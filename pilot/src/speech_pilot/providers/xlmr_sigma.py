"""Greek final-sigma (ς) restoration as a separate, non-canonical text layer.

Cohere Transcribe rarely emits "ς" on conversational audio (upstream bug, model discussion #25).
For each word `w` where `w + "ς"` is a real Greek word, XLM-RoBERTa-large scores both spellings in
context (pseudo-log-likelihood) and the more likely one is kept.

Ported from the speech_to_text project, where it was measured on Greek FLEURS (held-out 350 sentences
with every ς removed): word accuracy 82.1% (none) -> 88.8% (word list) -> 95.6% (this method).
The output never replaces the canonical ASR text; it is stored as `text_normalized`.
"""

import logging
import re
from pathlib import Path

import torch

from ..config import NormalizationCfg
from ..schema import ModelInfo
from .base import TextNormalizer, local_revision

log = logging.getLogger(__name__)

SIGMA_WORDS_FILE = Path(__file__).resolve().parents[1] / "resources" / "el_sigma_words.txt"
_GREEK_WORD = re.compile(r"[Ͱ-Ͽἀ-῿]+")


def _with_sigma(word: str) -> str:
    return word + ("Σ" if word.isupper() and len(word) > 1 else "ς")


def auto_margin(texts: list[str]) -> tuple[float, float]:
    """Decision margin from the share of words already ending in ς (normal Greek: ~15-18%).

    < 5%: the ASR dropped them, lean towards adding ς (-1). Otherwise only confident fixes (+3).
    Both values were tuned on FLEURS (see the speech_to_text Accuracy wiki page).
    """
    words = _GREEK_WORD.findall(" ".join(texts))
    rate = sum(w.endswith(("ς", "Σ")) for w in words) / len(words) if words else 0.0
    return (-1.0 if rate < 0.05 else 3.0), rate


class XlmrSigmaRestorer(TextNormalizer):
    def __init__(self, cfg: NormalizationCfg, device: str):
        self.cfg = cfg
        self.device = device
        self.tokenizer = None
        self.model = None
        self.sigma_words: set[str] = set()

    def info(self) -> ModelInfo:
        return ModelInfo(
            role="normalization",
            model_id=self.cfg.model_id,
            local_path=str(self.cfg.model_path),
            revision=local_revision(self.cfg.model_path),
            device=self.device,
            dtype=self.cfg.dtype,
            capabilities={"task": "greek_final_sigma_restoration", "canonical": False},
        )

    def load(self) -> None:
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        from transformers.utils import logging as hf_logging

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_path)
        verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()  # unused pooler weights trigger a harmless load report
        try:
            dtype = torch.float16 if self.cfg.dtype == "float16" else torch.float32
            self.model = AutoModelForMaskedLM.from_pretrained(self.cfg.model_path, dtype=dtype)
        finally:
            hf_logging.set_verbosity(verbosity)
        self.model.to(self.device).eval()
        with open(SIGMA_WORDS_FILE, encoding="utf-8") as f:
            self.sigma_words = {line.split()[0] for line in f if line.strip()}

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None

    def _inputs(self, text: str, spans: list[tuple[int, int]], wi: int, variant: str):
        w = self.cfg.window_words
        lo, hi = spans[max(0, wi - w)][0], spans[min(len(spans) - 1, wi + w)][1]
        start, end = spans[wi]
        window = text[lo:start] + variant + text[end:hi]
        v0, v1 = start - lo, start - lo + len(variant)
        enc = self.tokenizer(window, return_offsets_mapping=True)
        pos = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > a and b > v0 and a < v1]
        return enc["input_ids"], pos

    @torch.inference_mode()
    def _pll(self, items: list[tuple[list[int], list[int]]]) -> list[float]:
        flat = []
        for k, (ids, positions) in enumerate(items):
            for p in positions:
                masked = list(ids)
                masked[p] = self.tokenizer.mask_token_id
                flat.append((k, masked, p, ids[p]))
        totals = [0.0] * len(items)
        bs = self.cfg.batch_size
        for i in range(0, len(flat), bs):
            chunk = flat[i : i + bs]
            length = max(len(c[1]) for c in chunk)
            ids = torch.full((len(chunk), length), self.tokenizer.pad_token_id)
            att = torch.zeros((len(chunk), length), dtype=torch.long)
            for j, (_, masked, _, _) in enumerate(chunk):
                ids[j, : len(masked)] = torch.tensor(masked)
                att[j, : len(masked)] = 1
            hidden = self.model.roberta(input_ids=ids.to(self.device), attention_mask=att.to(self.device)).last_hidden_state
            rows = torch.arange(len(chunk), device=self.device)
            pos = torch.tensor([c[2] for c in chunk], device=self.device)
            true = torch.tensor([c[3] for c in chunk], device=self.device)
            # Project to the 250k vocabulary only at masked positions (full logits would not fit in 8 GB).
            logp = torch.log_softmax(self.model.lm_head(hidden[rows, pos]).float(), dim=-1)[rows, true]
            for (k, *_), v in zip(chunk, logp.tolist()):
                totals[k] += v
        return totals

    def normalize(self, texts: list[str]) -> tuple[list[str], list[int], dict]:
        jobs, items, all_spans = [], [], []
        for ti, text in enumerate(texts):
            matches = list(_GREEK_WORD.finditer(text))
            spans = [m.span() for m in matches]
            all_spans.append(spans)
            for wi, m in enumerate(matches):
                word = m.group(0)
                if word.endswith(("ς", "Σ")) or word.lower() + "ς" not in self.sigma_words:
                    continue
                jobs.append((ti, wi))
                items.append(self._inputs(text, spans, wi, word))
                items.append(self._inputs(text, spans, wi, _with_sigma(word)))

        margin, rate = auto_margin(texts)
        details = {"raw_sigma_rate": round(rate, 4), "margin": margin, "candidates": len(jobs)}
        if not jobs:
            return list(texts), [0] * len(texts), details | {"changed_words": 0}

        scores = self._pll(items)
        fix: dict[int, set[int]] = {}
        for k, (ti, wi) in enumerate(jobs):
            if scores[2 * k + 1] > scores[2 * k] + margin:
                fix.setdefault(ti, set()).add(wi)

        out, changes = [], []
        for ti, text in enumerate(texts):
            words = fix.get(ti, set())
            parts, last = [], 0
            for wi, (start, end) in enumerate(all_spans[ti]):
                if wi in words:
                    parts += [text[last:end], _with_sigma(text[start:end])[-1]]
                    last = end
            out.append("".join(parts) + text[last:])
            changes.append(len(words))
        details["changed_words"] = sum(changes)
        return out, changes, details
