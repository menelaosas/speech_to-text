"""Workaround for a known model bug: Cohere Transcribe rarely or never emits the Greek final sigma "ς"
on conversational audio. See https://huggingface.co/CohereLabs/cohere-transcribe-03-2026/discussions/25

Two restoration methods:

- `SigmaRestorer` (default, "lm"): for every word `w` where `w + "ς"` is a real Greek word, a masked
  language model (XLM-RoBERTa-large) scores both spellings in their sentence context and the more
  likely one is kept. Resolves ambiguous pairs such as "τη"/"της", "του"/"τους", "-ο"/"-ος".
- `restore_final_sigma` ("dict"): context-free word-frequency heuristic. No extra model needed,
  but only handles unambiguous cases ("όμω" -> "όμως").

Measured on Greek FLEURS text with every final ς removed (held-out 350 sentences, word accuracy):
none 82.1%, dict 88.8%, lm 95.6%. See docs/wiki/Accuracy.md.

Word lists: FrequencyWords by Hermit Dave (CC BY-SA 4.0), https://github.com/hermitdave/FrequencyWords
"""

import logging
import re
from functools import lru_cache
from pathlib import Path

from .config import SIGMA_MODEL_DIR, SIGMA_MODEL_ID

log = logging.getLogger(__name__)

RESOURCES = Path(__file__).parent / "resources"
FREQ_FILE = RESOURCES / "el_50k.txt"
# Words ending in ς that occur at least 3 times in the full FrequencyWords Greek list.
SIGMA_WORDS_FILE = RESOURCES / "el_sigma_words.txt"

# How many times more frequent `w + "ς"` must be than `w` before the dictionary method adds the sigma.
DEFAULT_RATIO = 20.0

_GREEK_WORD = re.compile(r"[Ͱ-Ͽἀ-῿]+")


@lru_cache(maxsize=2)
def load_frequencies(path: Path = FREQ_FILE) -> dict[str, int]:
    freqs: dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                freqs[parts[0]] = int(parts[1])
    return freqs


def _with_sigma(word: str) -> str:
    return word + ("Σ" if word.isupper() and len(word) > 1 else "ς")


def restore_final_sigma(text: str, ratio: float = DEFAULT_RATIO, freqs: dict[str, int] | None = None) -> str:
    """Dictionary method: add ς where `w + ς` is far more common than `w`."""
    freqs = freqs if freqs is not None else load_frequencies()

    def fix(match: re.Match) -> str:
        word = match.group(0)
        if word.endswith(("ς", "Σ")):
            return word
        lower = word.lower()
        with_sigma = freqs.get(lower + "ς", 0)
        if with_sigma and with_sigma >= ratio * max(freqs.get(lower, 0), 1):
            return _with_sigma(word)
        return word

    return _GREEK_WORD.sub(fix, text)


class SigmaRestorer:
    """Context-aware ς restoration with a masked language model."""

    def __init__(
        self,
        model: str | Path | None = None,
        device: str | None = None,
        margin: float | None = None,
        window: int = 8,
        batch_size: int = 64,
    ):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        from transformers.utils import logging as hf_logging

        if model is None:
            model = SIGMA_MODEL_DIR if (SIGMA_MODEL_DIR / "config.json").exists() else SIGMA_MODEL_ID
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32

        log.info("Loading ς-restoration model from %s", model)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(model))
        # The checkpoint's unused pooler weights trigger a harmless "UNEXPECTED keys" report.
        verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        try:
            self.model = AutoModelForMaskedLM.from_pretrained(str(model), dtype=dtype).to(self.device).eval()
        finally:
            hf_logging.set_verbosity(verbosity)
        self.sigma_words = set(load_frequencies(SIGMA_WORDS_FILE))
        # Log-likelihood advantage w+ς needs over w. None = choose per transcript (see `auto_margin`).
        self.margin = margin
        self.window = window
        self.batch_size = batch_size

    @staticmethod
    def auto_margin(texts: list[str]) -> float:
        """Pick the decision margin from how many words already end in ς.

        Normal Greek has ~15-18% of words ending in ς. When the ASR model has dropped them (rate < 5%,
        typical for conversational audio) we lean towards adding ς (-1, tuned on FLEURS with all ς
        removed). When the model is behaving (clean speech) we only add ς when the LM is confident (+3).
        """
        words = _GREEK_WORD.findall(" ".join(texts))
        if not words:
            return 0.0
        rate = sum(w.endswith(("ς", "Σ")) for w in words) / len(words)
        margin = -1.0 if rate < 0.05 else 3.0
        log.info("ς rate in raw transcript %.1f%% -> margin %+.0f", 100 * rate, margin)
        return margin

    def _candidate_inputs(self, text: str, spans: list[tuple[int, int]], wi: int, variant: str):
        """Token ids of a context window around word `wi` (with `variant` substituted) and the
        positions of the variant's tokens."""
        lo = spans[max(0, wi - self.window)][0]
        hi = spans[min(len(spans) - 1, wi + self.window)][1]
        start, end = spans[wi]
        window_text = text[lo:start] + variant + text[end:hi]
        v_start, v_end = start - lo, start - lo + len(variant)
        enc = self.tokenizer(window_text, return_offsets_mapping=True)
        positions = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > a and b > v_start and a < v_end]
        return enc["input_ids"], positions

    def _pseudo_log_likelihoods(self, items: list[tuple[list[int], list[int]]]) -> list[float]:
        """For each (ids, positions): sum of log P(token | rest) with each position masked in turn."""
        torch = self.torch
        flat = []
        for k, (ids, positions) in enumerate(items):
            for p in positions:
                masked = list(ids)
                masked[p] = self.tokenizer.mask_token_id
                flat.append((k, masked, p, ids[p]))

        totals = [0.0] * len(items)
        with torch.inference_mode():
            for i in range(0, len(flat), self.batch_size):
                chunk = flat[i : i + self.batch_size]
                length = max(len(c[1]) for c in chunk)
                ids = torch.full((len(chunk), length), self.tokenizer.pad_token_id)
                attention = torch.zeros((len(chunk), length), dtype=torch.long)
                for j, (_, masked, _, _) in enumerate(chunk):
                    ids[j, : len(masked)] = torch.tensor(masked)
                    attention[j, : len(masked)] = 1
                hidden = self.model.roberta(
                    input_ids=ids.to(self.device), attention_mask=attention.to(self.device)
                ).last_hidden_state
                rows = torch.arange(len(chunk), device=self.device)
                pos = torch.tensor([c[2] for c in chunk], device=self.device)
                true = torch.tensor([c[3] for c in chunk], device=self.device)
                # Vocabulary projection only at the masked positions (250k vocab: full logits would not fit).
                logits = self.model.lm_head(hidden[rows, pos])
                logp = torch.log_softmax(logits.float(), dim=-1)[rows, true]
                for (k, _, _, _), value in zip(chunk, logp.tolist()):
                    totals[k] += value
        return totals

    def restore(self, texts: list[str]) -> list[str]:
        """Restore final ς in each text. Context windows do not cross text boundaries."""
        jobs, items = [], []  # job = (text index, word index)
        all_spans = []
        for ti, text in enumerate(texts):
            matches = list(_GREEK_WORD.finditer(text))
            spans = [m.span() for m in matches]
            all_spans.append(spans)
            for wi, m in enumerate(matches):
                word = m.group(0)
                if word.endswith(("ς", "Σ")) or word.lower() + "ς" not in self.sigma_words:
                    continue
                jobs.append((ti, wi))
                items.append(self._candidate_inputs(text, spans, wi, word))
                items.append(self._candidate_inputs(text, spans, wi, _with_sigma(word)))

        if not jobs:
            return list(texts)
        scores = self._pseudo_log_likelihoods(items)
        margin = self.margin if self.margin is not None else self.auto_margin(texts)

        to_fix: dict[int, set[int]] = {}
        for k, (ti, wi) in enumerate(jobs):
            if scores[2 * k + 1] > scores[2 * k] + margin:
                to_fix.setdefault(ti, set()).add(wi)

        out = []
        for ti, text in enumerate(texts):
            fix = to_fix.get(ti)
            if not fix:
                out.append(text)
                continue
            parts, last = [], 0
            for wi, (start, end) in enumerate(all_spans[ti]):
                if wi in fix:
                    parts.append(text[last:end])
                    parts.append(_with_sigma(text[start:end])[-1])
                    last = end
            parts.append(text[last:])
            out.append("".join(parts))
        log.info("ς restoration: %d candidates, %d fixed", len(jobs), sum(len(v) for v in to_fix.values()))
        return out
