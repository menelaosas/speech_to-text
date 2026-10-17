"""Greek CTC forced alignment: word start/end times for a known text.

Used only to *time* words that Cohere (or the operator) already wrote; it never produces text, so it is
not a secondary ASR. torchaudio.functional.forced_align was removed in torchaudio 2.9, so the Viterbi
alignment is implemented here. Digits are verbalised first (the acoustic model has no digit tokens).
"""

import logging
import re
import unicodedata
from dataclasses import dataclass

import numpy as np
import torch

from ..config import AlignmentCfg
from ..greek_text import number_to_words
from ..schema import ModelInfo
from .base import Provider, local_revision

log = logging.getLogger(__name__)


@dataclass
class WordTime:
    start: float
    end: float
    score: float  # mean frame probability of the word's characters (0-1)
    aligned: bool  # False: no alignable characters, time interpolated from neighbours


def _chars_for(token: str, vocab: dict[str, int]) -> list[int]:
    """Vocabulary ids for one written token (uppercase Greek; accents kept when the vocab has them)."""
    token = re.sub(r"\d+", lambda m: " " + number_to_words(int(m.group())) + " ", token)
    ids = []
    # upper() decomposes some letters (ΐ -> Ι + dialytika + tonos): recompose, then skip leftover marks
    for ch in unicodedata.normalize("NFC", token.upper()):
        if unicodedata.combining(ch):
            continue
        if ch in vocab:
            ids.append(vocab[ch])
            continue
        base = unicodedata.normalize("NFD", ch)
        # drop the tonos but keep dialytika (Ϊ, Ϋ are in the vocab), then try the bare letter
        for candidate in (unicodedata.normalize("NFC", base.replace("́", "")), base[:1]):
            if candidate in vocab:
                ids.append(vocab[candidate])
                break
    return ids


def viterbi_align(log_probs: torch.Tensor, targets: list[int], blank: int = 0) -> list[int] | None:
    """CTC forced alignment. log_probs: [T, V]. Returns, for each target position, the frame index at
    which that token is emitted (first frame), or None if the text is longer than the audio allows."""
    T, n = log_probs.shape[0], len(targets)
    if n == 0 or T < n:
        return None
    S = 2 * n + 1  # blank-extended sequence: b t1 b t2 ... tn b
    ext = torch.full((S,), blank, dtype=torch.long, device=log_probs.device)
    ext[1::2] = torch.tensor(targets, device=log_probs.device)
    neg = torch.tensor(-1e30, device=log_probs.device)
    alpha = torch.full((S,), -1e30, device=log_probs.device)
    alpha[0] = log_probs[0, blank]
    alpha[1] = log_probs[0, ext[1]]
    skip_ok = torch.zeros(S, dtype=torch.bool, device=log_probs.device)
    skip_ok[3::2] = ext[3::2] != ext[1:-2:2]  # may skip the blank between two different tokens
    back = torch.zeros((T, S), dtype=torch.int8, device=log_probs.device)  # 0 stay, 1 from s-1, 2 from s-2
    emit = log_probs[:, ext]  # [T, S]
    for t in range(1, T):
        stay = alpha
        prev1 = torch.cat([neg.view(1), alpha[:-1]])
        prev2 = torch.where(skip_ok, torch.cat([neg.view(1).expand(2), alpha[:-2]]), neg)
        stacked = torch.stack([stay, prev1, prev2])
        best, arg = stacked.max(dim=0)
        alpha = best + emit[t]
        back[t] = arg.to(torch.int8)
    end = S - 1 if alpha[S - 1] >= alpha[S - 2] else S - 2
    back = back.cpu().numpy()
    s, first = end, [None] * n
    for t in range(T - 1, -1, -1):
        if s % 2 == 1:
            first[s // 2] = t  # keeps overwriting while going back: ends at the first frame of the token
        if t == 0:
            break
        s -= int(back[t, s])
    return first if all(f is not None for f in first) else None


class CtcAligner(Provider):
    role = "alignment"

    def __init__(self, cfg: AlignmentCfg, device: str):
        self.cfg = cfg
        self.device = device
        self.model = None
        self.processor = None
        self.vocab: dict[str, int] = {}

    def info(self) -> ModelInfo:
        return ModelInfo(role="alignment", model_id=self.cfg.model_id, local_path=str(self.cfg.model_path),
                         revision=local_revision(self.cfg.model_path), device=self.device, dtype="float32",
                         capabilities={"word_timestamps": True, "produces_text": False})

    def load(self) -> None:
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.cfg.model_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.cfg.model_path).to(self.device).eval()
        import json

        self.vocab = json.loads((self.cfg.model_path / "vocab.json").read_text(encoding="utf-8"))
        self.blank = self.vocab["<pad>"]
        self.sep = self.vocab["|"]

    def unload(self) -> None:
        self.model = None
        self.processor = None

    @torch.inference_mode()
    def emissions(self, audio: np.ndarray, sr: int) -> tuple[torch.Tensor, float]:
        """Frame log-probabilities, computed in windows with context so long audio fits in VRAM.
        Deterministic kernels are forced: GPU float non-determinism otherwise flips Viterbi ties and moves a
        few word boundaries by one frame (20 ms) between runs (DECISIONS D-19)."""
        previous = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            return self._emissions(audio, sr)
        finally:
            torch.use_deterministic_algorithms(previous)

    def _emissions(self, audio: np.ndarray, sr: int) -> tuple[torch.Tensor, float]:
        win, ctx = int(self.cfg.window_s * sr), int(self.cfg.context_s * sr)
        frames, frame_s = [], None
        for start in range(0, len(audio), win):
            lo, hi = max(0, start - ctx), min(len(audio), start + win + ctx)
            x = self.processor(audio[lo:hi], sampling_rate=sr, return_tensors="pt").input_values.to(self.device)
            lp = torch.log_softmax(self.model(x).logits[0].float(), dim=-1)
            frame_s = (hi - lo) / sr / lp.shape[0]
            a, b = int(round((start - lo) / sr / frame_s)), int(round((min(start + win, len(audio)) - lo) / sr / frame_s))
            frames.append(lp[a:b])
        return torch.cat(frames), frame_s

    def align(self, audio: np.ndarray, sr: int, tokens: list[str], offset: float = 0.0) -> list[WordTime]:
        """Times for each whitespace token of the text. Tokens without alignable characters (punctuation,
        Latin letters) are interpolated from their neighbours and marked aligned=False."""
        if not tokens:
            return []
        lp, frame_s = self.emissions(audio, sr)
        chars = [_chars_for(t, self.vocab) for t in tokens]
        targets, owner = [], []
        for i, ids in enumerate(chars):
            if not ids:
                continue
            if targets:
                targets.append(self.sep)
                owner.append(None)
            targets += ids
            owner += [i] * len(ids)
        first = viterbi_align(lp.to(self.device), targets, self.blank) if targets else None
        dur = len(audio) / sr
        if first is None:
            log.warning("alignment failed (%d tokens, %.1f s): times spread evenly", len(tokens), dur)
            step = dur / len(tokens)
            return [WordTime(offset + i * step, offset + (i + 1) * step, 0.0, False) for i in range(len(tokens))]

        probs = lp.exp()
        spans: dict[int, list[int]] = {}
        for pos, (frame, i) in enumerate(zip(first, owner)):
            if i is not None:
                spans.setdefault(i, []).append((frame, targets[pos]))
        out: list[WordTime | None] = [None] * len(tokens)
        ordered = sorted(spans)
        for k, i in enumerate(ordered):
            frames = [f for f, _ in spans[i]]
            start = frames[0]
            # word ends where the next word starts (minus the separator) or at its last char frame + 1
            nxt = spans[ordered[k + 1]][0][0] if k + 1 < len(ordered) else min(frames[-1] + 2, lp.shape[0])
            end = max(frames[-1] + 1, min(nxt, frames[-1] + int(0.5 / frame_s)))
            score = float(np.mean([probs[f, tok].item() for f, tok in spans[i]]))
            out[i] = WordTime(offset + start * frame_s, offset + end * frame_s, round(score, 3), True)
        for i, w in enumerate(out):  # interpolate unaligned tokens
            if w is None:
                prev = next((out[j] for j in range(i - 1, -1, -1) if out[j]), None)
                nxt = next((out[j] for j in range(i + 1, len(out)) if out[j]), None)
                t = prev.end if prev else (nxt.start if nxt else offset)
                out[i] = WordTime(t, t, 0.0, False)
        return out
