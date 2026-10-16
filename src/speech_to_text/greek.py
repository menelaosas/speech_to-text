"""Workaround for a known model bug: Cohere Transcribe never emits the Greek final sigma "ς".

See https://huggingface.co/CohereLabs/cohere-transcribe-03-2026/discussions/25

We restore it with a word-frequency heuristic: a word `w` becomes `w + "ς"` when `w + "ς"`
is far more common in Greek than `w` itself (e.g. "όμω" -> "όμως", "σα" -> "σας").
Ambiguous pairs where both forms are common ("τη"/"της", "αυτό"/"αυτός") are left untouched.

Word frequencies: FrequencyWords by Hermit Dave (CC BY-SA 4.0),
https://github.com/hermitdave/FrequencyWords
"""

import re
from functools import lru_cache
from pathlib import Path

FREQ_FILE = Path(__file__).parent / "resources" / "el_50k.txt"

# How many times more frequent `w + "ς"` must be than `w` before we add the sigma.
DEFAULT_RATIO = 20.0

_GREEK_WORD = re.compile(r"[Ͱ-Ͽἀ-῿]+")


@lru_cache(maxsize=1)
def load_frequencies(path: Path = FREQ_FILE) -> dict[str, int]:
    freqs: dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                freqs[parts[0]] = int(parts[1])
    return freqs


def restore_final_sigma(text: str, ratio: float = DEFAULT_RATIO, freqs: dict[str, int] | None = None) -> str:
    freqs = freqs if freqs is not None else load_frequencies()

    def fix(match: re.Match) -> str:
        word = match.group(0)
        if word.endswith(("ς", "Σ")):
            return word
        lower = word.lower()
        with_sigma = freqs.get(lower + "ς", 0)
        if with_sigma and with_sigma >= ratio * max(freqs.get(lower, 0), 1):
            return word + ("Σ" if word.isupper() and len(word) > 1 else "ς")
        return word

    return _GREEK_WORD.sub(fix, text)
