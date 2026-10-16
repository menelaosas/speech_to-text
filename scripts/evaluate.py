"""Measure accuracy on the Greek FLEURS test set (650 read-speech clips with reference transcripts).

Download the data once:
    uv run hf download google/fleurs parquet-data/el_gr/test-00000-of-00001.parquet \
        --repo-type dataset --local-dir data/eval/fleurs

Two benchmarks:

  asr    Transcribe the audio and compute WER / CER against the references.
           uv run python scripts/evaluate.py asr -n 200 --config "beams=1 fix=off" --config "beams=4 fix=lm"
         Config keys: beams (int), fix (off | dict | lm).

  sigma  Remove every final ς from the reference texts and measure how well each method restores it
         (simulates the model's ς bug on conversational audio). Rows 0-299 were used for tuning,
         so report results on 300-649.
           uv run python scripts/evaluate.py sigma --offset 300 -n 350
"""

import argparse
import io
import re
import time
import unicodedata
from pathlib import Path

import jiwer
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from speech_to_text.audio import Segment
from speech_to_text.config import PROJECT_ROOT
from speech_to_text.greek import SigmaRestorer, restore_final_sigma

DEFAULT_DATA = PROJECT_ROOT / "data/eval/fleurs/parquet-data/el_gr/test-00000-of-00001.parquet"


def normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse spaces; keep accents (they matter in Greek)."""
    text = unicodedata.normalize("NFC", text.lower())
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    return re.sub(r"\s+", " ", text).strip()


def load_rows(path: Path, offset: int, n: int, audio: bool) -> list[dict]:
    columns = ["audio", "transcription"] if audio else ["transcription"]
    return pq.read_table(path, columns=columns).slice(offset, n).to_pylist()


def sigma_misses(refs: list[str], hyps: list[str]) -> int:
    """Hypothesis words that are exactly a reference word without its final ς."""
    misses = 0
    for ref, hyp in zip(refs, hyps):
        hyp_words = set(hyp.split())
        misses += sum(1 for w in ref.split() if w.endswith("ς") and w[:-1] in hyp_words)
    return misses


def apply_fix(texts: list[str], method: str, restorer: SigmaRestorer | None) -> list[str]:
    if method == "dict":
        return [restore_final_sigma(t) for t in texts]
    if method == "lm":
        return restorer.restore(texts)
    return texts


def cmd_asr(args: argparse.Namespace) -> None:
    from speech_to_text.transcriber import Transcriber

    rows = load_rows(args.data, args.offset, args.n, audio=True)
    audios, refs = [], []
    for row in rows:
        audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        assert sr == 16_000, sr
        audios.append(audio if audio.ndim == 1 else audio.mean(axis=1))
        refs.append(normalize(row["transcription"]))
    print(f"{len(audios)} clips, {sum(len(a) for a in audios) / 16_000 / 60:.1f} min of audio")

    transcriber = Transcriber()
    restorer = None
    # Clip index as start time so results can be matched back (empty outputs are dropped).
    segments = [Segment(float(i), i + len(a) / 16_000, a) for i, a in enumerate(audios)]

    print(f"\n{'config':<20} {'WER':>7} {'CER':>7} {'ς-miss':>7} {'time':>7}")
    for spec in args.config or ["beams=4 fix=lm"]:
        cfg = dict(part.split("=") for part in spec.split())
        beams, fix = int(cfg.get("beams", 4)), cfg.get("fix", "lm")
        if fix == "lm" and restorer is None:
            restorer = SigmaRestorer()

        t0 = time.perf_counter()
        result = transcriber.transcribe_segments(
            segments, language="el", batch_size=args.batch_size, num_beams=beams, show_progress=False
        )
        texts = {int(r.start): r.text for r in result}
        raw = [texts.get(i, "") for i in range(len(segments))]
        hyps = [normalize(t) for t in apply_fix(raw, fix, restorer)]
        elapsed = time.perf_counter() - t0

        label = f"beams={beams} fix={fix}"
        wer, cer = jiwer.wer(refs, hyps), jiwer.cer(refs, hyps)
        print(f"{label:<20} {wer:7.2%} {cer:7.2%} {sigma_misses(refs, hyps):7d} {elapsed:6.1f}s")
        for ref, hyp in list(zip(refs, hyps))[: args.show]:
            print(f"   REF: {ref}\n   HYP: {hyp}")


def cmd_sigma(args: argparse.Namespace) -> None:
    refs = [r["transcription"] for r in load_rows(args.data, args.offset, args.n, audio=False)]
    stripped = [" ".join(w[:-1] if w.endswith("ς") else w for w in r.split()) for r in refs]
    total = sum(len(r.split()) for r in refs)
    print(f"{len(refs)} sentences, {total} words, {sum(w.endswith('ς') for r in refs for w in r.split())} end in ς")

    restorer = SigmaRestorer()
    print(f"\n{'method':<8} {'restored':>9} {'wrong':>7} {'missing':>8} {'word acc':>9} {'time':>7}")
    for method in ("off", "dict", "lm"):
        t0 = time.perf_counter()
        fixed = apply_fix(stripped, method, restorer)
        elapsed = time.perf_counter() - t0
        tp = fp = fn = 0
        for ref, out in zip(refs, fixed):
            for a, b in zip(ref.split(), out.split()):
                if a.endswith("ς"):
                    tp, fn = (tp + 1, fn) if a == b else (tp, fn + 1)
                elif a != b:
                    fp += 1
        print(f"{method:<8} {tp:9d} {fp:7d} {fn:8d} {1 - (fp + fn) / total:9.2%} {elapsed:6.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--offset", type=int, default=0, help="first row to use")
    parser.add_argument("-n", type=int, default=200, help="number of rows")
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("asr", help="transcription WER / CER")
    a.add_argument("-b", "--batch-size", type=int, default=8)
    a.add_argument("--config", action="append", help='e.g. "beams=4 fix=lm" (repeatable)')
    a.add_argument("--show", type=int, default=0, help="print this many ref/hyp pairs per config")
    a.set_defaults(func=cmd_asr)

    s = sub.add_parser("sigma", help="final-ς restoration accuracy on text")
    s.set_defaults(func=cmd_sigma)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
