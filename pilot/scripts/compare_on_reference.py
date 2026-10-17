"""Measure pipeline choices against the operator's reference excerpt.

Runs the pipeline on exactly the reference window with different settings and scores each against
`reference/excerpt.txt` of a finished run (WER/CER, spec and lenient normalisers, raw and ς layer).
Also scores Cohere FP32 (CPU) on the same units as BF16 (GPU), and the error rate inside vs outside
overlap segments.

    uv run python scripts/compare_on_reference.py --run runs/final
"""

import argparse
import json
from pathlib import Path

import jiwer
import torch

from speech_pilot import audio as A
from speech_pilot.config import load_config
from speech_pilot.greek_text import normalize
from speech_pilot.pipeline import PilotRun
from speech_pilot.providers import AsrUnit
from speech_pilot.providers.cohere_asr import CohereASR
from speech_pilot.schema import Transcript
from speech_pilot.score import parse_reference, score_texts

VARIANTS = {
    "hybrid_beam4": {"asr.segmentation": "hybrid"},
    "hybrid_greedy": {"asr.segmentation": "hybrid", "asr.num_beams": 1},
    "turns_beam4": {"asr.segmentation": "turns"},
    "vad_beam4": {"asr.segmentation": "vad"},
}


def headline(scores: dict, layer: str) -> dict:
    s = scores[f"{layer}/spec/numbers_to_words"]
    return {"wer": s["wer"], "cer": s["cer"], "sub": s["substitutions"], "del": s["deletions"], "ins": s["insertions"]}


def overlap_breakdown(reference: str, t: Transcript) -> dict:
    """Align reference to hypothesis words and attribute every error to the segment of the nearest
    hypothesis word, so errors can be split by overlap / non-overlap segments."""
    seg_of_word, hyp_words = [], []
    for s in t.segments:
        words = normalize(s.text_normalized or s.text).split()
        hyp_words += words
        seg_of_word += [s.overlap] * len(words)
    ref = normalize(reference)
    out = jiwer.process_words(ref, " ".join(hyp_words))
    stats = {True: {"errors": 0, "ref_words": 0}, False: {"errors": 0, "ref_words": 0}}
    for chunk in out.alignments[0]:
        hyp_idx = min(max(chunk.hyp_start_idx, 0), len(seg_of_word) - 1)
        cls = seg_of_word[hyp_idx] if seg_of_word else False
        n_ref = chunk.ref_end_idx - chunk.ref_start_idx
        n_hyp = chunk.hyp_end_idx - chunk.hyp_start_idx
        stats[cls]["ref_words"] += n_ref
        if chunk.type == "substitute":
            stats[cls]["errors"] += n_ref
        elif chunk.type == "delete":
            stats[cls]["errors"] += n_ref
        elif chunk.type == "insert":
            stats[cls]["errors"] += n_hyp
    return {("overlap" if k else "no_overlap"): v | {"approx_wer": round(v["errors"] / max(v["ref_words"], 1), 4)}
            for k, v in stats.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--skip-fp32", action="store_true")
    args = p.parse_args()

    ref_dir = args.run / "reference"
    excerpt = json.loads((ref_dir / "excerpt.json").read_text(encoding="utf-8"))
    reference = " ".join(t for _, t in parse_reference(ref_dir / "excerpt.txt"))
    window = {"start_s": excerpt["start_s"], "end_s": excerpt["end_s"]}
    results: dict = {"excerpt": window, "variants": {}}

    full = Transcript.model_validate_json((args.run / "transcript.json").read_text(encoding="utf-8"))
    in_window = [s for s in full.segments if s.id in set(excerpt["segments"])]
    sub = full.model_copy(update={"segments": in_window})
    sc = score_texts(reference, {"raw": " ".join(s.text for s in in_window),
                                 "sigma": " ".join(s.text_normalized or s.text for s in in_window)})
    results["variants"]["full_run (diarization over whole file)"] = {
        "raw": headline(sc, "raw"), "sigma": headline(sc, "sigma"), "overlap": overlap_breakdown(reference, sub)}

    for name, overrides in VARIANTS.items():
        cfg = load_config(overrides={**overrides, "input.excerpt": window})
        run_id = f"ref-{name}"
        t = PilotRun(cfg, run_id).run()
        sc = score_texts(reference, {"raw": " ".join(s.text for s in t.segments),
                                     "sigma": " ".join(s.text_normalized or s.text for s in t.segments)})
        results["variants"][name] = {"segments": len(t.segments), "raw": headline(sc, "raw"),
                                     "sigma": headline(sc, "sigma"), "overlap": overlap_breakdown(reference, t)}

    if not args.skip_fp32:
        # Same units as the BF16 hybrid run, transcribed in FP32 on CPU (FP32 does not fit in 8 GB VRAM).
        run = load_config().run.output_root / "ref-hybrid_beam4"
        rows = json.loads((run / "05_asr" / "asr_raw.json").read_text(encoding="utf-8"))["units"]
        audio, sr = A.read_wav(run / "02_prepare" / "processing.wav")
        units = [AsrUnit(r["id"], r["start"], r["end"], audio[int(r["start"] * sr):int(r["end"] * sr)]) for r in rows]
        cfg = load_config()
        torch.cuda.empty_cache()
        fp32 = CohereASR(cfg.asr.model_copy(update={"dtype": "float32"}), "cpu")
        fp32.load()
        texts = [o.text for o in fp32.transcribe(units, "el")]
        fp32.unload()
        bf16 = [r["text"] for r in sorted(rows, key=lambda r: r["start"])]
        order = sorted(range(len(rows)), key=lambda i: rows[i]["start"])
        sc = score_texts(reference, {"raw": " ".join(texts[i] for i in order), "bf16": " ".join(bf16)})
        results["precision"] = {"units": len(units),
                                "identical_units": sum(texts[i] == rows[i]["text"] for i in range(len(rows))),
                                "fp32_cpu_raw": headline(sc, "raw"), "bf16_gpu_raw": headline(sc, "bf16")}

    out = ref_dir / "comparison.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
