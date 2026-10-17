"""speech-pilot: run | smoke | compare | score."""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from .config import DEFAULT_CONFIG, load_config

log = logging.getLogger("speech_pilot")


def _parse_set(items: list[str]) -> dict:
    out = {}
    for item in items or []:
        key, _, value = item.partition("=")
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def _overrides(args) -> dict:
    o = _parse_set(args.set)
    if getattr(args, "excerpt", None):
        a, b = args.excerpt.split(":")
        o["input.excerpt"] = {"start_s": float(a), "end_s": float(b)}
    return o


def cmd_run(args) -> int:
    from .pipeline import PilotRun

    cfg = load_config(args.config, _overrides(args))
    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    force = set(args.force or [])
    if args.reuse_from:
        import shutil

        # Copy step folders of a finished run; each is then reused only if its config digest still matches.
        dst = cfg.run.output_root / run_id
        dst.mkdir(parents=True, exist_ok=True)
        for step in sorted(Path(args.reuse_from).glob("0*_*")):
            if step.is_dir() and not (dst / step.name).exists() and step.name < "08":
                shutil.copytree(step, dst / step.name)
    t = PilotRun(cfg, run_id, force=force).run()
    run_dir = cfg.run.output_root / run_id
    log.info("Done: %d segments, %d speakers, integrity %s", len(t.segments), len(t.speakers),
             "OK" if t.source.integrity_ok else "FAILED")
    log.info("Outputs in %s", run_dir)
    return 0 if t.source.integrity_ok else 2


def cmd_smoke(args) -> int:
    """Cohere on 30 s and diarization on 2 min of the file, with timing and VRAM."""
    import torch

    from . import audio as A
    from .gpu import StepMonitor, free_gpu, gpu_used_mib
    from .pipeline import set_determinism
    from .providers import AsrUnit, build_providers

    cfg = load_config(args.config, _overrides(args))
    set_determinism(cfg.run.seed)
    out = cfg.run.output_root / f"smoke-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    sha = A.sha256_file(cfg.input.path)
    wav = out / "smoke_2min.wav"
    start = args.start
    A.prepare_wav(cfg.input.path, wav, cfg.prepare.sample_rate, excerpt=(start, start + 120))
    audio, sr = A.read_wav(wav)
    providers = build_providers(cfg)
    result = {"sha256": sha, "excerpt_start_s": start}

    free_gpu()
    baseline = gpu_used_mib()
    result["baseline_vram_mib"] = baseline
    asr = providers["asr"]
    with StepMonitor("smoke_asr_30s", cfg.device.name, audio_s=30.0) as mon:
        with mon.phase("load"):
            asr.load()
        with mon.phase("process"):
            out_asr = asr.transcribe([AsrUnit("smoke", 0.0, 30.0, audio[: 30 * sr])], cfg.input.language)
        result["asr_device_actual"] = str(next(asr.model.parameters()).device)
        asr.unload()
    mon.after_unload(baseline)
    result["asr"] = mon.m.to_dict() | {"text": out_asr[0].text}

    if not args.skip_diarization:
        diar = providers["diarization"]
        free_gpu()
        with StepMonitor("smoke_diarization_120s", cfg.device.name, audio_s=len(audio) / sr) as mon:
            with mon.phase("load"):
                diar.load()
            with mon.phase("process"):
                res = diar.diarize(wav)
            diar.unload()
        mon.after_unload(baseline)
        result["diarization"] = mon.m.to_dict() | {
            "speakers": sorted({t.speaker for t in res.regular}), "turns": len(res.regular)}
    result["sha256_unchanged"] = A.sha256_file(cfg.input.path) == sha
    result["torch"] = torch.__version__
    (out / "smoke.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_compare(args) -> int:
    from .pipeline import compare_transcripts

    res = compare_transcripts(Path(args.a) / "transcript.json", Path(args.b) / "transcript.json")
    res["other_run"] = Path(args.b).name
    (Path(args.a) / "determinism.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    return 0 if res["identical"] else 1


def cmd_report(args) -> int:
    """Rebuild REPORT.md from a finished run's saved files (no models, metrics untouched)."""
    from .report import write_report
    from .schema import Transcript

    run = Path(args.run)
    t = Transcript.model_validate_json((run / "transcript.json").read_text(encoding="utf-8"))
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    ref = run / "reference" / "excerpt.json"
    write_report(run, t, metrics, json.loads(ref.read_text(encoding="utf-8")) if ref.exists() else None)
    log.info("wrote %s", run / "REPORT.md")
    return 0


def cmd_score(args) -> int:
    from .score import import_numbered_reference, score_run

    run = Path(args.run)
    if args.numbered:
        info = import_numbered_reference(Path(args.numbered), run / "reference" / "excerpt.txt")
        (run / "reference" / "excerpt_import.json").write_text(json.dumps(info, indent=2, ensure_ascii=False),
                                                              encoding="utf-8")
        log.info("imported %s -> reference/excerpt.txt (%s)", args.numbered, info["rule"])
    print(json.dumps(score_run(run, Path(args.reference) if args.reference else None), indent=2, ensure_ascii=False))
    return 0


def cmd_benchmark(args) -> int:
    from .benchmark import run_benchmark

    cfg = load_config(args.config, _parse_set(args.set))
    out = run_benchmark(cfg, args.dataset, args.split, args.limit, args.name)
    print((out / "REPORT.md").read_text(encoding="utf-8").split("## Worst")[0])
    log.info("Results in %s", out)
    return 0


def cmd_benchmark_compare(args) -> int:
    from .benchmark import compare

    res = compare(Path(args.a), Path(args.b))
    (Path(args.b) / f"compare_vs_{Path(args.a).name}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    for layer, units in res["layers"].items():
        for unit, v in units.items():
            print(f"{layer:<6} {unit.upper()}  A {100 * v['a']:.2f}%  B {100 * v['b']:.2f}%  "
                  f"Δ {100 * v['delta']:+.2f} pts  95% CI [{100 * v['delta_95ci'][0]:+.2f}, {100 * v['delta_95ci'][1]:+.2f}]"
                  f"  → {v['verdict']}  (better on {v['utterances_better']}, worse on {v['utterances_worse']} clips)")
    return 0


def cmd_align_reference(args) -> int:
    from .score import align_reference

    cfg = load_config(args.config)
    print(json.dumps(align_reference(Path(args.run) / "reference", cfg), indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="speech-pilot", description="Judicial Speech AI pilot")
    p.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run the full pipeline (steps 1-9)")
    r.add_argument("--run-id", help="reuse a run folder: finished steps with unchanged config are skipped")
    r.add_argument("--excerpt", help="process only START:END seconds of the file, e.g. 0:600")
    r.add_argument("--set", action="append", help="override a config value, e.g. asr.segmentation=vad")
    r.add_argument("--reuse-from", help="start from another run's step folders (reused where config matches)")
    r.add_argument("--force", action="append", help="re-run a step even if cached (step folder name or 'all'); needed after code changes")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("smoke", help="Cohere on 30 s + diarization on 2 min")
    s.add_argument("--start", type=float, default=60.0, help="excerpt start in seconds")
    s.add_argument("--skip-diarization", action="store_true")
    s.add_argument("--set", action="append")
    s.set_defaults(func=cmd_smoke)

    c = sub.add_parser("compare", help="determinism check between two run folders")
    c.add_argument("a")
    c.add_argument("b")
    c.set_defaults(func=cmd_compare)

    rp = sub.add_parser("report", help="rebuild REPORT.md from a finished run (e.g. after `compare`)")
    rp.add_argument("--run", required=True)
    rp.set_defaults(func=cmd_report)

    sc = sub.add_parser("score", help="WER/CER (and DER) against the operator's reference excerpt")
    sc.add_argument("--run", required=True)
    sc.add_argument("--numbered", help="operator transcript in '<speaker number> <text>' lines; converted "
                                       "to reference/excerpt.txt first (the source file is only read)")
    sc.add_argument("--reference", help="score against another run's reference folder (e.g. runs/final/reference)")

    bm = sub.add_parser("benchmark", help="WER/CER on Common Voice or FLEURS with the pipeline's ASR settings")
    bm.add_argument("--dataset", choices=["cv", "fleurs"], default="cv")
    bm.add_argument("--split", choices=["dev", "test", "train"], default="dev",
                    help="Common Voice split: dev while trying changes, test to confirm (FLEURS: always test)")
    bm.add_argument("--limit", type=int, help="seeded random subset of N clips (same clips every time)")
    bm.add_argument("--name", help="output folder name under runs/benchmarks/")
    bm.add_argument("--set", action="append", help="config override, e.g. asr.num_beams=1")
    bm.set_defaults(func=cmd_benchmark)

    bc = sub.add_parser("benchmark-compare", help="paired comparison of two benchmark runs (B minus A)")
    bc.add_argument("a")
    bc.add_argument("b")
    bc.set_defaults(func=cmd_benchmark_compare)

    ar = sub.add_parser("align-reference", help="time the operator's reference lines (forced alignment) -> "
                                                "reference/excerpt.rttm for DER")
    ar.add_argument("--run", required=True)
    ar.set_defaults(func=cmd_align_reference)
    sc.set_defaults(func=cmd_score)
    return p


def main(argv: list[str] | None = None) -> int:
    import os

    # Models load from local folders only; the pilot never reaches the network at run time.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    # Needed for deterministic cuBLAS; must be set before CUDA initialises.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    t0 = time.perf_counter()
    code = args.func(args)
    log.info("finished in %.1f s", time.perf_counter() - t0)
    return code


if __name__ == "__main__":
    sys.exit(main())
