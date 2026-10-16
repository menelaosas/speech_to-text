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
    from .score import score_run

    print(json.dumps(score_run(Path(args.run)), indent=2, ensure_ascii=False))
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
    sc.set_defaults(func=cmd_score)
    return p


def main(argv: list[str] | None = None) -> int:
    import os

    # Models load from local folders only; the pilot never reaches the network at run time.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    t0 = time.perf_counter()
    code = args.func(args)
    log.info("finished in %.1f s", time.perf_counter() - t0)
    return code


if __name__ == "__main__":
    sys.exit(main())
