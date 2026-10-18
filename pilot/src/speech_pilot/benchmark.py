"""ASR benchmarks on public Greek read-speech sets (Common Voice, FLEURS).

Measures the parts of the pipeline that decide the words: audio preparation (same ffmpeg conversion to
16 kHz mono 16-bit as step 02), Cohere ASR with the configured decoding, and the Greek ς layer. Each clip
is one ASR unit. VAD, diarization and attribution are not involved (one speaker per clip), so this does
not measure speaker accuracy; use the operator reference excerpt for that.

    speech-pilot benchmark --dataset cv --split dev            # while trying changes
    speech-pilot benchmark --dataset cv --split test           # to confirm a change
    speech-pilot benchmark-compare runs/benchmarks/A runs/benchmarks/B
"""

import csv
import hashlib
import io
import json
import logging
import random
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import jiwer
import numpy as np

from .audio import split_long
from .config import PilotConfig
from .gpu import StepMonitor, free_gpu
from .greek_text import normalize

log = logging.getLogger(__name__)
SR = 16000


@dataclass
class Utterance:
    id: str
    reference: str
    speaker: str
    source: Path | bytes  # file path, or encoded audio bytes (FLEURS parquet)


# ---- datasets ------------------------------------------------------------------------------------------------
def load_common_voice(root: Path, split: str) -> tuple[list[Utterance], dict]:
    tsv = root / f"{split}.tsv"
    if not tsv.exists():
        raise FileNotFoundError(f"{tsv} not found (expected a Common Voice language folder with {split}.tsv)")
    with open(tsv, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    utts = [Utterance(r["path"], r["sentence"], r["client_id"][:12], root / "clips" / r["path"])
            for r in rows if r.get("sentence", "").strip()]
    return utts, {"dataset": "common_voice", "release": root.parent.name, "split": split, "path": str(root),
                  "tsv_sha256": hashlib.sha256(tsv.read_bytes()).hexdigest()}


def load_fleurs(parquet: Path) -> tuple[list[Utterance], dict]:
    import pyarrow.parquet as pq

    rows = pq.read_table(parquet, columns=["id", "audio", "transcription", "raw_transcription"]).to_pylist()
    utts = [Utterance(f"fleurs_{r['id']}_{i}", r["raw_transcription"], "", r["audio"]["bytes"])
            for i, r in enumerate(rows)]
    return utts, {"dataset": "fleurs", "split": "test", "path": str(parquet),
                  "parquet_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest()}


def select(utts: list[Utterance], limit: int | None, seed: int) -> list[Utterance]:
    """Deterministic subset: a seeded sample, kept in id order so the same clips are chosen every time."""
    utts = sorted(utts, key=lambda u: u.id)
    if limit and limit < len(utts):
        utts = sorted(random.Random(seed).sample(utts, limit), key=lambda u: u.id)
    return utts


# ---- audio ---------------------------------------------------------------------------------------------------
def decode(source: Path | bytes) -> np.ndarray:
    """Same conversion as pipeline step 02 (mono, 16 kHz, 16-bit PCM), read back as float32."""
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i",
           "pipe:0" if isinstance(source, bytes) else str(source),
           "-ac", "1", "-ar", str(SR), "-f", "s16le", "-acodec", "pcm_s16le", "-"]
    out = subprocess.run(cmd, input=source if isinstance(source, bytes) else None, capture_output=True, check=True)
    return np.frombuffer(out.stdout, dtype=np.int16).astype(np.float32) / 32768.0


# ---- scoring -------------------------------------------------------------------------------------------------
def utterance_errors(ref: str, hyp: str) -> dict:
    """Word and character error counts after the Greek normaliser. Empty output = all reference words deleted;
    an empty reference counts every hypothesis word as an insertion."""
    r, h = normalize(ref), normalize(hyp)
    rw, hw = r.split(), h.split()
    if not rw or not hw:
        return {"ref_words": len(rw), "word_errors": len(rw) + len(hw), "sub": 0, "del": len(rw), "ins": len(hw),
                "ref_chars": len(r), "char_errors": len(r) + len(h)}
    words, chars = jiwer.process_words(r, h), jiwer.process_characters(r, h)
    return {"ref_words": len(rw), "word_errors": words.substitutions + words.deletions + words.insertions,
            "sub": words.substitutions, "del": words.deletions, "ins": words.insertions,
            "ref_chars": len(r), "char_errors": chars.substitutions + chars.deletions + chars.insertions}


def bootstrap(errors: np.ndarray, totals: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    """95% interval of a corpus error rate by resampling utterances."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(errors), size=(samples, len(errors)))
    rates = errors[idx].sum(axis=1) / np.maximum(totals[idx].sum(axis=1), 1)
    return float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))


def summarize(rows: list[dict], layer: str, samples: int, seed: int) -> dict:
    e = np.array([r[layer]["word_errors"] for r in rows]); n = np.array([r[layer]["ref_words"] for r in rows])
    ce = np.array([r[layer]["char_errors"] for r in rows]); cn = np.array([r[layer]["ref_chars"] for r in rows])
    wer, cer = e.sum() / max(n.sum(), 1), ce.sum() / max(cn.sum(), 1)
    return {"wer": round(float(wer), 4), "wer_95ci": [round(x, 4) for x in bootstrap(e, n, samples, seed)],
            "cer": round(float(cer), 4), "cer_95ci": [round(x, 4) for x in bootstrap(ce, cn, samples, seed + 1)],
            "substitutions": int(sum(r[layer]["sub"] for r in rows)), "deletions": int(sum(r[layer]["del"] for r in rows)),
            "insertions": int(sum(r[layer]["ins"] for r in rows)), "ref_words": int(n.sum()),
            "utterances_with_errors": int((e > 0).sum())}


# ---- run -----------------------------------------------------------------------------------------------------
def run_benchmark(cfg: PilotConfig, dataset: str, split: str, limit: int | None, name: str | None,
                  parakeet_model: Path | None = None) -> Path:
    from .pipeline import set_determinism
    from .providers import AsrUnit, build_providers

    set_determinism(cfg.run.seed)
    b = cfg.benchmark
    if dataset == "cv":
        utts, info = load_common_voice(b.common_voice_dir, split)
    elif dataset == "fleurs":
        utts, info = load_fleurs(b.fleurs_parquet)
    else:
        raise ValueError(f"unknown dataset {dataset!r} (cv or fleurs)")
    utts = select(utts, limit, cfg.run.seed)
    name = name or f"{dataset}-{info['split']}-n{len(utts)}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out = b.output_root / name
    out.mkdir(parents=True, exist_ok=False)
    log.info("%s %s: %d clips -> %s", dataset, info["split"], len(utts), out)

    t0 = time.perf_counter()
    audio, failed = {}, []
    with ThreadPoolExecutor(b.decode_workers) as pool:
        for u, result in zip(utts, pool.map(lambda u: _safe_decode(u.source), utts)):
            (audio.__setitem__(u.id, result) if isinstance(result, np.ndarray) else failed.append([u.id, result]))
    utts = [u for u in utts if u.id in audio]
    decode_s = time.perf_counter() - t0
    audio_s = sum(len(a) for a in audio.values()) / SR

    providers = build_providers(cfg)
    if parakeet_model is not None:
        from .providers.parakeet_asr import ParakeetASR
        asr = ParakeetASR(parakeet_model, cfg.device.name)
    else:
        asr = providers["asr"]
        asr.cfg = asr.cfg.model_copy(update={"batch_size": b.asr_batch_size})
    # Clips longer than the pipeline's unit limit are split at quiet points exactly like the pipeline does
    # (FLEURS has some > 35 s); the pieces' texts are joined back in order.
    units, owner = [], []
    for u in utts:
        a = audio[u.id]
        for k, (s, e) in enumerate(split_long(a, SR, 0.0, len(a) / SR, cfg.asr.max_unit_s, cfg.asr.split_search_s)):
            units.append(AsrUnit(f"{u.id}#{k}", s, e, a[int(s * SR):int(e * SR)]))
            owner.append(u.id)
    split_clips = len({o for o in owner}) != len(owner)
    free_gpu()
    with StepMonitor("benchmark_asr", cfg.device.name, audio_s=audio_s) as mon_asr:
        with mon_asr.phase("load"):
            asr.load()
        with mon_asr.phase("process"):
            pieces = [o.text for o in asr.transcribe(units, cfg.input.language)]
            joined: dict[str, list[str]] = {}
            for uid, text in zip(owner, pieces):
                joined.setdefault(uid, []).append(text)
            raw = [" ".join(t for t in joined[u.id] if t) for u in utts]
        asr_info, decoding = asr.info().model_dump(), asr.decoding()
        asr.unload()
    free_gpu()

    sigma, norm_info, norm_details = None, None, None
    if providers["normalization"] is not None and cfg.input.language == "el":
        norm = providers["normalization"]
        with StepMonitor("benchmark_normalize", cfg.device.name) as mon_norm:
            with mon_norm.phase("load"):
                norm.load()
            with mon_norm.phase("process"):
                sigma, _, norm_details = norm.normalize(raw)
            norm_info = norm.info().model_dump()
            norm.unload()
        free_gpu()

    rows = []
    for i, u in enumerate(utts):
        row = {"id": u.id, "speaker": u.speaker, "duration_s": round(len(audio[u.id]) / SR, 3),
               "reference": u.reference, "raw": raw[i], "raw_errors": utterance_errors(u.reference, raw[i])}
        if sigma is not None:
            row["sigma"] = sigma[i]
            row["sigma_errors"] = utterance_errors(u.reference, sigma[i])
        rows.append(row)
    layers = ["raw"] + (["sigma"] if sigma is not None else [])
    results = {
        "name": name, "created_at": datetime.now(timezone.utc).isoformat(), "dataset": info,
        "utterances": len(rows), "audio_s": round(audio_s, 1), "failed_to_decode": failed,
        "asr_units": len(units), "clips_split_over_max_unit_s": sum(1 for u in utts if owner.count(u.id) > 1)
        if split_clips else 0,
        "selection": {"limit": limit, "seed": cfg.run.seed,
                      "ids_sha256": hashlib.sha256("\n".join(u.id for u in utts).encode()).hexdigest()},
        "normalizer": "greek_text.normalize: lowercase, no tonos/dialytika, ς->σ, numbers to words, no punctuation",
        "scores": {layer: summarize([{layer: r[f"{layer}_errors"]} for r in rows], layer, b.bootstrap_samples,
                                    cfg.run.seed) for layer in layers},
        "models": {"asr": asr_info, "normalization": norm_info}, "decoding": decoding,
        "normalization_details": norm_details,
        "timing": {"decode_s": round(decode_s, 1), "asr": mon_asr.m.to_dict(),
                   "normalize": mon_norm.m.to_dict() if sigma is not None else None},
        "config": cfg.model_dump(mode="json"), "config_digest": cfg.digest(),
        "caveat": ("Common Voice / FLEURS are read speech; scores are for comparing pipeline changes, not the "
                   "accuracy on meeting recordings. Cohere was evaluated on Common Voice 17; its training data "
                   "is not disclosed, so absolute scores may be optimistic."),
    }
    (out / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out / "utterances.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "REPORT.md").write_text(_report(results, rows), encoding="utf-8")
    return out


def _safe_decode(source):
    try:
        return decode(source)
    except subprocess.CalledProcessError as exc:
        return exc.stderr.decode(errors="replace")[-200:]


def _report(res: dict, rows: list[dict]) -> str:
    L = [f"# Benchmark `{res['name']}`", "",
         f"{res['dataset']['dataset']} {res['dataset'].get('release', '')} · split **{res['dataset']['split']}** · "
         f"{res['utterances']} clips · {res['audio_s'] / 3600:.2f} h · decoding {res['decoding']}", "",
         "| Layer | WER | 95% CI | CER | 95% CI | Sub | Del | Ins | Ref words |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for layer, s in res["scores"].items():
        L.append(f"| {layer} | {100 * s['wer']:.2f}% | {100 * s['wer_95ci'][0]:.2f}–{100 * s['wer_95ci'][1]:.2f} | "
                 f"{100 * s['cer']:.2f}% | {100 * s['cer_95ci'][0]:.2f}–{100 * s['cer_95ci'][1]:.2f} | "
                 f"{s['substitutions']} | {s['deletions']} | {s['insertions']} | {s['ref_words']} |")
    asr = res["timing"]["asr"]
    L += ["", f"ASR on {asr['device']}: load {asr['load_s']} s, process {asr['process_s']} s "
          f"(RTF {asr['rtf']}), peak {asr['peak_allocated_mib']} MiB. Audio decoding {res['timing']['decode_s']} s.",
          "", f"> {res['caveat']}", "", "## Worst utterances (raw layer)", "",
          "| Clip | WER | Reference | Hypothesis |", "| --- | --- | --- | --- |"]
    worst = sorted(rows, key=lambda r: -r["raw_errors"]["word_errors"] / max(r["raw_errors"]["ref_words"], 1))[:15]
    for r in worst:
        e = r["raw_errors"]
        L.append(f"| {r['id']} | {100 * e['word_errors'] / max(e['ref_words'], 1):.0f}% | "
                 f"{r['reference'].replace('|', '/')} | {r['raw'].replace('|', '/')} |")
    return "\n".join(L) + "\n"


# ---- compare -------------------------------------------------------------------------------------------------
def compare(a_dir: Path, b_dir: Path, samples: int = 2000, seed: int = 1234) -> dict:
    """Paired comparison of two benchmark runs on the same clips: B minus A, with a bootstrap interval.
    A change is only called better/worse when the 95% interval excludes zero."""
    a = [json.loads(l) for l in open(a_dir / "utterances.jsonl", encoding="utf-8")]
    b = {r["id"]: r for r in (json.loads(l) for l in open(b_dir / "utterances.jsonl", encoding="utf-8"))}
    common = [r for r in a if r["id"] in b]
    if len(common) != len(a) or len(common) != len(b):
        raise ValueError(f"runs cover different clips ({len(a)} vs {len(b)}, {len(common)} shared): "
                         "use the same dataset, split and --limit")
    out = {"a": a_dir.name, "b": b_dir.name, "utterances": len(common), "layers": {}}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(common), size=(samples, len(common)))
    for layer in ("raw", "sigma"):
        if f"{layer}_errors" not in common[0] or f"{layer}_errors" not in b[common[0]["id"]]:
            continue
        for unit, err, tot in (("wer", "word_errors", "ref_words"), ("cer", "char_errors", "ref_chars")):
            ea = np.array([r[f"{layer}_errors"][err] for r in common])
            eb = np.array([b[r["id"]][f"{layer}_errors"][err] for r in common])
            n = np.array([r[f"{layer}_errors"][tot] for r in common])
            delta = (eb[idx].sum(1) - ea[idx].sum(1)) / np.maximum(n[idx].sum(1), 1)
            lo, hi = np.percentile(delta, [2.5, 97.5])
            d = (eb.sum() - ea.sum()) / max(n.sum(), 1)
            out["layers"].setdefault(layer, {})[unit] = {
                "a": round(float(ea.sum() / n.sum()), 4), "b": round(float(eb.sum() / n.sum()), 4),
                "delta": round(float(d), 4), "delta_95ci": [round(float(lo), 4), round(float(hi), 4)],
                "verdict": "B better" if hi < 0 else "B worse" if lo > 0 else "no significant difference",
                "utterances_better": int((eb < ea).sum()), "utterances_worse": int((eb > ea).sum()),
            }
    return out
