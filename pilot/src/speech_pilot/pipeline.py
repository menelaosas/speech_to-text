"""Runs the pilot steps in order. Depends only on the provider interfaces.

Every step writes into runs/<run_id>/<step>/ and a `step.json` holding a digest of the configuration
sections it depends on. A step whose digest still matches is reused, so later steps can be re-run
without repeating earlier ones (and without reloading their models).
"""

import hashlib
import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from . import audio as A
from .attribution import attribute, group_words
from .config import PilotConfig
from .export import write_json, write_rttm, write_srt, write_txt
from .gpu import StepMonitor, free_gpu, gpu_state, gpu_used_mib, host_power_state
from .providers import AsrUnit, build_providers
from .qc import flag_segment
from .reference import export_reference
from .schema import (Interval, ModelInfo, Provenance, Segment, SourceInfo, SpeakerSummary, SpeakerTurn, Transcript,
                     Word)
from .units import overlap_regions, units_from_turns, units_from_vad, units_hybrid

log = logging.getLogger(__name__)

# Config sections each step depends on (for reuse of earlier results).
DEPENDS = {
    "01_inspect": ["input"],
    "02_prepare": ["input", "prepare"],
    "03_vad": ["input", "prepare", "vad"],
    "04_diarization": ["input", "prepare", "diarization"],
    "05_asr": ["input", "prepare", "vad", "diarization", "asr"],
    "05b_normalize": ["input", "prepare", "vad", "diarization", "asr", "normalization"],
    "05c_align": ["input", "prepare", "vad", "diarization", "asr", "normalization", "alignment"],
    "06_attribution": ["input", "prepare", "vad", "diarization", "asr", "normalization", "alignment", "attribution"],
    "07_qc": ["input", "prepare", "vad", "diarization", "asr", "normalization", "alignment", "attribution", "qc"],
}


STEP_ORDER = ["01_inspect", "02_prepare", "03_vad", "04_diarization", "05_asr", "05b_normalize", "05c_align",
              "06_attribution", "07_qc"]


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _dump(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class PilotRun:
    def __init__(self, cfg: PilotConfig, run_id: str, force: set[str] | None = None):
        self.cfg = cfg
        self.run_id = run_id
        self.dir = cfg.run.output_root / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.force = force or set()
        self.providers = build_providers(cfg)
        self.metrics: dict[str, dict] = {}
        self.reused: list[str] = []
        self.isolated: list[str] = []  # GPU steps computed in an isolated child process in this invocation
        self.computed: set[str] = set()  # steps computed in this process in this invocation
        self.device = cfg.device.name
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            if not cfg.device.allow_cpu:
                raise RuntimeError("CUDA requested but not available, and allow_cpu is false (no silent CPU fallback)")
            log.warning("CUDA not available: running on CPU (allowed by config)")
            self.device = "cpu"

    # ---- step bookkeeping -------------------------------------------------------------------------------
    def _digest(self, step: str) -> str:
        """Cache key: the config sections the step depends on + the output hashes of all earlier steps,
        so a recomputed upstream step invalidates everything after it."""
        sections = {s: getattr(self.cfg, s).model_dump(mode="json") for s in DEPENDS.get(step, [])}
        upstream = {}
        for prev in STEP_ORDER[: STEP_ORDER.index(step)] if step in STEP_ORDER else []:
            meta = self.dir / prev / "step.json"
            if meta.exists():
                upstream[prev] = _load(meta).get("output_hash")
        blob = json.dumps({"config": sections, "upstream": upstream}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def _output_hash(self, step: str) -> str:
        h = hashlib.sha256()
        for p in sorted((self.dir / step).iterdir()):
            if p.is_file() and p.name != "step.json":
                h.update(p.name.encode())
                h.update(A.sha256_file(p).encode())
        return h.hexdigest()[:16]

    def _dir(self, step: str) -> Path:
        d = self.dir / step
        d.mkdir(exist_ok=True)
        return d

    def _cached(self, step: str) -> bool:
        p = self.dir / step / "step.json"
        if step in self.force or "all" in self.force or not p.exists():
            return False
        meta = _load(p)
        if meta.get("digest") == self._digest(step) and meta.get("status") == "ok":
            if step not in self.isolated and step not in self.computed:
                log.info("[%s] reusing previous result", step)
                self.reused.append(step)
            if meta.get("metrics"):
                self.metrics[step] = meta["metrics"]
            return True
        return False

    def _finish(self, step: str, metrics: dict | None = None, **info) -> None:
        self.force.discard(step)  # computed now: later calls in this invocation reuse it
        self.computed.add(step)
        if metrics:
            self.metrics[step] = metrics
        _dump(self.dir / step / "step.json", {"step": step, "status": "ok", "digest": self._digest(step),
                                              "output_hash": self._output_hash(step),
                                              "finished_at": datetime.now(timezone.utc).isoformat(),
                                              "metrics": metrics, **info})

    # ---- steps --------------------------------------------------------------------------------------------
    def inspect(self) -> dict:
        step, d = "01_inspect", self._dir("01_inspect")
        if self._cached(step):
            return _load(d / "inspect.json")
        src = self.cfg.input.path
        t0 = time.perf_counter()
        probe = A.ffprobe(src)
        stream = next(s for s in probe["streams"] if s["codec_type"] == "audio")
        result = {
            "path": str(src),
            "sha256_before": A.sha256_file(src),
            "size_bytes": src.stat().st_size,
            "format": probe["format"].get("format_name"),
            "codec": stream.get("codec_name"),
            "sample_rate": int(stream.get("sample_rate", 0)),
            "sample_fmt": stream.get("sample_fmt"),
            "bits_per_sample": stream.get("bits_per_raw_sample") or stream.get("bits_per_sample") or
                               "n/a (lossy codec, decoded as float)",
            "channels": stream.get("channels"),
            "channel_layout": stream.get("channel_layout"),
            "duration_s": float(probe["format"]["duration"]),
            "bit_rate": int(probe["format"].get("bit_rate", 0)),
            "loudness": A.measure_loudness(src),
            "clipping": A.count_clipping(src),
            "ffprobe": probe,
        }
        if result["channels"] > 1:
            result["multichannel_note"] = "per-channel energy recorded; per-speaker channel check runs after diarization"
        else:
            result["multichannel_note"] = "mono recording: speakers cannot be separated by channel"
        _dump(d / "inspect.json", result)
        self._finish(step, {"step": step, "device": "cpu", "process_s": round(time.perf_counter() - t0, 3)})
        return result

    def prepare(self, info: dict) -> dict:
        step, d = "02_prepare", self._dir("02_prepare")
        if self._cached(step):
            return _load(d / "prepare.json")
        ex = self.cfg.input.excerpt
        excerpt = (ex.start_s, ex.end_s) if ex else None
        wav = d / "processing.wav"
        t0 = time.perf_counter()
        cmd = A.prepare_wav(self.cfg.input.path, wav, self.cfg.prepare.sample_rate, excerpt=excerpt)
        result = {"processing_wav": str(wav), "command": cmd, "sha256": A.sha256_file(wav),
                  "sample_rate": self.cfg.prepare.sample_rate, "channels": 1, "codec": "pcm_s16le",
                  "offset_s": excerpt[0] if excerpt else 0.0}
        audio, sr = A.read_wav(wav)
        result["duration_s"] = round(len(audio) / sr, 3)
        if info["channels"] > 1:
            result["channel_copies"] = []
            for ch in range(info["channels"]):
                p = d / f"channel_{ch}.wav"
                c = A.prepare_wav(self.cfg.input.path, p, self.cfg.prepare.sample_rate, channel=ch, excerpt=excerpt)
                result["channel_copies"].append({"path": str(p), "command": c, "sha256": A.sha256_file(p)})
        _dump(d / "prepare.json", result)
        self._finish(step, {"step": step, "device": "cpu", "process_s": round(time.perf_counter() - t0, 3)})
        return result

    def vad(self, prep: dict) -> list[Interval]:
        step, d = "03_vad", self._dir("03_vad")
        if self._cached(step):
            return [Interval(**i) for i in _load(d / "vad.json")["intervals"]]
        audio, sr = A.read_wav(Path(prep["processing_wav"]))
        provider = self.providers["vad"]
        with StepMonitor(step, provider.cfg.device, audio_s=len(audio) / sr) as mon:
            with mon.phase("load"):
                provider.load()
            with mon.phase("process"):
                intervals = provider.detect(audio, sr)
            provider.unload()
        speech = sum(i.duration for i in intervals)
        total = len(audio) / sr
        _dump(d / "vad.json", {"model": provider.info().model_dump(), "intervals": [i.model_dump() for i in intervals],
                               "speech_s": round(speech, 3), "nonspeech_s": round(total - speech, 3),
                               "speech_fraction": round(speech / total, 4)})
        mon.m.extra.update(speech_s=round(speech, 2), regions=len(intervals),
                           note="Silero VAD runs on CPU by design (see DECISIONS.md D-03)")
        self._finish(step, mon.m.to_dict())
        return intervals

    def diarization(self, prep: dict) -> tuple[list[SpeakerTurn], list[SpeakerTurn]]:
        step, d = "04_diarization", self._dir("04_diarization")
        if self._cached(step):
            data = _load(d / "diarization.json")
            return [SpeakerTurn(**t) for t in data["regular"]], [SpeakerTurn(**t) for t in data["exclusive"]]
        provider = self.providers["diarization"]
        provider.device = self.device
        inp = self.cfg.input
        free_gpu()
        baseline = gpu_used_mib()
        with StepMonitor(step, self.device, audio_s=prep["duration_s"]) as mon:
            with mon.phase("load"):
                provider.load()
            with mon.phase("process"):
                try:
                    res = provider.diarize(Path(prep["processing_wav"]), inp.num_speakers, inp.min_speakers,
                                           inp.max_speakers)
                except torch.OutOfMemoryError:
                    free_gpu()
                    seg_bs = max(1, self.cfg.diarization.segmentation_batch_size // 2)
                    emb_bs = max(1, self.cfg.diarization.embedding_batch_size // 2)
                    mon.event(f"CUDA OOM in diarization; retrying once with batch sizes {seg_bs}/{emb_bs}")
                    provider.set_batch_sizes(seg_bs, emb_bs)
                    res = provider.diarize(Path(prep["processing_wav"]), inp.num_speakers, inp.min_speakers,
                                           inp.max_speakers)
            model_info = provider.info()
            provider.unload()
        mon.after_unload(baseline)
        overlaps = overlap_regions(res.regular)
        speakers = sorted({t.speaker for t in res.regular})
        per_speaker = {s: round(sum(t.duration for t in res.regular if t.speaker == s), 2) for s in speakers}
        summary = {"speakers": speakers, "speaking_time_s": per_speaker,
                   "overlap_total_s": round(sum(o.duration for o in overlaps), 2),
                   "overlap_regions": len(overlaps), "turns_regular": len(res.regular),
                   "turns_exclusive": len(res.exclusive)}
        _dump(d / "diarization.json", {"model": model_info.model_dump(), "params": res.raw.get("params"),
                                       "summary": summary,
                                       "regular": [t.model_dump() for t in res.regular],
                                       "exclusive": [t.model_dump() for t in res.exclusive],
                                       "overlaps": [o.model_dump() for o in overlaps]})
        (d / "regular.rttm").write_text(res.raw["regular_rttm"], encoding="utf-8")
        (d / "exclusive.rttm").write_text(res.raw["exclusive_rttm"], encoding="utf-8")
        mon.m.extra.update(summary)
        self._finish(step, mon.m.to_dict())
        return res.regular, res.exclusive

    def asr(self, prep: dict, vad: list[Interval], exclusive: list[SpeakerTurn],
            overlaps: list[Interval]) -> list[dict]:
        step, d = "05_asr", self._dir("05_asr")
        if self._cached(step):
            return _load(d / "asr_raw.json")["units"]
        audio, sr = A.read_wav(Path(prep["processing_wav"]))
        if self.cfg.asr.segmentation == "turns":
            spans, unit_stats = units_from_turns(exclusive, overlaps, audio, sr, self.cfg.asr)
        elif self.cfg.asr.segmentation == "hybrid":
            spans, unit_stats = units_hybrid(exclusive, overlaps, vad, audio, sr, self.cfg.asr)
        else:
            spans, unit_stats = units_from_vad(vad, audio, sr, self.cfg.asr)
        units = [AsrUnit(id=f"u{i:05d}", start=s.start, end=s.end, speaker_hint=s.speaker_hint,
                         audio=audio[int(s.start * sr):int(s.end * sr)]) for i, s in enumerate(spans)]
        provider = self.providers["asr"]
        provider.device = self.device
        free_gpu()
        baseline = gpu_used_mib()
        unit_audio = sum(u.end - u.start for u in units)
        with StepMonitor(step, self.device, audio_s=prep["duration_s"]) as mon:
            with mon.phase("load"):
                provider.load()
            with mon.phase("process"):
                outputs = provider.transcribe(units, self.cfg.input.language)
            model_info, decoding = provider.info(), provider.decoding()
            for e in provider.events:
                mon.event(e)
            provider.unload()
        mon.after_unload(baseline)
        rows = [{"id": u.id, "start": round(u.start, 3), "end": round(u.end, 3), "speaker_hint": u.speaker_hint,
                 "text": o.text, "n_tokens": len(o.tokens), "tokens": o.tokens,
                 "generation": o.generation} for u, o in zip(units, outputs)]
        _dump(d / "asr_raw.json", {"model": model_info.model_dump(), "decoding": decoding,
                                   "segmentation": self.cfg.asr.segmentation, "unit_stats": unit_stats,
                                   "units": rows})
        mon.m.extra.update(unit_stats, transcribed_audio_s=round(unit_audio, 2),
                           rtf_on_transcribed_audio=round(mon.m.process_s / max(unit_audio, 1e-6), 5))
        self._finish(step, mon.m.to_dict())
        return rows

    def normalize(self, rows: list[dict]) -> dict[str, dict]:
        step, d = "05b_normalize", self._dir("05b_normalize")
        provider = self.providers["normalization"]
        if provider is None:
            return {}
        if self._cached(step):
            return _load(d / "normalized.json")["units"]
        provider.device = self.device
        free_gpu()
        baseline = gpu_used_mib()
        with StepMonitor(step, self.device) as mon:
            with mon.phase("load"):
                provider.load()
            with mon.phase("process"):
                texts, changes, details = provider.normalize([r["text"] for r in rows])
            model_info = provider.info()
            provider.unload()
        mon.after_unload(baseline)
        result = {r["id"]: {"text_normalized": t, "changes": c} for r, t, c in zip(rows, texts, changes)}
        _dump(d / "normalized.json", {"model": model_info.model_dump(), "details": details, "units": result})
        mon.m.extra.update(details)
        self._finish(step, mon.m.to_dict())
        return result

    def align(self, prep: dict, rows: list[dict], normalized: dict[str, dict]) -> dict[str, list[dict]]:
        """Word timestamps for every unit's raw text (aligned mode only)."""
        step, d = "05c_align", self._dir("05c_align")
        if self._cached(step):
            return _load(d / "words.json")["units"]
        audio, sr = A.read_wav(Path(prep["processing_wav"]))
        provider = self.providers["alignment"]
        provider.device = self.device
        free_gpu()
        baseline = gpu_used_mib()
        result, failed, mismatched = {}, 0, 0
        with StepMonitor(step, self.device, audio_s=prep["duration_s"]) as mon:
            with mon.phase("load"):
                provider.load()
            with mon.phase("process"):
                for r in rows:
                    tokens = r["text"].split()
                    norm_text = normalized.get(r["id"], {}).get("text_normalized")
                    norm_tokens = norm_text.split() if norm_text else None
                    if norm_tokens is not None and len(norm_tokens) != len(tokens):
                        norm_tokens, mismatched = None, mismatched + 1
                    seg_audio = audio[int(r["start"] * sr):int(r["end"] * sr)]
                    times = provider.align(seg_audio, sr, tokens, offset=r["start"])
                    failed += int(bool(tokens) and not any(t.aligned for t in times))
                    result[r["id"]] = [
                        {"text": tok, "text_normalized": norm_tokens[i] if norm_tokens else None,
                         "start": round(t.start, 3), "end": round(t.end, 3), "score": t.score, "aligned": t.aligned}
                        for i, (tok, t) in enumerate(zip(tokens, times))]
            model_info = provider.info()
            provider.unload()
        mon.after_unload(baseline)
        words = [w for ws in result.values() for w in ws]
        stats = {"units": len(rows), "words": len(words), "aligned_words": sum(w["aligned"] for w in words),
                 "units_failed": failed, "units_normalized_token_mismatch": mismatched,
                 "mean_score": round(sum(w["score"] for w in words) / max(len(words), 1), 3)}
        _dump(d / "words.json", {"model": model_info.model_dump(), "stats": stats, "units": result})
        mon.m.extra.update(stats)
        self._finish(step, mon.m.to_dict())
        return result

    def attribution(self, prep, rows, normalized, regular, overlaps, vad, words=None) -> list[Segment]:
        step, d = "06_attribution", self._dir("06_attribution")
        if self._cached(step):
            return [Segment(**s) for s in _load(d / "segments.json")]
        asr_meta = _load(self.dir / "05_asr" / "asr_raw.json")
        diar_meta = _load(self.dir / "04_diarization" / "diarization.json")["model"]
        vad_meta = _load(self.dir / "03_vad" / "vad.json")["model"]
        norm_model = self.cfg.normalization.model_id if normalized else None
        offset = prep["offset_s"]
        segments = []
        if words is not None:
            segments = self._word_segments(rows, words, regular, overlaps, vad, asr_meta, diar_meta, vad_meta,
                                           norm_model, offset)
            _dump(d / "segments.json", [s.model_dump() for s in segments])
            self._finish(step)
            return segments
        for r in sorted(rows, key=lambda r: r["start"]):
            a = attribute(r["start"], r["end"], regular, overlaps, vad, self.cfg.attribution)
            n = normalized.get(r["id"], {})
            segments.append(Segment(
                id=f"seg_{len(segments) + 1:05d}",
                start=round(r["start"] + offset, 3), end=round(r["end"] + offset, 3),
                speakers=a.speakers, primary_speaker=a.primary, speaker_shares=a.shares,
                speaker_uncertain=a.uncertain, overlap=a.overlap, overlap_seconds=a.overlap_seconds,
                vad_speech_seconds=a.vad_speech_seconds, text=r["text"],
                text_normalized=n.get("text_normalized"), normalization_changes=n.get("changes", 0),
                provenance=Provenance(
                    asr_unit_id=r["id"], asr_model=asr_meta["model"]["model_id"],
                    asr_revision=asr_meta["model"]["revision"], segmentation=asr_meta["segmentation"],
                    decoding=asr_meta["decoding"], diarization_model=diar_meta["model_id"],
                    diarization_revision=diar_meta["revision"], vad_model=vad_meta["model_id"],
                    normalization_model=norm_model,
                ),
            ))
        _dump(d / "segments.json", [s.model_dump() for s in segments])
        self._finish(step)
        return segments

    def _word_segments(self, rows, words, regular, overlaps, vad, asr_meta, diar_meta, vad_meta, norm_model,
                       offset) -> list[Segment]:
        """Aligned mode: split each long unit into segments of consecutive words by the same speaker."""
        align_model = _load(self.dir / "05c_align" / "words.json")["model"]["model_id"]
        segments = []
        for r in sorted(rows, key=lambda r: r["start"]):
            for g in group_words(r["id"], words.get(r["id"], []), regular, self.cfg.alignment.split_pause_s,
                                 self.cfg.alignment.snap_to_sentence_words):
                a = attribute(g.start, g.end, regular, overlaps, vad, self.cfg.attribution)
                speakers = ([g.speaker] if g.speaker else []) + [s for s in a.speakers if s != g.speaker]
                norm_tokens = [w["text_normalized"] for w in g.words]
                segments.append(Segment(
                    id=f"seg_{len(segments) + 1:05d}",
                    start=round(g.start + offset, 3), end=round(g.end + offset, 3),
                    speakers=speakers, primary_speaker=g.speaker, speaker_shares=a.shares,
                    speaker_uncertain=g.speaker is None or (not a.overlap and a.dominance < self.cfg.attribution.dominant_min_fraction),
                    overlap=a.overlap, overlap_seconds=a.overlap_seconds, vad_speech_seconds=a.vad_speech_seconds,
                    text=" ".join(w["text"] for w in g.words),
                    text_normalized=" ".join(norm_tokens) if all(t is not None for t in norm_tokens) else None,
                    normalization_changes=sum(1 for w in g.words if w["text_normalized"] not in (None, w["text"])),
                    words=[Word(**(w | {"start": round(w["start"] + offset, 3), "end": round(w["end"] + offset, 3)}))
                           for w in g.words],
                    provenance=Provenance(
                        asr_unit_id=r["id"], asr_model=asr_meta["model"]["model_id"],
                        asr_revision=asr_meta["model"]["revision"], segmentation=asr_meta["segmentation"],
                        alignment_model=align_model, decoding=asr_meta["decoding"],
                        diarization_model=diar_meta["model_id"], diarization_revision=diar_meta["revision"],
                        vad_model=vad_meta["model_id"], normalization_model=norm_model,
                    ),
                ))
        return segments

    def qc(self, segments: list[Segment], rows: list[dict]) -> list[Segment]:
        step, d = "07_qc", self._dir("07_qc")
        if self._cached(step):
            return [Segment(**s) for s in _load(d / "segments_flagged.json")]
        tokens = {r["id"]: r["n_tokens"] for r in rows}
        generation = {r["id"]: r.get("generation", {}) for r in rows}
        counts: dict[str, int] = {}
        for seg in segments:
            # In aligned mode several segments share one ASR unit; truncation is a property of the unit.
            seg.flags = flag_segment(seg, self.cfg.qc, self.cfg.asr.max_new_tokens, tokens[seg.provenance.asr_unit_id],
                                     min_word_score=self.cfg.alignment.min_word_score,
                                     truncated=generation[seg.provenance.asr_unit_id].get("truncated"))
            for f in seg.flags:
                counts[f.code] = counts.get(f.code, 0) + 1
        _dump(d / "segments_flagged.json", [s.model_dump() for s in segments])
        _dump(d / "flag_counts.json", counts)
        self._finish(step)
        return segments

    # ---- export, integrity, report --------------------------------------------------------------------
    def build_transcript(self, info: dict, prep: dict, segments: list[Segment]) -> Transcript:
        models = [ModelInfo(**_load(self.dir / s / f)["model"]) for s, f in
                  [("03_vad", "vad.json"), ("04_diarization", "diarization.json"), ("05_asr", "asr_raw.json")]]
        if (self.dir / "05b_normalize" / "normalized.json").exists() and self.providers["normalization"]:
            models.append(ModelInfo(**_load(self.dir / "05b_normalize" / "normalized.json")["model"]))
        if self.cfg.asr.segmentation == "aligned":
            models.append(ModelInfo(**_load(self.dir / "05c_align" / "words.json")["model"]))
        speakers = []
        diar = _load(self.dir / "04_diarization" / "diarization.json")["summary"]
        for label in diar["speakers"]:
            speakers.append(SpeakerSummary(label=label, speaking_time_s=diar["speaking_time_s"][label],
                                           segments=sum(1 for s in segments if label in s.speakers)))
        ex = self.cfg.input.excerpt
        return Transcript(
            run_id=self.run_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            language=self.cfg.input.language,
            source=SourceInfo(path=info["path"], sha256_before=info["sha256_before"], duration_s=info["duration_s"],
                              format={k: info[k] for k in ("format", "codec", "sample_rate", "channels", "bit_rate")},
                              excerpt=ex.model_dump() if ex else None),
            models=models,
            config=self.cfg.model_dump(mode="json"),
            config_digest=self.cfg.digest(),
            speakers=speakers,
            segments=segments,
        )

    def gpu_steps(self) -> list[str]:
        steps = ["04_diarization", "05_asr"]
        if self.providers["normalization"] is not None:
            steps.append("05b_normalize")
        if self.cfg.asr.segmentation == "aligned":
            steps.append("05c_align")
        return steps

    def _is_cached(self, step: str) -> bool:
        p = self.dir / step / "step.json"
        if step in self.force or "all" in self.force or not p.exists():
            return False
        meta = _load(p)
        return meta.get("digest") == self._digest(step) and meta.get("status") == "ok"

    def _run_gpu_steps_isolated(self) -> None:
        """Run every GPU step that is not cached in its own fresh process (spawn), in order. Each model then
        starts from the same clean CUDA state whatever ran before, which makes results independent of step
        reuse (DECISIONS D-19), and VRAM is fully released when the process exits."""
        import multiprocessing as mp

        if "all" in self.force:
            self.force = set(DEPENDS) | {"05c_align"}
        # cheap CPU steps first, in this process, so every child finds them on disk
        prep = self.prepare(self.inspect())
        self.vad(prep)
        ctx = mp.get_context("spawn")
        for step in self.gpu_steps():
            if self._is_cached(step):
                continue
            log.info("[%s] running in an isolated process", step)
            proc = ctx.Process(target=_child_step, args=(self.cfg.model_dump_json(), self.run_id, step,
                                                         sorted(self.force)))
            proc.start()
            proc.join()
            if proc.exitcode != 0:
                raise RuntimeError(f"isolated step {step} failed (exit code {proc.exitcode}); see the log above")
            self.isolated.append(step)
            # the step is finished on disk now; later steps of this process must not recompute it
            self.force.discard(step)
            self.metrics[step] = _load(self.dir / step / "step.json").get("metrics")

    def _steps(self, until: str | None = None):
        """Steps 01-07 in order (cached steps are loaded). Stops after `until` (used by isolated children)."""
        info = self.inspect()
        prep = self.prepare(info)
        vad = self.vad(prep)
        regular, exclusive = self.diarization(prep)
        if until == "04_diarization":
            return None
        overlaps = overlap_regions(regular)
        rows = self.asr(prep, vad, exclusive, overlaps)
        if until == "05_asr":
            return None
        normalized = self.normalize(rows)
        if until == "05b_normalize":
            return None
        words = self.align(prep, rows, normalized) if self.cfg.asr.segmentation == "aligned" else None
        if until == "05c_align":
            return None
        segments = self.attribution(prep, rows, normalized, regular, overlaps, vad, words)
        segments = self.qc(segments, rows)
        return info, prep, vad, regular, exclusive, overlaps, rows, normalized, words, segments

    def run(self) -> Transcript:
        set_determinism(self.cfg.run.seed)
        started = time.perf_counter()
        _dump(self.dir / "run.json", {"run_id": self.run_id, "config_digest": self.cfg.digest(),
                                      "started_at": datetime.now(timezone.utc).isoformat(),
                                      "gpu_state_start": gpu_state(), "host_power_start": host_power_state(),
                                      "config": self.cfg.model_dump(mode="json")})
        hardware_start = {"gpu": gpu_state(), "power": host_power_state()}

        if self.cfg.device.isolate_gpu_steps:
            self._run_gpu_steps_isolated()
        info, prep, vad, regular, exclusive, overlaps, rows, normalized, words, segments = self._steps()

        # 08 export
        d = self._dir("08_export")
        transcript = self.build_transcript(info, prep, segments)
        write_rttm(regular, Path(info["path"]).stem, self.dir / "diarization.rttm", offset=prep["offset_s"])
        reference = export_reference(transcript, self.cfg.reference, self.cfg.input.path, self.dir / "reference")

        # 09 integrity (before writing the final files so they carry the result)
        sha_after = A.sha256_file(self.cfg.input.path)
        transcript.source.sha256_after = sha_after
        transcript.source.integrity_ok = sha_after == info["sha256_before"]
        _dump(self._dir("09_integrity") / "integrity.json",
              {"sha256_before": info["sha256_before"], "sha256_after": sha_after,
               "unchanged": transcript.source.integrity_ok, "checked_at": datetime.now(timezone.utc).isoformat()})
        if not transcript.source.integrity_ok:
            log.error("ORIGINAL FILE CHANGED DURING THE RUN: %s != %s", sha_after, info["sha256_before"])

        write_json(transcript, self.dir / "transcript.json")
        write_txt(transcript, self.dir / "transcript.txt")
        write_srt(transcript, self.dir / "transcript.srt")
        if normalized:
            write_txt(transcript, self.dir / "transcript.normalized.txt", normalized=True)

        metrics = {
            "run_id": self.run_id,
            "audio_s": prep["duration_s"],
            "wall_s": round(time.perf_counter() - started, 2),  # this invocation only
            "reused_steps": self.reused,  # results taken from an earlier invocation of the same run id
            "isolated_gpu_steps": self.isolated,  # computed in this invocation, each in its own process
            "steps_compute_s": round(sum((m or {}).get("load_s", 0) + (m or {}).get("process_s", 0)
                                         for m in self.metrics.values()), 2),
            "hardware_start": hardware_start,
            "hardware_end": {"gpu": gpu_state(), "power": host_power_state()},
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "steps": self.metrics,
        }
        _dump(self.dir / "metrics.json", metrics)
        if (self.dir / "reference" / "excerpt.txt").exists():
            from .score import score_run

            try:
                score_run(self.dir)  # operator reference available: measure this run automatically
            except Exception as exc:  # scoring must never hide the transcript; report the failure instead
                log.error("scoring against the reference failed: %s", exc)
        from .report import write_report

        write_report(self.dir, transcript, metrics, reference)
        _dump(d / "step.json", {"step": "08_export", "status": "ok",
                                "files": sorted(p.name for p in self.dir.iterdir() if p.is_file())})
        return transcript


def _child_step(cfg_json: str, run_id: str, step: str, force: list[str]) -> None:
    """Entry point of an isolated GPU step (fresh process): load cached inputs, compute `step`, exit."""
    import os

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    cfg = PilotConfig.model_validate_json(cfg_json)
    set_determinism(cfg.run.seed)
    child = PilotRun(cfg, run_id, force={step} if step in force or "all" in force else set())
    child._steps(until=step)


def compare_transcripts(a: Path, b: Path) -> dict:
    """Determinism check: segment boundaries, speakers and text must match exactly."""
    ta = Transcript.model_validate_json(a.read_text(encoding="utf-8"))
    tb = Transcript.model_validate_json(b.read_text(encoding="utf-8"))
    def key(s):
        words = tuple((w.start, w.end, w.speaker, w.text) for w in s.words) if s.words else None
        return (s.start, s.end, tuple(s.speakers), s.overlap, s.text, s.text_normalized, words)
    sa, sb = [key(s) for s in ta.segments], [key(s) for s in tb.segments]
    diffs = [i for i, (x, y) in enumerate(zip(sa, sb)) if x != y]
    return {"segments_a": len(sa), "segments_b": len(sb), "identical": sa == sb,
            "differing_segments": len(diffs) + abs(len(sa) - len(sb)), "first_differences": diffs[:10]}
