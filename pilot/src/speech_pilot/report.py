"""REPORT.md: what was run, measurements, flags and what was not tested."""

import json
from pathlib import Path

from .export import hms, speaker_label
from .qc import suspicion
from .schema import Transcript


def _fmt(v, nd=1):
    return "–" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def write_report(run_dir: Path, t: Transcript, metrics: dict, reference: dict | None) -> None:
    steps = metrics["steps"]
    diar = json.loads((run_dir / "04_diarization" / "diarization.json").read_text(encoding="utf-8"))["summary"]
    vad = json.loads((run_dir / "03_vad" / "vad.json").read_text(encoding="utf-8"))
    asr = json.loads((run_dir / "05_asr" / "asr_raw.json").read_text(encoding="utf-8"))
    norm_path = run_dir / "05b_normalize" / "normalized.json"
    norm = json.loads(norm_path.read_text(encoding="utf-8"))["details"] if norm_path.exists() else None
    counts = json.loads((run_dir / "07_qc" / "flag_counts.json").read_text(encoding="utf-8"))
    segs = t.segments
    excerpt = t.source.excerpt

    L = [f"# Pilot report — run `{t.run_id}`", ""]
    L += ["> **Accuracy has not been measured.** No human reference transcript exists yet. Nothing in this "
          "report says the transcript is correct; it reports what ran, how long it took and where a "
          "reviewer should listen first.", ""]

    L += ["## Input and integrity", "",
          "| Item | Value |", "| --- | --- |",
          f"| File | `{Path(t.source.path).name}` |",
          f"| Format | {t.source.format['codec']}, {t.source.format['sample_rate']} Hz, "
          f"{t.source.format['channels']} ch, {t.source.format['bit_rate'] // 1000} kb/s |",
          f"| Duration | {hms(t.source.duration_s)} ({t.source.duration_s:.1f} s) |",
          f"| Processed | {'excerpt ' + hms(excerpt['start_s']) + '–' + hms(excerpt['end_s']) if excerpt else 'whole file'} "
          f"({metrics['audio_s']:.1f} s) |",
          f"| SHA-256 before | `{t.source.sha256_before}` |",
          f"| SHA-256 after | `{t.source.sha256_after}` |",
          f"| Original unchanged | **{'yes' if t.source.integrity_ok else 'NO'}** |", ""]

    L += ["## Steps, timing and VRAM", "",
          "| Step | Device | Load s | Process s | RTF | Peak alloc MiB | Peak reserved MiB | VRAM after unload MiB | Events |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, m in steps.items():
        if not m:
            continue
        L.append(f"| {name} | {m.get('device', '–')} | {_fmt(m.get('load_s'), 2)} | {_fmt(m.get('process_s'), 2)} | "
                 f"{_fmt(m.get('rtf'), 4)} | {_fmt(m.get('peak_allocated_mib'))} | {_fmt(m.get('peak_reserved_mib'))} | "
                 f"{_fmt(m.get('gpu_used_after_mib'), 0)} | {'; '.join(m.get('events', [])) or '–'} |")
    hs, he = metrics["hardware_start"], metrics["hardware_end"]
    reused = metrics.get("reused_steps", [])
    if reused:
        timing = (f"This invocation reused {len(reused)} step(s) from an earlier invocation "
                  f"({', '.join(reused)}), so its wall time ({metrics['wall_s']:.1f} s) is not the processing "
                  f"time. Summed load + process time of all steps: {metrics['steps_compute_s']:.1f} s.")
    else:
        timing = (f"Total wall time {metrics['wall_s']:.1f} s for {metrics['audio_s']:.1f} s of audio "
                  f"(overall RTF {metrics['wall_s'] / metrics['audio_s']:.4f}); all steps computed in this run.")
    L += ["", timing + " RTF = processing time / audio duration.", "",
          "| GPU state | Start | End |", "| --- | --- | --- |"]
    for key in ("temperature.gpu", "clocks.sm", "power.draw", "pstate", "memory.used"):
        L.append(f"| {key} | {hs['gpu'].get(key, '–')} | {he['gpu'].get(key, '–')} |")
    L.append(f"| AC power / profile | {hs['power'].get('ac_online', '–')} / {hs['power'].get('platform_profile', '–')} | "
             f"{he['power'].get('ac_online', '–')} / {he['power'].get('platform_profile', '–')} |")
    L.append("")

    L += ["## VAD and diarization", "",
          f"- Speech (Silero VAD, CPU): {vad['speech_s']:.1f} s; non-speech {vad['nonspeech_s']:.1f} s "
          f"({100 * vad['speech_fraction']:.1f}% speech) in {len(vad['intervals'])} regions.",
          f"- Speakers found by pyannote: **{len(diar['speakers'])}** "
          f"({'fixed by config' if t.config['input']['num_speakers'] else 'estimated, count not given'}).",
          f"- Overlap (2+ speakers active): {diar['overlap_total_s']:.1f} s in {diar['overlap_regions']} regions.", "",
          "| Speaker | Speaking time | Segments (incl. overlap) |", "| --- | --- | --- |"]
    for s in t.speakers:
        L.append(f"| {s.label} | {hms(s.speaking_time_s)} ({s.speaking_time_s:.0f} s) | {s.segments} |")
    L.append("")

    us = asr["unit_stats"]
    if asr["segmentation"] == "vad":
        seg_line = f"{us['units']} units from {us.get('vad_regions_in')} VAD regions."
    else:
        seg_line = (f"{us['units']} units from {us.get('turns_in')} exclusive diarization turns; "
                    f"{us.get('merged_across_overlap', 0)} merges across overlap boundaries")
        if asr["segmentation"] == "hybrid":
            seg_line += (f", {us.get('vad_only_units_added', 0)} units added for VAD speech without a diarization "
                         f"turn ({us.get('vad_speech_without_diarization_s', 0)} s), {us.get('merged_for_context', 0)} "
                         f"merges to give units < {t.config['asr']['min_context_s']} s more context, "
                         f"{us.get('still_short_units', 0)} units still shorter than that")
        seg_line += (f"; {us.get('dropped_short_units', 0)} units < {t.config['asr']['min_unit_s']} s not transcribed "
                     f"({us.get('dropped_short_seconds', 0)} s).")
        if "vad_speech_not_transcribed_s" in us:
            from .schema import Interval
            from .units import uncovered

            gaps = uncovered([Interval(**i) for i in vad["intervals"]],
                             [Interval(start=u["start"], end=u["end"]) for u in asr["units"]])
            largest = max((b - a for a, b in gaps), default=0.0)
            seg_line += (f" VAD speech outside every ASR unit: {us['vad_speech_not_transcribed_s']} s in "
                         f"{len(gaps)} pieces, the largest {largest:.2f} s (edges of VAD regions, which are "
                         f"padded by {t.config['vad']['speech_pad_ms']} ms).")
    L += ["## ASR", "",
          f"- Cohere Transcribe `{asr['model']['revision']}`, language `{t.language}`, {asr['model']['dtype']} on "
          f"{asr['model']['device']}; decoding {asr['decoding']}.",
          f"- Segmentation: **{asr['segmentation']}** (see DECISIONS.md D-11). " + seg_line,
          f"- Segments in transcript: {len(segs)}; overlap segments: {sum(s.overlap for s in segs)}; "
          f"uncertain speaker: {sum(s.speaker_uncertain for s in segs)}.", ""]
    if norm:
        L += ["## Normalised text layer (not canonical)", "",
              f"Greek final ς restored by XLM-RoBERTa-large in `text_normalized` / `transcript.normalized.txt`. "
              f"Raw ς rate {100 * norm['raw_sigma_rate']:.1f}% (normal Greek ≈ 15–18%), margin {norm['margin']:+.0f}, "
              f"{norm['candidates']} candidates, {norm['changed_words']} words changed. The canonical `text` is "
              f"the raw Cohere output and is never edited.", ""]

    L += ["## Quality flags", "", "| Flag | Segments |", "| --- | --- |"]
    for code, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        L.append(f"| {code} | {n} |")
    if not counts:
        L.append("| (none) | 0 |")
    L += ["", "### 10 most suspicious segments (listen to these first)", "",
          "| # | Time | Speaker | Flags | Text (start) |", "| --- | --- | --- | --- | --- |"]
    ranked = sorted((s for s in segs if s.flags), key=lambda s: (-suspicion(s), s.start))[:10]
    for i, s in enumerate(ranked, 1):
        flags = ", ".join(f.code for f in s.flags)
        text = s.text[:90].replace("|", "/") + ("…" if len(s.text) > 90 else "")
        L.append(f"| {i} | {hms(s.start)}–{hms(s.end)} | {speaker_label(s)} | {flags} | {text} |")
    L.append("")

    L += ["## Reference excerpt for manual transcription", ""]
    if reference:
        L += [f"`reference/excerpt.wav`: {hms(reference['start_s'])}–{hms(reference['end_s'])} "
              f"({reference['end_s'] - reference['start_s']:.0f} s), speakers {', '.join(reference['speakers'])}, "
              f"overlap {reference['overlap_s']} s. Instructions in `reference/README.md`; "
              "score with `uv run speech-pilot score --run <run dir>`.", ""]
    else:
        L += ["No window of 3–5 minutes with two or more speakers was found.", ""]

    det_path = run_dir / "determinism.json"
    det = json.loads(det_path.read_text()) if det_path.exists() else None
    L += ["## Determinism", ""]
    if det:
        L += [f"Compared with run `{det['other_run']}` (same config, fresh run): "
              f"**{'identical' if det['identical'] else 'NOT identical'}**: {det['segments_a']} vs "
              f"{det['segments_b']} segments, {det['differing_segments']} differing (boundaries, speakers, "
              "overlap, raw and normalised text compared).", ""]
    else:
        L += ["Not checked yet (`uv run speech-pilot compare runs/<a> runs/<b>`).", ""]

    L += ["## Tested vs not tested", "",
          "**Tested:** the pipeline ran end to end with one command; the devices above were recorded per step "
          "(Cohere and pyannote on CUDA, Silero VAD on CPU by design); VRAM returned to baseline after each "
          "GPU model; the original file's SHA-256 is unchanged; all outputs were written and transcript.json "
          "re-validates against the schema"
          + ("; a second run produced the same transcript." if det and det["identical"] else "."), "",
          "**Not tested:** transcription accuracy (WER/CER), diarization accuracy (DER), whether the "
          f"{len(diar['speakers'])} speaker labels correspond to {len(diar['speakers'])} real people, whether the "
          "overlap regions are real, and the quality of the ς-normalised layer on this recording. "
          "These need the reference excerpt. BF16 vs FP32 accuracy is also open (DECISIONS D-09).", ""]

    rec = []
    if not t.config["input"]["num_speakers"]:
        rec.append(f"Confirm how many people speak. pyannote estimated {len(diar['speakers'])}; the smallest "
                   f"label has {min(diar['speaking_time_s'].values()):.0f} s of speech. If the count is known, set "
                   "`input.num_speakers` and re-run (steps 04 onward are recomputed automatically).")
    ov_share = sum(s.overlap for s in segs) / max(len(segs), 1)
    if ov_share > 0.25:
        rec.append(f"{100 * ov_share:.0f}% of segments contain overlapping speech ({diar['overlap_total_s']:.0f} s). "
                   "Cohere transcribes overlap as one stream, so these segments are the least reliable. Speech "
                   "separation (out of scope for the pilot) is the next lever if DER/WER confirm this.")
    if norm and norm["raw_sigma_rate"] < 0.05:
        rec.append("The raw text has almost no final ς (upstream model bug). Read `transcript.normalized.txt` for "
                   "comprehension, and cite the raw `transcript.txt`/`.json` as the canonical machine output.")
    rec.append("Transcribe `reference/excerpt.wav` by hand (and optionally its RTTM), then run "
               "`speech-pilot score`: this is the only way to turn the open questions above into measurements.")
    rec.append("Stretch steps after scoring: DeepFilterNet3 on a noisy excerpt (compare WER with/without), and "
               "Greek CTC forced alignment for word timestamps on a short excerpt.")
    L += ["## Recommendations for the next step", ""] + [f"{i}. {r}" for i, r in enumerate(rec, 1)] + [""]
    (run_dir / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
