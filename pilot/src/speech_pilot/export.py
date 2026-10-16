"""Write the transcript as JSON (canonical), TXT, SRT and the diarization as RTTM."""

from pathlib import Path

from .schema import SpeakerTurn, Transcript


def hms(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def speaker_label(seg) -> str:
    if seg.overlap and len(seg.speakers) > 1:
        return " + ".join(seg.speakers) + " [OVERLAP]"
    label = seg.primary_speaker or "UNKNOWN"
    return label + "?" if seg.speaker_uncertain else label


def write_json(t: Transcript, path: Path) -> None:
    path.write_text(t.model_dump_json(indent=2), encoding="utf-8")


def write_txt(t: Transcript, path: Path, normalized: bool = False) -> None:
    layer = ("machine-normalised layer (Greek final ς restored by XLM-R) - NOT the canonical text"
             if normalized else "raw ASR output (canonical, unedited)")
    lines = [
        f"# Source: {Path(t.source.path).name}",
        f"# SHA-256: {t.source.sha256_before}",
        f"# Run: {t.run_id}   Text: {layer}",
        f"# {t.accuracy_note}",
        "# Legend: 'A + B [OVERLAP]' = overlapping speech, 'SPEAKER_X?' = uncertain speaker, '<<flags>>' = review flags",
        "",
    ]
    for seg in t.segments:
        text = seg.text_normalized if normalized and seg.text_normalized is not None else seg.text
        warn = [f.code for f in seg.flags if f.severity == "warning" and f.code != "speaker_uncertain"]
        suffix = f"  <<{', '.join(warn)}>>" if warn else ""
        lines.append(f"[{hms(seg.start)} – {hms(seg.end)}] {speaker_label(seg)}: {text}{suffix}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_srt(t: Transcript, path: Path) -> None:
    cues = []
    for i, seg in enumerate(t.segments, start=1):
        cues.append(f"{i}\n{srt_time(seg.start)} --> {srt_time(seg.end)}\n[{speaker_label(seg)}] {seg.text}\n")
    path.write_text("\n".join(cues), encoding="utf-8")


def write_rttm(turns: list[SpeakerTurn], uri: str, path: Path, offset: float = 0.0) -> None:
    """Standard RTTM: SPEAKER <uri> 1 <onset> <duration> <NA> <NA> <speaker> <NA> <NA>."""
    uri = uri.replace(" ", "_")
    lines = [
        f"SPEAKER {uri} 1 {t.start + offset:.3f} {t.end - t.start:.3f} <NA> <NA> {t.speaker} <NA> <NA>"
        for t in sorted(turns, key=lambda t: (t.start, t.speaker))
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
