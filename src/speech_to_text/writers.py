"""Write transcripts to disk in different formats."""

import json
from pathlib import Path

from .transcriber import TranscribedSegment

FORMATS = ("txt", "srt", "json")


def format_timestamp(seconds: float, sep: str = ",") -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_txt(segments: list[TranscribedSegment], timestamps: bool = False) -> str:
    if timestamps:
        lines = [f"[{format_timestamp(s.start, '.')[:-4]}] {s.text}" for s in segments]
        return "\n".join(lines) + "\n"
    return "\n\n".join(s.text for s in segments) + "\n"


def to_srt(segments: list[TranscribedSegment]) -> str:
    blocks = [
        f"{i}\n{format_timestamp(s.start)} --> {format_timestamp(s.end)}\n{s.text}\n"
        for i, s in enumerate(segments, start=1)
    ]
    return "\n".join(blocks)


def to_json(segments: list[TranscribedSegment], source: str, language: str) -> str:
    data = {
        "source": source,
        "language": language,
        "text": " ".join(s.text for s in segments),
        "segments": [{"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text} for s in segments],
    }
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def write_outputs(
    segments: list[TranscribedSegment],
    source: Path,
    output_dir: Path,
    formats: list[str],
    language: str,
    timestamps: bool = False,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        if fmt == "txt":
            content = to_txt(segments, timestamps)
        elif fmt == "srt":
            content = to_srt(segments)
        elif fmt == "json":
            content = to_json(segments, source.name, language)
        else:
            raise ValueError(f"Unknown format {fmt!r}")
        path = output_dir / f"{source.stem}.{fmt}"
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written
