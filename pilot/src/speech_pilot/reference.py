"""Section 7: choose and export an excerpt for a human reference, then score against it."""

import json
from pathlib import Path

from .audio import export_excerpt
from .config import ReferenceCfg
from .schema import Transcript


def choose_excerpt(t: Transcript, cfg: ReferenceCfg) -> dict | None:
    """Pick 3-5 minutes snapped to segment boundaries, preferring 2+ speakers, overlap and dense speech."""
    segs = t.segments
    best, best_score = None, -1.0
    for i, first in enumerate(segs):
        for j in range(i, len(segs)):
            start, end = first.start, segs[j].end
            length = end - start
            if length > cfg.excerpt_max_s:
                break
            if length < cfg.excerpt_min_s:
                continue
            window = segs[i : j + 1]
            speakers = {s.primary_speaker for s in window if s.primary_speaker}
            overlap = sum(s.overlap_seconds for s in window)
            speech = sum(s.vad_speech_seconds for s in window)
            warnings = sum(1 for s in window for f in s.flags if f.severity == "warning")
            score = (min(len(speakers), 4) * 10 + (5 if overlap > 0 else 0) + min(overlap, 30) / 3
                     + 10 * speech / length - warnings * 0.5)
            if len(speakers) >= 2 and score > best_score:
                best_score = score
                best = {"start_s": round(start, 3), "end_s": round(end, 3), "segments": [s.id for s in window],
                        "speakers": sorted(speakers), "overlap_s": round(overlap, 2),
                        "speech_s": round(speech, 2), "warning_flags": warnings, "score": round(score, 2)}
    return best


def export_reference(t: Transcript, cfg: ReferenceCfg, original: Path, out_dir: Path) -> dict | None:
    choice = choose_excerpt(t, cfg)
    if choice is None:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    # Segment times are absolute times in the original file, so the excerpt is cut from the original.
    choice["ffmpeg"] = export_excerpt(original, out_dir / "excerpt.wav", choice["start_s"], choice["end_s"])
    hyp = [s for s in t.segments if s.id in set(choice["segments"])]
    (out_dir / "hypothesis.txt").write_text(
        "\n".join(f"{s.primary_speaker}: {s.text}" for s in hyp) + "\n", encoding="utf-8")
    (out_dir / "excerpt.json").write_text(json.dumps(choice, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "README.md").write_text(_instructions(choice), encoding="utf-8")
    return choice


def _instructions(c: dict) -> str:
    from .export import hms

    return f"""# Reference excerpt

`excerpt.wav` = original recording from {hms(c['start_s'])} to {hms(c['end_s'])}
({c['end_s'] - c['start_s']:.0f} s; speakers by diarization: {', '.join(c['speakers'])}; overlap {c['overlap_s']} s).

Please transcribe it **by hand, verbatim**, without looking at `hypothesis.txt` (the machine output),
so the reference is not biased by it.

1. `excerpt.txt`: one speaker turn per line, `LABEL: text`. Use your own labels (e.g. `A:`, `B:`);
   they don't need to match the diarization labels. Write numbers as you hear them (digits or words).
   Mark unintelligible words as `[...]`.
2. Optional `excerpt.rttm`: speaker turns in RTTM format, with times **relative to the start of
   excerpt.wav** (0.0 = start), e.g. `SPEAKER excerpt 1 12.40 3.10 <NA> <NA> A <NA> <NA>`.

Then run: `uv run speech-pilot score --run <this run folder>`
"""
