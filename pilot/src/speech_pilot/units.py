"""Turn diarization turns or VAD regions into ASR units (each <= max_unit_s, transcribed once)."""

from dataclasses import dataclass

import numpy as np

from .audio import split_long
from .config import AsrCfg
from .schema import Interval, SpeakerTurn


@dataclass
class UnitSpan:
    start: float
    end: float
    speaker_hint: str | None  # exclusive-diarization speaker, "MIXED" if merged across an overlap


def overlap_s(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def total_overlap(start: float, end: float, intervals: list[Interval]) -> float:
    return sum(overlap_s(start, end, iv.start, iv.end) for iv in intervals)


def overlap_regions(turns: list[SpeakerTurn]) -> list[Interval]:
    """Intervals where two or more speakers are active in the (overlap-aware) diarization."""
    events = sorted([(t.start, 1) for t in turns] + [(t.end, -1) for t in turns], key=lambda e: (e[0], e[1]))
    regions, active, opened = [], 0, 0.0
    for time, delta in events:
        before = active
        active += delta
        if before < 2 <= active:
            opened = time
        elif before >= 2 > active and time > opened:
            regions.append(Interval(start=opened, end=time))
    return regions


def _in_any(t: float, regions: list[Interval], tol: float = 0.05) -> bool:
    return any(r.start - tol <= t <= r.end + tol for r in regions)


def _split(spans: list[UnitSpan], audio: np.ndarray, sr: int, cfg: AsrCfg) -> list[UnitSpan]:
    out = []
    for s in spans:
        if s.end - s.start <= cfg.max_unit_s:
            out.append(s)
            continue
        for a, b in split_long(audio, sr, s.start, s.end, cfg.max_unit_s, cfg.split_search_s):
            out.append(UnitSpan(a, b, s.speaker_hint))
    return out


def units_from_turns(exclusive: list[SpeakerTurn], overlaps: list[Interval], audio: np.ndarray, sr: int,
                     cfg: AsrCfg) -> tuple[list[UnitSpan], dict]:
    """One unit per speaker turn. Consecutive turns are merged when they belong to the same speaker
    (short gap) or when the boundary between them lies in an overlap region, so overlapping speech is
    transcribed once and marked as overlap instead of being cut into false sequential turns."""
    turns = sorted((t for t in exclusive if t.end > t.start), key=lambda t: t.start)
    spans: list[UnitSpan] = []
    merged_overlap = 0
    for t in turns:
        if spans:
            cur = spans[-1]
            gap = t.start - cur.end
            fits = t.end - cur.start <= cfg.max_unit_s
            same = cur.speaker_hint == t.speaker
            across_overlap = gap <= cfg.merge_gap_s and (_in_any(cur.end, overlaps) or _in_any(t.start, overlaps))
            if fits and gap <= cfg.merge_gap_s and same:
                cur.end = max(cur.end, t.end)
                continue
            if fits and across_overlap and not same:
                cur.end = max(cur.end, t.end)
                cur.speaker_hint = "MIXED"
                merged_overlap += 1
                continue
        spans.append(UnitSpan(t.start, t.end, t.speaker))
    spans = _split(spans, audio, sr, cfg)
    kept = [s for s in spans if s.end - s.start >= cfg.min_unit_s]
    dropped = [s for s in spans if s.end - s.start < cfg.min_unit_s]
    return kept, {
        "turns_in": len(turns),
        "units": len(kept),
        "merged_across_overlap": merged_overlap,
        "dropped_short_units": len(dropped),
        "dropped_short_seconds": round(sum(s.end - s.start for s in dropped), 3),
    }


def merge_short_units(spans: list[UnitSpan], cfg: AsrCfg) -> tuple[list[UnitSpan], int]:
    """Give very short units context: merge any unit shorter than `min_context_s` into its nearest
    neighbour (gap <= `context_merge_gap_s`, result <= `max_unit_s`). Units that change speaker this way
    get speaker_hint "MIXED"; attribution then lists every speaker in the unit."""
    spans = [UnitSpan(s.start, s.end, s.speaker_hint) for s in sorted(spans, key=lambda s: s.start)]
    merges = 0
    changed = True
    while changed:
        changed = False
        for i, s in enumerate(spans):
            if s.end - s.start >= cfg.min_context_s:
                continue
            candidates = []
            for j in (i - 1, i + 1):
                if 0 <= j < len(spans):
                    n = spans[j]
                    gap = s.start - n.end if j < i else n.start - s.end
                    if gap <= cfg.context_merge_gap_s and max(s.end, n.end) - min(s.start, n.start) <= cfg.max_unit_s:
                        candidates.append((gap, n.end - n.start, j))
            if not candidates:
                continue
            _, _, j = min(candidates)
            n = spans[j]
            merged = UnitSpan(min(s.start, n.start), max(s.end, n.end),
                              n.speaker_hint if n.speaker_hint == s.speaker_hint else "MIXED")
            lo, hi = min(i, j), max(i, j)
            spans[lo : hi + 1] = [merged]
            merges += 1
            changed = True
            break
    return spans, merges


def uncovered(intervals: list[Interval], cover: list) -> list[tuple[float, float]]:
    """Parts of `intervals` not covered by any interval in `cover` (both have .start/.end)."""
    cover = sorted(cover, key=lambda c: c.start)
    gaps = []
    for iv in intervals:
        cur = iv.start
        for c in cover:
            if c.end <= cur or c.start >= iv.end:
                continue
            if c.start > cur:
                gaps.append((cur, c.start))
            cur = max(cur, c.end)
        if cur < iv.end:
            gaps.append((cur, iv.end))
    return gaps


def units_hybrid(exclusive: list[SpeakerTurn], overlaps: list[Interval], vad: list[Interval], audio: np.ndarray,
                 sr: int, cfg: AsrCfg) -> tuple[list[UnitSpan], dict]:
    """Diarization turns (speaker attribution) + VAD speech that no turn covers (nothing dropped
    silently) + a minimum context per unit (fewer hallucinations on very short clips)."""
    spans, stats = units_from_turns(exclusive, overlaps, audio, sr, cfg.model_copy(update={"min_unit_s": 0.0}))
    vad_only = [(a, b) for a, b in uncovered(vad, exclusive) if b - a >= cfg.vad_fill_min_s]
    spans += [UnitSpan(a, b, None) for a, b in vad_only]
    spans, merges = merge_short_units(spans, cfg)
    spans = _split(spans, audio, sr, cfg)
    kept = [s for s in spans if s.end - s.start >= cfg.min_unit_s]
    dropped = [s for s in spans if s.end - s.start < cfg.min_unit_s]
    left = uncovered(vad, [Interval(start=s.start, end=s.end) for s in kept])
    stats.update(
        units=len(kept),
        vad_speech_without_diarization_s=round(sum(b - a for a, b in uncovered(vad, exclusive)), 2),
        vad_only_units_added=len(vad_only),
        merged_for_context=merges,
        dropped_short_units=len(dropped),
        dropped_short_seconds=round(sum(s.end - s.start for s in dropped), 3),
        still_short_units=sum(1 for s in kept if s.end - s.start < cfg.min_context_s),
        vad_speech_not_transcribed_s=round(sum(b - a for a, b in left), 2),
    )
    return kept, stats


def units_from_vad(vad: list[Interval], audio: np.ndarray, sr: int, cfg: AsrCfg) -> tuple[list[UnitSpan], dict]:
    """Merge consecutive VAD regions into units up to max_unit_s (more context per ASR call)."""
    spans: list[UnitSpan] = []
    for iv in sorted(vad, key=lambda v: v.start):
        if spans and iv.end - spans[-1].start <= cfg.max_unit_s:
            spans[-1].end = iv.end
        else:
            spans.append(UnitSpan(iv.start, iv.end, None))
    spans = _split(spans, audio, sr, cfg)
    kept = [s for s in spans if s.end - s.start >= cfg.min_unit_s]
    return kept, {"vad_regions_in": len(vad), "units": len(kept), "dropped_short_units": len(spans) - len(kept)}
