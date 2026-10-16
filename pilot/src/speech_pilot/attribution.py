"""Assign each transcribed unit to speakers using the overlap-aware diarization."""

from dataclasses import dataclass

from .config import AttributionCfg
from .schema import Interval, SpeakerShare, SpeakerTurn
from .units import overlap_s, total_overlap


@dataclass
class Attribution:
    speakers: list[str]
    primary: str | None
    shares: list[SpeakerShare]
    uncertain: bool
    overlap: bool
    overlap_seconds: float
    vad_speech_seconds: float
    dominance: float


def attribute(start: float, end: float, regular: list[SpeakerTurn], overlaps: list[Interval],
              vad: list[Interval], cfg: AttributionCfg) -> Attribution:
    duration = max(end - start, 1e-6)
    per_speaker: dict[str, float] = {}
    for t in regular:
        o = overlap_s(start, end, t.start, t.end)
        if o > 0:
            per_speaker[t.speaker] = per_speaker.get(t.speaker, 0.0) + o
    ranked = sorted(per_speaker.items(), key=lambda kv: (-kv[1], kv[0]))
    ov = total_overlap(start, end, overlaps)
    diarized = max(sum(per_speaker.values()) - ov, 1e-6)  # union of diarized speech in the unit

    primary = ranked[0][0] if ranked else None
    dominance = ranked[0][1] / diarized if ranked else 0.0
    is_overlap = ov >= cfg.overlap_min_s or ov / duration >= cfg.overlap_min_fraction
    # Secondary speakers are listed when they are a real part of the unit, not a few frames.
    speakers = [s for s, sec in ranked if s == primary or sec >= 0.5 or sec / duration >= 0.15]
    uncertain = primary is None or (not is_overlap and dominance < cfg.dominant_min_fraction)
    return Attribution(
        speakers=speakers,
        primary=primary,
        shares=[SpeakerShare(speaker=s, seconds=round(sec, 3)) for s, sec in ranked],
        uncertain=uncertain,
        overlap=is_overlap,
        overlap_seconds=round(ov, 3),
        vad_speech_seconds=round(total_overlap(start, end, vad), 3),
        dominance=round(min(dominance, 1.0), 3),
    )
