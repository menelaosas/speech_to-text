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


@dataclass
class WordGroup:
    unit_id: str
    speaker: str | None
    words: list[dict]  # {text, text_normalized, start, end, score, aligned, speaker}

    @property
    def start(self) -> float:
        return self.words[0]["start"]

    @property
    def end(self) -> float:
        return max(self.words[-1]["end"], self.start + 0.02)


def word_speaker(start: float, end: float, regular: list[SpeakerTurn], tolerance: float = 0.3) -> str | None:
    """Speaker most active during the word (overlap-aware diarization); if none, the nearest turn within
    `tolerance` seconds (words often sit just outside a turn edge)."""
    end = max(end, start + 0.02)
    best, best_o = None, 0.0
    for t in regular:
        o = overlap_s(start, end, t.start, t.end)
        if o > best_o:
            best, best_o = t.speaker, o
    if best:
        return best
    mid = (start + end) / 2
    near = min(regular, key=lambda t: min(abs(mid - t.start), abs(mid - t.end)), default=None)
    if near and min(abs(mid - near.start), abs(mid - near.end)) <= tolerance:
        return near.speaker
    return None


_SENTENCE_END = (".", ";", ";", "?", "!", "…")


def _snap_boundaries(groups: list[WordGroup], snap_words: int) -> list[WordGroup]:
    """Move each speaker-change boundary to the nearest sentence end (ASR punctuation) within
    `snap_words` words, so a sentence is not cut between two speakers by word-timing jitter."""
    if snap_words <= 0 or len(groups) < 2:
        return groups
    for i in range(len(groups) - 1):
        a, b = groups[i], groups[i + 1]
        if a.speaker == b.speaker or not a.words or not b.words:
            continue
        if a.words[-1]["text"].endswith(_SENTENCE_END):
            continue  # already at a sentence end
        seq = a.words + b.words
        cut = len(a.words)
        best = None
        for k in range(max(1, cut - snap_words), min(len(seq) - 1, cut + snap_words) + 1):
            if seq[k - 1]["text"].endswith(_SENTENCE_END) and (best is None or abs(k - cut) < abs(best - cut)):
                best = k
        if best is not None and best != cut:
            a.words, b.words = seq[:best], seq[best:]
            for w in a.words:
                w["speaker"] = a.speaker
            for w in b.words:
                w["speaker"] = b.speaker
    merged: list[WordGroup] = []
    for g in groups:
        if not g.words:
            continue
        if merged and merged[-1].speaker == g.speaker:
            merged[-1].words += g.words
        else:
            merged.append(g)
    return merged


def group_words(unit_id: str, words: list[dict], regular: list[SpeakerTurn], split_pause_s: float,
                snap_words: int = 0) -> list[WordGroup]:
    """Consecutive words of the same speaker form one group; a long pause also starts a new group.
    Unaligned tokens (punctuation, Latin letters) stay with the previous word. Speaker changes are then
    snapped to the nearest sentence end within `snap_words` words."""
    groups: list[WordGroup] = []
    for w in words:
        spk = word_speaker(w["start"], w["end"], regular) if w["aligned"] else (
            groups[-1].speaker if groups else None)
        w = w | {"speaker": spk}
        if groups and (groups[-1].speaker == spk or not w["aligned"]) and \
                w["start"] - groups[-1].words[-1]["end"] <= split_pause_s:
            groups[-1].words.append(w)
        else:
            groups.append(WordGroup(unit_id, spk, [w]))
    return _snap_boundaries(groups, snap_words)


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
