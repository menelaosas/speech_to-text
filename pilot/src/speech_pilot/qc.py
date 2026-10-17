"""Quality flags. Flags point a reviewer at segments to listen to; they never change the text."""

import re

from .config import QcCfg
from .schema import Flag, Segment

_WORD = re.compile(r"\w+", re.UNICODE)
_SENTENCE = re.compile(r"[^.!?;;]+[.!?;;]?")

# Weight of each flag when ranking the most suspicious segments.
WEIGHTS = {
    "known_hallucination_phrase": 4.0,
    "repetition_loop": 3.0,
    "text_in_nonspeech": 3.0,
    "empty_output": 3.0,
    "possible_truncation": 3.0,
    "words_per_second_high": 2.0,
    "words_per_second_low": 2.0,
    "speaker_uncertain": 1.0,
    "no_diarized_speaker": 1.5,
    "low_alignment_confidence": 1.5,
    "overlap": 0.5,
}


def repetition(text: str, min_repeats: int) -> str | None:
    """Detect decoder loops: the same sentence or the same 1-4 word n-gram repeated consecutively."""
    sentences = [s.strip().lower() for s in _SENTENCE.findall(text) if s.strip()]
    run = 1
    for a, b in zip(sentences, sentences[1:]):
        run = run + 1 if a == b else 1
        if run >= min_repeats:
            return f"sentence repeated {run}+ times: {a[:60]!r}"
    words = [w.lower() for w in _WORD.findall(text)]
    for n in range(1, 5):
        for i in range(len(words) - n * min_repeats + 1):
            gram = words[i : i + n]
            if all(words[i + k * n : i + (k + 1) * n] == gram for k in range(1, min_repeats)):
                return f"{n}-word sequence repeated {min_repeats}+ times: {' '.join(gram)!r}"
    return None


def flag_segment(seg: Segment, cfg: QcCfg, max_new_tokens: int, n_tokens: int,
                 min_word_score: float | None = None) -> list[Flag]:
    flags: list[Flag] = []
    words = len(_WORD.findall(seg.text))
    duration = seg.end - seg.start
    speech = seg.vad_speech_seconds

    if words and speech < (1 - cfg.nonspeech_max_fraction) * duration:
        flags.append(Flag(code="text_in_nonspeech", severity="warning",
                          detail=f"{words} words but VAD speech only {speech:.1f}s of {duration:.1f}s"))
    lowered = seg.text.lower()
    hits = [p for p in cfg.hallucination_phrases if p.lower() in lowered]
    if hits:
        flags.append(Flag(code="known_hallucination_phrase", severity="warning",
                          detail=f"contains {', '.join(repr(h) for h in hits)} (memorised subtitle text)"))
    if (rep := repetition(seg.text, cfg.repetition_min_repeats)) is not None:
        flags.append(Flag(code="repetition_loop", severity="warning", detail=rep))
    if speech >= cfg.min_speech_for_rate_s and words:
        rate = words / speech
        if rate > cfg.max_words_per_s:
            flags.append(Flag(code="words_per_second_high", severity="warning",
                              detail=f"{rate:.1f} words/s over {speech:.1f}s of speech"))
        elif rate < cfg.min_words_per_s:
            flags.append(Flag(code="words_per_second_low", severity="warning",
                              detail=f"{rate:.2f} words/s over {speech:.1f}s of speech"))
    if speech >= cfg.empty_min_speech_s and words <= 1:
        flags.append(Flag(code="empty_output", severity="warning",
                          detail=f"{words} words for {speech:.1f}s of speech"))
    if n_tokens >= max_new_tokens:
        flags.append(Flag(code="possible_truncation", severity="warning",
                          detail=f"generation hit max_new_tokens={max_new_tokens}"))
    if seg.words and min_word_score is not None:
        aligned = [w.score for w in seg.words if w.aligned]
        mean = sum(aligned) / len(aligned) if aligned else 0.0
        if mean < min_word_score:
            flags.append(Flag(code="low_alignment_confidence", severity="warning",
                              detail=f"mean word alignment score {mean:.2f}: word times and speaker split uncertain"))
    if seg.primary_speaker is None:
        flags.append(Flag(code="no_diarized_speaker", severity="warning",
                          detail="VAD found speech here but diarization assigned no speaker"))
    elif seg.speaker_uncertain:
        share = seg.speaker_shares[0].seconds if seg.speaker_shares else 0.0
        flags.append(Flag(code="speaker_uncertain", severity="warning",
                          detail=f"main speaker {seg.primary_speaker} covers {share:.1f}s of {duration:.1f}s"))
    if seg.overlap:
        flags.append(Flag(code="overlap", severity="info",
                          detail=f"{seg.overlap_seconds:.1f}s with 2+ speakers: {', '.join(seg.speakers)}"))
    return flags


def suspicion(seg: Segment) -> float:
    """Ranking score for review: flag weights, plus how heavy the overlap is (share of the segment with
    2+ speakers, and how many speakers). Used only to order segments, never to change them."""
    score = sum(WEIGHTS.get(f.code, 1.0) for f in seg.flags)
    if seg.overlap:
        score += 2.0 * seg.overlap_seconds / max(seg.end - seg.start, 1e-6) + 0.5 * max(len(seg.speakers) - 2, 0)
    return score
