import numpy as np

from speech_to_text.audio import is_silent, split_audio

SR = 16_000


def test_short_audio_is_one_segment():
    audio = np.random.default_rng(0).normal(0, 0.1, SR * 10).astype(np.float32)
    segments = split_audio(audio, SR, max_segment_s=30)
    assert len(segments) == 1
    assert segments[0].start == 0 and segments[0].end == 10


def test_long_audio_is_split_without_gaps_or_overlap():
    audio = np.random.default_rng(0).normal(0, 0.1, SR * 95).astype(np.float32)
    segments = split_audio(audio, SR, max_segment_s=30, search_s=5)
    assert all(s.end - s.start <= 30 for s in segments)
    assert segments[0].start == 0
    assert segments[-1].end == 95
    for a, b in zip(segments, segments[1:]):
        assert a.end == b.start
    assert sum(len(s.samples) for s in segments) == len(audio)


def test_split_prefers_silence():
    rng = np.random.default_rng(0)
    audio = rng.normal(0, 0.3, SR * 40).astype(np.float32)
    audio[SR * 27 : SR * 28] = 0  # silence between 27 s and 28 s
    first = split_audio(audio, SR, max_segment_s=30, search_s=5)[0]
    assert 27 <= first.end <= 28


def test_is_silent():
    assert is_silent(np.zeros(SR, dtype=np.float32))
    assert not is_silent(np.full(SR, 0.1, dtype=np.float32))
