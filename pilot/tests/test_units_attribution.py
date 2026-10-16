import numpy as np

from speech_pilot.attribution import attribute
from speech_pilot.config import load_config
from speech_pilot.schema import Interval, SpeakerTurn
from speech_pilot.units import overlap_regions, units_from_turns, units_from_vad

CFG = load_config()
SR = 16_000


def T(s, e, spk):
    return SpeakerTurn(start=s, end=e, speaker=spk)


def test_overlap_regions():
    regular = [T(0, 10, "A"), T(8, 15, "B"), T(20, 25, "A")]
    assert [(r.start, r.end) for r in overlap_regions(regular)] == [(8, 10)]


def test_same_speaker_turns_are_merged_and_long_units_split():
    audio = np.random.default_rng(0).normal(0, 0.1, SR * 80).astype(np.float32)
    exclusive = [T(0, 10, "A"), T(10.2, 20, "A"), T(30, 75, "B")]
    units, stats = units_from_turns(exclusive, [], audio, SR, CFG.asr)
    assert (units[0].start, units[0].end, units[0].speaker_hint) == (0, 20, "A")
    b_units = [u for u in units if u.speaker_hint == "B"]
    assert len(b_units) == 2 and all(u.end - u.start <= CFG.asr.max_unit_s for u in b_units)
    assert b_units[0].start == 30 and b_units[-1].end == 75


def test_turns_meeting_in_an_overlap_are_one_unit_not_false_sequential_turns():
    audio = np.zeros(SR * 30, dtype=np.float32)
    regular = [T(0, 10, "A"), T(8, 15, "B")]
    exclusive = [T(0, 9, "A"), T(9, 15, "B")]
    units, stats = units_from_turns(exclusive, overlap_regions(regular), audio, SR, CFG.asr)
    assert len(units) == 1 and units[0].speaker_hint == "MIXED"
    assert stats["merged_across_overlap"] == 1
    a = attribute(units[0].start, units[0].end, regular, overlap_regions(regular), [Interval(start=0, end=15)],
                  CFG.attribution)
    assert a.overlap and set(a.speakers) == {"A", "B"} and a.primary == "A"


def test_vad_units_merge_up_to_max():
    audio = np.zeros(SR * 100, dtype=np.float32)
    vad = [Interval(start=i * 10, end=i * 10 + 8) for i in range(9)]
    units, _ = units_from_vad(vad, audio, SR, CFG.asr)
    assert all(u.end - u.start <= CFG.asr.max_unit_s for u in units)
    assert units[0].start == 0 and units[-1].end == 88


def test_uncertain_speaker_when_no_dominant_speaker():
    regular = [T(0, 4, "A"), T(4, 8, "B"), T(8, 10, "C")]
    a = attribute(0, 10, regular, [], [Interval(start=0, end=10)], CFG.attribution)
    assert a.uncertain and not a.overlap


def test_hybrid_merges_short_units_into_neighbour():
    from speech_pilot.units import UnitSpan, merge_short_units

    spans = [UnitSpan(0, 6, "A"), UnitSpan(6.3, 7.0, "B"), UnitSpan(7.2, 14, "A"), UnitSpan(20, 21, "B")]
    merged, n = merge_short_units(spans, CFG.asr)
    assert n == 1
    # 6.3-7.0 joins the nearer neighbour (gap 0.2 s); the isolated short unit at 20 s has no neighbour
    assert [(s.start, s.end) for s in merged] == [(0, 6), (6.3, 14), (20, 21)]
    assert merged[1].speaker_hint == "MIXED"
