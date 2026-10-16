import json
from pathlib import Path

from speech_pilot.config import load_config
from speech_pilot.export import srt_time, write_rttm, write_srt, write_txt
from speech_pilot.qc import flag_segment, repetition
from speech_pilot.schema import Provenance, Segment, SourceInfo, SpeakerShare, SpeakerTurn, Transcript

CFG = load_config()


def seg(text, start=0.0, end=10.0, speech=10.0, overlap=False, speakers=("SPEAKER_00",), uncertain=False):
    return Segment(
        id="seg_00001", start=start, end=end, speakers=list(speakers), primary_speaker=speakers[0],
        speaker_shares=[SpeakerShare(speaker=speakers[0], seconds=end - start)], speaker_uncertain=uncertain,
        overlap=overlap, overlap_seconds=2.0 if overlap else 0.0, vad_speech_seconds=speech, text=text,
        provenance=Provenance(asr_unit_id="u00000", asr_model="m", asr_revision="r", segmentation="turns",
                              decoding={}, diarization_model="d", diarization_revision="r", vad_model="v"),
    )


def codes(s, n_tokens=10):
    return {f.code for f in flag_segment(s, CFG.qc, CFG.asr.max_new_tokens, n_tokens)}


def test_repetition_detection():
    assert repetition("Να κάτσουμε. " * 6, 4)
    assert repetition("ναι ναι ναι ναι ναι", 4)
    assert repetition("Ναι. Ναι. Ναι. Εντάξει, πάμε.", 4) is None


def test_flags():
    normal = "Καλησπέρα σας, ευχαριστούμε πολύ για τη συνάντηση και τον χρόνο σας σήμερα."
    assert codes(seg(normal)) == set()
    assert "text_in_nonspeech" in codes(seg(normal, speech=1.0))
    assert "empty_output" in codes(seg("", speech=8.0))
    assert "words_per_second_high" in codes(seg(" ".join(["λέξη"] * 80), speech=10.0))
    assert "words_per_second_low" in codes(seg("ναι εντάξει", speech=10.0))
    assert "possible_truncation" in codes(seg(normal), n_tokens=CFG.asr.max_new_tokens)
    assert "overlap" in codes(seg(normal, overlap=True, speakers=("SPEAKER_00", "SPEAKER_01")))
    assert "speaker_uncertain" in codes(seg(normal, uncertain=True))


def transcript(segments):
    return Transcript(run_id="t", created_at="now", language="el",
                      source=SourceInfo(path="/x/a.m4a", sha256_before="abc", duration_s=60, format={}),
                      models=[], config={}, config_digest="d", speakers=[], segments=segments)


def test_txt_srt_rttm(tmp_path: Path):
    t = transcript([seg("Γεια σας.", 751.0, 764.0),
                    seg("Όχι, όχι.", 764.0, 770.0, overlap=True, speakers=("SPEAKER_01", "SPEAKER_00"))])
    write_txt(t, tmp_path / "t.txt")
    lines = (tmp_path / "t.txt").read_text(encoding="utf-8").splitlines()
    assert "[00:12:31 – 00:12:44] SPEAKER_00: Γεια σας." in lines
    assert "[00:12:44 – 00:12:50] SPEAKER_01 + SPEAKER_00 [OVERLAP]: Όχι, όχι." in lines
    write_srt(t, tmp_path / "t.srt")
    srt = (tmp_path / "t.srt").read_text(encoding="utf-8")
    assert "00:12:31,000 --> 00:12:44,000\n[SPEAKER_00] Γεια σας." in srt
    write_rttm([SpeakerTurn(start=1.5, end=3.0, speaker="SPEAKER_00")], "my file", tmp_path / "d.rttm", offset=10)
    assert (tmp_path / "d.rttm").read_text() == "SPEAKER my_file 1 11.500 1.500 <NA> <NA> SPEAKER_00 <NA> <NA>\n"
    # the canonical JSON round-trips through the schema
    again = Transcript.model_validate_json(t.model_dump_json())
    assert again.segments[1].speakers == ["SPEAKER_01", "SPEAKER_00"]
    assert json.loads(t.model_dump_json())["accuracy_measured"] is False


def test_srt_time():
    assert srt_time(3725.25) == "01:02:05,250"


def test_known_hallucination_phrase_is_flagged_not_removed():
    s = seg("Υπότιτλοι AUTHORWAVE", speech=2.0, end=2.0)
    assert "known_hallucination_phrase" in codes(s)
    assert s.text == "Υπότιτλοι AUTHORWAVE"
