import torch

from speech_pilot.attribution import group_words
from speech_pilot.providers.ctc_aligner import _chars_for, viterbi_align
from speech_pilot.schema import SpeakerTurn

VOCAB = {"<pad>": 0, "|": 4, "Α": 15, "Β": 16, "Ά": 8, "Ϊ": 39, "Ι": 23, "Ε": 19, "Ν": 27, "Κ": 24, "Ο": 29,
         "Σ": 32, "Ι": 23, "Τ": 33, "Ρ": 31, "Δ": 18, "Υ": 34}


def emissions(path: list[int], vocab_size: int = 41) -> torch.Tensor:
    """Log-probs that put almost all mass on the given token per frame."""
    lp = torch.full((len(path), vocab_size), -20.0)
    for t, tok in enumerate(path):
        lp[t, tok] = 0.0
    return lp


def test_viterbi_finds_token_frames():
    # frames: blank A A blank B blank blank A
    lp = emissions([0, 15, 15, 0, 16, 0, 0, 15])
    assert viterbi_align(lp, [15, 16, 15]) == [1, 4, 7]


def test_viterbi_repeated_token_needs_blank():
    lp = emissions([15, 0, 15, 0])
    assert viterbi_align(lp, [15, 15]) == [0, 2]


def test_viterbi_too_short_audio():
    assert viterbi_align(emissions([0, 15]), [15, 16, 15]) is None


def test_greek_char_mapping():
    assert _chars_for("Άνα", VOCAB) == [8, 27, 15]  # uppercase with tonos kept when in vocab
    assert _chars_for("ΐ", VOCAB) == [39]  # tonos dropped, dialytika kept
    assert _chars_for("12", VOCAB) == _chars_for("δώδεκα", VOCAB)  # digits verbalised first
    assert _chars_for("CD", VOCAB) == []  # Latin letters cannot be aligned


def test_group_words_splits_on_speaker_change_and_pause():
    regular = [SpeakerTurn(start=0, end=5, speaker="A"), SpeakerTurn(start=5, end=12, speaker="B")]
    w = lambda t, s, e, al=True: {"text": t, "text_normalized": None, "start": s, "end": e, "score": 0.9, "aligned": al}
    words = [w("ένα", 0.5, 1.0), w("δύο", 1.2, 1.6), w(",", 1.6, 1.6, False), w("τρία", 5.5, 6.0),
             w("τέσσερα", 9.0, 9.5)]
    groups = group_words("u1", words, regular, split_pause_s=2.0)
    assert [(g.speaker, [x["text"] for x in g.words]) for g in groups] == [
        ("A", ["ένα", "δύο", ","]), ("B", ["τρία"]), ("B", ["τέσσερα"])]  # 3 s pause splits B


def test_cp_wer_matches_speakers_by_content():
    from speech_pilot.score import cp_wer

    ref = {"1": "καλημέρα σας κύριε", "2": "ναι βεβαίως"}
    hyp = {"SPEAKER_07": "ναι βεβαίως", "SPEAKER_03": "καλημέρα σας κύριε"}
    r = cp_wer(ref, hyp)
    assert r["cpwer"] == 0 and r["mapping"] == {"1": "SPEAKER_03", "2": "SPEAKER_07"}
    # the same words attributed to the wrong speaker cost errors
    wrong = cp_wer(ref, {"SPEAKER_03": "καλημέρα σας κύριε ναι βεβαίως"})
    assert wrong["errors"] == 4 and wrong["cpwer"] == 0.8  # 2 insertions + 2 deletions over 5 words


def test_rttm_loader_keeps_overlapping_speakers(tmp_path):
    from speech_pilot.score import _rttm_annotation

    p = tmp_path / "x.rttm"
    p.write_text("SPEAKER x 1 1.000 2.000 <NA> <NA> 1 <NA> <NA>\nSPEAKER x 1 1.000 2.000 <NA> <NA> 2 <NA> <NA>\n")
    assert sorted(_rttm_annotation(p, "x").labels()) == ["1", "2"]


def test_speaker_change_snaps_to_sentence_end():
    regular = [SpeakerTurn(start=0, end=2.2, speaker="A"), SpeakerTurn(start=2.2, end=6, speaker="B")]
    w = lambda t, s, e: {"text": t, "text_normalized": None, "start": s, "end": e, "score": 0.9, "aligned": True}
    # "τελευταία ερώτηση." belongs to A, but timing puts "ερώτηση." just inside B's turn
    words = [w("Και", 0.2, 0.5), w("μια", 0.6, 0.9), w("τελευταία", 1.0, 2.1), w("ερώτηση.", 2.2, 2.8),
             w("Ναι,", 3.0, 3.3), w("πείτε", 3.4, 3.7), w("μου.", 3.8, 4.1)]
    groups = group_words("u", words, regular, split_pause_s=2.0, snap_words=3)
    assert [(g.speaker, " ".join(x["text"] for x in g.words)) for g in groups] == [
        ("A", "Και μια τελευταία ερώτηση."), ("B", "Ναι, πείτε μου.")]
    assert groups[0].words[-1]["speaker"] == "A"
