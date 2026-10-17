"""Benchmark scoring, subset selection, dataset loading and paired comparison (no models needed)."""

import json

import pytest

from speech_pilot.benchmark import Utterance, compare, load_common_voice, select, utterance_errors

HEADER = "client_id\tpath\tsentence_id\tsentence\tsentence_domain\tup_votes\tdown_votes\tage\tgender\taccents\tvariant\tlocale\tsegment\n"


def test_utterance_errors_uses_greek_normaliser():
    same = utterance_errors("Καλησπέρας σας, κύριε!", "καλησπερας σας κυριε")
    assert same["word_errors"] == 0 and same["ref_words"] == 3
    one = utterance_errors("ήρθαν 3 άνθρωποι", "ήρθαν τρία άνθρωποι")  # numbers verbalised on both sides
    assert one["word_errors"] == 0
    empty = utterance_errors("δύο λέξεις", "")
    assert (empty["word_errors"], empty["del"]) == (2, 2)


def test_selection_is_deterministic_and_seeded():
    utts = [Utterance(f"c{i:03d}", "λέξη", "s", b"") for i in range(50)]
    a, b = select(utts, 10, 1234), select(list(reversed(utts)), 10, 1234)
    assert [u.id for u in a] == [u.id for u in b] and len(a) == 10
    assert [u.id for u in select(utts, 10, 99)] != [u.id for u in a]
    assert len(select(utts, None, 1234)) == 50


def test_load_common_voice(tmp_path):
    (tmp_path / "clips").mkdir()
    (tmp_path / "dev.tsv").write_text(HEADER + "abcdef0123456789\tx.mp3\tsid\tΓεια σου κόσμε.\t\t2\t0\t\t\t\t\tel\t\n"
                                      + "abcdef0123456789\ty.mp3\tsid\t\t\t2\t0\t\t\t\t\tel\t\n", encoding="utf-8")
    utts, info = load_common_voice(tmp_path, "dev")
    assert [u.id for u in utts] == ["x.mp3"] and utts[0].reference == "Γεια σου κόσμε."
    assert utts[0].source == tmp_path / "clips" / "x.mp3" and info["split"] == "dev" and len(info["tsv_sha256"]) == 64
    with pytest.raises(FileNotFoundError):
        load_common_voice(tmp_path, "test")


def write_run(path, errors):
    path.mkdir()
    with open(path / "utterances.jsonl", "w", encoding="utf-8") as f:
        for i, e in enumerate(errors):
            f.write(json.dumps({"id": f"u{i}", "raw_errors": {"word_errors": e, "ref_words": 10,
                                                               "char_errors": e, "ref_chars": 50}}) + "\n")


def test_compare_detects_real_improvement_and_noise(tmp_path):
    write_run(tmp_path / "a", [3] * 200)
    write_run(tmp_path / "b", [1] * 200)  # 2 fewer errors on every clip
    better = compare(tmp_path / "a", tmp_path / "b")["layers"]["raw"]["wer"]
    assert better["verdict"] == "B better" and better["delta"] == -0.2 and better["utterances_better"] == 200
    write_run(tmp_path / "c", [3, 2, 4] * 66 + [3, 3])  # same total, just shuffled per clip
    noise = compare(tmp_path / "a", tmp_path / "c")["layers"]["raw"]["wer"]
    assert noise["verdict"] == "no significant difference"
    write_run(tmp_path / "d", [3] * 10)
    with pytest.raises(ValueError, match="different clips"):
        compare(tmp_path / "a", tmp_path / "d")
