import json
from pathlib import Path

from speech_to_text.transcriber import TranscribedSegment
from speech_to_text.writers import format_timestamp, to_json, to_srt, to_txt, write_outputs

SEGMENTS = [TranscribedSegment(0.0, 29.5, "Γεια σας."), TranscribedSegment(29.5, 3725.25, "Καλημέρα.")]


def test_format_timestamp():
    assert format_timestamp(0) == "00:00:00,000"
    assert format_timestamp(3725.25) == "01:02:05,250"
    assert format_timestamp(1.5, ".") == "00:00:01.500"


def test_srt():
    srt = to_srt(SEGMENTS)
    assert srt.startswith("1\n00:00:00,000 --> 00:00:29,500\nΓεια σας.\n")
    assert "2\n00:00:29,500 --> 01:02:05,250\nΚαλημέρα.\n" in srt


def test_txt():
    assert to_txt(SEGMENTS) == "Γεια σας.\n\nΚαλημέρα.\n"
    assert to_txt(SEGMENTS, timestamps=True).startswith("[00:00:00] Γεια σας.")


def test_json_keeps_greek_characters():
    data = json.loads(to_json(SEGMENTS, "a.m4a", "el"))
    assert data["text"] == "Γεια σας. Καλημέρα."
    assert len(data["segments"]) == 2
    assert "Γεια" in to_json(SEGMENTS, "a.m4a", "el")


def test_write_outputs(tmp_path: Path):
    written = write_outputs(SEGMENTS, Path("rec.m4a"), tmp_path, ["txt", "srt", "json"], "el")
    assert [p.name for p in written] == ["rec.txt", "rec.srt", "rec.json"]
    assert all(p.exists() for p in written)
