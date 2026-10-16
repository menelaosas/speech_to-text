import pytest

from speech_to_text.config import SIGMA_MODEL_DIR
from speech_to_text.greek import SigmaRestorer, restore_final_sigma


def test_auto_margin_leans_towards_sigma_when_model_dropped_it():
    assert SigmaRestorer.auto_margin(["Σα ευχαριστώ για το χρόνο σα όμω"]) < 0
    assert SigmaRestorer.auto_margin(["Σας ευχαριστώ για τον χρόνο σας, όμως της είπα"]) > 0


@pytest.mark.skipif(not (SIGMA_MODEL_DIR / "config.json").exists(), reason="XLM-R model not downloaded")
def test_lm_restorer_resolves_ambiguous_words():
    restorer = SigmaRestorer(margin=-1.0)
    out = restorer.restore(["Σα ευχαριστούμε πολύ για τη συνάντηση και το χρόνο σα. Η απόφαση τη κυβέρνηση."])[0]
    assert out.startswith("Σας ευχαριστούμε")
    assert "χρόνο σας." in out
    assert "τη συνάντηση" in out  # "τη" stays: accusative before a noun
    assert "της κυβέρνησης" in out or "τη κυβέρνηση" in out


def test_restores_clear_cases():
    assert restore_final_sigma("Θα δούμε όμω από") == "Θα δούμε όμως από"
    assert restore_final_sigma("Σε 11 μήνε εδώ.") == "Σε 11 μήνες εδώ."
    assert restore_final_sigma("Σα ευχαριστώ πολύ για εσά.") == "Σας ευχαριστώ πολύ για εσάς."


def test_leaves_ambiguous_words_alone():
    text = "για τη συνάντηση, αυτό του είπα"
    assert restore_final_sigma(text) == text


def test_ignores_non_greek_and_existing_sigma():
    text = "Hello world, ο κόσμος 2026"
    assert restore_final_sigma(text) == text


def test_uppercase_word():
    assert restore_final_sigma("ΟΜΩ", freqs={"όμως": 100}) == "ΟΜΩ"  # accents differ, no match
    assert restore_final_sigma("ΣΑ", freqs={"σας": 100}) == "ΣΑΣ"
