from speech_to_text.greek import restore_final_sigma


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
