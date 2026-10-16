from speech_to_text.postprocess import collapse_repetitions


def test_collapses_decoder_loop():
    text = "Ναι, ναι. Να κάτσουμε." + " Να κάτσουμε." * 21 + " Τέλος."
    assert collapse_repetitions(text) == "Ναι, ναι. Να κάτσουμε. Να κάτσουμε. Τέλος."


def test_keeps_short_natural_repeats():
    text = "Ναι. Ναι. Ναι. Εντάξει."
    assert collapse_repetitions(text) == text
