from speech_pilot.greek_text import normalize, number_to_words
from speech_pilot.providers.xlmr_sigma import auto_margin


def test_normalizer_steps():
    assert normalize("Καλησπέρας σας, κύριε Στρατηγέ!") == "καλησπερασ σασ κυριε στρατηγε"
    assert normalize("Ϊ ϋ ΐ «Ναι»;") == "ι υ ι ναι"
    assert normalize("εφτά, οχτώ, εννιά") == "επτα οκτω εννεα"


def test_numbers():
    assert number_to_words(0) == "μηδεν"
    assert number_to_words(21) == "εικοσι ενα"
    assert number_to_words(100) == "εκατο"
    assert number_to_words(115) == "εκατον δεκαπεντε"
    assert number_to_words(1453) == "χιλια τετρακοσια πενηντα τρια"
    assert number_to_words(2026) == "δυο χιλιαδεσ εικοσι εξι"
    assert number_to_words(3000) == "τρεισ χιλιαδεσ"
    assert number_to_words(234000) == "διακοσιεσ τριαντα τεσσερισ χιλιαδεσ"
    assert number_to_words(1_000_000) == "ενα εκατομμυριο"


def test_numbers_optional():
    assert normalize("Σε 11 μήνες", numbers=True) == "σε εντεκα μηνεσ"
    assert normalize("Σε 11 μήνες", numbers=False) == "σε 11 μηνεσ"
    assert normalize("το 1.000") == "το χιλια"


def test_auto_margin():
    assert auto_margin(["Σα ευχαριστώ για το χρόνο σα όμω"])[0] < 0
    assert auto_margin(["Σας ευχαριστώ για τον χρόνο σας, όμως της είπα"])[0] > 0
