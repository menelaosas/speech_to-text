"""Documented Greek normaliser for WER/CER scoring (section 7 of the pilot prompt).

Steps, in order:
 1. Unicode NFD, lowercase.
 2. Remove tonos and dialytika (all combining marks), then NFC.
 3. Final sigma ς -> σ.
 4. (optional) Numbers to words: digit groups are verbalised in Greek, neuter forms
    ("21" -> "εικοσι ενα"); thousands use feminine forms ("3000" -> "τρεισ χιλιαδεσ").
 5. Remove punctuation (every Unicode P* character becomes a space), collapse whitespace.
 6. Spelling variants are unified: εφτα->επτα, οχτω->οκτω, εννια->εννεα (also inside compounds).
"""

import re
import unicodedata

_UNITS_N = ["μηδεν", "ενα", "δυο", "τρια", "τεσσερα", "πεντε", "εξι", "επτα", "οκτω", "εννεα"]
_UNITS_F = ["μηδεν", "μια", "δυο", "τρεισ", "τεσσερισ", "πεντε", "εξι", "επτα", "οκτω", "εννεα"]
_TEENS_N = ["δεκα", "εντεκα", "δωδεκα", "δεκατρια", "δεκατεσσερα", "δεκαπεντε", "δεκαεξι", "δεκαεπτα",
            "δεκαοκτω", "δεκαεννεα"]
_TEENS_F = _TEENS_N[:3] + ["δεκατρεισ", "δεκατεσσερισ"] + _TEENS_N[5:]
_TENS = ["", "", "εικοσι", "τριαντα", "σαραντα", "πενηντα", "εξηντα", "εβδομηντα", "ογδοντα", "ενενηντα"]
_HUND_N = ["", "εκατο", "διακοσια", "τριακοσια", "τετρακοσια", "πεντακοσια", "εξακοσια", "επτακοσια",
           "οκτακοσια", "εννιακοσια"]
_HUND_F = ["", "εκατο"] + [h[:-1] + "εσ" for h in _HUND_N[2:]]

_VARIANTS = [("εφτα", "επτα"), ("οχτω", "οκτω"), ("εννια", "εννεα")]


def _below_1000(n: int, feminine: bool) -> list[str]:
    units, teens, hundreds = (_UNITS_F, _TEENS_F, _HUND_F) if feminine else (_UNITS_N, _TEENS_N, _HUND_N)
    words = []
    h, rest = divmod(n, 100)
    if h:
        words.append("εκατον" if h == 1 and rest else hundreds[h])
    if 10 <= rest < 20:
        words.append(teens[rest - 10])
    else:
        t, u = divmod(rest, 10)
        if t:
            words.append(_TENS[t])
        if u:
            words.append(units[u])
    return words


def number_to_words(n: int) -> str:
    """Greek cardinal (stress-free, final σ already folded), 0 <= n < 10^12."""
    if n == 0:
        return "μηδεν"
    words = []
    billions, n = divmod(n, 1_000_000_000)
    millions, n = divmod(n, 1_000_000)
    thousands, n = divmod(n, 1000)
    if billions:
        words += ["ενα", "δισεκατομμυριο"] if billions == 1 else _below_1000(billions, False) + ["δισεκατομμυρια"]
    if millions:
        words += ["ενα", "εκατομμυριο"] if millions == 1 else _below_1000(millions, False) + ["εκατομμυρια"]
    if thousands:
        words += ["χιλια"] if thousands == 1 else _below_1000(thousands, True) + ["χιλιαδεσ"]
    words += _below_1000(n, False)
    return " ".join(words)


def normalize(text: str, numbers: bool = True) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = unicodedata.normalize("NFC", text).replace("ς", "σ")
    if numbers:
        text = re.sub(r"(?<=\d)[.,](?=\d{3}\b)", "", text)  # 1.000 / 1,000 -> 1000
        text = re.sub(r"\d+", lambda m: f" {number_to_words(int(m.group()))} ", text)
    text = "".join(" " if unicodedata.category(c).startswith("P") or c in "«»" else c for c in text)
    words = []
    for w in text.split():
        for a, b in _VARIANTS:
            w = w.replace(a, b)
        words.append(w)
    return " ".join(words)
