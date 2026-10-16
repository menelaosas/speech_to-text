"""Language-independent clean-up of model output."""

import re

# A sentence (ending in . ! ? ; or the Greek question mark) followed by itself 3+ more times.
_REPEATED_SENTENCE = re.compile(r"([^\s.!?;;][^.!?;;]*[.!?;;])(?:\s+\1){3,}")


def collapse_repetitions(text: str, keep: int = 2) -> str:
    """Collapse decoder loops such as "Να κάτσουμε. Να κάτσουμε. ... (x22)" to `keep` copies."""
    return _REPEATED_SENTENCE.sub(lambda m: " ".join([m.group(1)] * keep), text)
