"""Inspect generation stop conditions and select among actual ASR hypotheses."""

from ..greek_text import normalize


def generated_tokens(sequence, eos_ids, pad_id, budget):
    """Count real decoder tokens, ignoring batch padding; EOS means generation completed."""
    tokens = []
    ended = False
    for token in sequence:
        token = int(token)
        if token in eos_ids:
            ended = True
            break
        if token != pad_id:
            tokens.append(token)
    return tokens, not ended and len(tokens) >= budget


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, word in enumerate(a, 1):
        row = [i]
        for j, other in enumerate(b, 1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (word != other)))
        previous = row
    return previous[-1]


def select_candidate(candidates, policy):
    """Uniform empirical WER risk over unique, completed candidates; ties keep beam order.

    This is a consensus heuristic, not a calibrated confidence or estimated true WER.
    Never synthesizes words. Number/Greek normalization affects comparison only.
    """
    eligible = [i for i, c in enumerate(candidates) if not c['truncated']]
    eligible = eligible or list(range(len(candidates)))
    if policy == 'best' or len(eligible) == 1:
        return eligible[0]
    unique = {}
    for i in eligible:
        words = tuple(normalize(candidates[i]['text']).split())
        if words:  # Empty hypotheses cannot vote away all speech.
            unique.setdefault(words, i)
    if not unique:
        return eligible[0]
    hypotheses = list(unique)
    risks = [sum(edit_distance(reference, hypothesis) / max(len(reference), 1)
                 for reference in hypotheses) / len(hypotheses) for hypothesis in hypotheses]
    return unique[hypotheses[min(range(len(risks)), key=risks.__getitem__)]]
