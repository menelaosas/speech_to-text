from types import SimpleNamespace

import numpy as np
import pytest
import torch

from speech_pilot.config import AsrCfg, load_config
from speech_pilot.providers.base import AsrOutput, AsrUnit
from speech_pilot.providers.cohere_asr import CohereASR
from speech_pilot.providers.decoding import generated_tokens, select_candidate
from speech_pilot.qc import flag_segment
from test_qc_export import seg


def unit(name='u1', duration=2):
    return AsrUnit(name, 0, duration, np.zeros(duration * 16000, dtype=np.float32))


def candidate(text, truncated=False):
    return dict(text=text, truncated=truncated)


def test_stop_detection_ignores_padding_and_respects_eos_at_budget():
    assert generated_tokens([10, 11, 3, 2, 2], {3}, 2, 5) == ([10, 11], False)
    assert generated_tokens([10, 11, 12], {3}, 2, 3) == ([10, 11, 12], True)
    assert generated_tokens([10, 11, 3], {3}, 2, 3) == ([10, 11], False)
    assert generated_tokens([10, 7, 2, 2], {3, 7}, 2, 4) == ([10], False)


def test_consensus_selects_actual_candidate_and_deduplicates_spellings():
    candidates = [candidate('Μια εντελώς διαφορετική πρόταση εδώ'),
                  candidate('καλημέρα σας'), candidate('Καλημέρα σας!'),
                  candidate('καλημέρα σας κύριε'), candidate('καλημέρα σας κυρία')]
    assert select_candidate(candidates, 'best') == 0
    assert select_candidate(candidates, 'consensus') == 1
    assert select_candidate([candidate('κομμένη', True), candidate('πλήρης πρόταση')], 'best') == 1
    assert select_candidate([candidate(''), candidate('πραγματική ομιλία')], 'consensus') == 1


def test_config_validates_candidate_count_and_old_configs_still_load():
    cfg = load_config().asr
    values = cfg.model_dump(exclude={'num_candidates', 'candidate_selection', 'retry_truncated', 'retry_max_new_tokens'})
    assert AsrCfg(**values).num_candidates == 1
    with pytest.raises(ValueError, match='num_candidates'):
        AsrCfg(**(values | {'num_beams': 1, 'num_candidates': 4}))
    with pytest.raises(ValueError, match='two candidates'):
        AsrCfg(**(values | {'candidate_selection': 'consensus'}))


def test_batch_groups_beams_by_unit_and_preserves_text(monkeypatch):
    provider = CohereASR(load_config().asr.model_copy(update={'num_candidates': 2}), 'cpu')
    class Inputs(dict):
        def to(self, *args, **kwargs):
            return self
    class Processor:
        def __call__(self, *args, **kwargs):
            return Inputs(decoder_input_ids=torch.tensor([[4, 5], [4, 5]]))
        def decode(self, out, **kwargs):
            return [' πρώτη ', 'δεύτερη', 'τρίτη', 'τέταρτη']
    def generate(**kwargs):
        assert kwargs['max_new_tokens'] == 4  # decoder capacity minus prompt
        assert kwargs['num_return_sequences'] == 2
        return torch.tensor([[4, 5, 10, 3, 2, 2], [4, 5, 11, 12, 13, 14],
                             [4, 5, 20, 21, 3, 2], [4, 5, 22, 23, 3, 2]])
    provider.processor = Processor()
    provider.model = SimpleNamespace(device='cpu', dtype=torch.float32,
        config=SimpleNamespace(max_position_embeddings=6),
        generation_config=SimpleNamespace(eos_token_id=3, pad_token_id=2), generate=generate)
    outputs = provider._batch([unit('a'), unit('b')], 'el')
    assert [(o.unit_id, o.text, o.tokens) for o in outputs] == [('a', 'πρώτη', [10]), ('b', 'τρίτη', [20, 21])]
    assert outputs[0].generation['candidates'][1]['truncated']
    assert not outputs[0].generation['truncated']


def test_only_truncated_units_retry_and_original_is_retained(monkeypatch):
    provider = CohereASR(load_config().asr, 'cpu')
    calls = []
    def batch(units, language, budget=None):
        calls.append(([u.id for u in units], budget))
        return [AsrOutput(u.id, 'πλήρες' if budget else 'αρχικό', [1, 2],
                {'truncated': not budget and u.id == 'long', 'token_budget': budget or 256}) for u in units]
    monkeypatch.setattr(provider, '_batch', batch)
    result = provider.transcribe([unit('short'), unit('long', 3)], 'el')
    assert [o.unit_id for o in result] == ['short', 'long']
    assert [o.text for o in result] == ['αρχικό', 'πλήρες']
    assert calls == [(['long', 'short'], None), (['long'], 512)]
    assert result[1].generation['first_pass']['truncated']
    assert not result[1].generation['truncated']
    assert result[1].generation['retry']['accepted']


def test_retry_oom_preserves_first_pass(monkeypatch):
    provider = CohereASR(load_config().asr, 'cpu')
    original = AsrOutput('u1', 'αρχικό', [1], {'truncated': True, 'token_budget': 256})
    def fail(*args, **kwargs):
        raise torch.OutOfMemoryError()
    monkeypatch.setattr(provider, '_batch', fail)
    assert provider._recover(unit(), original, 'el') is original
    assert 'retry_error' in original.generation


def test_retry_can_be_disabled_and_never_accepts_empty_output(monkeypatch):
    cfg = load_config().asr
    provider = CohereASR(cfg.model_copy(update={'retry_truncated': False}), 'cpu')
    original = AsrOutput('u1', 'αρχικό', [1], {'truncated': True, 'token_budget': 256})
    assert provider._recover(unit(), original, 'el') is original
    provider.cfg = cfg
    monkeypatch.setattr(provider, '_batch', lambda *args, **kwargs: [
        AsrOutput('u1', '', [], {'truncated': False, 'token_budget': 512})])
    result = provider._recover(unit(), original, 'el')
    assert result.text == 'αρχικό' and not result.generation['retry']['accepted']


def test_qc_uses_actual_stop_condition_including_recovered_units():
    cfg = load_config()
    complete = flag_segment(seg('καλημέρα σας'), cfg.qc, 256, 300, truncated=False)
    unfinished = flag_segment(seg('καλημέρα σας'), cfg.qc, 256, 200, truncated=True)
    assert 'possible_truncation' not in {f.code for f in complete}
    assert 'possible_truncation' in {f.code for f in unfinished}


def test_parakeet_comparison_uses_matching_units_without_language_prompt(tmp_path):
    from speech_pilot.providers.parakeet_asr import ParakeetASR

    provider = ParakeetASR(tmp_path, 'cpu')
    class Inputs(dict):
        def to(self, *args, **kwargs):
            return self
    class Processor:
        def __call__(self, *args, **kwargs):
            assert kwargs['sampling_rate'] == 16000 and 'language' not in kwargs
            return Inputs()
        def batch_decode(self, sequences, **kwargs):
            return ['Ελληνική πρόταση.']
    provider.processor = Processor()
    provider.model = SimpleNamespace(dtype=torch.float32,
        generate=lambda **kwargs: SimpleNamespace(sequences=torch.tensor([[8192, 10, 11, 3]])))
    outputs = provider.transcribe([unit('a'), unit('b')], 'el')
    assert [o.unit_id for o in outputs] == ['a', 'b']
    assert outputs[0].text == 'Ελληνική πρόταση.'
    assert outputs[0].generation['language_detection'] == 'automatic'
    assert provider.info().capabilities['comparison_only']
