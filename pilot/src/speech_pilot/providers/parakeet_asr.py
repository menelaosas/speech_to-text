"""Offline Parakeet TDT v3 comparison provider, using the existing Transformers dependency.

Kept separate from production provider selection until evaluation justifies adopting it.
"""

from pathlib import Path

import torch

from ..schema import ModelInfo
from .base import ASRProvider, AsrOutput, AsrUnit, local_revision


class ParakeetASR(ASRProvider):
    def __init__(self, model_path: Path, device: str, batch_size: int = 1):
        self.path = Path(model_path).resolve()
        self.device = device
        self.batch_size = batch_size
        self.processor = None
        self.model = None
        self.events = []

    def info(self):
        return ModelInfo(role='asr', model_id='nvidia/parakeet-tdt-0.6b-v3', local_path=str(self.path),
                         revision=local_revision(self.path), device=self.device, dtype='float32',
                         capabilities={'language_detection': True, 'comparison_only': True})

    def decoding(self):
        return {'strategy': 'greedy', 'dtype': 'float32', 'batch_size': self.batch_size,
                'language_detection': 'automatic; no language prompt', 'max_new_tokens': 1024}

    def load(self):
        from transformers import AutoModelForTDT, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.path, local_files_only=True)
        self.model = AutoModelForTDT.from_pretrained(
            self.path, dtype=torch.float32, local_files_only=True).to(self.device).eval()

    def unload(self):
        self.model = None
        self.processor = None

    @torch.inference_mode()
    def transcribe(self, units: list[AsrUnit], language: str):
        self.events = []
        outputs = []
        for start in range(0, len(units), self.batch_size):
            batch = units[start:start + self.batch_size]
            inputs = self.processor([u.audio for u in batch], sampling_rate=16000,
                                    return_tensors='pt', padding=True).to(self.device, dtype=self.model.dtype)
            result = self.model.generate(**inputs, return_dict_in_generate=True, max_new_tokens=1024)
            texts = self.processor.batch_decode(result.sequences, skip_special_tokens=True)
            if len(texts) != len(batch):
                raise RuntimeError('Unexpected number of Parakeet outputs')
            for u, text, sequence in zip(batch, texts, result.sequences):
                # Retain exact decoder tokens for audit; do not apply Cohere EOS rules to TDT.
                tokens = sequence.tolist()
                outputs.append(AsrOutput(u.id, text.strip(), tokens, {'requested_language': language,
                    'language_detection': 'automatic', 'decoder_sequence_length': len(tokens)}))
        return outputs
