import importlib.metadata

import numpy as np
import torch

from ..config import VadCfg
from ..schema import Interval, ModelInfo
from .base import VADProvider


class SileroVAD(VADProvider):
    def __init__(self, cfg: VadCfg):
        self.cfg = cfg
        self.model = None

    def info(self) -> ModelInfo:
        return ModelInfo(
            role="vad",
            model_id=self.cfg.model_id,
            local_path=None,
            revision=f"silero-vad=={importlib.metadata.version('silero-vad')} (weights bundled in package)",
            device=self.cfg.device,
            capabilities={"timestamps": True, "speaker_labels": False},
        )

    def load(self) -> None:
        from silero_vad import load_silero_vad

        self.model = load_silero_vad()
        self.model.to(self.cfg.device)

    def unload(self) -> None:
        self.model = None

    def detect(self, audio: np.ndarray, sample_rate: int) -> list[Interval]:
        from silero_vad import get_speech_timestamps

        stamps = get_speech_timestamps(
            torch.from_numpy(audio),
            self.model,
            sampling_rate=sample_rate,
            threshold=self.cfg.threshold,
            min_speech_duration_ms=self.cfg.min_speech_ms,
            min_silence_duration_ms=self.cfg.min_silence_ms,
            speech_pad_ms=self.cfg.speech_pad_ms,
            return_seconds=False,
        )
        return [Interval(start=s["start"] / sample_rate, end=s["end"] / sample_rate) for s in stamps]
