import logging
from pathlib import Path

import torch

from ..audio import read_wav
from ..config import DiarizationCfg
from ..schema import ModelInfo, SpeakerTurn
from .base import DiarizationProvider, DiarizationResult, local_revision

log = logging.getLogger(__name__)


def _turns(annotation) -> list[SpeakerTurn]:
    return [
        SpeakerTurn(start=round(seg.start, 3), end=round(seg.end, 3), speaker=str(label))
        for seg, _, label in annotation.itertracks(yield_label=True)
    ]


class PyannoteDiarization(DiarizationProvider):
    def __init__(self, cfg: DiarizationCfg, device: str):
        self.cfg = cfg
        self.device = device
        self.pipeline = None

    def info(self) -> ModelInfo:
        return ModelInfo(
            role="diarization",
            model_id=self.cfg.model_id,
            local_path=str(self.cfg.model_path),
            revision=local_revision(self.cfg.model_path),
            device=self.device,
            dtype="float32",
            capabilities={"overlap_aware": True, "exclusive_output": True, "timestamps": True},
        )

    def load(self) -> None:
        from pyannote.audio import Pipeline

        if not (self.cfg.model_path / "config.yaml").exists():
            raise FileNotFoundError(
                f"pyannote model not found in {self.cfg.model_path}. Accept the terms at "
                f"https://huggingface.co/{self.cfg.model_id} and download it (see README)."
            )
        self.pipeline = Pipeline.from_pretrained(self.cfg.model_path)
        self.pipeline.segmentation_batch_size = self.cfg.segmentation_batch_size
        self.pipeline.embedding_batch_size = self.cfg.embedding_batch_size
        self.pipeline.to(torch.device(self.device))

    def unload(self) -> None:
        self.pipeline = None

    def set_batch_sizes(self, segmentation: int, embedding: int) -> None:
        self.pipeline.segmentation_batch_size = segmentation
        self.pipeline.embedding_batch_size = embedding

    def diarize(self, wav: Path, num_speakers=None, min_speakers=None, max_speakers=None) -> DiarizationResult:
        audio, sr = read_wav(wav)
        # Pass the waveform in memory so decoding is identical to the other steps.
        file = {"waveform": torch.from_numpy(audio)[None, :], "sample_rate": sr, "uri": Path(wav).stem}
        kwargs = {k: v for k, v in {"num_speakers": num_speakers, "min_speakers": min_speakers,
                                     "max_speakers": max_speakers}.items() if v is not None}
        output = self.pipeline(file, **kwargs)
        regular, exclusive = output.speaker_diarization, output.exclusive_speaker_diarization
        return DiarizationResult(
            regular=_turns(regular),
            exclusive=_turns(exclusive),
            raw={"regular_rttm": regular.to_rttm(), "exclusive_rttm": exclusive.to_rttm(), "params": kwargs},
        )
