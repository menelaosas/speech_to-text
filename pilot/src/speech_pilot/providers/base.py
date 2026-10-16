"""Provider interfaces. The pipeline depends only on these, never on a concrete model library."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..schema import Interval, ModelInfo, SpeakerTurn


def local_revision(model_dir: Path) -> str | None:
    """Commit hash recorded by `hf download --local-dir` (first line of any .metadata file)."""
    meta_dir = model_dir / ".cache" / "huggingface" / "download"
    for candidate in ("config.json.metadata", "config.yaml.metadata"):
        p = meta_dir / candidate
        if p.exists():
            return p.read_text().splitlines()[0].strip()
    return None


class Provider(ABC):
    role: str

    @abstractmethod
    def info(self) -> ModelInfo: ...

    def load(self) -> None:  # noqa: B027 - optional hook
        pass

    def unload(self) -> None:  # noqa: B027 - optional hook
        pass


class VADProvider(Provider):
    role = "vad"

    @abstractmethod
    def detect(self, audio: np.ndarray, sample_rate: int) -> list[Interval]: ...


@dataclass
class DiarizationResult:
    regular: list[SpeakerTurn]  # overlap-aware: turns of different speakers may overlap
    exclusive: list[SpeakerTurn]  # at most one speaker at any time
    raw: dict = field(default_factory=dict)  # provider-specific raw output, saved for rebuilds


class DiarizationProvider(Provider):
    role = "diarization"

    @abstractmethod
    def diarize(self, wav: Path, num_speakers: int | None, min_speakers: int | None,
                max_speakers: int | None) -> DiarizationResult: ...


@dataclass
class AsrUnit:
    id: str
    start: float
    end: float
    audio: np.ndarray
    speaker_hint: str | None = None


@dataclass
class AsrOutput:
    unit_id: str
    text: str
    tokens: list[int]


class ASRProvider(Provider):
    role = "asr"

    @abstractmethod
    def transcribe(self, units: list[AsrUnit], language: str) -> list[AsrOutput]: ...


class TextNormalizer(Provider):
    role = "normalization"

    @abstractmethod
    def normalize(self, texts: list[str]) -> tuple[list[str], list[int], dict]:
        """Returns normalised texts, number of changes per text, and run details."""
