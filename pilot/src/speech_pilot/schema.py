"""Canonical transcript schema (Pydantic).

Shaped like the full architecture: several speakers may be active in the same interval, every segment
carries provenance and review flags, and the canonical text is always the unmodified ASR output.
"""

from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "pilot-1.0"


class Interval(BaseModel):
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


class SpeakerTurn(Interval):
    speaker: str


class ModelInfo(BaseModel):
    role: Literal["vad", "diarization", "asr", "normalization", "alignment"]
    model_id: str
    local_path: str | None
    revision: str | None
    device: str
    dtype: str | None = None
    capabilities: dict[str, bool | str] = Field(default_factory=dict)


class Flag(BaseModel):
    code: str
    severity: Literal["info", "warning"]
    detail: str


class SpeakerShare(BaseModel):
    speaker: str
    seconds: float


class Word(BaseModel):
    text: str  # raw token as written by the ASR
    text_normalized: str | None = None
    start: float
    end: float
    score: float  # alignment confidence (mean CTC frame probability)
    aligned: bool  # False: no alignable characters, time interpolated
    speaker: str | None


class Provenance(BaseModel):
    asr_unit_id: str
    asr_model: str
    asr_revision: str | None
    segmentation: Literal["turns", "vad", "hybrid", "aligned"]
    alignment_model: str | None = None
    decoding: dict[str, int | float | bool | str]
    diarization_model: str | None
    diarization_revision: str | None
    vad_model: str
    normalization_model: str | None = None


class Segment(BaseModel):
    id: str
    start: float
    end: float
    speakers: list[str]  # every speaker active in the interval, main speaker first
    primary_speaker: str | None
    speaker_shares: list[SpeakerShare]
    speaker_uncertain: bool
    overlap: bool
    overlap_seconds: float
    vad_speech_seconds: float
    text: str  # canonical: raw ASR output, never edited
    text_normalized: str | None = None  # machine-normalised layer (e.g. Greek final ς), not canonical
    normalization_changes: int = 0
    flags: list[Flag] = Field(default_factory=list)
    words: list[Word] | None = None  # word timestamps (aligned mode only)
    provenance: Provenance


class SpeakerSummary(BaseModel):
    label: str
    speaking_time_s: float
    segments: int


class SourceInfo(BaseModel):
    path: str
    sha256_before: str
    sha256_after: str | None = None
    integrity_ok: bool | None = None
    duration_s: float
    format: dict
    excerpt: dict | None = None


class Transcript(BaseModel):
    schema_version: str = SCHEMA_VERSION
    run_id: str
    created_at: str
    language: str
    source: SourceInfo
    accuracy_measured: bool = False
    accuracy_note: str = "Accuracy has not been measured: no human reference transcript exists yet."
    models: list[ModelInfo]
    config: dict
    config_digest: str
    speakers: list[SpeakerSummary]
    segments: list[Segment]
