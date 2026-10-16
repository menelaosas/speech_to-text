"""Typed pilot configuration loaded from config/pilot.yaml. No paths or hardware values live in code."""

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

PILOT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PILOT_ROOT / "config" / "pilot.yaml"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Excerpt(Strict):
    start_s: float
    end_s: float


class RunCfg(Strict):
    output_root: Path
    seed: int
    hardware_profile: str


class InputCfg(Strict):
    path: Path
    language: str
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    recording_setup: str
    excerpt: Excerpt | None = None


class DeviceCfg(Strict):
    name: str
    allow_cpu: bool


class PrepareCfg(Strict):
    sample_rate: int


class VadCfg(Strict):
    provider: Literal["silero"]
    model_id: str
    device: str
    threshold: float
    min_speech_ms: int
    min_silence_ms: int
    speech_pad_ms: int


class DiarizationCfg(Strict):
    provider: Literal["pyannote"]
    model_id: str
    model_path: Path
    segmentation_batch_size: int
    embedding_batch_size: int


class AsrCfg(Strict):
    provider: Literal["cohere"]
    model_id: str
    model_path: Path
    dtype: Literal["bfloat16", "float16", "float32"]
    segmentation: Literal["turns", "vad", "hybrid"]
    num_beams: int
    batch_size: int
    max_new_tokens: int
    max_unit_s: float
    split_search_s: float
    merge_gap_s: float
    min_unit_s: float
    min_context_s: float
    context_merge_gap_s: float
    vad_fill_min_s: float
    punctuation: bool


class NormalizationCfg(Strict):
    greek_final_sigma: bool
    model_id: str
    model_path: Path
    dtype: Literal["float16", "float32"]
    window_words: int
    batch_size: int


class AttributionCfg(Strict):
    overlap_min_s: float
    overlap_min_fraction: float
    dominant_min_fraction: float


class QcCfg(Strict):
    nonspeech_max_fraction: float
    max_words_per_s: float
    min_words_per_s: float
    min_speech_for_rate_s: float
    empty_min_speech_s: float
    repetition_min_repeats: int
    hallucination_phrases: list[str]


class ReferenceCfg(Strict):
    excerpt_min_s: float
    excerpt_max_s: float


class PilotConfig(Strict):
    run: RunCfg
    input: InputCfg
    device: DeviceCfg
    prepare: PrepareCfg
    vad: VadCfg
    diarization: DiarizationCfg
    asr: AsrCfg
    normalization: NormalizationCfg
    attribution: AttributionCfg
    qc: QcCfg
    reference: ReferenceCfg

    def digest(self) -> str:
        """Stable hash of the configuration, stored with every step output."""
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _resolve(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else (base / path).resolve()


def load_config(path: Path = DEFAULT_CONFIG, overrides: dict | None = None) -> PilotConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for dotted, value in (overrides or {}).items():
        node = data
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node[key]
        node[leaf] = value
    cfg = PilotConfig.model_validate(data)
    base = Path(path).resolve().parent.parent
    cfg.run.output_root = _resolve(cfg.run.output_root, base)
    cfg.input.path = _resolve(cfg.input.path, base)
    cfg.diarization.model_path = _resolve(cfg.diarization.model_path, base)
    cfg.asr.model_path = _resolve(cfg.asr.model_path, base)
    cfg.normalization.model_path = _resolve(cfg.normalization.model_path, base)
    return cfg
