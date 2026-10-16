from .base import (
    ASRProvider,
    AsrOutput,
    AsrUnit,
    DiarizationProvider,
    DiarizationResult,
    TextNormalizer,
    VADProvider,
)

__all__ = [
    "ASRProvider", "AsrOutput", "AsrUnit", "DiarizationProvider", "DiarizationResult", "TextNormalizer",
    "VADProvider", "build_providers",
]


def build_providers(cfg):
    """Instantiate the providers named in the config (imports are lazy so tests stay light)."""
    from .cohere_asr import CohereASR
    from .pyannote_diarization import PyannoteDiarization
    from .silero_vad import SileroVAD
    from .xlmr_sigma import XlmrSigmaRestorer

    device = cfg.device.name
    return {
        "vad": SileroVAD(cfg.vad),
        "diarization": PyannoteDiarization(cfg.diarization, device),
        "asr": CohereASR(cfg.asr, device),
        "normalization": XlmrSigmaRestorer(cfg.normalization, device) if cfg.normalization.greek_final_sigma else None,
    }
