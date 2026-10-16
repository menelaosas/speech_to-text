"""Wrapper around the Cohere Transcribe model."""

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoProcessor, CohereAsrForConditionalGeneration

from .audio import Segment, is_silent
from .config import MODEL_DIR, MODEL_ID, SAMPLE_RATE, SUPPORTED_LANGUAGES
from .greek import restore_final_sigma
from .postprocess import collapse_repetitions

log = logging.getLogger(__name__)


@dataclass
class TranscribedSegment:
    start: float
    end: float
    text: str


def resolve_model_path(model: str | Path | None = None) -> str:
    """Prefer the local copy in models/, fall back to the Hugging Face Hub id."""
    if model:
        return str(model)
    if (MODEL_DIR / "config.json").exists():
        return str(MODEL_DIR)
    log.warning("Local model not found in %s, loading %s from the Hub.", MODEL_DIR, MODEL_ID)
    return MODEL_ID


class Transcriber:
    def __init__(self, model: str | Path | None = None, device: str | None = None):
        path = resolve_model_path(model)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32

        log.info("Loading model from %s on %s (%s)", path, self.device, dtype)
        self.processor = AutoProcessor.from_pretrained(path)
        self.model = CohereAsrForConditionalGeneration.from_pretrained(path, dtype=dtype).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def transcribe_segments(
        self,
        segments: list[Segment],
        language: str,
        batch_size: int = 8,
        punctuation: bool = True,
        max_new_tokens: int = 256,
        show_progress: bool = True,
        fix_greek_sigma: bool = True,
    ) -> list[TranscribedSegment]:
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language {language!r}. Choose one of: {', '.join(SUPPORTED_LANGUAGES)}")

        results: list[TranscribedSegment] = []
        speech = [s for s in segments if not is_silent(s.samples)]
        bar = tqdm(total=len(speech), unit="seg", desc="Transcribing", disable=not show_progress)

        for i in range(0, len(speech), batch_size):
            batch = speech[i : i + batch_size]
            inputs = self.processor(
                [s.samples for s in batch],
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                language=language,
                punctuation=punctuation,
            )
            inputs.pop("audio_chunk_index", None)  # segments are short enough to never be chunked
            inputs = inputs.to(self.model.device, dtype=self.model.dtype)

            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            texts = self.processor.decode(outputs, skip_special_tokens=True)
            if isinstance(texts, str):
                texts = [texts]

            for seg, text in zip(batch, texts):
                text = collapse_repetitions(text.strip())
                if language == "el" and fix_greek_sigma:
                    text = restore_final_sigma(text)
                results.append(TranscribedSegment(seg.start, seg.end, text))
            bar.update(len(batch))

        bar.close()
        return [r for r in results if r.text]
