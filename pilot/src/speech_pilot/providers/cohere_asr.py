import logging

import torch
from tqdm import tqdm

from ..config import AsrCfg
from ..schema import ModelInfo
from .base import ASRProvider, AsrOutput, AsrUnit, local_revision

log = logging.getLogger(__name__)

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class CohereASR(ASRProvider):
    """Cohere Transcribe 03-2026 via the native transformers implementation.

    Known constraints: no timestamps, no diarization, needs a language tag, expects monolingual audio,
    tends to transcribe non-speech sounds, clips limited to 35 s.
    """

    def __init__(self, cfg: AsrCfg, device: str):
        self.cfg = cfg
        self.device = device
        self.processor = None
        self.model = None
        self.events: list[str] = []

    def info(self) -> ModelInfo:
        return ModelInfo(
            role="asr",
            model_id=self.cfg.model_id,
            local_path=str(self.cfg.model_path),
            revision=local_revision(self.cfg.model_path),
            device=self.device,
            dtype=self.cfg.dtype,
            capabilities={"timestamps": False, "diarization": False, "language_detection": False,
                          "max_clip_s": "35", "punctuation": True},
        )

    def decoding(self) -> dict:
        return {"num_beams": self.cfg.num_beams, "do_sample": False, "max_new_tokens": self.cfg.max_new_tokens,
                "punctuation": self.cfg.punctuation, "dtype": self.cfg.dtype}

    def load(self) -> None:
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(self.cfg.model_path)
        self.model = CohereAsrForConditionalGeneration.from_pretrained(
            self.cfg.model_path, dtype=DTYPES[self.cfg.dtype]
        ).to(self.device).eval()

    def unload(self) -> None:
        self.model = None
        self.processor = None

    @torch.inference_mode()
    def _batch(self, units: list[AsrUnit], language: str) -> list[AsrOutput]:
        inputs = self.processor(
            [u.audio for u in units], sampling_rate=16000, return_tensors="pt",
            language=language, punctuation=self.cfg.punctuation,
        )
        chunk_index = inputs.pop("audio_chunk_index", None)
        if chunk_index and any(c is not None for _, c in chunk_index):
            raise RuntimeError("an ASR unit was re-chunked by the processor; units must stay under 35 s")
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        prompt_len = inputs["decoder_input_ids"].shape[1]
        out = self.model.generate(**inputs, max_new_tokens=self.cfg.max_new_tokens,
                                  num_beams=self.cfg.num_beams, do_sample=False)
        texts = self.processor.decode(out, skip_special_tokens=True)
        if isinstance(texts, str):
            texts = [texts]
        return [
            AsrOutput(unit_id=u.id, text=t.strip(), tokens=[int(x) for x in out[i, prompt_len:].tolist()])
            for i, (u, t) in enumerate(zip(units, texts))
        ]

    def transcribe(self, units: list[AsrUnit], language: str) -> list[AsrOutput]:
        batch_size = self.cfg.batch_size
        retried = False
        # Longest first keeps padding low and makes an OOM show up early.
        order = sorted(range(len(units)), key=lambda i: -(units[i].end - units[i].start))
        bar = tqdm(total=len(units), unit="unit", desc="ASR")
        i = 0
        by_index: dict[int, AsrOutput] = {}
        while i < len(order):
            idx = order[i : i + batch_size]
            try:
                for j, res in zip(idx, self._batch([units[j] for j in idx], language)):
                    by_index[j] = res
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if retried or batch_size == 1:
                    raise
                retried = True
                batch_size = max(1, batch_size // 2)
                msg = f"CUDA OOM in ASR; freed memory and retrying once with batch_size={batch_size}"
                log.warning(msg)
                self.events.append(msg)
                continue
            bar.update(len(idx))
            i += len(idx)
        bar.close()
        return [by_index[j] for j in range(len(units))]
