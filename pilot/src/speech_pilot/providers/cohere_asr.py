import logging

import torch
from tqdm import tqdm

from ..config import AsrCfg
from ..schema import ModelInfo
from .base import ASRProvider, AsrOutput, AsrUnit, local_revision
from .decoding import generated_tokens, select_candidate

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
                "punctuation": self.cfg.punctuation, "dtype": self.cfg.dtype,
                "num_candidates": self.cfg.num_candidates, "candidate_selection": self.cfg.candidate_selection,
                "retry_truncated": self.cfg.retry_truncated, "retry_max_new_tokens": self.cfg.retry_max_new_tokens}

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
    def _batch(self, units: list[AsrUnit], language: str, budget: int | None = None) -> list[AsrOutput]:
        inputs = self.processor(
            [u.audio for u in units], sampling_rate=16000, return_tensors="pt",
            language=language, punctuation=self.cfg.punctuation,
        )
        chunk_index = inputs.pop("audio_chunk_index", None)
        if chunk_index and any(c is not None for _, c in chunk_index):
            raise RuntimeError("an ASR unit was re-chunked by the processor; units must stay under 35 s")
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        prompt_len = inputs["decoder_input_ids"].shape[1]
        requested = budget or self.cfg.max_new_tokens
        available = self.model.config.max_position_embeddings - prompt_len
        budget = min(requested, available)
        if budget < 1:
            raise ValueError("Decoder prompt leaves no room for transcript tokens")
        out = self.model.generate(**inputs, max_new_tokens=budget, num_return_sequences=self.cfg.num_candidates,
                                  num_beams=self.cfg.num_beams, do_sample=False)
        texts = self.processor.decode(out, skip_special_tokens=True)
        if isinstance(texts, str):
            texts = [texts]
        count = self.cfg.num_candidates
        if len(texts) != len(units) * count:
            raise RuntimeError("Unexpected number of ASR hypotheses returned")
        eos = self.model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
        results = []
        for i, unit in enumerate(units):
            candidates = []
            for j in range(count):
                index = i * count + j
                tokens, truncated = generated_tokens(out[index, prompt_len:].tolist(), eos_ids,
                                                     self.model.generation_config.pad_token_id, budget)
                candidates.append({"text": texts[index].strip(), "tokens": tokens, "truncated": truncated})
            selected = select_candidate(candidates, self.cfg.candidate_selection)
            chosen = candidates[selected]
            results.append(AsrOutput(unit.id, chosen['text'], chosen['tokens'], {
                "token_budget": budget, "truncated": chosen['truncated'], "selected_candidate": selected,
                "candidates": candidates}))
        return results

    def _recover(self, unit: AsrUnit, output: AsrOutput, language: str) -> AsrOutput:
        """Only retry unfinished generation, once, independently of the original batch."""
        if not self.cfg.retry_truncated or not output.generation.get('truncated'):
            return output
        previous_budget = output.generation['token_budget']
        if self.cfg.retry_max_new_tokens <= previous_budget:
            return output
        first = dict(output.generation)
        try:
            retry = self._batch([unit], language, budget=self.cfg.retry_max_new_tokens)[0]
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            output.generation['retry_error'] = 'CUDA out of memory; first-pass transcript retained'
            self.events.append(f'{unit.id}: retry ran out of memory; retained first-pass output')
            return output
        # Retain the original if decoder capacity did not allow a larger budget, or retry lost text.
        accepted = (retry.generation['token_budget'] > previous_budget
                    and bool(retry.text) and (not retry.generation['truncated']
                                             or len(retry.tokens) > len(output.tokens)))
        chosen = retry if accepted else output
        chosen.generation = dict(chosen.generation, first_pass=first, retry={
            'accepted': accepted, 'text': retry.text, 'tokens': retry.tokens, 'generation': retry.generation})
        self.events.append(f'{unit.id}: token-limit retry {previous_budget} -> '
                           f"{retry.generation['token_budget']}; {'accepted' if accepted else 'retained first pass'}")
        return chosen

    def transcribe(self, units: list[AsrUnit], language: str) -> list[AsrOutput]:
        self.events = []
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
        return [self._recover(unit, by_index[j], language) for j, unit in enumerate(units)]
