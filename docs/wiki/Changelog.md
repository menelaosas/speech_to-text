# Changelog

## 0.2.0 (2026-10-16): accuracy

- Beam search (4 beams) by default: `--beams`
- Context-aware Greek ς restoration with XLM-RoBERTa-large and an adaptive margin: `--sigma-fix lm|dict|off`.
  ς-related word errors −61% when the model drops ς. FLEURS WER 9.15% → 8.78%.
- `download` also fetches XLM-R-large (`--skip-sigma-model` to skip)
- `scripts/evaluate.py`: WER/CER and ς-restoration benchmarks on Greek FLEURS
- Rejected after measurement: ς logit bias, prompt context, `<|itn|>`/emotion tokens, audio filters
- New wiki page: [Accuracy](Accuracy.md)

## 0.1.0 (2026-10-16)

- Initial project: uv package, `speech-to-text` CLI with `download` and `transcribe`
- Cohere Transcribe 03-2026 via native transformers support, bf16 on CUDA
- ffmpeg decoding, splitting into ≤ 30 s segments at quiet points, batched inference, silent segments skipped
- Output formats: txt, srt, json
- Greek final-sigma (ς) restoration workaround for upstream model bug
- Collapse decoder repetition loops
- Unit tests and wiki
