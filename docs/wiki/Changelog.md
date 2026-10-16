# Changelog

## 0.1.0 (2026-10-16)

- Initial project: uv package, `speech-to-text` CLI with `download` and `transcribe`
- Cohere Transcribe 03-2026 via native transformers support, bf16 on CUDA
- ffmpeg decoding, splitting into ≤ 30 s segments at quiet points, batched inference, silent segments skipped
- Output formats: txt, srt, json
- Greek final-sigma (ς) restoration workaround for upstream model bug
- Collapse decoder repetition loops
- Unit tests and wiki
