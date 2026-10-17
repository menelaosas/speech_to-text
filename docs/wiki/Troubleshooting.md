# Troubleshooting

| Problem | Fix |
| --- | --- |
| `torch.OutOfMemoryError: CUDA out of memory` | Lower `-b` (e.g. `-b 2`) or `--beams 1`. Use `--sigma-fix dict` to skip loading the XLM-R model (~1 GB). Close other GPU programs (`nvidia-smi`). Last resort: `--device cpu`. |
| `401` / `GatedRepoError` when downloading | Accept the model terms on Hugging Face, then `uv run hf auth login` |
| `ffmpeg is not installed` | `sudo apt install ffmpeg` |
| `ffmpeg failed to decode ...` | The file is corrupt or not audio. Test it with `ffprobe file.m4a` |
| `torch.cuda.is_available()` is False | Check `nvidia-smi`. The driver must support the CUDA version of the installed torch wheel. |
| `Unsupported language` | The model only supports 14 languages, see [Model](Model.md) |
| Text in the wrong language / gibberish | You passed the wrong `-l`. The model can't detect the language. |
| Greek words missing a final ς | Known model bug, see [Known Issues](Known-Issues.md). Make sure `--sigma-fix lm` (the default) is in use. |
| Wrong ς added to words | Try `--sigma-fix dict` (fewer fixes, almost no wrong ones), see [Accuracy](Accuracy.md) |
| `Local model not found ... loading from the Hub` | Run `uv run speech-to-text download` |
| Downloads `FacebookAI/xlm-roberta-large` on every first run | `models/xlm-roberta-large/` is missing: run `uv run speech-to-text download` |
| Too slow | Check the log says "on cuda". For speed over accuracy: `--beams 1 --sigma-fix dict` (~2.5× faster) |
| Pilot: `pyannote model not found` / `403 requires approval` | Accept the terms at huggingface.co/pyannote/speaker-diarization-community-1, then download it (see [Installation](Installation.md) step 5) |
| Pilot: `isolated step … failed (exit code …)` | A GPU step crashed in its own process; the error is printed just above. Common cause: CUDA OOM, so lower `asr.batch_size` or the diarization batch sizes in `pilot/config/pilot.yaml` |
| Pilot: a changed setting doesn't seem to take effect | Steps are reused when their config and upstream outputs are unchanged; after a code change use `--force <step>` or a new `--run-id` |
| Pilot: `CUDA requested but not available` | The pilot never falls back to CPU silently. Fix the GPU, or set `device.allow_cpu: true` knowingly |
| Paths with spaces or Greek letters | Quote them: `"data/input/ΓΕΝΙΚΟ ... 11.m4a"` |
