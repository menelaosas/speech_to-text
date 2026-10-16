# Troubleshooting

| Problem | Fix |
|---|---|
| `torch.OutOfMemoryError: CUDA out of memory` | Lower `-b` (e.g. `-b 2`). Close other GPU programs (`nvidia-smi`). Last resort: `--device cpu`. |
| `401` / `GatedRepoError` when downloading | Accept the model terms on Hugging Face, then `uv run hf auth login` |
| `ffmpeg is not installed` | `sudo apt install ffmpeg` |
| `ffmpeg failed to decode ...` | The file is corrupt or not audio. Test it with `ffprobe file.m4a` |
| `torch.cuda.is_available()` is False | Check `nvidia-smi`. The driver must support the CUDA version of the installed torch wheel. |
| `Unsupported language` | The model only supports 14 languages, see [Model](Model.md) |
| Text in the wrong language / gibberish | You passed the wrong `-l`. The model can't detect the language. |
| Greek words missing a final ς | Known model bug, see [Known Issues](Known-Issues.md) |
| `Local model not found ... loading from the Hub` | Run `uv run speech-to-text download` |
| Very slow | You are on CPU. Check the "on cuda" line in the log. |
| Paths with spaces or Greek letters | Quote them: `"data/input/ΓΕΝΙΚΟ ... 11.m4a"` |
