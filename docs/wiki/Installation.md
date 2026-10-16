# Installation

## 1. System requirements

- **Python 3.12** (pinned in `.python-version`; uv installs it automatically if missing)
- **uv**: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **ffmpeg**, used to decode m4a/mp3/etc.: `sudo apt install ffmpeg`
- **GPU (recommended)**: NVIDIA with ≥ 6 GB VRAM and a recent driver. The model takes about 4 GB in bf16.
  Developed on an RTX 4070 Laptop (8 GB) with CUDA 13 wheels.
- **Disk**: about 4 GB for the model and about 6 GB for the PyTorch/CUDA environment.

## 2. Install dependencies

```bash
cd "speech to text_"
uv sync
```

This creates `.venv/` and installs the exact versions from `uv.lock`.

Check that the GPU is visible:

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

## 3. Hugging Face access

The model is public but **gated**: you must accept its terms once.

1. Log in at https://huggingface.co and open https://huggingface.co/CohereLabs/cohere-transcribe-03-2026
2. Accept the conditions.
3. Create a *read* token at https://huggingface.co/settings/tokens
4. `uv run hf auth login` and paste the token.

## 4. Download the model

```bash
uv run speech-to-text download
```

Weights go to `models/cohere-transcribe-03-2026/` (git-ignored). The tool always prefers this
local copy. If it is missing, the tool falls back to downloading into the Hugging Face cache.
