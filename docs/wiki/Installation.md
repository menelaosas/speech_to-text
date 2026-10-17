# Installation

## 1. System requirements

- **Python 3.12** (pinned in `.python-version`; uv installs it automatically if missing)
- **uv**: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **ffmpeg**, used to decode m4a/mp3/etc.: `sudo apt install ffmpeg`
- **GPU (recommended)**: NVIDIA with ≥ 6 GB VRAM and a recent driver. Cohere Transcribe takes about 4 GB
  in bf16 and the Greek ς-restoration model (XLM-RoBERTa-large) about 1 GB more in fp16.
  Developed on an RTX 4070 Laptop (8 GB) with CUDA 13 wheels.
- **Disk**: about 6.3 GB for the models (4.1 GB + 2.2 GB) and about 6 GB for the PyTorch/CUDA environment.

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

## 4. Download the models

```bash
uv run speech-to-text download                     # both models
uv run speech-to-text download --skip-sigma-model  # only Cohere Transcribe (non-Greek use)
```

| Model | Folder (git-ignored) | Size | Used for |
| --- | --- | --- | --- |
| Cohere Transcribe 03-2026 | `models/cohere-transcribe-03-2026/` | 4.1 GB | speech → text |
| XLM-RoBERTa-large | `models/xlm-roberta-large/` | 2.2 GB | restoring Greek final ς (`--sigma-fix lm`) |

The tool always prefers these local copies. If one is missing, it falls back to downloading into the
Hugging Face cache.

## 5. (Optional) Pilot: speaker-attributed pipeline

The [pilot](Pilot.md) has its own environment (`cd pilot && uv sync`) and needs two more models:

| Model | Folder | Size | Notes |
| --- | --- | --- | --- |
| pyannote/speaker-diarization-community-1 | `models/pyannote-speaker-diarization-community-1/` | 33 MB | **gated**: accept the terms on Hugging Face first |
| jonatasgrosman/wav2vec2-large-xlsr-53-greek | `models/wav2vec2-large-xlsr-53-greek/` | 2.4 GB on disk | word timing only |

```bash
cd pilot
uv run hf download pyannote/speaker-diarization-community-1 --local-dir ../models/pyannote-speaker-diarization-community-1
uv run hf download jonatasgrosman/wav2vec2-large-xlsr-53-greek --revision 489b34fb35fc5876af6193d419772cb9d6d1d531 \
    --local-dir ../models/wav2vec2-large-xlsr-53-greek
```

The pilot runs offline after that (`HF_HUB_OFFLINE=1` is set by its CLI).

## 6. (Optional) Evaluation data

To run the accuracy benchmarks in `scripts/evaluate.py` (see [Accuracy](Accuracy.md)):

```bash
uv run hf download google/fleurs parquet-data/el_gr/test-00000-of-00001.parquet \
    --repo-type dataset --local-dir data/eval/fleurs
```
