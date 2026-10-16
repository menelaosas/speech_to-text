# speech-to-text

Local speech-to-text for audio files (m4a, mp3, wav, …) using
[Cohere Transcribe](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
(`CohereLabs/cohere-transcribe-03-2026`, 2B parameters, Apache 2.0, 14 languages including Greek).

Audio in → transcript files (`.txt`, `.srt`, `.json`) out. Everything runs on your machine.

## Quick start

```bash
uv sync                                   # create .venv and install dependencies
uv run hf auth login                      # once; accept the model terms on Hugging Face first
uv run speech-to-text download            # download the models (~6 GB) into models/

cp ~/recording.m4a data/input/
uv run speech-to-text transcribe          # transcribes everything in data/input/
```

Transcripts appear in `data/output/`.

## Common commands

```bash
# one file, Greek (default), all formats
uv run speech-to-text transcribe "data/input/recording.m4a" -f txt srt json

# English audio
uv run speech-to-text transcribe interview.mp3 -l en

# timestamps in the .txt, smaller GPU batches
uv run speech-to-text transcribe -f txt --timestamps -b 4

uv run speech-to-text transcribe --help
```

## Project layout

```text
├── src/speech_to_text/     # the Python package
│   ├── cli.py              # command-line interface
│   ├── audio.py            # ffmpeg decoding + splitting at quiet points
│   ├── transcriber.py      # model loading and batched inference
│   ├── postprocess.py      # removes decoder repetition loops
│   ├── greek.py            # final-sigma (ς) restoration for Greek (XLM-R)
│   ├── writers.py          # txt / srt / json output
│   ├── config.py           # defaults (paths, language, segment length)
│   └── resources/          # Greek word frequency list
├── scripts/evaluate.py     # accuracy benchmarks (WER/CER on Greek FLEURS)
├── tests/                  # pytest unit tests
├── docs/wiki/              # project wiki
├── data/input/             # put audio here (git-ignored)
├── data/output/            # transcripts are written here (git-ignored)
└── models/                 # downloaded model weights (git-ignored)
```

## Documentation

See the [wiki](docs/wiki/Home.md): installation, usage, how it works, the model, known issues and troubleshooting.

## Requirements

- Linux/macOS/Windows with Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- `ffmpeg` on PATH
- NVIDIA GPU with ≥ 6 GB VRAM recommended (CPU works, but slowly)

## Licenses

- Code: see repository.
- Model: Apache 2.0 (Cohere Labs).
- ς-restoration model: [XLM-RoBERTa-large](https://huggingface.co/FacebookAI/xlm-roberta-large), MIT.
- `src/speech_to_text/resources/el_50k.txt`, `el_sigma_words.txt`: [FrequencyWords](https://github.com/hermitdave/FrequencyWords) by Hermit Dave, CC BY-SA 4.0.
