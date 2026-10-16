# Environment audit

Taken 2026-10-17 00:00 (Europe/Athens), before installing anything for the pilot.

## Input file (evidence)

| Item | Value |
| --- | --- |
| Path | `data/input/ΓΕΝΙΚΟ ΕΠΙΤΕΛΕΙΟ ΕΘΝΙΚΗΣ ΑΜΥΝΑΣ 11.m4a` |
| SHA-256 | `8b4ff418763d6880608a271b90fb318b0dc92bda0d5773317804a35b12004785` |
| Size | 40,354,650 bytes |
| Container / codec | MP4 (m4a) / AAC LC, fltp |
| Sample rate / channels | 48,000 Hz / 1 (mono) |
| Bit rate | 68.6 kb/s |
| Duration | 4707.05 s (78 min 27 s) |
| Speakers | unknown (operator) |
| Recording setup | phone on the table (operator) |
| Language | Greek |

**Chain-of-custody note.** On 2026-10-16 (before this pilot) the earlier speech-to-text project moved the
file from the project root to `data/input/` with `mv` on the same filesystem. A move does not change
file contents, but there is no hash from before the move. The hash above is the baseline from here on.
The pilot only reads this file.

## Hardware

| Item | Value |
| --- | --- |
| Machine | Lenovo Legion Slim 5 |
| GPU | NVIDIA GeForce RTX 4070 Laptop, 8188 MiB |
| Driver | 580.178.04 |
| GPU at audit | 110 MiB used, 42 °C, P8, 225 MHz SM, 2.9 W, power limit 60 W, no throttling active |
| Power | AC online; platform profile `balanced`, power-profiles-daemon `balanced` |
| CPU / RAM | 16 logical cores / 30 GiB (18 GiB available at audit) |
| Kernel | Linux 7.0.0-38-generic |
| Disk | 443 GB free of 915 GB |
| ffmpeg | 8.0.1 |

## Python

| Interpreter | Use |
| --- | --- |
| `/usr/bin/python3.14` (3.14.4) | host, **not touched** |
| uv-managed 3.12.14 | pilot environment (`pilot/.venv`) |
| uv-managed 3.11.16 | VS Code snap, unrelated |

The existing `speech_to_text` project (repo root, `.venv` on Python 3.12) has torch 2.14.1+cu130,
transformers 5.20.1, numpy 2.5.4, huggingface-hub 2.2.0. The pilot has its **own** environment because
pyannote.audio 4.0.7 needs torchaudio/torchcodec builds that pin torch to a matching version.

## Models

| Model | Location | Revision | Status |
| --- | --- | --- | --- |
| CohereLabs/cohere-transcribe-03-2026 | `models/cohere-transcribe-03-2026/` | `d552dadc2922ba6f6c1f994f8db904d52dcaf942` | present (4.1 GB), will not be re-downloaded |
| FacebookAI/xlm-roberta-large | `models/xlm-roberta-large/` | `c23d21b0620b635a76227c604d44e43a9f0ee389` | present (2.2 GB), ς-restoration layer |
| pyannote/speaker-diarization-community-1 | not downloaded | remote `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee` | **blocked**: gated, HF user `menelaosas` has not accepted the terms (403) |
| Silero VAD | pip package `silero-vad` 6.2.3 (weights bundled) | n/a | to install |

## Hugging Face

Token present in `~/.cache/huggingface/token`, user `menelaosas`. Access to the pyannote repo requires
accepting the conditions at https://huggingface.co/pyannote/speaker-diarization-community-1.
