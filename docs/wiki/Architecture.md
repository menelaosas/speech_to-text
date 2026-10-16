# Architecture

```text
audio file ─► load_audio ─► split_audio ─► Transcriber ─► collapse_repetitions ─► restore_final_sigma ─► writers ─► .txt/.srt/.json
 (m4a…)       (ffmpeg,       (≤30 s at      (batched GPU                            (Greek only)
              16 kHz mono)   quiet points)  generate)
```

## Modules (`src/speech_to_text/`)

| Module | Responsibility |
|---|---|
| `config.py` | Paths (`models/`, `data/input`, `data/output`), default language, segment lengths, supported languages |
| `audio.py` | `load_audio`: calls `ffmpeg` to decode to 16 kHz mono float32. `split_audio`: cuts audio into segments of ≤ 30 s. `is_silent`: skips silent segments. |
| `transcriber.py` | `Transcriber` loads `CohereAsrForConditionalGeneration` (native in transformers ≥ 5.4) in bf16 on GPU, then runs segments in batches |
| `postprocess.py` | Collapses decoder repetition loops (all languages) |
| `greek.py` | Workaround for the missing ς (see [Known Issues](Known-Issues.md)) |
| `writers.py` | Formats results as txt / srt / json |
| `cli.py` | argparse CLI with `download` and `transcribe` subcommands |

## Design decisions

**Splitting the audio ourselves.** The processor can chunk long audio internally (35 s chunks), but
it pushes *all* chunks through the GPU at once, which runs out of memory on an 8 GB card for a
1-hour file. It also loses the chunk timestamps. We split at the quietest 100 ms window in the
last 5 s before each 30 s mark, so words are rarely cut, and process segments in batches of `--batch-size`.
Segments stay under the model's 35 s limit, so the processor never re-chunks them.

**ffmpeg for decoding.** It handles m4a/AAC reliably and avoids librosa/audioread backend issues.

**bf16 on GPU.** The weights are stored in bf16. fp32 gives identical text, needs 8 GB and does not fit an 8 GB card.

**Local model directory.** `models/` keeps the project self-contained. The Hub id is the fallback.
