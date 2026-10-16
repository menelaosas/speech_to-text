# Architecture

```text
audio file ─► load_audio ─► split_audio ─► Transcriber ─► collapse_repetitions ─► SigmaRestorer ─► writers ─► .txt/.srt/.json
 (m4a…)       (ffmpeg,       (≤30 s at      (batched, beam                          (Greek only,
              16 kHz mono)   quiet points)  search 4, GPU)                          XLM-R-large)
```

## Modules (`src/speech_to_text/`)

| Module | Responsibility |
|---|---|
| `config.py` | Paths (`models/`, `data/input`, `data/output`), default language, segment lengths, supported languages |
| `audio.py` | `load_audio`: calls `ffmpeg` to decode to 16 kHz mono float32. `split_audio`: cuts audio into segments of ≤ 30 s. `is_silent`: skips silent segments. |
| `transcriber.py` | `Transcriber` loads `CohereAsrForConditionalGeneration` (native in transformers ≥ 5.4) in bf16 on GPU, then runs segments in batches |
| `postprocess.py` | Collapses decoder repetition loops (all languages) |
| `greek.py` | `SigmaRestorer` (XLM-R context scoring, adaptive margin) and the `restore_final_sigma` dictionary fallback. Workaround for the missing ς (see [Known Issues](Known-Issues.md)) |
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

**Fixing ς after transcription, not during it.** Biasing the decoder towards ς, prompting and audio
filters were all tried and failed (see [Accuracy](Accuracy.md)). A separate masked language model
scoring both spellings in context was the only approach that worked. To fit next to the ASR model in 8 GB,
it projects to the 250k vocabulary only at the masked positions.

**Beam search by default.** 4 beams cost ~1.8× decoding time and lower WER by ~3% relative.

**Local model directory.** `models/` keeps the project self-contained. The Hub id is the fallback.
