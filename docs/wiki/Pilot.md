# Judicial Speech AI pilot

`pilot/` is a separate, pinned project that turns **one recording into a speaker-attributed Greek
transcript**. It treats the input as evidence and measures every model on this laptop. It is a pilot,
not the platform: there is no API, database, UI or Docker.

Full detail lives in the pilot folder:
[pilot/README.md](../../pilot/README.md) · [pilot/DECISIONS.md](../../pilot/DECISIONS.md) (D-01 … D-20) ·
[pilot/PROGRESS.md](../../pilot/PROGRESS.md) · [pilot/ENVIRONMENT_AUDIT.md](../../pilot/ENVIRONMENT_AUDIT.md).
Current deliverable: `pilot/runs/pilot-final/REPORT.md`.

## Which tool to use

| | `speech-to-text` (repo root) | `speech-pilot` (`pilot/`) |
| --- | --- | --- |
| Output | plain transcript | transcript **with speakers**, overlap marks, word timestamps, review flags |
| Models | Cohere + XLM-R (ς) | Silero VAD + pyannote + Cohere + XLM-R (ς) + Greek CTC aligner |
| Input handling | any audio files | one file, read-only, SHA-256 checked before and after |
| Time for 78.5 min | ~80 s | ~5.5 min |
| Output folder | `data/output/` | `pilot/runs/<run_id>/` (browser app: `pilot/.web-data/`) |
| Interface | CLI | CLI, or the [Speech Studio web UI](Web-UI.md) |

`data/output/ΓΕΝΙΚΟ ΕΠΙΤΕΛΕΙΟ ΕΘΝΙΚΗΣ ΑΜΥΝΑΣ 11.txt` is the root tool's output (no speakers).
The speaker-attributed version is `pilot/runs/pilot-final/transcript.txt` (raw) and
`transcript.normalized.txt` (with Greek final ς restored).

## Run

```bash
cd pilot
uv sync
uv run speech-pilot run --run-id pilot-final      # steps 1-9, one command
uv run speech-pilot compare runs/a runs/b         # determinism check
uv run speech-pilot score --run runs/pilot-final  # automatic when a reference exists
uv run speech-pilot-web                           # browser UI: http://127.0.0.1:8765
uv run speech-pilot benchmark --split dev         # WER/CER on Common Voice (see Benchmarks)
```

The browser app (**Speech Studio**) runs this same pipeline for uploaded recordings and adds listening
with word highlighting and correction. See [Web UI](Web-UI.md).

## Pipeline

```text
original (read-only, SHA-256) ─► 01 inspect ─► 02 16 kHz WAV copy ─► 03 Silero VAD (CPU)
   ─► 04 pyannote diarization ─► 05 Cohere ASR on VAD units ≤30 s ─► 05b ς layer (XLM-R)
   ─► 05c CTC word alignment ─► 06 words → speakers → segments ─► 07 quality flags
   ─► 08 export JSON/TXT/SRT/RTTM + report ─► 09 SHA-256 re-check
```

- Steps 04, 05, 05b and 05c run on the GPU, **each in its own process**, with one model loaded at a time.
- Every step writes to `runs/<id>/<step>/` and is reused while its config and upstream outputs are unchanged.
- The **canonical text is the raw Cohere output** and is never edited. The ς restoration is a separate
  layer (`text_normalized`). Repetition loops and hallucinations are **flagged, not removed**.
- Speech overlap is marked (`[OVERLAP]`, all active speakers listed) and never split into false turns.

## Measured results

Reference: the operator's hand transcript of 00:31:38–00:35:08 (571 words, 7 speakers).
This is one excerpt only; the rest of the file is not measured.

| Metric | Result |
| --- | --- |
| WER raw / ς layer | 42.0% / 37.0% |
| CER raw / ς layer | 22.1% / 21.2% |
| cpWER (words + speaker) | 61.7% (was 81.8% before word timestamps) |
| DER (collar 0.25 s) | 36.6% (26.8% with pyannote's exclusive output) |
| Determinism | two fresh full runs identical, word times included |
| Runtime (78.5 min) | 321 s; Cohere 4.3 GB peak VRAM, pyannote 0.86 GB, aligner 1.6 GB |

About 22 of the reference's 561 words are typing slips, so part of the WER comes from the reference.
The DER reference times come from forced alignment of the operator's words, not from hand-marked times.

On public read-speech sets (no speakers involved, see [Benchmarks](Benchmarks.md)): Common Voice 27 Greek
WER 8.92% (dev) / 9.77% (test), FLEURS 7.42%.

## Main decisions (see DECISIONS.md)

| # | Decision | Evidence |
| --- | --- | --- |
| D-02 | Raw text canonical, ς restoration as a separate layer | WER 42.0% → 37.0% with the layer |
| D-09 | BF16 on GPU | BF16 44.1% vs FP32 44.3% WER on the same units |
| D-11 | Short turn units hallucinate (`Υπότιτλοι AUTHORWAVE`) | 5 hallucinations on units < 2.5 s |
| D-17 | Long units + word timestamps + split by speaker | cpWER 81.8% → 61.7%, WER unchanged |
| D-18 | DER reference = operator labels + aligned times | pyannote marks 7.5× more overlap than the reference |
| D-19 | Each GPU step in its own process | removes 20 ms word-time differences between runs |
| D-20 | Cache keys include upstream outputs | a recomputed step invalidates everything after it |
| D-21 | Benchmarks on Common Voice / FLEURS | CV dev 8.92%, test 9.77%, FLEURS 7.42% WER |
| D-22 | Parakeet as comparison only; correct token counts; truncation retry | Parakeet 49.2% vs Cohere 42.7% on the excerpt |
| D-23 | Consensus beam selection stays **off** | worse on CV dev: 8.92% → 9.50% WER (significant) |

## Open questions

1. Is pyannote over-detecting overlap, or did the reference under-mark it? Listen to a sample of
   `[OVERLAP]` segments.
2. How many people speak in total? pyannote estimates 9; the operator heard 7 in the excerpt.
3. The numbers need a second reference excerpt to confirm them.
