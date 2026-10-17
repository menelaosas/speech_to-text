# speech-to-text wiki

Local transcription of audio files with Cohere Transcribe (`CohereLabs/cohere-transcribe-03-2026`).

## Pages

| Page | What's in it |
| --- | --- |
| [Installation](Installation.md) | Setting up Python, uv, ffmpeg, Hugging Face access and the models |
| [Usage](Usage.md) | All CLI commands and options, output formats, using it from Python |
| [Architecture](Architecture.md) | How the pipeline works, module by module |
| [Model](Model.md) | Facts about the Cohere Transcribe model: languages, limits, performance |
| [Pilot](Pilot.md) | Speaker-attributed pipeline (`pilot/`): diarization, word timestamps, evidence handling, measured results |
| [Accuracy](Accuracy.md) | Benchmarks: what improves accuracy, what was tried and rejected |
| [Known Issues](Known-Issues.md) | Model bugs and their workarounds (Greek ς) |
| [Troubleshooting](Troubleshooting.md) | Common errors and fixes |
| [Development](Development.md) | Tests, adding features, project conventions |
| [Changelog](Changelog.md) | What changed and when |

## In one minute

```bash
uv sync
uv run speech-to-text download
cp recording.m4a data/input/
uv run speech-to-text transcribe
# -> data/output/recording.txt, recording.srt
```

Need to know **who said what**? Use the [pilot](Pilot.md):
`cd pilot && uv run speech-pilot run --run-id <name>` → `pilot/runs/<name>/transcript.txt`.
