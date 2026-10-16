"""Command-line interface: `speech-to-text download` and `speech-to-text transcribe`."""

import argparse
import logging
import sys
import time
from pathlib import Path

from .config import (
    AUDIO_EXTENSIONS,
    DEFAULT_LANGUAGE,
    INPUT_DIR,
    MODEL_DIR,
    MODEL_ID,
    OUTPUT_DIR,
    SAMPLE_RATE,
    SIGMA_MODEL_DIR,
    SIGMA_MODEL_ID,
    SUPPORTED_LANGUAGES,
)
from .writers import FORMATS

log = logging.getLogger("speech_to_text")


def collect_inputs(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS))
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(p)
    return files


def cmd_download(args: argparse.Namespace) -> int:
    from huggingface_hub import snapshot_download

    log.info("Downloading %s to %s", MODEL_ID, args.model_dir)
    snapshot_download(repo_id=MODEL_ID, local_dir=args.model_dir)
    if not args.skip_sigma_model:
        log.info("Downloading %s to %s", SIGMA_MODEL_ID, SIGMA_MODEL_DIR)
        snapshot_download(
            repo_id=SIGMA_MODEL_ID,
            local_dir=SIGMA_MODEL_DIR,
            allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                            "tokenizer_config.json", "sentencepiece.bpe.model"],
        )
    log.info("Done.")
    return 0


def cmd_transcribe(args: argparse.Namespace) -> int:
    from .audio import load_audio, split_audio
    from .greek import SigmaRestorer, restore_final_sigma
    from .transcriber import Transcriber
    from .writers import write_outputs

    files = collect_inputs(args.inputs or [INPUT_DIR])
    if not files:
        log.error("No audio files found. Put files in %s or pass paths explicitly.", INPUT_DIR)
        return 1

    transcriber = Transcriber(model=args.model, device=args.device)
    sigma_fix = args.sigma_fix if args.language == "el" else "off"
    restorer = SigmaRestorer(device=args.device) if sigma_fix == "lm" else None

    for path in files:
        log.info("Processing %s", path.name)
        t0 = time.perf_counter()
        audio = load_audio(path)
        duration = len(audio) / SAMPLE_RATE
        segments = split_audio(audio)
        log.info("Audio %.1f min, %d segments", duration / 60, len(segments))

        result = transcriber.transcribe_segments(
            segments,
            language=args.language,
            batch_size=args.batch_size,
            punctuation=not args.no_punctuation,
            num_beams=args.beams,
        )
        if restorer:
            for seg, text in zip(result, restorer.restore([r.text for r in result])):
                seg.text = text
        elif sigma_fix == "dict":
            for seg in result:
                seg.text = restore_final_sigma(seg.text)
        written = write_outputs(result, path, args.output_dir, args.formats, args.language, args.timestamps)
        elapsed = time.perf_counter() - t0
        log.info("Finished in %.1f s (%.1fx real time)", elapsed, duration / elapsed)
        for w in written:
            log.info("  -> %s", w)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="speech-to-text", description="Transcribe audio with Cohere Transcribe.")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download", help="download the model into models/")
    d.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    d.add_argument("--skip-sigma-model", action="store_true", help="don't download the Greek ς-restoration model")
    d.set_defaults(func=cmd_download)

    t = sub.add_parser("transcribe", help="transcribe audio files")
    t.add_argument("inputs", nargs="*", type=Path, help=f"audio files or folders (default: {INPUT_DIR})")
    t.add_argument("-l", "--language", default=DEFAULT_LANGUAGE, choices=sorted(SUPPORTED_LANGUAGES),
                   help=f"spoken language (default: {DEFAULT_LANGUAGE})")
    t.add_argument("-o", "--output-dir", type=Path, default=OUTPUT_DIR)
    t.add_argument("-f", "--formats", nargs="+", choices=FORMATS, default=["txt", "srt"])
    t.add_argument("--timestamps", action="store_true", help="prefix each paragraph in .txt with its start time")
    t.add_argument("--no-punctuation", action="store_true", help="lowercase output without punctuation")
    t.add_argument("--sigma-fix", choices=["lm", "dict", "off"], default="lm",
                   help="Greek final-sigma (ς) restoration: lm = context-aware language model (best), "
                        "dict = word list only, off (see docs/wiki/Accuracy.md)")
    t.add_argument("--beams", type=int, default=4, help="beam search width; 1 = greedy (faster, slightly less accurate)")
    t.add_argument("-b", "--batch-size", type=int, default=8, help="segments per GPU batch (lower if out of memory)")
    t.add_argument("--model", help="model path or Hub id (default: local models/ copy)")
    t.add_argument("--device", help="cuda, cuda:1, cpu (default: auto)")
    t.set_defaults(func=cmd_transcribe)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
