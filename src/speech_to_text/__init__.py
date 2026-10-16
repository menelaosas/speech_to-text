"""Speech-to-text with Cohere Transcribe (CohereLabs/cohere-transcribe-03-2026)."""

__version__ = "0.2.0"


def main() -> int:
    from .cli import main as cli_main

    return cli_main()
