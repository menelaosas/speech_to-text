"""CLI entry point: `uv run python scripts/run_pilot.py` == `uv run speech-pilot run`.

Any arguments are passed through, e.g. `scripts/run_pilot.py --excerpt 0:600 --run-id excerpt-10min`.
"""

import sys

from speech_pilot.cli import main

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] not in {"run", "smoke", "compare", "score", "-h", "--help", "-c", "--config", "-v"}:
        args = ["run", *args]
    sys.exit(main(args))
