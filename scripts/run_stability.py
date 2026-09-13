"""Run local clustering-stability evaluation from any working directory."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_xai.stability import run


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
