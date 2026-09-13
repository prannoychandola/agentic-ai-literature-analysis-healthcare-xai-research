"""Run semantic embedding and clustering analysis from any working directory."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_xai.clustering import run


def main() -> int:
    """Parse the explicit overwrite opt-in and run the analysis stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="replace existing analysis outputs")
    return run(overwrite=parser.parse_args().overwrite)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
