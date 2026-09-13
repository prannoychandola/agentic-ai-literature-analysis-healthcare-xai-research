"""Small, dependency-free I/O and validation helpers shared by package modules."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV file into dictionaries with a useful missing-file error."""

    if not path.is_file():
        raise RuntimeError(f"Required file not found: {path}")
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except OSError as exc:
        raise RuntimeError(f"Unable to read {path}: {exc}") from exc


def write_csv_rows(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    """Write dictionaries as a UTF-8 CSV file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object with a clear validation error."""

    if not path.is_file():
        raise RuntimeError(f"Required file not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}.")
    return value


def require_outputs_available(paths: Iterable[Path], overwrite: bool = False) -> None:
    """Refuse accidental output replacement unless explicitly requested."""

    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        rendered = ", ".join(str(path) for path in existing)
        raise RuntimeError(f"Outputs already exist; use --overwrite to replace them: {rendered}")
