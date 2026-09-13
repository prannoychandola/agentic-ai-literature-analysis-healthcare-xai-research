"""Repository-relative paths and safe environment configuration."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"


def get_semantic_scholar_api_key() -> str:
    """Return the Semantic Scholar key from the environment, if configured."""

    return os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()


def get_arxiv_contact_email() -> str:
    """Return a configured contact address without embedding personal data in code."""

    return os.environ.get("ARXIV_CONTACT_EMAIL", "contact-not-configured@example.com").strip()
