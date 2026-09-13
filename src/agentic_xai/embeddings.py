"""Shared Sentence Transformers loading and normalized embedding generation."""

from __future__ import annotations

from typing import Any

import numpy as np


def load_sentence_transformer(model_name: str) -> Any:
    """Load the requested embedding model without changing its configuration."""

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required. Install dependencies with: "
            "python -m pip install -r requirements.txt"
        ) from exc
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:
        raise RuntimeError(f"Unable to load embedding model '{model_name}': {exc}") from exc


def create_normalized_embeddings(model: Any, texts: list[str], context: str = "") -> np.ndarray:
    """Encode texts using the experiment's batch and L2-normalization settings."""

    try:
        encoded = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
    except Exception as exc:
        detail = f" for {context}" if context else ""
        raise RuntimeError(f"Embedding generation failed{detail}: {exc}") from exc
    matrix = np.asarray(encoded, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(texts) or matrix.shape[1] == 0:
        raise RuntimeError(f"Embedding model returned an invalid matrix shape: {matrix.shape}")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0) or not np.all(np.isfinite(matrix)):
        raise RuntimeError("Embedding matrix contains zero-norm or non-finite vectors.")
    return (matrix / norms[:, np.newaxis]).astype(np.float32, copy=False)
