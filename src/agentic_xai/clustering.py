"""Create semantic embeddings and evaluate K-Means thematic clusters."""

from __future__ import annotations

import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import PROJECT_ROOT
from .embeddings import create_normalized_embeddings, load_sentence_transformer
from .utils import require_outputs_available


CORPUS_PATH = PROJECT_ROOT / "data" / "corpus.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
EMBEDDINGS_PATH = OUTPUTS_DIR / "embeddings.npy"
EMBEDDING_INDEX_PATH = OUTPUTS_DIR / "embedding_index.csv"
ASSIGNMENTS_PATH = OUTPUTS_DIR / "cluster_assignments.csv"
CLUSTERING_RESULTS_PATH = OUTPUTS_DIR / "clustering_results.json"
OUTPUT_PATHS = (
    EMBEDDINGS_PATH,
    EMBEDDING_INDEX_PATH,
    ASSIGNMENTS_PATH,
    CLUSTERING_RESULTS_PATH,
    FIGURES_DIR / "silhouette_vs_k.png",
    FIGURES_DIR / "clustering_metrics_vs_k.png",
    FIGURES_DIR / "calinski_harabasz_vs_k.png",
    FIGURES_DIR / "davies_bouldin_vs_k.png",
    FIGURES_DIR / "inertia_vs_k.png",
    FIGURES_DIR / "cluster_size_distribution.png",
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
K_VALUES = tuple(range(2, 9))
RANDOM_STATE = 42
N_INIT = 20
SILHOUETTE_TIE_TOLERANCE = 1e-12
REQUIRED_COLUMNS = {"paper_id", "title", "abstract", "source", "year"}


def fail(message: str) -> None:
    raise RuntimeError(message)


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_reproducible_seeds() -> None:
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    try:
        import torch

        torch.manual_seed(RANDOM_STATE)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(RANDOM_STATE)
    except ImportError:
        pass


def load_corpus() -> list[dict[str, str]]:
    if not CORPUS_PATH.is_file():
        fail(f"Corpus file not found: {CORPUS_PATH}")
    try:
        with CORPUS_PATH.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            missing = sorted(REQUIRED_COLUMNS - fieldnames)
            if missing:
                fail("Corpus is missing required columns: " + ", ".join(missing))
            rows = [dict(row) for row in reader]
    except OSError as exc:
        fail(f"Unable to read corpus file {CORPUS_PATH}: {exc}")
    if not rows:
        fail(f"Corpus contains no paper rows: {CORPUS_PATH}")
    return rows


def prepare_embedding_rows(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[str]]:
    eligible: list[dict[str, str]] = []
    for row in rows:
        abstract = clean(row.get("abstract"))
        if abstract:
            title = clean(row.get("title"))
            text = f"{title}. {abstract}" if title else abstract
            eligible.append({**row, "embedding_text": text})
    if not eligible:
        fail("No usable abstracts found; at least one non-empty abstract is required.")
    return eligible, [row["embedding_text"] for row in eligible]


def load_embedding_model() -> Any:
    try:
        return load_sentence_transformer(MODEL_NAME)
    except RuntimeError as exc:
        fail(str(exc))


def create_embeddings(model: Any, texts: list[str]) -> np.ndarray:
    try:
        return create_normalized_embeddings(model, texts)
    except RuntimeError as exc:
        fail(str(exc))


def validate_clustering_configuration(n_samples: int) -> None:
    if not K_VALUES or any(k < 2 for k in K_VALUES):
        fail("Invalid clustering configuration: K values must be at least 2.")
    if len(set(K_VALUES)) != len(K_VALUES):
        fail("Invalid clustering configuration: K values must be unique.")
    if max(K_VALUES) >= n_samples:
        fail(
            "Invalid clustering configuration: every K must be smaller than "
            f"the number of embedded papers ({n_samples})."
        )


def evaluate_clusters(
    embeddings: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray], int]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )

    validate_clustering_configuration(embeddings.shape[0])
    metrics: list[dict[str, Any]] = []
    assignments: dict[int, np.ndarray] = {}
    for k in K_VALUES:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        labels = model.fit_predict(embeddings)
        assignments[k] = labels
        counts = np.bincount(labels, minlength=k)
        metrics.append(
            {
                "k": k,
                "silhouette_score": float(silhouette_score(embeddings, labels)),
                "calinski_harabasz_score": float(
                    calinski_harabasz_score(embeddings, labels)
                ),
                "davies_bouldin_score": float(
                    davies_bouldin_score(embeddings, labels)
                ),
                "inertia": float(model.inertia_),
                "cluster_sizes": {
                    str(index): int(count) for index, count in enumerate(counts)
                },
            }
        )
    maximum_silhouette = max(item["silhouette_score"] for item in metrics)
    tied = [
        item
        for item in metrics
        if maximum_silhouette - item["silhouette_score"]
        <= SILHOUETTE_TIE_TOLERANCE
    ]
    selected_k = min(item["k"] for item in tied)
    return metrics, assignments, selected_k


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def save_embeddings(embeddings: np.ndarray, rows: list[dict[str, str]]) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    mapping = [
        {
            "embedding_index": index,
            "paper_id": row.get("paper_id", ""),
            "title": row.get("title", ""),
            "source": row.get("source", ""),
            "year": row.get("year", ""),
        }
        for index, row in enumerate(rows)
    ]
    write_csv(
        EMBEDDING_INDEX_PATH,
        mapping,
        ["embedding_index", "paper_id", "title", "source", "year"],
    )


def save_assignments(rows: list[dict[str, str]], labels: np.ndarray) -> None:
    assignments = [
        {
            "paper_id": row.get("paper_id", ""),
            "title": row.get("title", ""),
            "source": row.get("source", ""),
            "year": row.get("year", ""),
            "cluster": int(label),
        }
        for row, label in zip(rows, labels)
    ]
    write_csv(
        ASSIGNMENTS_PATH,
        assignments,
        ["paper_id", "title", "source", "year", "cluster"],
    )


def generate_figures(metrics: list[dict[str, Any]], selected_k: int) -> None:
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ks = [item["k"] for item in metrics]
    figure, axis = plt.subplots()
    axis.plot(ks, [item["silhouette_score"] for item in metrics], marker="o")
    axis.set_xlabel("Number of clusters (K)")
    axis.set_ylabel("Silhouette score")
    axis.set_title("Silhouette Score by Number of Clusters")
    axis.set_xticks(ks)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "silhouette_vs_k.png", dpi=300)
    plt.close(figure)

    figure, axes = plt.subplots(3, 1, figsize=(6, 10), sharex=True)
    metric_panels = (
        ("silhouette_score", "Silhouette score"),
        ("calinski_harabasz_score", "Calinski-Harabasz score"),
        ("davies_bouldin_score", "Davies-Bouldin score"),
    )
    for axis, (metric_name, label) in zip(axes, metric_panels):
        axis.plot(ks, [item[metric_name] for item in metrics], marker="o")
        axis.set_ylabel(label)
        axis.set_xticks(ks)
    axes[-1].set_xlabel("Number of clusters (K)")
    figure.suptitle("Clustering Metrics by Number of Clusters")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "clustering_metrics_vs_k.png", dpi=300)
    plt.close(figure)

    for metric_name, label, filename in (
        (
            "calinski_harabasz_score",
            "Calinski-Harabasz score",
            "calinski_harabasz_vs_k.png",
        ),
        (
            "davies_bouldin_score",
            "Davies-Bouldin score",
            "davies_bouldin_vs_k.png",
        ),
        ("inertia", "Inertia", "inertia_vs_k.png"),
    ):
        figure, axis = plt.subplots()
        axis.plot(ks, [item[metric_name] for item in metrics], marker="o")
        axis.set_xlabel("Number of clusters (K)")
        axis.set_ylabel(label)
        axis.set_title(f"{label} by Number of Clusters")
        axis.set_xticks(ks)
        figure.tight_layout()
        figure.savefig(FIGURES_DIR / filename, dpi=300)
        plt.close(figure)

    selected_metrics = next(item for item in metrics if item["k"] == selected_k)
    cluster_ids = [int(cluster) for cluster in selected_metrics["cluster_sizes"]]
    cluster_sizes = [
        selected_metrics["cluster_sizes"][str(cluster)] for cluster in cluster_ids
    ]
    figure, axis = plt.subplots()
    axis.bar(cluster_ids, cluster_sizes)
    axis.set_xlabel("Cluster")
    axis.set_ylabel("Number of papers")
    axis.set_title(f"Cluster Size Distribution (K={selected_k})")
    axis.set_xticks(cluster_ids)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "cluster_size_distribution.png", dpi=300)
    plt.close(figure)


def save_results(
    metrics: list[dict[str, Any]],
    selected_k: int,
    corpus_count: int,
    embedded_count: int,
    embedding_dimension: int,
) -> None:
    results = {
        "timestamp": utc_now(),
        "embedding_model": MODEL_NAME,
        "number_of_corpus_papers": corpus_count,
        "number_of_papers_embedded": embedded_count,
        "number_excluded_due_to_missing_abstracts": corpus_count - embedded_count,
        "embedding_dimension": embedding_dimension,
        "normalization_method": "L2 normalization per embedding vector",
        "k_values_tested": list(K_VALUES),
        "metrics": metrics,
        "selected_k": selected_k,
        "selection_criterion": "highest silhouette score",
        "tie_breaking_rule": (
            "If silhouette scores differ by no more than "
            f"{SILHOUETTE_TIE_TOLERANCE}, select the smallest K."
        ),
        "random_state": RANDOM_STATE,
        "kmeans_n_init": N_INIT,
        "embedding_text": "title + abstract; rows with empty abstracts excluded",
        "semantic_analysis_scope": (
            "This experiment identifies semantic structure/themes; it does not prove research gaps."
        ),
    }
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    CLUSTERING_RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def run(overwrite: bool = False) -> int:
    """Create semantic embeddings and clustering artifacts when replacement is allowed."""
    require_outputs_available(OUTPUT_PATHS, overwrite=overwrite)
    set_reproducible_seeds()
    corpus = load_corpus()
    eligible_rows, texts = prepare_embedding_rows(corpus)
    model = load_embedding_model()
    embeddings = create_embeddings(model, texts)
    save_embeddings(embeddings, eligible_rows)
    metrics, assignments, selected_k = evaluate_clusters(embeddings)
    save_assignments(eligible_rows, assignments[selected_k])
    save_results(
        metrics,
        selected_k,
        len(corpus),
        len(eligible_rows),
        embeddings.shape[1],
    )
    generate_figures(metrics, selected_k)
    print(
        f"Analysis complete: embedded {len(eligible_rows)} of {len(corpus)} papers; "
        f"selected K={selected_k}."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
