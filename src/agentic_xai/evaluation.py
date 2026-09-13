"""Compare the saved semantic representation with reproducible baselines."""

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
from .utils import require_outputs_available


CORPUS_PATH = PROJECT_ROOT / "data" / "corpus.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
EMBEDDINGS_PATH = OUTPUTS_DIR / "embeddings.npy"
EMBEDDING_INDEX_PATH = OUTPUTS_DIR / "embedding_index.csv"
STEP3_RESULTS_PATH = OUTPUTS_DIR / "clustering_results.json"
RESULTS_PATH = OUTPUTS_DIR / "baseline_results.json"
SUMMARY_PATH = OUTPUTS_DIR / "baseline_summary.csv"
OUTPUT_PATHS = (
    RESULTS_PATH,
    SUMMARY_PATH,
    FIGURES_DIR / "baseline_silhouette_comparison.png",
    FIGURES_DIR / "baseline_davies_bouldin_comparison.png",
    FIGURES_DIR / "baseline_calinski_harabasz_comparison.png",
    FIGURES_DIR / "baseline_best_method_comparison.png",
)
K_VALUES = tuple(range(2, 9))
RANDOM_STATE = 42
N_INIT = 20
TFIDF_CONFIG = {
    "lowercase": True,
    "stop_words": "english",
    "max_features": 10000,
    "ngram_range": (1, 2),
}
REQUESTED_SVD_COMPONENTS = 100
REQUIRED_CORPUS_COLUMNS = {"paper_id", "title", "abstract", "source", "year"}


def fail(message: str) -> None:
    raise RuntimeError(message)


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_reproducible_seeds() -> None:
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        fail(f"Required file not found: {path}")
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except OSError as exc:
        fail(f"Unable to read {path}: {exc}")


def load_eligible_corpus() -> list[dict[str, str]]:
    rows = read_csv(CORPUS_PATH)
    if not rows:
        fail(f"Corpus contains no paper rows: {CORPUS_PATH}")
    missing = sorted(REQUIRED_CORPUS_COLUMNS - set(rows[0]))
    if missing:
        fail("Corpus is missing required columns: " + ", ".join(missing))
    eligible = []
    for row in rows:
        abstract = clean(row.get("abstract"))
        if abstract:
            title = clean(row.get("title"))
            eligible.append({**row, "embedding_text": f"{title}. {abstract}" if title else abstract})
    if not eligible:
        fail("No papers with non-empty abstracts are available for baseline evaluation.")
    return eligible


def validate_embedding_alignment(rows: list[dict[str, str]]) -> np.ndarray:
    if not EMBEDDINGS_PATH.is_file():
        fail(f"Step 3 embedding matrix not found: {EMBEDDINGS_PATH}")
    try:
        embeddings = np.load(EMBEDDINGS_PATH, allow_pickle=False)
    except Exception as exc:
        fail(f"Unable to load Step 3 embeddings: {exc}")
    index_rows = read_csv(EMBEDDING_INDEX_PATH)
    corpus_ids = [clean(row.get("paper_id")) for row in rows]
    embedding_ids = [clean(row.get("paper_id")) for row in index_rows]
    if embeddings.ndim != 2 or embeddings.shape[0] != len(rows):
        fail(
            "Embedding row count does not match eligible corpus: "
            f"matrix={embeddings.shape}, eligible={len(rows)}"
        )
    if len(index_rows) != len(rows) or embedding_ids != corpus_ids:
        fail("Step 3 embedding_index.csv paper-ID order does not match the eligible corpus order.")
    if not np.all(np.isfinite(embeddings)):
        fail("Step 3 embedding matrix contains non-finite values.")
    return embeddings


def validate_configuration(n_samples: int) -> None:
    if K_VALUES != tuple(range(2, 9)):
        fail("Invalid configuration: baseline K values must be exactly 2 through 8.")
    if max(K_VALUES) >= n_samples:
        fail(f"Invalid configuration: maximum K must be below sample count {n_samples}.")


def calculate_metrics(representation: Any) -> list[dict[str, Any]]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
    from scipy.sparse import issparse

    metrics = []
    metric_representation = representation.toarray() if issparse(representation) else representation
    for k in K_VALUES:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        labels = model.fit_predict(representation)
        metrics.append(
            {
                "k": k,
                "silhouette_score": float(silhouette_score(metric_representation, labels)),
                "calinski_harabasz_score": float(calinski_harabasz_score(metric_representation, labels)),
                "davies_bouldin_score": float(davies_bouldin_score(metric_representation, labels)),
                "inertia": float(model.inertia_),
            }
        )
    return metrics


def best_result(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    best = max(metrics, key=lambda item: (item["silhouette_score"], -item["k"]))
    return {"k": best["k"], "silhouette_score": best["silhouette_score"]}


def build_representations(
    rows: list[dict[str, str]], embeddings: np.ndarray
) -> tuple[dict[str, Any], int, int]:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    texts = [row["embedding_text"] for row in rows]
    vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
    tfidf = vectorizer.fit_transform(texts)
    if tfidf.shape[1] < 1:
        fail("TF-IDF produced zero features; cannot run baseline clustering.")
    requested = min(REQUESTED_SVD_COMPONENTS, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
    if requested < 1:
        fail("Invalid SVD configuration: fewer than two samples or two TF-IDF features.")
    svd = TruncatedSVD(n_components=requested, random_state=RANDOM_STATE)
    tfidf_svd = normalize(svd.fit_transform(tfidf), norm="l2")
    return {
        "MiniLM embeddings + KMeans": embeddings,
        "TF-IDF + KMeans": tfidf,
        "TF-IDF + SVD + KMeans": tfidf_svd,
    }, requested, tfidf.shape[1]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def generate_figures(all_metrics: dict[str, list[dict[str, Any]]]) -> None:
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for metric, ylabel, filename in (
        ("silhouette_score", "Silhouette score", "baseline_silhouette_comparison.png"),
        ("davies_bouldin_score", "Davies-Bouldin score", "baseline_davies_bouldin_comparison.png"),
        ("calinski_harabasz_score", "Calinski-Harabasz score", "baseline_calinski_harabasz_comparison.png"),
    ):
        figure, axis = plt.subplots()
        for method, metrics in all_metrics.items():
            axis.plot([item["k"] for item in metrics], [item[metric] for item in metrics], marker="o", label=method)
        axis.set_xlabel("Number of clusters (K)")
        axis.set_ylabel(ylabel)
        axis.set_xticks(K_VALUES)
        axis.legend()
        figure.tight_layout()
        figure.savefig(FIGURES_DIR / filename, dpi=300)
        plt.close(figure)

    figure, axis = plt.subplots()
    best = {method: best_result(metrics) for method, metrics in all_metrics.items()}
    axis.bar(list(best), [item["silhouette_score"] for item in best.values()])
    axis.set_ylabel("Best silhouette score")
    axis.set_title("Best Silhouette Score by Method")
    axis.tick_params(axis="x", labelrotation=15)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "baseline_best_method_comparison.png", dpi=300)
    plt.close(figure)


def build_interpretation(all_metrics: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    best = {method: best_result(metrics) for method, metrics in all_metrics.items()}
    ranked = sorted(best.items(), key=lambda pair: pair[1]["silhouette_score"], reverse=True)
    best_score = ranked[0][1]["silhouette_score"]
    interpretation = {
        "highest_best_silhouette_method": ranked[0][0],
        "highest_best_silhouette_score": best_score,
        "highest_best_silhouette_k": ranked[0][1]["k"],
        "method_comparison_by_best_silhouette": best,
        "all_methods_below_0_1_silhouette": all(
            item["silhouette_score"] < 0.1 for item in best.values()
        ),
        "note": "These metrics describe clustering separation only; they do not establish research gaps.",
    }
    if interpretation["all_methods_below_0_1_silhouette"]:
        interpretation["semantic_separation_statement"] = (
            "Best silhouette scores for all tested representations are below 0.1; "
            "semantic separation is weak under this descriptive threshold."
        )
    return interpretation


def run(overwrite: bool = False) -> int:
    """Evaluate baseline representations when replacement is explicitly allowed."""
    require_outputs_available(OUTPUT_PATHS, overwrite=overwrite)
    set_reproducible_seeds()
    rows = load_eligible_corpus()
    embeddings = validate_embedding_alignment(rows)
    validate_configuration(len(rows))
    representations, svd_components, tfidf_features = build_representations(rows, embeddings)
    all_metrics = {method: calculate_metrics(representation) for method, representation in representations.items()}
    summary_rows = [
        {"method": method, **metric}
        for method, metrics in all_metrics.items()
        for metric in metrics
    ]
    write_csv(
        SUMMARY_PATH,
        summary_rows,
        ["method", "k", "silhouette_score", "calinski_harabasz_score", "davies_bouldin_score", "inertia"],
    )
    step3_results = {}
    if STEP3_RESULTS_PATH.is_file():
        step3_results = json.loads(STEP3_RESULTS_PATH.read_text(encoding="utf-8"))
    result = {
        "timestamp": utc_now(),
        "number_of_papers": len(rows),
        "paper_ids_order_used": [row["paper_id"] for row in rows],
        "methods": list(all_metrics),
        "representation_configuration": {
            "text": "title + abstract",
            "tfidf": TFIDF_CONFIG,
            "svd_requested_components": REQUESTED_SVD_COMPONENTS,
            "svd_actual_components": svd_components,
            "tfidf_feature_count": tfidf_features,
            "svd_output_normalization": "L2 normalization",
            "inertia_comparison_note": "Inertia is not directly comparable across different feature spaces.",
        },
        "k_values": list(K_VALUES),
        "kmeans_configuration": {"random_state": RANDOM_STATE, "n_init": N_INIT},
        "metrics": all_metrics,
        "best_by_method": {method: best_result(metrics) for method, metrics in all_metrics.items()},
        "actual_step3_result_used": {
            "embedding_file": str(EMBEDDINGS_PATH),
            "embedding_model": step3_results.get("embedding_model"),
            "step3_selected_k": step3_results.get("selected_k"),
            "step3_selected_k_silhouette": next(
                (item.get("silhouette_score") for item in step3_results.get("metrics", []) if item.get("k") == step3_results.get("selected_k")),
                None,
            ),
        },
        "interpretation": build_interpretation(all_metrics),
    }
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    generate_figures(all_metrics)
    print(f"Baseline evaluation complete for {len(rows)} papers.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except (RuntimeError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
