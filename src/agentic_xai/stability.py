"""Evaluate K-Means clustering stability across random seeds."""

from __future__ import annotations

import csv
import itertools
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import PROJECT_ROOT


OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
CORPUS_PATH = PROJECT_ROOT / "data" / "corpus.csv"
EMBEDDINGS_PATH = OUTPUTS_DIR / "embeddings.npy"
EMBEDDING_INDEX_PATH = OUTPUTS_DIR / "embedding_index.csv"
STEP3_RESULTS_PATH = OUTPUTS_DIR / "clustering_results.json"
RESULTS_PATH = OUTPUTS_DIR / "stability_results.json"
SUMMARY_PATH = OUTPUTS_DIR / "stability_summary.csv"
ASSIGNMENTS_PATH = OUTPUTS_DIR / "stability_assignments_k2.csv"
PAIRWISE_ARI_PATH = OUTPUTS_DIR / "stability_pairwise_ari_k2.csv"
SEEDS = (42, 0, 1, 2, 10, 20, 50, 100)
K_VALUES = tuple(range(2, 9))
N_INIT = 20
EXPECTED_EMBEDDED_PAPERS = 175
REQUIRED_INDEX_COLUMNS = {"embedding_index", "paper_id"}
OUTPUT_PATHS = (
    RESULTS_PATH,
    SUMMARY_PATH,
    ASSIGNMENTS_PATH,
    PAIRWISE_ARI_PATH,
    FIGURES_DIR / "stability_silhouette_by_seed.png",
    FIGURES_DIR / "stability_best_silhouette_by_seed.png",
    FIGURES_DIR / "stability_k2_ari_vs_reference.png",
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_reproducible_seeds() -> None:
    random.seed(42)
    np.random.seed(42)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        fail(f"Required file not found: {path}")
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except OSError as exc:
        fail(f"Unable to read {path}: {exc}")


def load_corpus_size() -> int:
    rows = read_csv(CORPUS_PATH)
    if not rows:
        fail(f"Corpus contains no paper rows: {CORPUS_PATH}")
    return len(rows)


def load_embeddings_and_index() -> tuple[np.ndarray, list[dict[str, str]]]:
    if not EMBEDDINGS_PATH.is_file():
        fail(f"Saved Step 3 embeddings not found: {EMBEDDINGS_PATH}")
    try:
        embeddings = np.load(EMBEDDINGS_PATH, allow_pickle=False)
    except Exception as exc:
        fail(f"Unable to load saved embeddings: {exc}")
    if embeddings.ndim != 2 or embeddings.shape[0] != EXPECTED_EMBEDDED_PAPERS:
        fail(
            "Expected a two-dimensional embedding matrix with "
            f"{EXPECTED_EMBEDDED_PAPERS} rows; found {embeddings.shape}."
        )
    if not np.all(np.isfinite(embeddings)):
        fail("Saved embeddings contain non-finite values.")

    index_rows = read_csv(EMBEDDING_INDEX_PATH)
    if len(index_rows) != EXPECTED_EMBEDDED_PAPERS:
        fail(
            f"Expected {EXPECTED_EMBEDDED_PAPERS} embedding-index rows; "
            f"found {len(index_rows)}."
        )
    missing = sorted(REQUIRED_INDEX_COLUMNS - set(index_rows[0] if index_rows else {}))
    if missing:
        fail("Embedding index is missing required columns: " + ", ".join(missing))
    indices = [clean(row.get("embedding_index")) for row in index_rows]
    expected_indices = [str(index) for index in range(EXPECTED_EMBEDDED_PAPERS)]
    if indices != expected_indices:
        fail("Embedding index rows are not in deterministic 0..174 order.")
    paper_ids = [clean(row.get("paper_id")) for row in index_rows]
    if not all(paper_ids) or len(set(paper_ids)) != EXPECTED_EMBEDDED_PAPERS:
        fail("Embedding index does not contain exactly 175 unique paper IDs.")
    return embeddings, index_rows


def load_step3_provenance() -> dict[str, Any]:
    if not STEP3_RESULTS_PATH.is_file():
        fail(f"Step 3 provenance file not found: {STEP3_RESULTS_PATH}")
    try:
        payload = json.loads(STEP3_RESULTS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"Unable to read Step 3 provenance: {exc}")
    return {
        "model": payload.get("embedding_model"),
        "representation": payload.get("embedding_text", "title + abstract"),
        "embedding_dimension": payload.get("embedding_dimension"),
        "normalization_method": payload.get("normalization_method"),
        "selected_k": payload.get("selected_k"),
        "selected_k_silhouette": next(
            (
                item.get("silhouette_score")
                for item in payload.get("metrics", [])
                if item.get("k") == payload.get("selected_k")
            ),
            None,
        ),
        "source": str(STEP3_RESULTS_PATH),
    }


def validate_configuration(n_samples: int) -> None:
    if SEEDS != (42, 0, 1, 2, 10, 20, 50, 100):
        fail("Invalid seed configuration.")
    if K_VALUES != tuple(range(2, 9)):
        fail("Invalid K configuration: expected K=2 through K=8.")
    if N_INIT != 20:
        fail("Invalid KMeans configuration: expected n_init=20.")
    if max(K_VALUES) >= n_samples:
        fail(f"Invalid K configuration for {n_samples} embedded papers.")


def calculate_seed_metrics(
    embeddings: np.ndarray,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray]]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    metrics: list[dict[str, Any]] = []
    assignments: dict[int, np.ndarray] = {}
    for k in K_VALUES:
        model = KMeans(n_clusters=k, random_state=seed, n_init=N_INIT)
        labels = model.fit_predict(embeddings)
        assignments[k] = labels
        metrics.append(
            {
                "seed": seed,
                "k": k,
                "silhouette_score": float(silhouette_score(embeddings, labels)),
                "calinski_harabasz_score": float(calinski_harabasz_score(embeddings, labels)),
                "davies_bouldin_score": float(davies_bouldin_score(embeddings, labels)),
                "inertia": float(model.inertia_),
            }
        )
    return metrics, assignments


def best_metric(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    # Lowest K is selected only when silhouette values are exactly tied.
    return max(metrics, key=lambda item: (item["silhouette_score"], -item["k"]))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pairwise_ari(assignments: dict[int, np.ndarray]) -> list[dict[str, Any]]:
    from sklearn.metrics import adjusted_rand_score

    pairs = []
    for seed_a, seed_b in itertools.combinations(SEEDS, 2):
        pairs.append(
            {
                "seed_a": seed_a,
                "seed_b": seed_b,
                "ari": float(
                    adjusted_rand_score(assignments[seed_a], assignments[seed_b])
                ),
            }
        )
    return pairs


def assignment_stability(
    assignments: dict[int, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    reference = assignments[42]
    versus_reference = [
        {
            "seed": seed,
            "ari_vs_reference": float(adjusted_rand_score(reference, assignments[seed])),
            "nmi_vs_reference": float(
                normalized_mutual_info_score(reference, assignments[seed])
            ),
        }
        for seed in SEEDS
    ]
    pairwise = pairwise_ari(assignments)
    values = np.asarray([item["ari"] for item in pairwise], dtype=float)
    statistics = {
        "mean_pairwise_ari": float(np.mean(values)),
        "std_pairwise_ari": float(np.std(values)),
        "minimum_pairwise_ari": float(np.min(values)),
        "maximum_pairwise_ari": float(np.max(values)),
    }
    return versus_reference, pairwise, statistics


def generate_figures(metrics_by_seed: dict[int, list[dict[str, Any]]], versus_reference: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots()
    for seed, metrics in metrics_by_seed.items():
        axis.plot(
            [item["k"] for item in metrics],
            [item["silhouette_score"] for item in metrics],
            marker="o",
            label=str(seed),
        )
    axis.set_xlabel("Number of clusters (K)")
    axis.set_ylabel("Silhouette score")
    axis.set_xticks(K_VALUES)
    axis.legend(title="Seed")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "stability_silhouette_by_seed.png", dpi=300)
    plt.close(figure)

    best_values = [best_metric(metrics)["silhouette_score"] for metrics in metrics_by_seed.values()]
    figure, axis = plt.subplots()
    axis.bar([str(seed) for seed in SEEDS], best_values)
    axis.set_xlabel("Random seed")
    axis.set_ylabel("Best silhouette score")
    axis.set_title("Best Silhouette Score by Seed")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "stability_best_silhouette_by_seed.png", dpi=300)
    plt.close(figure)

    figure, axis = plt.subplots()
    axis.bar([str(item["seed"]) for item in versus_reference], [item["ari_vs_reference"] for item in versus_reference])
    axis.set_xlabel("Random seed")
    axis.set_ylabel("ARI against seed 42")
    axis.set_title("K=2 Assignment ARI Against Reference Seed")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "stability_k2_ari_vs_reference.png", dpi=300)
    plt.close(figure)


def run() -> int:
    existing = [path for path in OUTPUT_PATHS if path.exists()]
    if existing:
        fail(
            "Stability outputs already exist; refusing to overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    set_reproducible_seeds()
    corpus_size = load_corpus_size()
    embeddings, index_rows = load_embeddings_and_index()
    validate_configuration(embeddings.shape[0])
    provenance = load_step3_provenance()

    metrics_by_seed: dict[int, list[dict[str, Any]]] = {}
    assignments_by_seed: dict[int, dict[int, np.ndarray]] = {}
    for seed in SEEDS:
        metrics, assignments = calculate_seed_metrics(embeddings, seed)
        metrics_by_seed[seed] = metrics
        assignments_by_seed[seed] = assignments

    selected_rows = []
    for seed in SEEDS:
        selected = best_metric(metrics_by_seed[seed])
        selected_rows.append(
            {
                "seed": seed,
                "selected_k": selected["k"],
                "best_silhouette": selected["silhouette_score"],
                "selected_k_calinski_harabasz": selected["calinski_harabasz_score"],
                "selected_k_davies_bouldin": selected["davies_bouldin_score"],
                "selected_k_inertia": selected["inertia"],
            }
        )

    best_silhouettes = np.asarray([row["best_silhouette"] for row in selected_rows], dtype=float)
    selected_silhouettes = best_silhouettes.copy()
    selected_k_frequency = {
        str(k): sum(row["selected_k"] == k for row in selected_rows)
        for k in K_VALUES
    }
    k2_assignments = {
        seed: assignments_by_seed[seed][2]
        for seed in SEEDS
    }
    versus_reference, pairwise, ari_statistics = assignment_stability(k2_assignments)

    summary_rows = []
    for seed in SEEDS:
        selected_k = best_metric(metrics_by_seed[seed])["k"]
        for metric in metrics_by_seed[seed]:
            summary_rows.append(
                {
                    **metric,
                    "is_best_k_for_seed": metric["k"] == selected_k,
                }
            )
    write_csv(
        SUMMARY_PATH,
        summary_rows,
        [
            "seed",
            "k",
            "silhouette_score",
            "calinski_harabasz_score",
            "davies_bouldin_score",
            "inertia",
            "is_best_k_for_seed",
        ],
    )

    assignment_rows = []
    for index, row in enumerate(index_rows):
        assignment_rows.append(
            {
                "paper_id": row["paper_id"],
                **{
                    f"seed_{seed}_cluster": int(k2_assignments[seed][index])
                    for seed in SEEDS
                },
            }
        )
    write_csv(
        ASSIGNMENTS_PATH,
        assignment_rows,
        ["paper_id"] + [f"seed_{seed}_cluster" for seed in SEEDS],
    )
    write_csv(PAIRWISE_ARI_PATH, pairwise, ["seed_a", "seed_b", "ari"])

    results = {
        "timestamp": utc_now(),
        "corpus_size": corpus_size,
        "embedded_paper_count": int(embeddings.shape[0]),
        "paper_ids_order_used": [row["paper_id"] for row in index_rows],
        "embedding_source_path": str(EMBEDDINGS_PATH),
        "model_provenance": provenance,
        "representation": "Title + Abstract",
        "seeds": list(SEEDS),
        "k_values": list(K_VALUES),
        "n_init": N_INIT,
        "metrics_by_seed_and_k": {
            str(seed): metrics for seed, metrics in metrics_by_seed.items()
        },
        "selected_k_by_seed": selected_rows,
        "selected_k_summary": {
            "frequency_by_k": selected_k_frequency,
            "mean_best_silhouette": float(np.mean(best_silhouettes)),
            "std_best_silhouette": float(np.std(best_silhouettes)),
            "minimum_best_silhouette": float(np.min(best_silhouettes)),
            "maximum_best_silhouette": float(np.max(best_silhouettes)),
            "mean_selected_k_silhouette": float(np.mean(selected_silhouettes)),
            "std_selected_k_silhouette": float(np.std(selected_silhouettes)),
            "number_selecting_k2": selected_k_frequency["2"],
            "proportion_selecting_k2": selected_k_frequency["2"] / len(SEEDS),
        },
        "k2_assignment_stability": {
            "reference_seed": 42,
            "ari_nmi_vs_reference": versus_reference,
            "pairwise_ari_statistics": ari_statistics,
            "pairwise_comparison_count": len(pairwise),
        },
        "interpretation": {
            "selected_k_frequency": selected_k_frequency,
            "note": "Observed metric and assignment consistency is reported without an invented stability threshold. This analysis does not prove or disprove research gaps.",
        },
    }
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    generate_figures(metrics_by_seed, versus_reference)
    print(f"Stability analysis complete for {len(SEEDS)} seeds and {len(K_VALUES)} K values.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except (RuntimeError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
