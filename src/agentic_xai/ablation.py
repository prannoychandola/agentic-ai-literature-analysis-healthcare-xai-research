"""Evaluate title, abstract, and title-plus-abstract representations."""

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
RESULTS_PATH = OUTPUTS_DIR / "ablation_results.json"
SUMMARY_PATH = OUTPUTS_DIR / "ablation_summary.csv"
STEP3_RESULTS_PATH = OUTPUTS_DIR / "clustering_results.json"
OUTPUT_PATHS = (
    RESULTS_PATH,
    SUMMARY_PATH,
    FIGURES_DIR / "ablation_silhouette_comparison.png",
    FIGURES_DIR / "ablation_davies_bouldin_comparison.png",
    FIGURES_DIR / "ablation_calinski_harabasz_comparison.png",
    FIGURES_DIR / "ablation_best_silhouette.png",
)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
K_VALUES = tuple(range(2, 9))
RANDOM_STATE = 42
N_INIT = 20
EXPECTED_ELIGIBLE_PAPERS = 175
REQUIRED_COLUMNS = {"paper_id", "title", "abstract", "source", "year"}
CONDITIONS = {
    "Title only": "title",
    "Abstract only": "abstract",
    "Title + Abstract": "title_abstract",
}


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


def load_corpus_rows() -> list[dict[str, str]]:
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
    return rows


def load_eligible_corpus() -> list[dict[str, str]]:
    rows = load_corpus_rows()
    eligible = [row for row in rows if clean(row.get("abstract"))]
    if len(eligible) != EXPECTED_ELIGIBLE_PAPERS:
        fail(
            f"Expected exactly {EXPECTED_ELIGIBLE_PAPERS} papers with non-empty abstracts; "
            f"found {len(eligible)}."
        )
    if len({clean(row.get("paper_id")) for row in eligible}) != len(eligible):
        fail("Eligible corpus contains duplicate paper IDs.")
    return eligible


def build_texts(rows: list[dict[str, str]], condition_key: str) -> list[str]:
    if condition_key == "title":
        return [clean(row.get("title")) for row in rows]
    if condition_key == "abstract":
        return [clean(row.get("abstract")) for row in rows]
    if condition_key == "title_abstract":
        return [
            f"{clean(row.get('title'))}. {clean(row.get('abstract'))}"
            if clean(row.get("title"))
            else clean(row.get("abstract"))
            for row in rows
        ]
    fail(f"Invalid ablation condition: {condition_key}")


def load_model() -> Any:
    try:
        return load_sentence_transformer(MODEL_NAME)
    except RuntimeError as exc:
        fail(str(exc))


def create_embeddings(model: Any, texts: list[str], condition: str) -> np.ndarray:
    try:
        return create_normalized_embeddings(model, texts, context=f"condition '{condition}'")
    except RuntimeError as exc:
        fail(str(exc))


def validate_configuration(n_samples: int) -> None:
    if K_VALUES != tuple(range(2, 9)):
        fail("Invalid K configuration: expected K=2 through K=8.")
    if any(k >= n_samples for k in K_VALUES):
        fail(f"Invalid K configuration for {n_samples} samples.")
    if N_INIT != 20 or RANDOM_STATE != 42:
        fail("Invalid KMeans configuration: expected random_state=42 and n_init=20.")


def calculate_metrics(embeddings: np.ndarray) -> list[dict[str, float | int]]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    metrics: list[dict[str, float | int]] = []
    for k in K_VALUES:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        labels = model.fit_predict(embeddings)
        metrics.append(
            {
                "k": k,
                "silhouette_score": float(silhouette_score(embeddings, labels)),
                "calinski_harabasz_score": float(calinski_harabasz_score(embeddings, labels)),
                "davies_bouldin_score": float(davies_bouldin_score(embeddings, labels)),
                "inertia": float(model.inertia_),
            }
        )
    return metrics


def best_result(metrics: list[dict[str, float | int]]) -> dict[str, float | int]:
    best = max(metrics, key=lambda item: (float(item["silhouette_score"]), -int(item["k"])))
    return {"k": best["k"], "silhouette_score": best["silhouette_score"]}


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def generate_figures(all_metrics: dict[str, list[dict[str, float | int]]]) -> None:
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for metric, ylabel, filename in (
        ("silhouette_score", "Silhouette score", "ablation_silhouette_comparison.png"),
        ("davies_bouldin_score", "Davies-Bouldin score", "ablation_davies_bouldin_comparison.png"),
        ("calinski_harabasz_score", "Calinski-Harabasz score", "ablation_calinski_harabasz_comparison.png"),
    ):
        figure, axis = plt.subplots()
        for condition, metrics in all_metrics.items():
            axis.plot(
                [item["k"] for item in metrics],
                [item[metric] for item in metrics],
                marker="o",
                label=condition,
            )
        axis.set_xlabel("Number of clusters (K)")
        axis.set_ylabel(ylabel)
        axis.set_xticks(K_VALUES)
        axis.legend()
        figure.tight_layout()
        figure.savefig(FIGURES_DIR / filename, dpi=300)
        plt.close(figure)

    best = {condition: best_result(metrics) for condition, metrics in all_metrics.items()}
    figure, axis = plt.subplots()
    axis.bar(list(best), [float(item["silhouette_score"]) for item in best.values()])
    axis.set_ylabel("Best silhouette score")
    axis.set_title("Best Silhouette Score by Text Condition")
    axis.tick_params(axis="x", labelrotation=15)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "ablation_best_silhouette.png", dpi=300)
    plt.close(figure)


def load_step3_provenance() -> dict[str, Any]:
    if not STEP3_RESULTS_PATH.is_file():
        return {}
    try:
        payload = json.loads(STEP3_RESULTS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"Unable to read Step 3 provenance file: {exc}")
    selected_k = payload.get("selected_k")
    selected_metric = next(
        (
            item
            for item in payload.get("metrics", [])
            if item.get("k") == selected_k
        ),
        {},
    )
    return {
        "selected_k": selected_k,
        "selected_k_silhouette": selected_metric.get("silhouette_score"),
        "source": str(STEP3_RESULTS_PATH),
    }


def build_interpretation(
    all_metrics: dict[str, list[dict[str, float | int]]],
) -> dict[str, Any]:
    best = {condition: best_result(metrics) for condition, metrics in all_metrics.items()}
    ranking = sorted(
        best.items(),
        key=lambda item: float(item[1]["silhouette_score"]),
        reverse=True,
    )
    all_weak = all(float(item["silhouette_score"]) < 0.1 for item in best.values())
    interpretation: dict[str, Any] = {
        "highest_best_silhouette_condition": ranking[0][0],
        "highest_best_silhouette": ranking[0][1]["silhouette_score"],
        "best_result_by_condition": best,
        "all_best_silhouettes_below_0_1": all_weak,
        "note": "This ablation evaluates semantic clustering behavior; it does not establish research-gap identification.",
    }
    if all_weak:
        interpretation["weak_separation_statement"] = (
            "All conditions have best silhouette scores below 0.1; "
            "semantic separation is weak under this descriptive threshold."
        )
    return interpretation


def run(overwrite: bool = False) -> int:
    """Run representation ablations when output replacement is explicit."""
    require_outputs_available(OUTPUT_PATHS, overwrite=overwrite)
    set_reproducible_seeds()
    corpus = load_corpus_rows()
    rows = [row for row in corpus if clean(row.get("abstract"))]
    if len(rows) != EXPECTED_ELIGIBLE_PAPERS:
        fail(
            f"Expected exactly {EXPECTED_ELIGIBLE_PAPERS} papers with non-empty abstracts; "
            f"found {len(rows)}."
        )
    validate_configuration(len(rows))
    model = load_model()
    all_metrics: dict[str, list[dict[str, float | int]]] = {}
    embedding_dimensions: dict[str, int] = {}
    for condition, condition_key in CONDITIONS.items():
        texts = build_texts(rows, condition_key)
        embeddings = create_embeddings(model, texts, condition)
        embedding_dimensions[condition] = int(embeddings.shape[1])
        all_metrics[condition] = calculate_metrics(embeddings)

    summary_rows = [
        {"condition": condition, **metric}
        for condition, metrics in all_metrics.items()
        for metric in metrics
    ]
    write_csv(
        SUMMARY_PATH,
        summary_rows,
        [
            "condition",
            "k",
            "silhouette_score",
            "calinski_harabasz_score",
            "davies_bouldin_score",
            "inertia",
        ],
    )
    results = {
        "timestamp": utc_now(),
        "corpus_size": len(corpus),
        "embedded_paper_count": len(rows),
        "paper_ids_order_used": [clean(row.get("paper_id")) for row in rows],
        "model_name": MODEL_NAME,
        "embedding_dimensions": embedding_dimensions,
        "normalization_method": "L2 normalization per embedding vector",
        "conditions": {
            "Title only": "title field only",
            "Abstract only": "abstract field only",
            "Title + Abstract": "title field followed by '. ' and abstract field; title omitted from the separator when empty",
        },
        "k_values": list(K_VALUES),
        "kmeans_configuration": {"random_state": RANDOM_STATE, "n_init": N_INIT},
        "metrics": all_metrics,
        "best_by_condition": {
            condition: best_result(metrics)
            for condition, metrics in all_metrics.items()
        },
        "step3_title_abstract_provenance": load_step3_provenance(),
        "interpretation": build_interpretation(all_metrics),
    }
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    generate_figures(all_metrics)
    print(f"Ablation complete for {len(rows)} papers across {len(CONDITIONS)} conditions.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except (RuntimeError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
