"""Identify evidence-supported potential research-gap hypotheses locally."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import PROJECT_ROOT


DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
CORPUS_PATH = DATA_DIR / "corpus.csv"
EMBEDDINGS_PATH = OUTPUTS_DIR / "embeddings.npy"
EMBEDDING_INDEX_PATH = OUTPUTS_DIR / "embedding_index.csv"
CLUSTERING_RESULTS_PATH = OUTPUTS_DIR / "clustering_results.json"
STABILITY_RESULTS_PATH = OUTPUTS_DIR / "stability_results.json"
RESULTS_PATH = OUTPUTS_DIR / "gap_analysis.json"
CANDIDATES_PATH = OUTPUTS_DIR / "gap_candidates.csv"
EVIDENCE_PATH = OUTPUTS_DIR / "gap_evidence.csv"
FIGURE_PATH = FIGURES_DIR / "gap_evidence_support.png"
EXPECTED_CORPUS_SIZE = 187
EXPECTED_ABSTRACT_COUNT = 175
REQUIRED_COLUMNS = {"paper_id", "title", "abstract", "source", "year"}
OUTPUT_PATHS = (RESULTS_PATH, CANDIDATES_PATH, EVIDENCE_PATH, FIGURE_PATH)
CONFIDENCE_LABELS = {"HIGH", "MEDIUM", "LOW"}

LIMITATION_TERMS = (
    "limitation", "limitations", "limited", "lack of", "lacks", "insufficient",
    "challenge", "challenges", "difficulty", "difficult", "constraint", "weakness", "drawback",
)
FUTURE_TERMS = (
    "future work", "future research", "further work", "further research",
    "should be investigated", "should be explored", "remains to be explored",
    "needs further", "requires further", "we plan to", "future studies",
)
GENERALIZATION_TERMS = (
    "generalizability", "generalisation", "generalization", "external validation",
    "external dataset", "independent dataset", "multi-center", "multicenter",
    "prospective", "real-world", "clinical validation", "clinical deployment",
)
CLINICAL_TERMS = (
    "interpretability", "explainability", "explanation", "transparency", "clinician",
    "clinical utility", "trust", "usability", "user study",
)
ROBUSTNESS_TERMS = (
    "robustness", "reliability", "fairness", "bias", "demographic", "subgroup",
    "uncertainty", "calibration",
)
EVALUATION_TERMS = (
    "evaluation", "evaluate", "evaluated", "validation", "validated", "benchmark",
    "reproducibility", "reproducible",
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def contains_term(text: str, term: str) -> bool:
    return re.search(r"\b" + re.escape(term) + r"\b", text.casefold()) is not None


def matching_terms(text: str, terms: tuple[str, ...]) -> list[str]:
    return [term for term in terms if contains_term(text, term)]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        fail(f"Required file not found: {path}")
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except OSError as exc:
        fail(f"Unable to read {path}: {exc}")


def load_corpus() -> list[dict[str, str]]:
    rows = read_csv(CORPUS_PATH)
    if len(rows) != EXPECTED_CORPUS_SIZE:
        fail(f"Expected exactly {EXPECTED_CORPUS_SIZE} corpus papers; found {len(rows)}.")
    missing = sorted(REQUIRED_COLUMNS - set(rows[0] if rows else {}))
    if missing:
        fail("Corpus is missing required columns: " + ", ".join(missing))
    return rows


def load_provenance() -> dict[str, Any]:
    for path in (EMBEDDINGS_PATH, EMBEDDING_INDEX_PATH, CLUSTERING_RESULTS_PATH):
        if not path.is_file():
            fail(f"Required previous-stage artifact not found: {path}")
    try:
        embeddings = np.load(EMBEDDINGS_PATH, allow_pickle=False)
    except Exception as exc:
        fail(f"Unable to load existing embeddings for provenance validation: {exc}")
    index_rows = read_csv(EMBEDDING_INDEX_PATH)
    if embeddings.ndim != 2 or embeddings.shape[0] != EXPECTED_ABSTRACT_COUNT:
        fail(f"Expected 175 existing embedding rows; found {embeddings.shape}.")
    index_ids = [clean(row.get("paper_id")) for row in index_rows]
    if len(index_rows) != EXPECTED_ABSTRACT_COUNT or len(set(index_ids)) != EXPECTED_ABSTRACT_COUNT:
        fail("Existing embedding index does not contain exactly 175 unique papers.")
    try:
        clustering = json.loads(CLUSTERING_RESULTS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"Unable to read clustering provenance: {exc}")
    provenance = {
        "embedding_model": clustering.get("embedding_model"),
        "embedding_source": str(EMBEDDINGS_PATH),
        "embedding_index_source": str(EMBEDDING_INDEX_PATH),
        "embedding_dimension": int(embeddings.shape[1]),
        "representation": "Title + Abstract",
        "selected_k": clustering.get("selected_k"),
        "selected_k_silhouette": next(
            (item.get("silhouette_score") for item in clustering.get("metrics", [])
             if item.get("k") == clustering.get("selected_k")),
            None,
        ),
        "clustering_statement": "Cluster membership is contextual structural evidence only and does not establish a research gap.",
    }
    if STABILITY_RESULTS_PATH.is_file():
        try:
            stability = json.loads(STABILITY_RESULTS_PATH.read_text(encoding="utf-8"))
            provenance["stability_context"] = {
                "source": str(STABILITY_RESULTS_PATH),
                "selected_k_frequency": stability.get("selected_k_summary", {}).get("frequency_by_k"),
                "k2_pairwise_ari": stability.get("k2_assignment_stability", {}).get("pairwise_ari_statistics"),
            }
        except (OSError, json.JSONDecodeError) as exc:
            fail(f"Unable to read stability provenance: {exc}")
    return provenance


def split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]


def extract_evidence(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    theme_terms = {
        "external_validation_generalization": GENERALIZATION_TERMS,
        "clinical_utility_human_validation": CLINICAL_TERMS,
        "robustness_fairness_uncertainty": ROBUSTNESS_TERMS,
        "evaluation_reproducibility": EVALUATION_TERMS,
    }
    for row in rows:
        for sentence in split_sentences(clean(row.get("abstract"))):
            limitation_matches = matching_terms(sentence, LIMITATION_TERMS)
            future_matches = matching_terms(sentence, FUTURE_TERMS)
            if not limitation_matches and not future_matches:
                continue
            for theme, terms in theme_terms.items():
                theme_matches = matching_terms(sentence, terms)
                if not theme_matches:
                    continue
                evidence.append(
                    {
                        "paper_id": clean(row.get("paper_id")),
                        "title": clean(row.get("title")),
                        "theme": theme,
                        "evidence_type": "explicit future work" if future_matches else "explicit limitation",
                        "evidence": sentence,
                        "is_direct_quote": True,
                        "matched_terms": sorted(set(limitation_matches + future_matches + theme_matches)),
                    }
                )
    return evidence


def deduplicate_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    result = []
    for item in evidence:
        key = (item["theme"], item["paper_id"], item["evidence"])
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


GAP_DEFINITIONS = {
    "GAP-01": {
        "gap_statement": "Potential gap hypothesis: external validation and generalizability of healthcare XAI methods remain insufficiently addressed across the reviewed studies.",
        "gap_type": "generalization and external validation",
        "theme": "external_validation_generalization",
    },
    "GAP-02": {
        "gap_statement": "Potential gap hypothesis: clinical utility, clinician-facing validation, and real-world usability of healthcare XAI explanations remain insufficiently evaluated across the reviewed studies.",
        "gap_type": "clinical utility and human validation",
        "theme": "clinical_utility_human_validation",
    },
    "GAP-03": {
        "gap_statement": "Potential gap hypothesis: robustness, fairness, subgroup behavior, and uncertainty of healthcare XAI systems remain insufficiently evaluated across the reviewed studies.",
        "gap_type": "robustness, fairness, and uncertainty",
        "theme": "robustness_fairness_uncertainty",
    },
    "GAP-04": {
        "gap_statement": "Potential gap hypothesis: reproducible and comparative evaluation of healthcare XAI methods remains insufficiently developed across the reviewed studies.",
        "gap_type": "evaluation and reproducibility",
        "theme": "evaluation_reproducibility",
    },
}


def make_candidate(gap_id: str, definition: dict[str, str], evidence: list[dict[str, Any]]) -> dict[str, Any] | None:
    paper_ids: list[str] = []
    for item in evidence:
        if item["paper_id"] not in paper_ids:
            paper_ids.append(item["paper_id"])
    explicit_count = len(evidence)
    if len(paper_ids) < 3 and explicit_count < 2:
        return None
    if len(paper_ids) >= 3 and explicit_count >= 3:
        confidence = "HIGH"
    elif len(paper_ids) >= 2:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"
    return {
        "gap_id": gap_id,
        "gap_statement": definition["gap_statement"],
        "gap_type": definition["gap_type"],
        "confidence": confidence,
        "supporting_paper_ids": paper_ids,
        "supporting_titles": [next(item["title"] for item in evidence if item["paper_id"] == paper_id) for paper_id in paper_ids],
        "supporting_evidence": [
            {
                "paper_id": item["paper_id"],
                "evidence_type": item["evidence_type"],
                "evidence": item["evidence"],
                "is_direct_quote": True,
            }
            for item in evidence
        ],
        "evidence_count": len(evidence),
        "evidence_sources": ["abstract"],
        "rationale": f"This candidate is an evidence-supported hypothesis based on {len(paper_ids)} independent paper(s) and {explicit_count} explicit limitation/future-work passage(s).",
        "caveat": "This is potential research-gap identification, not a proven or confirmed gap. The evidence is abstract-level and requires further investigation and domain review.",
        "cluster_context": {
            "not_used_as_primary_evidence": True,
            "statement": "Cluster membership is contextual only and does not establish this candidate gap.",
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def generate_figure(candidates: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots()
    axis.bar(
        [candidate["gap_id"] for candidate in candidates],
        [len(candidate["supporting_paper_ids"]) for candidate in candidates],
    )
    axis.set_xlabel("Candidate gap")
    axis.set_ylabel("Number of supporting papers")
    axis.set_title("Evidence Support for Candidate Gaps")
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=300)
    plt.close(figure)


def run(overwrite: bool = False) -> int:
    existing = [path for path in OUTPUT_PATHS if path.exists()]
    if existing and not overwrite:
        fail("Gap-analysis outputs already exist; use --overwrite to replace them: " + ", ".join(str(path) for path in existing))
    corpus = load_corpus()
    papers_with_abstracts = sum(bool(clean(row.get("abstract"))) for row in corpus)
    if papers_with_abstracts != EXPECTED_ABSTRACT_COUNT:
        fail(f"Expected exactly {EXPECTED_ABSTRACT_COUNT} papers with abstracts; found {papers_with_abstracts}.")
    provenance = load_provenance()
    evidence = deduplicate_evidence(extract_evidence(corpus))
    candidates = []
    for gap_id, definition in GAP_DEFINITIONS.items():
        theme_evidence = [item for item in evidence if item["theme"] == definition["theme"]]
        candidate = make_candidate(gap_id, definition, theme_evidence)
        if candidate is not None:
            candidates.append(candidate)

    candidate_ids = {paper_id for candidate in candidates for paper_id in candidate["supporting_paper_ids"]}
    limitations = [
        "Evidence extraction is keyword-guided and limited to title/abstract corpus text.",
        "A candidate gap is a hypothesis requiring further investigation, not a proven or confirmed gap.",
        "Cluster size, citation count, and clustering stability are not treated as proof of a research gap.",
        "No external literature, LLM judgment, or expert validation was used.",
    ]
    if not candidates:
        limitations.append("Insufficient evidence to formulate a qualifying candidate gap under the implemented minimum-evidence rules.")

    results = {
        "metadata": {
            "timestamp": utc_now(),
            "corpus_size": len(corpus),
            "papers_with_abstracts": papers_with_abstracts,
            "paper_ids_with_candidate_evidence": len(candidate_ids),
            "statement": "This artifact performs potential/candidate research-gap identification and does not prove research gaps.",
        },
        "method": {
            "external_apis_used": False,
            "llm_used": False,
            "evidence_extraction_rules": {
                "limitation_terms": list(LIMITATION_TERMS),
                "future_work_terms": list(FUTURE_TERMS),
                "theme_terms": {
                    "external_validation_generalization": list(GENERALIZATION_TERMS),
                    "clinical_utility_human_validation": list(CLINICAL_TERMS),
                    "robustness_fairness_uncertainty": list(ROBUSTNESS_TERMS),
                    "evaluation_reproducibility": list(EVALUATION_TERMS),
                },
                "source_text": "abstract text from data/corpus.csv",
                "direct_quote_policy": "Evidence marked as direct quotes is copied exactly from an abstract sentence.",
            },
            "candidate_generation_rules": {
                "minimum": "At least 3 supporting papers, or at least 2 explicit limitation/future-work passages.",
                "keyword_presence_alone_is_not_evidence": True,
                "cluster_size_is_not_primary_evidence": True,
            },
            "confidence_rules": {
                "HIGH": "At least 3 papers and at least 3 explicit limitation/future-work passages.",
                "MEDIUM": "At least 2 supporting papers with reasonably consistent evidence.",
                "LOW": "Limited or indirect evidence; never treated as proof.",
                "allowed_labels": sorted(CONFIDENCE_LABELS),
            },
            "clustering_provenance": provenance,
            "clustering_limitation": "Semantic clusters provide contextual structure only and do not establish a gap.",
        },
        "candidate_gaps": candidates,
        "limitations": limitations,
    }
    candidate_rows = [
        {
            "gap_id": candidate["gap_id"],
            "gap_statement": candidate["gap_statement"],
            "gap_type": candidate["gap_type"],
            "confidence": candidate["confidence"],
            "evidence_count": candidate["evidence_count"],
            "supporting_paper_ids": ";".join(candidate["supporting_paper_ids"]),
            "supporting_titles": ";".join(candidate["supporting_titles"]),
            "rationale": candidate["rationale"],
            "caveat": candidate["caveat"],
        }
        for candidate in candidates
    ]
    evidence_rows = [
        {
            "gap_id": candidate["gap_id"],
            "paper_id": item["paper_id"],
            "title": next(title for title, paper_id in zip(candidate["supporting_titles"], candidate["supporting_paper_ids"]) if paper_id == item["paper_id"]),
            "evidence_type": item["evidence_type"],
            "evidence_text": item["evidence"],
            "is_direct_quote": item["is_direct_quote"],
        }
        for candidate in candidates
        for item in candidate["supporting_evidence"]
    ]
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(CANDIDATES_PATH, candidate_rows, ["gap_id", "gap_statement", "gap_type", "confidence", "evidence_count", "supporting_paper_ids", "supporting_titles", "rationale", "caveat"])
    write_csv(EVIDENCE_PATH, evidence_rows, ["gap_id", "paper_id", "title", "evidence_type", "evidence_text", "is_direct_quote"])
    generate_figure(candidates)
    print(f"Gap analysis complete: {len(candidates)} candidate gap hypotheses.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="replace existing gap-analysis outputs after an explicit request")
    args = parser.parse_args()
    return run(overwrite=args.overwrite)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
