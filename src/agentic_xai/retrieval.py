"""Retrieve, screen, and deduplicate the healthcare-XAI literature corpus."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from .config import DATA_DIR, OUTPUTS_DIR, PROJECT_ROOT, get_arxiv_contact_email, get_semantic_scholar_api_key


CORPUS_PATH = DATA_DIR / "corpus.csv"
REJECTED_PATH = OUTPUTS_DIR / "rejected_papers.csv"
MANIFEST_PATH = OUTPUTS_DIR / "corpus_manifest.csv"
STATS_PATH = OUTPUTS_DIR / "corpus_stats.json"
PROGRESS_PATH = OUTPUTS_DIR / "retrieval_progress.json"

SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV_URL = "https://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_QUERIES = [
    "explainable artificial intelligence healthcare",
    "explainable AI healthcare",
    "XAI healthcare",
    "interpretable machine learning healthcare",
]
ARXIV_QUERIES = [
    'all:"explainable artificial intelligence" AND all:healthcare',
    'all:"explainable AI" AND all:healthcare',
    "all:XAI AND all:healthcare",
    'all:"interpretable machine learning" AND all:healthcare',
]
TARGET_PER_SOURCE = 100
SEMANTIC_SCHOLAR_PAGE_SIZE = 100
SEMANTIC_SCHOLAR_FALLBACK_PAGE_SIZE = 50
SEMANTIC_SCHOLAR_FIELDS = (
    "paperId,title,abstract,authors,year,venue,externalIds,url,citationCount,publicationDate"
)
SEMANTIC_SCHOLAR_FALLBACK_FIELDS = "paperId,title,abstract,year,externalIds,citationCount"
ARXIV_PAGE_SIZE = 50
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_RETRIES = 5
SEMANTIC_SCHOLAR_MIN_INTERVAL = 1.0
ARXIV_MIN_INTERVAL = 3.0

XAI_TERMS = [
    "explainable artificial intelligence",
    "explainable ai",
    "explainability",
    "interpretable machine learning",
    "interpretable ai",
    "interpretability",
    "xai",
    "explainable machine learning",
    "model interpretability",
]
HEALTHCARE_TERMS = [
    "healthcare",
    "health care",
    "clinical",
    "clinician",
    "medical",
    "medicine",
    "hospital",
    "patient",
    "diagnosis",
    "diagnostic",
    "radiology",
    "pathology",
    "electronic health record",
    "ehr",
    "disease",
    "health",
    "biomedical",
]
CSV_FIELDS = [
    "source",
    "paper_id",
    "title",
    "abstract",
    "authors",
    "year",
    "venue",
    "doi",
    "url",
    "citation_count",
    "publication_date",
    "relevance_status",
    "rejection_reason",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_doi(value: Any) -> str:
    doi = clean_text(value).casefold()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    return doi.removeprefix("doi:").strip().rstrip(".")


def normalize_title(value: Any) -> str:
    title = clean_text(value).casefold()
    return re.sub(r"[^a-z0-9]+", " ", title).strip()


def empty_record() -> dict[str, Any]:
    return {field: "" for field in CSV_FIELDS}


def record_completeness(record: dict[str, Any]) -> tuple[int, int, int]:
    populated = sum(bool(clean_text(record.get(field))) for field in CSV_FIELDS)
    abstract_length = len(clean_text(record.get("abstract")))
    title_length = len(clean_text(record.get("title")))
    return populated, abstract_length, title_length


class RateLimiter:
    def __init__(self, minimum_interval: float) -> None:
        self.minimum_interval = minimum_interval
        self.last_request = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self.last_request
        if elapsed < self.minimum_interval:
            time.sleep(self.minimum_interval - elapsed)
        self.last_request = time.monotonic()


def request_with_retry(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    headers: dict[str, str],
    limiter: RateLimiter,
    source: str,
    failures: dict[str, Any],
) -> requests.Response | None:
    for attempt in range(MAX_RETRIES + 1):
        limiter.wait()
        try:
            response = session.get(url, params=params, headers=headers, timeout=30)
        except requests.RequestException as exc:
            if attempt >= MAX_RETRIES:
                failures[source] = {
                    "status": "failed",
                    "error": f"request error after retries: {exc}",
                }
                return None
            time.sleep(min(60.0, 2**attempt))
            continue

        if response.status_code == 200:
            return response

        if response.status_code in {401, 403}:
            failures[source] = {
                "status": "failed",
                "http_status": response.status_code,
                "error": "authentication or authorization failure; no retry performed",
            }
            return None

        if response.status_code not in RETRYABLE_STATUS_CODES or attempt >= MAX_RETRIES:
            failures[source] = {
                "status": "failed",
                "http_status": response.status_code,
                "error": "non-retryable HTTP failure or retry limit reached",
            }
            return None

        retry_after = response.headers.get("Retry-After")
        try:
            delay = float(retry_after) if retry_after else 2**attempt
        except ValueError:
            delay = 2**attempt
        time.sleep(min(60.0, max(1.0, delay)))

    return None


def semantic_scholar_record(item: dict[str, Any]) -> dict[str, Any]:
    external_ids = item.get("externalIds") or {}
    authors = [clean_text(author.get("name")) for author in item.get("authors") or []]
    record = empty_record()
    record.update(
        {
            "source": "semantic_scholar",
            "paper_id": clean_text(item.get("paperId")),
            "title": clean_text(item.get("title")),
            "abstract": clean_text(item.get("abstract")),
            "authors": "; ".join(author for author in authors if author),
            "year": item.get("year") or "",
            "venue": clean_text(item.get("venue")),
            "doi": clean_text(external_ids.get("DOI")),
            "url": clean_text(item.get("url")),
            "citation_count": item.get("citationCount")
            if item.get("citationCount") is not None
            else "",
            "publication_date": clean_text(item.get("publicationDate")),
        }
    )
    return record


def arxiv_record(entry: ET.Element, namespace: str) -> dict[str, Any]:
    def find_text(name: str) -> str:
        element = entry.find(f"{{{namespace}}}{name}")
        return clean_text(element.text if element is not None else "")

    identifier_url = find_text("id")
    identifier = urlparse(identifier_url).path.rsplit("/", 1)[-1]
    identifier = re.sub(r"v\d+$", "", identifier)
    authors = [
        clean_text(author.findtext(f"{{{namespace}}}name", default=""))
        for author in entry.findall(f"{{{namespace}}}author")
    ]
    doi_element = next(
        (element for element in entry if element.tag.rsplit("}", 1)[-1] == "doi"),
        None,
    )
    record = empty_record()
    record.update(
        {
            "source": "arxiv",
            "paper_id": identifier,
            "title": find_text("title"),
            "abstract": find_text("summary"),
            "authors": "; ".join(author for author in authors if author),
            "year": find_text("published")[:4],
            "venue": "arXiv",
            "doi": clean_text(doi_element.text if doi_element is not None else ""),
            "url": identifier_url,
            "citation_count": "",
            "publication_date": find_text("published"),
        }
    )
    return record


def save_progress(
    raw_records: list[dict[str, Any]],
    source_status: dict[str, Any],
    started_at: str,
) -> None:
    payload = {
        "retrieval_started_at": started_at,
        "last_updated_at": utc_now(),
        "raw_records": raw_records,
        "source_status": source_status,
    }
    write_json_atomic(PROGRESS_PATH, payload)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary_path.replace(path)


def retrieve_semantic_scholar(
    session: requests.Session,
    raw_records: list[dict[str, Any]],
    source_status: dict[str, Any],
    started_at: str,
    api_key: str,
) -> None:
    source = "semantic_scholar"
    status = {
        "status": "started",
        "target": TARGET_PER_SOURCE,
        "retrieved": 0,
        "queries": SEMANTIC_SCHOLAR_QUERIES,
    }
    source_status[source] = status
    if not api_key:
        status.update(
            {
                "status": "failed",
                "error": "SEMANTIC_SCHOLAR_API_KEY is not set",
            }
        )
        save_progress(raw_records, source_status, started_at)
        return

    headers = {"x-api-key": api_key, "Accept": "application/json"}
    limiter = RateLimiter(SEMANTIC_SCHOLAR_MIN_INTERVAL)
    for query in SEMANTIC_SCHOLAR_QUERIES:
        offset = 0
        while len([r for r in raw_records if r["source"] == source]) < TARGET_PER_SOURCE:
            response = None
            request_failures: dict[str, Any] = {}
            remaining = TARGET_PER_SOURCE - len(
                [r for r in raw_records if r["source"] == source]
            )
            page_limit = min(SEMANTIC_SCHOLAR_PAGE_SIZE, remaining)
            field_selection = SEMANTIC_SCHOLAR_FIELDS
            request_variants = [
                (min(SEMANTIC_SCHOLAR_PAGE_SIZE, remaining), SEMANTIC_SCHOLAR_FIELDS),
                (
                    min(SEMANTIC_SCHOLAR_FALLBACK_PAGE_SIZE, remaining),
                    SEMANTIC_SCHOLAR_FALLBACK_FIELDS,
                ),
            ]
            for candidate_limit, candidate_fields in request_variants:
                params = {
                    "query": query,
                    "offset": offset,
                    "limit": candidate_limit,
                    "fields": candidate_fields,
                }
                response = request_with_retry(
                    session,
                    SEMANTIC_SCHOLAR_URL,
                    params,
                    headers,
                    limiter,
                    source,
                    request_failures,
                )
                if response is not None:
                    page_limit = candidate_limit
                    field_selection = candidate_fields
                    break
            if response is None:
                status.update(
                    {
                        "status": "failed",
                        "error": "Semantic Scholar request failed after primary and fallback requests",
                        "request_failures": request_failures,
                    }
                )
                save_progress(raw_records, source_status, started_at)
                return
            try:
                payload = response.json()
            except ValueError as exc:
                source_status[source] = {
                    "status": "failed",
                    "error": f"invalid JSON response: {exc}",
                }
                save_progress(raw_records, source_status, started_at)
                return

            page = payload.get("data") or []
            for item in page:
                raw_records.append(semantic_scholar_record(item))
            status["retrieved"] = len(
                [r for r in raw_records if r["source"] == source]
            )
            status["pages"] = status.get("pages", 0) + 1
            status["last_request"] = {
                "limit": page_limit,
                "fields": field_selection,
            }
            save_progress(raw_records, source_status, started_at)
            if not page or len(page) < page_limit:
                break
            offset += len(page)
    status["status"] = "completed"
    save_progress(raw_records, source_status, started_at)


def retrieve_arxiv(
    session: requests.Session,
    raw_records: list[dict[str, Any]],
    source_status: dict[str, Any],
    started_at: str,
    contact_email: str,
) -> None:
    source = "arxiv"
    status = {
        "status": "started",
        "target": TARGET_PER_SOURCE,
        "retrieved": 0,
        "queries": ARXIV_QUERIES,
    }
    source_status[source] = status
    headers = {
        "User-Agent": f"agentic-ai-xai-research/0.1 ({contact_email})",
        "Accept": "application/atom+xml",
    }
    limiter = RateLimiter(ARXIV_MIN_INTERVAL)
    namespace = "http://www.w3.org/2005/Atom"
    for query in ARXIV_QUERIES:
        start = 0
        while len([r for r in raw_records if r["source"] == source]) < TARGET_PER_SOURCE:
            params = {
                "search_query": query,
                "start": start,
                "max_results": ARXIV_PAGE_SIZE,
                "sortBy": "relevance",
                "sortOrder": "descending",
            }
            response = request_with_retry(
                session,
                ARXIV_URL,
                params,
                headers,
                limiter,
                source,
                source_status,
            )
            if response is None:
                save_progress(raw_records, source_status, started_at)
                return
            try:
                root = ET.fromstring(response.content)
            except ET.ParseError as exc:
                source_status[source] = {
                    "status": "failed",
                    "error": f"invalid Atom response: {exc}",
                }
                save_progress(raw_records, source_status, started_at)
                return

            entries = root.findall(f"{{{namespace}}}entry")
            for entry in entries:
                raw_records.append(arxiv_record(entry, namespace))
            status["retrieved"] = len(
                [r for r in raw_records if r["source"] == source]
            )
            status["pages"] = status.get("pages", 0) + 1
            save_progress(raw_records, source_status, started_at)
            if not entries or len(entries) < ARXIV_PAGE_SIZE:
                break
            start += len(entries)
    status["status"] = "completed"
    save_progress(raw_records, source_status, started_at)


def screen_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for record in records:
        searchable_text = f"{record.get('title', '')} {record.get('abstract', '')}".casefold()
        xai_matches = [term for term in XAI_TERMS if re.search(r"\b" + re.escape(term) + r"\b", searchable_text)]
        healthcare_matches = [
            term
            for term in HEALTHCARE_TERMS
            if re.search(r"\b" + re.escape(term) + r"\b", searchable_text)
        ]
        reasons = []
        if not xai_matches:
            reasons.append("no XAI or interpretability signal")
        if not healthcare_matches:
            reasons.append("no healthcare or medical-application signal")
        record = dict(record)
        if reasons:
            record["relevance_status"] = "rejected"
            record["rejection_reason"] = "; ".join(reasons)
            rejected.append(record)
        else:
            record["relevance_status"] = "accepted"
            record["rejection_reason"] = ""
            accepted.append(record)
    return accepted, rejected


def deduplicate_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    selected: list[dict[str, Any]] = []
    key_to_index: dict[tuple[str, ...], int] = {}
    for record in records:
        doi = normalize_doi(record.get("doi"))
        paper_id = clean_text(record.get("paper_id")).casefold()
        title = normalize_title(record.get("title"))
        keys: list[tuple[str, ...]] = []
        if doi:
            keys.append(("doi", doi))
        if paper_id:
            keys.append(("paper_id", clean_text(record.get("source")), paper_id))
        if title:
            keys.append(("title", title))
        existing_indices = [key_to_index[key] for key in keys if key in key_to_index]
        if existing_indices:
            index = min(existing_indices)
            if record_completeness(record) > record_completeness(selected[index]):
                selected[index] = record
            for key in keys:
                key_to_index[key] = index
        else:
            index = len(selected)
            selected.append(record)
            if keys:
                for key in keys:
                    key_to_index[key] = index
            else:
                key_to_index[("unique", str(index))] = index
    return selected, len(records) - len(selected)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def has_existing_results() -> bool:
    return any(path.exists() for path in (STATS_PATH, REJECTED_PATH, MANIFEST_PATH)) or (
        CORPUS_PATH.exists()
        and sum(1 for _ in CORPUS_PATH.open(encoding="utf-8")) > 1
    )


def build_stats(
    raw_records: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    duplicates_removed: int,
    source_status: dict[str, Any],
    started_at: str,
    finished_at: str,
) -> dict[str, Any]:
    source_counts = Counter(record.get("source", "unknown") for record in accepted)
    return {
        "retrieval_started_at": started_at,
        "retrieval_finished_at": finished_at,
        "semantic_scholar_retrieved": sum(
            record.get("source") == "semantic_scholar" for record in raw_records
        ),
        "arxiv_retrieved": sum(record.get("source") == "arxiv" for record in raw_records),
        "raw_combined": len(raw_records),
        "relevance_qualified": len(accepted),
        "rejected": len(rejected),
        "duplicates_removed": duplicates_removed,
        "final_unique_corpus": len(accepted),
        "papers_with_abstracts": sum(bool(clean_text(r.get("abstract"))) for r in accepted),
        "papers_with_doi": sum(bool(normalize_doi(r.get("doi"))) for r in accepted),
        "papers_with_citation_counts": sum(
            r.get("citation_count") not in (None, "") for r in accepted
        ),
        "source_distribution": dict(source_counts),
        "api_failure_status": source_status,
        "retrieval_queries": {
            "semantic_scholar": SEMANTIC_SCHOLAR_QUERIES,
            "arxiv": ARXIV_QUERIES,
        },
        "retrieval_configuration": {
            "target_per_source": TARGET_PER_SOURCE,
            "semantic_scholar_page_size": SEMANTIC_SCHOLAR_PAGE_SIZE,
            "semantic_scholar_fallback_page_size": SEMANTIC_SCHOLAR_FALLBACK_PAGE_SIZE,
            "semantic_scholar_fields": SEMANTIC_SCHOLAR_FIELDS.split(","),
            "semantic_scholar_fallback_fields": SEMANTIC_SCHOLAR_FALLBACK_FIELDS.split(","),
            "arxiv_page_size": ARXIV_PAGE_SIZE,
            "arxiv_minimum_request_interval_seconds": ARXIV_MIN_INTERVAL,
            "screening_terms": {
                "xai": XAI_TERMS,
                "healthcare": HEALTHCARE_TERMS,
            },
            "deduplication_strategy": ["doi", "source paper ID", "normalized title"],
            "deduplication_is_fuzzy": False,
        },
    }


def run(overwrite: bool = False) -> int:
    if has_existing_results() and not overwrite:
        print(
            "Existing retrieval outputs detected. Use --overwrite to replace them.",
            file=sys.stderr,
        )
        return 2

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    raw_records: list[dict[str, Any]] = []
    source_status: dict[str, Any] = {}
    session = requests.Session()
    retrieve_semantic_scholar(
        session,
        raw_records,
        source_status,
        started_at,
        get_semantic_scholar_api_key(),
    )
    retrieve_arxiv(
        session,
        raw_records,
        source_status,
        started_at,
        get_arxiv_contact_email(),
    )

    deduplicated, duplicates_removed = deduplicate_records(raw_records)
    accepted, rejected = screen_records(deduplicated)
    write_csv(CORPUS_PATH, accepted, CSV_FIELDS)
    write_csv(REJECTED_PATH, rejected, CSV_FIELDS)
    manifest = [
        {
            "corpus_index": index,
            "source": record.get("source", ""),
            "paper_id": record.get("paper_id", ""),
            "title": record.get("title", ""),
            "year": record.get("year", ""),
            "doi": record.get("doi", ""),
            "has_abstract": bool(clean_text(record.get("abstract"))),
        }
        for index, record in enumerate(accepted, start=1)
    ]
    write_csv(
        MANIFEST_PATH,
        manifest,
        ["corpus_index", "source", "paper_id", "title", "year", "doi", "has_abstract"],
    )
    write_json_atomic(
        STATS_PATH,
        build_stats(
            raw_records,
            accepted,
            rejected,
            duplicates_removed,
            source_status,
            started_at,
            utc_now(),
        ),
    )
    print(f"Retrieval complete: {len(accepted)} papers in {CORPUS_PATH}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing retrieval outputs after an explicit request",
    )
    args = parser.parse_args()
    return run(overwrite=args.overwrite)


if __name__ == "__main__":
    raise SystemExit(main())
