"""Cache DBLP publication DOIs and abstracts for reviewer profiles.

    python enrich_publications.py --limit 10
    S2_API_KEY=... python enrich_publications.py

DBLP remains the authority for reviewer identities and publication lists. Its
person XML supplies DOI links but not abstracts. IEEE and ACM paper abstracts
are requested from Semantic Scholar's DOI batch API. Nothing scrapes publisher
web pages.

Two atomic, incremental caches are written. Transient failures remain
retryable, while successful and confirmed-missing results are reused.
"""

from __future__ import annotations

import argparse
import datetime
import html
import json
import os
import random
import re
import sys
import tempfile
import time
from pathlib import Path

import requests

from dblp import fetch_doi_records, normalise_doi
from area_chairs import load_area_chairs
from reviewers import load_reviewers

DEFAULT_CSV = "HPCA'27 PC Member Acceptance Form (Responses) - Form Responses 1.csv"
DEFAULT_METADATA_CACHE = "reviewer_publications.json"
DEFAULT_ABSTRACT_CACHE = "publication_abstracts.json"
METADATA_SCHEMA_VERSION = 1
ABSTRACT_SCHEMA_VERSION = 1
ALLOWED_PUBLISHERS = frozenset({"ieee", "acm"})
S2_BATCH_SIZE = 500
S2_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


def load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: str) -> None:
    p = Path(path)
    fd, tmp_name = tempfile.mkstemp(prefix=p.name + ".", suffix=".tmp", dir=p.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp, p)
    finally:
        tmp.unlink(missing_ok=True)


def publisher_for_doi(doi: str | None) -> str | None:
    doi = normalise_doi(doi)
    if doi and doi.startswith("10.1109/"):
        return "ieee"
    if doi and doi.startswith("10.1145/"):
        return "acm"
    return None


def clean_abstract(value: str | None) -> str:
    """Flatten common JATS/HTML markup and whitespace in API abstracts."""
    if not value:
        return ""
    return _SPACE_RE.sub(" ", html.unescape(_TAG_RE.sub(" ", value))).strip()


def safe_request_error(exc: requests.RequestException) -> str:
    """Summarize an HTTP failure without URLs, query strings, or API keys."""
    response = getattr(exc, "response", None)
    if response is not None:
        return f"HTTP {response.status_code} from {response.request.method} API request"
    return type(exc).__name__


def chunks(items: list[str], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def fetch_s2_abstracts(
    dois: list[str], api_key: str, session: requests.Session
) -> dict[str, dict]:
    """Return DOI-keyed Semantic Scholar batch results."""
    output: dict[str, dict] = {}
    headers = {"x-api-key": api_key} if api_key else {}
    for batch in chunks(dois, S2_BATCH_SIZE):
        for attempt in range(5):
            response = session.post(
                S2_URL,
                params={"fields": "title,year,abstract,externalIds"},
                headers=headers, json={"ids": [f"DOI:{doi}" for doi in batch]}, timeout=30,
            )
            if response.status_code != 429:
                response.raise_for_status()
                break
            retry_after = response.headers.get("Retry-After", "")
            time.sleep(int(retry_after) if retry_after.isdigit() else 2 ** attempt)
        else:
            response.raise_for_status()
        records = response.json()
        for requested, record in zip(batch, records):
            if record is None:
                output[requested] = {
                    "status": "not_found", "abstract": "", "source": "semantic_scholar"
                }
                continue
            returned = normalise_doi((record.get("externalIds") or {}).get("DOI"))
            if returned and returned != requested:
                output[requested] = {
                    "status": "not_found", "abstract": "", "source": "semantic_scholar"
                }
                continue
            abstract = clean_abstract(record.get("abstract"))
            output[requested] = {
                "status": "found" if abstract else "not_found",
                "abstract": abstract, "source": "semantic_scholar",
                "title": (record.get("title") or "").strip(),
                "year": record.get("year"),
            }
    return output


def selected_dois(metadata_cache: dict, years: int) -> dict[str, str]:
    cutoff = datetime.date.today().year - years + 1
    selected: dict[str, str] = {}
    for entry in metadata_cache.values():
        if not entry.get("complete"):
            continue
        for record in entry.get("records", []):
            try:
                year = int(record.get("year"))
            except (TypeError, ValueError):
                continue
            doi = normalise_doi(record.get("doi"))
            publisher = publisher_for_doi(doi)
            if year >= cutoff and doi and publisher in ALLOWED_PUBLISHERS:
                selected[doi] = publisher
    return selected


def enrich_abstract_cache(
    doi_publishers: dict[str, str], abstract_cache: dict, *, s2_key: str,
    session: requests.Session,
) -> tuple[int, int]:
    """Populate abstract_cache in place; return (found, attempted)."""
    pending = [
        doi for doi in sorted(doi_publishers)
        if abstract_cache.get(doi, {}).get("status") not in {"found", "not_found"}
    ]
    if not pending:
        return 0, 0

    results = fetch_s2_abstracts(pending, s2_key, session)

    retrieved_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for doi, result in results.items():
        abstract_cache[doi] = {
            **result, "doi": doi, "publisher": doi_publishers[doi],
            "retrieved_at": retrieved_at, "schema_version": ABSTRACT_SCHEMA_VERSION,
        }
    return sum(result["status"] == "found" for result in results.values()), len(results)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument(
        "--role", choices=("reviewer", "area-chair"), default="reviewer",
        help="acceptance-form schema to load (default: reviewer)",
    )
    parser.add_argument("--metadata-cache", default=DEFAULT_METADATA_CACHE)
    parser.add_argument("--abstract-cache", default=DEFAULT_ABSTRACT_CACHE)
    parser.add_argument("--years", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delay", type=float, default=3.0)
    args = parser.parse_args()
    if args.years <= 0:
        parser.error("--years must be greater than 0")
    if args.delay < 0:
        parser.error("--delay must be non-negative")

    s2_key = os.environ.get("S2_API_KEY", "").strip()
    if not s2_key:
        print(
            "INFO: S2_API_KEY is not set; using the shared unauthenticated S2 rate limit",
            file=sys.stderr,
        )

    reviewers = (
        load_area_chairs(args.csv) if args.role == "area-chair"
        else load_reviewers(args.csv)
    )
    if args.limit is not None:
        reviewers = reviewers[:args.limit]
    pids = list(dict.fromkeys(r.pid for r in reviewers if r.pid))
    metadata = load_json(args.metadata_cache)
    session = requests.Session()
    fetched = failed = attempted_pids = consecutive_failures = 0
    # Never-attempted reviewers go first so a few persistently unavailable
    # PIDs cannot starve the rest of the PC on every resumable run.
    pending_pids = (
        [pid for pid in pids if pid not in metadata]
        + [pid for pid in pids if pid in metadata and not metadata[pid].get("complete")]
    )
    for pid in pending_pids:
        entry = metadata.get(pid)
        if attempted_pids and args.delay:
            time.sleep(max(0.5, args.delay + random.uniform(-0.5, 0.5) * args.delay))
        attempted_pids += 1
        try:
            records = fetch_doi_records(pid, session)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            consecutive_failures += 1
            metadata[pid] = {
                "complete": False, "records": entry.get("records", []) if entry else [],
                "schema_version": METADATA_SCHEMA_VERSION,
            }
            print(f"WARN: DBLP DOI fetch failed for pid={pid}: {exc}", file=sys.stderr)
        else:
            fetched += 1
            consecutive_failures = 0
            metadata[pid] = {
                "complete": True, "records": records,
                "schema_version": METADATA_SCHEMA_VERSION,
            }
        save_json(metadata, args.metadata_cache)
        if consecutive_failures >= 3:
            print(
                "WARN: pausing DBLP metadata retrieval after 3 consecutive failures; "
                "a later run will resume from this cache",
                file=sys.stderr,
            )
            break

    abstracts = load_json(args.abstract_cache)
    dois = selected_dois(metadata, args.years)
    try:
        found, attempted = enrich_abstract_cache(
            dois, abstracts, s2_key=s2_key, session=session,
        )
    except requests.RequestException as exc:
        print(
            "ERROR: abstract API request failed; results remain retryable: "
            f"{safe_request_error(exc)}",
            file=sys.stderr,
        )
        return 1
    if attempted:
        save_json(abstracts, args.abstract_cache)

    total_found = sum(
        abstracts.get(doi, {}).get("status") == "found" for doi in dois
    )
    by_publisher = {
        p: sum(
            abstracts.get(doi, {}).get("status") == "found"
            for doi, publisher in dois.items() if publisher == p
        )
        for p in sorted(ALLOWED_PUBLISHERS)
    }
    print(
        f"DBLP metadata: {len(pids)} PIDs, {fetched} fetched, {failed} failed; "
        f"abstracts: {total_found}/{len(dois)} found "
        f"(IEEE {by_publisher['ieee']}, ACM {by_publisher['acm']}), "
        f"{found}/{attempted} found this run.", file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
