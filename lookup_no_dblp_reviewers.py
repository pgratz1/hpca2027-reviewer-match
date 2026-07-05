"""Identify reviewers with no usable DBLP link and pull their recent papers.

    python lookup_no_dblp_reviewers.py --limit 10   # validate on a handful first
    python lookup_no_dblp_reviewers.py              # full run (~53 reviewers)

For each accepted reviewer with no DBLP PID (reviewers.load_reviewers),
identity_recovery.resolve classifies their raw DBLP-link field and either:
  - recovers a DBLP PID (an ORCID, a legacy dblp.uni-trier.de URL, or a DBLP
    search URL that resolved to exactly one person) — titles then come from
    the normal dblp.fetch_titles path;
  - finds an OpenAlex author by ORCID or by name+affiliation search;
  - is flagged for manual review (no name to search on, or a Google Scholar
    link this script doesn't auto-fetch); or
  - hit a network failure partway through (identity_recovery.resolve's own
    'lookup_error' path) — distinct from a confident "not found" so it's
    obvious from the report which rows are worth simply re-running.

Recent (--years, default 3) titles are then validated against the reviewer's
declared primary/secondary/tertiary area + free-text keywords by plain
keyword/concept overlap (openalex.tokenize) — concepts (OpenAlex's curated
field-of-study tags) are the primary evidence, titles are the fallback. No
model load, just a word list a human can audit in the CSV.

Writes one row per no-PID reviewer to --report (default
no_dblp_lookup_report.csv) incrementally (flushed after each reviewer, so a
crash partway through a run still leaves a usable partial report) and prints
a summary to stderr. A single reviewer whose processing raises an unexpected
exception is recorded as an "error" row rather than aborting the whole run.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from collections import Counter

import requests

import dblp
import identity_recovery as ir
import openalex as oa
from reviewers import Reviewer, load_reviewers

DEFAULT_CSV = "HPCA'27 PC Member Acceptance Form (Responses) - Form Responses 1.csv"
DEFAULT_DBLP_CACHE = "dblp_cache.json"
DEFAULT_COLLEAGUE_CACHE = "dblp_pubs_cache.json"
DEFAULT_OPENALEX_CACHE = "openalex_cache.json"
DEFAULT_REPORT = "no_dblp_lookup_report.csv"

REPORT_FIELDS = [
    "email", "name", "tier", "affiliation",
    "primary", "secondary", "tertiary",
    "resolution_path", "identity_confidence",
    "dblp_pid", "openalex_author_id", "matched_institution",
    "n_recent_works", "area_validation", "matched_terms",
    "sample_titles", "candidates_considered", "note",
]

# Live-dblp.org paths this script can hit outside of dblp.fetch_titles's own
# (properly paced) live-fetch tracking — resolve_legacy_url/resolve_via_author_search
# have no cache of their own, so these paths always make a fresh request.
_LIVE_DBLP_RESOLUTION_PATHS = ("dblp_legacy", "dblp_search")


def reviewer_area_tokens(r: Reviewer) -> set[str]:
    text = " ".join([r.primary, r.secondary, r.tertiary, r.keywords])
    return oa.tokenize(text)


def validate_area(reviewer_tokens: set[str], works: list[dict]) -> tuple[str, set[str]]:
    """Keyword/concept overlap between a reviewer's declared area and their
    recent works. Concepts (curated field tags) are preferred evidence over
    raw title words when both are available."""
    if not works:
        return "NO_WORKS", set()
    concept_overlap = reviewer_tokens & oa.concept_tokens(works)
    if concept_overlap:
        return "AREA_MATCH", concept_overlap
    title_overlap = reviewer_tokens & oa.title_tokens(works)
    if title_overlap:
        return "AREA_MATCH", title_overlap
    return "NO_AREA_SIGNAL", set()


def dblp_titles_to_works(titles: list[tuple[int, str]]) -> list[dict]:
    """Normalize DBLP (year, title) pairs to the same work-dict shape OpenAlex
    returns (minus concepts, which DBLP doesn't have), so area validation and
    report formatting can treat both sources uniformly."""
    return [{"title": title, "publication_year": year, "concepts": []} for year, title in titles]


def candidate_summary(candidates: list[dict]) -> str:
    parts = []
    for c in candidates[:5]:
        insts = ", ".join(oa.institution_names(c)[:2])
        parts.append(f"{c.get('display_name')} ({c.get('works_count', 0)} works; {insts})")
    return " | ".join(parts)


def process_reviewer(
    r: Reviewer,
    session: requests.Session,
    years: int,
    dblp_write_cache: dict,
    dblp_readonly_cache: dict,
    openalex_cache: dict,
    dblp_cache_path: str,
) -> tuple[dict, bool]:
    """Resolve one reviewer's identity and recent works.

    Returns (report_row, made_live_dblp_call) — the latter drives the caller's
    inter-reviewer pacing (see main()).
    """
    res = ir.resolve(r, session, openalex_cache=openalex_cache)

    works: list[dict] = []
    institution = ""
    author_id = ""
    made_live_dblp_call = res.path in _LIVE_DBLP_RESOLUTION_PATHS

    if res.dblp_pid:
        titles, source = dblp.fetch_titles(
            res.dblp_pid, years=years, session=session,
            write_cache=dblp_write_cache, readonly_cache=dblp_readonly_cache,
            fallback_when_thin=True,
        )
        made_live_dblp_call = made_live_dblp_call or source == "live"
        if source == "live":
            dblp.save_cache(dblp_write_cache, dblp_cache_path)
        works = dblp_titles_to_works(titles)
    elif res.openalex_author:
        author_id = res.openalex_author["id"]
        institution = "; ".join(oa.institution_names(res.openalex_author)[:2])
        works = oa.fetch_recent_works(author_id, years, session, cache=openalex_cache)

    if res.dblp_pid or res.openalex_author:
        tokens = reviewer_area_tokens(r)
        area_flag, matched = validate_area(tokens, works)
    else:
        area_flag, matched = "N/A", set()

    titles_sample = "; ".join(w["title"] for w in works[:5] if w.get("title"))

    row = {
        "email": r.email,
        "name": r.name,
        "tier": r.tier,
        "affiliation": r.affiliation,
        "primary": r.primary,
        "secondary": r.secondary,
        "tertiary": r.tertiary,
        "resolution_path": res.path,
        "identity_confidence": res.confidence or "",
        "dblp_pid": res.dblp_pid or "",
        "openalex_author_id": author_id,
        "matched_institution": institution,
        "n_recent_works": len(works),
        "area_validation": area_flag,
        "matched_terms": ", ".join(sorted(matched)),
        "sample_titles": titles_sample,
        "candidates_considered": candidate_summary(res.candidates),
        "note": res.note,
    }

    print(
        f"  {r.name} <{r.email}>  path={res.path} confidence={res.confidence} "
        f"n_works={len(works)} area={area_flag}",
        file=sys.stderr,
    )
    return row, made_live_dblp_call


def error_row(r: Reviewer, exc: Exception) -> dict:
    """A report row for a reviewer whose processing raised unexpectedly, so
    the run can continue instead of losing every already-written row."""
    row = dict.fromkeys(REPORT_FIELDS, "")
    row.update(
        email=r.email, name=r.name, tier=r.tier, affiliation=r.affiliation,
        primary=r.primary, secondary=r.secondary, tertiary=r.tertiary,
        resolution_path="error", note=str(exc),
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=DEFAULT_CSV, help="path to the reviewer CSV")
    parser.add_argument("--dblp-cache", default=DEFAULT_DBLP_CACHE, help="path to the writable DBLP title cache")
    parser.add_argument(
        "--colleague-cache", default=DEFAULT_COLLEAGUE_CACHE,
        help="path to the colleague's read-only pre-built DBLP cache",
    )
    parser.add_argument("--openalex-cache", default=DEFAULT_OPENALEX_CACHE, help="path to the writable OpenAlex cache")
    parser.add_argument("--report", default=DEFAULT_REPORT, help="path to write the CSV report")
    parser.add_argument("--years", type=int, default=3, help="most recent N calendar years of publications (default: 3)")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N no-PID reviewers")
    parser.add_argument(
        "--delay", type=float, default=3.0,
        help="seconds to jitter between reviewers after one made a live dblp.org request (default: 3.0)",
    )
    args = parser.parse_args()

    reviewers = load_reviewers(args.csv)
    no_pid = [r for r in reviewers if not r.pid]
    if args.limit is not None:
        no_pid = no_pid[: args.limit]

    print(
        f"Loaded {len(reviewers)} accepted reviewers; {len(no_pid)} have no DBLP PID.",
        file=sys.stderr,
    )

    dblp_write_cache = dblp.load_cache(args.dblp_cache)
    dblp_readonly_cache = dblp.load_colleague_cache(args.colleague_cache)
    openalex_cache = dblp.load_cache(args.openalex_cache)

    session = requests.Session()
    path_counts: Counter = Counter()
    area_counts: Counter = Counter()
    last_was_live_dblp = False
    written = 0

    with open(args.report, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        f.flush()

        for r in no_pid:
            if last_was_live_dblp and args.delay:
                jitter = random.uniform(-0.5, 0.5) * args.delay
                time.sleep(max(0.5, args.delay + jitter))
            last_was_live_dblp = False

            try:
                row, last_was_live_dblp = process_reviewer(
                    r, session, args.years, dblp_write_cache, dblp_readonly_cache,
                    openalex_cache, args.dblp_cache,
                )
            except Exception as exc:  # noqa: BLE001 - one reviewer's failure must not lose the whole report
                print(f"  ERROR: {r.name} <{r.email}>: {exc}", file=sys.stderr)
                row = error_row(r, exc)
            finally:
                # Whatever this iteration did or didn't complete, persist any
                # OpenAlex cache entries it may have already written (e.g. an
                # ORCID/name-search lookup that succeeded before a later
                # fetch_recent_works call failed).
                dblp.save_cache(openalex_cache, args.openalex_cache)

            writer.writerow(row)
            f.flush()
            written += 1

            path_key = (
                f"{row['resolution_path']}:{row['identity_confidence']}"
                if row["identity_confidence"] else row["resolution_path"]
            )
            path_counts[path_key] += 1
            area_counts[row["area_validation"]] += 1

    print(f"\nWrote {written} rows to {args.report}", file=sys.stderr)
    print(f"Resolution paths: {dict(path_counts)}", file=sys.stderr)
    print(f"Area validation:  {dict(area_counts)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
