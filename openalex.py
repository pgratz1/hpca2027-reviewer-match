"""OpenAlex access: identify a reviewer who has no usable DBLP link, and pull
their recent works.

Used as a fallback for reviewers whose acceptance-form DBLP field is blank,
a personal homepage, or otherwise unresolvable to a DBLP PID (see
identity_recovery.py for the dispatch logic that decides when this module is
used). OpenAlex (https://openalex.org) is a free, keyless, open bibliographic
index with author search, direct ORCID lookup, and per-work "concepts"
(curated field-of-study tags) — the concepts are the hook the caller uses to
validate a candidate against the reviewer's declared HPCA area.

No API key is required; a `mailto` param opts into OpenAlex's "polite pool"
(higher, more reliable rate limits — no signup).
"""

from __future__ import annotations

import datetime
import re

import requests

from dblp import get_with_retry, windowed_with_fallback

_BASE = "https://api.openalex.org"
_MAILTO = "pgratz@gratz1.com"
_USER_AGENT = (
    f"HPCA2027-reviewer-match/0.1 (PC reviewer-paper matching; mailto:{_MAILTO})"
)

_STOPWORDS = {
    "and", "the", "for", "with", "using", "based", "via", "from", "into",
    "non", "ml", "emerging", "workloads",
    # Generic institution-name words: without these, affiliation matching
    # (pick_candidate) false-positives on any two university names sharing
    # a word like "university" rather than their distinctive part.
    "university", "universities", "college", "institute", "institutes",
    "national", "technology", "technological", "technical", "polytechnic",
    "school", "academy", "laboratory", "laboratories", "research",
    "center", "centre",
}


# Taxonomy plurals too short for the general len(token) > 4 rule below (which
# exists to avoid mangling 4-letter words like "bias"/"axis" that only look
# plural) — "gpus" is exactly 4 characters but is a real HPCA area name (see
# CLAUDE.md's taxonomy) and needs to match OpenAlex's singular "gpu" concept tag.
_SHORT_PLURALS = {"gpus": "gpu"}


def _singularize(token: str) -> str:
    """Naive plural stripper: "systems" -> "system", "networks" -> "network".

    The HPCA area taxonomy is plural-heavy ("Memory systems", "Interconnection
    networks", "ML architectures") while OpenAlex concepts are often singular
    ("Operating system") — without this, a plain word-overlap check misses
    otherwise-exact matches on nothing but grammatical number. Skips words
    ending "ss" (e.g. "class") to avoid mangling those.
    """
    if token in _SHORT_PLURALS:
        return _SHORT_PLURALS[token]
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> set[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords and short tokens.

    Shared by both sides of the area-validation overlap check in
    lookup_no_dblp_reviewers.py (reviewer's declared area/keywords vs. this
    module's OpenAlex concepts/titles) so the two vocabularies are normalized
    the same way.
    """
    tokens = re.split(r"[^a-z0-9]+", (text or "").lower())
    return {_singularize(t) for t in tokens if len(t) >= 3 and t not in _STOPWORDS}


# ---------------------------------------------------------------------------
# Author identification
# ---------------------------------------------------------------------------

def lookup_by_orcid(
    orcid: str, session: requests.Session, cache: dict | None = None
) -> dict | None:
    """Fetch the OpenAlex author record for an ORCID iD. Deterministic — no
    disambiguation needed. Returns None if OpenAlex has no author for it.

    `cache` (shared with search_by_name/fetch_recent_works, keyed by call
    type) is checked first and populated on a live hit — pass the same dict
    across calls and persist it (e.g. via dblp.save_cache) to avoid re-fetching
    on a later run.

    Only a genuine 404 (OpenAlex has no author for this ORCID) is treated as
    "not found" and cached as such. Any other failure (timeout, 5xx, exhausted
    429 retries) propagates as requests.RequestException instead of being
    cached as a false negative — a transient outage shouldn't permanently
    poison this ORCID as unresolvable.
    """
    key = f"orcid:{orcid}"
    if cache is not None and key in cache:
        return cache[key]
    try:
        resp = get_with_retry(
            session, f"{_BASE}/authors/orcid:{orcid}", params={"mailto": _MAILTO},
            headers={"User-Agent": _USER_AGENT}, backoff_floor=2,
        )
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            author = None
        else:
            raise
    else:
        author = resp.json()
    if cache is not None:
        cache[key] = author
    return author


def search_by_name(
    name: str, session: requests.Session, cache: dict | None = None
) -> list[dict]:
    """Return up to 10 OpenAlex author candidates matching `name`."""
    key = f"search:{name}"
    if cache is not None and key in cache:
        return cache[key]
    resp = get_with_retry(
        session, f"{_BASE}/authors",
        params={"search": name, "mailto": _MAILTO, "per-page": 10},
        headers={"User-Agent": _USER_AGENT}, backoff_floor=2,
    )
    results = resp.json().get("results", [])
    if cache is not None:
        cache[key] = results
    return results


def institution_names(candidate: dict) -> list[str]:
    names = []
    for aff in candidate.get("affiliations") or []:
        inst = aff.get("institution") or {}
        if inst.get("display_name"):
            names.append(inst["display_name"])
    return names


def pick_candidate(
    candidates: list[dict], affiliation: str
) -> tuple[dict | None, str, list[dict]]:
    """Disambiguate name-search candidates by the reviewer's declared affiliation.

    Returns (chosen, confidence, considered):
      - MATCH: exactly one candidate's institution history overlaps the
        reviewer's declared affiliation. chosen is that candidate.
      - AMBIGUOUS: more than one candidate overlaps. chosen is None — never
        guess between them; `considered` holds all of them for manual review.
      - LOW_CONFIDENCE: no candidate's institutions overlap. chosen is the
        highest-works-count candidate, flagged for a manual check.
      - NOT_FOUND: the search returned nothing. chosen is None.
    """
    if not candidates:
        return None, "NOT_FOUND", []

    aff_tokens = tokenize(affiliation)
    matches = []
    for c in candidates:
        inst_tokens: set[str] = set()
        for name in institution_names(c):
            inst_tokens |= tokenize(name)
        if aff_tokens & inst_tokens:
            matches.append(c)
    if len(matches) == 1:
        return matches[0], "MATCH", matches
    if len(matches) > 1:
        return None, "AMBIGUOUS", matches

    best = max(candidates, key=lambda c: c.get("works_count", 0))
    return best, "LOW_CONFIDENCE", [best]


# ---------------------------------------------------------------------------
# Works
# ---------------------------------------------------------------------------

def fetch_recent_works(
    author_id: str, years: int, session: requests.Session,
    current_year: int | None = None, cache: dict | None = None,
    min_count: int = 10, fallback_count: int = 10,
) -> list[dict]:
    """Works from an author's most recent `years` calendar years; if that
    window has fewer than `min_count`, falls back to their `fallback_count`
    most recent works overall (or however many they have, if fewer).

    A reviewer who's gone quiet recently but has a real publication history
    otherwise gets an almost-empty profile; this gives them a usable one
    instead. `author_id` may be a bare OpenAlex ID ("A123...") or a full URL
    ("https://openalex.org/A123...").

    Fetches the top 50 works by publication date (unfiltered by year) rather
    than applying the year window at the API level, so the cache holds one
    reusable batch per author regardless of `years` — mirrors dblp.py's
    fetch-all-then-filter-at-read-time caching. The window/fallback policy
    itself is dblp.windowed_with_fallback, shared with dblp.py's select_recent.
    """
    aid = author_id.rsplit("/", 1)[-1]

    key = f"works:{aid}"
    if cache is not None and key in cache:
        all_works = cache[key]
    else:
        resp = get_with_retry(
            session, f"{_BASE}/works",
            params={
                "filter": f"author.id:{aid}",
                "select": "title,publication_year,concepts",
                "sort": "publication_date:desc",
                "per-page": 50,
                "mailto": _MAILTO,
            },
            headers={"User-Agent": _USER_AGENT}, backoff_floor=2,
        )
        all_works = resp.json().get("results", [])
        if cache is not None:
            cache[key] = all_works

    all_works = dedup_works(all_works)
    return windowed_with_fallback(
        all_works, lambda w: w.get("publication_year") or 0,
        years, current_year, min_count, fallback_count,
    )


def dedup_works(works: list[dict]) -> list[dict]:
    """Drop duplicate works (OpenAlex lists preprint + venue versions, and
    sometimes differing punctuation/capitalization of the same title,
    separately). Keeps the first occurrence (works is assumed sorted
    publication_date-descending, so that's the most recent copy) and merges
    in any concepts a later duplicate has that the kept copy lacks.

    Never mutates a work dict in place — `works` may be the same list object
    held by a caller's cache dict (see fetch_recent_works), and mutating a
    kept entry in place would silently corrupt what's supposed to be a
    pristine copy of the raw API response.
    """
    seen: dict[str, int] = {}  # normalised_title -> index in result
    result: list[dict] = []
    for w in works:
        key = re.sub(r"[^a-z0-9]+", " ", (w.get("title") or "").lower()).strip()
        if not key:
            result.append(w)
            continue
        if key in seen:
            idx = seen[key]
            existing = result[idx]
            existing_concept_names = {c.get("display_name") for c in existing.get("concepts") or []}
            new_concepts = [
                c for c in (w.get("concepts") or []) if c.get("display_name") not in existing_concept_names
            ]
            if new_concepts:
                result[idx] = {**existing, "concepts": [*(existing.get("concepts") or []), *new_concepts]}
        else:
            seen[key] = len(result)
            result.append(w)
    return result


def concept_tokens(works: list[dict]) -> set[str]:
    """Union of tokenized concept display_names across a set of works."""
    tokens: set[str] = set()
    for w in works:
        for concept in w.get("concepts") or []:
            tokens |= tokenize(concept.get("display_name", ""))
    return tokens


def title_tokens(works: list[dict]) -> set[str]:
    """Union of tokenized titles across a set of works."""
    tokens: set[str] = set()
    for w in works:
        tokens |= tokenize(w.get("title", ""))
    return tokens
