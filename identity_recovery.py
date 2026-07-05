"""Classify a no-PID reviewer's raw DBLP-link field and resolve an identity.

Reviewers only reach this module after `dblp.parse_pid` has already had a
chance to resolve the link (bare PID, "/pod/" typo) — see reviewers.py. What's
left is heterogeneous: ORCID iDs, legacy/search DBLP URLs, Google Scholar
links, blank/junk fields, and — for a handful of rows — no name at all. This
module routes each to the right resolver and never guesses when a result is
ambiguous; ambiguous or unresolved cases come back flagged for manual review
instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import requests

import dblp
import openalex as oa
from reviewers import Reviewer

_ORCID_RE = re.compile(r"(\d{4}-\d{4}-\d{4}-\d{3}[\dXx])")
_SCHOLAR_HOSTS = ("scholar.google.", "share.google")
# Stops at '#' as well as '&': a URL fragment isn't sent to the server, so
# letting it leak into the captured query would silently truncate whatever
# this module appends after it (see resolve_via_author_search).
_SEARCH_QUERY_RE = re.compile(r"[?&]q=([^&#]+)")


@dataclass
class Resolution:
    # 'orcid' | 'dblp_legacy' | 'dblp_search' | 'openalex_name'
    # | 'manual_scholar_link' | 'manual_no_name' | 'lookup_error'
    path: str
    dblp_pid: str | None = None
    openalex_author: dict | None = None
    confidence: str | None = None  # MATCH / AMBIGUOUS / LOW_CONFIDENCE / NOT_FOUND (openalex paths only)
    candidates: list[dict] = field(default_factory=list)
    note: str = ""


def _extract_search_query(url: str) -> str | None:
    match = _SEARCH_QUERY_RE.search(url)
    return match.group(1) if match else None


def resolve(
    reviewer: Reviewer, session: requests.Session, openalex_cache: dict | None = None
) -> Resolution:
    """Resolve one no-PID reviewer to a DBLP PID, an OpenAlex author, or a
    manual-review flag. Assumes `reviewer.pid` is already None.

    A network failure (timeout, connection error, exhausted 429 retries) in
    any of the resolvers below propagates as requests.RequestException — that
    is caught here, once, and reported as its own 'lookup_error' path rather
    than being mistaken for a confident "not found" (see dblp.resolve_legacy_url
    / resolve_via_author_search / openalex.lookup_by_orcid's docstrings).
    """
    try:
        return _resolve(reviewer, session, openalex_cache)
    except requests.RequestException as exc:
        return Resolution(path="lookup_error", note=f"network error during resolution: {exc}")


def _resolve(
    reviewer: Reviewer, session: requests.Session, openalex_cache: dict | None
) -> Resolution:
    url = (reviewer.dblp_url or "").strip()
    has_name = bool(reviewer.first.strip() or reviewer.last.strip())

    orcid_match = _ORCID_RE.search(url)
    if orcid_match:
        orcid = orcid_match.group(1)
        author = oa.lookup_by_orcid(orcid, session, cache=openalex_cache)
        if author:
            return Resolution(path="orcid", openalex_author=author, confidence="MATCH")
        return Resolution(path="orcid", confidence="NOT_FOUND", note=f"ORCID {orcid} not found in OpenAlex")

    url_lower = url.lower()
    if "dblp.uni-trier.de/pers/hd/" in url_lower or "dblp.org/pers/hd/" in url_lower:
        pid = dblp.resolve_legacy_url(url, session)
        if pid:
            return Resolution(path="dblp_legacy", dblp_pid=pid)
    elif "dblp.org/search/author" in url_lower:
        # Deliberately narrower than a bare "dblp.org/search" substring: the
        # author-search API is only a safe shortcut when the link genuinely
        # targets author search. A generic /search or /search/publ link's
        # query text isn't reliably a name, so it falls through to the
        # affiliation-disambiguated OpenAlex path below instead of being
        # trusted on a single-hit author-search match.
        query = _extract_search_query(url)
        pid = dblp.resolve_via_author_search(query, session) if query else None
        if pid:
            return Resolution(path="dblp_search", dblp_pid=pid)
    elif any(host in url_lower for host in _SCHOLAR_HOSTS):
        return Resolution(path="manual_scholar_link", note=url)

    if not has_name:
        return Resolution(path="manual_no_name", note="no first/last name on the acceptance form")

    candidates = oa.search_by_name(reviewer.name, session, cache=openalex_cache)
    chosen, confidence, considered = oa.pick_candidate(candidates, reviewer.affiliation)
    return Resolution(path="openalex_name", openalex_author=chosen, confidence=confidence, candidates=considered)
