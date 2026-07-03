"""DBLP access: turn a reviewer's DBLP link into recent publication titles.

Pure DBLP logic with no CSV knowledge. DBLP publishes a per-person XML record at
``https://dblp.org/pid/{PID}.xml``; we extract the PID from whatever link shape
the reviewer supplied, fetch that record, and return titles within a year window.

Cache lookup order (all read before hitting the network):
  1. colleague's read-only cache (dblp_pubs_cache.json) — richer, uncapped
  2. our write cache (dblp_cache.json) — built incrementally by this script
  3. live DBLP fetch — stores ALL publications (uncapped) in the write cache
"""

from __future__ import annotations

import datetime
import json
import random
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable

import requests

# Matches the PID inside a DBLP person URL. A PID is two path segments,
# "<prefix>/<name>", in one of two DBLP formats:
#   numeric:  164/7945, 04/2187.html, 81/6765-2.html   (homonym ids, may have -N)
#   named:    s/SmrutiRSarangi.html, m/OnurMutlu, g/AntonioGonzalez1
# The name segment ([\w-]+) stops naturally at '.', '&', '?', '#', so this also
# pulls the PID out of Google-redirect wrappers (...url=.../pid/285/5783&ved=...).
_PID_RE = re.compile(r"/pid/(\w+/[\w-]+)", re.IGNORECASE)

_USER_AGENT = (
    "HPCA2027-reviewer-match/0.1 (PC reviewer-paper matching; contact PC chairs)"
)


# ---------------------------------------------------------------------------
# PID extraction
# ---------------------------------------------------------------------------

def parse_pid(url: str) -> str | None:
    """Extract a canonical DBLP PID from a reviewer's link.

    Returns None for missing links ("", "none"), personal homepages, and DBLP
    *search* URLs — anything that is not a direct /pid/ record.
    """
    if not url:
        return None
    url = url.strip()
    if url.lower() == "none" or "/pid/" not in url.lower():
        return None
    match = _PID_RE.search(url)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def load_cache(path: str) -> dict:
    """Load our writable title cache from disk (returns {} if not found).

    Format: {pid: [[year, title], ...], ...}
    """
    p = Path(path)
    if p.exists():
        with p.open(encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict, path: str) -> None:
    """Write the writable title cache to disk atomically."""
    p = Path(path)
    tmp = p.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    tmp.replace(p)


def load_colleague_cache(path: str) -> dict:
    """Load and normalise the colleague's richer cache to our [[year, title]] format.

    The colleague's format is {pid: [{title, year, venue, type}, ...]}. Some
    entries may lack 'year' or 'title' — those publications are skipped.
    Returns {} if the file is not found.
    """
    p = Path(path)
    if not p.exists():
        return {}
    with p.open(encoding="utf-8") as f:
        raw = json.load(f)

    normalised: dict[str, list] = {}
    for pid, pubs in raw.items():
        entries = []
        for pub in pubs:
            title = (pub.get("title") or "").strip()
            year_raw = pub.get("year")
            if not title or year_raw is None:
                continue
            try:
                year = int(year_raw)
            except (ValueError, TypeError):
                continue
            entries.append([year, title])
        entries.sort(key=lambda yt: yt[0], reverse=True)
        normalised[pid] = entries
    return normalised


# ---------------------------------------------------------------------------
# Title utilities
# ---------------------------------------------------------------------------

def dedup_titles(pubs: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Remove duplicate titles (DBLP lists preprint + conference versions).

    Normalises titles to lowercase with trailing punctuation stripped for
    comparison. When duplicates exist, keeps the entry with the highest year.
    Output order is preserved (year-descending assumed on input).
    """
    seen: dict[str, int] = {}  # normalised_title -> index in result
    result: list[tuple[int, str]] = []
    for year, title in pubs:
        key = title.lower().rstrip(". ")
        if key in seen:
            existing_idx = seen[key]
            if year > result[existing_idx][0]:
                result[existing_idx] = (year, title)
        else:
            seen[key] = len(result)
            result.append((year, title))
    return result


def filter_by_years(
    pubs: list, years: int, current_year: int | None = None
) -> list[tuple[int, str]]:
    """Return publications from the most recent `years` calendar years.

    `pubs` may be [[year, title], ...] (from JSON cache) or [(year, title), ...].
    """
    if current_year is None:
        current_year = datetime.date.today().year
    cutoff = current_year - years + 1
    return [(int(y), t) for y, t in pubs if int(y) >= cutoff]


# ---------------------------------------------------------------------------
# Network fetch
# ---------------------------------------------------------------------------

def _get_with_retry(
    session: requests.Session, url: str, *, max_retries: int = 5, timeout: int = 30
) -> requests.Response:
    """GET `url`, honoring DBLP's 429 rate-limit responses with aggressive backoff.

    DBLP will block an IP temporarily if hit too fast. On 429 we wait at least
    15 s (or whatever Retry-After says), doubling each attempt.
    """
    for attempt in range(max_retries):
        resp = session.get(url, timeout=timeout, headers={"User-Agent": _USER_AGENT})
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            wait = int(retry_after) if retry_after and retry_after.isdigit() else max(15, 15 * (2 ** attempt))
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp
    resp.raise_for_status()
    return resp


def _title_text(pub: ET.Element) -> str | None:
    """Full title text of a publication element, flattening nested markup."""
    title_el = pub.find("title")
    if title_el is None:
        return None
    text = "".join(title_el.itertext()).strip()
    return text or None


def _fetch_all_from_dblp(
    pid: str, session: requests.Session
) -> list[tuple[int, str]]:
    """Fetch ALL (year, title) pairs for a PID directly from DBLP XML."""
    resp = _get_with_retry(session, f"https://dblp.org/pid/{pid}.xml")
    root = ET.fromstring(resp.content)

    pubs: list[tuple[int, str]] = []
    for record in root.findall("r"):
        for pub in record:
            if pub.tag == "www":
                continue
            title = _title_text(pub)
            year_el = pub.find("year")
            if title is None or year_el is None or not (year_el.text or "").strip():
                continue
            try:
                year = int(year_el.text.strip())
            except ValueError:
                continue
            pubs.append((year, title))

    pubs.sort(key=lambda yt: yt[0], reverse=True)
    return pubs


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def fetch_titles(
    pid: str,
    years: int = 4,
    session: requests.Session | None = None,
    write_cache: dict | None = None,
    readonly_cache: dict | None = None,
) -> tuple[list[tuple[int, str]], str]:
    """Return (year, title) pairs for publications within the last `years` years.

    Lookup order:
      1. readonly_cache (colleague's pre-built cache) — uncapped, checked first
      2. write_cache (our incrementally built cache) — may be 10-capped for old entries
      3. Live DBLP fetch — stores ALL publications into write_cache (uncapped)

    Returns (filtered_pubs, source) where source is 'colleague', 'cache', or 'live'.
    Titles are deduplicated before year-filtering.
    """
    # 1. Colleague's read-only cache
    if readonly_cache is not None and pid in readonly_cache:
        all_pubs = [(int(y), t) for y, t in readonly_cache[pid]]
        return filter_by_years(dedup_titles(all_pubs), years), "colleague"

    # 2. Our write cache
    if write_cache is not None and pid in write_cache:
        all_pubs = [(int(y), t) for y, t in write_cache[pid]]
        return filter_by_years(dedup_titles(all_pubs), years), "cache"

    # 3. Live fetch
    session = session or requests.Session()
    all_pubs = _fetch_all_from_dblp(pid, session)
    if write_cache is not None:
        write_cache[pid] = all_pubs  # store uncapped
    return filter_by_years(dedup_titles(all_pubs), years), "live"


def fetch_titles_for_pids(
    pids: list[str],
    *,
    years: int = 4,
    session: requests.Session | None = None,
    write_cache: dict | None = None,
    readonly_cache: dict | None = None,
    cache_path: str | None = None,
    delay: float = 3.0,
    on_result: Callable[[str, list[tuple[int, str]], str], None] | None = None,
    on_error: Callable[[str, Exception], None] | None = None,
) -> dict[str, tuple[list[tuple[int, str]], str]]:
    """Fetch titles for each PID in turn, applying `fetch_titles`'s cache order.

    Jitters `delay` seconds between consecutive *live* DBLP fetches (no delay
    when a fetch was served from cache) and persists `write_cache` to
    `cache_path` after every live fetch, so progress survives an interruption.

    `on_result(pid, titles, source)` and `on_error(pid, exc)` are optional
    hooks for the caller's own progress reporting — this function has no
    knowledge of reviewers, only PIDs. Returns `{pid: (titles, source)}` for
    the PIDs that succeeded; failed PIDs are omitted.
    """
    session = session or requests.Session()
    results: dict[str, tuple[list[tuple[int, str]], str]] = {}
    last_was_live = False

    for pid in pids:
        if last_was_live and delay:
            jitter = random.uniform(-0.5, 0.5) * delay
            time.sleep(max(0.5, delay + jitter))

        try:
            titles, source = fetch_titles(
                pid,
                years=years,
                session=session,
                write_cache=write_cache,
                readonly_cache=readonly_cache,
            )
        except Exception as exc:  # noqa: BLE001
            last_was_live = False
            if on_error is not None:
                on_error(pid, exc)
            continue

        last_was_live = source == "live"
        if last_was_live and write_cache is not None and cache_path is not None:
            save_cache(write_cache, cache_path)

        results[pid] = (titles, source)
        if on_result is not None:
            on_result(pid, titles, source)

    return results
