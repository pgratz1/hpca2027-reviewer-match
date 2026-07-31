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
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
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

# A bare PID with no /pid/ wrapper at all, e.g. a reviewer pasting "241/3024"
# directly. Anchored to the whole (stripped) string so it doesn't misfire on
# a URL path fragment that merely contains something PID-shaped.
_BARE_PID_RE = re.compile(r"^(\w+/[\w-]+?)(\.html)?$", re.IGNORECASE)

_USER_AGENT = (
    "HPCA2027-reviewer-match/0.1 (PC reviewer-paper matching; contact PC chairs)"
)

_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s?#]+", re.IGNORECASE)

# Ceiling on any single wait, including one a server asked for via Retry-After.
# Past this the run is not making progress and should say so rather than sleep.
MAX_BACKOFF = 300.0

# Server-side statuses that mean "not now" rather than "never". DBLP answers 503
# when it is shedding load, which is the same signal as a 429 and has to back
# off the same way — raising immediately instead burns a whole batch in seconds
# without ever pausing, which is the opposite of what the server asked for.
RETRY_STATUSES = frozenset({500, 502, 503, 504})

# Consecutive live-fetch failures after which a batch loop gives up. DBLP drops
# connections when it has blocked an IP, and no amount of retrying clears that
# — only waiting does. Stopping keeps the partial cache and lets a later run
# resume, instead of grinding for hours and filling the roster with reviewers
# who look publication-less.
MAX_CONSECUTIVE_FAILURES = 5


def _backoff_delay(floor: float, attempt: int) -> float:
    """Exponential backoff with jitter, capped at MAX_BACKOFF.

    `attempt` is always the index within a single call, so the sequence starts
    over for each new request rather than ratcheting upward across a run.
    """
    delay = floor * (2 ** attempt) * random.uniform(0.75, 1.25)
    # Cap after jittering, not before, or the jitter reintroduces the overshoot
    # the cap exists to prevent.
    return min(max(floor, delay), MAX_BACKOFF)

# A submission's dblp field lists one entry per author. Its entries are
# separated by ';', ',', or a newline — and often by ',\n' or ';\n' together,
# which is ONE separator, not two. Collapsing only the newline-bearing runs
# keeps a bare ';;' meaning "this author has no page", which is what makes the
# positional alignment hold: matching the author count for 1,138 of the 1,354
# papers that have the field, against 970 if ',\n' is read as two separators.
# No DBLP URL or bare PID contains any of these characters, so one rule covers
# every observed shape.
_DBLP_SPLIT_RE = re.compile(r"[ \t]*(?:[;,][ \t]*\n|\n|[;,])[ \t]*")

# Placeholders authors type for "this author has no DBLP page". Filtered before
# parse_pid, whose bare-PID branch would happily read "N/A" as the PID "N/A".
_DBLP_PLACEHOLDERS = {"", "-", "--", "n/a", "na", "none", "null", "no", "nil", "?"}

# A trailing all-digit token is DBLP's homonym discriminator ("Yang Wang 0089"),
# not part of the name.
_HOMONYM_SUFFIX_RE = re.compile(r"\s+\d+$")


# ---------------------------------------------------------------------------
# PID extraction
# ---------------------------------------------------------------------------

def parse_pid(url: str) -> str | None:
    """Extract a canonical DBLP PID from a reviewer's link.

    Returns None for missing links ("", "none"), personal homepages, and DBLP
    *search* URLs — anything that is not a direct /pid/ record. Tolerates two
    reviewer typos seen in the acceptance form: a bare PID pasted with no URL
    wrapper ("241/3024"), and "/pod/" in place of "/pid/" — both verified live
    against dblp.org (dblp.org/pid/241/3024.xml and .../170/0138.xml both
    resolve).
    """
    if not url:
        return None
    url = url.strip()
    if url.lower() == "none":
        return None
    if "/pid/" not in url.lower():
        if "/pod/" in url.lower():
            url = re.sub(r"/pod/", "/pid/", url, flags=re.IGNORECASE)
        else:
            bare = _BARE_PID_RE.match(url)
            return bare.group(1) if bare else None
    match = _PID_RE.search(url)
    return match.group(1) if match else None


def name_tokens(name: str) -> frozenset[str]:
    """Comparable token set for a person's name.

    Accent-folded and lowercased, with DBLP's homonym suffix and bare initials
    dropped, so "Matthew D. Sinclair" matches "Matthew Sinclair" and a name
    written family-first matches the same name written given-first (the common
    case for CJK names in author lists). Returns a set, so token *order* never
    matters — deliberate, since that ordering is exactly what varies.
    """
    name = _HOMONYM_SUFFIX_RE.sub("", name.strip())
    folded = unicodedata.normalize("NFKD", name)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    tokens = re.findall(r"\w+", folded.lower())
    return frozenset(t for t in tokens if len(t) > 1)


def split_dblp_field(raw: str | None) -> list[str]:
    """Split a paper's `dblp` field into one entry per author, placeholders kept.

    Placeholder entries become "" rather than being dropped, so the list stays
    positionally aligned with the author list — dropping them is what breaks
    the alignment.
    """
    if not raw:
        return []
    entries = [e.strip() for e in _DBLP_SPLIT_RE.split(raw.strip())]
    # A trailing delimiter yields one empty tail entry that is punctuation, not
    # an author; anything else empty is a real "no DBLP page" slot.
    while entries and entries[-1] == "":
        entries.pop()
    return ["" if e.lower() in _DBLP_PLACEHOLDERS else e for e in entries]


def index_self_declared_pids(papers: list[dict]) -> dict[str, Counter]:
    """{author email: Counter(pid)} read off the submissions' own `dblp` fields.

    The field is a delimited list positionally aligned with the author list, so
    it is only read when its length matches the author count: about one paper
    in six lists a different number of entries than authors, and indexing into
    those returns a co-author's page.
    """
    index: dict[str, Counter] = defaultdict(Counter)
    for paper in papers:
        authors = paper.get("authors") or []
        entries = split_dblp_field(paper.get("dblp"))
        if len(entries) != len(authors):
            continue
        for author, entry in zip(authors, entries):
            email = (author.get("email") or "").strip().lower()
            pid = parse_pid(entry)
            if email and pid:
                index[email][pid] += 1
    return index


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


def load_rich_cache(path: str) -> dict:
    """Load a rich publication cache verbatim (returns {} if not found).

    Format: {pid: [{title, year, venue, type}, ...], ...} — both the
    colleague's read-only cache and our writable venue cache use it. No
    normalisation happens here: some colleague-cache entries are
    person-shaped or lack fields, so callers must read records with .get().
    """
    p = Path(path)
    if p.exists():
        with p.open(encoding="utf-8") as f:
            return json.load(f)
    return {}


def snapshot_gaps(snapshot: dict, *already_cached: dict) -> dict:
    """The snapshot entries for PIDs none of the existing caches has.

    The snapshot is a fallback, not a replacement. A PID whose publications are
    already cached keeps them, because a fingerprint's cache key includes its
    publication list: re-sourcing an already-cached person would invalidate
    their fingerprint and re-embed them for no benefit. Restricting the merge to
    genuine gaps is what lets the existing artifacts stay byte-identical.
    """
    return {
        pid: pubs for pid, pubs in snapshot.items()
        if not any(pid in cache for cache in already_cached)
    }


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


def windowed_with_fallback(
    items_desc: list,
    year_of: Callable[[object], int],
    years: int,
    current_year: int | None = None,
    min_count: int = 10,
    fallback_count: int = 10,
) -> list:
    """Items whose `year_of(item)` falls in the last `years` calendar years,
    falling back to the `fallback_count` most-recent items overall when that
    window has fewer than `min_count`. `items_desc` must already be sorted
    year-descending.

    A reviewer who's gone quiet recently but has a real publication history
    (e.g. moved to industry after a PhD) otherwise gets an almost-empty
    profile; this gives them a usable one instead. `year_of` keeps the policy
    reusable across item shapes (see `select_recent` for the title tuples).
    """
    if current_year is None:
        current_year = datetime.date.today().year
    cutoff = current_year - years + 1
    windowed = [it for it in items_desc if year_of(it) >= cutoff]
    if len(windowed) < min_count:
        return items_desc[:fallback_count]
    return windowed


def select_recent(
    deduped_desc: list[tuple[int, str]],
    years: int,
    current_year: int | None = None,
    min_count: int = 10,
    fallback_count: int = 10,
) -> list[tuple[int, str]]:
    """Titles from the last `years` years, falling back to the `fallback_count`
    most recent titles overall when that window has fewer than `min_count`.
    See `windowed_with_fallback`. `deduped_desc` must already be deduplicated
    (dedup_titles's output shape).
    """
    return windowed_with_fallback(
        deduped_desc, lambda yt: int(yt[0]), years, current_year, min_count, fallback_count
    )


# ---------------------------------------------------------------------------
# Network fetch
# ---------------------------------------------------------------------------

def get_with_retry(
    session: requests.Session,
    url: str,
    params: dict | None = None,
    *,
    headers: dict | None = None,
    max_retries: int = 5,
    timeout: int = 30,
    backoff_floor: float = 15,
) -> requests.Response:
    """GET `url`, retrying on a 429 or a dropped connection with backoff.

    Waits at least `backoff_floor` seconds (or whatever Retry-After says),
    doubling each attempt. DBLP needs an aggressive floor or it'll block the
    IP temporarily; `backoff_floor` exists so a gentler API can pass a much
    shorter one at the call site.

    The doubling restarts at every call — `attempt` is local — so a slow
    endpoint does not ratchet the delay up across the run. Two things bound it
    anyway: `Retry-After` is honoured but capped at `MAX_BACKOFF`, since a
    server is free to name an hour and a silent hour is indistinguishable from
    a hang, and every wait is announced on stderr for the same reason.

    A connection *reset* is reported separately from a timeout. DBLP drops the
    connection outright when it has blocked the caller's IP, and that is worth
    telling apart from a slow network: retrying through it cannot succeed, and
    the loop callers use it to stop early rather than grind.
    """
    for attempt in range(max_retries):
        try:
            resp = session.get(url, params=params, timeout=timeout, headers=headers)
        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt == max_retries - 1:
                raise
            wait = _backoff_delay(backoff_floor, attempt)
            print(
                f"    connection failed ({type(exc).__name__}); retrying in "
                f"{wait:.0f}s (attempt {attempt + 1}/{max_retries})",
                file=sys.stderr,
            )
            time.sleep(wait)
            continue
        if resp.status_code == 429 or resp.status_code in RETRY_STATUSES:
            retry_after = resp.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                wait = min(int(retry_after), MAX_BACKOFF)
            else:
                wait = _backoff_delay(backoff_floor, attempt)
            if attempt == max_retries - 1:
                break
            label = "rate-limited" if resp.status_code == 429 else "server busy"
            print(
                f"    {label} ({resp.status_code}); waiting {wait:.0f}s "
                f"(attempt {attempt + 1}/{max_retries})",
                file=sys.stderr,
            )
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


def normalise_doi(value: str | None) -> str | None:
    """Return a lowercase bare DOI extracted from a DOI or publisher URL."""
    if not value:
        return None
    match = _DOI_RE.search(value.strip())
    if not match:
        return None
    return match.group(0).rstrip(".>,)").lower()


def _publication_doi(pub: ET.Element) -> str | None:
    """Extract the first DOI encoded in a DBLP publication record."""
    for ee in pub.findall("ee"):
        doi = normalise_doi("".join(ee.itertext()))
        if doi:
            return doi
    for note in pub.findall("note"):
        if "doi" in (note.get("type") or "").split():
            doi = normalise_doi("".join(note.itertext()))
            if doi:
                return doi
    return None


def _fetch_all_from_dblp(
    pid: str, session: requests.Session
) -> list[tuple[int, str]]:
    """Fetch ALL (year, title) pairs for a PID directly from DBLP XML."""
    resp = get_with_retry(session, f"https://dblp.org/pid/{pid}.xml", headers={"User-Agent": _USER_AGENT})
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


def _fetch_all_records_from_dblp(
    pid: str, session: requests.Session, *, max_retries: int = 5,
    backoff_floor: float = 15, base_url: str = "https://dblp.org",
) -> list[dict]:
    """Fetch ALL publication records for a PID, keeping venue and record type.

    Rich analogue of _fetch_all_from_dblp for callers that need to know
    *where* something was published (e.g. seniority classification). Each
    record is {"title", "year": int, "venue", "type", "doi"} — matching the
    colleague-cache format — where type is the DBLP XML element tag
    (inproceedings/article/proceedings/...) and venue is the <booktitle> for
    inproceedings, the <journal> for articles, and "" otherwise. Same skip
    rules as _fetch_all_from_dblp (www records, missing title/year).
    Sorted year-descending.
    """
    resp = get_with_retry(
        session, f"{base_url}/pid/{pid}.xml", headers={"User-Agent": _USER_AGENT},
        max_retries=max_retries, backoff_floor=backoff_floor,
    )
    root = ET.fromstring(resp.content)

    records: list[dict] = []
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
            venue_el = pub.find("booktitle")
            if venue_el is None:
                venue_el = pub.find("journal")
            venue = (venue_el.text or "").strip() if venue_el is not None else ""
            records.append(
                {
                    "title": title, "year": year, "venue": venue,
                    "type": pub.tag, "doi": _publication_doi(pub) or "",
                }
            )

    records.sort(key=lambda r: r["year"], reverse=True)
    return records


def fetch_doi_records(
    pid: str, session: requests.Session | None = None
) -> list[dict]:
    """Fetch a PID's DBLP records live, including DOI identifiers.

    This deliberately bypasses the older title and venue caches: their
    schemas predate DOI retention, so treating them as authoritative would
    permanently hide identifiers for already-cached reviewers.
    """
    session = session or requests.Session()
    last_error: Exception | None = None
    for base_url in (
        "https://dblp.org",
        "https://dblp.uni-trier.de",
        "https://dblp.dagstuhl.de",
    ):
        try:
            return _fetch_all_records_from_dblp(
                pid, session, max_retries=2, backoff_floor=5, base_url=base_url,
            )
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def fetch_titles(
    pid: str,
    years: int = 4,
    session: requests.Session | None = None,
    write_cache: dict | None = None,
    readonly_cache: dict | None = None,
    fallback_when_thin: bool = False,
) -> tuple[list[tuple[int, str]], str]:
    """Return (year, title) pairs for publications within the last `years` years.

    Lookup order:
      1. readonly_cache (colleague's pre-built cache) — uncapped, checked first
      2. write_cache (our incrementally built cache) — may be 10-capped for old entries
      3. Live DBLP fetch — stores ALL publications into write_cache (uncapped)

    Returns (selected_pubs, source) where source is 'colleague', 'cache', or
    'live'. Titles are deduplicated before year-filtering.

    By default (fallback_when_thin=False) this is a strict year-window filter
    — a reviewer with nothing published in the window gets an empty list.
    build_fingerprints.py relies on that to detect when a reviewer needs its
    own area-profile-only fallback (see its module docstring), so this default
    must not change. Pass fallback_when_thin=True to instead backfill with the
    10 most recent titles overall when the window has fewer than 10 (see
    select_recent) — for a caller with no downstream fallback of its own, a
    thin-but-present profile is more useful than an empty one.
    """
    # 1. Colleague's read-only cache
    if readonly_cache is not None and pid in readonly_cache:
        all_pubs = [(int(y), t) for y, t in readonly_cache[pid]]
        source = "colleague"
    # 2. Our write cache
    elif write_cache is not None and pid in write_cache:
        all_pubs = [(int(y), t) for y, t in write_cache[pid]]
        source = "cache"
    # 3. Live fetch
    else:
        session = session or requests.Session()
        all_pubs = _fetch_all_from_dblp(pid, session)
        if write_cache is not None:
            write_cache[pid] = all_pubs  # store uncapped
        source = "live"

    deduped = dedup_titles(all_pubs)
    selected = select_recent(deduped, years) if fallback_when_thin else filter_by_years(deduped, years)
    return selected, source


def fetch_titles_for_pids(
    pids: list[str],
    *,
    years: int = 4,
    session: requests.Session | None = None,
    write_cache: dict | None = None,
    readonly_cache: dict | None = None,
    cache_path: str | None = None,
    delay: float = 3.0,
    fallback_when_thin: bool = False,
    on_result: Callable[[str, list[tuple[int, str]], str], None] | None = None,
    on_error: Callable[[str, Exception], None] | None = None,
) -> dict[str, tuple[list[tuple[int, str]], str]]:
    """Fetch titles for each PID in turn, applying `fetch_titles`'s cache order.

    Jitters `delay` seconds between consecutive *live* DBLP fetches (no delay
    when a fetch was served from cache) and persists `write_cache` to
    `cache_path` after every live fetch, so progress survives an interruption.
    `fallback_when_thin` is passed straight through to `fetch_titles` (see its
    docstring) — defaults to False so existing callers' behavior is unchanged.

    `on_result(pid, titles, source)` and `on_error(pid, exc)` are optional
    hooks for the caller's own progress reporting — this function has no
    knowledge of reviewers, only PIDs. Returns `{pid: (titles, source)}` for
    the PIDs that succeeded; failed PIDs are omitted.
    """
    session = session or requests.Session()
    results: dict[str, tuple[list[tuple[int, str]], str]] = {}
    last_was_live = False
    consecutive_failures = 0

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
                fallback_when_thin=fallback_when_thin,
            )
        except Exception as exc:  # noqa: BLE001
            # A failure means the last thing we did was hit the network and get
            # nowhere, so the next PID must still be paced. Clearing the flag
            # here — as this once did — sent the next request out with no delay
            # at exactly the moment the server had proved it wanted fewer.
            last_was_live = True
            consecutive_failures += 1
            if on_error is not None:
                on_error(pid, exc)
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(
                    f"{consecutive_failures} consecutive DBLP fetches failed "
                    f"(last: {pid}). DBLP blocks an IP that asks too fast, and "
                    f"retrying cannot clear that — only waiting can. "
                    f"{len(results)} PID(s) succeeded in this pass and "
                    f"{len(write_cache) if write_cache is not None else 0} are "
                    f"in the cache overall; re-run later to continue from there."
                ) from exc
            continue

        consecutive_failures = 0
        last_was_live = source == "live"
        if last_was_live and write_cache is not None and cache_path is not None:
            save_cache(write_cache, cache_path)

        results[pid] = (titles, source)
        if on_result is not None:
            on_result(pid, titles, source)

    return results


def fetch_records(
    pid: str,
    session: requests.Session | None = None,
    write_cache: dict | None = None,
    readonly_cache: dict | None = None,
) -> tuple[list[dict], str]:
    """Return all rich publication records for a PID (see load_rich_cache).

    Rich analogue of fetch_titles with the same cache order:
      1. readonly_cache (colleague's pre-built rich cache)
      2. write_cache (our incrementally built venue cache)
      3. Live DBLP fetch — stores ALL records into write_cache (uncapped)

    Returns (records, source) where source is 'colleague', 'cache', or
    'live'. No windowing or dedup here — callers count what they need.
    """
    if readonly_cache is not None and pid in readonly_cache:
        return readonly_cache[pid], "colleague"
    if write_cache is not None and pid in write_cache:
        return write_cache[pid], "cache"
    session = session or requests.Session()
    records = _fetch_all_records_from_dblp(pid, session)
    if write_cache is not None:
        write_cache[pid] = records
    return records, "live"


def fetch_records_for_pids(
    pids: list[str],
    *,
    session: requests.Session | None = None,
    write_cache: dict | None = None,
    readonly_cache: dict | None = None,
    cache_path: str | None = None,
    delay: float = 3.0,
    on_result: Callable[[str, list[dict], str], None] | None = None,
    on_error: Callable[[str, Exception], None] | None = None,
) -> dict[str, tuple[list[dict], str]]:
    """Fetch rich records for each PID in turn, applying fetch_records's cache order.

    Mirror of fetch_titles_for_pids: jitters `delay` seconds between
    consecutive *live* DBLP fetches (no delay when served from cache) and
    persists `write_cache` to `cache_path` after every live fetch, so
    progress survives an interruption. `on_result(pid, records, source)` and
    `on_error(pid, exc)` are optional progress hooks. Returns
    {pid: (records, source)} for the PIDs that succeeded; failed PIDs are
    omitted.
    """
    session = session or requests.Session()
    results: dict[str, tuple[list[dict], str]] = {}
    last_was_live = False
    consecutive_failures = 0

    for pid in pids:
        if last_was_live and delay:
            jitter = random.uniform(-0.5, 0.5) * delay
            time.sleep(max(0.5, delay + jitter))

        try:
            records, source = fetch_records(
                pid,
                session=session,
                write_cache=write_cache,
                readonly_cache=readonly_cache,
            )
        except Exception as exc:  # noqa: BLE001
            # Keep pacing after a failure; see fetch_titles_for_pids.
            last_was_live = True
            consecutive_failures += 1
            if on_error is not None:
                on_error(pid, exc)
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(
                    f"{consecutive_failures} consecutive DBLP fetches failed "
                    f"(last: {pid}). DBLP blocks an IP that asks too fast, and "
                    f"retrying cannot clear that — only waiting can. "
                    f"{len(results)} PID(s) succeeded in this pass and "
                    f"{len(write_cache) if write_cache is not None else 0} are "
                    f"in the cache overall; re-run later to continue from there."
                ) from exc
            continue

        consecutive_failures = 0
        last_was_live = source == "live"
        if last_was_live and write_cache is not None and cache_path is not None:
            save_cache(write_cache, cache_path)

        results[pid] = (records, source)
        if on_result is not None:
            on_result(pid, records, source)

    return results
