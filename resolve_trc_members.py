"""Fill in the Training Review Committee roster: DBLP identity + advisor HotCRP account.

The TRC is a cohort of PhD students reviewing alongside the PC. Two things are
missing from the roster before they can be matched to papers: each student's
DBLP page (the identity every fingerprint in this pipeline is built from) and
their advisor's HotCRP account, since a student inherits their advisor's
conflicts and must never be assigned an advisor's paper.

**Advisor -> HotCRP email** is resolved offline from two local sources, in
order: the PC acceptance form (whose first question *is* "confirm your HotCRP
email address", so it is authoritative even for those who declined) and the
author/contact lists of the submissions themselves. Where a name matches
several accounts, the roster's advisor-affiliation cell breaks the tie; where
it still doesn't, the row is left blank and reported rather than guessed --
"Rakesh Kumar" is two different researchers in this data.

**Student -> DBLP** is a name-matching problem that a name search alone cannot
solve: DBLP lists dozens of researchers called "Cheng Chen", and keeps an
undisambiguated bucket page of 700+ papers under that bare name. So three
independent routes propose candidate PIDs --

  1. *self-declared*: the student is an author on a submission, and that
     submission's `dblp` field (positionally aligned with its author list)
     names their page;
  2. *co-author*: the student appears in their advisor's DBLP record, which
     lists every co-author with their PID;
  3. *search*: DBLP's author search, which is fuzzy and proposes near-misses.

-- and every candidate is then verified against its own DBLP record: the name
must match, the page must not be one of DBLP's homonym buckets (implausibly
many publications for a student), and having the advisor as a co-author
confirms it. A candidate that two routes agree on and verification confirms is
reported as such; anything weaker is written but flagged, and anything
ambiguous is left blank for hand resolution. Nothing here is a guess that
isn't labelled as one.

Rows already carrying a value in either new column are left alone (hand-entered
identities win, as in dblp_overrides.csv) unless --overwrite is given.

    python resolve_trc_members.py
    python resolve_trc_members.py --out /tmp/trc-dry-run.csv   # leave the roster alone
    python resolve_trc_members.py --no-network                 # cached lookups only
    python resolve_trc_members.py --limit 5                    # try a few rows first
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import random
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import requests

from dblp import (
    _USER_AGENT,
    get_with_retry,
    load_cache,
    name_tokens,
    parse_pid,
    save_cache,
    split_dblp_field,
)

DEFAULT_TRC = "hpca2027-trc - hpca2027-trc.csv"
DEFAULT_CSV = "HPCA'27 PC Member Acceptance Form (Responses) - Form Responses 1.csv"
DEFAULT_DATA = "hpca2027-data.json"
DEFAULT_SENIORITY = "reviewer_seniority.csv"
DEFAULT_PROFILE_CACHE = "dblp_profile_cache.json"
DEFAULT_SEARCH_CACHE = "dblp_author_search_cache.json"
DEFAULT_DELAY = 3.0
DEFAULT_SEARCH_HITS = 100

# A DBLP page carrying more publications than any PhD student could have
# written is not that student: it is either a homonym bucket (DBLP's bare
# "Cheng Chen" holds 725 papers by many people) or an established researcher
# the matching picked up by mistake. Either way it must not be accepted
# silently.
DEFAULT_MAX_STUDENT_PUBS = 100

# When nothing corroborates a name — no joint publication with the advisor, no
# self-declared page — checking more than a handful of same-named DBLP people
# one by one buys nothing: none of them can be confirmed either.
DEFAULT_MAX_NAME_MATCHES = 8

# How alike two spellings of one name token must be ("maryam"/"mariam" scores
# 0.83). Only ever used to *propose* a match that co-authorship must then
# confirm, so it can afford to be generous.
NEAR_SPELLING_RATIO = 0.8

# Shortest email local part allowed to identify someone by itself. "hanjun" is
# a name; "kim" or "jp" is a surname or an initial that half a department
# shares.
MIN_ADDRESS_NAME_LENGTH = 5

DBLP_COLUMN = "DBLP"
ADVISOR_EMAIL_COLUMN = "Advisor HotCRP Email"

# "Xiaofei Liao (or Hai Jin)" — a co-advised student naming both.
_ADVISOR_SPLIT_RE = re.compile(r"\s*\(\s*or\s+|\)\s*|\s+or\s+|\s*/\s*", re.IGNORECASE)

# Titles people prefix to an affiliation cell ("Professor, National University
# of Singapore"), and the words shared by half the world's universities. Neither
# distinguishes one institution from another, so neither may vote in a tie-break.
_AFFILIATION_STOPWORDS = frozenset(
    """
    university universite universitat universidad univ college institute institut
    school department dept faculty laboratory lab center centre research academy
    academic technology technological science sciences engineering computer
    computing national state professor prof assistant associate phd student
    of the and for at in a
    """.split()
)


def affiliation_tokens(text: str) -> frozenset[str]:
    """Distinctive tokens of an affiliation, for tie-breaking only.

    Reuses dblp.name_tokens' accent folding, then drops the words that every
    institution shares, so "University of California San Diego" and "UCSD"
    both keep what identifies them ({california, san, diego} / {ucsd}) — the
    comparison is an overlap test, never an equality test.
    """
    return frozenset(t for t in name_tokens(text) if t not in _AFFILIATION_STOPWORDS)


def split_advisor_names(raw: str) -> list[str]:
    """Split an advisor cell into the individual people it names."""
    return [part.strip() for part in _ADVISOR_SPLIT_RE.split(raw or "") if part.strip()]


def author_name(author: dict) -> str:
    """One name string from a HotCRP author/contact record."""
    return f"{author.get('given_name') or ''} {author.get('family_name') or ''}".strip()


def same_name_respelt(one: str, other: str) -> bool:
    """Whether two name tokens are one name spelt two ways.

    Transliteration swaps letters inside a name — "Maryam"/"Mariam" — so the
    two spellings run the same length and start the same way. Both conditions
    are load-bearing: without them "Cheng"/"Heng" and "Jing"/"Jin" score just
    as well, and those are different people's names, not different spellings
    of one.
    """
    return (
        len(one) == len(other)
        and one[:1] == other[:1]
        and difflib.SequenceMatcher(None, one, other).ratio() >= NEAR_SPELLING_RATIO
    )


def spellings_of_one_name(a: frozenset[str], b: frozenset[str]) -> bool:
    """Whether two token sets are one person's name transliterated two ways.

    Pairs the tokens up, allowing a respelling per pair but requiring at least
    one token to match outright, so no two people are merged on a pile of
    approximations.
    """
    if len(a) != len(b):
        return False
    remaining = sorted(b)
    exact = 0
    for token in sorted(a):
        if token in remaining:
            remaining.remove(token)
            exact += 1
            continue
        close = [other for other in remaining if same_name_respelt(token, other)]
        if not close:
            return False
        remaining.remove(close[0])
    return exact >= 1


def tokens_compatible(a: frozenset[str], b: frozenset[str]) -> str | None:
    """How two name-token sets relate: 'exact', 'partial', or None.

    'partial' means the two are probably the same name written differently:
    either one set is a strict subset of the other sharing at least two tokens
    (a middle name present on one side only), or they are the same name
    transliterated two ways. Two shared tokens is the floor because a single
    shared family name is worth nothing, and a partial match never stands on
    its own — it has to be confirmed by co-authorship with the advisor.
    """
    if not a or not b:
        return None
    if a == b:
        return "exact"
    if (a < b or b < a) and len(a & b) >= 2:
        return "partial"
    if spellings_of_one_name(a, b):
        return "partial"
    return None


# ---------------------------------------------------------------------------
# Local indexes: HotCRP accounts and self-declared DBLP pages
# ---------------------------------------------------------------------------

def index_pc_form(path: str) -> tuple[dict[frozenset[str], dict[str, str]], list[tuple[str, str]]]:
    """Index the PC acceptance form by name. Returns (by name tokens, unnamed).

    Every respondent is indexed, including those who declined: the question
    asks them to confirm the address their HotCRP account already uses, which
    is exactly what a conflict needs, whether or not they joined the PC. Those
    who declined, though, leave the name and affiliation columns blank, so
    they come back separately as (email, affiliation) pairs for the address
    itself to be matched against a name.
    """
    index: dict[frozenset[str], dict[str, str]] = defaultdict(dict)
    unnamed: list[tuple[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            email = first = last = affiliation = ""
            for header, value in row.items():
                if not header:
                    continue
                header = header.lower()
                value = (value or "").strip()
                if "hotcrp email" in header:
                    email = value.lower()
                elif header.startswith("first name"):
                    first = value
                elif header.startswith("last name"):
                    last = value
                elif "institutional affiliation" in header:
                    affiliation = value
            if not email:
                continue
            tokens = name_tokens(f"{first} {last}".strip())
            if tokens:
                index[tokens][email] = affiliation
            else:
                unnamed.append((email, affiliation))
    return index, unnamed


def match_by_address(
    name: str, affiliation: str, unnamed: list[tuple[str, str]]
) -> str | None:
    """The one PC-form address that names this person, if exactly one does.

    Last resort for a respondent who declined and so filled in nothing but
    their address: "hanjun@yonsei.ac.kr" against "Hanjun Kim" of Yonsei
    University. Two independent things have to line up — the local part has to
    be part of their name, and the domain has to be their institution — and
    the local part has to be long enough to be a name rather than an initial
    or a bare surname.
    """
    wanted_name = name_tokens(name)
    wanted_affiliation = affiliation_tokens(affiliation)
    if not wanted_name or not wanted_affiliation:
        return None

    hits = []
    for email, _ in unnamed:
        local, _, domain = email.partition("@")
        local_tokens = name_tokens(local)
        if not local_tokens or not local_tokens <= wanted_name:
            continue
        if sum(len(t) for t in local_tokens) < MIN_ADDRESS_NAME_LENGTH:
            continue
        if affiliation_tokens(domain.replace(".", " ")) & wanted_affiliation:
            hits.append(email)
    return hits[0] if len(hits) == 1 else None


def index_hotcrp_authors(papers: list[dict]) -> dict[frozenset[str], dict[str, Counter]]:
    """{name tokens: {email: Counter(affiliation)}} over every author and contact.

    Anyone who appears here has a HotCRP account for this conference — that is
    how they got onto a submission — which makes the author list the fallback
    for advisors who were never invited to the PC.
    """
    index: dict[frozenset[str], dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for paper in papers:
        for person in (paper.get("authors") or []) + (paper.get("contacts") or []):
            email = (person.get("email") or "").strip().lower()
            tokens = name_tokens(author_name(person))
            if not email or not tokens:
                continue
            index[tokens][email][(person.get("affiliation") or "").strip()] += 1
    return index


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


def pick_by_affiliation(
    candidates: dict[str, frozenset[str]], wanted: frozenset[str]
) -> str | None:
    """The one candidate email whose affiliation overlaps `wanted`, if unique."""
    if not wanted:
        return None
    hits = [email for email, tokens in candidates.items() if tokens & wanted]
    return hits[0] if len(hits) == 1 else None


def resolve_advisor_email(
    name: str,
    affiliation: str,
    pc_index: dict[frozenset[str], dict[str, str]],
    author_index: dict[frozenset[str], dict[str, Counter]],
    unnamed_pc: list[tuple[str, str]] | None = None,
) -> tuple[str | None, str]:
    """Resolve one advisor name to a HotCRP email. Returns (email, resolution)."""
    tokens = name_tokens(name)
    if not tokens:
        return None, "no-name"
    wanted = affiliation_tokens(affiliation)

    pc_hits = pc_index.get(tokens, {})
    if len(pc_hits) == 1:
        return next(iter(pc_hits)), "pc-form"
    if len(pc_hits) > 1:
        picked = pick_by_affiliation(
            {e: affiliation_tokens(a) for e, a in pc_hits.items()}, wanted
        )
        if picked:
            return picked, "pc-form+affiliation"
        return None, f"ambiguous-pc-form({len(pc_hits)})"

    author_hits = author_index.get(tokens, {})
    if not author_hits:
        by_address = match_by_address(name, affiliation, unnamed_pc or [])
        if by_address:
            return by_address, "pc-form-address"
        return None, "not-found"
    if len(author_hits) == 1:
        return next(iter(author_hits)), "hotcrp-author"

    # Several accounts share the name. Either they are the same person with two
    # addresses (same institution) or two researchers (different ones); only the
    # affiliation cell can tell those apart.
    affiliations = {
        email: affiliation_tokens(" ".join(counter)) for email, counter in author_hits.items()
    }
    picked = pick_by_affiliation(affiliations, wanted)
    if picked:
        return picked, "hotcrp-author+affiliation"
    matching = [e for e, tok in affiliations.items() if tok & wanted]
    if not matching:
        return None, f"ambiguous-hotcrp-author({len(author_hits)})"
    # Same institution, several addresses: the one used on the most submissions
    # is the account they actually work from.
    ranked = sorted(matching, key=lambda e: sum(author_hits[e].values()), reverse=True)
    if len(ranked) == 1 or sum(author_hits[ranked[0]].values()) > sum(author_hits[ranked[1]].values()):
        return ranked[0], "hotcrp-author+most-used"
    return None, f"ambiguous-hotcrp-author({len(author_hits)})"


def load_pc_pids(path: str) -> dict[str, str]:
    """{email: pid} for PC members already resolved by classify_reviewers.py."""
    pids: dict[str, str] = {}
    if not Path(path).exists():
        return pids
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            email = (row.get("email") or "").strip().lower()
            pid = (row.get("pid") or "").strip()
            if email and pid:
                pids[email] = pid
    return pids


# ---------------------------------------------------------------------------
# DBLP (network)
# ---------------------------------------------------------------------------

class Dblp:
    """Cached, rate-limited access to the two DBLP endpoints this script needs.

    Both caches are keyed by exactly what was asked for (a PID, a query string)
    and written atomically after every live fetch, so an interrupted run keeps
    its progress — DBLP fetches are the expensive part of this pipeline.
    """

    def __init__(
        self, profile_path: str, search_path: str, *, delay: float = DEFAULT_DELAY,
        offline: bool = False, hits: int = DEFAULT_SEARCH_HITS,
    ) -> None:
        self.profiles = load_cache(profile_path)
        self.searches = load_cache(search_path)
        self.profile_path = profile_path
        self.search_path = search_path
        self.delay = delay
        self.offline = offline
        self.hits = hits
        self.session = requests.Session()
        self.fetches = 0
        self._last_was_live = False

    def _pace(self) -> None:
        if self._last_was_live and self.delay:
            jitter = random.uniform(-0.5, 0.5) * self.delay
            time.sleep(max(0.5, self.delay + jitter))
        self._last_was_live = True
        self.fetches += 1

    def profile(self, pid: str) -> dict | None:
        """{names, affiliations, coauthors: {pid: name}, pubs} for a DBLP PID."""
        if pid in self.profiles:
            self._last_was_live = False
            return self.profiles[pid]
        if self.offline:
            return None
        self._pace()
        try:
            resp = get_with_retry(
                self.session, f"https://dblp.org/pid/{pid}.xml",
                headers={"User-Agent": _USER_AGENT},
            )
        except requests.HTTPError as exc:
            print(f"    DBLP profile {pid} failed: {exc}", file=sys.stderr)
            return None
        self.profiles[pid] = parse_profile(resp.content)
        save_cache(self.profiles, self.profile_path)
        return self.profiles[pid]

    def search(self, query: str) -> list[dict] | None:
        """DBLP author search: [{name, pid, aliases, affiliations}] for a name."""
        if query in self.searches:
            self._last_was_live = False
            return self.searches[query]
        if self.offline:
            return None
        self._pace()
        try:
            resp = get_with_retry(
                self.session, "https://dblp.org/search/author/api",
                params={"q": query, "format": "json", "h": self.hits},
                headers={"User-Agent": _USER_AGENT},
            )
        except requests.HTTPError as exc:
            print(f"    DBLP search {query!r} failed: {exc}", file=sys.stderr)
            return None
        self.searches[query] = parse_search(resp.json())
        save_cache(self.searches, self.search_path)
        return self.searches[query]


def _as_list(value) -> list:
    """DBLP's JSON collapses one-element lists to the bare element."""
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def parse_profile(xml: bytes) -> dict:
    """Parse a DBLP person record into names, affiliations, co-authors, pub count.

    Co-authors come from the publication list, keyed by PID: the PID is the
    identity, and the same person may be written several ways across papers.
    Homepage records (`www`) are skipped, as everywhere else in this project.
    """
    root = ET.fromstring(xml)
    names: list[str] = []
    canonical = (root.get("name") or "").strip()
    if canonical:
        names.append(canonical)

    affiliations: list[str] = []
    person = root.find("person")
    if person is not None:
        for author in person.findall("author"):
            name = "".join(author.itertext()).strip()
            if name:
                names.append(name)
        for note in person.findall("note"):
            if note.get("type") == "affiliation":
                text = "".join(note.itertext()).strip()
                if text:
                    affiliations.append(text)

    coauthors: dict[str, str] = {}
    pubs = 0
    for record in root.findall("r"):
        for pub in record:
            if pub.tag == "www":
                continue
            pubs += 1
            for author in pub.findall("author"):
                pid = (author.get("pid") or "").strip()
                name = "".join(author.itertext()).strip()
                if pid and name:
                    coauthors[pid] = name

    seen: set[str] = set()
    return {
        "names": [n for n in names if not (n in seen or seen.add(n))],
        "affiliations": affiliations,
        "coauthors": coauthors,
        "pubs": pubs,
    }


def parse_search(payload: dict) -> list[dict]:
    """Flatten DBLP's author-search JSON into [{name, pid, aliases, affiliations}]."""
    hits = _as_list(((payload.get("result") or {}).get("hits") or {}).get("hit"))
    results: list[dict] = []
    for hit in hits:
        info = hit.get("info") or {}
        pid = parse_pid(info.get("url") or "")
        if not pid:
            continue
        aliases = _as_list((info.get("aliases") or {}).get("alias"))
        affiliations = [
            (note.get("text") or "").strip()
            for note in _as_list((info.get("notes") or {}).get("note"))
            if isinstance(note, dict) and note.get("@type") == "affiliation"
        ]
        results.append({
            "name": (info.get("author") or "").strip(),
            "pid": pid,
            "aliases": [a for a in aliases if isinstance(a, str)],
            "affiliations": [a for a in affiliations if a],
        })
    return results


def name_relation(names, wanted: frozenset[str]) -> str | None:
    """Best relation between any spelling of a person's name and `wanted`."""
    relations = {tokens_compatible(name_tokens(n), wanted) for n in names}
    return "exact" if "exact" in relations else ("partial" if "partial" in relations else None)


def search_candidates(dblp: Dblp, name: str) -> list[dict]:
    """DBLP author-search hits for a name, keeping only plausible name matches.

    The endpoint is a similarity search, not a lookup — asking it for "Cheng
    Chen" volunteers "Fu-Chen Cheng" and "Cheng-Zhong Xu" — so every hit is
    re-checked against the name asked for. A second query with camel-cased runs
    split apart ("SeyyedAghaeiRezaei" -> "Seyyed Aghaei Rezaei") runs only when
    the name as written matched nothing, which is how transliterated names are
    often typed into a roster but not into DBLP.
    """
    wanted = name_tokens(name)
    queries = [name]
    split = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
    if split != name:
        queries.append(split)

    for query in queries:
        hits = dblp.search(query)
        if hits is None:
            return []
        matched = []
        for hit in hits:
            relation = name_relation([hit["name"], *hit["aliases"]], wanted)
            if relation:
                matched.append({**hit, "relation": relation})
        if matched:
            exact = [hit for hit in matched if hit["relation"] == "exact"]
            return exact or matched
    return []


@dataclass
class PidCandidate:
    """One proposed DBLP identity for a student, and what is known about it."""

    pid: str
    routes: set[str] = field(default_factory=set)
    relation: str | None = None       # how its DBLP name compares to the roster's
    advisor_coauthor: bool = False    # the advisor is a co-author on its record
    pubs: int | None = None           # publication count, for the bucket check
    dblp_name: str = ""
    affiliations: list[str] = field(default_factory=list)
    fetched: bool = False

    def describe(self) -> str:
        routes = "+".join(sorted(self.routes))
        marks = [routes, self.dblp_name or "?", self.relation or "name mismatch"]
        if self.advisor_coauthor:
            marks.append("advisor-coauthor")
        if self.pubs is not None:
            marks.append(f"{self.pubs} pubs")
        return f"{self.pid} [{', '.join(marks)}]"


def resolve_advisor_pids(
    dblp: Dblp,
    advisors: list[tuple[str, str | None]],
    affiliation: str,
    pc_pids: dict[str, str],
    self_declared: dict[str, Counter],
) -> list[str]:
    """DBLP PIDs for a student's advisor(s), cheapest reliable source first.

    Takes (name, hotcrp email or None) pairs. Only used to confirm a student's
    identity by co-authorship, so a missing PID costs a confirmation, never a
    wrong answer.
    """
    pids: list[str] = []
    for advisor, email in advisors:
        if email and email in pc_pids:
            pids.append(pc_pids[email])          # vetted by classify_reviewers.py
            continue
        if email and self_declared.get(email):
            pids.append(self_declared[email].most_common(1)[0][0])
            continue
        hits = search_candidates(dblp, advisor)
        if len(hits) == 1:
            pids.append(hits[0]["pid"])
            continue
        wanted = affiliation_tokens(affiliation)
        narrowed = [
            hit for hit in hits
            if wanted and affiliation_tokens(" ".join(hit["affiliations"])) & wanted
        ]
        if len(narrowed) == 1:
            pids.append(narrowed[0]["pid"])
    return pids


def resolve_student_pid(
    dblp: Dblp,
    student: str,
    affiliation: str,
    email: str,
    advisor_pids: list[str],
    self_declared: dict[str, Counter],
    max_pubs: int,
    max_names: int,
) -> tuple[str | None, str, list[str]]:
    """Resolve a student to a DBLP PID. Returns (pid, resolution, notes).

    Candidates are proposed by the three routes, verified against their own
    DBLP record, and then accepted in order of how much evidence stands behind
    them. Nothing is accepted on a partial name match alone, and nothing that
    looks like a homonym bucket is accepted at all.
    """
    wanted = name_tokens(student)
    notes: list[str] = []
    candidates: dict[str, PidCandidate] = {}

    def candidate(pid: str, route: str, names: list[str] | None = None) -> PidCandidate:
        entry = candidates.setdefault(pid, PidCandidate(pid=pid))
        entry.routes.add(route)
        relation = name_relation(names, wanted) if names else None
        if relation and entry.relation != "exact":
            entry.relation = relation
        return entry

    # The student's own submission names a page, but not who is on it.
    for pid in self_declared.get(email) or ():
        candidate(pid, "self-declared")

    advisor_coauthors: dict[str, str] = {}
    for advisor_pid in advisor_pids:
        profile = dblp.profile(advisor_pid)
        if profile is not None:
            advisor_coauthors.update(profile.get("coauthors") or {})
    for pid, name in advisor_coauthors.items():
        if name_relation([name], wanted):
            candidate(pid, "coauthor", [name])

    for hit in search_candidates(dblp, student):
        candidate(hit["pid"], "search", [hit["name"], *hit["aliases"]])

    if not candidates:
        return None, "not-found", notes

    # Co-authorship is read out of the advisor's record, which is already in
    # hand — no per-candidate fetch needed to establish it.
    for entry in candidates.values():
        entry.advisor_coauthor = entry.pid in advisor_coauthors

    # Only candidates that can still win are worth a DBLP fetch: those the
    # advisor has published with, and those the student named themselves. Ten
    # unrelated researchers called "Cheng Chen" are not worth thirty seconds.
    contenders = [
        e for e in candidates.values() if e.advisor_coauthor or "self-declared" in e.routes
    ]
    if not contenders:
        contenders = [e for e in candidates.values() if e.relation == "exact"]
        if len(contenders) > max_names:
            notes.append(
                f"{len(contenders)} DBLP people share this name and none has "
                f"published with the advisor"
            )
            return None, "ambiguous", notes
    if len(contenders) > max_names:
        notes.append(
            f"{len(contenders)} candidate page(s) match the name — too many to "
            f"tell apart: " + ", ".join(sorted(e.pid for e in contenders))
        )
        return None, "ambiguous", notes

    for entry in contenders:
        profile = dblp.profile(entry.pid)
        if profile is None:
            continue
        entry.fetched = True
        entry.relation = name_relation(profile["names"], wanted)
        entry.pubs = profile["pubs"]
        entry.dblp_name = profile["names"][0] if profile["names"] else ""
        entry.affiliations = list(profile["affiliations"])

    for entry in contenders:
        if entry.pubs is not None and entry.pubs > max_pubs:
            notes.append(
                f"ignored {entry.pid} ({entry.pubs} publications — a DBLP homonym "
                f"bucket or the wrong person, not a student)"
            )
    # A page the student claimed on their own submission *and* that their
    # advisor has published with is theirs even if DBLP spells the name in a
    # way the roster shares no word with — transliterated names do that, and
    # two independent facts outweigh a spelling.
    viable = [
        e for e in contenders
        if e.fetched and e.pubs is not None and e.pubs <= max_pubs
        and (e.relation is not None
             or (e.advisor_coauthor and "self-declared" in e.routes))
    ]

    confirmed = [e for e in viable if e.advisor_coauthor]
    if len(confirmed) > 1:
        # DBLP splits a common name across numbered pages, and an advisor can
        # have published with more than one of them. The affiliation DBLP
        # records for each is what separates the student from their namesake.
        wanted_affiliation = affiliation_tokens(affiliation)
        matching = [
            e for e in confirmed
            if wanted_affiliation and affiliation_tokens(" ".join(e.affiliations)) & wanted_affiliation
        ]
        if len(matching) == 1:
            notes.append(
                "chosen over same-named co-author(s) of the advisor by DBLP's "
                "recorded affiliation: "
                + ", ".join(e.describe() for e in confirmed if e is not matching[0])
            )
            confirmed = matching
    if len(confirmed) == 1:
        entry = confirmed[0]
        resolution = "confirmed" if len(entry.routes) > 1 else f"confirmed-{'+'.join(entry.routes)}"
        if entry.relation != "exact":
            notes.append(f"DBLP spells the name differently ({entry.describe()})")
        return entry.pid, resolution, notes
    if confirmed:
        notes.append("several co-authors of the advisor match: " + ", ".join(e.describe() for e in confirmed))
        return None, "ambiguous", notes

    # No co-authorship with the advisor to lean on: only an exact name match
    # can carry a candidate now, and only if it is the sole one.
    exact = [e for e in viable if e.relation == "exact"]
    declared = [e for e in exact if "self-declared" in e.routes]
    if len(declared) == 1:
        notes.append("from the student's own submission; no joint publication with the advisor on DBLP")
        return declared[0].pid, "self-declared", notes
    if len(exact) == 1:
        notes.append(f"single name match, unconfirmed ({exact[0].describe()})")
        return exact[0].pid, "name-only", notes
    if exact:
        notes.append("same-named DBLP people: " + ", ".join(e.describe() for e in exact))
        return None, "ambiguous", notes
    # Nothing is acceptable, but the pages that were looked at are exactly what
    # a human needs to finish the job by eye, so name them.
    rejected = [e for e in contenders if e.fetched and e.pubs is not None and e.pubs <= max_pubs]
    if rejected:
        notes.append("candidate page(s) rejected: " + ", ".join(e.describe() for e in rejected))
    return None, "unverified", notes


# ---------------------------------------------------------------------------
# Roster I/O
# ---------------------------------------------------------------------------

def write_roster(path: str, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    """Rewrite the roster atomically (tmp + replace), as every cache write does."""
    target = Path(path)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(target)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--trc", default=DEFAULT_TRC, help="path to the TRC roster CSV")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="path to the PC acceptance CSV")
    parser.add_argument("--data", default=DEFAULT_DATA, help="path to the HotCRP paper export JSON")
    parser.add_argument(
        "--seniority", default=DEFAULT_SENIORITY,
        help="classify_reviewers.py output, read for PC members' known DBLP PIDs",
    )
    parser.add_argument("--out", default=None, help="write here instead of over the roster")
    parser.add_argument(
        "--profile-cache", default=DEFAULT_PROFILE_CACHE,
        help=f"DBLP person-record cache (default: {DEFAULT_PROFILE_CACHE})",
    )
    parser.add_argument(
        "--search-cache", default=DEFAULT_SEARCH_CACHE,
        help=f"DBLP author-search cache (default: {DEFAULT_SEARCH_CACHE})",
    )
    parser.add_argument(
        "--max-pubs", type=int, default=DEFAULT_MAX_STUDENT_PUBS,
        help=f"reject a DBLP page with more publications than this "
             f"(default: {DEFAULT_MAX_STUDENT_PUBS})",
    )
    parser.add_argument(
        "--max-name-matches", type=int, default=DEFAULT_MAX_NAME_MATCHES,
        help=f"give up on an uncorroborated name shared by more than this many "
             f"DBLP people (default: {DEFAULT_MAX_NAME_MATCHES})",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace values already present in the two resolved columns",
    )
    parser.add_argument(
        "--no-network", action="store_true",
        help="use only what the caches already hold; resolve nothing new from DBLP",
    )
    parser.add_argument(
        "--delay", type=float, default=DEFAULT_DELAY,
        help=f"seconds between live DBLP fetches (default: {DEFAULT_DELAY})",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="process at most this many roster rows"
    )
    args = parser.parse_args()

    if args.delay < 0:
        parser.error("--delay must be non-negative")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")

    with open(args.trc, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    for column in (DBLP_COLUMN, ADVISOR_EMAIL_COLUMN):
        if column not in fieldnames:
            fieldnames.append(column)

    pc_index, unnamed_pc = index_pc_form(args.csv)
    with open(args.data, encoding="utf-8") as f:
        papers = json.load(f)
    author_index = index_hotcrp_authors(papers)
    self_declared = index_self_declared_pids(papers)
    pc_pids = load_pc_pids(args.seniority)
    print(
        f"{len(rows)} roster row(s); {len(pc_index)} PC-form name(s) "
        f"({len(unnamed_pc)} address-only), {len(author_index)} HotCRP author name(s), "
        f"{len(self_declared)} self-declared DBLP page(s), {len(pc_pids)} known PC DBLP PID(s)",
        file=sys.stderr,
    )

    dblp = Dblp(
        args.profile_cache, args.search_cache, delay=args.delay, offline=args.no_network
    )

    unresolved_advisor: list[tuple[dict, str]] = []
    unresolved_student: list[tuple[dict, str, list[str]]] = []
    flagged: list[tuple[dict, str, list[str]]] = []
    # advisor name -> the HotCRP addresses they were *not* resolved to. A PC
    # member who submits under a second address holds two HotCRP accounts, and
    # only the chair can decide whether both need the conflict.
    alternates: dict[str, tuple[str, set[str]]] = {}
    processed = 0

    for row in rows:
        if args.limit is not None and processed >= args.limit:
            break
        processed += 1
        student = (row.get("Name") or "").strip()

        # --- advisor -> HotCRP email ---
        advisors = split_advisor_names((row.get("Advisor Name") or "").strip())
        advisor_affiliation = (row.get("Advisor Affiliation") or "").strip()
        resolved: list[tuple[str, str | None]] = []
        resolutions: list[str] = []
        for advisor in advisors:
            email, resolution = resolve_advisor_email(
                advisor, advisor_affiliation, pc_index, author_index, unnamed_pc
            )
            resolutions.append(f"{advisor}: {resolution}")
            resolved.append((advisor, email))
            others = set(author_index.get(name_tokens(advisor), {})) - {email}
            if email and others:
                alternates[advisor] = (email, others)
        emails = [email for _, email in resolved if email]
        if emails and (args.overwrite or not (row.get(ADVISOR_EMAIL_COLUMN) or "").strip()):
            row[ADVISOR_EMAIL_COLUMN] = "; ".join(dict.fromkeys(emails))
        if len(emails) < len(advisors):
            unresolved_advisor.append((row, "; ".join(resolutions)))

        # --- student -> DBLP ---
        existing = (row.get(DBLP_COLUMN) or "").strip()
        if existing and not args.overwrite:
            print(f"  {student}: keeping existing DBLP {existing}", file=sys.stderr)
            continue
        advisor_pids = resolve_advisor_pids(
            dblp, resolved, advisor_affiliation, pc_pids, self_declared
        )
        pid, resolution, notes = resolve_student_pid(
            dblp, student, (row.get("Affiliation") or "").strip(),
            (row.get("Email") or "").strip().lower(), advisor_pids, self_declared,
            args.max_pubs, args.max_name_matches,
        )
        row[DBLP_COLUMN] = f"https://dblp.org/pid/{pid}.html" if pid else ""
        if pid and resolution != "confirmed":
            flagged.append((row, resolution, notes))
        elif not pid:
            unresolved_student.append((row, resolution, notes))
        print(f"  {student}: {resolution} {pid or '-'}", file=sys.stderr)

    out = args.out or args.trc
    write_roster(out, fieldnames, rows)

    resolved_dblp = sum(1 for r in rows if (r.get(DBLP_COLUMN) or "").strip())
    resolved_email = sum(1 for r in rows if (r.get(ADVISOR_EMAIL_COLUMN) or "").strip())
    print(
        f"\n{dblp.fetches} live DBLP fetch(es); {resolved_dblp}/{len(rows)} DBLP, "
        f"{resolved_email}/{len(rows)} advisor email(s) resolved",
        file=sys.stderr,
    )

    print(f"\n{len(unresolved_advisor)} advisor(s) with no HotCRP account found:", file=sys.stderr)
    for row, why in unresolved_advisor:
        print(f"    {row.get('Name')} — advisor {row.get('Advisor Name')} [{why}]", file=sys.stderr)

    print(
        f"\n{len(alternates)} advisor(s) reachable at more than one HotCRP address "
        f"(the conflict may need all of them):",
        file=sys.stderr,
    )
    for advisor, (chosen, others) in sorted(alternates.items()):
        print(f"    {advisor}: {chosen} — also {', '.join(sorted(others))}", file=sys.stderr)

    print(f"\n{len(unresolved_student)} student(s) left without a DBLP identity:", file=sys.stderr)
    for row, resolution, notes in unresolved_student:
        detail = f" — {'; '.join(notes)}" if notes else ""
        print(
            f"    {row.get('Name')} ({row.get('Affiliation')}) [{resolution}]{detail}",
            file=sys.stderr,
        )

    print(f"\n{len(flagged)} DBLP identity/identities worth a second look:", file=sys.stderr)
    for row, resolution, notes in flagged:
        detail = f" — {'; '.join(notes)}" if notes else ""
        print(f"    {row.get('Name')}: {row.get(DBLP_COLUMN)} [{resolution}]{detail}", file=sys.stderr)

    print(f"\nWrote {len(rows)} roster row(s) to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
