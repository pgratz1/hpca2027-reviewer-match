"""Parse the reserve-reviewer roster into Reviewer records with derived areas.

Reserve reviewers are recruited from the submissions' `reserve_reviewer`
nominations and added to HotCRP directly, so unlike the PC and the area chairs
they never filled in an acceptance form. `reserve_reviewer_info.csv` (built by
build_reserve_reviewer_info.py) gives their email, name and DBLP page, and that
is all there is: no declared areas, no keywords, no tier.

Areas matter, because the area gate in paper_matching.eligible_scores intersects
a reviewer's primary/secondary with a paper's topics, and a reviewer holding
neither matches no paper at all. They are derived here from the HotCRP topics of
the submissions the person authored — the papers that put them in the reserve
pool in the first place. Those topics come from HotCRP's own topic list, so they
are already in the vocabulary the gate compares against, with none of the
free-text drift the acceptance form's areas need reconciling for.

Records come back as real `Reviewer` objects carrying `tier="reserve"`, so every
consumer that already takes a Reviewer — enrich_publications.py,
build_fingerprints.py, classify_reviewers.py — takes a reserve unchanged.
"""

from __future__ import annotations

from .paths import assignment_path, cache_path, curated_path, input_path, report_path, smoke_cache_path

import csv
from collections import Counter, defaultdict

import json

from . import pc_membership
from .dblp import parse_pid
from .reviewers import Reviewer

DEFAULT_INFO = report_path("reserve_reviewer_info.csv")
DEFAULT_DATA = input_path("hpca2027-data.json")

# Areas kept per reserve. The gate reads primary and secondary only; the third
# still reaches the fingerprint's area document, which is why it is collected.
DEFAULT_MAX_AREAS = 3

TIER = "reserve"


def _nominated_emails(paper: dict) -> list[str]:
    """Emails in a submission's `reserve_reviewer` field.

    One nomination per line, written "Name / Affiliation / email".
    """
    emails = []
    for line in (paper.get("reserve_reviewer") or "").splitlines():
        parts = [part.strip() for part in line.split("/")]
        if len(parts) >= 3 and "@" in parts[-1]:
            emails.append(parts[-1].lower())
    return emails


def index_topics(papers: list[dict]) -> tuple[dict[str, Counter], dict[str, Counter]]:
    """({email: Counter(topic)} from authorship, and the same from nominations).

    Kept apart so authorship — the person's own work — is preferred, and being
    named on somebody else's paper is only the fallback.
    """
    authored: dict[str, Counter] = defaultdict(Counter)
    nominated: dict[str, Counter] = defaultdict(Counter)
    for paper in papers:
        topics = paper.get("topics") or []
        if not topics:
            continue
        for author in paper.get("authors") or []:
            email = (author.get("email") or "").strip().lower()
            if email:
                authored[email].update(topics)
        for email in _nominated_emails(paper):
            nominated[email].update(topics)
    return authored, nominated


def index_affiliations(papers: list[dict]) -> dict[str, Counter]:
    """{email: Counter(affiliation)} over author and contact records."""
    index: dict[str, Counter] = defaultdict(Counter)
    for paper in papers:
        for person in (paper.get("authors") or []) + (paper.get("contacts") or []):
            email = (person.get("email") or "").strip().lower()
            affiliation = (person.get("affiliation") or "").strip()
            if email and affiliation:
                index[email][affiliation] += 1
    return index


def top_areas(counts: Counter, max_areas: int) -> list[str]:
    """The `max_areas` most frequent topics, ties broken alphabetically.

    Counter.most_common leaves ties in insertion order, which depends on the
    order papers happen to appear in the export; sorting the tie makes the
    derived areas — and so every fingerprint built from them — reproducible.
    """
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [topic for topic, _ in ranked[:max_areas]]


def load_reserve_reviewers(
    info_path: str = DEFAULT_INFO,
    data_path: str = DEFAULT_DATA,
    *,
    max_areas: int = DEFAULT_MAX_AREAS,
    pcinfo_path: str | None = pc_membership.DEFAULT_PCINFO,
) -> list[Reviewer]:
    """Load the reserve roster, deriving each one's areas from their papers.

    Topics are read from the whole export rather than through
    paper_matching.load_papers. The paper policy decides which submissions get
    *reviewed*, which is a different question from what a person works on: a
    withdrawn or still-draft paper says just as much about their expertise, and
    filtering by policy left three of them with no areas at all and so gated out
    of everything.

    Reserves are checked against the HotCRP user export the same way PC members
    are, and it matters more here: a recruit who is later stood down keeps the
    `reserve-reviewer` tag on their account, so the roster file alone cannot
    tell that the `pc` role is gone. `pcinfo_path=None` skips the check.
    """
    with open(data_path, encoding="utf-8") as f:
        papers = json.load(f)
    authored, nominated = index_topics(papers)
    affiliations = index_affiliations(papers)

    with open(info_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    index = pc_membership.load_pc_accounts(pcinfo_path) if pcinfo_path else None

    reserves = []
    dropped: list[str] = []
    for row in rows:
        email = (row.get("email") or "").strip().lower()
        if not email:
            continue
        name = (row.get("name") or "").strip()
        first, _, last = name.partition(" ")
        if index is not None and index.match(email, first, last)[0] is None:
            dropped.append(email)
            continue
        topics = authored.get(email) or nominated.get(email) or Counter()
        areas = top_areas(topics, max_areas)
        areas += [""] * (3 - len(areas))
        affiliation = affiliations.get(email)
        dblp_url = (row.get("dblp") or "").strip()
        reserves.append(
            Reviewer(
                email=email,
                first=first,
                last=last,
                dblp_url=dblp_url,
                pid=parse_pid(dblp_url),
                affiliation=affiliation.most_common(1)[0][0] if affiliation else "",
                primary=areas[0],
                secondary=areas[1],
                tertiary=areas[2],
                keywords="",
                tier=TIER,
                override_cap=None,
                # The roster is itself the hand-maintained identity layer
                # (reserve_dblp_overrides.csv feeds it), so every PID here is a
                # decision someone made rather than a form cell.
                pid_from_override=True,
            )
        )

    if index is not None:
        pc_membership.report_pruned(dropped, len(reserves), "reserve reviewers", index)
    return reserves
