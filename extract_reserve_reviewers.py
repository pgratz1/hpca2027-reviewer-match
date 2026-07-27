"""Turn HotCRP's free-text `reserve_reviewer` field into a candidate reviewer list.

Every HPCA submission nominates a senior author who can be called upon to
review ("reserve reviewer"). HotCRP stores that as one free-text string per
paper, in no fixed shape: "Name / Affiliation / email", "Name <email> (Affil)",
a bare name, or an exemption statement ("exempt", "one senior author is already
on the PC"). This script normalises the field into one row per person with a
DBLP identity, ready to invite — see estimate_reserve_need.py for how many are
needed.

Extraction, per nominating paper:
  1. if the field contains an email, that is the person; the paper's own author
     list supplies their name, affiliation, and position;
  2. otherwise, if the whole field exactly matches one author's name, that
     author is the person (`--no-name-match` disables this). Nothing fuzzier is
     attempted — exemption text and multi-person fields simply fall out here.

DBLP resolution uses the paper's `dblp` field, a delimited list positionally
aligned with `authors` (with `None` placeholders). Positional indexing is only
trusted when the list length equals the author count — about one paper in six
disagrees. Where it can't be trusted, or where a person's nominations disagree
about their PID, every PID listed on their papers is probed against DBLP's
person record and matched by name (`--no-name-probe` skips the network
entirely). Candidates who are already PC members, or whose DBLP stays
ambiguous, are dropped and reported.

    python extract_reserve_reviewers.py
    python extract_reserve_reviewers.py --no-name-probe --out /tmp/dry-run.csv
    python extract_reserve_reviewers.py --limit 20   # probe 20 candidates first
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import requests

from dblp import fetch_person_names_for_pids, load_cache, parse_pid, save_cache
from reviewers import load_reviewers

DEFAULT_CSV = "HPCA'27 PC Member Acceptance Form (Responses) - Form Responses 1.csv"
DEFAULT_DATA = "hpca2027-data.json"
DEFAULT_OUT = "reserve_reviewers.csv"
DEFAULT_PERSON_CACHE = "dblp_person_cache.json"
DEFAULT_DELAY = 3.0

OUTPUT_COLUMNS = ["name", "email", "affiliation", "dblp", "pid", "nominating_pids", "resolution"]

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")

# The dblp field's entries are separated by ';', ',', or a newline — and often
# by ',\n' or ';\n' together, which is ONE separator, not two. Collapsing only
# the newline-bearing runs keeps a bare ';;' meaning "this author has no page",
# which is what makes the positional alignment hold: matching the author count
# for 1,138 of the 1,354 papers that have the field, against 970 if ',\n' is
# read as two separators. No DBLP URL or bare PID contains any of these
# characters, so one rule covers every observed shape.
_DBLP_SPLIT_RE = re.compile(r"[ \t]*(?:[;,][ \t]*\n|\n|[;,])[ \t]*")

# Placeholders authors type for "this author has no DBLP page". Filtered before
# parse_pid, whose bare-PID branch would happily read "N/A" as the PID "N/A".
_DBLP_PLACEHOLDERS = {"", "-", "--", "n/a", "na", "none", "null", "no", "nil", "?"}

# A trailing all-digit token is DBLP's homonym discriminator ("Yang Wang 0089"),
# not part of the name.
_HOMONYM_SUFFIX_RE = re.compile(r"\s+\d+$")


@dataclass
class Candidate:
    """One nominated person, aggregated across every paper that nominated them."""

    email: str
    names: Counter = field(default_factory=Counter)
    affiliations: Counter = field(default_factory=Counter)
    nominating_pids: set[int] = field(default_factory=set)
    # PIDs read out of a trustworthy positional alignment, by how many
    # nominations agreed on them.
    positional_pids: Counter = field(default_factory=Counter)
    # Every PID appearing anywhere on their nominating papers — the pool the
    # DBLP name probe searches when the positional read is unavailable.
    paper_pids: set[str] = field(default_factory=set)

    @property
    def name(self) -> str:
        """The most frequently used spelling of their name."""
        return self.names.most_common(1)[0][0] if self.names else ""

    @property
    def affiliation(self) -> str:
        return self.affiliations.most_common(1)[0][0] if self.affiliations else ""


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


def author_name(author: dict) -> str:
    return f"{author.get('given_name') or ''} {author.get('family_name') or ''}".strip()


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


def parse_free_text(raw: str, email: str) -> tuple[str, str]:
    """Best-effort (name, affiliation) from a field whose email matched no author.

    Handles the shapes actually present: "Name / Affiliation / email",
    "Name, Affiliation, email", and "Name <email> (Affiliation)". Returns
    ("", "") rather than guessing when nothing usable is left.
    """
    text = raw.replace(email, " ")
    text = re.sub(r"[<>()]", " ", text)
    separator = "/" if "/" in text else ","
    parts = [p.strip(" \t\n\r-;:") for p in text.split(separator)]
    parts = [p for p in parts if p]
    name = parts[0] if parts else ""
    affiliation = parts[1] if len(parts) > 1 else ""
    return name, affiliation


def nomination(paper: dict, allow_name_match: bool) -> tuple[str, str, str, int | None] | None:
    """Resolve one paper's `reserve_reviewer` field.

    Returns (email, name, affiliation, author_index) — author_index is None
    when the person is not among the paper's authors — or None when the field
    names no identifiable person (blank, an exemption statement, several
    people, or a name matching no single author).
    """
    raw = (paper.get("reserve_reviewer") or "").strip()
    if not raw:
        return None
    authors = paper.get("authors") or []

    match = _EMAIL_RE.search(raw)
    if match:
        email = match.group(0).lower()
        for index, author in enumerate(authors):
            if (author.get("email") or "").strip().lower() == email:
                return email, author_name(author), (author.get("affiliation") or "").strip(), index
        # A typo'd email (or a nominee who isn't an author): fall back to the
        # contact list, then to parsing the free text around the email.
        for contact in paper.get("contacts") or []:
            if (contact.get("email") or "").strip().lower() == email:
                return email, author_name(contact), (contact.get("affiliation") or "").strip(), None
        name, affiliation = parse_free_text(raw, match.group(0))
        return email, name, affiliation, None

    if not allow_name_match:
        return None
    wanted = name_tokens(raw)
    if not wanted:
        return None
    hits = [i for i, a in enumerate(authors) if name_tokens(author_name(a)) == wanted]
    if len(hits) != 1:
        return None
    author = authors[hits[0]]
    email = (author.get("email") or "").strip().lower()
    if not email:
        return None
    return email, author_name(author), (author.get("affiliation") or "").strip(), hits[0]


def collect_candidates(
    papers: list[dict], *, allow_name_match: bool = True
) -> tuple[dict[str, Candidate], Counter]:
    """Aggregate every paper's nomination into one Candidate per email."""
    candidates: dict[str, Candidate] = {}
    stats: Counter = Counter()

    for paper in papers:
        raw = (paper.get("reserve_reviewer") or "").strip()
        if not raw:
            stats["blank"] += 1
            continue
        stats["nominations"] += 1

        resolved = nomination(paper, allow_name_match)
        if resolved is None:
            stats["unparseable"] += 1
            continue
        email, name, affiliation, index = resolved

        candidate = candidates.setdefault(email, Candidate(email=email))
        candidate.nominating_pids.add(paper["pid"])
        if name:
            candidate.names[name] += 1
        if affiliation:
            candidate.affiliations[affiliation] += 1

        entries = split_dblp_field(paper.get("dblp"))
        for entry in entries:
            pid = parse_pid(entry)
            if pid:
                candidate.paper_pids.add(pid)
        # Positional indexing only means anything when every author has a slot.
        if index is not None and len(entries) == len(paper.get("authors") or []):
            pid = parse_pid(entries[index])
            if pid:
                candidate.positional_pids[pid] += 1

    return candidates, stats


def probe_pool(candidate: Candidate) -> list[str]:
    """PIDs worth probing against DBLP for this candidate.

    Conflicting positional reads are settled among themselves; when there was
    no positional read at all, every PID on their papers is a suspect.
    """
    if len(candidate.positional_pids) > 1:
        return sorted(candidate.positional_pids)
    if not candidate.positional_pids:
        return sorted(candidate.paper_pids)
    return []


def probe_match(candidate: Candidate, pool: list[str], pid_names: dict[str, list[str]]) -> str | None:
    """The one PID in `pool` whose DBLP names match the candidate's, if unique."""
    wanted = {name_tokens(n) for n in candidate.names if name_tokens(n)}
    if not wanted:
        return None
    hits = [
        pid for pid in pool
        if any(name_tokens(n) in wanted for n in pid_names.get(pid, ()))
    ]
    return hits[0] if len(hits) == 1 else None


def merge_by_pid(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[list[dict[str, str]]]]:
    """Collapse rows that resolved to the same DBLP person into one.

    The same researcher is nominated under two email addresses more often than
    one would hope — a typo in one paper's field, or an institutional and a
    personal address. The PID is this project's real identity, so it wins: the
    address that was nominated most often becomes the row's email, and the
    nominating papers are pooled. Returns (merged_rows, merged_groups), the
    groups being the multi-row inputs so the caller can report them.
    """
    by_pid: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_pid.setdefault(row["pid"], []).append(row)

    merged: list[dict[str, str]] = []
    groups: list[list[dict[str, str]]] = []
    for group in by_pid.values():
        if len(group) == 1:
            merged.append(group[0])
            continue
        groups.append(group)
        papers = sorted({int(p) for row in group for p in row["nominating_pids"].split()})
        primary = max(group, key=lambda r: len(r["nominating_pids"].split()))
        merged.append({
            **primary,
            "nominating_pids": " ".join(str(p) for p in papers),
            "resolution": f"{primary['resolution']}+merged",
        })
    return merged, groups


def write_output(path: str, rows: list[dict[str, str]]) -> None:
    """Write the candidate list atomically (tmp + replace), as every cache does."""
    target = Path(path)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(target)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default=DEFAULT_DATA, help="path to the HotCRP paper export JSON")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="path to the reviewer CSV")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"output CSV (default: {DEFAULT_OUT})")
    parser.add_argument(
        "--person-cache", default=DEFAULT_PERSON_CACHE,
        help=f"DBLP person-name cache (default: {DEFAULT_PERSON_CACHE})",
    )
    parser.add_argument(
        "--no-name-probe", action="store_true",
        help="skip the DBLP name probe entirely; positional reads only, no network",
    )
    parser.add_argument(
        "--no-name-match", action="store_true",
        help="only extract nominations that contain an email address",
    )
    parser.add_argument(
        "--delay", type=float, default=DEFAULT_DELAY,
        help=f"seconds between live DBLP fetches (default: {DEFAULT_DELAY})",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="probe at most this many candidates (validate the probe cheaply first)",
    )
    args = parser.parse_args()

    if args.delay < 0:
        parser.error("--delay must be non-negative")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")

    with open(args.data, encoding="utf-8") as f:
        papers = json.load(f)

    # Every record counts here, not just the selected-policy ones: a nomination
    # identifies a person regardless of whether their paper is complete enough
    # to be assigned reviewers.
    candidates, stats = collect_candidates(papers, allow_name_match=not args.no_name_match)
    print(
        f"{stats['nominations']} nomination(s) across {len(papers)} record(s) "
        f"({stats['blank']} blank); {stats['unparseable']} named no identifiable "
        f"person (exemptions, multiple names); {len(candidates)} unique person(s)",
        file=sys.stderr,
    )

    reviewers = load_reviewers(args.csv)
    pc_emails = {r.email.lower() for r in reviewers}
    pc_pids = {r.pid for r in reviewers if r.pid}

    already_pc = sorted(e for e in candidates if e in pc_emails)
    for email in already_pc:
        del candidates[email]
    print(f"Dropped {len(already_pc)} already on the PC by email", file=sys.stderr)

    # Probe only what the positional read couldn't settle.
    to_probe = {e: probe_pool(c) for e, c in candidates.items()}
    to_probe = {e: pool for e, pool in to_probe.items() if pool}
    pid_names: dict[str, list[str]] = {}
    if to_probe and not args.no_name_probe:
        emails = sorted(to_probe)[: args.limit] if args.limit is not None else sorted(to_probe)
        pids = sorted({pid for e in emails for pid in to_probe[e]})
        print(
            f"Probing DBLP for {len(pids)} PID(s) to identify {len(emails)} "
            f"candidate(s) with no unambiguous positional DBLP",
            file=sys.stderr,
        )
        cache = load_cache(args.person_cache)
        session = requests.Session()
        fetched = fetch_person_names_for_pids(
            pids, session=session, write_cache=cache, cache_path=args.person_cache,
            delay=args.delay,
            on_error=lambda pid, exc: print(
                f"  DBLP lookup failed for {pid}: {type(exc).__name__}", file=sys.stderr
            ),
        )
        pid_names = {pid: names for pid, (names, _) in fetched.items()}
        save_cache(cache, args.person_cache)
    elif to_probe:
        print(
            f"Skipping the DBLP name probe for {len(to_probe)} candidate(s) "
            f"(--no-name-probe)",
            file=sys.stderr,
        )

    rows: list[dict[str, str]] = []
    ambiguous: list[Candidate] = []
    no_dblp: list[Candidate] = []
    pid_is_pc: list[tuple[Candidate, str]] = []

    for email in sorted(candidates):
        candidate = candidates[email]
        pool = to_probe.get(email, [])
        if len(candidate.positional_pids) == 1:
            pid = next(iter(candidate.positional_pids))
            resolution = (
                "positional-agreed" if candidate.positional_pids[pid] > 1 else "positional"
            )
        elif pool:
            pid = probe_match(candidate, pool, pid_names)
            resolution = "dblp-name-probe"
        else:
            pid = None
            resolution = ""

        if pid is None:
            (ambiguous if pool else no_dblp).append(candidate)
            continue
        if pid in pc_pids:
            pid_is_pc.append((candidate, pid))
            continue

        rows.append({
            "name": candidate.name,
            "email": candidate.email,
            "affiliation": candidate.affiliation,
            "dblp": f"https://dblp.org/pid/{pid}.html",
            "pid": pid,
            "nominating_pids": " ".join(str(p) for p in sorted(candidate.nominating_pids)),
            "resolution": resolution,
        })

    rows, merged_groups = merge_by_pid(rows)
    rows.sort(key=lambda r: (r["name"].lower(), r["email"]))
    write_output(args.out, rows)

    print(
        f"\nMerged {len(merged_groups)} person(s) nominated under several email addresses:",
        file=sys.stderr,
    )
    for group in merged_groups:
        addresses = ", ".join(sorted(r["email"] for r in group))
        print(f"    {group[0]['name']} ({group[0]['pid']}): {addresses}", file=sys.stderr)

    print(f"\nDropped {len(pid_is_pc)} whose DBLP is already a PC member's:", file=sys.stderr)
    for candidate, pid in pid_is_pc:
        print(f"    {candidate.name} <{candidate.email}> = {pid}", file=sys.stderr)
    print(f"\nDropped {len(no_dblp)} with no DBLP listed on any nominating paper:", file=sys.stderr)
    for candidate in no_dblp:
        print(f"    {candidate.name or '(name unknown)'} <{candidate.email}>", file=sys.stderr)
    unprobed = " (the DBLP name probe did not run for them)" if args.no_name_probe else ""
    print(
        f"\nDropped {len(ambiguous)} whose DBLP could not be disambiguated{unprobed}:",
        file=sys.stderr,
    )
    for candidate in ambiguous:
        pool = ", ".join(to_probe.get(candidate.email, []))
        print(
            f"    {candidate.name or '(name unknown)'} <{candidate.email}> "
            f"— candidates: {pool}",
            file=sys.stderr,
        )

    print(f"\nWrote {len(rows)} reserve reviewer(s) to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
