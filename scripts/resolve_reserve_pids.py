"""Resolve the DBLP pages that build_reserve_reviewer_info.py could not accept.

Verification against DBLP showed the reserve-reviewer vetting workbook's links
are unreliable: most of them are numeric PIDs that turned out to name somebody
else entirely. This proposes replacements, and takes its evidence from the place
the reserve reviewers came from in the first place — the submissions themselves.

Three routes propose a PID, and no route is trusted on its own:

  self-declared  the person is an author on a submission, and that submission's
                 `dblp` field (positionally aligned with the author list) names
                 their page. Beware: the workbook was largely *built* from this
                 field, so where the alignment slipped, workbook and submission
                 are wrong together and agreeing means nothing
  coauthor       a page that lists one of the co-authors from their own
                 submission. Two people publishing together is the strongest
                 signal here, and it costs no extra fetch — the co-author PIDs
                 come off the same aligned `dblp` field
  search         DBLP's author search, which is fuzzy and proposes near misses

Every surviving candidate is then fetched and checked against its own DBLP
record: the name must match, and where DBLP records an institution it is
compared with the affiliation HotCRP has. A page accepted here is one that two
routes agree on, or that a co-author confirms, or that is the single exact name
match; everything weaker is left for a human, with the pages it looked at named
so the decision can be made by eye.

Writes `reserve_dblp_overrides.csv` (email,dblp,note) — the hand-maintained
identity layer for reserves, which build_reserve_reviewer_info.py reads ahead of
the workbook. Rows it cannot resolve are appended with an empty `dblp` cell, so
the same file is the to-do list: fill the cell in, re-run `make reserve-info
VERIFY=--verify`, and the person joins the roster.

    python -m scripts.resolve_reserve_pids
    python -m scripts.resolve_reserve_pids --no-network      # cached lookups only
    python -m scripts.resolve_reserve_pids --out /tmp/dry-run.csv
"""

from __future__ import annotations

from reviewer_match.paths import assignment_path, cache_path, curated_path, input_path, report_path

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from reviewer_match.dblp import name_tokens, parse_pid, split_dblp_field
from scripts.resolve_trc_members import (
    DEFAULT_DELAY,
    DEFAULT_PROFILE_CACHE,
    DEFAULT_SEARCH_CACHE,
    Dblp,
    affiliation_tokens,
    author_name,
    index_self_declared_pids,
    name_relation,
    search_candidates,
)

DEFAULT_UNRESOLVED = report_path("reserve_reviewer_unresolved.csv")
DEFAULT_DATA = input_path("hpca2027-data.json")
DEFAULT_OUT = curated_path("reserve_dblp_overrides.csv")

OVERRIDE_FIELDS = ["email", "dblp", "note"]

# More same-named DBLP pages than this and no automated choice is defensible;
# the candidates are reported instead so the chair can pick by eye.
DEFAULT_MAX_NAMES = 8


@dataclass
class Candidate:
    """One proposed DBLP identity, and the evidence standing behind it."""

    pid: str
    routes: set[str] = field(default_factory=set)
    relation: str | None = None      # how its DBLP name compares to HotCRP's
    coauthor: str = ""               # a submission co-author found on its record
    affiliation_match: bool = False
    # Affiliations the *search* reported, known before any page is fetched --
    # which is what lets the right page survive the cap on a name like "Chao
    # Wang", where DBLP numbers more than forty different people.
    listed_affiliations: list[str] = field(default_factory=list)
    dblp_name: str = ""
    fetched: bool = False

    def describe(self) -> str:
        marks = ["+".join(sorted(self.routes)), self.dblp_name or "?",
                 self.relation or "name mismatch"]
        if self.coauthor:
            marks.append(f"co-authored with {self.coauthor}")
        if self.affiliation_match:
            marks.append("affiliation matches")
        return f"{self.pid} [{', '.join(marks)}]"


def index_submission_coauthors(papers: list[dict]) -> dict[str, dict[str, str]]:
    """{author email: {co-author PID: name}} from the submissions they wrote.

    Read off the same positionally-aligned `dblp` field as the self-declared
    index, so it is only as good as that alignment -- but a wrong co-author is
    merely a candidate that fails to confirm, never one that is wrongly
    accepted.
    """
    index: dict[str, dict[str, str]] = defaultdict(dict)
    for paper in papers:
        authors = paper.get("authors") or []
        entries = split_dblp_field(paper.get("dblp"))
        if len(entries) != len(authors):
            continue
        pids = [parse_pid(entry) for entry in entries]
        for position, author in enumerate(authors):
            email = (author.get("email") or "").strip().lower()
            if not email:
                continue
            for other, pid in enumerate(pids):
                if pid and other != position:
                    index[email][pid] = author_name(authors[other])
    return index


def index_hotcrp_affiliations(papers: list[dict]) -> dict[str, Counter]:
    """{email: Counter(affiliation)} from author records and reserve nominations."""
    index: dict[str, Counter] = defaultdict(Counter)
    for paper in papers:
        for author in (paper.get("authors") or []) + (paper.get("contacts") or []):
            email = (author.get("email") or "").strip().lower()
            affiliation = (author.get("affiliation") or "").strip()
            if email and affiliation:
                index[email][affiliation] += 1
        # "Name / Affiliation / email", one nomination per line.
        for line in (paper.get("reserve_reviewer") or "").splitlines():
            parts = [part.strip() for part in line.split("/")]
            if len(parts) >= 3 and "@" in parts[-1]:
                index[parts[-1].lower()][" ".join(parts[1:-1])] += 1
    return index


def resolve(
    dblp: Dblp, name: str, email: str, affiliation: str, rejected: str | None,
    self_declared: dict[str, Counter], coauthors: dict[str, dict[str, str]],
    max_names: int,
) -> tuple[str | None, str, list[str]]:
    """Resolve one reserve reviewer to a PID. Returns (pid, resolution, notes)."""
    wanted = name_tokens(name)
    wanted_affiliation = affiliation_tokens(affiliation)
    known_coauthors = coauthors.get(email) or {}
    notes: list[str] = []
    candidates: dict[str, Candidate] = {}

    def propose(pid: str, route: str, names: list[str] | None = None,
                listed: list[str] | None = None) -> None:
        entry = candidates.setdefault(pid, Candidate(pid=pid))
        entry.routes.add(route)
        relation = name_relation(names, wanted) if names else None
        if relation and entry.relation != "exact":
            entry.relation = relation
        if listed:
            entry.listed_affiliations.extend(listed)

    def listed_affiliation_match(entry: Candidate) -> bool:
        return bool(
            wanted_affiliation
            and affiliation_tokens(" ".join(entry.listed_affiliations)) & wanted_affiliation
        )

    for pid in self_declared.get(email) or ():
        propose(pid, "self-declared")
    for hit in search_candidates(dblp, name):
        propose(hit["pid"], "search", [hit["name"], *hit["aliases"]],
                hit.get("affiliations"))

    # A co-author's DBLP record lists everyone they have published with, our
    # person included. Their pages are only opened when the cheap routes have
    # proposed nothing, since each one costs a fetch.
    if not candidates:
        for coauthor_pid in list(known_coauthors)[:max_names]:
            profile = dblp.profile(coauthor_pid)
            if profile is None:
                continue
            for pid, listed in (profile.get("coauthors") or {}).items():
                if pid != coauthor_pid and name_relation([listed], wanted):
                    propose(pid, "coauthor", [listed])

    # The page the workbook proposed already failed verification; re-proposing
    # it needs evidence the workbook never had.
    if rejected in candidates and candidates[rejected].routes == {"self-declared"}:
        notes.append(
            f"{rejected} is what the workbook already claimed, and the submission "
            f"it came from says the same thing — the workbook was built from that "
            f"field, so this is one source, not two"
        )

    if not candidates:
        return None, "not-found", notes

    # Only pages that could still win are worth a fetch.
    contenders = [
        e for e in candidates.values()
        if "self-declared" in e.routes or e.relation is not None
    ]
    if len(contenders) > max_names:
        # A common name returns dozens of real people -- DBLP numbers over forty
        # called "Chao Wang" -- so the slice has to be ordered by what is known
        # before fetching. The search's own affiliation note comes first: it is
        # the only thing that distinguishes "Chong Zhang 0017 (Southwest
        # Petroleum University)" from sixteen other Chong Zhangs.
        ranked = sorted(
            contenders,
            key=lambda e: (not listed_affiliation_match(e),
                           "self-declared" not in e.routes,
                           e.relation != "exact"),
        )
        contenders = ranked[:max_names]
    if not contenders:
        return None, "not-found", notes

    for entry in contenders:
        profile = dblp.profile(entry.pid)
        if profile is None:
            continue
        entry.fetched = True
        entry.relation = name_relation(profile["names"], wanted)
        entry.dblp_name = profile["names"][0] if profile["names"] else ""
        entry.affiliation_match = bool(
            wanted_affiliation
            and affiliation_tokens(" ".join(profile["affiliations"])) & wanted_affiliation
        ) or listed_affiliation_match(entry)
        for pid, coauthor_name in (profile.get("coauthors") or {}).items():
            if pid in known_coauthors and pid != entry.pid:
                entry.coauthor = coauthor_name
                entry.routes.add("coauthor")
                break

    # A page that does not carry the person's name is not theirs, whatever
    # proposed it -- this is exactly the failure being repaired.
    viable = [e for e in contenders if e.fetched and e.relation is not None]
    if not viable:
        looked_at = [e for e in contenders if e.fetched]
        if looked_at:
            notes.append("rejected: " + ", ".join(e.describe() for e in looked_at))
        return None, "unverified", notes

    confirmed = [e for e in viable if e.coauthor]
    if len(confirmed) > 1 and any(e.affiliation_match for e in confirmed):
        confirmed = [e for e in confirmed if e.affiliation_match]
    if len(confirmed) == 1:
        return confirmed[0].pid, "confirmed-coauthor", notes
    if confirmed:
        notes.append("several co-authored pages match: "
                     + ", ".join(e.describe() for e in confirmed))
        return None, "ambiguous", notes

    exact = [e for e in viable if e.relation == "exact"]
    multi = [e for e in exact if len(e.routes) > 1]
    if len(multi) == 1:
        return multi[0].pid, "confirmed-two-routes", notes

    by_affiliation = [e for e in exact if e.affiliation_match]
    if len(by_affiliation) == 1 and len(exact) > 1:
        notes.append("chosen among same-named pages by DBLP's recorded affiliation: "
                     + ", ".join(e.describe() for e in exact))
        return by_affiliation[0].pid, "affiliation", notes

    if len(exact) == 1:
        entry = exact[0]
        route = "self-declared" if "self-declared" in entry.routes else "name-only"
        if route == "name-only":
            notes.append(f"single name match, nothing corroborates it ({entry.describe()})")
        return entry.pid, route, notes
    if exact:
        notes.append("same-named DBLP pages: " + ", ".join(e.describe() for e in exact))
        return None, "ambiguous", notes

    notes.append("only partial name matches: " + ", ".join(e.describe() for e in viable))
    return None, "unverified", notes


def write_overrides(path: str, rows: list[dict[str, str]]) -> None:
    """Write the override file atomically, as every cache write here does."""
    target = Path(path)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OVERRIDE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(target)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--unresolved", default=DEFAULT_UNRESOLVED,
                        help=f"rows to resolve (default: {DEFAULT_UNRESOLVED})")
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help=f"HotCRP paper export (default: {DEFAULT_DATA})")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help=f"reserve identity overrides (default: {DEFAULT_OUT})")
    parser.add_argument("--profile-cache", default=DEFAULT_PROFILE_CACHE)
    parser.add_argument("--search-cache", default=DEFAULT_SEARCH_CACHE)
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                        help=f"seconds between live DBLP fetches (default: {DEFAULT_DELAY})")
    parser.add_argument("--no-network", action="store_true",
                        help="use only what the caches already hold")
    parser.add_argument("--max-names", type=int, default=DEFAULT_MAX_NAMES,
                        help=f"give up past this many same-named pages (default: {DEFAULT_MAX_NAMES})")
    args = parser.parse_args()

    with open(args.unresolved, newline="", encoding="utf-8") as f:
        pending = list(csv.DictReader(f))
    with open(args.data, encoding="utf-8") as f:
        papers = json.load(f)

    self_declared = index_self_declared_pids(papers)
    coauthors = index_submission_coauthors(papers)
    affiliations = index_hotcrp_affiliations(papers)
    print(f"{len(pending)} unresolved reserve reviewer(s); "
          f"{len(papers)} submissions indexed", file=sys.stderr)

    # Anything already decided by hand outranks everything proposed here.
    existing: dict[str, dict[str, str]] = {}
    if Path(args.out).exists():
        with open(args.out, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                email = (row.get("email") or "").strip().lower()
                if email:
                    existing[email] = row

    dblp = Dblp(args.profile_cache, args.search_cache,
                delay=args.delay, offline=args.no_network)

    # Every row already in the file is carried forward, not just the ones still
    # listed as unresolved: once a person is resolved they drop out of that
    # list, and rebuilding from it alone would delete the very decision that
    # resolved them -- taking them off the roster on the next rebuild.
    rows = [dict(row) for row in existing.values()]
    by_email = {row["email"].strip().lower(): row for row in rows}
    resolutions = Counter()
    for entry in pending:
        email = entry["email"].strip().lower()
        name = entry["name"].strip()
        kept = existing.get(email)
        if kept and (kept.get("dblp") or "").strip():
            resolutions["already-decided"] += 1
            continue

        affiliation = ""
        if affiliations.get(email):
            affiliation = affiliations[email].most_common(1)[0][0]
        pid, resolution, notes = resolve(
            dblp, name, email, affiliation, parse_pid(entry.get("dblp_url") or ""),
            self_declared, coauthors, args.max_names,
        )
        resolutions[resolution] += 1
        note = f"{name} ({affiliation or 'affiliation unknown'}) — {resolution}"
        if notes:
            note += ": " + "; ".join(notes)
        proposed = {
            "email": email,
            "dblp": f"https://dblp.org/pid/{pid}.html" if pid else "",
            "note": note,
        }
        if email in by_email:
            by_email[email].update(proposed)
        else:
            rows.append(proposed)
            by_email[email] = proposed
        print(f"  {resolution:<20} {email:<32} {pid or '-'}", file=sys.stderr)

    rows.sort(key=lambda r: r["email"])
    write_overrides(args.out, rows)

    filled = sum(1 for r in rows if r["dblp"])
    print(f"\n{dblp.fetches} live DBLP fetch(es)", file=sys.stderr)
    print(f"{filled} of {len(rows)} resolved -> {args.out}", file=sys.stderr)
    for resolution, count in resolutions.most_common():
        print(f"    {count:>4}  {resolution}", file=sys.stderr)
    if filled < len(rows):
        print(
            f"\nThe {len(rows) - filled} row(s) with an empty dblp cell are the "
            f"to-do list: paste a DBLP link in, then re-run "
            f"`make reserve-info VERIFY=--verify`. The note column names the "
            f"pages already looked at and why each was rejected.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
