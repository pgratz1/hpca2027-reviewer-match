"""Report conflicts DBLP co-authorship implies but nobody declared.

    python -m scripts.audit_coauthor_conflicts
    python -m scripts.audit_coauthor_conflicts --coauthor-years 3
    python -m scripts.audit_coauthor_conflicts --role reviewer     # skip reserves

A reviewer who has published with one of a paper's authors in the last few years
is conflicted with that paper. HotCRP already knows most of these, because the
authors declared them. This asks DBLP the same question independently and writes
down every answer, so the two can be compared.

The `declared` column is the point of the file. `pc_conflicts` means the sweep
already had it and this is confirmation; `own_paper` means the person is on the
paper themselves. An **empty** `declared` is a conflict nobody recorded — a
reviewer the matcher would otherwise have considered available. Those are what
this exists to surface, and reading a handful of them is the way to judge how
much the declared markings can be trusted.

The confirmed rows are worth reading too, in the other direction: they are the
control group. If rows marked `pc_conflicts` look like real co-authorships then
the matching is working, and the empty ones deserve belief.

Matching is by name and errs towards firing — see `reviewer_match.coauthor_coi`
for why, and for the `exact`/`partial` distinction in the `match` column. Expect
false positives on common names; a wrongly withheld reviewer costs one slot out
of hundreds, a wrongly assigned conflict costs the review.

Offline and read-only: it needs `dblp_coauthors.json`, which `make dblp-snapshot`
writes. Reviewers whose PID is missing from that snapshot have no co-author data
and silently pass every check; the summary counts them, because an empty result
for them is not the same as a clean one.
"""

from __future__ import annotations

from reviewer_match.paths import assignment_path, cache_path, curated_path, input_path, report_path, smoke_cache_path

import argparse
import json
import sys
from collections import defaultdict

from reviewer_match import coauthor_coi, pc_membership
from reviewer_match.coauthor_coi import (
    DEFAULT_AUTHOR_NAMES,
    DEFAULT_COAUTHOR_YEARS,
    DEFAULT_COAUTHORS,
)
from reviewer_match.paper_matching import PAPER_POLICIES, load_papers
from reviewer_match.pc_membership import token_set
from reviewer_match.roster import ROLES, load_roster, role_label
from scripts.build_reserve_reviewer_info import write_csv

DEFAULT_DATA = input_path("hpca2027-data.json")
DEFAULT_OUT = report_path("coauthor_conflicts.csv")

# Rosters that actually review papers. Area chairs are covered by --role.
DEFAULT_ROLES = ("reviewer", "reserve")

FIELDS = [
    "pid", "paper_title", "reviewer_email", "reviewer_name", "role",
    "declared", "match", "author_name", "author_email", "author_affiliation",
    "shared_papers", "latest_year", "latest_shared_title", "reviewers_matched",
]

# A name reaching more reviewers than this is either a superstar everybody has
# written with or a name two different people share. The declared rate tells
# them apart: a superstar's conflicts are mostly already in pc_conflicts.
COLLISION_REVIEWERS = 20
COLLISION_DECLARED = 0.10


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default=DEFAULT_DATA, help="HotCRP submissions JSON")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help=f"where to write the conflicts (default: {DEFAULT_OUT})")
    parser.add_argument("--coauthor-cache", default=DEFAULT_COAUTHORS,
                        help=f"co-author cache from make dblp-snapshot "
                             f"(default: {DEFAULT_COAUTHORS})")
    parser.add_argument("--author-names-cache", default=DEFAULT_AUTHOR_NAMES,
                        help=f"DBLP name spellings of submission authors "
                             f"(default: {DEFAULT_AUTHOR_NAMES})")
    parser.add_argument("--coauthor-years", type=int, default=DEFAULT_COAUTHOR_YEARS,
                        help=f"calendar years of co-authorship that conflict "
                             f"(default: {DEFAULT_COAUTHOR_YEARS})")
    parser.add_argument(
        "--no-coauthor-identity", action="store_true",
        help="ignore DBLP's homonym numbering and treat every spelling of a name "
             "as one person; useful for diffing against the stricter default"
    )
    parser.add_argument("--role", action="append", choices=ROLES, default=None,
                        help=f"roster to check; repeatable "
                             f"(default: {', '.join(DEFAULT_ROLES)})")
    parser.add_argument("--paper-policy", choices=sorted(PAPER_POLICIES), default="registered",
                        help="which submissions count (default: registered)")
    parser.add_argument("--no-pc-check", action="store_true",
                        help="skip the HotCRP PC-membership check on the rosters")
    args = parser.parse_args()

    try:
        coauthors = coauthor_coi.load_coauthors(args.coauthor_cache)
    except FileNotFoundError:
        parser.error(f"{args.coauthor_cache}: not found; run `make dblp-snapshot`")
    try:
        author_names = coauthor_coi.load_author_names(args.author_names_cache)
    except FileNotFoundError:
        parser.error(f"{args.author_names_cache}: not found; run `make dblp-snapshot`")

    pcinfo = None if args.no_pc_check else pc_membership.DEFAULT_PCINFO
    people = []
    role_of: dict[str, str] = {}
    names: dict[str, str] = {}
    for role in (args.role or list(DEFAULT_ROLES)):
        roster = load_roster(role, data_path=args.data, pcinfo_path=pcinfo)
        people += roster
        for person in roster:
            role_of.setdefault(person.email, getattr(person, "tier", "area-chair"))
            names.setdefault(person.email, person.name)
    print(f"{len(people)} person/people across "
          f"{', '.join(role_label(r) for r in (args.role or DEFAULT_ROLES))}",
          file=sys.stderr)

    papers, _ = load_papers(args.data, paper_policy=args.paper_policy, with_skipped=True)
    with open(args.data, encoding="utf-8") as f:
        all_papers = json.load(f)
    papers_by_pid = {p["pid"]: p for p in papers}

    index = coauthor_coi.build_index(people, coauthors, years=args.coauthor_years)
    derived = coauthor_coi.derive_conflicts(
        papers, index, author_names, all_papers,
        use_identity=not args.no_coauthor_identity,
    )

    rows = []
    for pid, found in derived.items():
        title = papers_by_pid[pid].get("title", "")
        for email, coi in found.items():
            rows.append({
                "pid": pid,
                "paper_title": title,
                "reviewer_email": email,
                "reviewer_name": names.get(email, ""),
                "role": role_of.get(email, ""),
                "declared": coi.declared,
                "match": coi.match,
                "author_name": coi.author_name,
                "author_email": coi.author_email,
                "author_affiliation": coi.author_affiliation,
                "shared_papers": coi.shared,
                "latest_year": coi.latest_year,
                "latest_shared_title": coi.latest_title,
            })

    # How many distinct reviewers this author's name reaches across every
    # submission — the handle on name collisions, see COLLISION_REVIEWERS.
    reach: dict[frozenset[str], set[str]] = defaultdict(set)
    for row in rows:
        reach[token_set(row["author_name"])].add(row["reviewer_email"])
    for row in rows:
        row["reviewers_matched"] = len(reach[token_set(row["author_name"])])

    rows.sort(key=lambda r: (r["pid"], r["reviewer_email"]))
    write_csv(args.out, FIELDS, rows)

    report(rows, papers, people, index, reach, args)
    return 0


def report(rows, papers, people, index, reach, args) -> None:
    """Narrate the file to stderr: what is new, and what could not be checked."""
    new = [r for r in rows if not r["declared"]]
    by_declared: dict[str, int] = defaultdict(int)
    by_match: dict[str, int] = defaultdict(int)
    by_role: dict[str, int] = defaultdict(int)
    for row in rows:
        by_declared[row["declared"] or "(not declared)"] += 1
        if not row["declared"]:
            by_match[row["match"]] += 1
            by_role[row["role"]] += 1

    print(f"\nWrote {len(rows)} co-authorship conflict(s) to {args.out}", file=sys.stderr)
    for label, count in sorted(by_declared.items(), key=lambda kv: -kv[1]):
        print(f"    {count:>4}  {label}", file=sys.stderr)

    print(f"\n{len(new)} conflict(s) nobody declared, over "
          f"{len({r['pid'] for r in new})} of {len(papers)} paper(s) and "
          f"{len({r['reviewer_email'] for r in new})} reviewer(s):", file=sys.stderr)
    for label, count in sorted(by_match.items(), key=lambda kv: -kv[1]):
        print(f"    {count:>4}  {label} name match", file=sys.stderr)
    for label, count in sorted(by_role.items(), key=lambda kv: -kv[1]):
        print(f"    {count:>4}  {label}", file=sys.stderr)

    # A paper with no declared conflicts at all has not been swept, so every
    # conflict this finds on it is one the matcher would otherwise have used.
    unswept = {p["pid"] for p in papers if not (p.get("pc_conflicts") or {})}
    hit = unswept & {r["pid"] for r in new}
    if unswept:
        print(f"\n{len(hit)} of the {len(unswept)} paper(s) with no declared "
              f"conflicts at all have one here.", file=sys.stderr)

    # Where the false positives are, if there are any: one name reaching dozens
    # of reviewers that almost nobody declared is two people sharing a name, not
    # somebody everyone has written with.
    # Shares `reach` with the reviewers_matched column, so the count quoted here
    # is the one the reader will sort on.
    declared_rows: dict[frozenset[str], list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        key = token_set(row["author_name"])
        declared_rows[key][0] += 1
        declared_rows[key][1] += bool(row["declared"])
    suspect = {
        key for key, revs in reach.items()
        if len(revs) >= COLLISION_REVIEWERS
        and declared_rows[key][1] / declared_rows[key][0] < COLLISION_DECLARED
    }
    if suspect:
        affected = [r for r in new if token_set(r["author_name"]) in suspect]
        print(f"\n{len(suspect)} author name(s) conflict {COLLISION_REVIEWERS}+ "
              f"reviewers each while under {COLLISION_DECLARED:.0%} of those were "
              f"ever declared — the shape of a shared name rather than a shared "
              f"field, accounting for {len(affected)} of the new conflicts. Sort "
              f"by reviewers_matched to review them; they stay blocked either "
              f"way, since a withheld reviewer costs less than a missed conflict.",
              file=sys.stderr)

    # Always stated, never only on bad news: a reviewer with no co-author data
    # passes silently, so silence must not be what says everyone was checked.
    gap = coauthor_coi.coverage_gap(index, people)
    print(f"\nChecked {len(people) - len(gap)} of {len(people)} reviewer(s) "
          f"against {args.coauthor_cache}.", file=sys.stderr)
    if gap:
        print(f"    {len(gap):>4}  have no co-author data — their PID is missing "
              f"from the snapshot, so they pass this check silently and an empty "
              f"result for them is not a clean one. Re-run `make dblp-snapshot` "
              f"against a newer dump to close the gap.", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
