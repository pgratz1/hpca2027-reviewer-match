"""Report conflicts HotCRP's declared collaborators/affiliations imply but pc_conflicts missed.

    python -m scripts.audit_collaborator_conflicts
    python -m scripts.audit_collaborator_conflicts --role reviewer     # skip reserves

A reviewer whose declared `collaborators` (data/inputs/hpca2027-pcinfo.csv) names
one of a paper's authors, or who is named as a collaborator by an author who
declared one, is excluded from that paper by `assign_reviewers.py` and
`assign_area_chairs.py` -- the same signal HotCRP's own Assignments-upload "may
conflict" warning checks, reproduced here so it is caught before assignment
rather than read as a warning on a 6,000-row bulk upload.

A third, wider signal -- the reviewer's affiliation overlapping an author's --
is also HotCRP's, but is **not excluded**: measured on this data it touched the
majority of both reviewers and papers, dominated by generic words ("hong",
"california", "computing"), with no frequency cutoff able to separate those
from a genuinely specific match. It is reported here, `kind` = `affiliation`,
for a human to skim, not applied automatically. See
`reviewer_match.collaborator_coi`'s module docstring for the full reasoning.

The `declared` column is the point of the file, the same as
`audit_coauthor_conflicts.py`'s: `pc_conflicts` means the sweep already had it,
`own_paper` means the person is on the paper themselves, and an **empty**
`declared` is a conflict nobody recorded.

Offline and read-only: it needs only `data/inputs/hpca2027-pcinfo.csv`, already
required by every roster loader.
"""

from __future__ import annotations

from reviewer_match.paths import assignment_path, cache_path, curated_path, input_path, report_path

import argparse
import sys
from collections import defaultdict

from reviewer_match import collaborator_coi, pc_membership
from reviewer_match.paper_matching import PAPER_POLICIES, load_papers
from reviewer_match.roster import ROLES, load_roster, role_label
from scripts.build_reserve_reviewer_info import write_csv

DEFAULT_DATA = input_path("hpca2027-data.json")
DEFAULT_OUT = report_path("collaborator_conflicts.csv")

# Rosters that actually review papers. Area chairs are covered by --role.
DEFAULT_ROLES = ("reviewer", "reserve")

FIELDS = [
    "pid", "paper_title", "reviewer_email", "reviewer_name", "role",
    "declared", "kind", "direction", "evidence",
    "author_name", "author_email", "author_affiliation",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default=DEFAULT_DATA, help="HotCRP submissions JSON")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help=f"where to write the conflicts (default: {DEFAULT_OUT})")
    parser.add_argument("--pcinfo", default=pc_membership.DEFAULT_PCINFO,
                        help=f"HotCRP user export, source of collaborators/affiliation "
                             f"(default: {pc_membership.DEFAULT_PCINFO})")
    parser.add_argument("--role", action="append", choices=ROLES, default=None,
                        help=f"roster to check; repeatable "
                             f"(default: {', '.join(DEFAULT_ROLES)})")
    parser.add_argument("--paper-policy", choices=sorted(PAPER_POLICIES), default="registered",
                        help="which submissions count (default: registered)")
    parser.add_argument("--no-pc-check", action="store_true",
                        help="skip the HotCRP PC-membership check on the rosters")
    args = parser.parse_args()

    try:
        pcinfo_index = pc_membership.load_pc_accounts(args.pcinfo)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    pcinfo = None if args.no_pc_check else args.pcinfo
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
    papers_by_pid = {p["pid"]: p for p in papers}

    profiles = collaborator_coi.build_index(people, pcinfo_index)
    derived = collaborator_coi.derive_conflicts(papers, profiles, pcinfo_index)

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
                "kind": coi.kind,
                "direction": coi.direction,
                "evidence": coi.evidence,
                "author_name": coi.author_name,
                "author_email": coi.author_email,
                "author_affiliation": coi.author_affiliation,
            })

    rows.sort(key=lambda r: (r["pid"], r["reviewer_email"]))
    write_csv(args.out, FIELDS, rows)
    report(rows, papers, args)
    return 0


def report(rows, papers, args) -> None:
    """Narrate the file to stderr: what is new, and how it splits by signal."""
    new = [r for r in rows if not r["declared"]]
    by_declared: dict[str, int] = defaultdict(int)
    by_kind: dict[str, int] = defaultdict(int)
    for row in rows:
        by_declared[row["declared"] or "(not declared)"] += 1
        if not row["declared"]:
            by_kind[row["kind"]] += 1

    name_rows = [r for r in rows if r["kind"] == "name"]
    print(f"\nWrote {len(rows)} declared-collaborator conflict(s) to {args.out} "
          f"({len(name_rows)} excluded by name match, {len(rows) - len(name_rows)} "
          f"affiliation overlap, reported only)", file=sys.stderr)
    for label, count in sorted(by_declared.items(), key=lambda kv: -kv[1]):
        print(f"    {count:>4}  {label}", file=sys.stderr)

    print(f"\n{len(new)} conflict(s) nobody declared in pc_conflicts, over "
          f"{len({r['pid'] for r in new})} of {len(papers)} paper(s) and "
          f"{len({r['reviewer_email'] for r in new})} reviewer(s):", file=sys.stderr)
    for label, count in sorted(by_kind.items(), key=lambda kv: -kv[1]):
        print(f"    {count:>4}  {label}", file=sys.stderr)

    affiliation_new = [r for r in new if r["kind"] == "affiliation"]
    if affiliation_new:
        print(f"\n{len(affiliation_new)} of those are affiliation-only matches -- "
              f"not excluded from assignment, deliberately broad (see "
              f"reviewer_match.collaborator_coi). Worth skimming the `evidence` "
              f"column for ones too generic to mean anything.", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
