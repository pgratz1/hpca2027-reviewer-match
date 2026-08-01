"""Cross-check every reserve reviewer's DBLP identity against independent evidence.

    python -m scripts.audit_reserve_identities
    python -m scripts.audit_reserve_identities --out /tmp/audit.csv

The DBLP page on file for a reserve reviewer came from the recruiting workbook
or from resolve_reserve_pids.py, and both can be wrong in the same way: a page
belonging to somebody who merely shares the name. A name check cannot catch
that — "Rakesh Kumar" the computer architect and "Rakesh Kumar" the queuing
theorist are spelled identically — so this asks questions a namesake would fail.

Four signals, each from a source that knows nothing about the others:

  coauthors    the people they wrote HotCRP submissions with should appear on
               their DBLP page. Two researchers publishing together is hard to
               fake and does not depend on spelling; this is the strongest
               signal here, and the only one that positively identifies rather
               than merely failing to contradict
  affiliation  the institution HotCRP has for them should be the institution
               DBLP records
  declared     the DBLP link their own submission gives for them should be the
               page on file
  volume       a person invited to review should have a publication record;
               a page with a handful of papers is unlikely to be theirs

A signal only counts when it can actually speak — a DBLP page with no recorded
affiliation neither confirms nor denies. Rows are ranked by how much evidence
contradicts them, and nothing here is a verdict: it is a list of people worth
looking at, most doubtful first.
"""

from __future__ import annotations

from reviewer_match.paths import assignment_path, cache_path, curated_path, input_path, report_path

import argparse
import csv
import json
import sys
from collections import defaultdict

from reviewer_match.dblp import name_tokens, parse_pid
from scripts.resolve_trc_members import (
    affiliation_tokens, author_name, index_self_declared_pids, tokens_compatible,
)

DEFAULT_INFO = report_path("reserve_reviewer_info.csv")
DEFAULT_DATA = input_path("hpca2027-data.json")
DEFAULT_PROFILES = cache_path("dblp_profile_cache.json")
DEFAULT_SENIORITY = report_path("reserve_seniority.csv")
DEFAULT_OUT = report_path("reserve_identity_audit.csv")

# A page with fewer publications than this is thin for someone senior enough to
# be invited onto a review committee. Not damning on its own -- plenty of good
# researchers are early-career -- which is why it only ever adds to a verdict.
THIN_PUBS = 15

OUT_FIELDS = [
    "email", "name", "pid", "verdict", "concerns",
    "dblp_name", "dblp_pubs", "dblp_affiliation", "hotcrp_affiliation",
    "coauthor_evidence", "declared_pids",
]


def hotcrp_coauthors(papers: list[dict]) -> dict[str, set[str]]:
    """{email: {co-author name}} over the submissions each person wrote."""
    index: dict[str, set[str]] = defaultdict(set)
    for paper in papers:
        authors = paper.get("authors") or []
        names = [author_name(a) for a in authors]
        for position, author in enumerate(authors):
            email = (author.get("email") or "").strip().lower()
            if not email:
                continue
            for other, name in enumerate(names):
                if other != position and name:
                    index[email].add(name)
    return index


def hotcrp_affiliations(papers: list[dict]) -> dict[str, str]:
    """{email: affiliation} from author and contact records."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for paper in papers:
        for person in (paper.get("authors") or []) + (paper.get("contacts") or []):
            email = (person.get("email") or "").strip().lower()
            affiliation = (person.get("affiliation") or "").strip()
            if email and affiliation:
                counts[email][affiliation] += 1
    return {
        email: max(by_name.items(), key=lambda kv: kv[1])[0]
        for email, by_name in counts.items()
    }


def matching_coauthors(hotcrp_names: set[str], dblp_names) -> list[str]:
    """Co-authors DBLP and HotCRP agree on, by name tokens."""
    dblp_tokens = [(n, name_tokens(n)) for n in dblp_names]
    hits = []
    for name in sorted(hotcrp_names):
        wanted = name_tokens(name)
        if not wanted:
            continue
        for dblp_name, tokens in dblp_tokens:
            if tokens_compatible(wanted, tokens) == "exact":
                hits.append(dblp_name)
                break
    return hits


def audit_one(
    email: str, name: str, pid: str, profile: dict | None, *,
    hotcrp_affiliation: str, coauthor_names: set[str], declared: dict,
    career_total: int | None, thin_pubs: int,
) -> dict:
    """Everything known about one reserve reviewer's identity, and the doubts."""
    concerns: list[str] = []
    row = {
        "email": email, "name": name, "pid": pid or "",
        "dblp_name": "", "dblp_pubs": "", "dblp_affiliation": "",
        "hotcrp_affiliation": hotcrp_affiliation, "coauthor_evidence": "",
        "declared_pids": ", ".join(sorted(declared.get(email) or ())),
    }
    if profile is None:
        row["verdict"] = "no-profile"
        row["concerns"] = "no DBLP record cached for this PID"
        return row

    dblp_names = profile.get("names") or []
    row["dblp_name"] = dblp_names[0] if dblp_names else ""
    row["dblp_pubs"] = profile.get("pubs", "")
    affiliations = profile.get("affiliations") or []
    row["dblp_affiliation"] = "; ".join(affiliations)

    # 1. Co-authors — the one signal that can positively confirm.
    shared = matching_coauthors(coauthor_names, (profile.get("coauthors") or {}).values())
    if shared:
        row["coauthor_evidence"] = ", ".join(shared[:4])
    elif coauthor_names:
        concerns.append(
            f"none of their {len(coauthor_names)} HotCRP co-author(s) appear on this DBLP page"
        )

    # 2. Affiliation, when DBLP records one.
    if affiliations and hotcrp_affiliation:
        wanted = affiliation_tokens(hotcrp_affiliation)
        got = affiliation_tokens(" ".join(affiliations))
        if wanted and got and not (wanted & got):
            concerns.append("HotCRP and DBLP name different institutions")

    # 3. What their own submission says their page is.
    declared_pids = declared.get(email) or {}
    if declared_pids and pid not in declared_pids:
        concerns.append(
            f"their own submission declares {', '.join(sorted(declared_pids))}, not this page"
        )

    # 4. Volume.
    pubs = profile.get("pubs")
    if isinstance(pubs, int) and pubs < thin_pubs:
        concerns.append(f"only {pubs} publication(s) on this page")
    if career_total == 0 and isinstance(pubs, int) and pubs < thin_pubs:
        concerns.append("and none in ISCA/MICRO/HPCA/ASPLOS")

    row["concerns"] = "; ".join(concerns)
    if shared:
        # Co-authorship outweighs the rest: it is positive proof of identity,
        # where every other signal here can only fail to contradict.
        row["verdict"] = "confirmed" if not concerns else "confirmed-with-notes"
    elif not concerns:
        row["verdict"] = "unconfirmed"
    elif len(concerns) >= 2:
        row["verdict"] = "doubtful"
    else:
        row["verdict"] = "check"
    return row


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--info", default=DEFAULT_INFO)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--profiles", default=DEFAULT_PROFILES)
    parser.add_argument("--seniority", default=DEFAULT_SENIORITY)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--thin-pubs", type=int, default=THIN_PUBS,
                        help=f"publication count below which a page is thin (default: {THIN_PUBS})")
    args = parser.parse_args()

    with open(args.data, encoding="utf-8") as f:
        papers = json.load(f)
    with open(args.profiles, encoding="utf-8") as f:
        profiles = json.load(f)
    with open(args.info, newline="", encoding="utf-8-sig") as f:
        roster = list(csv.DictReader(f))

    career = {}
    try:
        with open(args.seniority, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    career[r["email"]] = int(r["career_total"] or 0)
                except ValueError:
                    pass
    except FileNotFoundError:
        print(f"{args.seniority}: not found; skipping the venue signal", file=sys.stderr)

    coauthors = hotcrp_coauthors(papers)
    affiliations = hotcrp_affiliations(papers)
    declared = index_self_declared_pids(papers)

    rows = []
    for entry in roster:
        email = (entry.get("email") or "").strip().lower()
        if not email:
            continue
        pid = parse_pid(entry.get("dblp") or "")
        rows.append(audit_one(
            email, (entry.get("name") or "").strip(), pid, profiles.get(pid),
            hotcrp_affiliation=affiliations.get(email, ""),
            coauthor_names=coauthors.get(email, set()),
            declared=declared, career_total=career.get(email),
            thin_pubs=args.thin_pubs,
        ))

    order = {"doubtful": 0, "check": 1, "no-profile": 2, "unconfirmed": 3,
             "confirmed-with-notes": 4, "confirmed": 5}
    rows.sort(key=lambda r: (order.get(r["verdict"], 9), -len(r["concerns"]), r["email"]))
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    tally: dict[str, int] = defaultdict(int)
    for r in rows:
        tally[r["verdict"]] += 1
    print(f"{len(rows)} reserve reviewer(s) audited -> {args.out}", file=sys.stderr)
    for verdict in sorted(tally, key=lambda v: order.get(v, 9)):
        print(f"    {tally[verdict]:>4}  {verdict}", file=sys.stderr)
    print(
        "\nconfirmed = a HotCRP co-author of theirs appears on the DBLP page, which "
        "is positive proof. unconfirmed = nothing contradicts it, but nothing "
        "corroborates it either. doubtful/check are worth opening by hand.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
