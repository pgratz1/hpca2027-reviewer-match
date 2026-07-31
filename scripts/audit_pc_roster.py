"""Cross-check the reviewer rosters against the HotCRP user export.

    python -m scripts.audit_pc_roster
    python -m scripts.audit_pc_roster --pcinfo data/inputs/hpca2027-pcinfo.csv
    python -m scripts.audit_pc_roster --pruned /tmp/pc_roster_pruned.csv \
      --missing /tmp/pc_roster_missing.csv

Two lists have to agree and drift apart constantly: who accepted an invitation
(the acceptance forms and the reserve upload) and who is on the committee in
HotCRP today (`hpca2027-pcinfo.csv`). This reports both directions of the
disagreement. It decides nothing and changes nothing -- the loaders in
`pc_membership` apply the rule; this explains what that rule did and what it
could not explain.

**Who the loaders drop** -> `pc_roster_pruned.csv`, one row per person, with
`problem` saying why the export does not have them on the PC:

  no_account    -- no HotCRP account under this address, and no PC account is
                   the same person by name or email local part. The real
                   "removed from HotCRP" case.
  role_removed  -- the account is still there but the `pc` role is gone. The
                   `detail` column keeps whatever tags survived, which is what
                   separates a deliberate stand-down (the reviewer tag left
                   behind) from an account that was wiped.
  disabled      -- on the PC but the account is disabled, so they cannot log in
                   to review.

Note what is *not* here: someone who accepted from one address and holds their
HotCRP account under another is not pruned and does not appear. That is not a
nicety -- on the current export it is twelve of the sixteen rows whose own
address is not `pc`, so an email-only rule would remove twelve sitting PC
members. They surface in the other file, under `alternate_account`, because the
useful fact about them is the HotCRP merge nobody has done yet.

**PC accounts no roster explains** -> `pc_roster_missing.csv`, one row per
HotCRP account marked `pc` that appears in neither acceptance form nor the
reserve upload, with `category`:

  chair, trc, sysadmin, disabled, area-chair
                    -- on the PC for a structural reason. Settled; nothing to do.
  alternate_account -- the person is on a roster under a different address.
                       Merge the two accounts in HotCRP.
  declined          -- their latest form response says they cannot serve, and
                       they are on the PC anyway. The sharpest anomaly here.
  no_roster_row     -- on the PC with no acceptance and no reserve-upload row
                       at all. Check by hand.

Both files are always written, even when empty, and sorted by email, so a re-run
against a fresh export diffs cleanly instead of going stale.
"""

from __future__ import annotations

from reviewer_match.paths import assignment_path, cache_path, curated_path, input_path, report_path, smoke_cache_path

import argparse
import csv
import sys
from collections import Counter

from reviewer_match import pc_membership
from reviewer_match import reserve_reviewers
from reviewer_match import roster
from reviewer_match.area_chairs import load_area_chairs
from scripts.build_reserve_reviewer_info import DEFAULT_UPLOAD, write_csv
from reviewer_match.pc_membership import local_part, on_pc, token_set
from reviewer_match.reviewers import _latest_rows_by_email, field, load_reviewers

DEFAULT_PRUNED = report_path("pc_roster_pruned.csv")
DEFAULT_MISSING = report_path("pc_roster_missing.csv")

PRUNED_FIELDS = ["email", "name", "tier", "source", "problem", "detail"]
MISSING_FIELDS = ["email", "name", "affiliation", "roles", "tags", "category", "detail"]

# Categories that explain themselves. Everything else is work for a human.
SETTLED = frozenset({"chair", "trc", "sysadmin", "disabled", "area-chair"})


def declined_emails(csv_path: str) -> set[str]:
    """Emails whose latest acceptance-form response says they cannot serve.

    Read straight from the form rather than by differencing the roster, because
    `load_reviewers` cannot distinguish "declined" from "never answered" and the
    two mean very different things about an account still marked `pc`.
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {
        field(row, "email address").lower()
        for row in _latest_rows_by_email(rows)
        if "unable" in field(row, "PC membership").lower()
    }


def upload_emails(path: str) -> set[str]:
    """Emails in the HotCRP reserve upload.

    The upload is the authoritative "we recruited this person" list, and it is
    what the reverse check needs. `reserve_reviewer_info.csv` is the wrong file
    here: it has already lost rows to unresolved DBLP identities, and every one
    of those would show up as an anomaly it is not.
    """
    try:
        f = open(path, newline="", encoding="utf-8-sig")
    except FileNotFoundError:
        print(f"Warning: {path} not found; reserve recruits will look unexplained",
              file=sys.stderr)
        return set()
    with f:
        return {(row.get("email") or "").strip().lower() for row in csv.DictReader(f)} - {""}


def prune_problem(index: pc_membership.PcIndex, email: str) -> tuple[str, str]:
    """(problem, detail) for a roster row the membership check dropped."""
    acct = index.by_email.get(email)
    if acct is None:
        return "no_account", "no HotCRP account under this address"
    if acct.disabled:
        return "disabled", f"account disabled; roles: {acct.roles or '(none)'}"
    return "role_removed", f"account kept, pc role gone; tags: {acct.tags or '(none)'}"


def collect_pruned(index, form_csv, ac_csv, reserve_info, data_path) -> list[dict[str, str]]:
    """Every roster row the loaders drop, across all three rosters.

    Each roster is loaded with the check switched off, then the same predicate
    the loaders use is applied here -- otherwise this could only report the
    people who survived it.
    """
    rosters = (
        ("acceptance-form", load_reviewers(form_csv, pcinfo_path=None)),
        ("area-chair-form", load_area_chairs(ac_csv, pcinfo_path=None)),
        ("reserve-roster",
         reserve_reviewers.load_reserve_reviewers(reserve_info, data_path, pcinfo_path=None)),
    )
    pruned = []
    for source, people in rosters:
        for person in people:
            if index.match(person.email, person.first, person.last)[0] is not None:
                continue
            problem, detail = prune_problem(index, person.email)
            pruned.append({
                "email": person.email,
                "name": person.name if person.name != person.email else "",
                "tier": getattr(person, "tier", "area-chair"),
                "source": source,
                "problem": problem,
                "detail": detail,
            })
    return sorted(pruned, key=lambda r: r["email"])


def roster_identity_index(rosters) -> tuple[dict, dict]:
    """Name-token and local-part indexes over everyone on a roster.

    The mirror image of `PcIndex`'s indexes: that one asks "is this roster row a
    PC account", this one asks "is this PC account on some roster".
    """
    by_tokens: dict[frozenset[str], list[tuple[str, str]]] = {}
    by_local: dict[str, list[tuple[str, str]]] = {}
    for source, entries in rosters:
        for email, name in entries:
            tokens = token_set(name)
            if tokens:
                by_tokens.setdefault(tokens, []).append((email, source))
            local = local_part(email)
            if local:
                by_local.setdefault(local, []).append((email, source))
    return by_tokens, by_local


def missing_category(acct, declined, by_tokens, by_local) -> tuple[str, str]:
    """(category, detail) for a PC account no roster accounts for.

    Structural roles are tested before identity facts: a chair holding a second
    account is still on the PC because they are the chair, and filing them under
    `alternate_account` would put a settled row on the to-do list.
    """
    tags = acct.tags.split()
    if "chairs" in tags or "chair" in acct.roles.split():
        return "chair", "on the PC as a chair"
    if any(t.startswith("trc") for t in tags):
        return "trc", "on the PC for the TRC"
    if "sysadmin" in acct.roles.split():
        return "sysadmin", "administrative account"
    if acct.disabled:
        return "disabled", "account is disabled"
    if any(t.endswith("area-chairs") for t in tags):
        return "area-chair", "tagged as an area chair"

    hits = by_tokens.get(acct.tokens) or by_local.get(acct.local) or []
    hits = [(e, s) for e, s in hits if e != acct.email.lower()]
    if hits:
        email, source = sorted(hits)[0]
        return "alternate_account", f"same person as {email} on the {source}"

    if acct.email.lower() in declined:
        return "declined", "their latest form response says they cannot serve"
    return "no_roster_row", "no acceptance and no reserve-upload row"


def collect_missing(index, form_csv, ac_csv, upload_path, data_path,
                    reserve_info) -> list[dict[str, str]]:
    """Every PC account that neither acceptance form nor the reserve upload explains."""
    accepted = load_reviewers(form_csv, pcinfo_path=None)
    chairs = load_area_chairs(ac_csv, pcinfo_path=None)
    reserves = reserve_reviewers.load_reserve_reviewers(
        reserve_info, data_path, pcinfo_path=None
    )
    upload = upload_emails(upload_path)

    known = {r.email for r in accepted} | {c.email for c in chairs} | upload
    by_tokens, by_local = roster_identity_index((
        ("acceptance form", [(r.email, r.name) for r in accepted]),
        ("area-chair form", [(c.email, c.name) for c in chairs]),
        ("reserve roster", [(r.email, r.name) for r in reserves]),
    ))
    declined = declined_emails(form_csv)

    missing = []
    for acct in index.pc_accounts:
        if acct.email.lower() in known:
            continue
        category, detail = missing_category(acct, declined, by_tokens, by_local)
        missing.append({
            "email": acct.email, "name": acct.name, "affiliation": acct.affiliation,
            "roles": acct.roles, "tags": acct.tags,
            "category": category, "detail": detail,
        })
    return sorted(missing, key=lambda r: r["email"].lower())


def report(pruned: list[dict], missing: list[dict], pruned_path: str,
           missing_path: str) -> None:
    """Summarise both directions, settled rows kept apart from real work."""
    print(f"\n{len(pruned)} roster {'row' if len(pruned) == 1 else 'rows'} dropped "
          f"by the membership check -> {pruned_path}", file=sys.stderr)
    for problem, count in Counter(r["problem"] for r in pruned).most_common():
        print(f"    {count:>4}  {problem}", file=sys.stderr)
    if pruned:
        # These are decisions the chair already made in HotCRP. Nothing to do
        # here -- which is why they are counted apart from the file below.
        print("\nThose are removals someone made in HotCRP on purpose; the "
              "pipeline has now caught up with them.", file=sys.stderr)

    print(f"\n{len(missing)} PC {'account' if len(missing) == 1 else 'accounts'} "
          f"with no acceptance and no reserve-upload row -> {missing_path}",
          file=sys.stderr)
    for category, count in Counter(r["category"] for r in missing).most_common():
        print(f"    {count:>4}  {category}", file=sys.stderr)

    settled = [r for r in missing if r["category"] in SETTLED]
    if settled:
        one = len(settled) == 1
        print(f"\n{len(settled)} of those {'is' if one else 'are'} a chair, TRC "
              f"member, area chair or disabled account and {'needs' if one else 'need'} "
              f"nothing.", file=sys.stderr)

    alternates = [r for r in missing if r["category"] == "alternate_account"]
    if alternates:
        one = len(alternates) == 1
        print(f"\n{len(alternates)} hold{'s' if one else ''} a second HotCRP account "
              f"that a roster does use. Merge each pair in HotCRP so their conflicts "
              f"and review load stop being split across two identities, then "
              f"re-export. `make duplicates` lists them pair by pair.", file=sys.stderr)

    declined = [r for r in missing if r["category"] == "declined"]
    if declined:
        one = len(declined) == 1
        print(f"\n{len(declined)} said on the form that {'they cannot' if one else 'they cannot'} "
              f"serve and {'is' if one else 'are'} still marked pc in HotCRP. Remove the "
              f"role, or the matcher will keep counting on {'them' if one else 'them'}.",
              file=sys.stderr)

    orphans = [r for r in missing if r["category"] == "no_roster_row"]
    if orphans:
        one = len(orphans) == 1
        print(f"\nThe other {len(orphans)} {'is' if one else 'are'} on the PC in HotCRP "
              f"with no acceptance and no reserve-upload row at all: either they never "
              f"answered the form, or their response is filed under an address nothing "
              f"here recognises. Check {'it' if one else 'each'} by hand.", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--pcinfo", default=pc_membership.DEFAULT_PCINFO,
        help=f"HotCRP user export (default: {pc_membership.DEFAULT_PCINFO})",
    )
    parser.add_argument(
        "--csv", default=roster.DEFAULT_CSV, help="PC member acceptance form CSV"
    )
    parser.add_argument(
        "--area-chair-csv", default=roster.DEFAULT_AREA_CHAIR_CSV,
        help="area-chair acceptance form CSV",
    )
    parser.add_argument(
        "--reserve-info", default=reserve_reviewers.DEFAULT_INFO,
        help=f"reserve roster (default: {reserve_reviewers.DEFAULT_INFO})",
    )
    parser.add_argument(
        "--upload", default=DEFAULT_UPLOAD,
        help=f"HotCRP reserve-reviewer upload (default: {DEFAULT_UPLOAD})",
    )
    parser.add_argument(
        "--data", default=reserve_reviewers.DEFAULT_DATA,
        help=f"submissions JSON, for reserve areas (default: {reserve_reviewers.DEFAULT_DATA})",
    )
    parser.add_argument(
        "--pruned", default=DEFAULT_PRUNED,
        help=f"where to write the dropped roster rows (default: {DEFAULT_PRUNED})",
    )
    parser.add_argument(
        "--missing", default=DEFAULT_MISSING,
        help=f"where to write the unexplained PC accounts (default: {DEFAULT_MISSING})",
    )
    args = parser.parse_args()

    try:
        index = pc_membership.load_pc_accounts(args.pcinfo)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    print(f"{len(index.pc_accounts)} PC accounts in {args.pcinfo} "
          f"({len(index.accounts)} accounts total)", file=sys.stderr)

    pruned = collect_pruned(index, args.csv, args.area_chair_csv,
                            args.reserve_info, args.data)
    missing = collect_missing(index, args.csv, args.area_chair_csv, args.upload,
                              args.data, args.reserve_info)

    write_csv(args.pruned, PRUNED_FIELDS, pruned)
    write_csv(args.missing, MISSING_FIELDS, missing)
    report(pruned, missing, args.pruned, args.missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
