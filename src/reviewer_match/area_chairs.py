"""Parse the HPCA area-chair acceptance form into research-profile records.

Who is an area chair has two sources of truth, and on the current data **each
one catches somebody the other misses**: 19 accounts carry HotCRP's
`~~area-chairs` tag, 18 people accepted on the form, and the union is 20. So
neither source alone is sufficient, and both are read.

The two are used asymmetrically, because the error directions differ:

* The **exclusion set** -- who may not be assigned papers to review -- is the
  union (`area_chair_emails`). Over-excluding costs one reviewer slot out of
  thousands of spare ones; under-excluding breaks the rule that an area chair
  reviews nothing.
* The **chair roster** -- who gets papers to chair -- is the form, supplemented
  by tagged accounts whose research profile can be sourced from another roster
  they are on. A chair needs a DBLP identity and declared areas before they can
  be fingerprinted at all, and the form is the only place those are collected
  first-hand.
"""

from __future__ import annotations

from .paths import assignment_path, cache_path, curated_path, input_path, report_path

import csv
import datetime
import sys
from dataclasses import dataclass

from . import pc_membership
from .dblp import parse_pid
from .reviewers import TIMESTAMP_FORMAT, _latest_rows_by_email, field, load_dblp_overrides


@dataclass
class AreaChair:
    email: str
    first: str
    last: str
    dblp_url: str
    pid: str | None
    affiliation: str
    primary: str
    secondary: str
    tertiary: str
    keywords: str
    pid_from_override: bool = False
    # See Reviewer.hotcrp_email in reviewers.py: the address a HotCRP bulk
    # upload must use, defaulting to `email`.
    hotcrp_email: str = ""

    def __post_init__(self) -> None:
        if not self.hotcrp_email:
            self.hotcrp_email = self.email

    @property
    def name(self) -> str:
        full = f"{self.first} {self.last}".strip()
        return full or self.email


def load_area_chairs(
    csv_path: str,
    overrides_path: str = curated_path("dblp_overrides.csv"),
    *,
    pcinfo_path: str | None = pc_membership.DEFAULT_PCINFO,
    supplement=None,
) -> list[AreaChair]:
    """Load the latest explicitly accepted area-chair responses.

    Area chairs go through the same HotCRP membership check as everyone else.
    Every one of them is on the PC today, so it changes nothing — but the check
    living in one loader and not another is how the scripts would come to
    disagree about who is on the committee. `pcinfo_path=None` skips it.

    `supplement` is an iterable of roster records (PC or reserve) that may
    donate a research profile to an account HotCRP tags `~~area-chairs` but that
    never returned the form. Without it those people would chair nothing while
    also being barred from reviewing, so they would do neither job. Supplementing
    needs the export, so it is skipped along with the membership check when
    `pcinfo_path` is None -- which keeps `audit_pc_roster.py`'s un-gated view of
    the form exactly as it was.

    A PC chair is never an area chair, even if they accepted the area-chair
    form -- they already make the final call on the same papers, so chairing
    one is a conflict, not a second job. `pc_membership.structural_role`
    already recognises a PC chair from HotCRP's own `roles`/`tags` columns
    (the same check that keeps a PC chair from being auto-added as a plain
    reviewer in `reviewers._pc_members_without_a_form_row`); applying it here
    too needs no hardcoded name or email and follows the role automatically
    if it ever changes.
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    overrides = load_dblp_overrides(overrides_path)
    index = pc_membership.load_pc_accounts(pcinfo_path) if pcinfo_path else None

    chairs = []
    dropped: list[str] = []
    chair_conflicts: list[str] = []
    timestamps: dict[str, datetime.datetime] = {}
    for row in _latest_rows_by_email(rows):
        membership = field(row, "Area Chair membership")
        if not membership.lower().startswith("yes"):
            continue
        email = field(row, "email address").lower()
        timestamps[email] = datetime.datetime.strptime(row["Timestamp"].strip(), TIMESTAMP_FORMAT)
        first, last = field(row, "First Name"), field(row, "Last Name")
        hotcrp_email = email
        if index is not None:
            acct, how = index.match(email, first, last)
            if acct is None:
                dropped.append(email)
                continue
            if pc_membership.structural_role(acct) == "chair":
                chair_conflicts.append(email)
                continue
            hotcrp_email = pc_membership.hotcrp_email_for(email, acct, how)
        dblp_url = field(row, "DBLP")
        chairs.append(
            AreaChair(
                email=email,
                hotcrp_email=hotcrp_email,
                first=first,
                last=last,
                dblp_url=dblp_url,
                pid=overrides.get(email) or parse_pid(dblp_url),
                affiliation=field(row, "institutional affiliation"),
                primary=field(row, "primary area"),
                secondary=field(row, "secondary area"),
                tertiary="",
                keywords=field(row, "keywords"),
                pid_from_override=email in overrides,
            )
        )

    if index is not None:
        pc_membership.report_pruned(dropped, len(chairs), "area chairs", index)
    if chair_conflicts:
        print(
            f"{len(chair_conflicts)} area-chair acceptance(s) skipped — already a "
            f"PC chair, so chairing would conflict with the same papers: "
            f"{', '.join(sorted(chair_conflicts))}",
            file=sys.stderr,
        )

    # Same reasoning as reviewers.py: a chair who resubmitted under a second
    # real address is two rows here for one HotCRP account, and the later
    # submission wins. Done before the tag-supplement extend below, which has
    # its own dedup against `chairs` and would otherwise be checked against
    # a chair who is about to be collapsed away.
    chairs, collapsed = pc_membership.collapse_by_hotcrp_email(chairs, timestamps)
    if collapsed:
        shown = ", ".join(f"{lost} -> {kept}" for kept, lost in sorted(collapsed)[:5])
        more = f", +{len(collapsed) - 5} more" if len(collapsed) > 5 else ""
        print(
            f"{len(collapsed)} area-chair submission(s) under a second address "
            f"collapsed into the same HotCRP account as another row: {shown}{more}",
            file=sys.stderr,
        )

    if index is not None and supplement is not None:
        chairs.extend(_tagged_without_a_form_row(index, chairs, supplement))
    return chairs


def _tagged_without_a_form_row(index, chairs: list[AreaChair], supplement) -> list[AreaChair]:
    """Area chairs HotCRP knows about that the acceptance form does not.

    Their profile is borrowed from whichever roster does have them -- the PC
    acceptance form or the reserve roster -- because a chair with no PID and no
    declared areas cannot be fingerprinted, and so cannot be assigned a paper.
    Anyone tagged who is on no roster at all is named and skipped: emitting a
    record that cannot be fingerprinted would fail later and further away.

    A PC chair is skipped here too, same as the form-row loop in
    `load_area_chairs` -- future-proofing against a PC chair's account someday
    also picking up the `~~area-chairs` tag in HotCRP.
    """
    accounted: set[str] = set()
    for chair in chairs:
        accounted |= pc_membership.resolved_emails(index, chair.email, chair.first, chair.last)

    donors: dict[str, object] = {}
    for record in supplement:
        for email in pc_membership.resolved_emails(index, record.email, record.first, record.last):
            donors.setdefault(email, record)

    added: list[AreaChair] = []
    unprofiled: list[str] = []
    chair_conflicts: list[str] = []
    for acct in index.accounts:
        if not acct.is_area_chair:
            continue
        if pc_membership.structural_role(acct) == "chair":
            chair_conflicts.append(acct.email.lower())
            continue
        email = acct.email.lower()
        if email in accounted:
            continue
        donor = donors.get(email)
        if donor is None or not donor.pid:
            unprofiled.append(email)
            continue
        accounted.add(email)
        accounted.add(donor.email.lower())
        added.append(
            AreaChair(
                email=donor.email,
                hotcrp_email=acct.email,
                first=donor.first or acct.given,
                last=donor.last or acct.family,
                dblp_url=donor.dblp_url,
                pid=donor.pid,
                affiliation=donor.affiliation or acct.affiliation,
                primary=donor.primary,
                secondary=donor.secondary,
                tertiary=getattr(donor, "tertiary", ""),
                keywords=donor.keywords,
                pid_from_override=donor.pid_from_override,
            )
        )

    if added:
        print(f"{len(added)} area chair(s) tagged in HotCRP with no accepted form row; "
              f"profile taken from the roster that has them: "
              f"{', '.join(sorted(c.email for c in added))}", file=sys.stderr)
    if unprofiled:
        print(f"WARNING: {len(unprofiled)} account(s) tagged `~~area-chairs` are on no "
              f"roster, so they have no DBLP page or areas and cannot chair anything: "
              f"{', '.join(sorted(unprofiled))} — add them to a roster or remove the tag",
              file=sys.stderr)
    if chair_conflicts:
        print(
            f"{len(chair_conflicts)} account(s) tagged `~~area-chairs` skipped — "
            f"already a PC chair, so chairing would conflict with the same papers: "
            f"{', '.join(sorted(chair_conflicts))}",
            file=sys.stderr,
        )
    return added


def area_chair_emails(
    csv_path: str,
    overrides_path: str = curated_path("dblp_overrides.csv"),
    *,
    pcinfo_path: str | None = pc_membership.DEFAULT_PCINFO,
) -> set[str]:
    """Every address belonging to an area chair, from both sources of truth.

    The form half is read with `pcinfo_path=None` deliberately. For a *roster*
    the membership gate is protective: a false prune drops someone who is no
    longer on the committee. For an *exclusion set* the safe direction inverts —
    a false prune silently readmits an area chair to the reviewer pool, which is
    the failure this whole check exists to prevent. So the gate is not applied
    here, and the tag half stands on its own regardless.
    """
    chairs = load_area_chairs(csv_path, overrides_path, pcinfo_path=None)
    emails = {c.email.lower() for c in chairs}
    if pcinfo_path:
        index = pc_membership.load_pc_accounts(pcinfo_path)
        emails |= {a.email.lower() for a in index.accounts if a.is_area_chair}
        for chair in chairs:
            emails |= pc_membership.resolved_emails(index, chair.email, chair.first, chair.last)
    return emails


def drop_area_chairs(people_by_email: dict, chair_emails: set[str], index=None) -> list[str]:
    """Remove every area chair from a reviewer pool, in place.

    Returns the dropped addresses, sorted. Takes a plain dict rather than
    reaching into the assignment so the rule stays testable on its own, and so
    it applies identically to PC members and reserves — one of today's five is a
    reserve, and reserves are merged into the pool *after* the PC roster is
    built.
    """
    dropped = []
    for email, person in list(people_by_email.items()):
        known = {email.lower()}
        if index is not None:
            known |= pc_membership.resolved_emails(
                index, email, getattr(person, "first", ""), getattr(person, "last", "")
            )
        if known & chair_emails:
            dropped.append(email)
            del people_by_email[email]
    return sorted(dropped)
