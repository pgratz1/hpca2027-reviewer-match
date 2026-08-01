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
import sys
from dataclasses import dataclass

from . import pc_membership
from .dblp import parse_pid
from .reviewers import _latest_rows_by_email, field, load_dblp_overrides


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
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    overrides = load_dblp_overrides(overrides_path)
    index = pc_membership.load_pc_accounts(pcinfo_path) if pcinfo_path else None

    chairs = []
    dropped: list[str] = []
    for row in _latest_rows_by_email(rows):
        membership = field(row, "Area Chair membership")
        if not membership.lower().startswith("yes"):
            continue
        email = field(row, "email address").lower()
        first, last = field(row, "First Name"), field(row, "Last Name")
        if index is not None and index.match(email, first, last)[0] is None:
            dropped.append(email)
            continue
        dblp_url = field(row, "DBLP")
        chairs.append(
            AreaChair(
                email=email,
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
    if index is not None and supplement is not None:
        chairs.extend(_tagged_without_a_form_row(index, chairs, supplement))
    return chairs


def _resolved_emails(index, email: str, first: str, last: str) -> set[str]:
    """Every address this person is known by: their own, plus their account's.

    Accepting from one address while holding the HotCRP account under another is
    ordinary here, which is the whole reason `pc_membership` exists. Comparing
    raw addresses alone would file one person as two.
    """
    found = {email.lower()}
    acct, _ = index.match(email, first, last)
    if acct is not None:
        found.add(acct.email.lower())
    return found


def _tagged_without_a_form_row(index, chairs: list[AreaChair], supplement) -> list[AreaChair]:
    """Area chairs HotCRP knows about that the acceptance form does not.

    Their profile is borrowed from whichever roster does have them -- the PC
    acceptance form or the reserve roster -- because a chair with no PID and no
    declared areas cannot be fingerprinted, and so cannot be assigned a paper.
    Anyone tagged who is on no roster at all is named and skipped: emitting a
    record that cannot be fingerprinted would fail later and further away.
    """
    accounted: set[str] = set()
    for chair in chairs:
        accounted |= _resolved_emails(index, chair.email, chair.first, chair.last)

    donors: dict[str, object] = {}
    for record in supplement:
        for email in _resolved_emails(index, record.email, record.first, record.last):
            donors.setdefault(email, record)

    added: list[AreaChair] = []
    unprofiled: list[str] = []
    for acct in index.accounts:
        if not acct.is_area_chair:
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
            emails |= _resolved_emails(index, chair.email, chair.first, chair.last)
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
            known |= _resolved_emails(
                index, email, getattr(person, "first", ""), getattr(person, "last", "")
            )
        if known & chair_emails:
            dropped.append(email)
            del people_by_email[email]
    return sorted(dropped)
