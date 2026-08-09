"""Parse the TRC (Training Review Committee) acceptance CSV into Reviewer records.

The TRC is a cohort of PhD students reviewing alongside the PC. Their
acceptance form is structurally close to the PC's -- read with `reviewers.field`,
the same substring-header-match helper, and turned into the same `Reviewer`
shape (as a `TrcMember` subclass carrying one extra field, the declared
advisor email(s)) so every downstream consumer that already takes a Reviewer
-- enrich_publications.py, build_fingerprints.py, fingerprint.py,
coauthor_coi.py, collaborator_coi.py -- takes a TRC member unchanged.

Unlike the PC roster, **this file is the membership authority by itself**.
TRC HotCRP accounts carry `tags=trc` but not the `pc` role (`roles=""`), so
`pc_membership`'s prune-only check -- built entirely around `"pc" in
roles.split()` -- must not be applied here the way it is applied to the PC
and reserve rosters: doing so would drop every TRC member. The acceptance
form is instead cross-checked against the export non-fatally (see
`_cross_check_membership`), which only ever warns.
"""

from __future__ import annotations

from .paths import curated_path, input_path

import csv
import datetime
import sys
from dataclasses import dataclass, field as dataclass_field

from . import pc_membership
from .dblp import parse_pid
from .reviewers import Reviewer, field, load_dblp_overrides

TIMESTAMP_FORMAT = "%m/%d/%Y %H:%M:%S"

DEFAULT_TRC_CSV = input_path(
    "HPCA'27 TRC Member Acceptance Form (Responses) - Form Responses 1.csv"
)

# Same email,dblp,note idiom as dblp_overrides.csv -- a hand-fix path for a
# wrong or missing DBLP link, never a required to-do: a TRC member with no
# DBLP link is matched on declared area/keywords alone (see build_fingerprints.py,
# which already fingerprints any Reviewer with pid=None from its area-profile
# document, no special-casing needed here).
DEFAULT_TRC_OVERRIDES = curated_path("trc_dblp_overrides.csv")

TIER = "trc"


@dataclass
class TrcMember(Reviewer):
    # Every declared advisor's address, lowercased, in form order. A student
    # inherits each advisor's entire conflict set (see
    # scripts/assign_trc_reviews.py's person_conflict_pids) -- conservative
    # by design, per explicit direction: an advisor's own students should
    # never end up reviewing something the advisor is tied to in any way.
    advisor_emails: tuple[str, ...] = dataclass_field(default_factory=tuple)


def _latest_rows_by_email(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Collapse repeat form submissions to each student's most recent row.

    Same policy as reviewers._latest_rows_by_email (reimplemented locally
    rather than importing a leading-underscore helper across modules): the
    row with the latest Timestamp wins, matching case-insensitively.
    """
    latest: dict[str, dict[str, str]] = {}
    latest_ts: dict[str, datetime.datetime] = {}
    for row in rows:
        email = field(row, "email address").lower()
        if not email:
            continue
        ts = datetime.datetime.strptime(row["Timestamp"].strip(), TIMESTAMP_FORMAT)
        if email not in latest_ts or ts > latest_ts[email]:
            latest[email] = row
            latest_ts[email] = ts
    return list(latest.values())


def missing_tag_emails(
    students: list[TrcMember], pcinfo_path: str | None
) -> tuple[list[str], list[str]]:
    """(accepted TRC members with no `trc`-tagged HotCRP account, and the reverse).

    Both directions of the membership/tag disagreement `_cross_check_membership`
    warns about, pulled out as its own function so a caller -- e.g.
    `scripts/assign_trc_reviews.py --missing-tag-csv` -- can act on the first
    list (who to bulk-tag in HotCRP) instead of only reading it off stderr.
    Never prunes; the acceptance form stays the membership authority (see
    module docstring). Returns `([], [])` if `pcinfo_path` is falsy or the
    export can't be read, the same degrade-and-warn every other roster
    loader's membership check uses.
    """
    if not pcinfo_path:
        return [], []
    try:
        index = pc_membership.load_pc_accounts(pcinfo_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"WARNING: TRC membership cross-check skipped: {exc}", file=sys.stderr)
        return [], []

    def is_trc_tagged(acct) -> bool:
        return any(t.startswith("trc") for t in pc_membership.tag_names(acct.tags))

    form_emails = {s.email for s in students}
    trc_tagged_emails = {
        acct.email.lower() for acct in index.accounts if is_trc_tagged(acct)
    }
    return (
        sorted(form_emails - trc_tagged_emails),
        sorted(trc_tagged_emails - form_emails),
    )


def _cross_check_membership(students: list[TrcMember], pcinfo_path: str | None) -> None:
    """Warn (never prune) on a disagreement between the form and HotCRP tags.

    The acceptance form is the membership authority here (see module
    docstring), so this only reports -- in the spirit of `make pc-roster`'s
    pruned/missing report pair, without the pruning behavior.
    """
    no_tag, no_form = missing_tag_emails(students, pcinfo_path)
    if no_tag:
        print(
            f"WARNING: {len(no_tag)} accepted TRC member(s) have no `trc`-tagged "
            f"HotCRP account: {', '.join(no_tag[:5])}"
            f"{' ...' if len(no_tag) > 5 else ''} — the acceptance form is still "
            f"authoritative, so they are included anyway; fix the tag in HotCRP",
            file=sys.stderr,
        )
    if no_form:
        print(
            f"WARNING: {len(no_form)} `trc`-tagged HotCRP account(s) never accepted "
            f"the TRC form: {', '.join(no_form[:5])}{' ...' if len(no_form) > 5 else ''}",
            file=sys.stderr,
        )


def load_trc_members(
    csv_path: str = DEFAULT_TRC_CSV,
    *,
    overrides_path: str = DEFAULT_TRC_OVERRIDES,
    pcinfo_path: str | None = pc_membership.DEFAULT_PCINFO,
) -> list[TrcMember]:
    """Load accepted TRC members from the acceptance CSV.

    Rows are collapsed to one per email via `_latest_rows_by_email`, then
    declines (rows whose 'TRC membership' says the student is unable to
    accept) are skipped, the same "unable" substring test `load_reviewers`
    uses on 'PC membership' -- today only one value exists ("Yes, I accept
    as a TRC member"), but this future-proofs a decline option the same way.

    `overrides_path` (see `DEFAULT_TRC_OVERRIDES`) wins over the form's own
    DBLP column, same precedence `dblp_overrides.csv` has for the PC.

    No membership pruning against `pcinfo_path` -- this roster is its own
    authority (see module docstring) -- but a non-fatal cross-check still
    runs and warns on any disagreement with HotCRP's `trc` tag.
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    overrides = load_dblp_overrides(overrides_path)

    students: list[TrcMember] = []
    no_pid: list[str] = []
    bad_email: list[str] = []
    for row in _latest_rows_by_email(rows):
        membership = field(row, "TRC membership")
        if "unable" in membership.lower():
            continue
        email = field(row, "email address").lower()
        if "@" not in email:
            # A form-entry slip (seen live: a stray quote where "@" should
            # be), not a resignation. Skipped rather than guessed at, the
            # same "report, don't guess" rule dblp_overrides.csv follows --
            # a garbled address would otherwise reach the HotCRP upload CSV
            # and either fail on import or silently address nobody.
            bad_email.append(email)
            continue
        first, last = field(row, "First Name"), field(row, "Last Name")
        dblp_url = field(row, "DBLP")
        pid = overrides.get(email) or parse_pid(dblp_url)
        if pid is None:
            no_pid.append(email)
        advisor_emails = tuple(
            addr.strip().lower()
            for addr in field(row, "advisor's email").split(";")
            if addr.strip()
        )
        students.append(
            TrcMember(
                email=email,
                first=first,
                last=last,
                dblp_url=dblp_url,
                pid=pid,
                affiliation=field(row, "institutional affiliation"),
                primary=field(row, "primary area"),
                secondary=field(row, "secondary area"),
                tertiary=field(row, "tertiary area"),
                keywords=field(row, "keywords"),
                tier=TIER,
                override_cap=None,
                pid_from_override=email in overrides,
                advisor_emails=advisor_emails,
            )
        )

    if bad_email:
        print(
            f"WARNING: {len(bad_email)} TRC form row(s) have no usable email address "
            f"and were skipped entirely (fix the form response by hand): "
            f"{', '.join(sorted(bad_email))}",
            file=sys.stderr,
        )
    if no_pid:
        print(
            f"{len(no_pid)} TRC member(s) have no DBLP link (or linked something "
            f"other than DBLP, e.g. an ORCID page); matched on declared "
            f"area/keywords alone unless {overrides_path} resolves one: "
            f"{', '.join(sorted(no_pid))}",
            file=sys.stderr,
        )

    _cross_check_membership(students, pcinfo_path)
    return students
