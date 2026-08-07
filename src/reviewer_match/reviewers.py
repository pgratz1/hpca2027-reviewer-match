"""Parse the HPCA PC-member acceptance CSV into Reviewer records.

The CSV is a Google-Forms export whose headers are long free-text questions and
whose exact format is not final. To stay robust against column drift, fields are
located by substring match on the header rather than by exact name.
"""

from __future__ import annotations

from .paths import assignment_path, cache_path, curated_path, input_path, report_path

import csv
import datetime
import sys
from dataclasses import dataclass

from . import pc_membership
from .dblp import parse_pid

TIMESTAMP_FORMAT = "%m/%d/%Y %H:%M:%S"

# Hand-maintained DBLP identity overrides, keyed by email so they survive
# re-exports of the acceptance CSV. Auto-loaded by load_reviewers if present.
DEFAULT_OVERRIDES = curated_path("dblp_overrides.csv")

# Hand-maintained area overrides for a PC member with no acceptance-form row
# (see _pc_members_without_a_form_row) whose HotCRP topic interests are blank
# or wrong. Nobody else needs this file: everyone with a form row already
# declared primary/secondary/tertiary there.
DEFAULT_AREA_OVERRIDES = curated_path("pc_area_overrides.csv")

# HotCRP's reserve-reviewer bulk upload -- read here only to recognise a raw
# recruit whose reserve-side identity never resolved (build_reserve_reviewer_
# info.py held them back in reserve_reviewer_unresolved.csv). Without this,
# _pc_members_without_a_form_row would mistake them for a form-less PC
# addition and build a Reviewer for them from the wrong roster's rules.
DEFAULT_RESERVE_UPLOAD = input_path("reserve_reviewer_upload.csv")


@dataclass
class Reviewer:
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
    tier: str  # 'full' | 'light'
    override_cap: int | None  # overrides the tier-based default paper cap, if set
    pid_from_override: bool = False  # pid came from dblp_overrides.csv, not the form
    # The address a HotCRP bulk upload must use, which is `email` unless
    # pc_membership matched this person under a different one they hold their
    # PC account under. Defaults to `email` for callers (mostly tests) that
    # never ran the membership check.
    hotcrp_email: str = ""

    def __post_init__(self) -> None:
        if not self.hotcrp_email:
            self.hotcrp_email = self.email

    @property
    def name(self) -> str:
        """Display name, falling back to email when the name columns are blank."""
        full = f"{self.first} {self.last}".strip()
        return full or self.email


def field(row: dict[str, str], needle: str) -> str:
    """Return the value of the first column whose header contains `needle`.

    Raises KeyError if no header matches so a renamed/removed column fails loudly
    rather than silently returning empty data.
    """
    for header, value in row.items():
        if header and needle.lower() in header.lower():
            return (value or "").strip()
    raise KeyError(f"no column header contains {needle!r}")


def _parse_override_cap(email: str, override_raw: str) -> int | None:
    """Parse the free-text 'Override paper assignment number' cell.

    Blank means no override. A non-blank, non-integer value (the column has
    no numeric validation on the form side) fails loudly with the offending
    reviewer's email rather than a bare ValueError, since load_reviewers is
    called by every script in the pipeline — one bad cell shouldn't take all
    of them down with an unhelpful traceback.
    """
    if not override_raw:
        return None
    try:
        value = int(override_raw)
    except ValueError:
        raise ValueError(
            f"{email}: 'Override paper assignment number' must be a whole number, got {override_raw!r}"
        ) from None
    if value < 0:
        raise ValueError(
            f"{email}: 'Override paper assignment number' must be non-negative, got {value}"
        )
    return value


def load_dblp_overrides(path: str = DEFAULT_OVERRIDES) -> dict[str, str]:
    """Load the hand-maintained DBLP override file: email -> DBLP PID.

    Format: a CSV with columns `email`, `dblp`, `note` (note is free text and
    ignored here). The dblp cell may be any link shape parse_pid accepts —
    full URL, bare PID, or Google-redirect wrapper. Rows with a blank email
    or dblp cell are skipped; a non-blank dblp value that doesn't parse to a
    PID fails loudly with the offending email, since a silent skip would make
    the override mysteriously not take effect.

    Returns {} if the file doesn't exist.
    """
    try:
        f = open(path, newline="", encoding="utf-8")
    except FileNotFoundError:
        return {}
    overrides: dict[str, str] = {}
    with f:
        for row in csv.DictReader(f):
            email = (row.get("email") or "").strip().lower()
            raw = (row.get("dblp") or "").strip()
            if not email or not raw:
                continue
            pid = parse_pid(raw)
            if pid is None:
                raise ValueError(
                    f"{path}: override for {email} has an unparseable DBLP value {raw!r}"
                )
            overrides[email] = pid
    return overrides


def load_area_overrides(path: str = DEFAULT_AREA_OVERRIDES) -> dict[str, tuple[str, str, str]]:
    """Load the hand-maintained area override file: email -> (primary, secondary, tertiary).

    Format: a CSV with columns `email`, `primary`, `secondary`, `tertiary`,
    `note` (note is free text and ignored here). A row with a blank email or a
    blank `primary` is skipped -- a hand decision always names at least a
    primary area, the same "blank means no override yet" rule
    `load_dblp_overrides` uses. `secondary`/`tertiary` may be blank; not every
    account has three.

    Returns {} if the file doesn't exist.
    """
    try:
        f = open(path, newline="", encoding="utf-8")
    except FileNotFoundError:
        return {}
    overrides: dict[str, tuple[str, str, str]] = {}
    with f:
        for row in csv.DictReader(f):
            email = (row.get("email") or "").strip().lower()
            primary = (row.get("primary") or "").strip()
            if not email or not primary:
                continue
            overrides[email] = (
                primary,
                (row.get("secondary") or "").strip(),
                (row.get("tertiary") or "").strip(),
            )
    return overrides


def _reserve_upload_emails(path: str) -> set[str]:
    """Emails HotCRP's reserve-reviewer upload named, resolved or not.

    A raw recruit belongs to the reserve pipeline even before
    `build_reserve_reviewer_info.py` has resolved a DBLP identity for them --
    so this counts them as accounted for, the same as an entry
    `load_reserve_reviewers` would eventually pick up, and keeps them from
    being mistaken for a form-less PC addition. Missing file -> empty set:
    this is a defensive exclusion, not something `load_reviewers` should fail
    without.
    """
    try:
        f = open(path, newline="", encoding="utf-8-sig")
    except FileNotFoundError:
        return set()
    with f:
        return {(row.get("email") or "").strip().lower() for row in csv.DictReader(f)} - {""}


def _latest_rows_by_email(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Collapse repeat form submissions to each person's most recent row.

    Some PC members resubmitted the acceptance form (correcting an area,
    adding a DBLP link, or changing their mind about accepting/declining).
    Google Forms appends a new row rather than editing the old one, so the
    same person's email can appear multiple times, including with a decline
    in one row and an accept in another — the row with the latest Timestamp
    is authoritative. Matching is case-insensitive since a few submissions
    vary only in email capitalization.
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


def load_reviewers(
    csv_path: str,
    overrides_path: str = DEFAULT_OVERRIDES,
    *,
    pcinfo_path: str | None = pc_membership.DEFAULT_PCINFO,
    area_overrides_path: str = DEFAULT_AREA_OVERRIDES,
    reserve_upload_path: str = DEFAULT_RESERVE_UPLOAD,
) -> list[Reviewer]:
    """Load accepted PC members from the acceptance CSV.

    Rows are first collapsed to one per email via `_latest_rows_by_email` so
    a later resubmission (including a later decline overriding an earlier
    accept, or vice versa) wins. Declines (rows whose 'PC membership' says
    the member is unable to accept) are then skipped; everyone else is
    returned, including those without a DBLP link.

    Whoever is left is checked against the HotCRP user export, which is the
    authority on who is still on the committee: accepting the invitation is
    not the same as being on the PC today, and people do get removed after
    they accept. Anyone the export does not have on the PC under any address
    is dropped — see `pc_membership`, which is careful about the second
    address, since accepting from one and holding the HotCRP account under
    another is common. `pcinfo_path=None` skips the check entirely.

    If a DBLP override file exists (see load_dblp_overrides), its PID wins
    over whatever the form's DBLP column says — filling in reviewers who left
    it blank and correcting wrong links (e.g. a namesake's page). Overrides
    whose email matches no accepted reviewer are reported to stderr so a
    typo'd email doesn't silently do nothing.

    Finally, `_pc_members_without_a_form_row` adds anyone the HotCRP export
    marks `pc` that the form never explains at all — someone added to the
    committee straight in HotCRP after the acceptance form closed. See its
    docstring for where their identity, tier and areas come from.
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    overrides = load_dblp_overrides(overrides_path)
    area_overrides = load_area_overrides(area_overrides_path)
    index = pc_membership.load_pc_accounts(pcinfo_path) if pcinfo_path else None

    reviewers: list[Reviewer] = []
    dropped: list[str] = []
    timestamps: dict[str, datetime.datetime] = {}
    for row in _latest_rows_by_email(rows):
        membership = field(row, "PC membership")
        if "unable" in membership.lower():
            continue
        tier = "light" if "light" in membership.lower() else "full"
        email = field(row, "email address").lower()
        timestamps[email] = datetime.datetime.strptime(row["Timestamp"].strip(), TIMESTAMP_FORMAT)
        first, last = field(row, "First Name"), field(row, "Last Name")
        # Before the Reviewer is built, not after: someone who is off the PC
        # shouldn't be able to take the whole pipeline down through a garbage
        # override-cap cell they will never use.
        hotcrp_email = email
        if index is not None:
            acct, how = index.match(email, first, last)
            if acct is None:
                dropped.append(email)
                continue
            hotcrp_email = pc_membership.hotcrp_email_for(email, acct, how)
        dblp_url = field(row, "DBLP")
        reviewers.append(
            Reviewer(
                email=email,
                hotcrp_email=hotcrp_email,
                first=first,
                last=last,
                dblp_url=dblp_url,
                pid=overrides.get(email) or parse_pid(dblp_url),
                affiliation=field(row, "institutional affiliation"),
                primary=field(row, "primary area"),
                secondary=field(row, "secondary area"),
                tertiary=field(row, "tertiary area"),
                keywords=field(row, "keywords"),
                tier=tier,
                override_cap=_parse_override_cap(email, field(row, "Override paper assignment number")),
                pid_from_override=email in overrides,
            )
        )

    if index is not None:
        pc_membership.report_pruned(
            dropped, len(reviewers), "acceptance-form reviewers", index
        )

    # A person who resubmitted under a second real address (not the same
    # email twice, which _latest_rows_by_email already handled) is two rows
    # here that resolve to one HotCRP account; the later submission wins, the
    # same policy as same-email duplicates.
    reviewers, collapsed = pc_membership.collapse_by_hotcrp_email(reviewers, timestamps)
    if collapsed:
        shown = ", ".join(f"{lost} -> {kept}" for kept, lost in sorted(collapsed)[:5])
        more = f", +{len(collapsed) - 5} more" if len(collapsed) > 5 else ""
        print(
            f"{len(collapsed)} acceptance-form submission(s) under a second address "
            f"collapsed into the same HotCRP account as another row: {shown}{more}",
            file=sys.stderr,
        )

    if index is not None:
        reserve_upload_emails = _reserve_upload_emails(reserve_upload_path)
        reviewers.extend(
            _pc_members_without_a_form_row(
                index, reviewers, overrides, area_overrides, reserve_upload_emails
            )
        )

    # A dropped reviewer's override is accounted for, not a typo — saying
    # otherwise would send someone hunting for a mistake they didn't make.
    # This check only runs with the membership check on: it is the only case
    # where _pc_members_without_a_form_row has had a chance to explain an
    # override belonging to nobody on the form. Without it (area_chairs.py's
    # `_profile_donors` reads the form this way on purpose, as a donor pool
    # rather than a membership claim), a no-form-row override for a real PC
    # member — the exact case this whole mechanism exists for — would always
    # read as an unmatched typo, since nothing here could tell the two apart.
    if index is not None:
        collapsed_away = {lost for _, lost in collapsed}
        unmatched = set(overrides) - {r.email for r in reviewers} - set(dropped) - collapsed_away
        for email in sorted(unmatched):
            print(
                f"Warning: DBLP override for {email} matches no accepted reviewer "
                f"(typo, or they declined?)",
                file=sys.stderr,
            )
    return reviewers


def _pc_members_without_a_form_row(
    index: pc_membership.PcIndex,
    reviewers: list[Reviewer],
    overrides: dict[str, str],
    area_overrides: dict[str, tuple[str, str, str]],
    reserve_upload_emails: set[str],
) -> list[Reviewer]:
    """PC members HotCRP knows about that the acceptance form does not.

    Someone added straight to HotCRP after the form closed never answered it,
    so none of the three things the form would have supplied exist on its own:

      * identity (a DBLP page) — from `dblp_overrides.csv`, the same file and
        the same "blank means not resolved yet" rule as everyone else. A
        missing PID does not exclude the account, the same as a form row that
        left DBLP blank: `classify_reviewers.py` already treats a PID-less
        Reviewer as unclassifiable and appends a stub row for it, so the
        to-do surfaces there rather than needing a second mechanism here.
      * areas — from the account's positive-scored HotCRP topic interests
        (`Account.top_topics`), which is what the form's primary/secondary/
        tertiary questions amount to when nobody manually curates them: a
        `pc_area_overrides.csv` row for the email wins when present (a hand
        decision always outranks a derived guess, the same precedence
        `dblp_overrides.csv` has over the form's own DBLP column), otherwise
        the top HotCRP-declared topics are used. Either way a missing area
        does not exclude the account — it just cannot pass the area gate for
        any paper until one exists, which is reported so it doesn't go
        unnoticed.
      * tier — `pc-full`/`pc-light` is the *only* place this can come from
        when there is no form. A `pc-full`/`pc-light` tag is also the
        candidacy signal, not just the tier value: this loader only ever sees
        the PC acceptance-form roster, so an account carrying neither tag is
        silently not a candidate here rather than reported — on the live
        export most such accounts are reserve reviewers this loader has no
        way to recognise as already accounted for, and `audit_pc_roster.py`'s
        `no_roster_row` category is the tag-agnostic backstop that still
        catches a genuine no-form, no-tag PC addition. An account carrying
        *both* tags is a real anomaly, though, the same as `Account.pc_tier`'s
        other caller (an untiered `~~ex-rr` account) — there is no safe
        default to guess, so it is left out of the reviewer pool entirely and
        reported.

    Chairs, TRC members, sysadmins and area chairs are excluded — they are on
    the PC for a structural reason `structural_role` already accounts for,
    not because a review form is missing. So are ex-reserves, who are
    promoted by `load_reserve_reviewers` instead: that path derives areas
    from authored papers and keeps their reserve-side fingerprint, neither of
    which this one can reconstruct. And so is anyone HotCRP's reserve-reviewer
    upload named, even if `build_reserve_reviewer_info.py` never resolved a
    DBLP identity for them (`reserve_reviewer_unresolved.csv`) — they belong
    to that to-do list, not this one.
    """
    accounted: set[str] = set()
    for r in reviewers:
        accounted |= pc_membership.resolved_emails(index, r.email, r.first, r.last)

    added: list[Reviewer] = []
    untiered: list[str] = []
    no_pid: list[str] = []
    no_area: list[str] = []
    for acct in index.pc_accounts:
        email = acct.email.lower()
        if email in accounted or email in reserve_upload_emails:
            continue
        if pc_membership.structural_role(acct) is not None or acct.is_ex_reserve:
            continue
        # A `pc-full`/`pc-light` tag is the candidacy signal, not just the tier
        # value: this loader only ever sees the PC acceptance-form roster, not
        # the reserve or area-chair rosters, so an untagged account here is
        # usually a reserve reviewer this loader has no way to recognise as
        # already accounted for -- not a genuine no-form-row PC member. Only a
        # self-contradictory account (both tags) is a real anomaly worth a
        # warning; a plain absence of either tag is silently not a candidate.
        if not (pc_membership.TIER_TAGS.keys() & pc_membership.tag_names(acct.tags)):
            continue
        tier = acct.pc_tier
        if tier is None:
            untiered.append(email)
            continue
        override = area_overrides.get(email)
        areas = list(override) if override else acct.top_topics(3)
        areas += [""] * (3 - len(areas))
        pid = overrides.get(email)
        if pid is None:
            no_pid.append(email)
        if not areas[0]:
            no_area.append(email)
        added.append(
            Reviewer(
                email=email,
                hotcrp_email=acct.email,
                first=acct.given,
                last=acct.family,
                dblp_url="",
                pid=pid,
                affiliation=acct.affiliation,
                primary=areas[0],
                secondary=areas[1],
                tertiary=areas[2],
                keywords="",
                tier=tier,
                override_cap=None,
                pid_from_override=email in overrides,
            )
        )

    if added:
        print(
            f"{len(added)} PC member(s) on the HotCRP roster with no acceptance-form "
            f"row; profile built from HotCRP tags and topic interests: "
            f"{', '.join(sorted(r.email for r in added))}",
            file=sys.stderr,
        )
    if no_pid:
        print(
            f"    {len(no_pid)} of them have no DBLP link yet — add one to "
            f"{DEFAULT_OVERRIDES} to fingerprint them: {', '.join(sorted(no_pid))}",
            file=sys.stderr,
        )
    if no_area:
        print(
            f"    {len(no_area)} of them have no positive HotCRP topic interest and "
            f"no {DEFAULT_AREA_OVERRIDES} row, so they match no paper's area gate "
            f"until one exists: {', '.join(sorted(no_area))}",
            file=sys.stderr,
        )
    if untiered:
        tags = "`/`".join(sorted(pc_membership.TIER_TAGS))
        print(
            f"WARNING: {len(untiered)} PC account(s) with no acceptance-form row "
            f"carry neither or both of `{tags}`, so their load tier is not settled "
            f"and they are left out of the reviewer pool entirely: "
            f"{', '.join(sorted(untiered))} — fix the tags in HotCRP",
            file=sys.stderr,
        )
    return added
