"""Parse the HPCA PC-member acceptance CSV into Reviewer records.

The CSV is a Google-Forms export whose headers are long free-text questions and
whose exact format is not final. To stay robust against column drift, fields are
located by substring match on the header rather than by exact name.
"""

from __future__ import annotations

import csv
import datetime
import sys
from dataclasses import dataclass

from dblp import parse_pid

TIMESTAMP_FORMAT = "%m/%d/%Y %H:%M:%S"

# Hand-maintained DBLP identity overrides, keyed by email so they survive
# re-exports of the acceptance CSV. Auto-loaded by load_reviewers if present.
DEFAULT_OVERRIDES = "dblp_overrides.csv"


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
        return int(override_raw)
    except ValueError:
        raise ValueError(
            f"{email}: 'Override paper assignment number' must be a whole number, got {override_raw!r}"
        ) from None


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


def load_reviewers(csv_path: str, overrides_path: str = DEFAULT_OVERRIDES) -> list[Reviewer]:
    """Load accepted PC members from the acceptance CSV.

    Rows are first collapsed to one per email via `_latest_rows_by_email` so
    a later resubmission (including a later decline overriding an earlier
    accept, or vice versa) wins. Declines (rows whose 'PC membership' says
    the member is unable to accept) are then skipped; everyone else is
    returned, including those without a DBLP link.

    If a DBLP override file exists (see load_dblp_overrides), its PID wins
    over whatever the form's DBLP column says — filling in reviewers who left
    it blank and correcting wrong links (e.g. a namesake's page). Overrides
    whose email matches no accepted reviewer are reported to stderr so a
    typo'd email doesn't silently do nothing.
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    overrides = load_dblp_overrides(overrides_path)

    reviewers: list[Reviewer] = []
    for row in _latest_rows_by_email(rows):
        membership = field(row, "PC membership")
        if "unable" in membership.lower():
            continue
        tier = "light" if "light" in membership.lower() else "full"
        dblp_url = field(row, "DBLP")
        email = field(row, "email address").lower()
        reviewers.append(
            Reviewer(
                email=email,
                first=field(row, "First Name"),
                last=field(row, "Last Name"),
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

    unmatched = set(overrides) - {r.email for r in reviewers}
    for email in sorted(unmatched):
        print(
            f"Warning: DBLP override for {email} matches no accepted reviewer "
            f"(typo, or they declined?)",
            file=sys.stderr,
        )
    return reviewers
