"""Parse the HPCA PC-member acceptance CSV into Reviewer records.

The CSV is a Google-Forms export whose headers are long free-text questions and
whose exact format is not final. To stay robust against column drift, fields are
located by substring match on the header rather than by exact name.
"""

from __future__ import annotations

import csv
import datetime
from dataclasses import dataclass

from dblp import parse_pid

TIMESTAMP_FORMAT = "%m/%d/%Y %H:%M:%S"


@dataclass
class Reviewer:
    email: str
    first: str
    last: str
    dblp_url: str
    pid: str | None
    primary: str
    secondary: str
    tertiary: str
    keywords: str
    tier: str  # 'full' | 'light'
    override_cap: int | None  # overrides the tier-based default paper cap, if set

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


def load_reviewers(csv_path: str) -> list[Reviewer]:
    """Load accepted PC members from the acceptance CSV.

    Rows are first collapsed to one per email via `_latest_rows_by_email` so
    a later resubmission (including a later decline overriding an earlier
    accept, or vice versa) wins. Declines (rows whose 'PC membership' says
    the member is unable to accept) are then skipped; everyone else is
    returned, including those without a DBLP link.
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    reviewers: list[Reviewer] = []
    for row in _latest_rows_by_email(rows):
        membership = field(row, "PC membership")
        if "unable" in membership.lower():
            continue
        tier = "light" if "light" in membership.lower() else "full"
        dblp_url = field(row, "DBLP")
        override_raw = field(row, "Override paper assignment number")
        reviewers.append(
            Reviewer(
                email=field(row, "email address").lower(),
                first=field(row, "First Name"),
                last=field(row, "Last Name"),
                dblp_url=dblp_url,
                pid=parse_pid(dblp_url),
                primary=field(row, "primary area"),
                secondary=field(row, "secondary area"),
                tertiary=field(row, "tertiary area"),
                keywords=field(row, "keywords"),
                tier=tier,
                override_cap=int(override_raw) if override_raw else None,
            )
        )
    return reviewers
