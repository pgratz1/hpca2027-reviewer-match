"""Parse the HPCA area-chair acceptance form into research-profile records."""

from __future__ import annotations

import csv
from dataclasses import dataclass

from dblp import parse_pid
from reviewers import _latest_rows_by_email, field, load_dblp_overrides


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


def load_area_chairs(csv_path: str, overrides_path: str = "dblp_overrides.csv") -> list[AreaChair]:
    """Load the latest explicitly accepted area-chair responses."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    overrides = load_dblp_overrides(overrides_path)

    chairs = []
    for row in _latest_rows_by_email(rows):
        membership = field(row, "Area Chair membership")
        if not membership.lower().startswith("yes"):
            continue
        email = field(row, "email address").lower()
        dblp_url = field(row, "DBLP")
        chairs.append(
            AreaChair(
                email=email,
                first=field(row, "First Name"),
                last=field(row, "Last Name"),
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
    return chairs
