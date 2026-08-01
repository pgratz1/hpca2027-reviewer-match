"""Parse the HPCA area-chair acceptance form into research-profile records."""

from __future__ import annotations

from .paths import assignment_path, cache_path, curated_path, input_path, report_path

import csv
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
) -> list[AreaChair]:
    """Load the latest explicitly accepted area-chair responses.

    Area chairs go through the same HotCRP membership check as everyone else.
    Every one of them is on the PC today, so it changes nothing — but the check
    living in one loader and not another is how the scripts would come to
    disagree about who is on the committee. `pcinfo_path=None` skips it.
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
    return chairs
