"""Load whichever roster a script was pointed at, by role.

Three kinds of person now flow through the same publication-fetching and
fingerprinting code: PC members and area chairs, who each filled in an
acceptance form, and reserve reviewers, who never did and whose record is
assembled from the submissions instead. The scripts differ only in which loader
they call, so the choice lives here rather than being spelled out in each one.

Every loader returns records exposing the same fields the downstream code reads
-- email, name, pid, primary/secondary/tertiary, keywords -- so a caller only
needs the role.
"""

from __future__ import annotations

from .paths import assignment_path, cache_path, curated_path, input_path, report_path

from . import pc_membership
from .area_chairs import load_area_chairs
from .reserve_reviewers import DEFAULT_DATA, DEFAULT_INFO
from .reserve_reviewers import load_reserve_reviewers
from .reviewers import load_reviewers

ROLES = ("reviewer", "area-chair", "reserve")

DEFAULT_CSV = input_path("HPCA'27 PC Member Acceptance Form (Responses) - Form Responses 1.csv")
DEFAULT_AREA_CHAIR_CSV = input_path("Area Chair Acceptance Form (Responses) - Form Responses 1.csv")

# Each role's default roster file, so --csv stays optional for all of them.
DEFAULT_ROSTERS = {
    "reviewer": DEFAULT_CSV,
    "area-chair": DEFAULT_AREA_CHAIR_CSV,
    "reserve": DEFAULT_INFO,
}


def load_roster(
    role: str,
    csv_path: str | None = None,
    data_path: str = DEFAULT_DATA,
    *,
    pcinfo_path: str | None = pc_membership.DEFAULT_PCINFO,
):
    """Load `role`'s roster, falling back to that role's default file.

    `pcinfo_path` reaches all three loaders, because the HotCRP membership
    check is only worth anything if every role is held to it — one branch
    skipping the gate is how two scripts end up with different committees.
    """
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; expected one of {', '.join(ROLES)}")
    path = csv_path or DEFAULT_ROSTERS[role]
    if role == "area-chair":
        return load_area_chairs(path, pcinfo_path=pcinfo_path)
    if role == "reserve":
        return load_reserve_reviewers(path, data_path, pcinfo_path=pcinfo_path)
    return load_reviewers(path, pcinfo_path=pcinfo_path)


def role_label(role: str) -> str:
    """Plural human name for progress and summary lines."""
    return {
        "reviewer": "reviewers",
        "area-chair": "area chairs",
        "reserve": "reserve reviewers",
    }[role]
