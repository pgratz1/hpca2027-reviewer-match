"""Parse the two per-pair assignment CSV shapes this pipeline writes.

`assign_reviewers.write_hotcrp_csv` writes the HotCRP upload shape
(`paper,action,email,round`), keyed on each reviewer's `hotcrp_email`;
`assign_reviewers.write_pairs_csv` writes the measurement shape
(`pid,email,phase,rank_score,affinity`), keyed on the roster `email`, with
`phase`/`affinity` alongside. Both are `(pid, email)` pairs underneath, and
`scripts/diff_assignments.py` and `scripts/fill_open_slots.py` both need to
read either one -- this is the one place that parsing lives, so neither
duplicates it.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Mapping

HOTCRP_CSV_HEADER = ("paper", "action", "email", "round")
PAIRS_CSV_HEADER = ("pid", "email", "phase", "rank_score", "affinity")


@dataclass(frozen=True)
class Pair:
    """One (paper, reviewer) assignment, as much as either CSV shape records."""

    phase: str | None = None
    affinity: float | None = None


def sniff_format(path: str) -> str:
    """"hotcrp" or "pairs", from the file's own header row.

    Raises ValueError on anything else rather than guessing: a shape this
    doesn't recognise should fail loudly, not be silently misparsed as the
    wrong one.
    """
    with open(path, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f), None)
    if header == list(HOTCRP_CSV_HEADER):
        return "hotcrp"
    if header == list(PAIRS_CSV_HEADER):
        return "pairs"
    raise ValueError(
        f"{path}: unrecognised header {header!r}; expected {HOTCRP_CSV_HEADER} "
        f"(a --hotcrp-csv file) or {PAIRS_CSV_HEADER} (a --pairs-csv file)"
    )


def load_assignment_pairs(path: str) -> tuple[str, dict[int, dict[str, Pair]]]:
    """Read either CSV shape into (format, {pid: {email: Pair}}).

    A hotcrp-csv's `clearreview` bookkeeping row, and any row whose `action`
    isn't `primaryreview`, are skipped -- the housekeeping row this pipeline
    always writes first is a no-op here rather than a phantom pid.
    """
    fmt = sniff_format(path)
    pairs: dict[int, dict[str, Pair]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if fmt == "hotcrp":
                if row["action"] != "primaryreview":
                    continue
                pid = int(row["paper"])
                pairs.setdefault(pid, {})[row["email"]] = Pair()
            else:
                pid = int(row["pid"])
                affinity = float(row["affinity"]) if row["affinity"] else None
                pairs.setdefault(pid, {})[row["email"]] = Pair(
                    phase=row["phase"] or None, affinity=affinity
                )
    return fmt, pairs


def hotcrp_email_map(reviewers_by_email: Mapping[str, object]) -> dict[str, str]:
    """{roster email: hotcrp_email}, for normalizing a pairs-csv onto the upload key space.

    `write_hotcrp_csv` keys pairs on `hotcrp_email`; `write_pairs_csv` keys
    them on the roster `email`. These differ for anyone `pc_membership`
    matched under a second address -- treating the two key spaces as the same
    one manufactures false churn for exactly those reviewers.
    """
    return {email: r.hotcrp_email for email, r in reviewers_by_email.items()}


def remap_pairs(
    pairs: dict[int, dict[str, "Pair"]], email_map: dict[str, str]
) -> dict[int, dict[str, "Pair"]]:
    """Remap every email in `pairs` through `email_map` (identity if absent).

    Used to normalize a --hotcrp-csv (keyed on hotcrp_email) or a --pairs-csv
    (keyed on the roster email) onto whichever key space a caller needs --
    diff_assignments.py normalizes both files onto hotcrp_email before
    comparing; fill_open_slots.py normalizes a hotcrp-csv baseline back onto
    the roster-email space its own COI/scoring logic keys on.
    """
    return {
        pid: {email_map.get(email, email): pair for email, pair in emails.items()}
        for pid, emails in pairs.items()
    }
