"""Read HotCRP's action log: replay the live review assignment, and index PC activity.

`data/inputs/hpca2027-log.csv` is HotCRP's own action log, exported with the
header `date,ipaddr,email,roles,affected_email,via,paper,action`. Two things
in this pipeline need it, and both of them need the same parse:
`scripts/extract_log_assignments.py` reconstructs what HotCRP actually holds
today (manual UI edits included, which no artifact under `outputs/` knows
about), and `scripts/audit_reviewer_activity.py` asks who has touched their
assignment since it landed.

Two properties of the file drive every function here.

**It is newest-first.** `load_log` reverses it, so every consumer sees events
in the order they happened -- an assign/unassign replay read backwards is
simply wrong, not merely reversed.

**It preserves HotCRP's display casing of addresses** (`Given.Family@uni.edu`),
while every roster key and `hotcrp_email` in this repo is lower-cased. So
`load_log` lower-cases `email` and `affected_email` at parse time, once, here.
Skipping that reads as ~110 phantom assignment changes against an identical
`outputs/assignments/assignment.csv` -- a difference in spelling presenting as
a difference in the assignment.

**There is no login event in a HotCRP log, and no "paper viewed" event.**
Nothing here can report who signed in. `last_activity` reports who *did*
something the log records, which is a floor on engagement and not the same
question; `engagement_actions` narrows that to the subset which can only come
from opening a paper. Both are proxies, and callers must say so.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass

LOG_HEADER = ("date", "ipaddr", "email", "roles", "affected_email", "via", "paper", "action")

# "Review 31247 assigned: primary, round R1" / "Review 11642 unassigned".
ASSIGN_RE = re.compile(r"^Review (\d+) assigned: (\w+), round (\S+)$")
UNASSIGN_RE = re.compile(r"^Review (\d+) unassigned$")

# Actions that can only follow from opening a paper or a review form. Deliberately
# narrower than "the account did something": a password reset or a topic-interest
# edit says the person logged in, not that they looked at what they were assigned.
ENGAGEMENT_RE = re.compile(
    r"^(Download submission|Download reviews|Review \d+ (accepted|declined|edited))"
)


@dataclass(frozen=True)
class Review:
    """One live review row, as the log records it."""

    pid: int
    email: str
    kind: str  # "primary", "external", ...
    round: str  # "R1", "TRC", ...
    assigned_at: str


def load_log(path: str) -> list[dict[str, str]]:
    """Every log row, oldest first, with both address columns lower-cased.

    Raises ValueError on an unexpected header rather than silently yielding
    rows with missing keys: this file is a HotCRP export whose shape can change
    between versions, and a KeyError three functions deep is a worse way to
    learn that than a named failure here.
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != LOG_HEADER:
            raise ValueError(
                f"{path}: unexpected header {reader.fieldnames!r}; expected {LOG_HEADER}"
            )
        rows = list(reader)
    for row in rows:
        row["email"] = (row["email"] or "").strip().lower()
        row["affected_email"] = (row["affected_email"] or "").strip().lower()
    rows.reverse()
    return rows


def replay_assignments(
    rows: list[dict[str, str]]
) -> tuple[dict[int, Review], list[tuple[str, dict[str, str]]]]:
    """Replay assign/unassign into ({review id: Review}, [(anomaly, row)]).

    HotCRP's review id is the identity that survives an unassign/reassign
    cycle, so it -- not (pid, email) -- is what the state is keyed on. A bulk
    upload's leading `clearreview` row shows up here as one `unassigned` event
    per existing review, which is exactly how a re-upload replaces rather than
    duplicates an assignment.

    Anomalies are returned, never raised and never swallowed: a log window that
    opens after a review was assigned genuinely does contain an unassign for an
    id it never saw, and that is a fact about the export, not a parse error.
    Callers report the count. `rows` must be chronological -- see `load_log`.
    """
    live: dict[int, Review] = {}
    anomalies: list[tuple[str, dict[str, str]]] = []
    for row in rows:
        action = row["action"]
        m = ASSIGN_RE.match(action)
        if m:
            rid = int(m.group(1))
            if rid in live:
                anomalies.append(("reassigned without an unassign", row))
            live[rid] = Review(
                pid=int(row["paper"]),
                email=row["affected_email"],
                kind=m.group(2),
                round=m.group(3),
                assigned_at=row["date"],
            )
            continue
        m = UNASSIGN_RE.match(action)
        if m:
            rid = int(m.group(1))
            previous = live.pop(rid, None)
            if previous is None:
                anomalies.append(("unassigned a review never assigned in this window", row))
            elif previous.email != row["affected_email"]:
                anomalies.append(("unassigned under a different address", row))
            elif previous.pid != int(row["paper"]):
                anomalies.append(("unassigned from a different paper", row))
    return live, anomalies


def live_pairs(live: dict[int, Review], kind: str = "primary", round: str = "R1") -> dict[int, set[str]]:
    """{pid: {email}} for the live reviews of one kind and round.

    `kind`/`round` of "all" keep everything. Collapses the review id away: two
    review rows for the same person on the same paper (which HotCRP does not
    create, but a log this tool did not write is not ours to trust) become one
    pair rather than two.
    """
    pairs: dict[int, set[str]] = defaultdict(set)
    for review in live.values():
        if kind != "all" and review.kind != kind:
            continue
        if round != "all" and review.round != round:
            continue
        pairs[review.pid].add(review.email)
    return dict(pairs)


def first_assigned_at(live: dict[int, Review]) -> dict[str, str]:
    """{email: timestamp of the EARLIEST of their live assignments}.

    The per-reviewer cutoff `audit_reviewer_activity.py` judges activity
    against. One global cutoff would misreport anyone assigned in a later
    patch: someone given their first paper on Wednesday has not been silent
    since Monday, they have been assigned for a day.

    Earliest, not latest, and the difference is not cosmetic. The question is
    whether someone has looked at their assignment at all; a reviewer who read
    their papers on Monday afternoon and was handed one more on Monday evening
    has looked. Keying on their most recent assignment would move the cutoff
    past the activity that answers the question and report them as silent --
    which on this log costs 17 of 283 reviewers, every one of them a false
    accusation.
    """
    earliest: dict[str, str] = {}
    for review in live.values():
        current = earliest.get(review.email)
        if current is None or review.assigned_at < current:
            earliest[review.email] = review.assigned_at
    return earliest


def last_activity(
    rows: list[dict[str, str]], *, since: dict[str, str] | str | None = None
) -> dict[str, tuple[str, str, int]]:
    """{acting email: (last date, last action, rows since the cutoff)}.

    Keyed on `email`, the account that *performed* the action -- never
    `affected_email`, which is who it was done to. A chair assigning someone a
    paper writes that person's address into `affected_email`, and counting it
    would score every reviewer as active the moment they were assigned.

    `since` is a per-email mapping (the reviewer's own assignment timestamp),
    one string for a single global cutoff, or None for the whole log. An email
    with no entry in a mapping cutoff is judged over the whole log; an address
    with nothing after its cutoff is simply absent from the result.
    """
    counts: dict[str, int] = defaultdict(int)
    last: dict[str, tuple[str, str]] = {}
    for row in rows:
        email = row["email"]
        if not email:
            continue
        if isinstance(since, str):
            cutoff = since
        elif since is not None:
            cutoff = since.get(email, "")
        else:
            cutoff = ""
        if cutoff and row["date"] < cutoff:
            continue
        counts[email] += 1
        # Chronological, so the last write wins.
        last[email] = (row["date"], row["action"])
    return {email: (*last[email], counts[email]) for email in last}


def engagement_actions(
    rows: list[dict[str, str]], *, since: dict[str, str] | str | None = None
) -> dict[str, tuple[str, str, int]]:
    """`last_activity`, restricted to actions that mean a paper was actually opened."""
    engaged = [row for row in rows if ENGAGEMENT_RE.match(row["action"])]
    return last_activity(engaged, since=since)
