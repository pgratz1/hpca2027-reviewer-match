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
from datetime import datetime

# `2026-09-22 11:06:00 -0400`: local time with its own offset.
DATE_FORMAT = "%Y-%m-%d %H:%M:%S %z"

LOG_HEADER = ("date", "ipaddr", "email", "roles", "affected_email", "via", "paper", "action")

# "Review 31247 assigned: primary, round R1" / "Review 11642 unassigned".
ASSIGN_RE = re.compile(r"^Review (\d+) assigned: (\w+), round (\S+)$")
UNASSIGN_RE = re.compile(r"^Review (\d+) unassigned$")
# "Review 5702 deleted": how HotCRP logs a chair removing a review that has
# content. No "unassigned" follows, so a replay that misses it keeps a review
# HotCRP no longer has.
DELETE_RE = re.compile(r"^Review (\d+) deleted$")

# Actions that can only follow from opening a paper or a review form. Deliberately
# narrower than "the account did something": a password reset or a topic-interest
# edit says the person logged in, not that they looked at what they were assigned.
ENGAGEMENT_RE = re.compile(
    r"^(Download submission|Download reviews|Review \d+ (accepted|declined|edited))"
)

# "Review 18025 submitted: 1337 words" (first submission) / "Review 4861 edited,
# submitted: ComAut, ..., 466 words" (a resubmission after edits) -- both
# distinct from "Review N edited, updated draft: ...", a draft save that does
# not mean submitted, and from "Review N edited, updated: ...", which is an
# edit to a review that is *already* submitted (every one of the 1,681 such
# rows today lands on a submitted review) -- the signal `UPDATE_RE` below reads
# as "the reviewer revisited their review". A draft saved first and submitted
# unchanged logs a bare "Review 17095 submitted", with no colon or word count
# -- rare (1 in 7,000 today), but missing it read a turned-in review as
# outstanding and put it in an unassign upload.
SUBMIT_RE = re.compile(r"^Review (\d+) (?:submitted|edited, submitted)(?::|$)")
# "Review 15 retracted" / "Review 726 deleted" / "Review 5702 unsubmitted draft:
# 482 words" (a chair returning it to draft) -- each reverses a submission.
UNSUBMIT_RE = re.compile(r"^Review (\d+) (?:retracted$|deleted$|unsubmitted\b)")

# "Review 4861 edited, updated: PreRebOve:3, ComAut, 512 words": a change to a
# review that was already submitted. The field tokens name what changed.
UPDATE_RE = re.compile(r"^Review (\d+) edited, updated:")

# "Comment 2272 edited: text" / "Comment 2273 submitted" / "Comment 1575 on
# submission thread edited: visibility". HotCRP logs a comment's *creation* as
# "edited" too -- there is no "created" verb -- so the first row for a comment
# id is when it came into being. The log never says what a comment's
# visibility is (only that it changed), nor who it is addressed to; the API's
# comment objects carry that, and `scripts/discussion_nudges.py` joins the two
# on the comment id.
COMMENT_RE = re.compile(r"^Comment (\d+)\b")

# Discussion-lead changes, exactly as `a_lead.php` logs them.
SET_LEAD = "Set lead"
CLEAR_LEAD = "Clear lead"


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
    """Replay assign/unassign/delete into ({review id: Review}, [(anomaly, row)]).

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
        m = UNASSIGN_RE.match(action) or DELETE_RE.match(action)
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


def replay_leads(rows: list[dict[str, str]]) -> dict[int, str]:
    """{pid: lead's address} from replaying `Set lead` / `Clear lead`.

    HotCRP logs a lead change as `Set lead` with the new lead as the affected
    user, and a removal as `Clear lead` (`a_lead.php`). A bulk upload can log
    one event against several papers, so `paper` is split on whitespace.
    `rows` must be chronological -- see `load_log`.
    """
    leads: dict[int, str] = {}
    for row in rows:
        action = row["action"]
        if action not in (SET_LEAD, CLEAR_LEAD):
            continue
        for pid in row["paper"].split():
            if action == SET_LEAD:
                leads[int(pid)] = row["affected_email"]
            else:
                leads.pop(int(pid), None)
    return leads


def submitted_review_ids(rows: list[dict[str, str]]) -> set[int]:
    """Review ids currently in "submitted" state, replayed chronologically.

    Paired with `replay_assignments`'s `live` dict (`rid -> Review(pid,
    email, ...)`), a caller builds `rid_by_pair = {(r.pid, r.email): rid for
    rid, r in live.items()}` once, and `rid_by_pair.get((pid, email)) in
    submitted_review_ids(rows)` answers "has this reviewer completed this
    paper's review" for any live pair -- the question a reviewer-swap
    proposal needs before offering to move someone off a review already
    turned in. `rows` must be chronological -- see `load_log`.
    """
    submitted: dict[int, bool] = {}
    for row in rows:
        action = row["action"]
        m = SUBMIT_RE.match(action)
        if m:
            submitted[int(m.group(1))] = True
            continue
        m = UNSUBMIT_RE.match(action)
        if m:
            submitted[int(m.group(1))] = False
    return {rid for rid, state in submitted.items() if state}


def first_submitted_at(rows: list[dict[str, str]]) -> dict[int, str]:
    """{review id: date of its FIRST submission}, over every review ever submitted.

    Later "edited, submitted" rows are revisions of a review already turned in
    and do not move this date -- a deadline is met by the first submission, not
    the last edit. It says nothing about whether the review is submitted *now*
    (a retraction does not erase it), so pair it with `submitted_review_ids`.
    `rows` must be chronological -- see `load_log`.
    """
    first: dict[int, str] = {}
    for row in rows:
        m = SUBMIT_RE.match(row["action"])
        if m:
            first.setdefault(int(m.group(1)), row["date"])
    return first


def comment_created_at(rows: list[dict[str, str]]) -> dict[int, str]:
    """{comment id: date of its FIRST log row}, i.e. when the comment was created.

    HotCRP has no "comment created" verb; the first "Comment N edited" (or
    "submitted") row is the creation. The `email` on that row is the author.
    Later edits do not move this date, which is the point: a nudge asks
    whether somebody *posted* after a cutoff, and an old comment touched up
    yesterday is not a new post. `rows` must be chronological -- see `load_log`.
    """
    first: dict[int, str] = {}
    for row in rows:
        m = COMMENT_RE.match(row["action"])
        if m:
            first.setdefault(int(m.group(1)), row["date"])
    return first


def review_updated_at(rows: list[dict[str, str]]) -> dict[int, str]:
    """{review id: date of its LAST post-submission edit} (`UPDATE_RE`), if any.

    Complements `first_submitted_at`: that is when the review was turned in,
    this is the most recent time its author changed it afterwards. A reviewer
    asked to revisit their review after the authors' response shows up here or
    nowhere. `rows` must be chronological -- see `load_log`.
    """
    last: dict[int, str] = {}
    for row in rows:
        m = UPDATE_RE.match(row["action"])
        if m:
            last[int(m.group(1))] = row["date"]
    return last


def parse_date(value: str) -> datetime:
    """A log `date` as an aware datetime; seconds may be omitted.

    Always compare these, never the strings: the offset is part of the value,
    and string order stops being time order the moment two offsets differ.
    """
    value = value.strip()
    for fmt in (DATE_FORMAT, "%Y-%m-%d %H:%M %z"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    raise ValueError(f"unrecognised timestamp {value!r}; expected e.g. '2026-09-22 11:00:00 -0400'")


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


# "Response 1843 (Revision) submitted: attachments" / "Response 2261 (Revision)
# edited draft: text" / "Response 2255 (Revision) deleted". HotCRP stores a
# response as a comment, one per paper per response round, and logs each save
# as the verb plus the fields that save changed. "attachments" in the field
# list means the attachment set changed -- the log never says whether a file
# was added or removed, so attachment presence is an inference (see
# `replay_responses`).
RESPONSE_RE = re.compile(
    r"^Response (\d+) \(([^)]*)\) (started draft|edited draft|submitted|edited|deleted)(?::\s*(.*))?$"
)
RESPONSE_SUBMITTED_VERBS = frozenset({"submitted", "edited"})


@dataclass
class Response:
    """One paper's live response in one response round, as the log records it."""

    pid: int
    rid: int
    name: str
    submitted: bool = False
    attachment_saves: int = 0  # saves whose field list named "attachments"
    attachment_saves_after_submit: int = 0  # ... of which came on a submitted response
    last_saved_at: str = ""

    @property
    def attached(self) -> bool:
        """Whether the response ever carried an attachment.

        A removal also logs "attachments", so a response edited that way after
        submission could in principle be empty again; callers report
        `attachment_saves_after_submit` rather than pretend the log can tell.
        """
        return self.attachment_saves > 0


def replay_responses(rows: list[dict[str, str]], name: str | None = None) -> dict[int, Response]:
    """Replay the response saves into {pid: Response} for one response round.

    `name` is the response round as HotCRP prints it in parentheses
    ("Revision"); None keeps every round, in which case a paper with responses
    in two rounds keeps the one saved last. A deleted response is dropped; a
    re-created one (new id) replaces it. "submitted"/"edited" leave the
    response submitted, "started draft"/"edited draft" leave it a draft.
    """
    live: dict[int, Response] = {}
    for row in rows:
        m = RESPONSE_RE.match(row["action"])
        if not m or not row.get("paper"):
            continue
        rid, rname, verb, fields = int(m.group(1)), m.group(2), m.group(3), m.group(4) or ""
        if name is not None and rname != name:
            continue
        pid = int(row["paper"])
        if verb == "deleted":
            if pid in live and live[pid].rid == rid:
                del live[pid]
            continue
        resp = live.get(pid)
        if resp is None or resp.rid != rid:
            resp = live[pid] = Response(pid=pid, rid=rid, name=rname)
        resp.submitted = verb in RESPONSE_SUBMITTED_VERBS
        if "attachments" in [f.strip() for f in fields.split(",")]:
            resp.attachment_saves += 1
            if verb == "edited":
                resp.attachment_saves_after_submit += 1
        resp.last_saved_at = row.get("date", "")
    return live
