"""Tag each paper by whether its PC/reserve authors turned in their own reviews on time.

    python -m scripts.timeliness_tags
    python -m scripts.timeliness_tags --min-reviews 5 --min-decided 3 --recent-after 2026-09-13
    python -m scripts.timeliness_tags --cutoff '2026-09-22 11:00:00 -0400' --delay-days 4
    python -m scripts.timeliness_tags --extensions data/curated/review_extensions.csv

Chairs-only tags that delay review release and shorten the rebuttal period
for papers whose authors are late reviewers. Two buckets:

  ~~ontime          the paper is NoRevision, or every PC/reserve author had
                    submitted every R1 review they hold within `--delay-days`
                    24-hour windows of `--cutoff` (fewer than 4 days late by
                    default: `~~threedayslate` was the last on-time day), or
                    the paper has no such author
  ~~revisiondelay   otherwise -- including a paper still waiting on an author
                    who has not finished, since nothing is waited for any more

**NoRevision papers are never delayed.** The revision decision is
`scripts/revision_tags.py`'s, off the same `--reviews`, log, export and bar
flags, so the two uploads cannot disagree about a paper.

**Who.** A PC/reserve reviewer is anyone on the PC or reserve roster (HotCRP
membership check on, so `~~ex-rr` promotions count) plus any address holding a
live R1 review. Authorship is matched on **email only**, against both the
roster address and the HotCRP address; an author whose name matches a
reviewer but whose email does not is listed in the report and not enforced.

**Which reviews.** Only the R1 reviews the reviewer holds today, on papers
still under review. TRC reviews are ignored entirely -- they neither block a
paper nor excuse one -- and neither a review pulled off someone nor one on a
paper since desk-rejected or withdrawn (absent from `--data`, or tagged
`desk-reject`) counts against them.

**When.** A reviewer finishes when their last held review was *first*
submitted. Later edits and resubmissions do not move that time. Log timestamps
are compared as aware datetimes, never as strings.

**Exempt, i.e. on time.** A reviewer holding an R1 review assigned after
`--exempt-after`, or listed with status `extension` in `--extensions`, counts
as on time whether they have finished or not. Status `late` there overrides
that: with a blank date they block their papers until the row is changed; with
a date they finish no earlier than that date (11:00 at the cutoff's offset
when no time is given). Each email in the file is matched against both address
spaces, and one that matches nobody is a warning.

**Several authors.** A paper takes its latest author's day, and is blocked
while any of them is unfinished and not exempt.

**Recomputed every run.** Nothing sticks: every paper is decided afresh from
the log, and a paper whose export tag flips between the two buckets is listed
on stderr. The upload clears the retired tags (`~~onedaylate`,
`~~twodayslate`, ..., `~~delayedmissingreview`) from every paper with one
`all,cleartag` row each, then clears each paper's opposite bucket, then tags.
Clearing a tag a paper does not have changes nothing.

**Unassignments.** `scripts/revision_tags.py` unassigns the outstanding R1
reviews on papers decided NoRevision early. Removing a reviewer's outstanding
reviews can finish them, and so change the papers they author; stdout lists
every reviewer whose state those removals would change, and each authored
paper's tag before and after. The tags themselves follow the log as it
stands: once the unassignment is uploaded, the next export shows it.

Writes `--upload` (`paper,action,email,tag,round`: every `cleartag` first,
then one `tag` row per paper) and `--report` (every paper, its authors' states,
its revision decision and any name-only matches). stdout gives the counts and
the reviewers holding papers up. Offline and instant; nothing is uploaded. Both
outputs name late reviewers, so they are gitignored with the other outputs.
"""

from __future__ import annotations

from reviewer_match.paths import assignment_path, curated_path, input_path, report_path

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from reviewer_match import hotcrp_log
from reviewer_match import paper_matching
from reviewer_match import pc_membership
from reviewer_match import review_scores
from reviewer_match.roster import load_roster
from scripts import revision_tags
from scripts.assign_paper_leads import write_csv

DEFAULT_LOG = input_path("hpca2027-log.csv")
DEFAULT_DATA = input_path("hpca2027-data.json")
DEFAULT_EXTENSIONS = curated_path("review_extensions.csv")
DEFAULT_UPLOAD = assignment_path("timeliness_tags_upload.csv")
DEFAULT_REPORT = report_path("timeliness_papers.csv")
# 10:00 CDT / 11:00 EDT; the log writes -0400.
DEFAULT_CUTOFF = "2026-09-22 11:00:00 -0400"
# A review assigned on any later day exempts its reviewer.
DEFAULT_EXEMPT_AFTER = "2026-09-13"
# A paper whose authors finished this many 24-hour windows late, or later, is delayed.
DEFAULT_DELAY_DAYS = 4

TAG_PREFIX = "~~"
ON_TIME = "ontime"
DELAY_TAG = "revisiondelay"
BUCKETS = (ON_TIME, DELAY_TAG)
# Retired: the placeholder a paper carried while an author still owed a review.
# Cleared from every paper, with the retired day tags.
BLOCKED_TAG = "delayedmissingreview"
NUMBER_WORDS = (
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen", "twenty",
)
TIMELINESS_RE = re.compile(r"^(ontime|onedaylate|\w+dayslate)$")

# The report status of a delayed paper still waiting on one of its authors.
# `scripts/timeliness_emails.py` drafts one email for each.
BLOCKED_STATUSES = ("blocked",)

UPLOAD_HEADER = ["paper", "action", "email", "tag", "round"]
EXTENSION_HEADER = ["email", "status", "date", "note"]
EXTENSION = "extension"
LATE = "late"
REPORT_FIELDS = [
    "paper", "title", "revision", "existing_tag", "new_tag", "status",
    "reviewer_authors", "blocked_by", "name_only_matches",
]


def tier_tag(days: int) -> str:
    """The bare tag name for a paper `days` 24-hour windows late."""
    if days <= 0:
        return ON_TIME
    if days == 1:
        return "onedaylate"
    if days <= len(NUMBER_WORDS):
        return f"{NUMBER_WORDS[days - 1]}dayslate"
    return f"{days}dayslate"


def is_day_tag(name: str) -> bool:
    """Whether a bare tag name (as `pc_membership.tag_names` gives it) states a day.

    `ontime` is one of these and still in use; the rest are retired.
    """
    return bool(TIMELINESS_RE.match(name))


def is_timeliness_tag(name: str) -> bool:
    """Whether a bare tag name is one of ours, current or retired."""
    return name in (BLOCKED_TAG, DELAY_TAG) or is_day_tag(name)


def days_late(completed_at: datetime | None, cutoff: datetime) -> int:
    """0 at or before the cutoff, else the 24-hour window it fell in, from 1."""
    if completed_at is None or completed_at <= cutoff:
        return 0
    return math.ceil((completed_at - cutoff) / timedelta(days=1))


@dataclass(frozen=True)
class ReviewerState:
    """Where one reviewer stands; `days` is None while they hold their papers up."""

    held: int
    outstanding: int
    completed_at: datetime | None
    days: int | None
    reason: str

    def describe(self) -> str:
        if self.days is None:
            return f"blocking ({self.reason}; {self.outstanding}/{self.held} outstanding)"
        return f"{tier_tag(self.days)} ({self.reason})"


def reviewer_status(
    reviews: list[tuple[int, hotcrp_log.Review]],
    submitted: set[int],
    first_submit: dict[int, str],
    *,
    cutoff: datetime,
    exempt_after: date,
    override: tuple[str, datetime | None] | None = None,
) -> ReviewerState:
    """One reviewer's state from their live R1 `(rid, Review)` pairs.

    Precedence: an `extension` excuses them outright; a `late` row is the
    chair's judgement and beats the post-`exempt_after` exemption; then the
    exemption; then the reviews themselves.
    """
    held = len(reviews)
    outstanding = sum(rid not in submitted for rid, _ in reviews)
    done = [hotcrp_log.parse_date(first_submit[rid]) for rid, _ in reviews if rid in submitted]
    completed_at = max(done) if done and not outstanding else None

    def state(days, reason):
        return ReviewerState(held, outstanding, completed_at, days, reason)

    if override and override[0] == EXTENSION:
        return state(0, "extension")
    if override and override[0] == LATE:
        if outstanding or override[1] is None:
            return state(None, "marked late")
        finished = max(completed_at, override[1]) if completed_at else override[1]
        return state(days_late(finished, cutoff), "marked late")
    if any(hotcrp_log.parse_date(r.assigned_at).date() > exempt_after for _, r in reviews):
        return state(0, f"assigned after {exempt_after.isoformat()}")
    if outstanding:
        return state(None, "unfinished")
    if not held:
        return state(0, "no R1 reviews")
    return state(days_late(completed_at, cutoff), f"finished {first_submit_label(completed_at)}")


def first_submit_label(when: datetime | None) -> str:
    return when.strftime("%Y-%m-%d %H:%M %z") if when else ""


def paper_days(authors: list[str], states: dict[str, ReviewerState]) -> tuple[int | None, list[str]]:
    """(the paper's day or None while blocked, [blocking authors]).

    An author with no state holds no R1 review, so is on time.
    """
    blockers = [a for a in authors if a in states and states[a].days is None]
    if blockers:
        return None, blockers
    return max((states[a].days for a in authors if a in states), default=0), []


def existing_timeliness_tags(paper: dict) -> list[str]:
    """The timeliness tags a paper already carries, sorted, bare -- placeholder included."""
    return sorted(t for t in pc_membership.tag_names(" ".join(paper.get("tags") or []))
                  if is_timeliness_tag(t))


def bucket(days: int | None, revision: str | None, delay_days: int) -> str:
    """ON_TIME or DELAY_TAG for a paper `days` late (None: blocked) with this revision tag."""
    if revision == revision_tags.NO_REVISION_TAG:
        return ON_TIME
    return DELAY_TAG if days is None or days >= delay_days else ON_TIME


def decide(
    papers: list[dict],
    author_keys: dict[int, list[str]],
    states: dict[str, ReviewerState],
    revision: dict[int, str | None] | None = None,
    delay_days: int = DEFAULT_DELAY_DAYS,
) -> list[dict]:
    """One report row per paper, in pid order, each with its `new_tag`.

    Without `revision` no paper counts as NoRevision.
    """
    revision = revision or {}
    rows = []
    for paper in sorted(papers, key=lambda p: p["pid"]):
        pid = paper["pid"]
        authors = author_keys.get(pid, [])
        days, blockers = paper_days(authors, states)
        tag = bucket(days, revision.get(pid), delay_days)
        if revision.get(pid) == revision_tags.NO_REVISION_TAG:
            status = "no revision"
        elif days is None:
            status = "blocked"
        else:
            status = "late" if tag == DELAY_TAG else "on time"
        rows.append({
            "days": days,
            "paper": pid,
            "title": paper.get("title") or "",
            "revision": revision.get(pid) or "",
            "existing_tag": " ".join(TAG_PREFIX + t for t in existing_timeliness_tags(paper)),
            "new_tag": TAG_PREFIX + tag,
            "status": status,
            "reviewer_authors": "; ".join(
                f"{a}: {states[a].describe() if a in states else tier_tag(0) + ' (no R1 reviews)'}"
                for a in authors
            ),
            "blocked_by": "; ".join(blockers),
            "name_only_matches": "",
        })
    return rows


def retired_tags(report: list[dict]) -> list[str]:
    """Every retired tag an earlier run could have written, twiddled.

    The spelled day tags, numeric ones up to the latest day this run computes
    (an earlier run cannot have computed a later one, since first submissions
    never move), and the blocked placeholder.
    """
    max_days = max([len(NUMBER_WORDS)] + [r["days"] for r in report if r.get("days") is not None])
    return [TAG_PREFIX + tier_tag(d) for d in range(1, max_days + 1)] + [TAG_PREFIX + BLOCKED_TAG]


def upload_rows(report: list[dict]) -> list[list[object]]:
    """The HotCRP delta: clear the retired tags everywhere, clear each paper's
    opposite bucket, then tag every paper.

    The retired tags go in one `all,cleartag` row each rather than one per
    paper, the `generate_clear_uploads.py` shape.
    """
    clears: list[list[object]] = [["all", "cleartag", "", t, ""] for t in retired_tags(report)]
    clears += [[r["paper"], "cleartag", "", TAG_PREFIX + other, ""]
               for r in report for other in BUCKETS if TAG_PREFIX + other != r["new_tag"]]
    tags = [[r["paper"], "tag", "", r["new_tag"], ""] for r in report]
    return clears + tags


def flips(report: list[dict]) -> list[tuple[int, str, str]]:
    """[(pid, bucket the export shows, new bucket)] where the two differ."""
    out = []
    for r in report:
        shown = [t for t in r["existing_tag"].split() if t.lstrip("~") in BUCKETS]
        if len(shown) == 1 and shown[0] != r["new_tag"]:
            out.append((r["paper"], shown[0], r["new_tag"]))
    return out


def parse_override_date(value: str, cutoff: datetime) -> datetime | None:
    """A `late` row's date: a full timestamp, or a bare day at the cutoff's time and offset."""
    value = value.strip()
    if not value:
        return None
    try:
        day = date.fromisoformat(value)
    except ValueError:
        return hotcrp_log.parse_date(value)
    return datetime.combine(day, cutoff.timetz())


def load_extensions(path: str, cutoff: datetime) -> dict[str, tuple[str, datetime | None]]:
    """{email: (status, date)} from the chair's file, created header-only if missing."""
    if not os.path.exists(path):
        write_csv(path, EXTENSION_HEADER, [])
        print(f"Created {path} (empty; add rows for approved extensions)", file=sys.stderr)
        return {}
    out: dict[str, tuple[str, datetime | None]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = {"email", "status"} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: no {', '.join(sorted(missing))} column")
        for line, row in enumerate(reader, start=2):
            email = (row.get("email") or "").strip().lower()
            if not email:
                continue
            status = (row.get("status") or "").strip().lower() or EXTENSION
            if status not in (EXTENSION, LATE):
                raise ValueError(f"{path}:{line}: status {status!r}; expected {EXTENSION} or {LATE}")
            try:
                when = parse_override_date(row.get("date") or "", cutoff) if status == LATE else None
            except ValueError as exc:
                raise ValueError(f"{path}:{line}: {exc}") from None
            out[email] = (status, when)
    return out


def load_pc_and_reserves(pcinfo_path: str | None) -> list:
    """Both rosters. A roster that fails to load is fatal: every author would read as on time."""
    records = []
    for role in ("reserve", "reviewer"):
        records += load_roster(role, pcinfo_path=pcinfo_path)
    return records


@dataclass(frozen=True)
class Evaluation:
    """Everything the log and the paper export say about where each paper stands.

    `scripts/timeliness_emails.py` reads the same object, so the tags and the
    emails to authors cannot name different people.
    """

    papers: dict[int, dict]                    # pid -> the paper export record
    report: list[dict]                         # decide()'s rows, in pid order
    states: dict[str, ReviewerState]           # roster key -> where they stand
    author_keys: dict[int, list[str]]          # pid -> its reviewer-authors' keys
    author_source: dict[int, dict[str, str]]   # pid -> {key: the author row's address}
    display: dict[str, str]                    # roster key -> roster display name
    name_only: dict[int, list[str]]            # pid -> name-but-not-email matches
    cutoff: datetime
    exempt_after: date
    # What `reviewer_status` was computed from, so a caller can recompute one
    # reviewer with some of their reviews taken away (`unassign_effects`).
    by_reviewer: dict[str, list[tuple[int, hotcrp_log.Review]]] = field(default_factory=dict)
    submitted: set[int] = field(default_factory=set)
    first_submit: dict[int, str] = field(default_factory=dict)
    overrides: dict[str, tuple[str, datetime | None]] = field(default_factory=dict)


def evaluate(
    *,
    log: str,
    data: str,
    exclude_pids: str,
    cutoff: datetime,
    exempt_after: date,
    extensions: dict[str, tuple[str, datetime | None]],
    extensions_path: str,
    pcinfo: str | None,
    revision: dict[int, str | None] | None = None,
    delay_days: int = DEFAULT_DELAY_DAYS,
) -> Evaluation:
    """Replay the log against the rosters and decide every eligible paper.

    `revision` is `revision_tags.decide`'s {pid: tag}; without it no paper is
    spared as NoRevision.
    """
    rows = hotcrp_log.load_log(log)
    live, _ = hotcrp_log.replay_assignments(rows)
    submitted = hotcrp_log.submitted_review_ids(rows)
    first_submit = hotcrp_log.first_submitted_at(rows)
    eligible, _ = review_scores.eligible_pids(data, paper_matching.parse_exclude_pids(exclude_pids))
    # A desk rejection or withdrawal leaves the paper's reviews live in the log,
    # but a review nobody will ever read must not count against its reviewer --
    # neither as outstanding nor towards when they finished.
    by_reviewer: dict[str, list[tuple[int, hotcrp_log.Review]]] = defaultdict(list)
    for rid, review in live.items():
        if review.round == review_scores.REVIEW_ROUND and review.pid in eligible:
            by_reviewer[review.email].append((rid, review))

    # Every address a PC/reserve reviewer is known by -> their HotCRP address.
    key_of: dict[str, str] = {email: email for email in by_reviewer}
    names: dict[frozenset[str], set[str]] = defaultdict(set)
    display: dict[str, str] = {}
    try:
        records = load_pc_and_reserves(pcinfo)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(f"could not load the rosters: {exc}") from None
    for record in records:
        key = record.hotcrp_email.lower()
        key_of[key] = key
        key_of.setdefault(record.email.lower(), key)
        names[pc_membership.token_set(record.name)].add(key)
        display.setdefault(key, record.name)

    overrides: dict[str, tuple[str, datetime | None]] = {}
    for email, override in extensions.items():
        if email in key_of:
            overrides[key_of[email]] = override
        else:
            print(f"warning: {extensions_path}: {email} is on no roster and holds no R1 review;"
                  " ignored", file=sys.stderr)

    states = {
        key: reviewer_status(by_reviewer.get(key, []), submitted, first_submit,
                             cutoff=cutoff, exempt_after=exempt_after, override=overrides.get(key))
        for key in set(key_of.values())
    }

    with open(data, encoding="utf-8") as f:
        papers = [p for p in json.load(f) if p["pid"] in eligible]
    author_keys: dict[int, list[str]] = {}
    author_source: dict[int, dict[str, str]] = {}
    name_only: dict[int, list[str]] = {}
    for paper in papers:
        matched, source, candidates = [], {}, []
        for author in paper.get("authors") or []:
            email = (author.get("email") or "").strip().lower()
            if email in key_of:
                if key_of[email] not in matched:
                    matched.append(key_of[email])
                    # The address this paper lists them under, which is the one
                    # their co-authors will recognise.
                    source[key_of[email]] = email
                continue
            tokens = pc_membership.token_set(f"{author.get('given_name', '')} {author.get('family_name', '')}")
            for key in sorted(names.get(tokens, ())):
                candidates.append(f"{email or '(no email)'} ~ {key}")
        author_keys[paper["pid"]] = matched
        author_source[paper["pid"]] = source
        name_only[paper["pid"]] = [c for c in candidates if c.split(" ~ ")[1] not in matched]

    report = decide(papers, author_keys, states, revision, delay_days)
    for row in report:
        row["name_only_matches"] = "; ".join(name_only[row["paper"]])
    return Evaluation(
        papers={p["pid"]: p for p in papers},
        report=report,
        states=states,
        author_keys=author_keys,
        author_source=author_source,
        display=display,
        name_only=name_only,
        cutoff=cutoff,
        exempt_after=exempt_after,
        by_reviewer=dict(by_reviewer),
        submitted=submitted,
        first_submit=first_submit,
        overrides=overrides,
    )


def unassign_effects(
    ev: Evaluation,
    pairs: list[tuple[int, str]],
    revision: dict[int, str | None],
    delay_days: int,
) -> list[tuple[str, ReviewerState, ReviewerState, list[tuple[int, str, str]]]]:
    """[(reviewer, state now, state without the `pairs` reviews, [(pid, tag now, tag after)])].

    Only reviewers whose day changes are listed; their authored papers are
    re-bucketed with every affected author's new state at once.
    """
    removed: dict[str, set[int]] = defaultdict(set)
    for pid, email in pairs:
        removed[email].add(pid)
    after = dict(ev.states)
    changed = []
    for key in sorted(removed):
        if key not in ev.states:
            continue
        kept = [(rid, r) for rid, r in ev.by_reviewer.get(key, []) if r.pid not in removed[key]]
        state = reviewer_status(kept, ev.submitted, ev.first_submit, cutoff=ev.cutoff,
                                exempt_after=ev.exempt_after, override=ev.overrides.get(key))
        if state.days != ev.states[key].days:
            after[key] = state
            changed.append(key)
    out = []
    for key in changed:
        papers = []
        for pid in sorted(p for p, keys in ev.author_keys.items() if key in keys):
            before = bucket(paper_days(ev.author_keys[pid], ev.states)[0], revision.get(pid), delay_days)
            now = bucket(paper_days(ev.author_keys[pid], after)[0], revision.get(pid), delay_days)
            papers.append((pid, TAG_PREFIX + before, TAG_PREFIX + now))
        out.append((key, ev.states[key], after[key], papers))
    return out


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """The inputs `evaluate()` needs, shared with `scripts/timeliness_emails.py`."""
    parser.add_argument("--log", default=DEFAULT_LOG, help="HotCRP action log CSV export")
    parser.add_argument("--data", default=DEFAULT_DATA, help="HotCRP paper export JSON (authors and current tags)")
    parser.add_argument("--exclude-pids", default="", help="comma-separated paper IDs to leave out")
    parser.add_argument("--cutoff", default=DEFAULT_CUTOFF,
                        help=f"on-time deadline, with offset (default: {DEFAULT_CUTOFF!r})")
    parser.add_argument("--exempt-after", default=DEFAULT_EXEMPT_AFTER,
                        help="an R1 review assigned on a later day exempts its reviewer "
                             f"(default: {DEFAULT_EXEMPT_AFTER})")
    parser.add_argument("--extensions", default=DEFAULT_EXTENSIONS,
                        help="chair's email,status,date,note file (extension or late)")
    parser.add_argument("--pcinfo", default=pc_membership.DEFAULT_PCINFO, help="HotCRP user export")
    parser.add_argument("--no-pc-check", action="store_true",
                        help="skip the HotCRP PC-membership check when loading the rosters")


def evaluate_from_args(args, revision: dict[int, str | None] | None = None,
                       delay_days: int = DEFAULT_DELAY_DAYS) -> Evaluation:
    """`evaluate()` off a parsed `add_common_arguments` namespace.

    Raises ValueError for a bad flag and FileNotFoundError for a missing input,
    so each caller reports one way.
    """
    cutoff = hotcrp_log.parse_date(args.cutoff)
    exempt_after = date.fromisoformat(args.exempt_after)
    extensions = load_extensions(args.extensions, cutoff)
    for path in (args.log, args.data):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found")
    return evaluate(
        log=args.log, data=args.data, exclude_pids=args.exclude_pids, cutoff=cutoff,
        exempt_after=exempt_after, extensions=extensions, extensions_path=args.extensions,
        pcinfo=None if args.no_pc_check else args.pcinfo,
        revision=revision, delay_days=delay_days,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_arguments(parser)
    parser.add_argument("--reviews", default=revision_tags.DEFAULT_REVIEWS,
                        help="HotCRP review CSV export (for the revision decision)")
    review_scores.add_bar_arguments(parser)
    parser.add_argument("--delay-days", type=int, default=DEFAULT_DELAY_DAYS,
                        help="a paper this many 24-hour windows late, or blocked, is delayed "
                             f"(default {DEFAULT_DELAY_DAYS})")
    parser.add_argument("--upload", default=DEFAULT_UPLOAD, help="HotCRP bulk-assignment file to write")
    parser.add_argument("--report", default=DEFAULT_REPORT, help="per-paper report to write")
    args = parser.parse_args()

    try:
        if not os.path.exists(args.reviews):
            raise FileNotFoundError(f"{args.reviews} not found")
        decided = revision_tags.load_inputs(args)
        ev = evaluate_from_args(args, decided.decisions, args.delay_days)
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    report, states, author_keys = ev.report, ev.states, ev.author_keys
    cutoff, exempt_after = ev.cutoff, ev.exempt_after

    write_csv(args.upload, UPLOAD_HEADER, upload_rows(report))
    write_csv(args.report, REPORT_FIELDS, [[row[k] for k in REPORT_FIELDS] for row in report])
    print(f"Wrote {args.upload}\nWrote {args.report}", file=sys.stderr)

    status_counts = Counter(r["status"] for r in report)
    tag_counts = Counter(r["new_tag"] for r in report)
    print(f"Cutoff {first_submit_label(cutoff)}; exempt if assigned an R1 review after {exempt_after}; "
          f"delayed from {tier_tag(args.delay_days)}.")
    print(f"{len(report)} papers: {tag_counts[TAG_PREFIX + ON_TIME]} {TAG_PREFIX + ON_TIME}, "
          f"{tag_counts[TAG_PREFIX + DELAY_TAG]} {TAG_PREFIX + DELAY_TAG} "
          f"({status_counts['late']} late, {status_counts['blocked']} blocked on an unfinished author).")
    spared = sum(1 for r in report if r["status"] == "no revision"
                 and bucket(r["days"], None, args.delay_days) == DELAY_TAG)
    no_author = sum(1 for r in report if not author_keys[r["paper"]])
    print(f"  {spared} late or blocked paper(s) are {revision_tags.NO_REVISION_TAG} and so on time; "
          f"{no_author} paper(s) have no PC/reserve author.")

    held_up: Counter = Counter()
    for row in report:
        if row["status"] in BLOCKED_STATUSES:
            held_up.update(row["blocked_by"].split("; "))
    if held_up:
        print(f"\n{len(held_up)} reviewer(s) still unfinished, delaying papers they author:")
        print(f"# outstanding  papers  reviewer")
        for key in sorted(held_up, key=lambda k: (-states[k].outstanding, -held_up[k], k)):
            s = states[key]
            print(f"{s.outstanding:>6}/{s.held:<6} {held_up[key]:>6}  {key}"
                  + ("  (marked late)" if s.reason == "marked late" else ""))

    print()
    print("\n".join(review_scores.manual_report(decided.manual, decided.by_bar)))
    effects = unassign_effects(ev, decided.pairs, decided.decisions, args.delay_days)
    print(f"\nUnassigning {len(decided.pairs)} outstanding review(s) on early "
          f"{revision_tags.NO_REVISION_TAG} papers changes {len(effects)} reviewer(s)' late status"
          + (":" if effects else "."))
    for key, before, after, papers in effects:
        print(f"  {key}: {before.describe()} -> {after.describe()}")
        for pid, old, new in papers:
            print(f"    authors #{pid}: {old}" + (f" -> {new}" if new != old else " (unchanged)"))

    flipped = flips(report)
    if flipped:
        print(f"\nwarning: {len(flipped)} paper(s) change bucket against the export: "
              + ", ".join(f"#{pid} {old}->{new}" for pid, old, new in flipped), file=sys.stderr)
    n_name_only = sum(1 for r in report if r["name_only_matches"])
    if n_name_only:
        print(f"note: {n_name_only} paper(s) have an author matching a reviewer by name but not email; "
              f"not enforced, see name_only_matches in {args.report}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
