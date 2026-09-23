"""Tag each paper by whether its PC/reserve authors turned in their own reviews on time.

    python -m scripts.timeliness_tags
    python -m scripts.timeliness_tags --cutoff '2026-09-22 11:00:00 -0400'
    python -m scripts.timeliness_tags --extensions data/curated/review_extensions.csv

Chairs-only tags that delay review release and shorten the rebuttal period
for papers whose authors are late reviewers:

  ~~ontime        every PC/reserve author had submitted every R1 review they
                  hold by `--cutoff` -- or the paper has no such author
  ~~onedaylate    the last of them finished within 24 hours after the cutoff
  ~~twodayslate   ... within the second 24 hours, and so on

  ~~delayedmissingreview
                  provisional: an author still owes a review, so the paper's
                  release is held but its day is not known yet

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

**Several authors.** A paper waits for all of its PC/reserve authors. While
any of them is unfinished and not exempt the paper is blocked and takes
`~~delayedmissingreview` rather than a day; otherwise it takes the latest
author's day.

**The day tags stick; the blocked tag does not.** A paper that already carries
a day tag in the paper export (`--data`) is never re-tagged, so the upload only
ever adds a day tag to a paper that has none. Because the day comes from log
timestamps rather than from when this runs, a missed day catches up, and a
stale export only repeats a tag, never contradicts it.
`~~delayedmissingreview` is the exception -- it is a placeholder for a day not
yet known, so a paper carrying it is still decided, and every paper no longer
blocked is `cleartag`ged whether or not the export shows the tag. Clearing a
tag a paper does not have changes nothing, and doing it unconditionally is what
keeps two runs against one stale export from leaving a paper carrying both.

Writes `--upload` (`paper,action,email,tag,round`: every `cleartag` first,
then one `tag` row per newly decided or newly held paper) and `--report`
(every paper, its authors' states and any
name-only matches). stdout gives the counts and the reviewers holding papers
up. Offline and instant; nothing is uploaded. Both outputs name late
reviewers, so they are gitignored with the other outputs.
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
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from reviewer_match import hotcrp_log
from reviewer_match import paper_matching
from reviewer_match import pc_membership
from reviewer_match import review_scores
from reviewer_match.roster import load_roster
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

TAG_PREFIX = "~~"
ON_TIME = "ontime"
# Provisional, and the one tag here that never sticks: it marks a paper blocked
# by an author who still owes a review, and is cleared and replaced by a day tag
# as soon as the day is known.
BLOCKED_TAG = "delayedmissingreview"
NUMBER_WORDS = (
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen", "twenty",
)
TIMELINESS_RE = re.compile(r"^(ontime|onedaylate|\w+dayslate)$")

# The two report statuses of a paper still waiting on one of its authors: it
# takes the placeholder now, or already carries it.
BLOCKED_STATUSES = ("blocked", "still blocked")

UPLOAD_HEADER = ["paper", "action", "email", "tag", "round"]
EXTENSION_HEADER = ["email", "status", "date", "note"]
EXTENSION = "extension"
LATE = "late"
REPORT_FIELDS = [
    "paper", "title", "existing_tag", "new_tag", "status",
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

    These are the tags that stick: once a paper has one, it is never re-tagged.
    """
    return bool(TIMELINESS_RE.match(name))


def is_timeliness_tag(name: str) -> bool:
    """Whether a bare tag name is one of ours -- a day tag or the blocked placeholder."""
    return name == BLOCKED_TAG or is_day_tag(name)


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


def decide(
    papers: list[dict],
    author_keys: dict[int, list[str]],
    states: dict[str, ReviewerState],
) -> list[dict]:
    """One report row per paper, in pid order; `new_tag` is set only for an upload row.

    Only a day tag makes a paper already tagged: the blocked placeholder is a
    stand-in for a day not yet known, so a paper carrying it is decided as soon
    as its authors finish.
    """
    rows = []
    for paper in sorted(papers, key=lambda p: p["pid"]):
        pid = paper["pid"]
        authors = author_keys.get(pid, [])
        existing = existing_timeliness_tags(paper)
        days, blockers = paper_days(authors, states)
        if any(is_day_tag(t) for t in existing):
            status, new_tag = "already tagged", ""
        elif days is None:
            status, new_tag = ("still blocked", "") if BLOCKED_TAG in existing \
                else ("blocked", TAG_PREFIX + BLOCKED_TAG)
        else:
            status, new_tag = "tagged", TAG_PREFIX + tier_tag(days)
        rows.append({
            "paper": pid,
            "title": paper.get("title") or "",
            "existing_tag": " ".join(TAG_PREFIX + t for t in existing),
            "new_tag": new_tag,
            "status": status,
            "reviewer_authors": "; ".join(
                f"{a}: {states[a].describe() if a in states else tier_tag(0) + ' (no R1 reviews)'}"
                for a in authors
            ),
            "blocked_by": "; ".join(blockers),
            "name_only_matches": "",
        })
    return rows


def upload_rows(report: list[dict]) -> list[list[object]]:
    """The HotCRP delta: clear the placeholder off every unblocked paper, then tag.

    The clear is unconditional rather than driven by the export's tags, so a
    second run against one stale paper export cannot leave a paper carrying both
    the placeholder and its day. A paper already carrying a day tag is cleared
    even if it is blocked again today: the day tag is the settled answer, and
    nothing later would clear a placeholder left beside it.
    """
    clears = [[r["paper"], "cleartag", "", TAG_PREFIX + BLOCKED_TAG, ""]
              for r in report if r["status"] not in BLOCKED_STATUSES]
    tags = [[r["paper"], "tag", "", r["new_tag"], ""] for r in report if r["new_tag"]]
    return clears + tags


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
) -> Evaluation:
    """Replay the log against the rosters and decide every eligible paper."""
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

    report = decide(papers, author_keys, states)
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
    )


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


def evaluate_from_args(args) -> Evaluation:
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
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_arguments(parser)
    parser.add_argument("--upload", default=DEFAULT_UPLOAD, help="HotCRP bulk-assignment file to write")
    parser.add_argument("--report", default=DEFAULT_REPORT, help="per-paper report to write")
    args = parser.parse_args()

    try:
        ev = evaluate_from_args(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    report, states, author_keys = ev.report, ev.states, ev.author_keys
    cutoff, exempt_after = ev.cutoff, ev.exempt_after

    write_csv(args.upload, UPLOAD_HEADER, upload_rows(report))
    write_csv(args.report, REPORT_FIELDS, [[row[k] for k in REPORT_FIELDS] for row in report])
    print(f"Wrote {args.upload}\nWrote {args.report}", file=sys.stderr)

    status_counts = Counter(r["status"] for r in report)
    tag_counts = Counter(r["new_tag"] for r in report if r["new_tag"])
    print(f"Cutoff {first_submit_label(cutoff)}; exempt if assigned an R1 review after {exempt_after}.")
    print(f"{len(report)} papers: {status_counts['tagged']} newly tagged with a day, "
          f"{status_counts['already tagged']} already carrying one, "
          f"{status_counts['blocked']} newly blocked, "
          f"{status_counts['still blocked']} blocked already.")
    # ~~ontime first, then the days in order, then the placeholder: it is not a verdict.
    def tag_order(kv):
        return kv[0] == TAG_PREFIX + BLOCKED_TAG, kv[0] != TAG_PREFIX + ON_TIME, kv[0]
    for tag, n in sorted(tag_counts.items(), key=tag_order):
        print(f"  {tag:24s} {n}")
    no_author = sum(1 for r in report if not author_keys[r["paper"]] and r["status"] == "tagged")
    print(f"  ({no_author} of the new day tags are papers with no PC/reserve author)")

    held_up: Counter = Counter()
    for row in report:
        if row["status"] in BLOCKED_STATUSES:
            held_up.update(row["blocked_by"].split("; "))
    if held_up:
        print(f"\n{len(held_up)} reviewer(s) holding papers up:")
        print(f"# outstanding  papers  reviewer")
        for key in sorted(held_up, key=lambda k: (-states[k].outstanding, -held_up[k], k)):
            s = states[key]
            print(f"{s.outstanding:>6}/{s.held:<6} {held_up[key]:>6}  {key}"
                  + ("  (marked late)" if s.reason == "marked late" else ""))

    # Only day tags conflict; a day tag beside the placeholder is the stale
    # case this run clears, not a contradiction.
    multi = [r["paper"] for r in report
             if sum(is_day_tag(t.lstrip(TAG_PREFIX)) for t in r["existing_tag"].split()) > 1]
    if multi:
        print(f"\nwarning: {len(multi)} paper(s) carry more than one day tag: "
              f"{', '.join(map(str, multi))}", file=sys.stderr)
    n_name_only = sum(1 for r in report if r["name_only_matches"])
    if n_name_only:
        print(f"note: {n_name_only} paper(s) have an author matching a reviewer by name but not email; "
              f"not enforced, see name_only_matches in {args.report}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
