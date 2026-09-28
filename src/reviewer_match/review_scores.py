"""Read HotCRP's review export, and decide which papers clear the revision bar.

Two scripts ask the same question of the same file. `scripts/revision_cutoffs.py`
measures how many papers each candidate bar would catch, and
`scripts/assign_paper_leads.py` gives a lead to every paper the chosen bar lets
through. The bar lives here, once, so those two can never disagree about which
papers advance.

`data/inputs/hpca2027-reviews.csv` is HotCRP's review CSV export: submitted
reviews only, one row each, keyed on the reviewer's HotCRP address. **It does
not name a review's round.** TRC (Training Review Committee) reviews are
therefore found in the action log (`review_rounds`), and every consumer here
leaves them out of both the count and the average unless told otherwise.

A *net* is the average test plus an optional extra condition that spares a
paper somebody argues for (`NETS`). A paper with `min_reviews` or more
submitted reviews is **under the bar** when the net catches it. A paper with
`min_decided` up to `min_reviews - 1` reviews is decided early, by a simpler
rule: under the bar when every score is `EARLY_UNDER_MAX` or lower (reject or
weak reject). A paper with fewer than `min_decided` reviews always advances
(`Bar.advances`).

**An outstanding late-assigned review keeps a paper advancing.** An R1 review
assigned after `--recent-after` and still outstanding is let play out: an
early paper under the bar advances, and a paper the net puts under the bar
*keeps* RevisionAdvance if the export already shows it (it is never promoted
for one). Otherwise the bar follows the reviews as they stand, so a
RevisionAdvance paper that new reviews put under the bar becomes NoRevision.

**A hand-set revision tag wins over both** (`manual_revision_tags`). Once
anyone -- a chair or a PC member -- has changed a paper's RevisionAdvance or
NoRevision tag in HotCRP's UI, the paper keeps what their last such edit
left, whatever its scores say and whatever a bulk upload did since.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from fractions import Fraction

from . import hotcrp_log
from . import paper_matching
from . import pc_membership

SCORE_FIELD = "Pre-Rebuttal Overall merit"
REVIEW_ROUND = "R1"
TRAINING_ROUND = "TRC"
# As `pc_membership.tag_names` spells it: `~~desk-reject#0` in the export.
DESK_REJECT_TAG = "desk-reject"

COMPARATORS = ("le", "lt")
COMPARATOR_SYMBOLS = {"le": "<=", "lt": "<"}

DEFAULT_MIN_REVIEWS = 4
DEFAULT_MIN_DECIDED = 3
# An early paper (min_decided <= reviews < min_reviews) is under the bar when no
# score exceeds this: all reject (1) or weak reject (2).
EARLY_UNDER_MAX = 2
DEFAULT_BAR_NET = "one3"
DEFAULT_BAR_CUTOFF = 2.5
DEFAULT_BAR_COMPARATOR = "le"

# `Bar.advances` reasons.
FEW_REVIEWS = "few-reviews"
OVER_BAR = "over-bar"
RECENT = "recent-review"
MANUAL_ADVANCE = "manual-advance"   # tagged RevisionAdvance by hand
MANUAL_UNTAGGED = "manual-untagged"  # both tags removed by hand: advances, untagged

# The revision tags `scripts/revision_tags.py` writes. They live here because
# `decide_paper` reads the export's copy of them back.
ADVANCE_TAG = "RevisionAdvance"
NO_REVISION_TAG = "NoRevision"

# (key, label, short label, extra condition on one paper's scores). The average
# test is applied to every net; this is what each one adds on top of it.
NETS = (
    ("average", "Average only", "Average only", lambda s: True),
    ("no4", "+ no score of 4 or better", "No score ≥4", lambda s: max(s) < 4),
    ("one3", "+ at most one score of 3 or better", "≤1 score ≥3",
     lambda s: sum(x >= 3 for x in s) <= 1),
    ("no3", "+ no score of 3 or better", "No score ≥3", lambda s: max(s) < 3),
)
NET_CONDITIONS = {key: condition for key, _, _, condition in NETS}
NET_LABELS = {key: label for key, label, _, _ in NETS}


@dataclass(frozen=True)
class SubmittedReview:
    """One row of the review export."""

    pid: int
    email: str  # HotCRP address, lower-cased
    score: int
    title: str


def parse_thresholds(value: str) -> list[float]:
    """Sorted, de-duplicated cutoffs from "1,1.5,2"; each must lie on the 1-5 scale."""
    thresholds = sorted({float(v) for v in value.split(",") if v.strip()})
    if not thresholds:
        raise ValueError("--thresholds names no cutoff")
    for t in thresholds:
        if not 1 <= t <= 5:
            raise ValueError(f"cutoff {t:g} is off the 1-5 overall-merit scale")
    return thresholds


def review_rounds(log_rows: list[dict[str, str]]) -> tuple[set[tuple[int, str]], Counter]:
    """({(pid, email)} of submitted TRC reviews, Counter{pid: R1 reviews outstanding}).

    Both come from one replay: the TRC set is how a training review is told
    apart in an export that does not name rounds, and the outstanding count is
    what says whether a paper's average is still moving. Keyed on the review id
    throughout, so an address change never splits a review in two.
    """
    live, _ = hotcrp_log.replay_assignments(log_rows)
    submitted = hotcrp_log.submitted_review_ids(log_rows)
    training = {
        (review.pid, review.email)
        for rid, review in live.items()
        if review.round == TRAINING_ROUND and rid in submitted
    }
    outstanding = Counter(
        review.pid
        for rid, review in live.items()
        if review.round == REVIEW_ROUND and rid not in submitted
    )
    return training, outstanding


def recent_pids(log_rows: list[dict[str, str]], after: date) -> set[int]:
    """{pid} holding an outstanding R1 review assigned on a day after `after`.

    Such a paper is not decided early: the chairs let a freshly assigned review
    play out rather than cut it short.
    """
    live, _ = hotcrp_log.replay_assignments(log_rows)
    submitted = hotcrp_log.submitted_review_ids(log_rows)
    return {
        review.pid
        for rid, review in live.items()
        if review.round == REVIEW_ROUND and rid not in submitted
        and hotcrp_log.parse_date(review.assigned_at).date() > after
    }


def load_reviews(
    path: str, skip_pairs: set[tuple[int, str]] = frozenset()
) -> tuple[list[SubmittedReview], set[tuple[int, str]]]:
    """([SubmittedReview] in file order, {(pid, email)} skipped) from HotCRP's review CSV.

    A review whose (pid, lower-cased email) is in `skip_pairs` is left out.
    A row with no usable score is an error, not a skip: the export carries
    submitted reviews only, and quietly dropping one would move its paper's
    average and review count.
    """
    reviews: list[SubmittedReview] = []
    skipped: set[tuple[int, str]] = set()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = {"paper", "email", SCORE_FIELD} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: no {', '.join(sorted(missing))} column in the review export")
        for row in reader:
            pid = int(row["paper"])
            pair = (pid, row["email"].strip().lower())
            if pair in skip_pairs:
                skipped.add(pair)
                continue
            raw = (row[SCORE_FIELD] or "").strip()
            try:
                score = int(raw)
            except ValueError:
                raise ValueError(
                    f"{path}: review {row.get('review') or pair} has {SCORE_FIELD} {raw!r}"
                ) from None
            reviews.append(SubmittedReview(pid, pair[1], score, row.get("title") or ""))
    return reviews, skipped


def load_review_scores(
    path: str, skip_pairs: set[tuple[int, str]] = frozenset()
) -> tuple[dict[int, list[int]], dict[int, str], set[tuple[int, str]]]:
    """({pid: [scores]}, {pid: title}, {(pid, email)} skipped): `load_reviews` by paper."""
    reviews, skipped = load_reviews(path, skip_pairs)
    scores: dict[int, list[int]] = {}
    titles: dict[int, str] = {}
    for review in reviews:
        scores.setdefault(review.pid, []).append(review.score)
        titles.setdefault(review.pid, review.title)
    return scores, titles, skipped


def eligible_pids(data_path: str, exclude_pids: frozenset[int]) -> tuple[set[int], set[int]]:
    """({pid} still under review, {pid} desk-rejected) from the paper export.

    "Still under review" is the `submitted` policy every other paper-side tool
    here uses, minus `--exclude-pids`, minus papers tagged desk-rejected: HotCRP
    leaves those `submitted`, and a paper that is already out is no revision
    candidate.
    """
    with open(data_path, encoding="utf-8") as f:
        papers = json.load(f)
    eligible, desk_rejected = set(), set()
    for paper in papers:
        if paper_matching.selection_gaps(paper, "submitted", exclude_pids):
            continue
        if DESK_REJECT_TAG in pc_membership.tag_names(" ".join(paper.get("tags") or [])):
            desk_rejected.add(paper["pid"])
            continue
        eligible.add(paper["pid"])
    return eligible, desk_rejected


def average(scores: list[int]) -> Fraction:
    """Exact average, so a cutoff of 2.5 against 15/6 is never a float accident."""
    return Fraction(sum(scores), len(scores))


def caught(scores: list[int], threshold: float, comparator: str, condition) -> bool:
    """Whether a net with this cutoff, comparator and extra condition catches the paper."""
    avg, cut = average(scores), Fraction(str(threshold))
    below = avg <= cut if comparator == "le" else avg < cut
    return below and condition(scores)


@dataclass(frozen=True)
class Bar:
    """The revision bar: which net, at what cutoff, and the review-count floor."""

    net: str = DEFAULT_BAR_NET
    cutoff: float = DEFAULT_BAR_CUTOFF
    comparator: str = DEFAULT_BAR_COMPARATOR
    min_reviews: int = DEFAULT_MIN_REVIEWS
    min_decided: int | None = None  # None: no early decisions (= min_reviews)

    def __post_init__(self) -> None:
        if self.net not in NET_CONDITIONS:
            raise ValueError(f"unknown net {self.net!r}; expected one of {', '.join(NET_CONDITIONS)}")
        if self.comparator not in COMPARATORS:
            raise ValueError(f"unknown comparator {self.comparator!r}; expected le or lt")
        if not 1 <= self.cutoff <= 5:
            raise ValueError(f"cutoff {self.cutoff:g} is off the 1-5 overall-merit scale")
        if self.min_reviews < 1:
            raise ValueError("min_reviews must be at least 1")
        if self.min_decided is not None and not 1 <= self.min_decided <= self.min_reviews:
            raise ValueError("min_decided must lie between 1 and min_reviews")

    @property
    def decided_from(self) -> int:
        """The fewest reviews on which a paper is decided at all."""
        return self.min_reviews if self.min_decided is None else self.min_decided

    def is_early(self, scores: list[int]) -> bool:
        """Whether the paper is decided by the early rule rather than the net."""
        return self.decided_from <= len(scores) < self.min_reviews

    def advances(self, scores: list[int], recent: bool = False, held: bool = False) -> str | None:
        """FEW_REVIEWS, RECENT, OVER_BAR, or None when the paper is under the bar.

        `recent` (see `recent_pids`) turns a paper under the bar into RECENT:
        an early paper always, a paper the net decides only when `held` says
        the export already tags it RevisionAdvance.
        """
        if len(scores) < self.decided_from:
            return FEW_REVIEWS
        if self.is_early(scores):
            under = max(scores) <= EARLY_UNDER_MAX
        else:
            under = caught(scores, self.cutoff, self.comparator, NET_CONDITIONS[self.net])
        if not under:
            return OVER_BAR
        return RECENT if recent and (held or self.is_early(scores)) else None

    def describe(self) -> str:
        """One line naming the bar, for reports."""
        extra = "" if self.net == "average" else " and" + NET_LABELS[self.net][1:]
        line = (f"under the bar: average {COMPARATOR_SYMBOLS[self.comparator]} "
                f"{self.cutoff:g}{extra}")
        if self.decided_from < self.min_reviews:
            line += (f" with {self.min_reviews}+ reviews; with {self.decided_from}-"
                     f"{self.min_reviews - 1}, every score {EARLY_UNDER_MAX} or lower")
        return (line + "; a recently assigned review outstanding advances an early paper"
                " and keeps a RevisionAdvance one"
                f"; fewer than {self.decided_from} reviews always advances")


def tagged_pids(data_path: str, tag: str) -> set[int]:
    """{pid} whose tags in the paper export include `tag` (twiddles and values ignored)."""
    with open(data_path, encoding="utf-8") as f:
        papers = json.load(f)
    want = tag.lower()
    return {p["pid"] for p in papers
            if want in pc_membership.tag_names(" ".join(p.get("tags") or []))}


@dataclass(frozen=True)
class ManualTag:
    """The revision tag a hand edit in HotCRP left on a paper."""

    tag: str    # ADVANCE_TAG, NO_REVISION_TAG, or "" when it left neither
    email: str  # who made the edit
    date: str   # the log's timestamp, as written


def manual_revision_tags(log_rows: list[dict[str, str]]) -> dict[int, ManualTag]:
    """{pid: ManualTag} for every paper whose revision tags were ever changed by hand.

    `log_rows` oldest first (`hotcrp_log.load_log`). The log does not say
    whether a tag change came from the UI or a bulk upload, but an upload writes
    every row under one account and one timestamp, and a UI edit touches one
    paper per request. So a change is **manual when no other paper's revision
    tags changed under the same account at the same second.** The cost of the
    rule: an upload that happens to change one paper's tag reads as manual, and
    pins that paper to what the upload said.

    A manual edit wins over any later upload: the paper keeps the tag the
    *last* manual edit added, or, when that edit only removed a tag, whatever
    the paper was left holding (replayed from the log). An edit that left both
    tags cannot happen in the UI, which replaces one with the other.
    """
    ours = {ADVANCE_TAG.lower(): ADVANCE_TAG, NO_REVISION_TAG.lower(): NO_REVISION_TAG}
    events = []
    for row in log_rows:
        changes = [(sign, ours[name.lower()])
                   for sign, name in re.findall(r"([+-])#([^\s#]+)", row["action"])
                   if row["action"].startswith("Tag ") and name.lower() in ours]
        if changes and row["paper"].strip().isdigit():
            events.append((row, int(row["paper"]), changes))
    papers_per_request = Counter()
    seen = set()
    for row, pid, _ in events:
        key = (row["email"], row["date"])
        if (key, pid) not in seen:
            seen.add((key, pid))
            papers_per_request[key] += 1

    state: dict[int, set[str]] = {}
    manual: dict[int, ManualTag] = {}
    for row, pid, changes in events:
        tags = state.setdefault(pid, set())
        for sign, tag in changes:
            (tags.add if sign == "+" else tags.discard)(tag)
        if papers_per_request[(row["email"], row["date"])] > 1:
            continue
        added = [tag for sign, tag in changes if sign == "+"]
        left = added[-1] if added else (next(iter(tags)) if len(tags) == 1 else "")
        manual[pid] = ManualTag(left, row["email"], row["date"])
    return manual


def reason_tag(reason: str | None) -> str | None:
    """The revision tag a `decide_paper` reason gets: None while untagged."""
    if reason in (FEW_REVIEWS, MANUAL_UNTAGGED):
        return None
    return NO_REVISION_TAG if reason is None else ADVANCE_TAG


def manual_report(manual: dict[int, ManualTag], by_bar: dict[int, str | None]) -> list[str]:
    """stdout lines naming every hand-set tag the bar was skipped for, oldest first.

    `by_bar` is {pid: tag the bar alone would give} over the papers in play;
    a hand-set tag on a paper outside it (desk-rejected, excluded) is left out.
    """
    pids = sorted((pid for pid in manual if pid in by_bar), key=lambda pid: (manual[pid].date, pid))
    differ = sum((manual[pid].tag or None) != by_bar[pid] for pid in pids)
    lines = [f"{len(pids)} paper(s) keep a hand-set revision tag instead of the bar's "
             f"({differ} where the bar disagrees, marked *):"]
    for pid in pids:
        m = manual[pid]
        tag, bar_tag = m.tag or None, by_bar[pid]
        lines.append(f"  {'*' if tag != bar_tag else ' '} #{pid}: {tag or 'untagged'}, "
                     f"set by {m.email} at {m.date} (bar: {bar_tag or 'untagged'})")
    return lines


def decide_paper(bar: Bar, scores: list[int], *, recent: bool = False, held: bool = False,
                 manual: str | None = None) -> str | None:
    """`Bar.advances`, unless a revision tag was set by hand.

    In priority order: a hand-set tag; an outstanding late-assigned review
    (`recent`), which keeps the paper advancing (`held`: the export already
    tags it RevisionAdvance; see `Bar.advances`); the reviews as they stand.
    Both `scripts/revision_tags.py` and `scripts/assign_paper_leads.py` decide
    through here, so a paper's tag and its lead always agree.

    `manual` (a `ManualTag.tag`, None when nobody tagged the paper by hand)
    overrides everything: RevisionAdvance gives MANUAL_ADVANCE, NoRevision
    None, and "" (both removed by hand) MANUAL_UNTAGGED, which advances but is
    left untagged.
    """
    if manual is not None:
        return {ADVANCE_TAG: MANUAL_ADVANCE, NO_REVISION_TAG: None}.get(manual, MANUAL_UNTAGGED)
    return bar.advances(scores, recent, held)


def add_bar_arguments(parser: argparse.ArgumentParser) -> None:
    """The four flags that define a `Bar`, with this module's defaults."""
    parser.add_argument(
        "--bar-net", choices=list(NET_CONDITIONS), default=DEFAULT_BAR_NET,
        help=f"the net that decides who is under the bar (default {DEFAULT_BAR_NET})"
    )
    parser.add_argument(
        "--bar-cutoff", type=float, default=DEFAULT_BAR_CUTOFF,
        help=f"average-score cutoff of the bar (default {DEFAULT_BAR_CUTOFF:g})"
    )
    parser.add_argument(
        "--bar-comparator", choices=COMPARATORS, default=DEFAULT_BAR_COMPARATOR,
        help="le: an average on the cutoff is under the bar; lt: it advances (default le)"
    )
    parser.add_argument(
        "--min-reviews", type=int, default=DEFAULT_MIN_REVIEWS,
        help=f"papers with this many submitted reviews face the net (default {DEFAULT_MIN_REVIEWS})"
    )
    parser.add_argument(
        "--min-decided", type=int, default=None,
        help="papers with fewer submitted reviews always advance; from here up to --min-reviews"
             f" they are under the bar when every score is {EARLY_UNDER_MAX} or lower"
             " (default: --min-reviews, i.e. no early decisions)"
    )
    parser.add_argument(
        "--recent-after", type=date.fromisoformat, default=None,
        help="an early paper holding an outstanding R1 review assigned on a later day advances"
             " (default: none)"
    )


def bar_from_args(args: argparse.Namespace) -> Bar:
    return Bar(args.bar_net, args.bar_cutoff, args.bar_comparator, args.min_reviews,
               args.min_decided)


def recent_from_args(args: argparse.Namespace, log_rows: list[dict[str, str]]) -> set[int]:
    """`recent_pids` under `--recent-after`, or nothing when the flag is unset."""
    return recent_pids(log_rows, args.recent_after) if args.recent_after else set()
