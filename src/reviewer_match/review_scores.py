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
paper somebody argues for (`NETS`). A paper is **under the bar** when a net
catches it. A paper with fewer than `min_reviews` submitted reviews always
advances, whatever its scores (`Bar.advances`).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
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
DEFAULT_BAR_NET = "one3"
DEFAULT_BAR_CUTOFF = 2.5
DEFAULT_BAR_COMPARATOR = "le"

# `Bar.advances` reasons.
FEW_REVIEWS = "few-reviews"
OVER_BAR = "over-bar"

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

    def __post_init__(self) -> None:
        if self.net not in NET_CONDITIONS:
            raise ValueError(f"unknown net {self.net!r}; expected one of {', '.join(NET_CONDITIONS)}")
        if self.comparator not in COMPARATORS:
            raise ValueError(f"unknown comparator {self.comparator!r}; expected le or lt")
        if not 1 <= self.cutoff <= 5:
            raise ValueError(f"cutoff {self.cutoff:g} is off the 1-5 overall-merit scale")
        if self.min_reviews < 1:
            raise ValueError("min_reviews must be at least 1")

    def advances(self, scores: list[int]) -> str | None:
        """FEW_REVIEWS, OVER_BAR, or None when the paper is under the bar."""
        if len(scores) < self.min_reviews:
            return FEW_REVIEWS
        if caught(scores, self.cutoff, self.comparator, NET_CONDITIONS[self.net]):
            return None
        return OVER_BAR

    def describe(self) -> str:
        """One line naming the bar, for reports."""
        extra = "" if self.net == "average" else " and" + NET_LABELS[self.net][1:]
        return (f"under the bar: average {COMPARATOR_SYMBOLS[self.comparator]} "
                f"{self.cutoff:g}{extra}; fewer than {self.min_reviews} reviews always advances")


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
        help=f"papers with fewer submitted reviews always advance (default {DEFAULT_MIN_REVIEWS})"
    )


def bar_from_args(args: argparse.Namespace) -> Bar:
    return Bar(args.bar_net, args.bar_cutoff, args.bar_comparator, args.min_reviews)
