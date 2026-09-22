"""Tag every paper the revision bar has decided: RevisionAdvance or NoRevision.

    python -m scripts.revision_tags --min-reviews 5
    python -m scripts.revision_tags --bar-cutoff 2.25 --bar-net no4
    python -m scripts.revision_tags --upload /tmp/revision_tags.csv

**Which papers.** Every paper still under review (`submitted`, not
desk-rejected, not `--exclude-pids`) with at least `--min-reviews` submitted PC
reviews. The bar is `reviewer_match.review_scores.Bar`, the definition
`scripts/assign_paper_leads.py` and `scripts/revision_cutoffs.py` use too, so
the tags and the leads agree on the same inputs and flags: over the bar gets
`RevisionAdvance`, under it `NoRevision`. TRC reviews count towards neither
the review floor nor the average.

**Papers short of reviews are left untagged.** They advance for a lead, but
the bar has not decided them yet. Tag them on a later run, once their reviews
are in.

**Reruns are safe.** HotCRP's bulk-assignment `tag` action only adds, so a
paper that crosses the bar between runs would otherwise carry both tags. The
upload is therefore a delta that states the whole decision per paper: every
decided paper gets a `cleartag` of the opposite tag before its `tag` row, and
every undecided paper gets a `cleartag` of both. Clearing a tag a paper does
not have changes nothing. Desk-rejected and excluded papers are not touched.

Writes `--upload` (`paper,action,email,tag,round`, clears first, then tags,
each by pid). Nothing is uploaded. Offline and instant. The file reveals
review outcomes, so it is gitignored with the other assignment outputs.
"""

from __future__ import annotations

from reviewer_match.paths import assignment_path, input_path

import argparse
import os
import sys
from collections import Counter

from reviewer_match import hotcrp_log
from reviewer_match import paper_matching
from reviewer_match import review_scores
from scripts.assign_paper_leads import write_csv

DEFAULT_REVIEWS = input_path("hpca2027-reviews.csv")
DEFAULT_LOG = input_path("hpca2027-log.csv")
DEFAULT_DATA = input_path("hpca2027-data.json")
DEFAULT_UPLOAD = assignment_path("revision_tags_upload.csv")

ADVANCE_TAG = "RevisionAdvance"
NO_REVISION_TAG = "NoRevision"
TAGS = (ADVANCE_TAG, NO_REVISION_TAG)
UPLOAD_HEADER = ["paper", "action", "email", "tag", "round"]


def decide(
    eligible: set[int], scores: dict[int, list[int]], bar: review_scores.Bar
) -> dict[int, str | None]:
    """{pid: ADVANCE_TAG, NO_REVISION_TAG, or None while short of reviews}."""
    decisions: dict[int, str | None] = {}
    for pid in sorted(eligible):
        reason = bar.advances(scores.get(pid, []))
        if reason == review_scores.FEW_REVIEWS:
            decisions[pid] = None
        else:
            decisions[pid] = ADVANCE_TAG if reason == review_scores.OVER_BAR else NO_REVISION_TAG
    return decisions


def upload_rows(decisions: dict[int, str | None]) -> list[list[object]]:
    """The HotCRP delta: every clear first, then every tag, each by pid."""
    clears = []
    tags = []
    for pid in sorted(decisions):
        tag = decisions[pid]
        clears += [[pid, "cleartag", "", other, ""] for other in TAGS if other != tag]
        if tag:
            tags.append([pid, "tag", "", tag, ""])
    return clears + tags


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--reviews", default=DEFAULT_REVIEWS, help="HotCRP review CSV export")
    parser.add_argument("--log", default=DEFAULT_LOG, help="HotCRP action log (identifies TRC reviews)")
    parser.add_argument("--data", default=DEFAULT_DATA, help="HotCRP paper export JSON")
    parser.add_argument("--exclude-pids", default="", help="comma-separated paper IDs to leave out")
    review_scores.add_bar_arguments(parser)
    parser.add_argument("--upload", default=DEFAULT_UPLOAD, help="HotCRP bulk-assignment delta to write")
    args = parser.parse_args()

    bar = review_scores.bar_from_args(args)
    for path in (args.reviews, args.log, args.data):
        if not os.path.exists(path):
            print(f"ERROR: {path} not found", file=sys.stderr)
            return 1

    training, _ = review_scores.review_rounds(hotcrp_log.load_log(args.log))
    reviews, _ = review_scores.load_reviews(args.reviews, training)
    eligible, _ = review_scores.eligible_pids(args.data, paper_matching.parse_exclude_pids(args.exclude_pids))
    scores: dict[int, list[int]] = {}
    for review in reviews:
        scores.setdefault(review.pid, []).append(review.score)

    decisions = decide(eligible, scores, bar)
    write_csv(args.upload, UPLOAD_HEADER, upload_rows(decisions))
    print(f"Wrote {args.upload}", file=sys.stderr)

    counts = Counter(decisions.values())
    print(f"Bar: {bar.describe()}.")
    print(f"{len(decisions)} papers: {counts[ADVANCE_TAG]} {ADVANCE_TAG}, "
          f"{counts[NO_REVISION_TAG]} {NO_REVISION_TAG}, "
          f"{counts[None]} untagged with fewer than {bar.min_reviews} reviews.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
