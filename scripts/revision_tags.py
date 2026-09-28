"""Tag every paper the revision bar has decided: RevisionAdvance or NoRevision.

    python -m scripts.revision_tags --min-reviews 5
    python -m scripts.revision_tags --min-reviews 5 --min-decided 3 --recent-after 2026-09-13
    python -m scripts.revision_tags --bar-cutoff 2.25 --bar-net no4
    python -m scripts.revision_tags --upload /tmp/revision_tags.csv

**Which papers.** Every paper still under review (`submitted`, not
desk-rejected, not `--exclude-pids`) with at least `--min-decided` (default:
`--min-reviews`) submitted PC reviews. The bar is
`reviewer_match.review_scores.Bar`, the definition
`scripts/assign_paper_leads.py` and `scripts/timeliness_tags.py` use too, so
the tags, the leads and the timeliness tags agree on the same inputs and
flags: over the bar gets `RevisionAdvance`, under it `NoRevision`. From
`--min-reviews` reviews up the net decides; below that, a paper is under the
bar when every score is reject or weak reject. An outstanding R1 review
assigned after `--recent-after` is let play out: it advances an early paper,
and keeps `RevisionAdvance` on a paper the export already tags so. TRC reviews count towards neither the review floor nor the average.

**Papers short of reviews are left untagged.** They advance for a lead, but
the bar has not decided them yet. Tag them on a later run, once their reviews
are in.

**The tag follows the reviews.** A `RevisionAdvance` paper that new reviews put
under the bar becomes `NoRevision` -- unless a late-assigned review is still
outstanding on it, or the tag was set by hand (below). Priority, highest first:
a hand-set tag, an outstanding late-assigned review, the reviews
(`review_scores.decide_paper`).

**A hand-set tag is never overridden.** A paper whose RevisionAdvance or
NoRevision tag somebody changed in HotCRP's UI keeps what their last edit left
(`review_scores.manual_revision_tags`) -- even where a later upload of this
file changed it, which this upload then puts back. Such papers are listed on
stdout, with who set the tag and when, and those the bar disagrees with marked.

**Early NoRevision papers lose their outstanding reviews.** Every outstanding
R1 review on a paper decided `NoRevision` below `--min-reviews` gets a
per-pair `clearreview` row in `--unassign-upload` -- never
`all,clearreview`, and never a submitted or TRC review. A paper decided on the
net is not touched: its slate was complete when it was decided.
`scripts/timeliness_tags.py` reports whose late status those removals change.
A hand-set NoRevision unassigns nothing the bar would not: taking reviews
away is the chairs' call, not the side effect of a PC member's tag.

**Reruns are safe.** HotCRP's bulk-assignment `tag` action only adds, so a
paper that crosses the bar between runs would otherwise carry both tags. The
upload is therefore a delta that states the whole decision per paper: every
decided paper gets a `cleartag` of the opposite tag before its `tag` row, and
every undecided paper gets a `cleartag` of both. Clearing a tag a paper does
not have changes nothing. Desk-rejected and excluded papers are not touched.

Writes `--upload` (`paper,action,email,tag,round`, clears first, then tags,
each by pid) and `--unassign-upload` (`paper,action,email,round`). Nothing is
uploaded. Offline and instant. The file reveals
review outcomes, so it is gitignored with the other assignment outputs.
"""

from __future__ import annotations

from reviewer_match.paths import assignment_path, input_path

import argparse
import os
import sys
from collections import Counter
from dataclasses import dataclass

from reviewer_match import hotcrp_log
from reviewer_match import paper_matching
from reviewer_match import review_scores
from scripts.assign_paper_leads import write_csv

DEFAULT_REVIEWS = input_path("hpca2027-reviews.csv")
DEFAULT_LOG = input_path("hpca2027-log.csv")
DEFAULT_DATA = input_path("hpca2027-data.json")
DEFAULT_UPLOAD = assignment_path("revision_tags_upload.csv")
DEFAULT_UNASSIGN_UPLOAD = assignment_path("revision_unassign_upload.csv")

ADVANCE_TAG = review_scores.ADVANCE_TAG
NO_REVISION_TAG = review_scores.NO_REVISION_TAG
TAGS = (ADVANCE_TAG, NO_REVISION_TAG)
UPLOAD_HEADER = ["paper", "action", "email", "tag", "round"]
UNASSIGN_HEADER = ["paper", "action", "email", "round"]


def reasons(
    eligible: set[int],
    scores: dict[int, list[int]],
    bar: review_scores.Bar,
    recent: set[int] = frozenset(),
    manual: dict[int, review_scores.ManualTag] | None = None,
    held: set[int] = frozenset(),
) -> dict[int, str | None]:
    """{pid: `review_scores.decide_paper`'s reason}, None meaning under the bar."""
    manual = manual or {}
    return {
        pid: review_scores.decide_paper(bar, scores.get(pid, []), recent=pid in recent, held=pid in held,
                                        manual=manual[pid].tag if pid in manual else None)
        for pid in sorted(eligible)
    }


def decide(
    eligible: set[int],
    scores: dict[int, list[int]],
    bar: review_scores.Bar,
    recent: set[int] = frozenset(),
    manual: dict[int, review_scores.ManualTag] | None = None,
    held: set[int] = frozenset(),
) -> dict[int, str | None]:
    """{pid: ADVANCE_TAG, NO_REVISION_TAG, or None while short of reviews or untagged by hand}."""
    return {
        pid: review_scores.reason_tag(reason)
        for pid, reason in reasons(eligible, scores, bar, recent, manual, held).items()
    }


def unassign_pairs(
    decisions: dict[int, str | None],
    scores: dict[int, list[int]],
    bar: review_scores.Bar,
    live: dict[int, hotcrp_log.Review],
    submitted: set[int],
) -> list[tuple[int, str]]:
    """[(pid, email)] of outstanding R1 reviews on papers decided NoRevision early, sorted."""
    early_no = {pid for pid, tag in decisions.items()
                if tag == NO_REVISION_TAG and bar.is_early(scores.get(pid, []))}
    return sorted(
        (review.pid, review.email)
        for rid, review in live.items()
        if review.pid in early_no and review.round == review_scores.REVIEW_ROUND
        and rid not in submitted
    )


@dataclass(frozen=True)
class Decided:
    """Every paper's revision decision and what it implies, off one set of inputs."""

    bar: review_scores.Bar
    scores: dict[int, list[int]]
    why: dict[int, str | None]            # pid -> `reasons`
    decisions: dict[int, str | None]      # pid -> tag, None while undecided
    pairs: list[tuple[int, str]]          # `unassign_pairs`
    manual: dict[int, review_scores.ManualTag]  # `review_scores.manual_revision_tags`
    by_bar: dict[int, str | None]         # `decide` without the hand-set tags


def load_inputs(args) -> Decided:
    """`Decided` off the shared flags (`--reviews`, `--log`, `--data`, the bar's).

    `scripts/timeliness_tags.py` calls this too, so its NoRevision papers and
    the unassignments it reports on are exactly this script's.
    """
    bar = review_scores.bar_from_args(args)
    log_rows = hotcrp_log.load_log(args.log)
    training, _ = review_scores.review_rounds(log_rows)
    reviews, _ = review_scores.load_reviews(args.reviews, training)
    eligible, _ = review_scores.eligible_pids(args.data, paper_matching.parse_exclude_pids(args.exclude_pids))
    scores: dict[int, list[int]] = {}
    for review in reviews:
        scores.setdefault(review.pid, []).append(review.score)
    recent = review_scores.recent_from_args(args, log_rows)
    manual = review_scores.manual_revision_tags(log_rows)
    held = review_scores.tagged_pids(args.data, ADVANCE_TAG)
    why = reasons(eligible, scores, bar, recent, manual, held)
    decisions = decide(eligible, scores, bar, recent, manual, held)
    by_bar = decide(eligible, scores, bar, recent, held=held)
    # Unassign only where the bar agrees: see the module docstring.
    agreed = {pid: tag if tag == by_bar[pid] else None for pid, tag in decisions.items()}
    live, _ = hotcrp_log.replay_assignments(log_rows)
    pairs = unassign_pairs(agreed, scores, bar, live, hotcrp_log.submitted_review_ids(log_rows))
    return Decided(bar, scores, why, decisions, pairs, manual, by_bar)


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
    parser.add_argument("--unassign-upload", default=DEFAULT_UNASSIGN_UPLOAD,
                        help="HotCRP bulk-assignment file of the reviews to unassign")
    args = parser.parse_args()

    for path in (args.reviews, args.log, args.data):
        if not os.path.exists(path):
            print(f"ERROR: {path} not found", file=sys.stderr)
            return 1
    try:
        d = load_inputs(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    bar, scores, why, decisions, pairs = d.bar, d.scores, d.why, d.decisions, d.pairs
    write_csv(args.upload, UPLOAD_HEADER, upload_rows(decisions))
    write_csv(args.unassign_upload, UNASSIGN_HEADER,
              [[pid, "clearreview", email, review_scores.REVIEW_ROUND] for pid, email in pairs])
    print(f"Wrote {args.upload}\nWrote {args.unassign_upload}", file=sys.stderr)

    counts = Counter(decisions.values())
    early = Counter(tag for pid, tag in decisions.items() if tag and bar.is_early(scores.get(pid, [])))
    print(f"Bar: {bar.describe()}.")
    print(f"{len(decisions)} papers: {counts[ADVANCE_TAG]} {ADVANCE_TAG}, "
          f"{counts[NO_REVISION_TAG]} {NO_REVISION_TAG}, "
          f"{counts[None]} untagged with fewer than {bar.decided_from} reviews.")
    if bar.decided_from < bar.min_reviews:
        print(f"  decided on {bar.decided_from}-{bar.min_reviews - 1} reviews: "
              f"{early[ADVANCE_TAG]} {ADVANCE_TAG}, {early[NO_REVISION_TAG]} {NO_REVISION_TAG}")
    recent = sorted(pid for pid, reason in why.items() if reason == review_scores.RECENT)
    if recent:
        print(f"  {len(recent)} paper(s) under the bar kept at {ADVANCE_TAG} for an outstanding "
              f"late-assigned review: {', '.join(map(str, recent))}")
    was = review_scores.tagged_pids(args.data, ADVANCE_TAG)
    downgraded = sorted(pid for pid, tag in decisions.items() if tag == NO_REVISION_TAG and pid in was)
    if downgraded:
        print(f"  {len(downgraded)} paper(s) go from {ADVANCE_TAG} to {NO_REVISION_TAG} on new reviews: "
              f"{', '.join(map(str, downgraded))}")
    print("\n".join(review_scores.manual_report(d.manual, d.by_bar)))
    print(f"{len(pairs)} outstanding R1 review(s) to unassign on "
          f"{len({pid for pid, _ in pairs})} early {NO_REVISION_TAG} paper(s), "
          f"{len({email for _, email in pairs})} reviewer(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
