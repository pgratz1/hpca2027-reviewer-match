"""Estimate how many reserve reviewers are needed to cover the review shortfall.

The accepted PC cannot cover the submission volume: the papers need
`--reviewers-per-paper` reviews each, and the PC supplies only as many review
slots as its per-member caps allow. The gap has to be filled by reserve
reviewers — the senior authors each submission nominates in HotCRP's
`reserve_reviewer` field — each of whom is expected to take
`--reviews-per-reserve` reviews.

Pure arithmetic on the same inputs the real assignment uses: no embeddings, no
network. It deliberately ignores COI and the area gate, so it is a *floor* on
the reserve count; the true shortfall from assign_reviewers.py's shortage
report is always at least this large. Pass that number to `--unfilled-slots`
to size the cohort against it instead.

    python -m scripts.estimate_reserve_need
    python -m scripts.estimate_reserve_need --reviews-per-reserve 3
    python -m scripts.estimate_reserve_need --paper-policy submitted
    python -m scripts.estimate_reserve_need --unfilled-slots 3800
"""

from __future__ import annotations

from reviewer_match.paths import assignment_path, cache_path, curated_path, input_path, report_path

import argparse
import math
import sys

from scripts.assign_reviewers import DEFAULT_REVIEWERS_PER_PAPER, reviewer_paper_cap
from reviewer_match.paper_matching import PAPER_POLICIES, load_papers
from reviewer_match.reviewers import load_reviewers

DEFAULT_CSV = input_path("HPCA'27 PC Member Acceptance Form (Responses) - Form Responses 1.csv")
DEFAULT_DATA = input_path("hpca2027-data.json")

DEFAULT_REVIEWS_PER_RESERVE = 4
DEFAULT_LIGHT_CAP = 7
DEFAULT_FULL_CAP = 15

# Alternative per-reserve loads printed alongside the answer, so the chair can
# see how sensitive the cohort size is to the one assumption driving it.
SENSITIVITY_LOADS = (2, 3, 4, 6, 8)


def reserves_needed(deficit: int, reviews_per_reserve: int) -> int:
    """Reserve reviewers required to absorb `deficit` review slots."""
    return math.ceil(deficit / reviews_per_reserve)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default=DEFAULT_DATA, help="path to the HotCRP paper export JSON")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="path to the reviewer CSV")
    parser.add_argument(
        "--paper-policy", default="registered", choices=PAPER_POLICIES,
        help="which papers count toward the review load (default: registered)",
    )
    parser.add_argument(
        "--reviews-per-reserve", type=int, default=DEFAULT_REVIEWS_PER_RESERVE,
        help=f"reviews each reserve reviewer takes (default: {DEFAULT_REVIEWS_PER_RESERVE})",
    )
    parser.add_argument(
        "--reviewers-per-paper", type=int, default=DEFAULT_REVIEWERS_PER_PAPER,
        help=f"reviews each paper needs (default: {DEFAULT_REVIEWERS_PER_PAPER})",
    )
    parser.add_argument(
        "--light-cap", type=int, default=DEFAULT_LIGHT_CAP,
        help=f"max papers per light PC member (default: {DEFAULT_LIGHT_CAP})",
    )
    parser.add_argument(
        "--full-cap", type=int, default=DEFAULT_FULL_CAP,
        help=f"max papers per full PC member (default: {DEFAULT_FULL_CAP})",
    )
    parser.add_argument(
        "--unfilled-slots", type=int, default=None,
        help="use this slot deficit (e.g. assign_reviewers.py's shortage-report "
             "total) instead of the papers-minus-capacity arithmetic",
    )
    args = parser.parse_args()

    if args.reviews_per_reserve <= 0:
        parser.error("--reviews-per-reserve must be positive")
    if args.reviewers_per_paper <= 0:
        parser.error("--reviewers-per-paper must be positive")
    if args.light_cap < 0 or args.full_cap < 0:
        parser.error("--light-cap and --full-cap must be non-negative")
    if args.unfilled_slots is not None and args.unfilled_slots < 0:
        parser.error("--unfilled-slots must be non-negative")

    papers = load_papers(args.data, paper_policy=args.paper_policy)
    reviewers = load_reviewers(args.csv)

    slots_needed = len(papers) * args.reviewers_per_paper
    tiers = {"full": [0, 0], "light": [0, 0]}  # tier -> [members, capacity]
    for r in reviewers:
        stat = tiers.setdefault(r.tier, [0, 0])
        stat[0] += 1
        stat[1] += reviewer_paper_cap(r, args.light_cap, args.full_cap)
    pc_capacity = sum(stat[1] for stat in tiers.values())

    if args.unfilled_slots is not None:
        deficit = args.unfilled_slots
        basis = "supplied --unfilled-slots"
    else:
        deficit = max(0, slots_needed - pc_capacity)
        basis = "papers x reviewers-per-paper - PC capacity"

    print("=== Reserve reviewer need ===")
    print(f"Paper policy:            {args.paper_policy}")
    print(f"Papers:                  {len(papers)}")
    print(f"Reviews per paper:       {args.reviewers_per_paper}")
    print(f"Review slots needed:     {slots_needed}")
    print(f"PC members:              {len(reviewers)}")
    for tier in sorted(tiers):
        members, capacity = tiers[tier]
        print(f"    {tier:<6} {members:>4} member(s) = {capacity:>5} slot(s)")
    print(f"PC capacity:             {pc_capacity}")
    print(f"Shortfall ({basis}): {deficit}")
    print()

    needed = reserves_needed(deficit, args.reviews_per_reserve)
    print(
        f"At {args.reviews_per_reserve} review(s) each, "
        f"{needed} reserve reviewer(s) are needed."
    )

    print("\nSensitivity to the per-reserve load:")
    for load in sorted({*SENSITIVITY_LOADS, args.reviews_per_reserve}):
        marker = "  <-- selected" if load == args.reviews_per_reserve else ""
        print(f"    {load:>2} review(s) each: {reserves_needed(deficit, load):>5} reserve(s){marker}")

    if deficit == 0:
        print(
            "\nThe PC already covers the requested load; no reserve reviewers "
            "are needed under these assumptions.",
            file=sys.stderr,
        )
    else:
        print(
            f"\nNote: this ignores COI and the area gate, so {needed} is a floor. "
            f"Compare it against the size of the recruited reserve list — if "
            f"that pool is smaller, either the per-reserve load or the "
            f"papers-per-reviewer caps have to rise.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
