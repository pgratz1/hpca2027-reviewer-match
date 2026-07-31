"""Build a smaller paper export for smoke-testing the assignment end to end.

    python -m scripts.make_smoke_dataset
    python -m scripts.make_smoke_dataset --fraction 0.25       # tighter, less headroom
    python -m scripts.make_smoke_dataset --seed 7 --out /tmp/other-draw.json

Registration is open, so the export holds far more papers than will ever need
reviewing: 1,421 currently pass the `registered` policy, against a PC and reserve
pool that can cover roughly 1,070 at six reviews each. Testing the assignment
against all of them only ever measures the shortfall.

This writes a copy of the export with a random `--fraction` of the currently
selectable papers marked withdrawn, standing in for the ones that will not be
submitted. At the default 30% that leaves ~995 papers and about 8% of headroom —
enough that the assignment should largely fill, so anything that fails points at
the machinery rather than at arithmetic.

Papers are *marked withdrawn* rather than deleted: paper_matching's own
`_is_withdrawn` then drops them through the same policy path a real withdrawal
takes, so the smoke run exercises production selection logic and not a filter
written for the test. Papers already withdrawn or otherwise unselectable are left
untouched and are not counted toward the fraction — the draw is over what is
actually in play.

The seed is fixed by default, so the same draw comes back every run and two
assignments can be compared without the paper set moving underneath them.
"""

from __future__ import annotations

from reviewer_match.paths import assignment_path, cache_path, curated_path, input_path, report_path, smoke_cache_path

import argparse
import json
import random
import sys
from pathlib import Path

from reviewer_match.paper_matching import PAPER_POLICIES, load_papers

DEFAULT_DATA = input_path("hpca2027-data.json")
DEFAULT_OUT = smoke_cache_path("hpca2027-data-smoke.json")
DEFAULT_FRACTION = 0.30

# Fixed so a re-run reproduces the draw exactly. Change it to sample a different
# set, not to get "a fresh random one" -- an unrepeatable paper set makes two
# assignment runs incomparable.
DEFAULT_SEED = 20260730

WITHDRAW_REASON = "smoke test: assumed not submitted"


def choose_withdrawn(pids: list[int], fraction: float, seed: int) -> set[int]:
    """The pids to mark withdrawn: a seeded sample of `fraction` of `pids`.

    Sorted before sampling so the draw depends on the seed and the paper set,
    never on the order the export happened to arrive in.
    """
    count = round(len(pids) * fraction)
    return set(random.Random(seed).sample(sorted(pids), count))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help=f"HotCRP paper export to draw from (default: {DEFAULT_DATA})")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help=f"where to write the smoke export (default: {DEFAULT_OUT})")
    parser.add_argument("--fraction", type=float, default=DEFAULT_FRACTION,
                        help=f"share of selectable papers to withdraw (default: {DEFAULT_FRACTION})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"random seed, fixed so runs repeat (default: {DEFAULT_SEED})")
    parser.add_argument("--paper-policy", default="registered", choices=PAPER_POLICIES,
                        help="which papers count as selectable (default: registered)")
    args = parser.parse_args()

    if not 0.0 <= args.fraction < 1.0:
        parser.error("--fraction must be at least 0 and below 1")
    if Path(args.out).resolve() == Path(args.data).resolve():
        parser.error("--out would overwrite --data; choose a different path")

    selectable = load_papers(args.data, paper_policy=args.paper_policy)
    selectable_pids = [p["pid"] for p in selectable]
    withdrawn = choose_withdrawn(selectable_pids, args.fraction, args.seed)

    with open(args.data, encoding="utf-8") as f:
        papers = json.load(f)
    for paper in papers:
        if paper["pid"] in withdrawn:
            paper["withdrawn"] = True
            paper["withdraw_reason"] = WITHDRAW_REASON

    target = Path(args.out)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=1, sort_keys=True)
    tmp.replace(target)

    remaining = len(selectable_pids) - len(withdrawn)
    print(
        f"{len(papers)} papers in {args.data}; {len(selectable_pids)} selectable "
        f"under --paper-policy {args.paper_policy}",
        file=sys.stderr,
    )
    print(
        f"Marked {len(withdrawn)} withdrawn ({args.fraction:.0%}, seed {args.seed}); "
        f"{remaining} remain -> {args.out}",
        file=sys.stderr,
    )
    print(
        f"At 6 reviews each that is {remaining * 6} review slots.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
