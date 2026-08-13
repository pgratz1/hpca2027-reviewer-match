"""Reconstruct HotCRP's live review assignment by replaying its action log.

    python -m scripts.extract_log_assignments
    python -m scripts.extract_log_assignments --kind external --round TRC \\
        --out outputs/assignments/current_trc_assignment.csv

`outputs/assignments/assignment.csv` is what this pipeline last *proposed*.
What HotCRP actually holds has drifted from it: reviews get added and removed
by hand through the UI, and none of those edits reach any artifact under
`outputs/`. This replays `Review N assigned` / `Review N unassigned` from the
action log to recover the live state, and writes it in the same
`paper,action,email,round` shape `assign_reviewers.py --hotcrp-csv` writes --
so it drops into `--pin-csv`, `scripts/diff_assignments.py` and
`scripts/fill_open_slots.py --baseline` with no adapter.

Offline, instant, read-only apart from --out.

**HotCRP can export this directly and that export is authoritative.** Search
-> select all -> Download -> "Review assignments", or the Assignments page's
own download link, produces this same shape from the database rather than from
a reconstruction. Prefer it when you have it; this exists because the log is
what a chair always has, and because diffing the two is a real check on both.

Only complete review rows are recovered: the log records that a review exists,
not its text or its round-trip through a reviewer's hands. `--kind`/`--round`
select which reviews to emit, defaulting to the R1 primary reviews the matcher
owns -- the TRC externals belong to `scripts/assign_trc_reviews.py`, and
mixing them into a file destined for a `clearreview R1` upload would propose
deleting them.
"""

from __future__ import annotations

from reviewer_match.paths import assignment_path, input_path

import argparse
import csv
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

from reviewer_match import hotcrp_log
from reviewer_match.assignment_io import HOTCRP_CSV_HEADER

DEFAULT_LOG = input_path("hpca2027-log.csv")
DEFAULT_OUT = assignment_path("current_assignment.csv")

# HotCRP's bulk-assignment verbs, matched to the review kinds the log records.
REVIEW_ACTIONS = {"primary": "primaryreview", "external": "review", "secondary": "secondaryreview"}


def write_hotcrp_csv(path: str, pairs: dict[int, set[str]], round: str) -> None:
    """The live pairs in HotCRP upload shape, atomically and idempotently.

    Opens with the `clearreview` row every upload this pipeline writes opens
    with, so the file is a complete statement of the assignment rather than a
    set of additions -- uploading it makes HotCRP match it exactly. Rows are
    sorted by (pid, email): affinity order is not recoverable here (the log
    records no score), and a stable order is what lets `cmp` verify a re-run.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", dir=target.parent,
            prefix=f".{target.name}.", suffix=".tmp", delete=False,
        ) as f:
            temporary = Path(f.name)
            writer = csv.writer(f)
            writer.writerow(list(HOTCRP_CSV_HEADER))
            writer.writerow(["all", "clearreview", "all", round])
            for pid in sorted(pairs):
                for email, action in sorted(pairs[pid]):
                    writer.writerow([pid, action, email, round])
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--log", default=DEFAULT_LOG, help="HotCRP action log CSV export")
    parser.add_argument("--out", default=DEFAULT_OUT, help="where to write the upload-shaped CSV")
    parser.add_argument(
        "--kind", default="primary", choices=[*sorted(REVIEW_ACTIONS), "all"],
        help="review kind to emit (default: primary)",
    )
    parser.add_argument(
        "--round", default="R1",
        help='review round to emit, or "all" (default: R1)',
    )
    args = parser.parse_args()

    try:
        rows = hotcrp_log.load_log(args.log)
    except OSError as exc:
        print(f"cannot read the action log: {exc}", file=sys.stderr)
        return 1
    live, anomalies = hotcrp_log.replay_assignments(rows)

    if args.kind == "all" or args.round == "all":
        # One file cannot be a complete statement of two rounds: its single
        # `clearreview` row names one round, so the other round's rows would
        # be uploaded as additions on top of whatever is already there.
        print(
            'refusing to write a mixed file: --kind/--round "all" is for reading, '
            "not for uploading. Run once per (kind, round).",
            file=sys.stderr,
        )
        return 1

    pairs_by_pid = hotcrp_log.live_pairs(live, kind=args.kind, round=args.round)
    action = REVIEW_ACTIONS[args.kind]
    rows_by_pid = {pid: {(email, action) for email in emails} for pid, emails in pairs_by_pid.items()}

    try:
        write_hotcrp_csv(args.out, rows_by_pid, args.round)
    except OSError as exc:
        print(f"cannot write {args.out}: {exc}", file=sys.stderr)
        return 1

    n_pairs = sum(len(emails) for emails in pairs_by_pid.values())
    load = Counter(email for emails in pairs_by_pid.values() for email in emails)
    print(
        f"{n_pairs} live {args.kind}/{args.round} review(s) over {len(pairs_by_pid)} paper(s) "
        f"and {len(load)} reviewer(s) -> {args.out}",
        file=sys.stderr,
    )
    print(
        f"    replayed {len(rows)} log rows spanning {rows[0]['date']} .. {rows[-1]['date']}",
        file=sys.stderr,
    )

    per_paper = Counter(len(emails) for emails in pairs_by_pid.values())
    print("    reviewers per paper:", file=sys.stderr)
    for size in sorted(per_paper):
        print(f"        {per_paper[size]:>5} paper(s) with {size}", file=sys.stderr)
    per_reviewer = Counter(load.values())
    print("    papers per reviewer:", file=sys.stderr)
    for size in sorted(per_reviewer):
        print(f"        {per_reviewer[size]:>5} reviewer(s) with {size}", file=sys.stderr)

    if anomalies:
        tally = Counter(reason for reason, _ in anomalies)
        print(
            f"\n{len(anomalies)} log anomal(ies) -- the export's window opens after some "
            "reviews were created, which is expected, not an error:",
            file=sys.stderr,
        )
        for reason, count in tally.most_common():
            print(f"    {count:>5}  {reason}", file=sys.stderr)

    other = Counter(
        (review.kind, review.round) for review in live.values()
        if review.kind != args.kind or review.round != args.round
    )
    if other:
        print("\nlive reviews this file does NOT cover:", file=sys.stderr)
        for (kind, round), count in sorted(other.items()):
            print(f"    {count:>5}  {kind}, round {round}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
