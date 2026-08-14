"""Pin-emails list for a *targeted* incremental rerun: protect everyone with
HotCRP activity or a manually-edited pair, except a named force-release list.

    python -m scripts.build_targeted_rerun_pins \\
        --force-release data/curated/dblp_merge_release_list.txt

`make rerun` (CLAUDE.md section 4d) freezes whoever
`scripts.audit_reviewer_activity`'s activity signal has already touched and
releases everyone else -- the right default for an ordinary rerun, and not
enough for a targeted one: a reviewer on a named "these people's fingerprints
were wrong, re-solve around them regardless" list may well have already
logged into HotCRP, and would stay pinned (silently kept) under a plain
rerun. This script computes the pin-emails list a targeted rerun needs
instead: (1) everyone `--activity-pinned` already names, union (2) everyone
touched by a *manual edit* -- a (paper, reviewer) pair present in
`--current-csv` (what HotCRP holds) but not `--baseline-csv` (what the
pipeline last proposed), or vice versa. (2) exists because
`audit_reviewer_activity.py` only ever credits activity to the *acting*
account and explicitly excludes the chair (see
`hotcrp_log.last_activity`/`engagement_actions`), so a pair a chair
hand-edited for a reviewer who has never personally logged in is invisible
to (1) alone.

The named `--force-release` set is then subtracted from (1) union (2) --
*except* a force-released reviewer who has any manually-edited pair is held
back entirely, reported, and NOT released. `--pin-emails` pins a reviewer,
not a single pair, so partial release isn't possible without changing
assign_reviewers.py itself; holding the whole reviewer back is the
conservative reading of "a manual edit always wins" -- naming someone for an
unrelated fix must never risk silently undoing a chair's deliberate pair.

Feed the result straight to `scripts.assign_reviewers --pin-emails`,
alongside `--pin-csv` pointed at the same `--current-csv`.
"""

from __future__ import annotations

import argparse
import sys

from reviewer_match import assignment_io
from reviewer_match import pc_membership
from reviewer_match.paths import assignment_path, curated_path, report_path
from reviewer_match.reserve_reviewers import DEFAULT_DATA
from reviewer_match.reserve_reviewers import DEFAULT_INFO as DEFAULT_RESERVE_INFO
from reviewer_match.reviewers import DEFAULT_CAP_OVERRIDES

from scripts.assign_reviewers import DEFAULT_CSV
from scripts.diff_assignments import diff_pairs, load_identity_map

DEFAULT_ACTIVITY_PINNED = report_path("reviewers_pinned.txt")
DEFAULT_BASELINE_CSV = assignment_path("assignment.csv")
DEFAULT_CURRENT_CSV = assignment_path("current_assignment.csv")
DEFAULT_OUT = report_path("reviewers_pinned_targeted.txt")


def load_email_list(path: str) -> set[str]:
    """One lowercased address per line; blank lines and `#` comments skipped."""
    with open(path, encoding="utf-8") as f:
        return {
            line.strip().lower() for line in f
            if line.strip() and not line.strip().startswith("#")
        }


def manually_modified(
    baseline_pairs: dict[int, dict[str, "assignment_io.Pair"]],
    current_pairs: dict[int, dict[str, "assignment_io.Pair"]],
) -> dict[str, list[tuple[int, str]]]:
    """{email: [(pid, "removed"/"added"), ...]} for every pair that deviates.

    A deviation is either direction: HotCRP holding a pair the pipeline never
    proposed, or the pipeline having proposed one HotCRP no longer holds.
    Both are read the same way here -- something changed this reviewer's
    slate since the baseline, deliberately or not, and that is reason enough
    to protect them from an unrelated re-solve.
    """
    changed, _load_delta = diff_pairs(baseline_pairs, current_pairs)
    touched: dict[str, list[tuple[int, str]]] = {}
    for pid, (removed, added) in changed.items():
        for email in removed:
            touched.setdefault(email, []).append((pid, "removed"))
        for email in added:
            touched.setdefault(email, []).append((pid, "added"))
    return touched


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--activity-pinned", default=DEFAULT_ACTIVITY_PINNED,
        help=f"scripts.audit_reviewer_activity's --pinned-out file (default: {DEFAULT_ACTIVITY_PINNED})"
    )
    parser.add_argument(
        "--baseline-csv", default=DEFAULT_BASELINE_CSV,
        help=f"the pipeline's last full proposal (default: {DEFAULT_BASELINE_CSV})"
    )
    parser.add_argument(
        "--current-csv", default=DEFAULT_CURRENT_CSV,
        help=f"what HotCRP currently holds, from scripts.extract_log_assignments "
             f"(default: {DEFAULT_CURRENT_CSV})"
    )
    parser.add_argument(
        "--force-release", required=True, metavar="PATH",
        help="one address per line (# comments allowed): reviewers to release "
             "regardless of activity, unless a specific pair of theirs was manually edited"
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"default: {DEFAULT_OUT}")
    parser.add_argument(
        "--no-normalize", action="store_true",
        help="skip roster-based hotcrp/roster email normalization (see diff_assignments.py); "
             "off by default, same as there"
    )
    parser.add_argument("--csv", default=DEFAULT_CSV, help="PC acceptance-form CSV, for identity normalization")
    parser.add_argument("--data", default=DEFAULT_DATA, help="paper export JSON, for the reserve roster")
    parser.add_argument("--pcinfo", default=pc_membership.DEFAULT_PCINFO, help="HotCRP user export")
    parser.add_argument("--no-pc-check", action="store_true", help="skip the PC-membership prune while loading the roster")
    parser.add_argument("--reserve-info", default=DEFAULT_RESERVE_INFO, help="reserve roster CSV")
    parser.add_argument("--cap-overrides", default=DEFAULT_CAP_OVERRIDES, help="per-reviewer paper-cap override CSV")
    args = parser.parse_args()

    activity_pinned = load_email_list(args.activity_pinned)
    force_release = load_email_list(args.force_release)

    _, baseline_pairs = assignment_io.load_assignment_pairs(args.baseline_csv)
    _, current_pairs = assignment_io.load_assignment_pairs(args.current_csv)

    if not args.no_normalize:
        pcinfo = None if args.no_pc_check else args.pcinfo
        email_map = load_identity_map(args.csv, args.data, pcinfo, args.reserve_info, args.cap_overrides)
        baseline_pairs = assignment_io.remap_pairs(baseline_pairs, email_map)
        current_pairs = assignment_io.remap_pairs(current_pairs, email_map)
        force_release = {email_map.get(e, e) for e in force_release}

    touched = manually_modified(baseline_pairs, current_pairs)
    manually_modified_emails = set(touched)

    held_back = force_release & manually_modified_emails
    released = force_release - held_back
    protected = activity_pinned | manually_modified_emails
    final_pinned = protected - released

    print(f"activity-pinned: {len(activity_pinned)}", file=sys.stderr)
    print(f"manually modified (baseline vs. current): {len(manually_modified_emails)}", file=sys.stderr)
    for email in sorted(manually_modified_emails):
        pids = ", ".join(f"[{pid}] {action}" for pid, action in sorted(touched[email]))
        print(f"    {email}: {pids}", file=sys.stderr)
    print(f"force-release named: {len(force_release)}", file=sys.stderr)
    print(f"    released: {len(released)}: {', '.join(sorted(released)) or '(none)'}", file=sys.stderr)
    if held_back:
        print(
            f"    WARNING: {len(held_back)} held back despite being named -- a manual edit "
            f"is present and always wins:",
            file=sys.stderr,
        )
        for email in sorted(held_back):
            pids = ", ".join(f"[{pid}] {action}" for pid, action in sorted(touched[email]))
            print(f"        {email}: {pids}", file=sys.stderr)
    unnamed = force_release - manually_modified_emails - activity_pinned - released
    if unnamed:
        # released already covers this in the ordinary case; only reachable
        # if released and protected somehow disagree, which would be a bug
        # in the set algebra above -- kept as a loud canary rather than
        # silently trusting the arithmetic.
        print(f"    WARNING: {len(unnamed)} named but neither protected nor released: {sorted(unnamed)}", file=sys.stderr)
    print(f"final pinned: {len(final_pinned)}", file=sys.stderr)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"# generated by build_targeted_rerun_pins.py --force-release {args.force_release}\n")
        for email in sorted(final_pinned):
            f.write(email + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
