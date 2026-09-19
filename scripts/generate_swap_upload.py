"""Turn confirmed reviewer-swap picks into a HotCRP bulk-assignment upload.

    python -m scripts.generate_swap_upload --departed-email person@example.edu \\
        --confirmed-csv data/curated/confirmed_swaps.csv \\
        --out outputs/assignments/swap_upload.csv

Reads a hand-edited "confirmed swaps" CSV -- the chair's own record of who
actually said yes, after scripts.propose_reviewer_swaps only proposed
candidates -- header `target_pid,add_email,source_pid,remove_source`, and
writes the small HotCRP Assignments -> Bulk update delta that applies it:
for each row, `add_email` is added to `target_pid` (clearing
`--departed-email`'s own review there first) and, unless `remove_source` is
"no", is also removed from `source_pid`. `remove_source: no` is the rare
exception -- a reviewer who agreed to take on the target paper without
giving up the one they are already partway into (an add, not a move); every
other row is a genuine swap.

`--clear-pids` adds a bare clear row for each further paper the departed
reviewer holds -- the usual case of leaving the committee outright, where
only the papers left short need a replacement but every one needs clearing.
A confirmed row may leave `source_pid` blank when `remove_source` is no,
for a reviewer with spare capacity taking a paper on without giving one up.

This is NOT a full replacement and never opens with `all,clearreview,all,R1`
the way assign_reviewers.py's own upload does: only the pairs named in
--confirmed-csv are touched, so uploading it changes nothing for any other
paper. As always, use HotCRP's own preview before approving the upload.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

from reviewer_match import pc_membership
from reviewer_match.paper_matching import parse_exclude_pids
from reviewer_match.paths import assignment_path, curated_path
from reviewer_match.reserve_reviewers import DEFAULT_INFO as DEFAULT_RESERVE_INFO
from reviewer_match.reserve_reviewers import load_reserve_reviewers
from reviewer_match.reviewers import DEFAULT_CAP_OVERRIDES, load_reviewers

from scripts import assign_reviewers as ar

DEFAULT_CONFIRMED_CSV = curated_path("confirmed_swaps.csv")
DEFAULT_OUT = assignment_path("swap_upload.csv")
DEFAULT_ROUND = "R1"

CONFIRMED_HEADER = ("target_pid", "add_email", "source_pid", "remove_source")
UPLOAD_HEADER = ("paper", "action", "email", "round")


def load_confirmed_swaps(path: str) -> list[tuple[int, str, int | None, bool]]:
    """[(target_pid, add_email, source_pid, remove_source)] from a hand-edited CSV.

    `remove_source` reads as false only for "no"/"n"/"false"/"0" (case
    insensitive); anything else, including a blank cell, is a normal swap.

    `source_pid` may be blank, and then reads as None: a reviewer with spare
    capacity who is taking the target paper on without giving anything up has
    no source paper to name. That is only coherent when `remove_source` says
    no -- a blank source with a row that asks for the source to be cleared is
    a mistake in the file, not a no-op, so it raises rather than quietly
    dropping half of a swap.
    """
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != list(CONFIRMED_HEADER):
            raise ValueError(f"{path}: expected header {CONFIRMED_HEADER}, got {reader.fieldnames}")
        for row in reader:
            remove_source = row["remove_source"].strip().lower() not in ("no", "n", "false", "0")
            raw_source = row["source_pid"].strip()
            if not raw_source and remove_source:
                raise ValueError(
                    f"{path}: paper {row['target_pid']} names no source_pid but asks to clear "
                    f"it; set remove_source to no for an add, or name the source paper"
                )
            rows.append((
                int(row["target_pid"]),
                row["add_email"].strip().lower(),
                int(raw_source) if raw_source else None,
                remove_source,
            ))
    return rows


def build_upload_rows(
    confirmed: list[tuple[int, str, int | None, bool]],
    departed_email: str,
    hotcrp_email: dict[str, str],
    clear_pids: frozenset[int] = frozenset(),
) -> list[tuple[int, str, str, str]]:
    """[(paper, action, email, round)] for the delta upload -- see module docstring.

    `hotcrp_email` maps a roster email to the address HotCRP actually has
    marked `pc` (identity if the address isn't in the map at all, so an
    unresolvable email still round-trips instead of silently vanishing).

    `clear_pids` are further papers to clear `departed_email` from, for the
    ordinary case of someone leaving the committee outright: only the papers
    they left *short* need a replacement, but every paper they hold needs the
    review cleared, and the ones already covered above are not repeated.
    """
    out = []
    departed_hotcrp = hotcrp_email.get(departed_email, departed_email)
    for target_pid, add_email, source_pid, remove_source in confirmed:
        add_hotcrp = hotcrp_email.get(add_email, add_email)
        out.append((target_pid, "clearreview", departed_hotcrp, DEFAULT_ROUND))
        out.append((target_pid, "primaryreview", add_hotcrp, DEFAULT_ROUND))
        if remove_source:
            out.append((source_pid, "clearreview", add_hotcrp, DEFAULT_ROUND))
    for pid in sorted(clear_pids - {target_pid for target_pid, _e, _s, _r in confirmed}):
        out.append((pid, "clearreview", departed_hotcrp, DEFAULT_ROUND))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--departed-email", required=True, help="reviewer being cleared from every --confirmed-csv target paper")
    parser.add_argument("--confirmed-csv", default=DEFAULT_CONFIRMED_CSV, help="hand-maintained confirmed-swaps CSV (see module docstring)")
    parser.add_argument("--clear-pids", type=parse_exclude_pids, default=frozenset(),
                        help="comma-separated further paper IDs to clear --departed-email from, for a "
                             "reviewer leaving the committee outright: every paper they hold needs the "
                             "review cleared, not just the ones a confirmed swap backfills")
    parser.add_argument("--out", default=DEFAULT_OUT, help="path to write the HotCRP bulk-assignment delta")
    parser.add_argument("--csv", default=ar.DEFAULT_CSV, help="path to the reviewer CSV")
    parser.add_argument("--cap-overrides", default=DEFAULT_CAP_OVERRIDES, help="hand-maintained per-reviewer paper-cap override CSV")
    parser.add_argument("--pcinfo", default=pc_membership.DEFAULT_PCINFO, help="HotCRP user export deciding who is still on the PC")
    parser.add_argument("--no-pc-check", action="store_true", help="keep everyone the roster lists, even if HotCRP no longer marks them pc")
    parser.add_argument("--data", default=ar.DEFAULT_DATA, help="path to the HotCRP paper export JSON")
    parser.add_argument("--reserve-info", default=DEFAULT_RESERVE_INFO, help="reserve roster, for a mover who is a reserve reviewer")
    args = parser.parse_args()

    try:
        confirmed = load_confirmed_swaps(args.confirmed_csv)
    except FileNotFoundError:
        print(f"{args.confirmed_csv} not found", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if not confirmed:
        print(f"{args.confirmed_csv} has no rows; nothing to write.", file=sys.stderr)
        return 0

    pcinfo = None if args.no_pc_check else args.pcinfo
    reviewers_by_email = {
        r.email: r for r in load_reviewers(args.csv, pcinfo_path=pcinfo, cap_overrides_path=args.cap_overrides)
    }
    reserves = load_reserve_reviewers(args.reserve_info, args.data, pcinfo_path=pcinfo, cap_overrides_path=args.cap_overrides)
    promoted, _already_pc, true_reserves = ar.split_promoted_reserves(reserves, reviewers_by_email)
    reviewers_by_email.update({r.email: r for r in promoted + true_reserves})

    hotcrp_email = {email: r.hotcrp_email for email, r in reviewers_by_email.items()}
    unresolved = sorted({
        email for _t, email, _s, _r in confirmed if email not in hotcrp_email
    } | ({args.departed_email} if args.departed_email not in hotcrp_email else set()))
    if unresolved:
        print(f"WARNING: {len(unresolved)} email(s) not found on the current roster, used as-is: "
              f"{', '.join(unresolved)}", file=sys.stderr)

    rows = build_upload_rows(confirmed, args.departed_email, hotcrp_email, args.clear_pids)

    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", dir=target.parent,
            prefix=f".{target.name}.", suffix=".tmp", delete=False,
        ) as f:
            temporary = Path(f.name)
            writer = csv.writer(f)
            writer.writerow(list(UPLOAD_HEADER))
            for row in rows:
                writer.writerow(row)
        os.replace(temporary, target)
        temporary = None
    except OSError as exc:
        print(f"ERROR: could not write {args.out}: {exc}", file=sys.stderr)
        return 1
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()

    print(f"{len(rows)} row(s) ({len(confirmed)} paper(s)) -> {args.out}. This is a delta, not a replacement -- "
          f"preview it in HotCRP before approving.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
