"""Pairwise diff between two assignment runs: what changed, and by how much.

    python -m scripts.diff_assignments old.csv new.csv
    python -m scripts.diff_assignments old.csv old.csv                       # smoke test: 0 churn
    python -m scripts.diff_assignments assignment.csv pairs-full-rerun.csv --score-affinity
    python -m scripts.diff_assignments old.csv new.csv --pids 247,1043,1136

`compare_baselines.py` compares aggregate affinity distributions across arms
but never reads the `email` column, so it cannot say *which* reviewer changed
on *which* paper -- only how the goodness distribution moved. This is the
(paper, reviewer)-pair-level complement: which pids gained or lost a
reviewer, which reviewers gained or lost load, and (when affinity is known on
both sides) how each changed paper's match goodness moved.

Accepts either CSV shape assign_reviewers.py writes (`--hotcrp-csv` or
`--pairs-csv`, auto-detected by header; see reviewer_match.assignment_io) and
either input can be either shape -- comparing a production hotcrp-csv against
a pairs-csv from a scratch rerun is the common case, and requiring identical
formats would mean generating a redundant file every time.

`write_hotcrp_csv` keys pairs on `hotcrp_email`; `write_pairs_csv` keys them
on the roster `email`. These differ for anyone `pc_membership` matched under
a second address, so comparing the two files raw would manufacture false
churn for exactly those reviewers -- --no-normalize is the off switch, but
normalization (which loads the current roster only to build the identity
map, no GPU) is on by default.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from reviewer_match import assignment_io
from reviewer_match import fingerprint as fp
from reviewer_match import pc_membership
from reviewer_match.assignment_io import Pair
from reviewer_match.paths import cache_path
from reviewer_match.reserve_reviewers import DEFAULT_INFO as DEFAULT_RESERVE_INFO
from reviewer_match.reserve_reviewers import load_reserve_reviewers
from reviewer_match.reviewers import DEFAULT_CAP_OVERRIDES, load_reviewers

from scripts.assign_reviewers import (
    DEFAULT_CSV,
    DEFAULT_DATA,
    DEFAULT_FINGERPRINT_CACHE,
    DEFAULT_PAPER_CACHE,
    paper_goodness,
)

DEFAULT_WORST = 20
DEFAULT_TOP_REVIEWERS = 20
DEFAULT_RESERVE_FINGERPRINT_CACHE = cache_path("reserve_fingerprints.json")


def diff_pairs(
    old: dict[int, dict[str, Pair]], new: dict[int, dict[str, Pair]]
) -> tuple[dict[int, tuple[set[str], set[str]]], Counter]:
    """Per-pid (removed, added) email sets, plus a per-reviewer load delta.

    A pid present in one file and not the other is treated as an empty slate
    on the missing side, so a paper that dropped out of the paper selection
    entirely still shows up as "every reviewer removed" rather than being
    silently absent from the report.
    """
    changed: dict[int, tuple[set[str], set[str]]] = {}
    load_delta: Counter = Counter()
    for pid in sorted(set(old) | set(new)):
        old_emails = set(old.get(pid, {}))
        new_emails = set(new.get(pid, {}))
        removed = old_emails - new_emails
        added = new_emails - old_emails
        if removed or added:
            changed[pid] = (removed, added)
        for email in added:
            load_delta[email] += 1
        for email in removed:
            load_delta[email] -= 1
    return changed, load_delta


def reviewer_lineup_changes(
    changed: dict[int, tuple[set[str], set[str]]]
) -> dict[str, tuple[set[int], set[int]]]:
    """{email: (pids lost, pids gained)} for every reviewer touched by any paper's change.

    Distinct from `load_delta`, which is a net count: a reviewer who loses
    one paper and gains a different one nets to zero there and is invisible
    in that report, but they still have a different lineup to look at --
    exactly the case this exists to surface. A reviewer with no lineup
    change at all (unaffected by anything in `changed`) never appears here.
    """
    lineups: dict[str, tuple[set[int], set[int]]] = defaultdict(lambda: (set(), set()))
    for pid, (removed, added) in changed.items():
        for email in removed:
            lineups[email][0].add(pid)
        for email in added:
            lineups[email][1].add(pid)
    return dict(lineups)


def load_identity_map(
    csv_path: str, data_path: str, pcinfo_path: str | None, reserve_info: str,
    cap_overrides: str,
) -> dict[str, str]:
    """{roster email: hotcrp_email} for everyone on the PC or reserve roster."""
    reviewers_by_email = {
        r.email: r
        for r in load_reviewers(csv_path, pcinfo_path=pcinfo_path, cap_overrides_path=cap_overrides)
    }
    for r in load_reserve_reviewers(
        reserve_info, data_path, pcinfo_path=pcinfo_path, cap_overrides_path=cap_overrides
    ):
        reviewers_by_email.setdefault(r.email, r)
    return assignment_io.hotcrp_email_map(reviewers_by_email)


def backfill_affinity(
    pairs: dict[int, dict[str, Pair]], paper_cache: dict, reviewer_fp: dict
) -> tuple[dict[int, dict[str, Pair]], int]:
    """Fill in Pair.affinity for pairs that lack it, via a fingerprint dot-product.

    Both vectors are already L2-normalized (see fingerprint.pool), so this is
    exactly the cosine `eligible_scores` computes -- no COI or eligibility is
    recomputed, just a lookup on a pair already known to have been assigned.
    A pair whose paper or reviewer has no cached fingerprint is left as-is and
    counted, not silently dropped.
    """
    misses = 0
    out: dict[int, dict[str, Pair]] = {}
    for pid, emails in pairs.items():
        paper_vec = paper_cache.get(str(pid), {}).get("vector")
        remapped = {}
        for email, pair in emails.items():
            if pair.affinity is not None:
                remapped[email] = pair
                continue
            reviewer_vec = reviewer_fp.get(email, {}).get("vector")
            if paper_vec is None or reviewer_vec is None:
                misses += 1
                remapped[email] = pair
                continue
            cosine = float(np.dot(np.asarray(paper_vec, dtype=np.float32), np.asarray(reviewer_vec, dtype=np.float32)))
            remapped[email] = Pair(phase=pair.phase, affinity=cosine)
        out[pid] = remapped
    return out, misses


def goodness_delta(
    changed: dict[int, tuple[set[str], set[str]]],
    old: dict[int, dict[str, Pair]],
    new: dict[int, dict[str, Pair]],
) -> tuple[dict[int, float], list[int]]:
    """{pid: new_goodness - old_goodness} for changed pids scorable on both sides."""
    scorable: dict[int, float] = {}
    skipped: list[int] = []
    for pid in changed:
        old_emails = list(old.get(pid, {}))
        new_emails = list(new.get(pid, {}))
        old_scores = {(e, pid): old[pid][e].affinity for e in old_emails}
        new_scores = {(e, pid): new[pid][e].affinity for e in new_emails}
        if any(v is None for v in old_scores.values()) or any(v is None for v in new_scores.values()):
            skipped.append(pid)
            continue
        old_g = paper_goodness({pid: old_emails}, old_scores)[pid]
        new_g = paper_goodness({pid: new_emails}, new_scores)[pid]
        if old_g is None or new_g is None:
            skipped.append(pid)
            continue
        scorable[pid] = new_g - old_g
    return scorable, skipped


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("old", help="the earlier assignment (--hotcrp-csv or --pairs-csv)")
    parser.add_argument("new", help="the later assignment (--hotcrp-csv or --pairs-csv)")
    parser.add_argument("--old-label", help="label for `old` in the report (default: its filename)")
    parser.add_argument("--new-label", help="label for `new` in the report (default: its filename)")
    parser.add_argument(
        "--no-normalize", action="store_true",
        help="compare raw emails as each file recorded them, without mapping onto "
             "hotcrp_email; off by default so a reviewer matched under a second "
             "address doesn't read as manufactured churn"
    )
    parser.add_argument("--csv", default=DEFAULT_CSV, help="PC acceptance-form CSV, for identity normalization")
    parser.add_argument("--data", default=DEFAULT_DATA, help="paper export JSON, for the reserve roster")
    parser.add_argument("--pcinfo", default=pc_membership.DEFAULT_PCINFO, help="HotCRP user export")
    parser.add_argument("--no-pc-check", action="store_true", help="skip the PC-membership prune while loading the roster")
    parser.add_argument("--reserve-info", default=DEFAULT_RESERVE_INFO, help="reserve roster CSV")
    parser.add_argument("--cap-overrides", default=DEFAULT_CAP_OVERRIDES, help="per-reviewer paper-cap override CSV")
    parser.add_argument(
        "--score-affinity", action="store_true",
        help="backfill missing affinity via a fingerprint dot-product lookup, "
             "so a hotcrp-csv (which carries no affinity) can still get a "
             "goodness delta"
    )
    parser.add_argument("--fingerprint-cache", default=DEFAULT_FINGERPRINT_CACHE, help="reviewer fingerprint cache")
    parser.add_argument(
        "--reserve-fingerprint-cache", default=DEFAULT_RESERVE_FINGERPRINT_CACHE,
        help="reserve fingerprint cache"
    )
    parser.add_argument("--paper-cache", default=DEFAULT_PAPER_CACHE, help="paper fingerprint cache")
    parser.add_argument("--pids", help="comma-separated pids to restrict the report to (default: all)")
    parser.add_argument("--worst", type=int, default=DEFAULT_WORST, help="worst goodness-delta papers to print (default: %(default)s)")
    parser.add_argument(
        "--top-reviewers", type=int, default=DEFAULT_TOP_REVIEWERS,
        help="reviewer load-delta rows to print before collapsing the tail (default: %(default)s)"
    )
    args = parser.parse_args()

    old_label = args.old_label or Path(args.old).stem
    new_label = args.new_label or Path(args.new).stem

    old_fmt, old_pairs = assignment_io.load_assignment_pairs(args.old)
    new_fmt, new_pairs = assignment_io.load_assignment_pairs(args.new)

    normalized = not args.no_normalize
    if normalized:
        pcinfo = None if args.no_pc_check else args.pcinfo
        email_map = load_identity_map(args.csv, args.data, pcinfo, args.reserve_info, args.cap_overrides)
        remapped = sum(1 for e, h in email_map.items() if e != h)
        old_pairs = assignment_io.remap_pairs(old_pairs, email_map)
        new_pairs = assignment_io.remap_pairs(new_pairs, email_map)

    if args.score_affinity:
        paper_cache = fp.load_fingerprint_cache(args.paper_cache)
        reviewer_fp = fp.load_fingerprint_cache(args.fingerprint_cache)
        reviewer_fp.update(fp.load_fingerprint_cache(args.reserve_fingerprint_cache))
        old_pairs, old_misses = backfill_affinity(old_pairs, paper_cache, reviewer_fp)
        new_pairs, new_misses = backfill_affinity(new_pairs, paper_cache, reviewer_fp)
        if old_misses or new_misses:
            print(
                f"WARNING: {old_misses} pair(s) in {old_label} and {new_misses} in "
                f"{new_label} have no cached paper/reviewer fingerprint and could "
                f"not be scored", file=sys.stderr,
            )

    if args.pids:
        wanted = {int(p) for p in args.pids.split(",")}
        old_pairs = {pid: v for pid, v in old_pairs.items() if pid in wanted}
        new_pairs = {pid: v for pid, v in new_pairs.items() if pid in wanted}

    changed, load_delta = diff_pairs(old_pairs, new_pairs)
    total_removed = sum(len(r) for r, _ in changed.values())
    total_added = sum(len(a) for _, a in changed.values())

    print(f"=== diff_assignments: {old_label} -> {new_label} ===")
    print(f"old: {args.old} ({old_fmt}, {len(old_pairs)} pid, {sum(len(v) for v in old_pairs.values())} pairs)")
    print(f"new: {args.new} ({new_fmt}, {len(new_pairs)} pid, {sum(len(v) for v in new_pairs.values())} pairs)")
    if normalized:
        print(f"email normalization: on ({remapped} roster email(s) mapped to a different hotcrp address)")
    else:
        print("email normalization: off (--no-normalize)")

    print(f"\n=== Papers changed ===")
    print(f"{len(changed)} of {len(set(old_pairs) | set(new_pairs))} paper(s) changed reviewer composition.")
    for pid in sorted(changed):
        removed, added = changed[pid]
        print(f"[{pid}] {len(old_pairs.get(pid, {}))} -> {len(new_pairs.get(pid, {}))} reviewers")
        for email in sorted(removed):
            print(f"    - removed: {email}")
        for email in sorted(added):
            affinity = new_pairs[pid][email].affinity
            note = f"  (affinity {affinity:.3f})" if affinity is not None else ""
            print(f"    + added:   {email}{note}")

    print(f"\n=== Pair-level churn ===")
    print(f"removed: {total_removed} pair(s)   added: {total_added} pair(s)   touching {len(changed)} paper(s)")

    print(f"\n=== Reviewer load deltas ===")
    ranked = sorted(load_delta.items(), key=lambda kv: (-abs(kv[1]), kv[0]))
    shown = [kv for kv in ranked if kv[1] != 0][: args.top_reviewers]
    for email, delta in shown:
        print(f"{email}   {delta:+d}")
    remaining = sum(1 for _, d in ranked if d != 0) - len(shown)
    if remaining > 0:
        print(f"... +{remaining} more reviewer(s) with a load delta")

    lineups = reviewer_lineup_changes(changed)
    gained_only = sum(1 for lost, gained in lineups.values() if not lost and gained)
    lost_only = sum(1 for lost, gained in lineups.values() if lost and not gained)
    swapped = sum(1 for lost, gained in lineups.values() if lost and gained)
    print(f"\n=== Reviewers with a changed lineup ===")
    print(
        f"{len(lineups)} distinct reviewer(s) have a different set of papers "
        f"({gained_only} gained only, {lost_only} lost only, {swapped} swapped "
        f"at least one paper for another -- net load delta 0, invisible above)."
    )
    swap_ranked = sorted(
        ((e, l, g) for e, (l, g) in lineups.items() if l and g),
        key=lambda t: (-min(len(t[1]), len(t[2])), t[0]),
    )
    for email, lost, gained in swap_ranked[: args.top_reviewers]:
        print(f"{email}   -{len(lost)} +{len(gained)} (swapped)")
    swap_remaining = swapped - min(swapped, args.top_reviewers)
    if swap_remaining > 0:
        print(f"... +{swap_remaining} more reviewer(s) with a swapped lineup")

    scorable, skipped = goodness_delta(changed, old_pairs, new_pairs)
    print(f"\n=== Goodness delta (changed papers, affinity known on both sides) ===")
    if scorable:
        deltas = list(scorable.values())
        mean_delta = sum(deltas) / len(deltas)
        print(
            f"mean delta: {mean_delta:+.3f}  ({len(scorable)} of {len(changed)} papers "
            f"scored; {len(skipped)} skipped — affinity unknown)"
        )
        worst = sorted(scorable.items(), key=lambda kv: kv[1])[: args.worst]
        print(f"worst {len(worst)} by delta:")
        for pid, delta in worst:
            print(f"  [{pid}] {delta:+.3f}")
    else:
        print(f"no papers scorable (0 of {len(changed)}; pass --score-affinity to backfill missing affinity)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
