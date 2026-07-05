"""Solve a global, load-capped assignment of reviewers to every paper.

    python assign_reviewers.py
    python assign_reviewers.py --light-cap 1 --full-cap 2 --reviewers-per-paper 6

Unlike score_papers.py's independent per-paper ranking, a per-reviewer paper
cap only makes sense considered across ALL papers at once: if two papers
both want the same top-scoring reviewer and that reviewer is capped,
something has to give. This solves that as a stable matching — the
"Hospital/Residents problem", the many-to-many capacitated generalization of
Gale-Shapley (the algorithm behind the US medical residency match) — via
paper-proposing deferred acceptance: each paper proposes to its
highest-scoring eligible reviewers first; a reviewer holds the best offers
received so far (up to their cap), bumping a worse one whenever a better
offer arrives. This is guaranteed to terminate in an assignment with zero
"blocking pairs" (no reviewer-paper pair that would both rather swap into
each other over a current match) and is paper-optimal: each paper gets the
best slate of reviewers achievable in any stable matching.

Eligibility (COI exclusion + area gate) is identical to score_papers.py, via
paper_matching.eligible_scores.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict, deque

import numpy as np

import fingerprint as fp
from paper_matching import build_paper_fingerprints, eligible_scores, load_papers
from reviewers import load_reviewers

DEFAULT_CSV = "HPCA'27 PC Member Acceptance Form (Responses) - Form Responses 1.csv"
DEFAULT_DATA = "hpca2027-data.json"
DEFAULT_FINGERPRINT_CACHE = "fingerprints.json"
DEFAULT_PAPER_CACHE = "paper_fingerprints.json"


def deferred_acceptance(
    pids: list[int],
    paper_prefs: dict[int, list[str]],
    paper_target: dict[int, int],
    reviewer_cap: dict[str, int],
    score_lookup: dict[tuple[str, int], float],
) -> dict[int, list[str]]:
    """Paper-proposing Hospital/Residents deferred acceptance.

    `paper_prefs[pid]` must already be sorted by descending score (eligible
    reviewers only). Returns `{pid: [email, ...]}`, the stable assignment.
    """
    paper_ptr = {pid: 0 for pid in pids}
    paper_held: dict[int, list[str]] = {pid: [] for pid in pids}
    reviewer_held: dict[str, list[tuple[int, float]]] = {}

    queue = deque(pid for pid in pids if paper_target[pid] > 0 and paper_prefs[pid])
    while queue:
        pid = queue.popleft()
        if len(paper_held[pid]) >= paper_target[pid] or paper_ptr[pid] >= len(paper_prefs[pid]):
            continue

        email = paper_prefs[pid][paper_ptr[pid]]
        paper_ptr[pid] += 1
        score = score_lookup[(email, pid)]
        held = reviewer_held.setdefault(email, [])

        bumped_pid = None
        if len(held) < reviewer_cap[email]:
            held.append((pid, score))
            paper_held[pid].append(email)
        else:
            worst_i = min(range(len(held)), key=lambda i: held[i][1])
            worst_pid, worst_score = held[worst_i]
            if score > worst_score:
                held[worst_i] = (pid, score)
                paper_held[pid].append(email)
                paper_held[worst_pid].remove(email)
                bumped_pid = worst_pid
            # else: rejected — pid tries its next candidate on a later turn

        if bumped_pid is not None:
            queue.append(bumped_pid)
        if len(paper_held[pid]) < paper_target[pid] and paper_ptr[pid] < len(paper_prefs[pid]):
            queue.append(pid)

    return paper_held


def count_blocking_pairs(
    eligible_by_pid: dict[int, list[tuple[str, float]]],
    paper_held: dict[int, list[str]],
    reviewer_cap: dict[str, int],
    paper_target: dict[int, int],
    score_lookup: dict[tuple[str, int], float],
) -> int:
    """Number of (reviewer, paper) pairs that would both prefer each other
    over one of their current matches — should always be 0 for a stable
    assignment; a self-check on `deferred_acceptance`'s guarantee."""
    reviewer_papers: dict[str, list[int]] = defaultdict(list)
    for pid, emails in paper_held.items():
        for email in emails:
            reviewer_papers[email].append(pid)

    blocking = 0
    for pid, pairs in eligible_by_pid.items():
        held = set(paper_held[pid])
        for email, score in pairs:
            if email in held:
                continue
            my_papers = reviewer_papers.get(email, [])
            reviewer_wants = len(my_papers) < reviewer_cap[email] or any(
                score_lookup[(email, p2)] < score for p2 in my_papers
            )
            if not reviewer_wants:
                continue
            current = paper_held[pid]
            paper_wants = len(current) < paper_target[pid] or any(
                score_lookup[(r2, pid)] < score for r2 in current
            )
            if paper_wants:
                blocking += 1
    return blocking


def reviewer_paper_cap(r, light_cap: int, full_cap: int) -> int:
    """Per-reviewer paper cap: the CSV's override, if set, else the tier default."""
    if r.override_cap is not None:
        return r.override_cap
    return light_cap if r.tier == "light" else full_cap


def build_canonical_area_map(reviewers_by_email: dict) -> dict[str, str]:
    """Lowercase area name -> canonical (reviewer CSV) spelling.

    HotCRP topic strings don't always match the CSV's casing exactly (e.g.
    "Memory Systems" vs "Memory systems") — the same mismatch the area gate
    already normalizes for. The report should be keyed by the reviewer-facing
    spelling, since that's what a chair would use when recruiting more PC
    members for a short area.
    """
    m: dict[str, str] = {}
    for r in reviewers_by_email.values():
        for area in (r.primary, r.secondary, r.tertiary):
            if area:
                m.setdefault(area.lower(), area)
    return m


def area_pool_stats(
    candidate_emails: list[str], reviewers_by_email: dict, light_cap: int, full_cap: int
) -> dict[str, dict]:
    """Reviewer-pool size and total capacity per area, counting primary/secondary only
    (matching the area gate's own rule — tertiary doesn't count there either).

    Computed from the whole candidate pool, independent of any single paper's
    COI exclusions, so a reviewer conflicted out of the one paper in their
    area doesn't make that area's pool look emptier than it really is.
    """
    stats: dict[str, dict] = {}
    for email in candidate_emails:
        r = reviewers_by_email[email]
        cap = reviewer_paper_cap(r, light_cap, full_cap)
        for area in {r.primary, r.secondary} - {""}:
            s = stats.setdefault(area, {"reviewers": 0, "light": 0, "full": 0, "capacity": 0})
            s["reviewers"] += 1
            s[r.tier] += 1
            s["capacity"] += cap
    return stats


def shortage_report(
    papers: list[dict],
    paper_held: dict[int, list[str]],
    paper_target: dict[int, int],
    area_stats: dict[str, dict],
    canonical_areas: dict[str, str],
) -> int:
    """Print which areas need more reviewer capacity to satisfy the requested load.

    A multi-topic paper's shortfall is attributed to *every* one of its
    topic areas (not split) — we can't cleanly tell which topic's scarcity
    actually caused the gap, so the goal is "where to look," not a precise
    partition. Returns the total number of unfilled reviewer-slots.
    """
    shortfalls: dict[str, list[tuple[int, str, int]]] = defaultdict(list)
    total_missing = 0
    under_filled_papers = 0

    for p in papers:
        pid = p["pid"]
        missing = paper_target[pid] - len(paper_held[pid])
        if missing <= 0:
            continue
        total_missing += missing
        under_filled_papers += 1
        for topic in p.get("topics", []):
            area = canonical_areas.get(topic.lower(), topic)
            shortfalls[area].append((pid, p["title"], missing))

    print("\n=== Shortage report ===")
    if not shortfalls:
        print("None — every paper reached its requested reviewer count.")
        return 0

    print(f"{total_missing} reviewer-slot(s) unfilled across {under_filled_papers} paper(s):")
    for area, entries in sorted(shortfalls.items(), key=lambda kv: -sum(e[2] for e in kv[1])):
        stat = area_stats.get(area, {"reviewers": 0, "light": 0, "full": 0, "capacity": 0})
        print(f"\n{area}")
        print(
            f"    current pool: {stat['reviewers']} reviewers "
            f"({stat['full']} full, {stat['light']} light) = {stat['capacity']} total capacity slots"
        )
        for pid, title, missing in entries:
            print(f"    [{pid}] {title} — missing {missing}")

    return total_missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=DEFAULT_DATA, help="path to the HotCRP paper export JSON")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="path to the reviewer CSV")
    parser.add_argument(
        "--fingerprint-cache", default=DEFAULT_FINGERPRINT_CACHE, help="path to the reviewer fingerprint cache"
    )
    parser.add_argument(
        "--paper-cache", default=DEFAULT_PAPER_CACHE, help="path to the writable paper fingerprint cache"
    )
    parser.add_argument(
        "--reviewers-per-paper", type=int, default=6, help="target reviewer slots per paper (default: 6)"
    )
    parser.add_argument("--light-cap", type=int, default=1, help="max papers per light PC member (default: 1)")
    parser.add_argument("--full-cap", type=int, default=2, help="max papers per full PC member (default: 2)")
    parser.add_argument(
        "--area-weight", type=float, default=1.0,
        help="weight of the topics document relative to the title+abstract document (default: 1.0)"
    )
    parser.add_argument(
        "--no-area-gate", action="store_true",
        help="skip the hard area-eligibility gate; consider all non-conflicted reviewers"
    )
    parser.add_argument("--device", default="cuda", help="torch device for SPECTER2 (default: cuda)")
    args = parser.parse_args()

    papers = load_papers(args.data)
    if not papers:
        print(f"No papers found in {args.data}", file=sys.stderr)
        return 1

    paper_cache = fp.load_fingerprint_cache(args.paper_cache)
    build_paper_fingerprints(
        papers, paper_cache, args.paper_cache, area_weight=args.area_weight, device=args.device
    )

    reviewer_fp = fp.load_fingerprint_cache(args.fingerprint_cache)
    reviewers_by_email = {r.email: r for r in load_reviewers(args.csv)}
    candidate_emails = [e for e in reviewer_fp if e in reviewers_by_email]
    candidate_matrix = np.array([reviewer_fp[e]["vector"] for e in candidate_emails], dtype=np.float32)

    eligible_by_pid: dict[int, list[tuple[str, float]]] = {}
    score_lookup: dict[tuple[str, int], float] = {}
    reviewer_cap: dict[str, int] = {}
    for p in papers:
        pid = p["pid"]
        paper_vec = np.array(paper_cache[str(pid)]["vector"], dtype=np.float32)
        pairs = eligible_scores(
            p, candidate_emails, candidate_matrix, paper_vec, reviewers_by_email,
            area_gate=not args.no_area_gate,
        )
        eligible_by_pid[pid] = pairs
        for email, score in pairs:
            score_lookup[(email, pid)] = score
            reviewer_cap[email] = reviewer_paper_cap(reviewers_by_email[email], args.light_cap, args.full_cap)

    pids = [p["pid"] for p in papers]
    paper_target = {pid: args.reviewers_per_paper for pid in pids}
    paper_prefs = {
        pid: [email for email, _ in sorted(eligible_by_pid[pid], key=lambda es: -es[1])] for pid in pids
    }

    paper_held = deferred_acceptance(pids, paper_prefs, paper_target, reviewer_cap, score_lookup)

    # --- Report ---------------------------------------------------------------
    reviewer_load: dict[str, int] = defaultdict(int)
    for p in papers:
        pid = p["pid"]
        assigned = sorted(paper_held[pid], key=lambda e: -score_lookup[(e, pid)])
        under_filled = "  *** UNDER-FILLED ***" if len(assigned) < args.reviewers_per_paper else ""
        print(f"\n=== [{pid}] {p['title']}")
        print(f"    topics: {', '.join(p.get('topics', []))}")
        print(f"    assigned {len(assigned)} of {args.reviewers_per_paper} requested{under_filled}")
        for rank, email in enumerate(assigned, 1):
            r = reviewers_by_email[email]
            print(f"  {rank:2d}. {score_lookup[(email, pid)]:.3f}  {r.name} <{email}>  [{r.tier}]  ({r.primary})")
            reviewer_load[email] += 1

    total_pairs = sum(len(v) for v in paper_held.values())
    light_over = sum(
        1
        for e, n in reviewer_load.items()
        if reviewers_by_email[e].tier == "light" and n > reviewer_paper_cap(reviewers_by_email[e], args.light_cap, args.full_cap)
    )
    full_over = sum(
        1
        for e, n in reviewer_load.items()
        if reviewers_by_email[e].tier == "full" and n > reviewer_paper_cap(reviewers_by_email[e], args.light_cap, args.full_cap)
    )
    blocking = count_blocking_pairs(eligible_by_pid, paper_held, reviewer_cap, paper_target, score_lookup)

    canonical_areas = build_canonical_area_map(reviewers_by_email)
    area_stats = area_pool_stats(candidate_emails, reviewers_by_email, args.light_cap, args.full_cap)
    total_missing = shortage_report(papers, paper_held, paper_target, area_stats, canonical_areas)

    print(
        f"\nDone. {total_pairs} reviewer-paper pairs assigned across {len(papers)} papers, "
        f"{len(reviewer_load)} distinct reviewers used "
        f"(light cap {args.light_cap}, full cap {args.full_cap}; "
        f"{light_over} light and {full_over} full over cap — should always be 0; "
        f"{blocking} blocking pairs — should always be 0; "
        f"{total_missing} reviewer-slot(s) unfilled — see shortage report above).",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
