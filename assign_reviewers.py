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

Seniority constraints (needs reviewer_seniority.csv from classify_reviewers.py;
skip them all with --no-seniority): every paper should get at least
--min-seniors senior reviewers and at most --max-juniors juniors, degrading
gracefully when the pool can't satisfy that. Assignment runs in phases, each a
deferred-acceptance pass whose results are frozen before the next:

  1.  senior anchors — every paper matches its best eligible senior;
  1b. papers that got none fall back to an "almost senior" (a typical-class
      reviewer with >= --almost-senior-window in-window papers);
  2.  main fill — everyone with remaining capacity competes on score, but a
      paper holds at most --max-juniors juniors at a time (a pure cap: a
      well-matched junior can still beat a weak-matched typical to a slot);
  3.  papers still under-filled may take extra "almost not junior" juniors
      (>= --almost-junior-career career papers).

Papers that break the criteria even after degradation are printed in a report
at the end. Each phase is individually stable, but freezing earlier phases
(the anchors) and the junior cap mean the final assignment trades classical
global stability for the composition constraints — the self-check verifies
phase 2's cap-aware stability plus the final composition invariants instead.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

import fingerprint as fp
from classify_reviewers import DEFAULT_OUT as DEFAULT_SENIORITY, load_seniority
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
    junior_emails: frozenset[str] = frozenset(),
    max_juniors: int | None = None,
) -> dict[int, list[str]]:
    """Paper-proposing Hospital/Residents deferred acceptance.

    `paper_prefs[pid]` must already be sorted by descending score (eligible
    reviewers only). Returns `{pid: [email, ...]}`, the stable assignment.

    If `max_juniors` is set, a paper holds at most that many reviewers from
    `junior_emails` at a time. Juniors it can't currently take are deferred,
    not rejected: they stay proposable in score order and get their offer if
    the paper's junior slot later opens (its held junior was bumped away).
    Each (paper, reviewer) pair is still proposed at most once, so
    termination is unchanged.
    """
    paper_ptr = {pid: 0 for pid in pids}
    paper_held: dict[int, list[str]] = {pid: [] for pid in pids}
    juniors_held = {pid: 0 for pid in pids}
    deferred: dict[int, deque[str]] = {pid: deque() for pid in pids}
    reviewer_held: dict[str, list[tuple[int, float]]] = {}

    def junior_ok(pid: int) -> bool:
        return max_juniors is None or juniors_held[pid] < max_juniors

    def next_candidate(pid: int) -> str | None:
        """Best-scoring proposable candidate, honoring the junior cap.

        While the cap is full, juniors at the head of the pref list are moved
        to `deferred` (preserving score order — the pref list is descending,
        so appends keep the deque sorted). When a junior may be taken, the
        deferred head competes with the pref-list head on score.
        """
        prefs = paper_prefs[pid]
        if not junior_ok(pid):
            while paper_ptr[pid] < len(prefs) and prefs[paper_ptr[pid]] in junior_emails:
                deferred[pid].append(prefs[paper_ptr[pid]])
                paper_ptr[pid] += 1
        head = prefs[paper_ptr[pid]] if paper_ptr[pid] < len(prefs) else None
        if junior_ok(pid) and deferred[pid]:
            best_deferred = deferred[pid][0]
            if head is None or score_lookup[(best_deferred, pid)] >= score_lookup[(head, pid)]:
                return deferred[pid].popleft()
        if head is not None:
            paper_ptr[pid] += 1
        return head

    queue = deque(pid for pid in pids if paper_target[pid] > 0 and paper_prefs[pid])
    while queue:
        pid = queue.popleft()
        if len(paper_held[pid]) >= paper_target[pid]:
            continue
        email = next_candidate(pid)
        if email is None:
            # Nothing proposable right now. If deferred juniors remain, the
            # paper is re-queued by the bump that frees its junior slot.
            continue

        score = score_lookup[(email, pid)]
        held = reviewer_held.setdefault(email, [])

        bumped_pid = None
        if len(held) < reviewer_cap[email]:
            held.append((pid, score))
            paper_held[pid].append(email)
            juniors_held[pid] += email in junior_emails
        else:
            worst_i = min(range(len(held)), key=lambda i: held[i][1])
            worst_pid, worst_score = held[worst_i]
            if score > worst_score:
                held[worst_i] = (pid, score)
                paper_held[pid].append(email)
                juniors_held[pid] += email in junior_emails
                paper_held[worst_pid].remove(email)
                juniors_held[worst_pid] -= email in junior_emails
                bumped_pid = worst_pid
            # else: rejected — pid tries its next candidate on a later turn

        if bumped_pid is not None:
            queue.append(bumped_pid)
        more = paper_ptr[pid] < len(paper_prefs[pid]) or (junior_ok(pid) and deferred[pid])
        if len(paper_held[pid]) < paper_target[pid] and more:
            queue.append(pid)

    return paper_held


def count_blocking_pairs(
    eligible_by_pid: dict[int, list[tuple[str, float]]],
    paper_held: dict[int, list[str]],
    reviewer_cap: dict[str, int],
    paper_target: dict[int, int],
    score_lookup: dict[tuple[str, int], float],
    junior_emails: frozenset[str] = frozenset(),
    max_juniors: int | None = None,
) -> int:
    """Number of (reviewer, paper) pairs that would both prefer each other
    over one of their current matches — should always be 0 for a stable
    assignment; a self-check on `deferred_acceptance`'s guarantee.

    With `max_juniors` set, a junior doesn't block a paper whose junior slots
    are full of better-scoring juniors: the paper could only take them by
    dropping a junior, so a non-junior's slot is not up for grabs.
    """
    reviewer_papers: dict[str, list[int]] = defaultdict(list)
    for pid, emails in paper_held.items():
        for email in emails:
            reviewer_papers[email].append(pid)

    blocking = 0
    for pid, pairs in eligible_by_pid.items():
        held = set(paper_held[pid])
        juniors_held = sum(1 for e in paper_held[pid] if e in junior_emails)
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
            if max_juniors is not None and email in junior_emails and juniors_held >= max_juniors:
                paper_wants = any(
                    r2 in junior_emails and score_lookup[(r2, pid)] < score for r2 in current
                )
            else:
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


@dataclass(frozen=True)
class SeniorityPools:
    seniors: frozenset[str]
    almost_seniors: frozenset[str]  # typical-class, window_total >= threshold
    juniors: frozenset[str]
    almost_not_juniors: frozenset[str]  # junior-class, career_total >= threshold


def seniority_pools(
    candidate_emails, seniority: dict[str, dict], almost_senior_window: int, almost_junior_career: int
) -> tuple[SeniorityPools, list[str]]:
    """Split the candidate pool by seniority class from reviewer_seniority.csv.

    Candidates classified 'unknown' or missing from the CSV count as neither
    senior nor junior (they can fill slots but not a senior one); the missing
    ones are also returned so the caller can warn — they usually mean the CSV
    is stale and classify_reviewers.py needs a rerun.
    """
    seniors, almost_seniors, juniors, almost_not = set(), set(), set(), set()
    missing: list[str] = []
    for email in candidate_emails:
        row = seniority.get(email)
        if row is None:
            missing.append(email)
            continue
        cls = row["class"]
        if cls == "senior":
            seniors.add(email)
        elif cls == "junior":
            juniors.add(email)
            if row["career_total"] is not None and row["career_total"] >= almost_junior_career:
                almost_not.add(email)
        elif cls == "typical" and row["window_total"] is not None and row["window_total"] >= almost_senior_window:
            almost_seniors.add(email)
    return (
        SeniorityPools(frozenset(seniors), frozenset(almost_seniors), frozenset(juniors), frozenset(almost_not)),
        missing,
    )


def assignment_phase(
    pids: list[int],
    full_prefs: dict[int, list[str]],
    phase_target: dict[int, int],
    slates: dict[int, list[str]],
    used: dict[str, int],
    reviewer_cap: dict[str, int],
    score_lookup: dict[tuple[str, int], float],
    candidates: frozenset[str] | set[str],
    junior_emails: frozenset[str] = frozenset(),
    max_juniors: int | None = None,
):
    """One accumulating deferred-acceptance pass, restricted to `candidates`.

    `phase_target[pid]` is the number of ADDITIONAL reviewers the paper may
    gain this phase; assignments from earlier phases are frozen — their
    reviewers can't be bumped, which is what makes a phase-1 senior a real
    anchor. Folds the result into `slates` and `used`, and returns this
    phase's (held, prefs, cap) view for self-checks.
    """
    cap = {}
    for email in candidates:
        remaining = reviewer_cap[email] - used[email]
        if remaining > 0:
            cap[email] = remaining
    prefs = {}
    for pid in pids:
        taken = set(slates[pid])
        prefs[pid] = [e for e in full_prefs[pid] if e in cap and e not in taken]

    held = deferred_acceptance(
        pids, prefs, phase_target, cap, score_lookup,
        junior_emails=junior_emails, max_juniors=max_juniors,
    )
    for pid, emails in held.items():
        slates[pid].extend(emails)
        for e in emails:
            used[e] += 1
    return held, prefs, cap


def seniority_report(
    papers: list[dict],
    slates: dict[int, list[str]],
    pools: SeniorityPools,
    reviewers_by_email: dict,
    seniority: dict[str, dict],
    full_prefs: dict[int, list[str]],
    reviewers_per_paper: int,
    min_seniors: int,
    max_juniors: int,
    almost_senior_window: int,
    almost_junior_career: int,
) -> tuple[int, int, int]:
    """Print which papers meet, degrade on, or break the seniority criteria.

    Judged on final slates, not on which phase assigned whom — a true senior
    picked up on score in the main fill satisfies the requirement no matter
    how the anchor phase went. Returns (ok, degraded, breaking) counts.
    """
    ok_count = 0
    degraded: list[tuple[int, str, list[str]]] = []
    breaking: list[tuple[int, str, list[str]]] = []

    for p in papers:
        pid = p["pid"]
        slate = slates[pid]
        true_seniors = [e for e in slate if e in pools.seniors]
        almost = [e for e in slate if e in pools.almost_seniors]
        juniors = [e for e in slate if e in pools.juniors]
        deep_juniors = [e for e in juniors if e not in pools.almost_not_juniors]

        degrade_notes: list[str] = []
        break_notes: list[str] = []

        if len(true_seniors) < min_seniors:
            if len(true_seniors) + len(almost) >= min_seniors:
                for e in almost[: min_seniors - len(true_seniors)]:
                    degrade_notes.append(
                        f"senior slot filled by almost-senior {reviewers_by_email[e].name} "
                        f"({seniority[e]['window_total']} window papers)"
                    )
            else:
                pool_size = sum(1 for e in full_prefs[pid] if e in pools.seniors)
                detail = (
                    f"{pool_size} eligible senior(s), all at capacity on better-matched papers"
                    if pool_size
                    else "no senior passes the area gate for this paper"
                )
                break_notes.append(
                    f"only {len(true_seniors) + len(almost)} of {min_seniors} senior slot(s) filled "
                    f"even counting almost-seniors — {detail}"
                )
        if len(juniors) > max_juniors:
            names = ", ".join(
                f"{reviewers_by_email[e].name} ({seniority[e]['career_total']} career papers)"
                for e in sorted(juniors, key=lambda e: seniority[e]["career_total"] or 0)
            )
            if len(deep_juniors) <= max_juniors:
                degrade_notes.append(
                    f"{len(juniors)} juniors (cap {max_juniors}): {names} — extras within the almost-not-junior allowance"
                )
            else:
                break_notes.append(
                    f"{len(deep_juniors)} juniors below the almost-not-junior line (cap {max_juniors}): {names}"
                )
        if len(slate) < reviewers_per_paper:
            break_notes.append(
                f"{reviewers_per_paper - len(slate)} slot(s) unfilled even after the almost-not-junior relaxation"
            )

        if break_notes:
            breaking.append((pid, p["title"], break_notes + degrade_notes))
        elif degrade_notes:
            degraded.append((pid, p["title"], degrade_notes))
        else:
            ok_count += 1

    print("\n=== Seniority criteria report ===")
    print(
        f"Target: >= {min_seniors} senior and <= {max_juniors} junior reviewer(s) per paper. "
        f"Fallbacks: almost-senior = typical with >= {almost_senior_window} window papers; "
        f"almost-not-junior = junior with >= {almost_junior_career} career papers."
    )
    print(
        f"{ok_count} paper(s) OK outright, {len(degraded)} degraded but within policy, "
        f"{len(breaking)} BREAKING the criteria."
    )
    if degraded:
        print("\nDegraded:")
        for pid, title, notes in degraded:
            print(f"  [{pid}] {title}")
            for n in notes:
                print(f"      {n}")
    if breaking:
        print("\nBREAKING:")
        for pid, title, notes in breaking:
            print(f"  [{pid}] {title}")
            for n in notes:
                print(f"      {n}")
    return ok_count, len(degraded), len(breaking)


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
    parser.add_argument(
        "--seniority", default=DEFAULT_SENIORITY,
        help="reviewer seniority CSV from classify_reviewers.py (default: %(default)s)"
    )
    parser.add_argument(
        "--no-seniority", action="store_true",
        help="skip the seniority constraints and report; plain single-pass assignment"
    )
    parser.add_argument(
        "--min-seniors", type=int, default=1,
        help="senior reviewers each paper should get (default: %(default)s)"
    )
    parser.add_argument(
        "--max-juniors", type=int, default=1,
        help="max junior reviewers per paper before the almost-not-junior relaxation (default: %(default)s)"
    )
    parser.add_argument(
        "--almost-senior-window", type=int, default=10,
        help="window papers for a typical-class reviewer to count as almost-senior; "
             "assumes classify_reviewers.py defaults, where senior needs 12 (default: %(default)s)"
    )
    parser.add_argument(
        "--almost-junior-career", type=int, default=5,
        help="career papers for a junior to count as almost-not-junior; "
             "assumes classify_reviewers.py defaults, where junior means < 7 (default: %(default)s)"
    )
    parser.add_argument("--device", default="cuda", help="torch device for SPECTER2 (default: cuda)")
    args = parser.parse_args()

    seniority: dict[str, dict] | None = None
    if not args.no_seniority:
        try:
            seniority = load_seniority(args.seniority)
        except FileNotFoundError:
            print(
                f"{args.seniority} not found — run classify_reviewers.py first, "
                f"or pass --no-seniority to assign without the seniority constraints",
                file=sys.stderr,
            )
            return 1

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

    if args.no_seniority:
        paper_held = deferred_acceptance(pids, paper_prefs, paper_target, reviewer_cap, score_lookup)
        blocking = count_blocking_pairs(eligible_by_pid, paper_held, reviewer_cap, paper_target, score_lookup)
        blocking_label = "blocking pairs"
        pools = None
    else:
        pools, missing = seniority_pools(
            set(reviewer_cap), seniority, args.almost_senior_window, args.almost_junior_career
        )
        if missing:
            print(
                f"Warning: {len(missing)} candidate reviewer(s) not in {args.seniority} — "
                f"treated as neither senior nor junior; rerun classify_reviewers.py to refresh it",
                file=sys.stderr,
            )
        slates: dict[int, list[str]] = {pid: [] for pid in pids}
        used: dict[str, int] = defaultdict(int)

        # Phase 1: anchor each paper's best eligible senior(s) — frozen afterwards.
        anchor_target = {pid: min(args.min_seniors, args.reviewers_per_paper) for pid in pids}
        assignment_phase(
            pids, paper_prefs, anchor_target, slates, used, reviewer_cap, score_lookup, pools.seniors
        )
        # Phase 1b: papers that got no senior fall back to an almost-senior.
        fallback_target = {pid: max(0, anchor_target[pid] - len(slates[pid])) for pid in pids}
        assignment_phase(
            pids, paper_prefs, fallback_target, slates, used, reviewer_cap, score_lookup, pools.almost_seniors
        )
        # Phase 2: main fill — everyone competes on score, juniors capped per paper.
        fill_target = {pid: args.reviewers_per_paper - len(slates[pid]) for pid in pids}
        held2, prefs2, cap2 = assignment_phase(
            pids, paper_prefs, fill_target, slates, used, reviewer_cap, score_lookup,
            set(reviewer_cap), junior_emails=pools.juniors, max_juniors=args.max_juniors,
        )
        # Phase 3: still-under-filled papers may take extra almost-not-juniors.
        relax_target = {pid: args.reviewers_per_paper - len(slates[pid]) for pid in pids}
        assignment_phase(
            pids, paper_prefs, relax_target, slates, used, reviewer_cap, score_lookup, pools.almost_not_juniors
        )
        paper_held = slates

        # Self-check the new junior-cap logic where its guarantee holds: the
        # phase-2 pass, in phase-2 terms (its own prefs, caps, and targets).
        pairs2 = {pid: [(e, score_lookup[(e, pid)]) for e in prefs2[pid]] for pid in pids}
        blocking = count_blocking_pairs(
            pairs2, held2, cap2, fill_target, score_lookup,
            junior_emails=pools.juniors, max_juniors=args.max_juniors,
        )
        blocking_label = "phase-2 blocking pairs"

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
            cls = "" if seniority is None else "/" + (seniority[email]["class"] if email in seniority else "?")
            print(f"  {rank:2d}. {score_lookup[(email, pid)]:.3f}  {r.name} <{email}>  [{r.tier}{cls}]  ({r.primary})")
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
    canonical_areas = build_canonical_area_map(reviewers_by_email)
    area_stats = area_pool_stats(candidate_emails, reviewers_by_email, args.light_cap, args.full_cap)
    total_missing = shortage_report(papers, paper_held, paper_target, area_stats, canonical_areas)

    seniority_summary = ""
    if pools is not None:
        ok_n, deg_n, brk_n = seniority_report(
            papers, paper_held, pools, reviewers_by_email, seniority, paper_prefs,
            args.reviewers_per_paper, args.min_seniors, args.max_juniors,
            args.almost_senior_window, args.almost_junior_career,
        )
        deep_over = sum(
            1
            for pid in pids
            if sum(1 for e in paper_held[pid] if e in pools.juniors and e not in pools.almost_not_juniors)
            > args.max_juniors
        )
        over_target = sum(1 for pid in pids if len(paper_held[pid]) > args.reviewers_per_paper)
        seniority_summary = (
            f"seniority: {ok_n} papers OK, {deg_n} degraded, {brk_n} breaking — see report above; "
            f"{deep_over} papers over the junior policy and {over_target} over target — should always be 0; "
        )

    print(
        f"\nDone. {total_pairs} reviewer-paper pairs assigned across {len(papers)} papers, "
        f"{len(reviewer_load)} distinct reviewers used "
        f"(light cap {args.light_cap}, full cap {args.full_cap}; "
        f"{light_over} light and {full_over} full over cap — should always be 0; "
        f"{blocking} {blocking_label} — should always be 0; "
        f"{seniority_summary}"
        f"{total_missing} reviewer-slot(s) unfilled — see shortage report above).",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
