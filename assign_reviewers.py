"""Solve a global, load-capped assignment of reviewers to every paper.

    python assign_reviewers.py
    python assign_reviewers.py --light-cap 7 --full-cap 15 --reviewers-per-paper 6

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
paper_matching.eligible_scores. COI and reviewer capacity are absolute; the
area gate is a soft constraint, released paper-by-paper when a paper can't
otherwise fill its slate or senior slot (see the ladder below).

Seniority constraints (needs reviewer_seniority.csv from classify_reviewers.py;
skip them all with --no-seniority): every paper should get at least
--min-seniors senior reviewers, at most --max-juniors juniors, and at most
--max-out-of-area out-of-area reviewers. Every paper should also end up with
a full slate of --reviewers-per-paper reviewers; when the normal constraints
can't deliver that, they are released in a fixed order — (1) the area gate,
(2) the junior/out-of-area caps (almost-nots only), (3) the senior
requirement — with every relaxed pool still ranked by fingerprint similarity
so match goodness holds up. Assignment runs in phases, each a
deferred-acceptance pass whose results are frozen before the next:

  A1. senior anchors — every paper matches its best eligible in-area senior;
  A2. papers short a senior try area-released true seniors (a close-
      fingerprint senior from another area beats an almost-senior);
  A3. papers still senior-less fall back to an "almost senior" (a
      typical-class reviewer with >= --almost-senior-window window papers),
      any area;
  F1. main fill — everyone with remaining capacity competes on score within
      the area gate, but a paper holds at most --max-juniors juniors and
      --max-out-of-area out-of-area reviewers at a time (pure caps: a
      well-matched junior can still beat a weak-matched typical to a slot);
  F2. under-filled papers fill from the area-released pool, the class caps
      still counting everything held so far;
  F3. papers still under-filled may exceed the caps with "almost not junior"
      juniors (>= --almost-junior-pubs pubs overall) and "almost not
      out-of-area" reviewers (>= --almost-out-of-area-career career papers).

Papers that break the criteria even after degradation are printed in a report
at the end; every paper gets a "match goodness" score — the mean similarity
of its assigned reviewers — summarized worst-first; and a relaxation &
exclusion report itemizes each paper skipped for missing information or
withdrawal and each paper that needed a relaxed constraint, reviewer by
reviewer. Each phase is individually stable, but freezing earlier phases
(the anchors) and the per-class caps mean the final assignment trades
classical global stability for the composition constraints — the self-check
verifies F1's cap-aware stability plus the final composition invariants
instead.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict, deque
from collections.abc import Sequence
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
    capped: Sequence[tuple[frozenset[str], int]] = (),
    held_counts: dict[int, list[int]] | None = None,
) -> dict[int, list[str]]:
    """Paper-proposing Hospital/Residents deferred acceptance.

    `paper_prefs[pid]` must already be sorted by descending score (eligible
    reviewers only). Returns `{pid: [email, ...]}`, the stable assignment.

    `capped` is a list of (emails, per-paper cap) pairs — disjoint reviewer
    classes (e.g. juniors, out-of-area) of which a paper holds at most `cap`
    members at a time. Class members a paper can't currently take are
    deferred, not rejected: they stay proposable in score order and get their
    offer if a class slot later opens (a held member was bumped away). Each
    (paper, reviewer) pair is still proposed at most once, so termination is
    unchanged. `held_counts` seeds each paper's per-class counts with frozen
    assignments from earlier phases, so the caps stay cumulative across
    phases.
    """
    class_of = {email: k for k, (emails, _) in enumerate(capped) for email in emails}
    paper_ptr = {pid: 0 for pid in pids}
    paper_held: dict[int, list[str]] = {pid: [] for pid in pids}
    held_counts = held_counts or {}
    class_held = {pid: list(held_counts.get(pid, [0] * len(capped))) for pid in pids}
    deferred: dict[int, list[deque[str]]] = {pid: [deque() for _ in capped] for pid in pids}
    reviewer_held: dict[str, list[tuple[int, float]]] = {}

    def class_ok(pid: int, k: int) -> bool:
        return class_held[pid][k] < capped[k][1]

    def count_class(pid: int, email: str, delta: int) -> None:
        k = class_of.get(email)
        if k is not None:
            class_held[pid][k] += delta

    def proposable(pid: int) -> bool:
        return paper_ptr[pid] < len(paper_prefs[pid]) or any(
            dq and class_ok(pid, k) for k, dq in enumerate(deferred[pid])
        )

    def next_candidate(pid: int) -> str | None:
        """Best-scoring proposable candidate, honoring the class caps.

        Members of a currently-full class at the head of the pref list are
        moved to that class's deferred deque (preserving score order — the
        pref list is descending, so appends keep each deque sorted). Each
        takeable deferred head then competes with the pref-list head on score.
        """
        prefs = paper_prefs[pid]
        while paper_ptr[pid] < len(prefs):
            k = class_of.get(prefs[paper_ptr[pid]])
            if k is None or class_ok(pid, k):
                break
            deferred[pid][k].append(prefs[paper_ptr[pid]])
            paper_ptr[pid] += 1
        head = prefs[paper_ptr[pid]] if paper_ptr[pid] < len(prefs) else None
        best, best_k = head, None
        for k, dq in enumerate(deferred[pid]):
            if dq and class_ok(pid, k):
                if best is None or score_lookup[(dq[0], pid)] >= score_lookup[(best, pid)]:
                    best, best_k = dq[0], k
        if best_k is not None:
            return deferred[pid][best_k].popleft()
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
            # Nothing proposable right now. If deferred class members remain,
            # the paper is re-queued by the bump that frees its class slot.
            continue

        score = score_lookup[(email, pid)]
        held = reviewer_held.setdefault(email, [])

        bumped_pid = None
        if len(held) < reviewer_cap[email]:
            held.append((pid, score))
            paper_held[pid].append(email)
            count_class(pid, email, +1)
        else:
            worst_i = min(range(len(held)), key=lambda i: held[i][1])
            worst_pid, worst_score = held[worst_i]
            if score > worst_score:
                held[worst_i] = (pid, score)
                paper_held[pid].append(email)
                count_class(pid, email, +1)
                paper_held[worst_pid].remove(email)
                count_class(worst_pid, email, -1)
                bumped_pid = worst_pid
            # else: rejected — pid tries its next candidate on a later turn

        if bumped_pid is not None:
            queue.append(bumped_pid)
        if len(paper_held[pid]) < paper_target[pid] and proposable(pid):
            queue.append(pid)

    return paper_held


def count_blocking_pairs(
    eligible_by_pid: dict[int, list[tuple[str, float]]],
    paper_held: dict[int, list[str]],
    reviewer_cap: dict[str, int],
    paper_target: dict[int, int],
    score_lookup: dict[tuple[str, int], float],
    capped: Sequence[tuple[frozenset[str], int]] = (),
) -> int:
    """Number of (reviewer, paper) pairs that would both prefer each other
    over one of their current matches — should always be 0 for a stable
    assignment; a self-check on `deferred_acceptance`'s guarantee.

    A member of a capped class doesn't block a paper whose class slots are
    full of better-scoring members of the same class: the paper could only
    take them by dropping a class member, so no other slot is up for grabs.
    """
    class_of = {email: k for k, (emails, _) in enumerate(capped) for email in emails}
    reviewer_papers: dict[str, list[int]] = defaultdict(list)
    for pid, emails in paper_held.items():
        for email in emails:
            reviewer_papers[email].append(pid)

    blocking = 0
    for pid, pairs in eligible_by_pid.items():
        held = set(paper_held[pid])
        class_counts = [sum(1 for e in paper_held[pid] if e in emails) for emails, _ in capped]
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
            k = class_of.get(email)
            if k is not None and class_counts[k] >= capped[k][1]:
                class_emails = capped[k][0]
                paper_wants = any(
                    r2 in class_emails and score_lookup[(r2, pid)] < score for r2 in current
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
    almost_not_juniors: frozenset[str]  # junior-class, total_pubs >= threshold
    out_of_area: frozenset[str]
    almost_not_out_of_area: frozenset[str]  # out-of-area-class, career_total >= threshold


def seniority_pools(
    candidate_emails,
    seniority: dict[str, dict],
    almost_senior_window: int,
    almost_junior_pubs: int,
    almost_out_of_area_career: int,
) -> tuple[SeniorityPools, list[str]]:
    """Split the candidate pool by seniority class from reviewer_seniority.csv.

    Candidates classified 'unknown' or missing from the CSV count as neither
    senior, junior, nor out-of-area (they can fill slots but not a senior
    one); the missing ones are also returned so the caller can warn — they
    usually mean the CSV is stale and classify_reviewers.py needs a rerun.
    """
    seniors, almost_seniors, juniors, almost_not = set(), set(), set(), set()
    out_of_area, almost_not_oob = set(), set()
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
            if row["total_pubs"] is not None and row["total_pubs"] >= almost_junior_pubs:
                almost_not.add(email)
        elif cls == "out-of-area":
            out_of_area.add(email)
            if row["career_total"] is not None and row["career_total"] >= almost_out_of_area_career:
                almost_not_oob.add(email)
        elif cls == "typical" and row["window_total"] is not None and row["window_total"] >= almost_senior_window:
            almost_seniors.add(email)
    return (
        SeniorityPools(
            frozenset(seniors), frozenset(almost_seniors),
            frozenset(juniors), frozenset(almost_not),
            frozenset(out_of_area), frozenset(almost_not_oob),
        ),
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
    capped: Sequence[tuple[frozenset[str], int]] = (),
):
    """One accumulating deferred-acceptance pass, restricted to `candidates`.

    `phase_target[pid]` is the number of ADDITIONAL reviewers the paper may
    gain this phase; assignments from earlier phases are frozen — their
    reviewers can't be bumped, which is what makes a phase-1 senior a real
    anchor, but they do keep counting against the per-class caps. Folds the
    result into `slates` and `used`, and returns this phase's (held, prefs,
    cap) view for self-checks.
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
    held_counts = {
        pid: [sum(1 for e in slates[pid] if e in emails) for emails, _ in capped]
        for pid in pids
    }

    held = deferred_acceptance(pids, prefs, phase_target, cap, score_lookup, capped, held_counts)
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
    max_out_of_area: int,
    almost_senior_window: int,
    almost_junior_pubs: int,
    almost_out_of_area_career: int,
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
        oob = [e for e in slate if e in pools.out_of_area]
        deep_oob = [e for e in oob if e not in pools.almost_not_out_of_area]

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
                    f"{pool_size} non-conflicted senior(s) (any area), all at capacity on better-matched papers"
                    if pool_size
                    else "no non-conflicted senior exists for this paper"
                )
                break_notes.append(
                    f"only {len(true_seniors) + len(almost)} of {min_seniors} senior slot(s) filled "
                    f"even counting almost-seniors — {detail}"
                )
        if len(juniors) > max_juniors:
            names = ", ".join(
                f"{reviewers_by_email[e].name} ({seniority[e]['total_pubs']} pubs)"
                for e in sorted(juniors, key=lambda e: seniority[e]["total_pubs"] or 0)
            )
            if len(deep_juniors) <= max_juniors:
                degrade_notes.append(
                    f"{len(juniors)} juniors (cap {max_juniors}): {names} — extras within the almost-not-junior allowance"
                )
            else:
                break_notes.append(
                    f"{len(deep_juniors)} juniors below the almost-not-junior line (cap {max_juniors}): {names}"
                )
        if len(oob) > max_out_of_area:
            names = ", ".join(
                f"{reviewers_by_email[e].name} ({seniority[e]['career_total']} career papers)"
                for e in sorted(oob, key=lambda e: seniority[e]["career_total"] or 0)
            )
            if len(deep_oob) <= max_out_of_area:
                degrade_notes.append(
                    f"{len(oob)} out-of-area (cap {max_out_of_area}): {names} — extras within the almost-not-out-of-area allowance"
                )
            else:
                break_notes.append(
                    f"{len(deep_oob)} out-of-area below the almost-not-out-of-area line (cap {max_out_of_area}): {names}"
                )
        if len(slate) < reviewers_per_paper:
            break_notes.append(
                f"{reviewers_per_paper - len(slate)} slot(s) unfilled even after the almost-not relaxations"
            )

        if break_notes:
            breaking.append((pid, p["title"], break_notes + degrade_notes))
        elif degrade_notes:
            degraded.append((pid, p["title"], degrade_notes))
        else:
            ok_count += 1

    print("\n=== Seniority criteria report ===")
    print(
        f"Target: >= {min_seniors} senior, <= {max_juniors} junior, and "
        f"<= {max_out_of_area} out-of-area reviewer(s) per paper. "
        f"Fallbacks: almost-senior = typical with >= {almost_senior_window} window papers; "
        f"almost-not-junior = junior with >= {almost_junior_pubs} pubs overall; "
        f"almost-not-out-of-area = out-of-area with >= {almost_out_of_area_career} career papers."
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


def paper_goodness(paper_held: dict[int, list[str]], score_lookup: dict[tuple[str, int], float]) -> dict[int, float | None]:
    """Per-paper match goodness: mean similarity of the assigned reviewers.

    None for papers with no reviewers — "no slate" is a shortage-report
    problem, not a goodness of 0.
    """
    return {
        pid: sum(score_lookup[(e, pid)] for e in emails) / len(emails) if emails else None
        for pid, emails in paper_held.items()
    }


def match_goodness_report(papers: list[dict], goodness: dict[int, float | None]) -> None:
    """Print every paper's match goodness worst-first, so the papers whose
    reviewer slates sit furthest from their topic are easy to spot."""
    scored = sorted(
        (p for p in papers if goodness[p["pid"]] is not None),
        key=lambda p: (goodness[p["pid"]], p["pid"]),
    )
    unscored = [p for p in papers if goodness[p["pid"]] is None]

    print("\n=== Match goodness (mean similarity of assigned reviewers, worst first) ===")
    if scored:
        values = [goodness[p["pid"]] for p in scored]
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        print(f"{len(scored)} paper(s): mean {mean:.3f}, std {std:.3f}")
        for p in scored:
            print(f"  {goodness[p['pid']]:.3f}  [{p['pid']}] {p['title']}")
    if unscored:
        print(f"{len(unscored)} paper(s) with no reviewers assigned:")
        for p in unscored:
            print(f"    n/a  [{p['pid']}] {p['title']}")


UNRELAXED_PHASES = frozenset({"senior anchor", "fill"})


def relaxation_report(
    skipped: list[dict],
    papers: list[dict],
    paper_held: dict[int, list[str]],
    paper_target: dict[int, int],
    assigned_via: dict[tuple[int, str], str],
    goodness: dict[int, float | None],
    score_lookup: dict[tuple[str, int], float],
    reviewers_by_email: dict,
    seniority: dict[str, dict] | None,
) -> tuple[int, int]:
    """Itemize papers excluded from assignment and papers that needed relaxed
    constraints (area gate, junior/out-of-area caps, senior requirement) to
    fill their slate or senior slot — the chair's checklist of what to eyeball.
    Returns (excluded, relaxed) paper counts.
    """
    print("\n=== Relaxation & exclusion report ===")
    if skipped:
        print(f"{len(skipped)} paper(s) excluded from assignment:")
        for s in skipped:
            print(f"  [{s['pid']}] {s['title'] or '(no title)'} — {', '.join(s['missing'])}")
    else:
        print("No papers excluded from assignment.")

    relaxed_papers = []
    for p in papers:
        pid = p["pid"]
        entries = []
        for e in paper_held[pid]:
            label = assigned_via.get((pid, e), "fill")
            if label not in UNRELAXED_PHASES:
                entries.append((score_lookup[(e, pid)], e, label))
        missing = paper_target[pid] - len(paper_held[pid])
        if entries or missing > 0:
            relaxed_papers.append((p, sorted(entries, reverse=True), missing))

    if not relaxed_papers:
        print("No papers needed relaxed constraints.")
        return len(skipped), 0
    print(f"\n{len(relaxed_papers)} paper(s) needed relaxed constraints:")
    for p, entries, missing in relaxed_papers:
        pid = p["pid"]
        g = goodness[pid]
        print(f"  [{pid}] {p['title']} — match goodness {'n/a' if g is None else format(g, '.3f')}")
        for score, e, label in entries:
            r = reviewers_by_email[e]
            cls = seniority[e]["class"] if seniority and e in seniority else "?"
            print(f"      {label:28s} {score:.3f}  {r.name} <{e}>  [{cls}]  ({r.primary})")
        if missing > 0:
            print(f"      still {missing} slot(s) unfilled — see shortage report")
    return len(skipped), len(relaxed_papers)


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
        topics = p.get("topics", [])
        if not topics:
            shortfalls["Unspecified/no matching topic"].append((pid, p["title"], missing))
        for topic in topics:
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
    parser.add_argument("--light-cap", type=int, default=7, help="max papers per light PC member (default: 7)")
    parser.add_argument("--full-cap", type=int, default=15, help="max papers per full PC member (default: 15)")
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
        "--max-out-of-area", type=int, default=3,
        help="max out-of-area reviewers per paper before the almost-not-out-of-area relaxation (default: %(default)s)"
    )
    parser.add_argument(
        "--almost-senior-window", type=int, default=10,
        help="window papers for a typical-class reviewer to count as almost-senior; "
             "assumes classify_reviewers.py defaults, where senior needs 12 (default: %(default)s)"
    )
    parser.add_argument(
        "--almost-junior-pubs", type=int, default=15,
        help="overall pubs for a junior to count as almost-not-junior; "
             "assumes classify_reviewers.py defaults, where junior means < 20 (default: %(default)s)"
    )
    parser.add_argument(
        "--almost-out-of-area-career", type=int, default=5,
        help="career target-venue papers for an out-of-area reviewer to count as almost-not-out-of-area; "
             "assumes classify_reviewers.py defaults, where out-of-area means < 5 (default: %(default)s)"
    )
    parser.add_argument("--device", default="cuda", help="torch device for SPECTER2 (default: cuda)")
    args = parser.parse_args()

    if args.reviewers_per_paper < 0:
        parser.error("--reviewers-per-paper must be non-negative")
    if args.light_cap < 0 or args.full_cap < 0:
        parser.error("--light-cap and --full-cap must be non-negative")
    if args.area_weight <= 0:
        parser.error("--area-weight must be greater than 0")
    if args.min_seniors < 0 or args.max_juniors < 0 or args.max_out_of_area < 0:
        parser.error("--min-seniors, --max-juniors, and --max-out-of-area must be non-negative")
    if args.almost_senior_window < 0 or args.almost_junior_pubs < 0 or args.almost_out_of_area_career < 0:
        print("Warning: negative near-threshold values make every applicable reviewer a fallback", file=sys.stderr)
    if args.min_seniors > args.reviewers_per_paper:
        print(
            "Warning: --min-seniors exceeds --reviewers-per-paper; the criteria report will mark papers breaking",
            file=sys.stderr,
        )

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

    papers, skipped_papers = load_papers(args.data, with_skipped=True)
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

    # Gated pairs drive the normal phases; area-released pairs (COI-only)
    # back the relaxation phases, so score_lookup and caps cover the superset.
    eligible_by_pid: dict[int, list[tuple[str, float]]] = {}
    released_by_pid: dict[int, list[tuple[str, float]]] = {}
    score_lookup: dict[tuple[str, int], float] = {}
    reviewer_cap: dict[str, int] = {}
    for p in papers:
        pid = p["pid"]
        paper_vec = np.array(paper_cache[str(pid)]["vector"], dtype=np.float32)
        pairs_all = eligible_scores(
            p, candidate_emails, candidate_matrix, paper_vec, reviewers_by_email,
            area_gate=False,
        )
        released_by_pid[pid] = pairs_all
        eligible_by_pid[pid] = pairs_all if args.no_area_gate else eligible_scores(
            p, candidate_emails, candidate_matrix, paper_vec, reviewers_by_email,
        )
        for email, score in pairs_all:
            score_lookup[(email, pid)] = score
            reviewer_cap[email] = reviewer_paper_cap(reviewers_by_email[email], args.light_cap, args.full_cap)

    pids = [p["pid"] for p in papers]
    paper_target = {pid: args.reviewers_per_paper for pid in pids}
    paper_prefs = {
        pid: [email for email, _ in sorted(eligible_by_pid[pid], key=lambda es: -es[1])] for pid in pids
    }
    released_prefs = {
        pid: [email for email, _ in sorted(released_by_pid[pid], key=lambda es: -es[1])] for pid in pids
    }

    assigned_via: dict[tuple[int, str], str] = {}

    if args.no_seniority:
        slates = deferred_acceptance(pids, paper_prefs, paper_target, reviewer_cap, score_lookup)
        # Judge stability on the gated pass alone — the area-released fill
        # below deliberately steps outside the gated preference lists.
        blocking = count_blocking_pairs(eligible_by_pid, slates, reviewer_cap, paper_target, score_lookup)
        blocking_label = "gated-pass blocking pairs"
        pools = None
        assigned_via = {(pid, e): "fill" for pid, emails in slates.items() for e in emails}
        used = defaultdict(int)
        for emails in slates.values():
            for e in emails:
                used[e] += 1
        relax_target = {pid: paper_target[pid] - len(slates[pid]) for pid in pids}
        held_r, _, _ = assignment_phase(
            pids, released_prefs, relax_target, slates, used, reviewer_cap, score_lookup,
            set(reviewer_cap),
        )
        for pid, emails in held_r.items():
            for e in emails:
                assigned_via[(pid, e)] = "fill (area released)"
        paper_held = slates
    else:
        pools, missing = seniority_pools(
            set(reviewer_cap), seniority, args.almost_senior_window,
            args.almost_junior_pubs, args.almost_out_of_area_career,
        )
        if missing:
            print(
                f"Warning: {len(missing)} candidate reviewer(s) not in {args.seniority} — "
                f"treated as neither senior nor junior; rerun classify_reviewers.py to refresh it",
                file=sys.stderr,
            )
        slates: dict[int, list[str]] = {pid: [] for pid in pids}
        used: dict[str, int] = defaultdict(int)

        def run_phase(label, prefs, target, candidates, capped=()):
            held, phase_prefs, phase_cap = assignment_phase(
                pids, prefs, target, slates, used, reviewer_cap, score_lookup, candidates, capped
            )
            for pid, emails in held.items():
                for e in emails:
                    assigned_via[(pid, e)] = label
            return held, phase_prefs, phase_cap

        # A1: anchor each paper's best eligible in-area senior(s) — frozen afterwards.
        anchor_target = {pid: min(args.min_seniors, args.reviewers_per_paper) for pid in pids}
        run_phase("senior anchor", paper_prefs, anchor_target, pools.seniors)
        # A2: papers short a senior try area-released true seniors (area is
        # released before the senior requirement is relaxed).
        a2_target = {pid: max(0, anchor_target[pid] - len(slates[pid])) for pid in pids}
        run_phase("senior anchor (area released)", released_prefs, a2_target, pools.seniors)
        # A3: papers still senior-less fall back to an almost-senior, any area.
        a3_target = {pid: max(0, anchor_target[pid] - len(slates[pid])) for pid in pids}
        run_phase("almost-senior anchor", released_prefs, a3_target, pools.almost_seniors)
        # F1: main fill — everyone competes on score within the area gate,
        # juniors and out-of-area reviewers each capped per paper.
        capped = [(pools.juniors, args.max_juniors), (pools.out_of_area, args.max_out_of_area)]
        fill_target = {pid: args.reviewers_per_paper - len(slates[pid]) for pid in pids}
        held2, prefs2, cap2 = run_phase("fill", paper_prefs, fill_target, set(reviewer_cap), capped)
        # F2: under-filled papers fill from the area-released pool; the caps
        # keep counting what earlier phases assigned.
        f2_target = {pid: args.reviewers_per_paper - len(slates[pid]) for pid in pids}
        run_phase("fill (area released)", released_prefs, f2_target, set(reviewer_cap), capped)
        # F3: papers still under-filled may exceed the caps with extra
        # almost-not-juniors and almost-not-out-of-area reviewers.
        f3_target = {pid: args.reviewers_per_paper - len(slates[pid]) for pid in pids}
        run_phase(
            "fill (cap relaxed)", released_prefs, f3_target,
            pools.almost_not_juniors | pools.almost_not_out_of_area,
        )
        paper_held = slates

        # Self-check the class-cap logic where its guarantee holds: the F1
        # pass, in F1 terms (its own prefs, caps, and targets).
        pairs2 = {pid: [(e, score_lookup[(e, pid)]) for e in prefs2[pid]] for pid in pids}
        blocking = count_blocking_pairs(pairs2, held2, cap2, fill_target, score_lookup, capped)
        blocking_label = "F1 blocking pairs"

    # --- Report ---------------------------------------------------------------
    goodness = paper_goodness(paper_held, score_lookup)
    reviewer_load: dict[str, int] = defaultdict(int)
    for p in papers:
        pid = p["pid"]
        assigned = sorted(paper_held[pid], key=lambda e: -score_lookup[(e, pid)])
        under_filled = "  *** UNDER-FILLED ***" if len(assigned) < args.reviewers_per_paper else ""
        g = goodness[pid]
        print(f"\n=== [{pid}] {p['title']}")
        print(f"    topics: {', '.join(p.get('topics', []))}")
        print(f"    assigned {len(assigned)} of {args.reviewers_per_paper} requested{under_filled}")
        print(f"    match goodness: {'n/a' if g is None else format(g, '.3f')}")
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
            papers, paper_held, pools, reviewers_by_email, seniority, released_prefs,
            args.reviewers_per_paper, args.min_seniors, args.max_juniors, args.max_out_of_area,
            args.almost_senior_window, args.almost_junior_pubs, args.almost_out_of_area_career,
        )
        deep_junior_over = sum(
            1
            for pid in pids
            if sum(1 for e in paper_held[pid] if e in pools.juniors and e not in pools.almost_not_juniors)
            > args.max_juniors
        )
        deep_oob_over = sum(
            1
            for pid in pids
            if sum(1 for e in paper_held[pid] if e in pools.out_of_area and e not in pools.almost_not_out_of_area)
            > args.max_out_of_area
        )
        over_target = sum(1 for pid in pids if len(paper_held[pid]) > args.reviewers_per_paper)
        seniority_summary = (
            f"seniority: {ok_n} papers OK, {deg_n} degraded, {brk_n} breaking — see report above; "
            f"{deep_junior_over} papers over the junior policy, {deep_oob_over} over the "
            f"out-of-area policy, and {over_target} over target — should always be 0; "
        )

    match_goodness_report(papers, goodness)
    n_excluded, n_relaxed = relaxation_report(
        skipped_papers, papers, paper_held, paper_target, assigned_via,
        goodness, score_lookup, reviewers_by_email, seniority,
    )

    print(
        f"\nDone. {total_pairs} reviewer-paper pairs assigned across {len(papers)} papers, "
        f"{len(reviewer_load)} distinct reviewers used "
        f"(light cap {args.light_cap}, full cap {args.full_cap}; "
        f"{light_over} light and {full_over} full over cap — should always be 0; "
        f"{blocking} {blocking_label} — should always be 0; "
        f"{seniority_summary}"
        f"{n_excluded} papers excluded and {n_relaxed} relaxed — see relaxation report above; "
        f"{total_missing} reviewer-slot(s) unfilled — see shortage report above).",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
