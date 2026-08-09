"""Solve a global, load-capped assignment of reviewers to every paper.

    python -m scripts.assign_reviewers
    python -m scripts.assign_reviewers --light-cap 7 --full-cap 15 --reviewers-per-paper 5
    python -m scripts.assign_reviewers --score-mode random --surplus-per-paper 0
    python -m scripts.assign_reviewers --score-mode random --no-area-gate --surplus-per-paper 0

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

COI here is three layers, not one: what the authors declared (`pc_conflicts`),
who is on the paper themselves (`own_paper_conflicts`), and who has recently
published with one of its authors (`coauthor_coi`, on by default, disable with
--no-coauthor-coi). The third needs nobody to have declared anything, which
matters because the sweep has not reached the reviewers added most recently —
see the two conflict reports the run prints.

Area chairs are excluded from the pool outright (--no-area-chair-exclusion turns
it off): an area chair chairs papers and reviews none. Membership is the union
of HotCRP's `~~area-chairs` tag and the area-chair acceptance form, because each
source has been observed to catch people the other misses.

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

Whatever reviewer capacity those phases leave unspent is then handed to the
papers that need it most (--surplus-per-paper N, default 1; 0 switches the
stage off). Rounds repeat: rank the papers that reached their target by current
match goodness, offer the worst of them one extra reviewer each, take the gated
pool first and the area-released pool second. A paper that cannot use its offer
is dropped so the slot flows on to the next-worst paper. COI, the same-country
cap and the junior/out-of-area caps all still bind -- the F3 relaxations are
deliberately not reused, because a bonus slot is not worth breaking composition
policy for. The stage is purely additive: every phase above is frozen, so
--reviewers-per-paper stays the number the shortage, relaxation and criteria
reports are judged against, and no paper is ever short because a surplus slot
went elsewhere.

The same-country cap (--same-country-cap N, default 2) is the one constraint no
phase releases: a paper whose authors are mostly from country C holds at most N
reviewers affiliated in C, in A1-A3 and F1-F3 alike, and under-fills rather than
exceed it. One rule for every country -- a US paper is capped on US reviewers
exactly as a Chinese paper is capped on Chinese ones, and no country is named in
the policy. C is where the institution is, never anyone's nationality; HK, MO,
TW and SG are separate ISO codes and are never counted as CN. A paper's majority
is taken over the authors whose country could be placed, above a
--region-min-resolved coverage floor; below it the paper is not capped and is
reported. --same-country-cap 0 admits no same-country reviewer at all;
--no-same-country-cap switches the policy off. See README, "Same-country cap",
for the stability caveat this introduces.

Randomized baselines (--score-mode random, --score-seed N) answer "how much of
the match quality is actually the SPECTER2 signal?" The matcher ranks on a
reproducible per-(reviewer, paper) draw instead of the cosine, while every COI
layer, the senior anchor, the junior/out-of-area caps, the same-country cap and
every reviewer load cap still bind exactly as they do in production. Reported
match goodness stays the TRUE SPECTER2 similarity of whatever slate that
produces, so the affinity drop is readable straight off the report. Two arms:
area-blind (--score-mode random --no-area-gate) and area-aware (--score-mode
random alone), which between them separate the declared-area gate's
contribution from the embedding's. Compare arms at --surplus-per-paper 0, and
see `make baselines`. This is a stable matching under random preferences, not a
uniform draw over feasible assignments, and it is not an assignment:
--hotcrp-csv is refused.

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

from reviewer_match.paths import assignment_path, cache_path, curated_path, input_path, report_path

import argparse
import csv
import hashlib
import heapq
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from reviewer_match import affiliation_country
from reviewer_match import coauthor_coi
from reviewer_match import collaborator_coi
from reviewer_match import fingerprint as fp
from scripts.classify_reviewers import DEFAULT_OUT as DEFAULT_SENIORITY, load_seniority
from reviewer_match.paper_matching import (
    PAPER_POLICIES,
    build_paper_fingerprints,
    eligible_scores,
    load_papers,
    parse_exclude_pids,
)
from reviewer_match import pc_membership
from reviewer_match.area_chairs import area_chair_emails, drop_area_chairs
from reviewer_match.roster import DEFAULT_AREA_CHAIR_CSV
from reviewer_match.reserve_reviewers import DEFAULT_DATA as DEFAULT_RESERVE_DATA
from reviewer_match.reserve_reviewers import DEFAULT_INFO as DEFAULT_RESERVE_INFO
from reviewer_match.reserve_reviewers import TIER as RESERVE_TIER
from reviewer_match.reserve_reviewers import load_reserve_reviewers
from reviewer_match.reviewers import DEFAULT_CAP_OVERRIDES, load_cap_overrides, load_reviewers

DEFAULT_CSV = input_path("HPCA'27 PC Member Acceptance Form (Responses) - Form Responses 1.csv")
DEFAULT_DATA = input_path("hpca2027-data.json")
DEFAULT_FINGERPRINT_CACHE = cache_path("fingerprints.json")
DEFAULT_PAPER_CACHE = cache_path("paper_fingerprints.json")

# Reviews a reserve reviewer takes, matching estimate_reserve_need.py's
# --reviews-per-reserve, which is what the cohort was sized against.
DEFAULT_RESERVE_CAP = 4

# Reviewer slots every paper is guaranteed. estimate_reserve_need.py imports
# this, so the cohort is always sized against the number an assignment aims for.
DEFAULT_REVIEWERS_PER_PAPER = 5

# Extra reviewers one paper may gain from capacity the fill phases left unspent.
DEFAULT_SURPLUS_PER_PAPER = 1

# What the matcher RANKS on. `random` is a baseline arm: it keeps every COI
# layer, the senior anchor, the junior/out-of-area caps, the same-country cap
# and every load cap, and drops only the affinity signal -- so the true-SPECTER2
# match goodness the reports still print measures what the constraints alone are
# worth. 2027 matches compare_abstract_rankings.py's seed; nothing depends on
# the value.
SCORE_MODES = ("specter2", "random")
DEFAULT_SCORE_MODE = "specter2"
DEFAULT_SCORE_SEED = 2027

# A per-paper limit that doesn't name a paper doesn't bind it. An int keeps every
# comparison integral -- no float('inf') leaking into counters or reports -- and
# no slate can come close to reaching it.
UNCAPPED = sys.maxsize

# A class limit is either one int binding every paper (the junior and
# out-of-area caps) or a {pid: limit} mapping binding only the papers it names
# (a same-country cap, which applies to that country's majority papers alone).
ClassLimit = int | Mapping[int, int]
CappedClasses = Sequence[tuple[frozenset[str], ClassLimit]]


def resolve_caps(capped: CappedClasses, pids) -> list[dict[int, int]]:
    """Per-class, per-paper limits, resolved once up front.

    Resolving here keeps the inner loop a plain integer compare instead of an
    isinstance check per candidate, and lets each call site pass whichever shape
    reads better.
    """
    resolved = []
    for _, limit in capped:
        if isinstance(limit, int):
            resolved.append({pid: limit for pid in pids})
        else:
            resolved.append({pid: limit.get(pid, UNCAPPED) for pid in pids})
    return resolved


def class_counts_of(
    slates: dict[int, list[str]], pids, capped: CappedClasses
) -> dict[int, list[int]]:
    """Per-paper counts of each capped class already held.

    A reviewer in several classes counts against every one of them — that is
    what makes a junior who is also same-country consume both caps.
    """
    return {
        pid: [sum(1 for e in slates[pid] if e in emails) for emails, _ in capped]
        for pid in pids
    }


def deferred_acceptance(
    pids: list[int],
    paper_prefs: dict[int, list[str]],
    paper_target: dict[int, int],
    reviewer_cap: dict[str, int],
    score_lookup: dict[tuple[str, int], float],
    capped: CappedClasses = (),
    held_counts: dict[int, list[int]] | None = None,
) -> dict[int, list[str]]:
    """Paper-proposing Hospital/Residents deferred acceptance.

    `paper_prefs[pid]` must already be sorted by descending score (eligible
    reviewers only). Returns `{pid: [email, ...]}`.

    `capped` is a list of (emails, limit) pairs — reviewer classes (juniors,
    out-of-area, a country) of which a paper holds at most `limit` members at a
    time. A reviewer may belong to several classes and consumes every one of
    them; the limit is either one int for all papers or a {pid: limit} mapping
    binding only the papers it names. Class members a paper can't currently take
    are deferred, not rejected: they stay proposable in score order and get their
    offer if a class slot later opens (a held member was bumped away). Each
    (paper, reviewer) pair is still proposed at most once, so termination is
    unchanged. `held_counts` seeds each paper's per-class counts with frozen
    assignments from earlier phases, so the caps stay cumulative across phases.

    Deferral queues are keyed by a candidate's *membership signature* — the tuple
    of classes it belongs to — rather than by a single class. Whether a paper can
    take a candidate depends only on that signature and on the paper's current
    counts, so every queue is takeability-homogeneous: a blocked head means a
    genuinely blocked queue. Keying by one class instead would let an unrelated
    full class stall a queue whose own class has room, and the paper would
    silently under-fill.

    The caps are hard: a paper under-fills rather than exceed one, in every
    phase. Stability is a weaker promise — see `count_blocking_pairs`.
    """
    limits = resolve_caps(capped, pids)
    sig_of: dict[str, tuple[int, ...]] = {}
    for k, (emails, _) in enumerate(capped):
        for email in emails:
            sig_of[email] = sig_of.get(email, ()) + (k,)
    cells = sorted(set(sig_of.values()))
    cell_index = {sig: i for i, sig in enumerate(cells)}

    paper_ptr = {pid: 0 for pid in pids}
    paper_held: dict[int, list[str]] = {pid: [] for pid in pids}
    held_counts = held_counts or {}
    class_held = {pid: list(held_counts.get(pid, [0] * len(capped))) for pid in pids}
    # Keyed by cell and created on demand, because the number of cells is set by
    # the whole candidate pool while any one paper only ever defers into the few
    # cells its own caps can block. Under one same-country cap per paper that is
    # 1-3 cells out of the ~60 a full country roster produces, so a list would be
    # ~98% empty deques and would make every scan below proportional to the
    # number of countries in the conference rather than to the work at hand.
    deferred: dict[int, dict[int, deque[str]]] = {pid: {} for pid in pids}
    reviewer_held: dict[str, list[tuple[int, float]]] = {}

    def sig_ok(pid: int, sig: tuple[int, ...]) -> bool:
        return all(class_held[pid][k] < limits[k][pid] for k in sig)

    def cell_ok(pid: int, c: int) -> bool:
        return sig_ok(pid, cells[c])

    def proposable(pid: int) -> bool:
        return paper_ptr[pid] < len(paper_prefs[pid]) or any(
            cell_ok(pid, c) for c in deferred[pid]
        )

    def count_class(pid: int, email: str, delta: int) -> None:
        for k in sig_of.get(email, ()):
            class_held[pid][k] += delta

    def next_candidate(pid: int) -> str | None:
        """Best-scoring proposable candidate, honoring the class caps.

        A pref-list head the paper can't currently take moves to its signature's
        deferred deque (preserving score order — the pref list is descending, so
        appends keep each deque sorted). Each takeable deferred head then
        competes with the pref-list head on score.
        """
        prefs = paper_prefs[pid]
        while paper_ptr[pid] < len(prefs):
            sig = sig_of.get(prefs[paper_ptr[pid]])
            if sig is None or sig_ok(pid, sig):
                break
            cell = deferred[pid].setdefault(cell_index[sig], deque())
            cell.append(prefs[paper_ptr[pid]])
            paper_ptr[pid] += 1
        head = prefs[paper_ptr[pid]] if paper_ptr[pid] < len(prefs) else None
        best, best_c = head, None
        for c, dq in deferred[pid].items():
            if cell_ok(pid, c):
                if best is None or score_lookup[(dq[0], pid)] >= score_lookup[(best, pid)]:
                    best, best_c = dq[0], c
        if best_c is not None:
            dq = deferred[pid][best_c]
            email = dq.popleft()
            if not dq:  # keep the scan proportional to what is actually waiting
                del deferred[pid][best_c]
            return email
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
    capped: CappedClasses = (),
    held_counts: dict[int, list[int]] | None = None,
) -> int:
    """Number of (reviewer, paper) pairs that would both prefer each other
    over one of their current matches — a self-check on `deferred_acceptance`.

    A member of a full class doesn't block a paper whose class slots are full of
    better-scoring members of that same class: the paper could only take them by
    dropping a class member, so no other slot is up for grabs. When a candidate
    is in several full classes, one dropped reviewer has to free all of them, or
    the swap isn't a swap.

    `held_counts` seeds the per-class counts with what earlier phases froze, the
    same way `deferred_acceptance` does. Without it a paper that already holds a
    class member from an anchor phase looks like it has room, and the check
    reports blocking pairs the matcher correctly refused.

    **Zero is guaranteed only when the classes form a laminar family** — pairwise
    disjoint or nested, as `{juniors, out-of-area}` alone are. A country class
    crosses the seniority classes, and greedy-by-score choice over a crossing
    family is not substitutable, so paper-proposing deferred acceptance no longer
    guarantees a stable outcome for the papers that carry one. Callers should
    judge capped papers separately from the rest; the caps themselves stay hard
    either way.
    """
    pids = list(eligible_by_pid)
    limits = resolve_caps(capped, pids)
    sig_of: dict[str, tuple[int, ...]] = {}
    for k, (emails, _) in enumerate(capped):
        for email in emails:
            sig_of[email] = sig_of.get(email, ()) + (k,)
    held_counts = held_counts or {}
    reviewer_papers: dict[str, list[int]] = defaultdict(list)
    for pid, emails in paper_held.items():
        for email in emails:
            reviewer_papers[email].append(pid)

    blocking = 0
    for pid, pairs in eligible_by_pid.items():
        held = set(paper_held[pid])
        seed = held_counts.get(pid) or [0] * len(capped)
        class_counts = [
            seed[k] + sum(1 for e in paper_held[pid] if e in emails)
            for k, (emails, _) in enumerate(capped)
        ]
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
            full = [k for k in sig_of.get(email, ()) if class_counts[k] >= limits[k][pid]]
            if full:
                paper_wants = any(
                    score_lookup[(r2, pid)] < score
                    and all(r2 in capped[k][0] for k in full)
                    for r2 in current
                )
            else:
                paper_wants = len(current) < paper_target[pid] or any(
                    score_lookup[(r2, pid)] < score for r2 in current
                )
            if paper_wants:
                blocking += 1
    return blocking


def reviewer_paper_cap(
    r, light_cap: int, full_cap: int, reserve_cap: int = DEFAULT_RESERVE_CAP
) -> int:
    """Per-reviewer paper cap: the CSV's override, if set, else the tier default.

    Reserve reviewers agreed to a handful of reviews, not a PC member's load, so
    their tier is named explicitly. Without that they would fall through to
    `full_cap` the moment they entered the pool — a silent fifteen-paper
    assignment for someone who signed up for four.
    """
    if r.override_cap is not None:
        return r.override_cap
    if r.tier == "light":
        return light_cap
    if r.tier == "reserve":
        return reserve_cap
    return full_cap


def random_pair_score(seed: int, email: str, pid: int) -> float:
    """A reproducible pseudo-random score in [0, 1) for one (reviewer, paper).

    A pure function of its arguments, never a draw from a stream, for two
    reasons. `build_pair_scores` scores each paper twice -- once gated, once
    area-released -- and both lists must carry the same number for the same
    pair, or a preference list disagrees with the score deferred acceptance
    bumps on and the deferral deques stop being sorted. And a rerun has to be
    identical, the contract every cache write in this repo keeps.

    blake2b, not the builtin `hash()`: Python salts string hashing per process,
    the trap that once made the DBLP snapshot files reorder on every run and
    left `cmp` unable to tell a real change from none.
    """
    digest = hashlib.blake2b(
        "\x00".join((str(seed), email, str(pid))).encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") / 2 ** 64


def split_promoted_reserves(reserves, reviewers_by_email) -> tuple[list, list[str], list]:
    """Split the reserve roster into (promoted, already on the PC roster, reserves).

    A reserve HotCRP tags `~~ex-rr` comes off this roster carrying a `light` or
    `full` tier, because they have been elevated to the committee. They are PC
    members and belong in the pool whether or not reserves are being assigned
    from -- excluding the reserve bench must not quietly exclude them too.

    One of them who has since also returned the acceptance form is on both
    rosters, and the form record wins: it carries the areas they declared
    themselves and a fingerprint built from them, where the reserve record's
    areas are inferred from the topics of papers they happened to author.
    Comparison is on `hotcrp_email`, never the roster address -- accepting under
    one address while holding the account under another is ordinary here, and
    that is the whole reason `pc_membership` exists. Nobody is in both today.
    """
    pc_accounts = {r.hotcrp_email.lower() for r in reviewers_by_email.values()}
    promoted, already_pc, true_reserves = [], [], []
    for r in reserves:
        if r.tier == RESERVE_TIER:
            true_reserves.append(r)
        elif r.hotcrp_email.lower() in pc_accounts:
            already_pc.append(r.email)
        else:
            promoted.append(r)
    return promoted, already_pc, true_reserves


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
    capped: CappedClasses = (),
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
    held_counts = class_counts_of(slates, pids, capped)

    held = deferred_acceptance(pids, prefs, phase_target, cap, score_lookup, capped, held_counts)
    for pid, emails in held.items():
        slates[pid].extend(emails)
        for e in emails:
            used[e] += 1
    return held, prefs, cap


def spare_capacity(reviewer_cap: dict[str, int], used: dict[str, int]) -> int:
    """Reviewer-slots the whole candidate pool has still not spent."""
    # .get, not [], so this stays callable with a plain dict for `used`.
    return sum(reviewer_cap[e] - used.get(e, 0) for e in reviewer_cap)


def distribute_surplus(
    pids: list[int],
    paper_prefs: dict[int, list[str]],
    released_prefs: dict[int, list[str]],
    slates: dict[int, list[str]],
    used: dict[str, int],
    reviewer_cap: dict[str, int],
    score_lookup: dict[tuple[str, int], float],
    assigned_via: dict[tuple[int, str], str],
    capped: CappedClasses = (),
    *,
    base_target: dict[int, int],
    surplus_per_paper: int,
) -> tuple[dict[int, list[str]], int]:
    """Spend leftover reviewer capacity on the worst-matched papers.

    Returns ({pid: [added email]}, rounds run). Like every phase above it this
    accumulates into `slates`, `used` and `assigned_via`, so it is purely
    additive: earlier assignments are frozen and cannot be bumped, and the
    class caps keep counting what those phases already placed.

    Only papers that REACHED their base target are offered a slot. One that is
    still short failed the fill phases minutes ago under these same rules and a
    strictly larger pool, so an offer could not place anything — and skipping
    them is what makes "no paper is short because a surplus slot went
    elsewhere" true by construction rather than by convention.

    Each round re-ranks by current match goodness and offers the worst `spare`
    papers one reviewer each, gated pool first and area-released second. A paper
    that places nothing is dropped for good: it has exhausted its preference
    list, and since capacity only shrinks while its own slate stands still, it
    would fail identically forever. Dropping it is precisely what lets the slot
    flow on to the next-worst paper. So a round that places nothing is still
    productive, and "stop when a round places nothing" would abandon capacity in
    exactly the case this stage exists for. Termination does not need it: every
    round either drops a paper or spends a slot, both finite and monotone.
    """
    added: dict[int, list[str]] = {}
    rounds = 0
    if surplus_per_paper <= 0:
        return added, rounds

    candidates = set(reviewer_cap)
    eligible = {
        pid for pid in pids
        if base_target[pid] > 0 and len(slates[pid]) >= base_target[pid]
    }
    while eligible:
        spare = spare_capacity(reviewer_cap, used)
        if spare <= 0:
            break
        goodness = paper_goodness(slates, score_lookup)
        # Worst-matched first, pid breaking ties — match_goodness_report's order.
        ranked = sorted(
            (pid for pid in eligible if goodness[pid] is not None),
            key=lambda pid: (goodness[pid], pid),
        )
        chosen = ranked[:spare]
        if not chosen:
            break
        rounds += 1
        # `chosen`, not every pid: assignment_phase uses its pid list only to
        # build prefs and seed the class counts, and a paper it never lists
        # could not have been touched anyway. That keeps a round proportional
        # to the papers actually being offered a slot.
        held, _, _ = assignment_phase(
            chosen, paper_prefs, {pid: 1 for pid in chosen}, slates, used,
            reviewer_cap, score_lookup, candidates, capped,
        )
        still = [pid for pid in chosen if not held[pid]]
        held_r: dict[int, list[str]] = {}
        if still:
            held_r, _, _ = assignment_phase(
                still, released_prefs, {pid: 1 for pid in still}, slates, used,
                reviewer_cap, score_lookup, candidates, capped,
            )
        for label, batch in (("surplus", held), ("surplus (area released)", held_r)):
            for pid, emails in batch.items():
                for e in emails:
                    assigned_via[(pid, e)] = label
                    added.setdefault(pid, []).append(e)
        for pid in chosen:
            if not held[pid] and not held_r.get(pid):
                eligible.discard(pid)
            elif len(slates[pid]) >= base_target[pid] + surplus_per_paper:
                eligible.discard(pid)
    return added, rounds


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

    Means whatever score the caller passes: the matcher's ranking score inside
    distribute_surplus, which is how a randomized arm stays blind to affinity
    even when choosing who to boost, and the true SPECTER2 affinity in every
    report. The same object unless --score-mode split them.

    None for papers with no reviewers — "no slate" is a shortage-report
    problem, not a goodness of 0.
    """
    return {
        pid: sum(score_lookup[(e, pid)] for e in emails) / len(emails) if emails else None
        for pid, emails in paper_held.items()
    }


def match_goodness_report(
    papers: list[dict],
    goodness: dict[int, float | None],
    *,
    floor: dict[int, float] | None = None,
    ceiling: dict[int, float] | None = None,
    ceiling_k: int | None = None,
) -> None:
    """Print every paper's match goodness worst-first, so the papers whose
    reviewer slates sit furthest from their topic are easy to spot.

    `floor` and `ceiling` bracket the mean so it can be read as a fraction of
    the range actually available, which is the only way to tell a 0.08 gap that
    is most of the headroom from one that is a sliver of it. Both are averaged
    over the same papers as the mean, so all three move together.
    """
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
        if floor is not None and ceiling is not None:
            lo = [floor[p["pid"]] for p in scored if p["pid"] in floor]
            hi = [ceiling[p["pid"]] for p in scored if p["pid"] in ceiling]
            if lo and hi:
                print(
                    f"Bracket: pool floor {sum(lo) / len(lo):.3f} over {len(lo)} paper(s) "
                    f"— one reviewer drawn at random from what this run's own COI and "
                    f"area gate left; best-{ceiling_k} ceiling {sum(hi) / len(hi):.3f} "
                    f"over {len(hi)} paper(s) — each paper's {ceiling_k} closest eligible "
                    f"reviewers ignoring every load and composition cap. The ceiling is a "
                    f"per-paper maximum and is NOT jointly achievable: it would put the "
                    f"same popular reviewers on dozens of papers each. It brackets, it "
                    f"does not target."
                )
        for p in scored:
            print(f"  {goodness[p['pid']]:.3f}  [{p['pid']}] {p['title']}")
    if unscored:
        print(f"{len(unscored)} paper(s) with no reviewers assigned:")
        for p in unscored:
            print(f"    n/a  [{p['pid']}] {p['title']}")


UNRELAXED_PHASES = frozenset({"senior anchor", "fill"})

# Surplus picks get their own report, so the relaxation report stays a list of
# papers that struggled. "surplus (area released)" really is an area release,
# but on a slot the paper was never owed — reporting it as a relaxation would
# imply a shortfall that does not exist.
SURPLUS_PHASES = frozenset({"surplus", "surplus (area released)"})


def surplus_report(
    papers: list[dict],
    added: dict[int, list[str]],
    base_goodness: dict[int, float | None],
    goodness: dict[int, float | None],
    score_lookup: dict[tuple[str, int], float],
    reviewers_by_email: dict,
    seniority: dict[str, dict] | None,
    assigned_via: dict[tuple[int, str], str],
    *,
    reviewers_per_paper: int,
    surplus_per_paper: int,
    spare_before: int,
    spare_after: int,
    rounds: int,
) -> int:
    """Itemize the papers that gained a reviewer from leftover capacity.

    Prints goodness over the base slate beside goodness over the full slate,
    because they move in opposite directions to the intuition: goodness is a
    MEAN, and a surplus reviewer almost always scores below the slate that
    outbid it, so a paper that gained a review shows a LOWER full-slate figure.
    The base figure is the one comparable against a run with the stage off.
    Returns the number of reviewer-slots placed.
    """
    placed = sum(len(v) for v in added.values())
    print("\n=== Surplus distribution report ===")
    if surplus_per_paper <= 0:
        print(
            f"Stage off (--surplus-per-paper 0). {spare_before} reviewer-slot(s) "
            f"left unspent beyond the {reviewers_per_paper}-reviewer target."
        )
        return placed
    print(
        f"Target: up to {surplus_per_paper} reviewer(s) beyond the "
        f"{reviewers_per_paper}-slot slate, offered worst-matched paper first in "
        f"re-ranked rounds, under the same COI, same-country and junior/out-of-area "
        f"rules as the main fill (the almost-not pools are not used). The base slate "
        f"stays the contract: no paper is short, relaxed, or over target because of "
        f"a surplus slot."
    )
    print(
        f"Leftover capacity: {spare_before} slot(s) before, {spare_after} after. "
        f"{placed} reviewer(s) placed on {len(added)} paper(s) over {rounds} round(s)."
    )
    if not placed:
        return placed

    by_pid = {p["pid"]: p for p in papers}
    for pid in sorted(added, key=lambda pid: (base_goodness[pid], pid)):
        p = by_pid[pid]
        before, after = base_goodness[pid], goodness[pid]
        print(f"  {before:.3f} -> {after:.3f}  [{pid}] {p['title']}")
        for e in added[pid]:
            r = reviewers_by_email[e]
            cls = seniority[e]["class"] if seniority and e in seniority else "?"
            label = assigned_via.get((pid, e), "surplus")
            print(f"      {label:28s} {score_lookup[(e, pid)]:.3f}  {r.name} <{e}>  [{cls}]  ({r.primary})")

    befores = [base_goodness[pid] for pid in added]
    afters = [goodness[pid] for pid in added]
    print(
        f"Mean over the {len(added)} boosted paper(s): "
        f"{sum(befores) / len(befores):.3f} base slate -> "
        f"{sum(afters) / len(afters):.3f} with the surplus reviewer(s)."
    )
    return placed


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
    *,
    itemize_excluded: bool = True,
) -> tuple[int, int]:
    """Itemize papers excluded from assignment and papers that needed relaxed
    constraints (area gate, junior/out-of-area caps, senior requirement) to
    fill their slate or senior slot — the chair's checklist of what to eyeball.
    Returns (excluded, relaxed) paper counts.
    """
    print("\n=== Relaxation & exclusion report ===")
    if skipped:
        print(f"{len(skipped)} paper(s) excluded from assignment:")
        if itemize_excluded:
            for s in skipped:
                print(f"  [{s['pid']}] {s['title'] or '(no title)'} — {', '.join(s['missing'])}")
        else:
            reasons = Counter(reason for s in skipped for reason in s["missing"])
            for reason, count in sorted(reasons.items()):
                print(f"  {count} — {reason}")
    else:
        print("No papers excluded from assignment.")

    relaxed_papers = []
    for p in papers:
        pid = p["pid"]
        entries = []
        for e in paper_held[pid]:
            label = assigned_via.get((pid, e), "fill")
            if label not in UNRELAXED_PHASES and label not in SURPLUS_PHASES:
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


DEFAULT_SAME_COUNTRY_CAP = 2
DEFAULT_REGION_MAJORITY = 0.5
DEFAULT_REGION_MIN_RESOLVED = 0.5


@dataclass(frozen=True)
class CountryCap:
    """The same-country cap as it applies to one country, for this run's data."""

    code: str  # ISO 3166-1 alpha-2 — where the institution is, not a nationality
    cap: int
    members: frozenset[str]  # candidate reviewers affiliated in `code`
    papers: dict[int, int]  # pid -> cap, majority-`code` papers only
    shares: dict[int, tuple[int, int, int]]  # pid -> (in country, placed, authors)


def build_country_caps(
    papers: list[dict],
    candidate_emails: list[str],
    reviewers_by_email: dict,
    cap: int,
    layers,
    *,
    majority: float,
    min_resolved: float,
) -> tuple[list[CountryCap], dict[str, str], dict[int, tuple[int, int]], list[int]]:
    """Build one capped class per country that is some paper's majority.

    The rule is the same for everyone: a paper whose authors are mostly from
    country C holds at most `cap` reviewers affiliated in C. Nothing singles out
    a country — the set of capped countries is whatever the submissions turn out
    to contain, so a US paper is capped on US reviewers exactly as a Chinese one
    is capped on Chinese reviewers.

    A paper is majority-C when more than `majority` of the authors whose country
    could be *placed* are in C. The denominator is the placed authors, not all of
    them: an unplaced author is a gap in our data, and counting them against the
    country would make thin coverage silently look like a non-majority
    everywhere. To stop that reading the other way — one placed author out of ten
    reading as 100% — a paper below `min_resolved` coverage is not capped at all
    and is reported by name instead.

    Countries are disjoint as reviewer sets (a reviewer resolves to one country)
    and a paper has at most one majority, so these classes never cross each
    other; they cross only the seniority classes.

    Returns (caps, {reviewer email: code or ""}, {pid: (placed, authors)},
    [pid too thin to judge]).
    """
    reviewer_country = {}
    for email in candidate_emails:
        code, _ = affiliation_country.reviewer_country(reviewers_by_email[email], layers)
        reviewer_country[email] = code

    coverage: dict[int, tuple[int, int]] = {}
    majority_of: dict[int, str] = {}
    counts: dict[int, Counter] = {}
    thin: list[int] = []
    for p in papers:
        pid = p["pid"]
        authors = p.get("authors") or []
        seen = Counter()
        for author in authors:
            code, _ = affiliation_country.author_country(author, layers)
            if code:
                seen[code] += 1
        placed = sum(seen.values())
        coverage[pid] = (placed, len(authors))
        counts[pid] = seen
        if not authors:
            continue
        if placed / len(authors) < min_resolved:
            thin.append(pid)
            continue
        if placed:
            code, n = seen.most_common(1)[0]
            if n / placed > majority:
                majority_of[pid] = code

    caps = []
    for code in sorted(set(majority_of.values())):
        members = frozenset(e for e, c in reviewer_country.items() if c == code)
        bound = {pid: cap for pid, c in majority_of.items() if c == code}
        shares = {
            pid: (counts[pid][code], coverage[pid][0], coverage[pid][1]) for pid in bound
        }
        caps.append(CountryCap(code, cap, members, bound, shares))
    return caps, reviewer_country, coverage, thin


def country_cap_report(
    papers: list[dict],
    slates: dict[int, list[str]],
    countries: list[CountryCap],
    reviewer_country: dict[str, str],
    paper_coverage: dict[int, tuple[int, int]],
    thin: list[int],
    released_prefs: dict[int, list[str]],
    paper_target: dict[int, int],
    score_lookup: dict[tuple[str, int], float],
    has_capacity: set[str],
    *,
    cap: int,
    majority: float,
    min_resolved: float,
) -> int:
    """Print how the same-country cap bound, and how well the data supported it.

    Returns the number of papers over the cap, which must always be 0.

    The coverage lines are the point of this report as much as the caps are, and
    more so than under a single named country: a reviewer whose country could not
    be placed is in no class and can never consume a cap, and a paper whose
    authors could not be placed is never judged. Uneven coverage therefore does
    not weaken the rule evenly -- it exempts whoever we happen to be worse at
    placing -- so the numbers have to be read, not assumed.
    """
    by_title = {p["pid"]: p["title"] for p in papers}
    placed_reviewers = sum(1 for c in reviewer_country.values() if c)
    total_reviewers = len(reviewer_country)
    judged = len(papers) - len(thin)
    capped_papers = sum(len(c.papers) for c in countries)

    print("\n=== Same-country cap report ===")
    print(
        f"Target: at most {cap} reviewer(s) from a paper's own majority-author "
        f"country, where a paper counts as majority-C when more than "
        f"{majority:.0%} of its placed-country authors are in C. The same rule "
        f"applies to every country; none is named in the policy."
    )
    print(
        "Country is where the institution is, not anyone's nationality; HK, MO, "
        "TW and SG are separate ISO codes and are never counted as CN."
    )
    print(
        f"Reviewer coverage: {placed_reviewers} of {total_reviewers} candidates "
        f"placed ({100 * placed_reviewers / total_reviewers:.1f}%); "
        f"{total_reviewers - placed_reviewers} cannot count against any cap."
    )
    print(
        f"Paper coverage: {judged} of {len(papers)} papers had at least "
        f"{min_resolved:.0%} of their authors placed and were judged; "
        f"{len(papers) - judged} were not."
    )

    rows, over_total, short_total, traded_total = [], 0, 0, 0
    for c in countries:
        counts = {pid: sum(1 for e in slates[pid] if e in c.members) for pid in c.papers}
        at_cap = [pid for pid, n in counts.items() if n == c.cap]
        over = [pid for pid, n in counts.items() if n > c.cap]

        # Two different costs, worth separating. A paper is SHORT when the cap is
        # why a slot is empty -- the serious case, and the one that turns up in
        # the shortage report. A paper TRADED when it filled its slate but a
        # better-matched reviewer from its own country was passed over for a
        # worse-matched one; that is the match quality the policy spends, and on
        # a country with many reviewers it is the usual outcome, not a fault.
        #
        # SHORT is an UPPER BOUND on the cap's fill cost, not a measurement of
        # it. `has_capacity` removes the clearest false positives -- a
        # same-country reviewer who ended the run at their own paper cap proves
        # nothing, since the slot would have been empty either way -- but a paper
        # can still land here having run out of phases rather than out of
        # reviewers: F3 only draws from the almost-not pools, so a free reviewer
        # outside them was never takeable. The true cost is a counterfactual and
        # needs the two-run diff against --no-same-country-cap.
        def spare(pid, members=c.members):
            return [e for e in released_prefs[pid]
                    if e in members and e not in slates[pid] and e in has_capacity]

        def eligible_spare(pid, members=c.members):
            return [e for e in released_prefs[pid] if e in members and e not in slates[pid]]

        short = [pid for pid in at_cap
                 if len(slates[pid]) < paper_target[pid] and spare(pid)]
        traded = []
        for pid in at_cap:
            if pid in short or not slates[pid]:
                continue
            worst = min(score_lookup[(e, pid)] for e in slates[pid])
            if any(score_lookup[(e, pid)] > worst for e in eligible_spare(pid)):
                traded.append(pid)
        over_total += len(over)
        short_total += len(short)
        traded_total += len(traded)
        rows.append((c, counts, at_cap, over, short, traded))

    print(
        f"{capped_papers} paper(s) capped across {len(countries)} countries; "
        f"{over_total} over cap; at most {short_total} left short by it "
        f"(upper bound — diff a --no-same-country-cap run for the true cost); "
        f"{traded_total} that traded a better-matched same-country reviewer."
    )

    rows.sort(key=lambda r: (-len(r[0].papers), r[0].code))
    for c, counts, at_cap, over, short, traded in rows:
        print(
            f"\n{c.code}: {len(c.members)} reviewer(s) affiliated there; "
            f"{len(c.papers)} paper(s) majority-{c.code}, {len(at_cap)} at the cap, "
            f"{len(short)} short, {len(traded)} traded."
        )
        if over:
            print(f"  OVER THE CAP — should never happen: {len(over)} paper(s)")
            for pid in sorted(over):
                print(f"    [{pid}] {by_title.get(pid, '')} — {counts[pid]} of cap {c.cap}")
        if short:
            print(f"  Possibly short because of the cap — under target with a "
                  f"free {c.code} reviewer refused:")
            for pid in sorted(short):
                here, placed, _ = c.shares[pid]
                print(f"    [{pid}] {by_title.get(pid, '')} — "
                      f"{len(slates[pid])} of {paper_target[pid]} reviewer(s); "
                      f"{here}/{placed} placed authors in {c.code}")

    if thin:
        print(f"\nNot judged, too few authors placed ({len(thin)} paper(s)):")
        for pid in sorted(thin)[:20]:
            placed, total = paper_coverage[pid]
            print(f"    [{pid}] {by_title.get(pid, '')} — {placed} of {total} author(s) placed")
        if len(thin) > 20:
            print(f"    ... and {len(thin) - 20} more")

    if placed_reviewers < total_reviewers:
        print(
            f"\nWARNING: {total_reviewers - placed_reviewers} of {total_reviewers} "
            f"candidate reviewers have no placed country, so no cap can count them "
            f"and the policy under-applies — unevenly, in favour of whichever "
            f"countries we are worse at placing. Run build_affiliation_countries.py "
            f"and fill the blank country cells in affiliation_countries.csv."
        )
    if thin:
        print(
            f"WARNING: {len(thin)} of {len(papers)} papers had fewer than "
            f"{min_resolved:.0%} of their authors placed and were NOT capped. Raise "
            f"coverage, or lower --region-min-resolved deliberately."
        )
    return over_total


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


def report_conflict_coverage(papers: list[dict], reviewers_by_email: dict) -> None:
    """Print how thoroughly conflicts are declared, per reviewer tier.

    A reviewer nobody has declared a conflict against looks universally
    available, and the matcher will use them accordingly. Authors have not yet
    been asked to declare conflicts against recently promoted PC and reserve
    reviewers, so coverage is uneven — and an assignment that leans on the
    under-covered tier is more feasible on paper than in life. Printing the
    asymmetry is what stops the result being read as a clean bill of health.
    """
    by_tier: dict[str, list[int]] = defaultdict(list)
    emails_by_tier: dict[str, set[str]] = defaultdict(set)
    for email, r in reviewers_by_email.items():
        emails_by_tier[r.tier].add(email)

    zero_cover: dict[str, int] = defaultdict(int)
    for paper in papers:
        declared = {e.lower() for e in (paper.get("pc_conflicts") or {})}
        for tier, emails in emails_by_tier.items():
            n = len(declared & emails)
            by_tier[tier].append(n)
            if n == 0:
                zero_cover[tier] += 1

    print("\n=== Declared-conflict coverage ===")
    print(f"{'tier':<9} {'reviewers':>9} {'mean/paper':>11} {'median':>7} {'papers with none':>18}")
    for tier in sorted(by_tier):
        counts = sorted(by_tier[tier])
        mean = sum(counts) / len(counts) if counts else 0.0
        median = counts[len(counts) // 2] if counts else 0
        share = 100 * zero_cover[tier] / len(papers) if papers else 0
        print(f"{tier:<9} {len(emails_by_tier[tier]):>9} {mean:>11.1f} {median:>7} "
              f"{zero_cover[tier]:>10} ({share:.0f}%)")
    print(
        "Uneven coverage means the thinner tier looks more available than it is; "
        "conflicts against recently added reviewers have not been collected yet.",
        file=sys.stderr,
    )


def report_coauthor_coi(
    papers: list[dict], derived: dict, index, reviewers_by_email: dict, years: int
) -> None:
    """Print what the derived co-author layer excluded, and what it could not see.

    Read alongside the declared-conflict coverage above: this layer is heaviest
    exactly where that one is thinnest, because a reviewer nobody was asked
    about is still in DBLP. The count of newly-found conflicts is the size of
    the gap the sweep left; `scripts.audit_coauthor_conflicts` itemises them.
    """
    total = new = 0
    by_tier: dict[str, int] = defaultdict(int)
    for found in derived.values():
        for email, coi in found.items():
            total += 1
            if coi.declared:
                continue
            new += 1
            reviewer = reviewers_by_email.get(email)
            if reviewer is not None:
                by_tier[reviewer.tier] += 1

    checked = len(index.covered)
    print(f"\n=== Derived co-author COI (last {years} years) ===")
    print(f"Reviewer-paper pairs excluded: {total} over {len(derived)} of "
          f"{len(papers)} paper(s)")
    print(f"  already declared: {total - new}   not declared anywhere: {new}")
    for tier, count in sorted(by_tier.items(), key=lambda kv: -kv[1]):
        print(f"    {count:>6}  {tier}")
    # Always stated, never only on bad news: a reviewer with no co-author data
    # passes this layer silently, so silence must not be what says everyone was
    # checked.
    print(f"Reviewers checked: {checked} of {checked + len(index.uncovered)} "
          f"({len(index.uncovered)} have no co-author data and pass silently)")
    print(
        "Derived from DBLP, not declared by anyone; name matching errs towards "
        "excluding, so expect some false positives. Run `make coauthor-coi` for "
        "the itemised report.", file=sys.stderr,
    )


def report_collaborator_coi(papers: list[dict], derived: dict, reviewers_by_email: dict) -> None:
    """Print what the declared-collaborator layer found, and how much of it excluded.

    Only the `name` kind is excluded (see collaborator_coi.hard_conflicts);
    `affiliation` is reported here but not applied, so the counts below
    separate "excluded" from "found but left as informational."

    Unlike the co-author layer this one has no coverage gap to report: every
    HotCRP account, reviewer or author, is either in the export or it is not,
    and `collaborator_coi.build_index`/`derive_conflicts` read directly from
    it rather than a separately-built cache that could be stale or missing a
    PID.
    """
    total = new = excluded = 0
    by_kind: dict[str, int] = defaultdict(int)
    for found in derived.values():
        for email, coi in found.items():
            total += 1
            by_kind[coi.kind] += 1
            if coi.kind == "name":
                excluded += 1
            if not coi.declared:
                new += 1
    print(f"\n=== Derived declared-collaborator COI ===")
    print(f"Reviewer-paper pairs found: {total} over {len(derived)} of "
          f"{len(papers)} paper(s); {excluded} excluded (name match), "
          f"{total - excluded} reported only (affiliation overlap)")
    print(f"  already declared: {total - new}   not declared anywhere: {new}")
    for kind, count in sorted(by_kind.items(), key=lambda kv: -kv[1]):
        print(f"    {count:>6}  {kind}")
    print(
        "Name matches are excluded like a declared conflict; affiliation "
        "overlap is HotCRP's own 'may conflict' signal too, but too coarse "
        "to hard-block on -- see reviewer_match.collaborator_coi. Run "
        "`make collaborator-coi` for the itemised report.", file=sys.stderr,
    )


def area_pool_stats(
    candidate_emails: list[str], reviewers_by_email: dict, light_cap: int, full_cap: int,
    reserve_cap: int = DEFAULT_RESERVE_CAP,
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
        cap = reviewer_paper_cap(r, light_cap, full_cap, reserve_cap)
        for area in {r.primary, r.secondary} - {""}:
            s = stats.setdefault(
                area, {"reviewers": 0, "light": 0, "full": 0, "reserve": 0, "capacity": 0}
            )
            s["reviewers"] += 1
            s[r.tier] = s.get(r.tier, 0) + 1
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


def write_hotcrp_csv(
    path: str,
    slates: Mapping[int, Sequence[str]],
    score_lookup: Mapping[tuple[str, int], float],
    reviewers_by_email: Mapping[str, object],
) -> None:
    """Atomically write the final slates as a replacing HotCRP R1 upload.

    Writes each reviewer's `hotcrp_email`, not the roster key `email`: for
    someone pc_membership matched under a different address than the one on
    their acceptance-form row, `email` is what they typed, but `hotcrp_email`
    is the address HotCRP actually has marked `pc` -- the only one a bulk
    upload will accept.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temporary = Path(f.name)
            writer = csv.writer(f)
            writer.writerow(["paper", "action", "email", "round"])
            writer.writerow(["all", "clearreview", "all", "R1"])
            for pid in sorted(slates):
                emails = sorted(
                    slates[pid],
                    key=lambda email: (-score_lookup[(email, pid)], email),
                )
                for email in emails:
                    hotcrp_email = reviewers_by_email[email].hotcrp_email
                    writer.writerow([pid, "primaryreview", hotcrp_email, "R1"])
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class PairScores:
    """The one place the matcher's ranking score can differ from true affinity.

    `rank` is what every phase proposes, bumps, defers and self-checks on;
    `affinity` is the SPECTER2 cosine every report prints. Under the default
    --score-mode specter2 they are the SAME OBJECT, so the production path
    cannot drift from the one this repo has always run -- two dicts holding
    equal floats could, the first time someone edits one branch.

    `eligible` and `released` are the gated and COI-only pair lists, both scored
    by `rank`, because they become the preference lists.
    """

    eligible: dict[int, list[tuple[str, float]]]
    released: dict[int, list[tuple[str, float]]]
    rank: dict[tuple[str, int], float]
    affinity: dict[tuple[str, int], float]
    reviewer_cap: dict[str, int]


def build_pair_scores(
    papers: list[dict],
    paper_cache: Mapping[str, dict],
    candidate_emails: list[str],
    candidate_matrix: np.ndarray,
    reviewers_by_email: Mapping[str, object],
    extra_conflicts: Mapping[int, set[str]],
    *,
    area_gate: bool = True,
    score_mode: str = DEFAULT_SCORE_MODE,
    score_seed: int = DEFAULT_SCORE_SEED,
    light_cap: int,
    full_cap: int,
    reserve_cap: int = DEFAULT_RESERVE_CAP,
) -> PairScores:
    """Gated and area-released (reviewer, paper) pairs, plus per-reviewer caps.

    Gated pairs drive the normal phases; area-released pairs (COI-only) back the
    relaxation phases, so the score tables and the caps cover the superset.

    Under --score-mode random the pairs are ranked by `random_pair_score`
    instead of the cosine, which is what makes a baseline arm: COI, the area
    gate, the anchors and every cap still bind, but nothing in the decision path
    knows anything about affinity. The cosine is still computed -- `affinity`
    keeps it -- because measuring the true affinity of the slate a blind matcher
    produces is the entire point.
    """
    eligible: dict[int, list[tuple[str, float]]] = {}
    released: dict[int, list[tuple[str, float]]] = {}
    affinity: dict[tuple[str, int], float] = {}
    rank = affinity if score_mode == "specter2" else {}
    reviewer_cap: dict[str, int] = {}
    for p in papers:
        pid = p["pid"]
        paper_vec = np.array(paper_cache[str(pid)]["vector"], dtype=np.float32)
        conflicts = extra_conflicts.get(pid, frozenset())
        pairs_all = eligible_scores(
            p, candidate_emails, candidate_matrix, paper_vec, reviewers_by_email,
            area_gate=False, extra_conflicts=conflicts,
        )
        pairs_gated = pairs_all if not area_gate else eligible_scores(
            p, candidate_emails, candidate_matrix, paper_vec, reviewers_by_email,
            extra_conflicts=conflicts,
        )
        for email, cosine in pairs_all:
            affinity[(email, pid)] = cosine
            if rank is not affinity:
                rank[(email, pid)] = random_pair_score(score_seed, email, pid)
            reviewer_cap[email] = reviewer_paper_cap(
                reviewers_by_email[email], light_cap, full_cap, reserve_cap
            )
        # Re-scored off `rank`, unconditionally and never off the cosine: these
        # lists become the preference lists, and deferred_acceptance's deferral
        # deques assume their order matches the score it bumps on ("the pref
        # list is descending, so appends keep each deque sorted"). Ordering by
        # affinity while ranking on the draw would pop the wrong deferred
        # candidate -- silently, with no self-check to catch it. In specter2
        # mode rank[...] IS the cosine, so this is a no-op by construction.
        released[pid] = [(e, rank[(e, pid)]) for e, _ in pairs_all]
        eligible[pid] = released[pid] if pairs_gated is pairs_all else [
            (e, rank[(e, pid)]) for e, _ in pairs_gated
        ]
    return PairScores(eligible, released, rank, affinity, reviewer_cap)


def write_pairs_csv(
    path: str,
    slates: Mapping[int, Sequence[str]],
    assigned_via: Mapping[tuple[int, str], str],
    rank_lookup: Mapping[tuple[str, int], float],
    affinity_lookup: Mapping[tuple[str, int], float],
) -> None:
    """Atomically write one row per assigned pair, for comparing arms.

    A measurement artifact, never an upload -- hence the roster `email` rather
    than write_hotcrp_csv's `hotcrp_email`, which is the address HotCRP accepts
    but not the key the fingerprint caches and seniority CSVs are keyed on.

    `rank_score` and `affinity` are the same number under --score-mode specter2
    and differ under a randomized arm, which makes the two columns side by side
    the readable evidence of which run this was. Full repr, not the reports'
    three decimals: the differences this repo already reasons about are
    0.0006-scale and three decimals cannot resolve them.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temporary = Path(f.name)
            writer = csv.writer(f)
            writer.writerow(["pid", "email", "phase", "rank_score", "affinity"])
            for pid in sorted(slates):
                emails = sorted(
                    slates[pid], key=lambda e: (-affinity_lookup[(e, pid)], e)
                )
                for email in emails:
                    writer.writerow([
                        pid,
                        email,
                        assigned_via.get((pid, email), "fill"),
                        repr(rank_lookup[(email, pid)]),
                        repr(affinity_lookup[(email, pid)]),
                    ])
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=DEFAULT_DATA, help="path to the HotCRP paper export JSON")
    parser.add_argument(
        "--hotcrp-csv",
        help="write the final slates as a HotCRP bulk-update CSV that replaces all R1 reviews",
    )
    parser.add_argument(
        "--paper-policy", choices=PAPER_POLICIES, default="registered",
        help="paper selection policy (default: registered)",
    )
    parser.add_argument(
        "--exclude-pids", type=parse_exclude_pids, default=frozenset(),
        help="comma-separated paper IDs to exclude regardless of policy, "
             "e.g. a known test submission (default: none)",
    )
    parser.add_argument("--csv", default=DEFAULT_CSV, help="path to the reviewer CSV")
    parser.add_argument(
        "--cap-overrides", default=DEFAULT_CAP_OVERRIDES,
        help="hand-maintained per-reviewer paper-cap override CSV (email,cap,note), "
             "covering both PC members and reserve reviewers; wins over the PC "
             "acceptance form's own 'Override paper assignment number' column "
             "(default: %(default)s)",
    )
    parser.add_argument("--pcinfo", default=pc_membership.DEFAULT_PCINFO, help="HotCRP user export deciding who is still on the PC (default: %(default)s)")
    parser.add_argument("--no-pc-check", action="store_true", help="keep everyone the roster lists, even if HotCRP no longer marks them pc")
    parser.add_argument(
        "--fingerprint-cache", default=DEFAULT_FINGERPRINT_CACHE, help="path to the reviewer fingerprint cache"
    )
    parser.add_argument(
        "--paper-cache", default=DEFAULT_PAPER_CACHE, help="path to the writable paper fingerprint cache"
    )
    parser.add_argument(
        "--reviewers-per-paper", type=int, default=DEFAULT_REVIEWERS_PER_PAPER,
        help="target reviewer slots per paper (default: %(default)s)"
    )
    parser.add_argument(
        "--surplus-per-paper", type=int, default=DEFAULT_SURPLUS_PER_PAPER, metavar="N",
        help="extra reviewer(s) a paper may gain once the fill phases leave capacity "
             "unspent, offered to the worst-matched papers first (default: %(default)s). "
             "0 leaves the leftover capacity unspent"
    )
    parser.add_argument("--light-cap", type=int, default=7, help="max papers per light PC member (default: 7)")
    parser.add_argument("--full-cap", type=int, default=15, help="max papers per full PC member (default: 15)")
    parser.add_argument(
        "--include-reserves", action="store_true",
        help="add the reserve reviewers to the pool (roster, fingerprints and seniority)"
    )
    parser.add_argument(
        "--reserve-cap", type=int, default=DEFAULT_RESERVE_CAP,
        help=f"max papers per reserve reviewer (default: {DEFAULT_RESERVE_CAP})"
    )
    parser.add_argument(
        "--reserve-info", default=DEFAULT_RESERVE_INFO,
        help=f"reserve roster for --include-reserves (default: {DEFAULT_RESERVE_INFO})"
    )
    parser.add_argument(
        "--reserve-fingerprint-cache", default=cache_path("reserve_fingerprints.json"),
        help="reserve fingerprint cache for --include-reserves (default: %(default)s)"
    )
    parser.add_argument(
        "--reserve-seniority", default=report_path("reserve_seniority.csv"),
        help="reserve seniority CSV for --include-reserves (default: %(default)s)"
    )
    parser.add_argument(
        "--area-weight", type=float, default=1.0,
        help="weight of the topics document relative to the title+abstract document (default: 1.0)"
    )
    parser.add_argument(
        "--no-area-gate", action="store_true",
        help="skip the hard area-eligibility gate; consider all non-conflicted reviewers"
    )
    parser.add_argument(
        "--score-mode", choices=SCORE_MODES, default=DEFAULT_SCORE_MODE,
        help="what the matcher RANKS on (default: %(default)s). `random` replaces the "
             "SPECTER2 similarity with a reproducible per-(reviewer, paper) draw, "
             "giving a randomized baseline that still honours every COI layer, the "
             "senior anchor, the junior/out-of-area caps, the same-country cap and "
             "every load cap. Reported match goodness stays the TRUE SPECTER2 "
             "similarity either way, so the affinity drop is readable directly"
    )
    parser.add_argument(
        "--score-seed", type=int, default=DEFAULT_SCORE_SEED, metavar="N",
        help="seed for --score-mode random (default: %(default)s). The draw is a pure "
             "function of (seed, reviewer, paper), so a rerun is identical and the "
             "gated and area-released passes agree on every pair"
    )
    parser.add_argument(
        "--pairs-csv", metavar="PATH",
        help="write one row per assigned pair (pid, email, phase, rank_score, "
             "affinity) for comparing arms; see scripts/compare_baselines.py. Unlike "
             "--hotcrp-csv this is a measurement artifact and never an upload, so it "
             "is written even when the slate is short"
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
        "--same-country-cap", type=int, default=DEFAULT_SAME_COUNTRY_CAP, metavar="N",
        help="max reviewers from a paper's own majority-author country (default: "
             "%(default)s). One rule for every country: a US paper is capped on US "
             "reviewers exactly as a Chinese paper is capped on Chinese ones. Keyed "
             "on where the institution is, never anyone's nationality; HK, MO, TW "
             "and SG are separate ISO codes. 0 means no same-country reviewer at "
             "all; use --no-same-country-cap to switch the policy off"
    )
    parser.add_argument(
        "--no-same-country-cap", action="store_true",
        help="disable the same-country cap entirely (distinct from --same-country-cap 0, "
             "which admits no same-country reviewer)"
    )
    parser.add_argument(
        "--region-majority", type=float, default=DEFAULT_REGION_MAJORITY,
        help="share of a paper's placed authors that must share a country for its "
             "cap to bind (default: %(default)s)"
    )
    parser.add_argument(
        "--region-min-resolved", type=float, default=DEFAULT_REGION_MIN_RESOLVED,
        help="share of a paper's authors whose country must be known before it is "
             "judged at all; below this the paper is uncapped and reported "
             "(default: %(default)s)"
    )
    parser.add_argument(
        "--affiliation-countries", default=affiliation_country.DEFAULT_COUNTRIES,
        help="hand-maintained affiliation -> country file (default: %(default)s)"
    )
    parser.add_argument(
        "--no-coauthor-coi", action="store_true",
        help="disable the derived co-author COI. On by default: a reviewer who "
             "published with one of a paper's authors inside --coauthor-years is "
             "conflicted with it whether or not anyone declared it"
    )
    parser.add_argument(
        "--coauthor-years", type=int, default=coauthor_coi.DEFAULT_COAUTHOR_YEARS,
        help="calendar years of co-authorship that conflict (default: %(default)s)"
    )
    parser.add_argument(
        "--no-collaborator-coi", action="store_true",
        help="disable the derived declared-collaborator COI. On by default: a "
             "reviewer whose declared collaborators (data/inputs/hpca2027-pcinfo.csv) "
             "name a paper's author, or who is named as a collaborator by an author "
             "who declared one, is excluded whether or not pc_conflicts records it. "
             "Affiliation overlap -- the same signal, but too coarse to hard-block "
             "on -- is reported instead; see reviewer_match.collaborator_coi"
    )
    parser.add_argument(
        "--area-chair-csv", default=DEFAULT_AREA_CHAIR_CSV,
        help="area-chair acceptance form, one half of the area-chair check "
             "(default: %(default)s)"
    )
    parser.add_argument(
        "--no-area-chair-exclusion", action="store_true",
        help="assign papers to area chairs too. On by default: an area chair "
             "chairs papers and reviews none, and the check is the union of the "
             "HotCRP `~~area-chairs` tag and the acceptance form, because each "
             "source catches people the other misses"
    )
    parser.add_argument(
        "--coauthor-cache", default=coauthor_coi.DEFAULT_COAUTHORS,
        help="co-author cache from make dblp-snapshot (default: %(default)s)"
    )
    parser.add_argument(
        "--author-names-cache", default=coauthor_coi.DEFAULT_AUTHOR_NAMES,
        help="DBLP name spellings of submission authors (default: %(default)s)"
    )
    parser.add_argument(
        "--no-coauthor-identity", action="store_true",
        help="ignore DBLP's homonym numbering and treat every spelling of a name "
             "as one person. On by default the numbering is honoured, so a "
             "reviewer who wrote with a different Wei Zhang than the paper's "
             "author is not conflicted — but only where that author declared a "
             "DBLP page, which is about half of them"
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
    if args.surplus_per_paper < 0:
        parser.error("--surplus-per-paper must be non-negative")
    if args.light_cap < 0 or args.full_cap < 0:
        parser.error("--light-cap and --full-cap must be non-negative")
    if args.reserve_cap < 0:
        parser.error("--reserve-cap must be non-negative")
    if args.area_weight <= 0:
        parser.error("--area-weight must be greater than 0")
    if args.min_seniors < 0 or args.max_juniors < 0 or args.max_out_of_area < 0:
        parser.error("--min-seniors, --max-juniors, and --max-out-of-area must be non-negative")
    if args.same_country_cap < 0:
        parser.error("--same-country-cap must be non-negative")
    if args.coauthor_years <= 0:
        parser.error("--coauthor-years must be greater than 0")
    if not 0 < args.region_majority <= 1:
        parser.error("--region-majority must be greater than 0 and at most 1")
    if not 0 <= args.region_min_resolved <= 1:
        parser.error("--region-min-resolved must be between 0 and 1")
    if args.almost_senior_window < 0 or args.almost_junior_pubs < 0 or args.almost_out_of_area_career < 0:
        print("Warning: negative near-threshold values make every applicable reviewer a fallback", file=sys.stderr)
    if args.min_seniors > args.reviewers_per_paper:
        print(
            "Warning: --min-seniors exceeds --reviewers-per-paper; the criteria report will mark papers breaking",
            file=sys.stderr,
        )
    if args.score_mode != "specter2" and args.hotcrp_csv:
        parser.error(
            f"--score-mode {args.score_mode} produces a measurement baseline, not an "
            f"assignment anyone should upload; drop --hotcrp-csv (--pairs-csv is the "
            f"machine-readable artifact for comparing arms)"
        )
    if args.score_mode == "specter2" and args.score_seed != DEFAULT_SCORE_SEED:
        print("Warning: --score-seed has no effect under --score-mode specter2",
              file=sys.stderr)

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

    papers, skipped_papers = load_papers(
        args.data, paper_policy=args.paper_policy, exclude_pids=args.exclude_pids,
        with_skipped=True
    )
    if not papers:
        print(f"No papers found in {args.data}", file=sys.stderr)
        return 1

    paper_cache = fp.load_fingerprint_cache(args.paper_cache)
    build_paper_fingerprints(
        papers, paper_cache, args.paper_cache, area_weight=args.area_weight, device=args.device
    )

    reviewer_fp = fp.load_fingerprint_cache(args.fingerprint_cache)
    pcinfo = None if args.no_pc_check else args.pcinfo
    try:
        reviewers_by_email = {
            r.email: r for r in load_reviewers(
                args.csv, pcinfo_path=pcinfo, cap_overrides_path=args.cap_overrides
            )
        }
    except (FileNotFoundError, ValueError) as exc:
        if pcinfo and str(exc).startswith(pcinfo):
            parser.error(str(exc))
        raise

    # The reserve roster is read whether or not reserves are being assigned from,
    # because it is also where the reserves who have since been elevated to the
    # PC live: they never filled in an acceptance form, so this file is the only
    # place their DBLP page and derived areas exist. They are PC members now, so
    # they join the pool unconditionally -- excluding reserves must not quietly
    # exclude five sitting PC members too.
    reserves: list = []
    if pcinfo:
        try:
            reserves = load_reserve_reviewers(
                args.reserve_info, args.data, pcinfo_path=pcinfo,
                cap_overrides_path=args.cap_overrides,
            )
        except FileNotFoundError:
            if args.include_reserves:
                raise
            print(f"WARNING: {args.reserve_info} not found, so reserve reviewers "
                  f"promoted to the PC cannot be identified and are absent from the "
                  f"pool; run `make reserve-info`", file=sys.stderr)
    elif args.include_reserves:
        reserves = load_reserve_reviewers(
            args.reserve_info, args.data, pcinfo_path=None,
            cap_overrides_path=args.cap_overrides,
        )
        print("WARNING: --no-pc-check leaves the `~~ex-rr` tag unreadable, so any "
              "reserve since promoted to the PC is assigned at the reserve cap",
              file=sys.stderr)

    promoted, already_pc, true_reserves = split_promoted_reserves(
        reserves, reviewers_by_email
    )
    if already_pc:
        print(f"{len(already_pc)} promoted reserve(s) also returned the acceptance "
              f"form; the form record wins: {', '.join(sorted(already_pc))}",
              file=sys.stderr)

    merged = promoted + (true_reserves if args.include_reserves else [])
    if merged:
        reserve_fp = fp.load_fingerprint_cache(args.reserve_fingerprint_cache)
        reserve_seniority = {}
        if seniority is not None:
            try:
                reserve_seniority = load_seniority(args.reserve_seniority)
            except FileNotFoundError:
                reserve_seniority = {}
        # Missing artifacts are fatal for the promoted half. A reserve silently
        # absent from the pool is a thinner assignment; a PC member silently
        # absent is the roster being wrong, which is the failure the membership
        # check exists to prevent.
        if promoted and not reserve_fp:
            parser.error(
                f"{args.reserve_fingerprint_cache}: not found or empty, so the "
                f"{len(promoted)} reserve reviewer(s) promoted to the PC cannot be "
                f"scored and would be dropped from the pool; run `make reserves`"
            )
        if promoted and seniority is not None and not reserve_seniority:
            parser.error(
                f"{args.reserve_seniority}: not found or empty, so the "
                f"{len(promoted)} reserve reviewer(s) promoted to the PC would be "
                f"unclassified and could never fill a senior slot; run `make reserves`"
            )

        # A reviewer with no fingerprint cannot be scored and one with no
        # seniority row cannot fill a senior slot. Either way they would simply
        # be absent from the pool, which is indistinguishable from having none —
        # so say so rather than let the roster silently shrink.
        no_fp = [r.email for r in merged if r.email not in reserve_fp]
        no_sen = [
            r.email for r in merged
            if seniority is not None and r.email not in reserve_seniority
        ]
        if no_fp:
            print(f"WARNING: {len(no_fp)} reserve-roster reviewer(s) have no "
                  f"fingerprint in {args.reserve_fingerprint_cache} and cannot be "
                  f"assigned: {', '.join(no_fp[:5])}{' ...' if len(no_fp) > 5 else ''} — "
                  f"run `make reserves`", file=sys.stderr)
        if no_sen:
            print(f"WARNING: {len(no_sen)} reserve-roster reviewer(s) have no row in "
                  f"{args.reserve_seniority}; they can fill slots but never a "
                  f"senior one: {', '.join(no_sen[:5])}"
                  f"{' ...' if len(no_sen) > 5 else ''}", file=sys.stderr)

        reviewers_by_email.update({r.email: r for r in merged})
        reviewer_fp.update(reserve_fp)
        if seniority is not None:
            seniority.update(reserve_seniority)
        if promoted:
            by_tier = Counter(r.tier for r in promoted)
            counts = ", ".join(f"{tier} {n}" for tier, n in sorted(by_tier.items()))
            print(f"Ex-reserve reviewers promoted to the PC: {len(promoted)} "
                  f"({counts}); assigned at the PC cap for their tier, not the "
                  f"reserve cap", file=sys.stderr)
        if args.include_reserves:
            print(f"Reserve reviewers included: {len(true_reserves)} roster, "
                  f"{len(reserve_fp)} fingerprinted, {len(reserve_seniority)} classified "
                  f"(cap {args.reserve_cap} papers each)", file=sys.stderr)

    # Checked once here, after both rosters are merged into one dict -- doing
    # it inside load_reviewers or load_reserve_reviewers individually would
    # misreport the other roster's emails as typos, since neither loader can
    # see the other's pool.
    cap_overrides = load_cap_overrides(args.cap_overrides)
    unmatched_caps = sorted(set(cap_overrides) - set(reviewers_by_email))
    if unmatched_caps:
        print(
            f"WARNING: {len(unmatched_caps)} {args.cap_overrides} email(s) match "
            f"neither a PC member nor a reserve reviewer in the pool (typo, or "
            f"they left?): {', '.join(unmatched_caps[:5])}"
            f"{' ...' if len(unmatched_caps) > 5 else ''}",
            file=sys.stderr,
        )

    # An area chair chairs papers and reviews none. This has to run after the
    # reserve merge above, not inside the PC load: a reserve can be an area
    # chair too (one is today), and `reviewers_by_email.update(...)` would put
    # them straight back. Everything downstream draws from this dict, so it is
    # the one place the rule needs stating.
    area_chairs_dropped: list[str] = []
    if not args.no_area_chair_exclusion:
        index = None
        try:
            index = pc_membership.load_pc_accounts(args.pcinfo)
        except (FileNotFoundError, ValueError) as exc:
            # --no-pc-check exists for an export staler than the rosters, which
            # is about pruning people off a roster. A stale export's tags are
            # still good evidence someone is a chair, so the tag half survives
            # it; only a missing or truncated file forces the form-only reading.
            print(f"WARNING: {exc.__class__.__name__} reading {args.pcinfo}; the "
                  f"area-chair check falls back to the acceptance form alone and "
                  f"cannot see the `~~area-chairs` tag", file=sys.stderr)
        try:
            chair_emails = area_chair_emails(
                args.area_chair_csv, pcinfo_path=args.pcinfo if index else None
            )
        except FileNotFoundError:
            parser.error(
                f"{args.area_chair_csv}: not found, so area chairs cannot be kept out "
                f"of the reviewer pool; download the acceptance form, or pass "
                f"--no-area-chair-exclusion to assign papers to them anyway"
            )
        area_chairs_dropped = drop_area_chairs(reviewers_by_email, chair_emails, index)
        if area_chairs_dropped:
            shown = ", ".join(area_chairs_dropped[:5])
            more = f", +{len(area_chairs_dropped) - 5} more" if len(area_chairs_dropped) > 5 else ""
            print(f"Area chairs excluded from the reviewer pool: "
                  f"{len(area_chairs_dropped)} of {len(chair_emails)} ({shown}{more})",
                  file=sys.stderr)
        else:
            print(f"Area-chair check: none of the {len(chair_emails)} area chairs "
                  f"was in the reviewer pool", file=sys.stderr)

    candidate_emails = [e for e in reviewer_fp if e in reviewers_by_email]
    candidate_matrix = np.array([reviewer_fp[e]["vector"] for e in candidate_emails], dtype=np.float32)

    # Conflicts DBLP implies but nobody declared, folded into the same exclusion
    # the declared ones get. Built before the pairs, because every phase draws
    # from lists this filters and none of them re-checks COI afterwards.
    derived: dict[int, dict[str, coauthor_coi.CoauthorConflict]] = {}
    coauthor_index = None
    if not args.no_coauthor_coi:
        try:
            coauthors = coauthor_coi.load_coauthors(args.coauthor_cache)
            author_names = coauthor_coi.load_author_names(args.author_names_cache)
        except FileNotFoundError as exc:
            # Silently assigning conflicted reviewers is worse than not running,
            # so a missing cache stops here rather than dropping the layer.
            parser.error(f"{exc.filename}: not found; run `make dblp-snapshot`, "
                         f"or pass --no-coauthor-coi to assign without this check")
        with open(args.data, encoding="utf-8") as f:
            all_papers = json.load(f)
        roster = list(reviewers_by_email.values())
        coauthor_index = coauthor_coi.build_index(
            roster, coauthors, years=args.coauthor_years
        )
        derived = coauthor_coi.derive_conflicts(
            papers, coauthor_index, author_names, all_papers,
            use_identity=not args.no_coauthor_identity,
        )

    # Same "conservative" direction, sourced from HotCRP's own declared
    # `collaborators` field rather than DBLP -- see collaborator_coi's module
    # docstring. Reads args.pcinfo directly, independent of --no-pc-check:
    # that flag turns off roster *pruning*, a different question from
    # whether this COI check can run. Only the name-matched subset is
    # excluded; the noisier affiliation-overlap signal is reported, not
    # hard-blocked -- see collaborator_coi.hard_conflicts.
    collab_derived: dict[int, dict[str, collaborator_coi.CollaboratorConflict]] = {}
    collab_hard: dict[int, set[str]] = {}
    if not args.no_collaborator_coi:
        try:
            pcinfo_index = pc_membership.load_pc_accounts(args.pcinfo)
        except (FileNotFoundError, ValueError) as exc:
            parser.error(f"{exc}; needed for the declared-collaborator COI check, "
                         f"or pass --no-collaborator-coi to assign without it")
        collab_profiles = collaborator_coi.build_index(
            list(reviewers_by_email.values()), pcinfo_index
        )
        collab_derived = collaborator_coi.derive_conflicts(papers, collab_profiles, pcinfo_index)
        collab_hard = collaborator_coi.hard_conflicts(collab_derived)

    pair_scores = build_pair_scores(
        papers, paper_cache, candidate_emails, candidate_matrix, reviewers_by_email,
        {p["pid"]: set(derived.get(p["pid"], ())) | collab_hard.get(p["pid"], set())
         for p in papers},
        area_gate=not args.no_area_gate,
        score_mode=args.score_mode, score_seed=args.score_seed,
        light_cap=args.light_cap, full_cap=args.full_cap, reserve_cap=args.reserve_cap,
    )
    # score_lookup is what every phase ranks, bumps and self-checks on;
    # affinity_lookup is the true SPECTER2 cosine every report prints. The same
    # object unless --score-mode random split them.
    eligible_by_pid = pair_scores.eligible
    released_by_pid = pair_scores.released
    score_lookup = pair_scores.rank
    affinity_lookup = pair_scores.affinity
    reviewer_cap = pair_scores.reviewer_cap

    if args.score_mode != "specter2":
        # On stdout, not stderr: the transcript is the archived artifact, and a
        # reader who finds one of these files months later has to be able to see
        # from the file alone that it is not an assignment.
        gate = "area-blind (--no-area-gate)" if args.no_area_gate else "area-aware"
        print(f"\n=== Randomized baseline: --score-mode {args.score_mode}, "
              f"seed {args.score_seed}, {gate} ===")
        print("The matcher ranked on a reproducible per-(reviewer, paper) draw, not on "
              "SPECTER2 similarity. Every COI layer, the senior anchor, the "
              "junior/out-of-area caps, the same-country cap and every reviewer load "
              "cap bound exactly as they do in production. Match goodness below is the "
              "TRUE SPECTER2 similarity of the slate that produced.")
        print("This is a stable matching under random preferences, NOT a uniform draw "
              "over feasible assignments: the constraints alone shape the slate, which "
              "is what the area-aware and area-blind arms exist to separate.")
        print("Three sections below are NOT comparable against a production run. The "
              "relaxation report: with --no-area-gate there is no gate left to release, "
              "so every pick is labelled 'senior anchor'/'fill' and the arm looks like "
              "it struggled less. The same-country 'traded a better-matched reviewer' "
              "count: there is no better-matched reviewer when the ranking is noise. "
              "And any full-slate goodness figure: the surplus stage offers slots to "
              "the worst-matched papers, 'worst-matched' is measured on the ranking "
              "score, and that means something different here -- compare arms at "
              "--surplus-per-paper 0.")

    if area_chairs_dropped:
        # On stdout as well as stderr: assignment.txt is the archived artifact,
        # and every other hard filter leaves its evidence there.
        print(f"\n=== Area chairs excluded from the reviewer pool ===")
        print(f"{len(area_chairs_dropped)} area chair(s) hold no reviews, by the rule "
              f"that an area chair chairs papers and reviews none. Membership is the "
              f"union of the HotCRP `~~area-chairs` tag and the acceptance form.")
        for email in area_chairs_dropped:
            print(f"  {email}")

    report_conflict_coverage(papers, reviewers_by_email)
    if coauthor_index is not None:
        report_coauthor_coi(papers, derived, coauthor_index, reviewers_by_email,
                            args.coauthor_years)
    if not args.no_collaborator_coi:
        report_collaborator_coi(papers, collab_derived, reviewers_by_email)

    # The same-country cap binds before any phase runs, and no phase releases it.
    countries: list[CountryCap] = []
    reviewer_country: dict[str, str] = {}
    paper_coverage: dict[int, tuple[int, int]] = {}
    thin_papers: list[int] = []
    if not args.no_same_country_cap:
        # A fresh, independent load rather than reusing `index`/`pcinfo_index`
        # from the area-chair or collaborator-COI blocks above: both are only
        # assigned when their own flag is enabled, so referencing either here
        # would risk a NameError depending on which flags are set.
        # pc_membership.load_pc_accounts caches on (path, mtime, size), so this
        # is a cache hit whenever one of those blocks already ran.
        try:
            country_pcinfo = pc_membership.load_pc_accounts(args.pcinfo)
        except (FileNotFoundError, ValueError) as exc:
            country_pcinfo = None
            print(f"WARNING: {exc.__class__.__name__} reading {args.pcinfo}; the "
                  f"same-country cap falls back to the affiliation-only layers and "
                  f"cannot see HotCRP's own declared country", file=sys.stderr)
        layers = affiliation_country.load_layers(
            args.affiliation_countries, pcinfo_index=country_pcinfo
        )
        countries, reviewer_country, paper_coverage, thin_papers = build_country_caps(
            papers, candidate_emails, reviewers_by_email, args.same_country_cap, layers,
            majority=args.region_majority, min_resolved=args.region_min_resolved,
        )
    country_capped: list[tuple[frozenset[str], dict[int, int]]] = [
        (c.members, c.papers) for c in countries
    ]
    capped_pids = {pid for c in countries for pid in c.papers}

    pids = [p["pid"] for p in papers]
    # The base target, and the only target any report or the exit code judges.
    # A surplus slot rides on top of it and is never folded in.
    paper_target = {pid: args.reviewers_per_paper for pid in pids}
    slate_ceiling = args.reviewers_per_paper + args.surplus_per_paper
    paper_prefs = {
        pid: [email for email, _ in sorted(eligible_by_pid[pid], key=lambda es: -es[1])] for pid in pids
    }
    released_prefs = {
        pid: [email for email, _ in sorted(released_by_pid[pid], key=lambda es: -es[1])] for pid in pids
    }

    assigned_via: dict[tuple[int, str], str] = {}
    # Only the F1 pass mixes a country class with the seniority classes, so only
    # it can lose the stability guarantee; every other path stays laminar.
    capped_blocking = 0

    if args.no_seniority:
        slates = deferred_acceptance(pids, paper_prefs, paper_target, reviewer_cap,
                                     score_lookup, country_capped)
        # Judge stability on the gated pass alone — the area-released fill
        # below deliberately steps outside the gated preference lists.
        blocking = count_blocking_pairs(eligible_by_pid, slates, reviewer_cap, paper_target,
                                        score_lookup, country_capped)
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
            set(reviewer_cap), country_capped,
        )
        for pid, emails in held_r.items():
            for e in emails:
                assigned_via[(pid, e)] = "fill (area released)"
        paper_held = slates
        # No seniority data means no junior/out-of-area classes to cap, which is
        # this branch's existing contract; the same-country cap still binds.
        surplus_capped = country_capped
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

        # Region caps ride along in every phase, including the anchors and the
        # cap-relaxed fill: a paper under-fills rather than exceed one.
        # A1: anchor each paper's best eligible in-area senior(s) — frozen afterwards.
        anchor_target = {pid: min(args.min_seniors, args.reviewers_per_paper) for pid in pids}
        run_phase("senior anchor", paper_prefs, anchor_target, pools.seniors, country_capped)
        # A2: papers short a senior try area-released true seniors (area is
        # released before the senior requirement is relaxed).
        a2_target = {pid: max(0, anchor_target[pid] - len(slates[pid])) for pid in pids}
        run_phase("senior anchor (area released)", released_prefs, a2_target, pools.seniors,
                  country_capped)
        # A3: papers still senior-less fall back to an almost-senior, any area.
        a3_target = {pid: max(0, anchor_target[pid] - len(slates[pid])) for pid in pids}
        run_phase("almost-senior anchor", released_prefs, a3_target, pools.almost_seniors,
                  country_capped)
        # F1: main fill — everyone competes on score within the area gate,
        # juniors and out-of-area reviewers each capped per paper.
        capped = [(pools.juniors, args.max_juniors),
                  (pools.out_of_area, args.max_out_of_area), *country_capped]
        # What the anchors froze, so the F1 self-check judges F1 in F1's terms.
        f1_seed = class_counts_of(slates, pids, capped)
        fill_target = {pid: args.reviewers_per_paper - len(slates[pid]) for pid in pids}
        held2, prefs2, cap2 = run_phase("fill", paper_prefs, fill_target, set(reviewer_cap), capped)
        # F2: under-filled papers fill from the area-released pool; the caps
        # keep counting what earlier phases assigned.
        f2_target = {pid: args.reviewers_per_paper - len(slates[pid]) for pid in pids}
        run_phase("fill (area released)", released_prefs, f2_target, set(reviewer_cap), capped)
        # F3: papers still under-filled may exceed the junior and out-of-area
        # caps with extra almost-nots. The same-country cap is not relaxed here.
        f3_target = {pid: args.reviewers_per_paper - len(slates[pid]) for pid in pids}
        run_phase(
            "fill (cap relaxed)", released_prefs, f3_target,
            pools.almost_not_juniors | pools.almost_not_out_of_area, country_capped,
        )
        paper_held = slates
        # Surplus picks answer to F1's caps, not F3's relaxations: an extra
        # reviewer nobody was owed is not worth breaking composition policy for.
        surplus_capped = capped

        # Self-check the class-cap logic where its guarantee holds: the F1
        # pass, in F1 terms (its own prefs, caps, and targets).
        #
        # Region-capped papers are counted separately because the guarantee does
        # not extend to them: a region class crosses the seniority classes, and
        # greedy-by-score choice over a crossing family is not substitutable, so
        # deferred acceptance no longer promises a stable outcome there. Their
        # caps are still hard — `country_over` below is the check that replaces
        # this one. With --no-same-country-cap, capped_pids is empty and this
        # is exactly the check it has always been.
        pairs2 = {pid: [(e, score_lookup[(e, pid)]) for e in prefs2[pid]] for pid in pids}
        laminar = {pid: v for pid, v in pairs2.items() if pid not in capped_pids}
        blocking = count_blocking_pairs(
            laminar, held2, cap2, fill_target, score_lookup, capped, f1_seed
        )
        blocking_label = "F1 blocking pairs"
        crossing = {pid: v for pid, v in pairs2.items() if pid in capped_pids}
        capped_blocking = count_blocking_pairs(
            crossing, held2, cap2, fill_target, score_lookup, capped, f1_seed
        ) if crossing else 0

    # Spend whatever capacity the fill phases left on the worst-matched papers.
    # Purely additive — every phase above is frozen — so the F1 self-check and
    # every base-target report still describe the assignment they described
    # before this ran. Snapshot goodness first: it is the figure comparable
    # against a run with the stage off, and paper_held mutates underneath.
    base_goodness = paper_goodness(paper_held, affinity_lookup)
    spare_before = spare_capacity(reviewer_cap, used)
    surplus_added, surplus_rounds = distribute_surplus(
        pids, paper_prefs, released_prefs, paper_held, used, reviewer_cap,
        score_lookup, assigned_via, surplus_capped,
        base_target=paper_target, surplus_per_paper=args.surplus_per_paper,
    )
    spare_after = spare_capacity(reviewer_cap, used)

    # --- Report ---------------------------------------------------------------
    goodness = paper_goodness(paper_held, affinity_lookup)
    reviewer_load: dict[str, int] = defaultdict(int)
    for p in papers:
        pid = p["pid"]
        assigned = sorted(paper_held[pid], key=lambda e: -affinity_lookup[(e, pid)])
        under_filled = "  *** UNDER-FILLED ***" if len(assigned) < args.reviewers_per_paper else ""
        # Mutually exclusive with the marker above, and the count stays the base
        # target: assign_area_chairs.py reads this line's leading number.
        extra = len(assigned) - args.reviewers_per_paper
        surplus_note = f"  (+{extra} surplus)" if extra > 0 else ""
        g = goodness[pid]
        print(f"\n=== [{pid}] {p['title']}")
        print(f"    topics: {', '.join(p.get('topics', []))}")
        print(f"    assigned {len(assigned)} of {args.reviewers_per_paper} requested{surplus_note}{under_filled}")
        print(f"    match goodness: {'n/a' if g is None else format(g, '.3f')}")
        for rank, email in enumerate(assigned, 1):
            r = reviewers_by_email[email]
            cls = "" if seniority is None else "/" + (seniority[email]["class"] if email in seniority else "?")
            print(f"  {rank:2d}. {affinity_lookup[(email, pid)]:.3f}  {r.name} <{email}>  [{r.tier}{cls}]  ({r.primary})")
            reviewer_load[email] += 1

    total_pairs = sum(len(v) for v in paper_held.values())
    # Counted per tier rather than by naming 'light' and 'full': a tier the
    # check doesn't know about would otherwise be exempt from it, which is
    # exactly how a reserve reviewer could quietly draw a full PC load.
    over_by_tier: Counter = Counter()
    for e, n in reviewer_load.items():
        r = reviewers_by_email[e]
        if n > reviewer_paper_cap(r, args.light_cap, args.full_cap, args.reserve_cap):
            over_by_tier[r.tier] += 1
    light_over = over_by_tier["light"]
    full_over = over_by_tier["full"]
    canonical_areas = build_canonical_area_map(reviewers_by_email)
    area_stats = area_pool_stats(
        candidate_emails, reviewers_by_email, args.light_cap, args.full_cap, args.reserve_cap
    )
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
        seniority_summary = (
            f"seniority: {ok_n} papers OK, {deg_n} degraded, {brk_n} breaking — see report above; "
            f"{deep_junior_over} papers over the junior policy and {deep_oob_over} over the "
            f"out-of-area policy — should always be 0; "
        )

    # Not inside the seniority block: the surplus stage runs on both paths, so
    # the ceiling it must respect has to be checked on both too.
    over_ceiling = sum(1 for pid in pids if len(paper_held[pid]) > slate_ceiling)

    country_summary = ""
    if not args.no_same_country_cap:
        country_over = country_cap_report(
            papers, paper_held, countries, reviewer_country, paper_coverage,
            thin_papers, released_prefs, paper_target, affinity_lookup,
            {e for e in reviewer_cap if reviewer_load[e] < reviewer_cap[e]},
            cap=args.same_country_cap, majority=args.region_majority,
            min_resolved=args.region_min_resolved,
        )
        country_summary = (
            f"{country_over} papers over the same-country cap — should always be 0; "
            f"{capped_blocking} F1 blocking pairs among capped papers "
            f"(a country class crosses the seniority classes, so a stable matching "
            f"is not guaranteed there — see README); "
        )

    # Cheap, and the only thing that would catch the layer being computed and
    # then not actually threaded into the preference lists.
    coauthor_violations = sum(
        1 for pid, slate in paper_held.items() for email in slate
        if email in derived.get(pid, ())
    )
    coauthor_summary = "" if args.no_coauthor_coi else (
        f"{coauthor_violations} co-authored assignments — should always be 0; "
    )
    chairs_assigned = sum(1 for email in reviewer_load if email in set(area_chairs_dropped))
    area_chair_summary = "" if args.no_area_chair_exclusion else (
        f"{len(area_chairs_dropped)} area chair(s) excluded from the pool, "
        f"{chairs_assigned} still assigned — should always be 0; "
    )

    # The bracket, from data already in memory. eligible_by_pid, not
    # released_by_pid, so each arm is bracketed by the pool it actually drew
    # from -- released under --no-area-gate, gated otherwise. affinity_lookup,
    # not the pairs' own second element: those carry the ranking score.
    goodness_floor = {
        pid: sum(affinity_lookup[(e, pid)] for e, _ in eligible_by_pid[pid])
             / len(eligible_by_pid[pid])
        for pid in pids if eligible_by_pid[pid]
    }
    goodness_ceiling = {
        pid: sum(heapq.nlargest(
            args.reviewers_per_paper,
            (affinity_lookup[(e, pid)] for e, _ in eligible_by_pid[pid]),
        )) / args.reviewers_per_paper
        for pid in pids if len(eligible_by_pid[pid]) >= args.reviewers_per_paper
    }
    match_goodness_report(
        papers, goodness, floor=goodness_floor, ceiling=goodness_ceiling,
        ceiling_k=args.reviewers_per_paper,
    )
    surplus_placed = surplus_report(
        papers, surplus_added, base_goodness, goodness, affinity_lookup,
        reviewers_by_email, seniority, assigned_via,
        reviewers_per_paper=args.reviewers_per_paper,
        surplus_per_paper=args.surplus_per_paper,
        spare_before=spare_before, spare_after=spare_after, rounds=surplus_rounds,
    )
    n_excluded, n_relaxed = relaxation_report(
        skipped_papers, papers, paper_held, paper_target, assigned_via,
        goodness, affinity_lookup, reviewers_by_email, seniority,
        itemize_excluded=args.paper_policy != "submitted",
    )

    print(
        f"\nDone. {total_pairs} reviewer-paper pairs assigned across {len(papers)} papers, "
        f"{len(reviewer_load)} distinct reviewers used "
        f"(light cap {args.light_cap}, full cap {args.full_cap}"
        f"{f', reserve cap {args.reserve_cap}' if args.include_reserves else ''}; "
        f"{sum(over_by_tier.values())} over cap"
        f"{' (' + ', '.join(f'{n} {t}' for t, n in sorted(over_by_tier.items())) + ')' if over_by_tier else ''}"
        f" — should always be 0; "
        f"{blocking} {blocking_label} — should always be 0; "
        f"{seniority_summary}"
        f"{country_summary}"
        f"{coauthor_summary}"
        f"{area_chair_summary}"
        f"{surplus_placed} surplus reviewer(s) on {len(surplus_added)} paper(s) with "
        f"{spare_after} reviewer-slot(s) still unused, {over_ceiling} papers over the "
        f"{slate_ceiling}-reviewer ceiling — should always be 0; "
        f"{n_excluded} papers excluded and {n_relaxed} relaxed — see relaxation report above; "
        f"{total_missing} reviewer-slot(s) unfilled — see shortage report above).",
        file=sys.stderr,
    )
    if args.pairs_csv:
        # Above the "refusing a partial result" guard, unlike --hotcrp-csv below
        # it: a short slate is still a valid measurement, it is just not a valid
        # upload. A randomized arm routinely under-fills, and that run is
        # precisely the one worth comparing.
        try:
            write_pairs_csv(args.pairs_csv, paper_held, assigned_via,
                            score_lookup, affinity_lookup)
        except OSError as exc:
            print(f"ERROR: could not write pairs CSV {args.pairs_csv}: {exc}",
                  file=sys.stderr)
            return 1
    if args.paper_policy == "submitted" and total_missing:
        print(
            "ERROR: submitted-paper assignment is incomplete; refusing a partial result",
            file=sys.stderr,
        )
        return 1
    if args.hotcrp_csv:
        try:
            write_hotcrp_csv(args.hotcrp_csv, paper_held, affinity_lookup, reviewers_by_email)
        except OSError as exc:
            print(f"ERROR: could not write HotCRP CSV {args.hotcrp_csv}: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
