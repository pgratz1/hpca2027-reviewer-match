"""Give every paper that advances to revision a randomly drawn, load-balanced lead.

    python -m scripts.assign_paper_leads
    python -m scripts.assign_paper_leads --seed 7
    python -m scripts.assign_paper_leads --bar-cutoff 2.25 --bar-net no4
    python -m scripts.assign_paper_leads --no-keep-leads

**Which papers.** Every paper still under review (`submitted`, not
desk-rejected, not `--exclude-pids`) that *advances*: it has fewer than
`--min-reviews` submitted PC reviews, or it is over the revision bar. The bar
is `reviewer_match.review_scores.Bar`, the same definition
`scripts/revision_cutoffs.py` measures. By default a paper is under the bar
when its average pre-rebuttal overall merit is <= 2.5 **and** at most one
reviewer scored it 3 or better; `--bar-net`/`--bar-cutoff`/`--bar-comparator`
change it. TRC reviews count towards neither the review floor nor the average.

**Who can lead.** Only someone who has *submitted their review of that paper*
and is on the PC, full or light. The 7 reserves promoted onto the PC
(`~~ex-rr`, §3d in CLAUDE.md) are light PC members here. Other reserves, TRC
students and anyone on no roster never lead. A paper with no such reviewer is
reported as unassignable and gets no upload row.

**How many each.** Lead load is proportional to review load. A person's weight
is the number of R1 reviews `--pcassignments` (HotCRP's PC-assignments
download) gives them on papers still under review, so a 15-review full member
carries about twice the leads of a 7-review light member. Each person's share
of the L papers is `L * weight / total weight`. Nobody's share may exceed the
number of those papers they reviewed; the excess is spread over everyone else.
Each share is then rounded up or down at random, with expected value exactly
the share and the rounded quotas adding up to L (systematic sampling).

**The draw.** Papers are taken in random order, and each takes a uniformly
random candidate with quota left. If none has any, a chain of swaps among
this run's own draws frees one (an augmenting path). Only when no assignment
within the quotas exists does a paper go to the candidate with the fewest
leads per review, and that is counted as over quota. Every set is sorted
before it is shuffled and all randomness comes from one `random.Random(--seed)`,
so the same inputs and seed reproduce the same draw byte for byte.

**Reruns keep existing leads.** Existing leads are read from `--existing-leads`
(default: the `lead` rows of `--pcassignments`). A lead who is still an
eligible reviewer of a paper that still advances is kept and counts towards
their load. A paper that no longer advances gets a `clearlead`. A lead who is
no longer eligible is replaced. `--no-keep-leads` draws every lead afresh.

Writes `--upload`, a HotCRP bulk-assignment **delta** (`clearlead` rows, then
`lead` rows for new and replaced leads; kept leads are not repeated), plus
`--papers-csv` and `--loads-csv` reports. Nothing is uploaded. Offline and
instant apart from loading the rosters. Every output names real reviewers and
papers, so all three are gitignored.
"""

from __future__ import annotations

from reviewer_match.paths import assignment_path, input_path, report_path

import argparse
import csv
import math
import os
import random
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction

from reviewer_match import assignment_io
from reviewer_match import hotcrp_log
from reviewer_match import paper_matching
from reviewer_match import pc_membership
from reviewer_match import review_scores
from scripts.audit_reviewer_activity import build_roster

DEFAULT_REVIEWS = input_path("hpca2027-reviews.csv")
DEFAULT_LOG = input_path("hpca2027-log.csv")
DEFAULT_DATA = input_path("hpca2027-data.json")
DEFAULT_PCASSIGNMENTS = input_path("hpca2027-pcassignments.csv")
DEFAULT_UPLOAD = assignment_path("lead_upload.csv")
DEFAULT_PAPERS_CSV = report_path("paper_leads.csv")
DEFAULT_LOADS_CSV = report_path("lead_loads.csv")
DEFAULT_SEED = 1

# Tiers that may lead. An ex-reserve promoted onto the PC carries "light".
LEAD_TIERS = ("full", "light")

PAPERS_FIELDS = [
    "pid", "title", "reason", "reviews", "outstanding", "average", "scores",
    "candidates", "lead", "lead_name", "lead_tier", "status", "note",
]
LOADS_FIELDS = [
    "email", "name", "tier", "assigned_reviews", "eligible_papers", "target",
    "quota", "leads", "kept", "over_quota",
]


@dataclass
class LeadDraw:
    """What `assign_leads` decided, for the papers it was given."""

    leads: dict[int, str] = field(default_factory=dict)  # pid -> lead
    status: dict[int, str] = field(default_factory=dict)  # pid -> kept/new/redrawn/unassignable
    clears: list[int] = field(default_factory=list)  # pids whose existing lead is removed
    targets: dict[str, Fraction] = field(default_factory=dict)
    quotas: dict[str, int] = field(default_factory=dict)
    eligible_papers: dict[str, int] = field(default_factory=dict)
    kept: Counter = field(default_factory=Counter)  # email -> kept leads
    over_quota: list[int] = field(default_factory=list)  # pids placed over quota by this run


def proportional_targets(
    pool: list[str], weights: dict[str, int], lower: dict[str, int],
    upper: dict[str, int], total: int
) -> dict[str, Fraction]:
    """{email: share of `total`}: `lam * weight`, clipped to [lower, upper].

    `lam` is the one scale factor at which the clipped shares sum to exactly
    `total`. `upper` is how many of the papers a person reviewed (nobody can
    lead more); `lower` is what they are already committed to (kept leads, and
    papers they are the only possible lead for). Without the floor a person's
    quota could fall below papers nobody else can take, and those would be
    placed over quota. The sum is piecewise linear and non-decreasing in
    `lam`, so a binary search over the breakpoints finds the right segment and
    one exact interpolation solves it. Solvable whenever sum(lower) <= total
    <= sum(upper), which holds by construction.
    """
    def clipped(lam: Fraction) -> dict[str, Fraction]:
        return {e: min(max(lam * weights[e], Fraction(lower[e])), Fraction(upper[e])) for e in pool}

    def summed(lam: Fraction) -> Fraction:
        return sum(clipped(lam).values(), Fraction(0))

    points = sorted({Fraction(0)} | {Fraction(lower[e], weights[e]) for e in pool}
                    | {Fraction(upper[e], weights[e]) for e in pool})
    lo, hi = 0, len(points) - 1  # summed(points[lo]) <= total <= summed(points[hi])
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if summed(points[mid]) < total:
            lo = mid
        else:
            hi = mid
    a, b = points[lo], points[hi]
    sa, sb = summed(a), summed(b)
    lam = b if sb == sa else a + (b - a) * (total - sa) / (sb - sa)
    return clipped(lam)


def systematic_quotas(targets: dict[str, Fraction], rng: random.Random) -> dict[str, int]:
    """{email: floor or ceiling of the target}, each with expected value the target.

    One uniform draw strides through the cumulative targets in a random order,
    so the quotas sum to exactly the (integer) sum of the targets and nobody's
    rounding depends on anyone else's except through that one draw.
    """
    order = sorted(targets)
    rng.shuffle(order)
    offset = Fraction(rng.random())
    quotas: dict[str, int] = {}
    cumulative = Fraction(0)
    for email in order:
        before = math.floor(cumulative + offset)
        cumulative += targets[email]
        quotas[email] = math.floor(cumulative + offset) - before
    return quotas


def assign_leads(
    candidates: dict[int, list[str]],
    weights: dict[str, int],
    existing: dict[int, str],
    rng: random.Random,
) -> LeadDraw:
    """Draw one lead per paper in `candidates` ({pid: eligible reviewers}).

    `candidates` holds every paper that needs a lead, including those with no
    eligible reviewer (an empty list: reported unassignable, and any existing
    lead on it cleared). An `existing` lead who is still a candidate is kept
    and counts towards their load; kept leads are never moved.
    """
    draw = LeadDraw()
    pids = sorted(candidates)
    staffed = [pid for pid in pids if candidates[pid]]
    for pid in pids:
        if not candidates[pid]:
            draw.status[pid] = "unassignable"
            if pid in existing:
                draw.clears.append(pid)
        elif existing.get(pid) in candidates[pid]:
            draw.leads[pid] = existing[pid]
            draw.status[pid] = "kept"
            draw.kept[existing[pid]] += 1

    pool = sorted({e for pid in staffed for e in candidates[pid]})
    draw.eligible_papers = Counter(e for pid in staffed for e in candidates[pid])
    committed = draw.kept + Counter(
        candidates[pid][0] for pid in staffed if pid not in draw.leads and len(candidates[pid]) == 1
    )
    draw.targets = proportional_targets(
        pool, weights, committed, draw.eligible_papers, len(staffed)
    )
    draw.quotas = systematic_quotas(draw.targets, rng)

    holding: dict[str, set[int]] = {e: set() for e in pool}  # this run's movable draws

    def spare(email: str) -> int:
        return draw.quotas[email] - draw.kept[email] - len(holding[email])

    def shuffled(items) -> list:
        items = sorted(items)
        rng.shuffle(items)
        return items

    def place(pid: int, visited: set[str]) -> bool:
        """Kuhn's augmenting path: seat `pid`, moving this run's draws if needed."""
        options = shuffled(candidates[pid])
        for email in options:
            # A visited reviewer's only spare slot is the one being vacated for
            # the caller; taking it back would put that reviewer over quota.
            if email not in visited and spare(email) > 0:
                holding[email].add(pid)
                return True
        for email in options:
            if email in visited:
                continue
            visited.add(email)
            for other in shuffled(holding[email]):
                holding[email].discard(other)
                if place(other, visited):
                    holding[email].add(pid)
                    return True
                holding[email].add(other)
        return False

    overflow: dict[int, str] = {}
    for pid in shuffled(pid for pid in staffed if pid not in draw.leads):
        if place(pid, set()):
            continue
        # No assignment within the quotas: the fewest leads per review wins.
        def load(email: str) -> int:
            return draw.kept[email] + len(holding[email]) + sum(1 for e in overflow.values() if e == email)
        options = shuffled(candidates[pid])
        overflow[pid] = min(options, key=lambda e: Fraction(load(e) + 1, max(1, weights[e])))
        draw.over_quota.append(pid)

    for email, held in holding.items():
        for pid in held:
            draw.leads[pid] = email
    draw.leads.update(overflow)
    for pid in staffed:
        if pid not in draw.status:
            draw.status[pid] = "redrawn" if pid in existing else "new"
    draw.over_quota.sort()
    return draw


def lead_counts(draw: LeadDraw) -> Counter:
    return Counter(draw.leads.values())


def unassignable_note(reviewers: list[str], roster: dict[str, object]) -> str:
    """Why a paper has nobody who can lead it."""
    if not reviewers:
        return "no submitted reviews"
    tiers = {getattr(roster.get(e), "tier", None) for e in reviewers}
    if tiers == {"reserve"}:
        return "only reserve reviewers have submitted"
    return "no PC member has submitted a review"


def upload_rows(draw: LeadDraw, extra_clears: list[int]) -> list[tuple[int, str, str]]:
    """The HotCRP delta: clears first, then new and redrawn leads, by pid."""
    rows = [(pid, "clearlead", "") for pid in sorted(set(draw.clears) | set(extra_clears))]
    rows += [
        (pid, "lead", draw.leads[pid])
        for pid in sorted(draw.leads)
        if draw.status[pid] in ("new", "redrawn")
    ]
    return rows


def write_csv(path: str, header, rows) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    os.replace(tmp, path)


def fmt(value: float) -> str:
    return f"{value:.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--reviews", default=DEFAULT_REVIEWS, help="HotCRP review CSV export")
    parser.add_argument("--log", default=DEFAULT_LOG, help="HotCRP action log (identifies TRC reviews)")
    parser.add_argument("--data", default=DEFAULT_DATA, help="HotCRP paper export JSON")
    parser.add_argument(
        "--pcassignments", default=DEFAULT_PCASSIGNMENTS,
        help="HotCRP PC-assignments download: review-load weights, and existing leads by default"
    )
    parser.add_argument(
        "--existing-leads",
        help="CSV whose `lead` rows are the leads HotCRP holds (default: --pcassignments)"
    )
    parser.add_argument("--no-keep-leads", action="store_true", help="ignore existing leads and draw every lead afresh")
    parser.add_argument("--pcinfo", default=pc_membership.DEFAULT_PCINFO, help="HotCRP user export (roster tiers)")
    parser.add_argument("--exclude-pids", default="", help="comma-separated paper IDs to leave out")
    review_scores.add_bar_arguments(parser)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"random seed (default {DEFAULT_SEED})")
    parser.add_argument("--upload", default=DEFAULT_UPLOAD, help="HotCRP bulk-assignment delta to write")
    parser.add_argument("--papers-csv", default=DEFAULT_PAPERS_CSV, help="per-paper report to write")
    parser.add_argument("--loads-csv", default=DEFAULT_LOADS_CSV, help="per-reviewer load report to write")
    args = parser.parse_args()

    bar = review_scores.bar_from_args(args)
    exclude = paper_matching.parse_exclude_pids(args.exclude_pids)
    leads_path = args.existing_leads or args.pcassignments
    for path in (args.reviews, args.log, args.data, args.pcassignments, leads_path):
        if not os.path.exists(path):
            print(f"ERROR: {path} not found", file=sys.stderr)
            return 1

    training, outstanding = review_scores.review_rounds(hotcrp_log.load_log(args.log))
    reviews, skipped = review_scores.load_reviews(args.reviews, training)
    eligible, _ = review_scores.eligible_pids(args.data, exclude)
    scores: dict[int, list[int]] = {}
    reviewers: dict[int, list[str]] = {}
    titles: dict[int, str] = {}
    for review in reviews:
        scores.setdefault(review.pid, []).append(review.score)
        reviewers.setdefault(review.pid, []).append(review.email)
        titles.setdefault(review.pid, review.title)

    need = {pid: reason for pid in sorted(eligible) if (reason := bar.advances(scores.get(pid, [])))}

    roster = build_roster(args.pcinfo)
    def can_lead(email: str) -> bool:
        return getattr(roster.get(email), "tier", None) in LEAD_TIERS
    candidates = {pid: sorted({e for e in reviewers.get(pid, []) if can_lead(e)}) for pid in need}

    _, pairs = assignment_io.load_assignment_pairs(args.pcassignments)
    assigned = Counter(e for pid, emails in pairs.items() if pid in eligible for e in emails)
    pool = sorted({e for c in candidates.values() for e in c})
    unweighted = [e for e in pool if not assigned[e]]
    if unweighted:
        print(f"WARNING: {len(unweighted)} candidate lead(s) hold no review in {args.pcassignments} "
              f"(weighted as 1): {', '.join(unweighted[:5])}" + (" ..." if len(unweighted) > 5 else ""),
              file=sys.stderr)
    weights = {e: max(1, assigned[e]) for e in pool}

    existing = {} if args.no_keep_leads else assignment_io.load_leads(leads_path)
    draw = assign_leads(candidates, weights, existing, random.Random(args.seed))
    # A lead on a paper still under review that no longer advances is cleared.
    no_longer = sorted(pid for pid in existing if pid in eligible and pid not in need)
    elsewhere = sorted(pid for pid in existing if pid not in eligible)
    rows = upload_rows(draw, no_longer)

    # Self-checks: each should always be 0.
    counts = lead_counts(draw)
    checks = {
        "advancing papers with a candidate but no lead": sum(
            1 for pid, c in candidates.items() if c and pid not in draw.leads),
        "leads who are not a submitted PC reviewer of the paper": sum(
            1 for pid, e in draw.leads.items() if e not in candidates.get(pid, ())),
        "leads who are not full or light PC": sum(1 for e in draw.leads.values() if not can_lead(e)),
        "papers with more than one lead row": sum(
            1 for n in Counter(pid for pid, action, _ in rows if action == "lead").values() if n > 1),
    }
    inherited = {e: n - draw.quotas.get(e, 0) for e, n in draw.kept.items() if n > draw.quotas.get(e, 0)}

    write_csv(args.upload, assignment_io.LEAD_CSV_HEADER, rows)
    paper_rows = []
    for pid, reason in need.items():
        lead = draw.leads.get(pid, "")
        record = roster.get(lead)
        s = scores.get(pid, [])
        paper_rows.append([
            pid, titles.get(pid, ""), reason, len(s), outstanding[pid],
            f"{float(review_scores.average(s)):.3f}" if s else "",
            " ".join(str(x) for x in sorted(s)), len(candidates[pid]),
            lead, getattr(record, "name", "") if lead else "", getattr(record, "tier", "") if lead else "",
            draw.status[pid],
            unassignable_note(reviewers.get(pid, []), roster) if not candidates[pid]
            else ("over quota" if pid in draw.over_quota else ""),
        ])
    write_csv(args.papers_csv, PAPERS_FIELDS, paper_rows)
    load_rows = []
    for email in pool:
        record = roster[email]
        load_rows.append([
            email, getattr(record, "name", ""), record.tier, assigned[email],
            draw.eligible_papers[email], fmt(float(draw.targets[email])), draw.quotas[email],
            counts[email], draw.kept[email], max(0, counts[email] - draw.quotas[email]),
        ])
    write_csv(args.loads_csv, LOADS_FIELDS, load_rows)

    # ---- summary (stdout) ----------------------------------------------------
    reasons = Counter(need.values())
    status = Counter(draw.status.values())
    unassignable = [pid for pid in need if not candidates[pid]]
    print(f"Bar: {bar.describe()}. Seed {args.seed}.")
    print(f"{len(need)} of {len(eligible)} papers advance: {reasons[review_scores.OVER_BAR]} over the bar, "
          f"{reasons[review_scores.FEW_REVIEWS]} with fewer than {bar.min_reviews} reviews "
          f"({len(skipped)} TRC reviews left out).")
    print(f"Leads: {status['new']} new, {status['redrawn']} replaced, {status['kept']} kept; "
          f"{len(no_longer) + len(draw.clears)} cleared "
          f"({len(no_longer)} on papers that no longer advance, "
          f"{len(draw.clears)} on papers nobody can lead now).")
    if unassignable:
        print(f"{len(unassignable)} unassignable (no submitted PC reviewer): "
              + ", ".join(f"#{pid}" for pid in unassignable))
    if elsewhere:
        print(f"{len(elsewhere)} existing lead(s) on papers no longer under review left alone: "
              + ", ".join(f"#{pid}" for pid in elsewhere))
    print()
    print(f"  {'tier':<6} {'reviewers':>9} {'mean reviews':>12} {'mean leads':>10} {'leads/review':>12}   leads held")
    means = {}
    for tier in LEAD_TIERS:
        members = [e for e in pool if roster[e].tier == tier]
        if not members:
            continue
        mean_assigned = sum(assigned[e] for e in members) / len(members)
        mean_leads = sum(counts[e] for e in members) / len(members)
        means[tier] = (mean_assigned, mean_leads)
        spread = Counter(counts[e] for e in members)
        held = "  ".join(f"{n}:{spread[n]}" for n in sorted(spread))
        per_review = sum(counts[e] for e in members) / max(1, sum(assigned[e] for e in members))
        print(f"  {tier:<6} {len(members):>9} {mean_assigned:>12.2f} {mean_leads:>10.2f} {per_review:>12.3f}   {held}")
    if all(t in means and means[t][0] and means[t][1] for t in LEAD_TIERS):
        print(f"  full/light: {means['full'][1] / means['light'][1]:.2f}x the leads for "
              f"{means['full'][0] / means['light'][0]:.2f}x the reviews")
    pc_total = sum(1 for r in roster.values() if r.tier in LEAD_TIERS)
    print(f"  {pc_total - len(pool)} PC members have submitted no review of an advancing paper and lead nothing.")
    print()
    print(f"Over quota: {len(draw.over_quota)} placed by this run"
          + (f" (#{', #'.join(str(p) for p in draw.over_quota)})" if draw.over_quota else "")
          + (f"; {sum(inherited.values())} more inherited from existing leads" if inherited else "") + ".")
    failures = sum(checks.values())
    print("Self-checks (should always be 0): " + "; ".join(f"{name} {n}" for name, n in checks.items()))
    print(f"Wrote {args.upload} ({len(rows)} rows), {args.papers_csv}, {args.loads_csv}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
