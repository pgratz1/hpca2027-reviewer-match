"""Propose reviewer swaps to backfill papers a departed reviewer leaves short.

    python -m scripts.propose_reviewer_swaps --departed-email person@example.edu \\
        --paper-policy submitted --include-reserves --reserve-cap 6 \\
        --same-country-cap 1 --max-juniors 2 --exclude-pids 1152

    # A reviewer keeps most of their load but must be pulled off a few named
    # papers only (a COI surfaced late, an ability-to-review issue, etc.):
    python -m scripts.propose_reviewer_swaps --departed-email person@example.edu \\
        --departed-pids 1991,3825,4044 --paper-policy submitted \\
        --include-reserves --reserve-cap 6 --same-country-cap 1 --max-juniors 2 \\
        --exclude-pids 1152

Unlike scripts.fill_open_slots, which only ever *adds* a reviewer with spare
capacity, this proposes *moving* an already-assigned reviewer off a paper
that currently holds `--reviewers-per-paper + 1` reviewers (the
surplus-distribution stage's bonus recipients) onto one of the departed
reviewer's now-short papers -- freeing a slot on the donor paper back down to
`--reviewers-per-paper` rather than adding fresh load anywhere. Every
standing hard rule from the live pipeline still applies on both ends of the
move: COI (declared + derived co-author + derived collaborator), the area
gate, the same-country cap, the senior floor, and the junior/out-of-area
caps on the target paper. Two rules are specific to a swap: the mover must
not have already submitted their review on the paper they would leave (read
from data/inputs/hpca2027-log.csv), and removing them must not drop that
paper's own senior count below the floor either.

Reads `--current-csv` for the live paper/reviewer pairs (either
scripts.extract_log_assignments's reconstruction or HotCRP's own "Review
assignments" export -- see README's Incremental rerun section), separately
from `--log`, which is consulted only for each candidate's own
submitted/not-submitted status. Run `make log-assignments` first so
`--current-csv` reflects today's log, not a stale snapshot.

Prints, per short paper, its two best-matched feasible candidates (primary
and backup, ranked by SPECTER2 cosine similarity, read from the existing
fingerprint caches -- no new embedding) -- so the chair can ask the primary
first and fall back to the second if they decline or already started. Picks
are assigned globally, not independently per paper: the same reviewer never
appears twice across the whole proposal, because a reviewer who says yes to
two different papers' asks ends up overloaded (see `assign_unique_picks`).
`--pairs-csv PATH` writes the same picks as
`target_pid,rank,add_email,source_pid,remove_email,affinity` for the
chair's own record. Nothing HotCRP-upload-shaped is written and nothing is
applied; this is a proposal to read and act on by hand.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass

import numpy as np

from reviewer_match import affiliation_country
from reviewer_match import assignment_io
from reviewer_match import coauthor_coi
from reviewer_match import collaborator_coi
from reviewer_match import fingerprint as fp
from reviewer_match import hotcrp_log
from reviewer_match import pc_membership
from reviewer_match.area_chairs import area_chair_emails, drop_area_chairs
from reviewer_match.paper_matching import build_paper_fingerprints, eligible_scores, load_papers, parse_exclude_pids, PAPER_POLICIES
from reviewer_match.paths import assignment_path, cache_path, input_path, report_path
from reviewer_match.reserve_reviewers import DEFAULT_INFO as DEFAULT_RESERVE_INFO
from reviewer_match.reserve_reviewers import load_reserve_reviewers
from reviewer_match.reviewers import DEFAULT_CAP_OVERRIDES, load_reviewers
from reviewer_match.roster import DEFAULT_AREA_CHAIR_CSV

from scripts import assign_reviewers as ar
from scripts.classify_reviewers import DEFAULT_OUT as DEFAULT_SENIORITY, load_seniority

DEFAULT_CURRENT_CSV = assignment_path("current_assignment.csv")
DEFAULT_LOG = input_path("hpca2027-log.csv")
TOP_N = 2


@dataclass(frozen=True)
class Candidate:
    """One feasible (target, mover) pick."""

    email: str
    source_pid: int
    affinity: float
    gated: bool  # area-gate passed, as opposed to only area-released


def target_papers(
    pairs: dict[int, dict], departed_email: str, reviewers_per_paper: int
) -> tuple[list[int], list[tuple[int, int]]]:
    """(short pids, [(pid, slate size)] for the departed reviewer's other papers).

    A "short" paper is one where the departed reviewer was one of exactly
    `reviewers_per_paper` -- removing them drops it below target. A paper
    where they were one of more than that already has enough left over and
    needs no swap.
    """
    short, fine = [], []
    for pid, emails in pairs.items():
        if departed_email not in emails:
            continue
        if len(emails) == reviewers_per_paper:
            short.append(pid)
        else:
            fine.append((pid, len(emails)))
    return sorted(short), sorted(fine)


def restrict_to_requested_pids(
    short: list[int], fine: list[tuple[int, int]], requested: frozenset[int]
) -> tuple[list[int], list[tuple[int, int]], list[int], list[int]]:
    """Narrow (short, fine) to just `requested` pids -- a partial departure from
    named papers only, as opposed to every paper the reviewer holds.

    Returns (short, fine, ignored_short, not_held): `ignored_short` are the
    reviewer's own short papers left alone because they were not asked for
    (the reviewer keeps them, untouched); `not_held` are requested pids the
    reviewer does not actually hold at all (a typo guard). An empty
    `requested` is a no-op -- the original full-departure behavior.
    """
    if not requested:
        return short, fine, [], []
    held = set(short) | {pid for pid, _ in fine}
    not_held = sorted(requested - held)
    ignored_short = [pid for pid in short if pid not in requested]
    new_short = [pid for pid in short if pid in requested]
    new_fine = [(pid, n) for pid, n in fine if pid in requested]
    return new_short, new_fine, ignored_short, not_held


def movable_source_pairs(
    pairs: dict[int, dict], departed_email: str, donor_size: int
) -> list[tuple[int, str]]:
    """[(source pid, email)] for every reviewer on a paper holding exactly `donor_size`.

    Removing one of them leaves the donor paper at exactly `donor_size - 1`
    (== --reviewers-per-paper) -- never fewer, per the chair's own rule that a
    move must leave 5 behind.
    """
    out = []
    for pid, emails in pairs.items():
        if len(emails) != donor_size:
            continue
        for email in emails:
            if email != departed_email:
                out.append((pid, email))
    return out


def unsubmitted_pairs(
    pairs: list[tuple[int, str]],
    reviewers_by_email: dict,
    rid_by_pair: dict[tuple[int, str], int],
    submitted: set[int],
) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """Split `pairs` (roster-email keyed) into (not yet submitted, dropped) by the log.

    A pair with no matching review id in the log's own replay (should be rare
    -- only when --current-csv disagrees with the log, e.g. it came from
    HotCRP's direct export rather than the reconstruction) can't be verified
    as unsubmitted, so it is dropped and reported rather than assumed safe.
    """
    keep, dropped = [], []
    for pid, email in pairs:
        hotcrp_email = reviewers_by_email[email].hotcrp_email.lower() if email in reviewers_by_email else email.lower()
        rid = rid_by_pair.get((pid, hotcrp_email))
        if rid is None or rid in submitted:
            dropped.append((pid, email))
        else:
            keep.append((pid, email))
    return keep, dropped


def dedupe_by_email(candidates: list[Candidate]) -> list[Candidate]:
    """One `Candidate` per unique email, keeping the best (gated, affinity)
    source when the same person is feasible off more than one donor paper --
    a mover is one ask regardless of which of their papers would donate the
    slot. Sorted best-first.
    """
    best_per_email: dict[str, Candidate] = {}
    for c in candidates:
        held = best_per_email.get(c.email)
        if held is None or (c.gated, c.affinity) > (held.gated, held.affinity):
            best_per_email[c.email] = c
    return sorted(best_per_email.values(), key=lambda c: (c.gated, c.affinity), reverse=True)


def assign_unique_picks(
    feasible_by_pid: dict[int, list[Candidate]], top_n: int
) -> dict[int, list[Candidate]]:
    """Up to `top_n` picks per paper, no reviewer named twice across the whole batch.

    A reviewer who says yes to two different papers' asks ends up overloaded,
    so this is a global greedy assignment, not `top_n` independent per-paper
    rankings: every (paper, candidate) pair across every paper is pooled and
    walked best-affinity-first, each paper takes up to `top_n` and each
    reviewer at most one slot anywhere. A candidate who is any paper's best
    match still gets first claim on it -- they are simply removed from every
    other paper's pool once taken, so the next-best *distinct* person is who
    shows up there instead. `feasible_by_pid` values must already be
    deduped by email (see `dedupe_by_email`) and sorted best-first.
    """
    pool = [(pid, c) for pid, candidates in feasible_by_pid.items() for c in candidates]
    pool.sort(key=lambda pc: (pc[1].gated, pc[1].affinity), reverse=True)
    used_emails: set[str] = set()
    picks: dict[int, list[Candidate]] = {pid: [] for pid in feasible_by_pid}
    for pid, c in pool:
        if len(picks[pid]) >= top_n or c.email in used_emails:
            continue
        picks[pid].append(c)
        used_emails.add(c.email)
    return picks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--departed-email", required=True, help="reviewer who is leaving the committee")
    parser.add_argument("--departed-pids", type=parse_exclude_pids, default=frozenset(),
                         help="comma-separated paper IDs to reassign, restricting which of the departed "
                              "reviewer's papers are touched (default: every paper they hold, i.e. a full departure)")
    parser.add_argument("--current-csv", default=DEFAULT_CURRENT_CSV, help="what HotCRP currently holds (--hotcrp-csv or --pairs-csv shape)")
    parser.add_argument("--log", default=DEFAULT_LOG, help="HotCRP action log, for each candidate's submitted/not-submitted status")
    parser.add_argument("--pairs-csv", metavar="PATH", help="write the proposed picks (target_pid,rank,add_email,source_pid,remove_email,affinity)")
    parser.add_argument("--data", default=ar.DEFAULT_DATA, help="path to the HotCRP paper export JSON")
    parser.add_argument("--paper-policy", choices=PAPER_POLICIES, default="registered", help="paper selection policy (default: registered)")
    parser.add_argument("--exclude-pids", type=parse_exclude_pids, default=frozenset(), help="comma-separated paper IDs to exclude regardless of policy (default: none)")
    parser.add_argument("--csv", default=ar.DEFAULT_CSV, help="path to the reviewer CSV")
    parser.add_argument("--cap-overrides", default=DEFAULT_CAP_OVERRIDES, help="hand-maintained per-reviewer paper-cap override CSV")
    parser.add_argument("--pcinfo", default=pc_membership.DEFAULT_PCINFO, help="HotCRP user export deciding who is still on the PC")
    parser.add_argument("--no-pc-check", action="store_true", help="keep everyone the roster lists, even if HotCRP no longer marks them pc")
    parser.add_argument("--fingerprint-cache", default=ar.DEFAULT_FINGERPRINT_CACHE, help="path to the reviewer fingerprint cache")
    parser.add_argument("--paper-cache", default=ar.DEFAULT_PAPER_CACHE, help="path to the writable paper fingerprint cache")
    parser.add_argument("--reviewers-per-paper", type=int, default=ar.DEFAULT_REVIEWERS_PER_PAPER, help="target slate size; a donor paper must hold exactly this plus one (default: %(default)s)")
    parser.add_argument("--light-cap", type=int, default=7, help="max papers per light PC member (default: 7)")
    parser.add_argument("--full-cap", type=int, default=15, help="max papers per full PC member (default: 15)")
    parser.add_argument("--include-reserves", action="store_true", help="add the reserve reviewers to the pool")
    parser.add_argument("--reserve-cap", type=int, default=ar.DEFAULT_RESERVE_CAP, help=f"max papers per reserve reviewer (default: {ar.DEFAULT_RESERVE_CAP})")
    parser.add_argument("--reserve-info", default=DEFAULT_RESERVE_INFO, help="reserve roster for --include-reserves")
    parser.add_argument("--reserve-fingerprint-cache", default=cache_path("reserve_fingerprints.json"), help="reserve fingerprint cache for --include-reserves")
    parser.add_argument("--reserve-seniority", default=report_path("reserve_seniority.csv"), help="reserve seniority CSV for --include-reserves")
    parser.add_argument("--area-weight", type=float, default=1.0, help="weight of the topics document relative to title+abstract (default: 1.0)")
    parser.add_argument("--no-area-gate", action="store_true", help="skip the hard area-eligibility gate; every candidate ranks as gated")
    parser.add_argument("--seniority", default=DEFAULT_SENIORITY, help="reviewer seniority CSV from classify_reviewers.py")
    parser.add_argument("--min-seniors", type=int, default=1, help="senior reviewers each paper should keep, on both ends of a move (default: %(default)s)")
    parser.add_argument("--max-juniors", type=int, default=1, help="max junior reviewers per paper before relaxation (default: %(default)s)")
    parser.add_argument("--max-out-of-area", type=int, default=3, help="max out-of-area reviewers per paper before relaxation (default: %(default)s)")
    parser.add_argument("--same-country-cap", type=int, default=ar.DEFAULT_SAME_COUNTRY_CAP, metavar="N", help="max reviewers from a paper's majority-author country (default: %(default)s)")
    parser.add_argument("--no-same-country-cap", action="store_true", help="disable the same-country cap entirely")
    parser.add_argument("--region-majority", type=float, default=ar.DEFAULT_REGION_MAJORITY, help="share of placed authors that must share a country for its cap to bind (default: %(default)s)")
    parser.add_argument("--region-min-resolved", type=float, default=ar.DEFAULT_REGION_MIN_RESOLVED, help="share of authors whose country must be known before a paper is judged (default: %(default)s)")
    parser.add_argument("--affiliation-countries", default=affiliation_country.DEFAULT_COUNTRIES, help="hand-maintained affiliation -> country file")
    parser.add_argument("--no-coauthor-coi", action="store_true", help="disable the derived co-author COI")
    parser.add_argument("--coauthor-years", type=int, default=coauthor_coi.DEFAULT_COAUTHOR_YEARS, help="calendar years of co-authorship that conflict (default: %(default)s)")
    parser.add_argument("--no-collaborator-coi", action="store_true", help="disable the derived declared-collaborator COI")
    parser.add_argument("--area-chair-csv", default=DEFAULT_AREA_CHAIR_CSV, help="area-chair acceptance form")
    parser.add_argument("--no-area-chair-exclusion", action="store_true", help="do not exclude area chairs from the candidate pool")
    parser.add_argument("--coauthor-cache", default=coauthor_coi.DEFAULT_COAUTHORS, help="co-author cache from make dblp-snapshot")
    parser.add_argument("--author-names-cache", default=coauthor_coi.DEFAULT_AUTHOR_NAMES, help="DBLP name spellings of submission authors")
    parser.add_argument("--no-coauthor-identity", action="store_true", help="ignore DBLP's homonym numbering in the co-author COI")
    parser.add_argument("--kind", default="primary", help="review kind the log/current-csv are read for (default: %(default)s)")
    parser.add_argument("--round", default="R1", help="review round the log/current-csv are read for (default: %(default)s)")
    parser.add_argument("--device", default="cuda", help="torch device for SPECTER2 (default: cuda)")
    args = parser.parse_args()

    if args.reviewers_per_paper <= 0:
        parser.error("--reviewers-per-paper must be positive")
    if args.light_cap < 0 or args.full_cap < 0 or args.reserve_cap < 0:
        parser.error("--light-cap, --full-cap and --reserve-cap must be non-negative")
    if args.min_seniors < 0 or args.max_juniors < 0 or args.max_out_of_area < 0:
        parser.error("--min-seniors, --max-juniors, and --max-out-of-area must be non-negative")
    if args.same_country_cap < 0:
        parser.error("--same-country-cap must be non-negative")
    if args.coauthor_years <= 0:
        parser.error("--coauthor-years must be greater than 0")

    try:
        seniority = load_seniority(args.seniority)
    except FileNotFoundError:
        print(f"{args.seniority} not found — run classify_reviewers.py first", file=sys.stderr)
        return 1

    old_fmt, current_pairs_raw = assignment_io.load_assignment_pairs(args.current_csv)

    papers, _skipped = load_papers(args.data, paper_policy=args.paper_policy, exclude_pids=args.exclude_pids, with_skipped=True)
    if not papers:
        print(f"No papers found in {args.data}", file=sys.stderr)
        return 1
    papers_by_pid = {p["pid"]: p for p in papers}

    pcinfo = None if args.no_pc_check else args.pcinfo
    reviewers_by_email = {
        r.email: r for r in load_reviewers(args.csv, pcinfo_path=pcinfo, cap_overrides_path=args.cap_overrides)
    }

    reserves = []
    if pcinfo:
        reserves = load_reserve_reviewers(args.reserve_info, args.data, pcinfo_path=pcinfo, cap_overrides_path=args.cap_overrides)
    elif args.include_reserves:
        reserves = load_reserve_reviewers(args.reserve_info, args.data, pcinfo_path=None, cap_overrides_path=args.cap_overrides)

    promoted, _already_pc, true_reserves = ar.split_promoted_reserves(reserves, reviewers_by_email)
    merged = promoted + (true_reserves if args.include_reserves else [])
    reserve_fp: dict = {}
    if merged:
        reserve_fp = fp.load_fingerprint_cache(args.reserve_fingerprint_cache)
        reserve_seniority = {}
        try:
            reserve_seniority = load_seniority(args.reserve_seniority)
        except FileNotFoundError:
            reserve_seniority = {}
        reviewers_by_email.update({r.email: r for r in merged})
        seniority.update(reserve_seniority)

    if not args.no_area_chair_exclusion:
        index = None
        try:
            index = pc_membership.load_pc_accounts(args.pcinfo)
        except (FileNotFoundError, ValueError):
            index = None
        try:
            chair_emails = area_chair_emails(args.area_chair_csv, pcinfo_path=args.pcinfo if index else None)
        except FileNotFoundError:
            parser.error(f"{args.area_chair_csv}: not found; pass --no-area-chair-exclusion to skip this check")
        drop_area_chairs(reviewers_by_email, chair_emails, index)

    email_map = assignment_io.hotcrp_email_map(reviewers_by_email)
    hotcrp_to_roster = {hotcrp: roster for roster, hotcrp in email_map.items()}
    if old_fmt == "hotcrp":
        current_pairs_raw = assignment_io.remap_pairs(current_pairs_raw, hotcrp_to_roster)
    current_pairs = {pid: set(emails) for pid, emails in current_pairs_raw.items()}
    departed_email = hotcrp_to_roster.get(args.departed_email, args.departed_email)

    donor_size = args.reviewers_per_paper + 1
    short_pids, fine_pids = target_papers(current_pairs, departed_email, args.reviewers_per_paper)
    short_pids, fine_pids, ignored_short, not_held = restrict_to_requested_pids(short_pids, fine_pids, args.departed_pids)
    if not_held:
        print(f"WARNING: {departed_email} is not currently assigned to paper(s) {not_held}; ignoring.", file=sys.stderr)
    if ignored_short:
        print(f"{len(ignored_short)} other paper(s) {departed_email} holds are left alone (not in "
              f"--departed-pids), kept as-is: {ignored_short}", file=sys.stderr)
    if fine_pids:
        print(f"{len(fine_pids)} of {departed_email}'s other paper(s) already hold enough reviewers, no swap needed: "
              f"{', '.join(f'[{pid}] {n}' for pid, n in fine_pids)}", file=sys.stderr)
    if not short_pids:
        print(f"No papers left short by {departed_email}.", file=sys.stderr)
        return 0
    missing_pids = [pid for pid in short_pids if pid not in papers_by_pid]
    if missing_pids:
        print(f"{len(missing_pids)} short paper(s) are not in the current paper selection, skipping: "
              f"{', '.join(map(str, missing_pids))}", file=sys.stderr)
        short_pids = [pid for pid in short_pids if pid in papers_by_pid]
    if not short_pids:
        return 0
    short_papers = [papers_by_pid[pid] for pid in short_pids]

    raw_movable = movable_source_pairs(current_pairs, departed_email, donor_size)
    print(f"{len(raw_movable)} reviewer-paper pair(s) on a {donor_size}-reviewer paper are candidate donors "
          f"across {len({pid for pid, _ in raw_movable})} paper(s).", file=sys.stderr)

    rows = hotcrp_log.load_log(args.log)
    live, anomalies = hotcrp_log.replay_assignments(rows)
    if anomalies:
        print(f"{len(anomalies)} log replay anomalies (see hotcrp_log.replay_assignments); continuing.", file=sys.stderr)
    submitted = hotcrp_log.submitted_review_ids(rows)
    rid_by_pair = {}
    for rid, review in live.items():
        if args.kind != "all" and review.kind != args.kind:
            continue
        if args.round != "all" and review.round != args.round:
            continue
        rid_by_pair[(review.pid, review.email)] = rid

    movable, dropped_submitted = unsubmitted_pairs(raw_movable, reviewers_by_email, rid_by_pair, submitted)
    if dropped_submitted:
        print(f"{len(dropped_submitted)} candidate donor pair(s) dropped: review already submitted or "
              f"not verifiable in the log.", file=sys.stderr)
    if not movable:
        print("No unsubmitted donor reviews available.", file=sys.stderr)
        return 0

    candidate_emails_all = sorted({email for _pid, email in movable if email in reviewers_by_email})

    fp_cache = fp.load_fingerprint_cache(args.fingerprint_cache)
    fp_cache.update(reserve_fp)
    candidate_emails_all = [e for e in candidate_emails_all if e in fp_cache]
    candidate_matrix_all = np.array([fp_cache[e]["vector"] for e in candidate_emails_all], dtype=np.float32)
    email_index = {e: i for i, e in enumerate(candidate_emails_all)}

    paper_cache = fp.load_fingerprint_cache(args.paper_cache)
    build_paper_fingerprints(short_papers, paper_cache, args.paper_cache, area_weight=args.area_weight, device=args.device)

    derived: dict = {}
    if not args.no_coauthor_coi:
        try:
            coauthors = coauthor_coi.load_coauthors(args.coauthor_cache)
            author_names = coauthor_coi.load_author_names(args.author_names_cache)
        except FileNotFoundError as exc:
            parser.error(f"{exc.filename}: not found; run `make dblp-snapshot`, "
                         f"or pass --no-coauthor-coi to propose without this check")
        with open(args.data, encoding="utf-8") as f:
            all_papers = json.load(f)
        roster = list(reviewers_by_email.values())
        coauthor_index = coauthor_coi.build_index(roster, coauthors, years=args.coauthor_years)
        derived = coauthor_coi.derive_conflicts(short_papers, coauthor_index, author_names, all_papers, use_identity=not args.no_coauthor_identity)

    collab_hard: dict = {}
    if not args.no_collaborator_coi:
        try:
            pcinfo_index = pc_membership.load_pc_accounts(args.pcinfo)
        except (FileNotFoundError, ValueError) as exc:
            parser.error(f"{exc}; needed for the declared-collaborator COI check, "
                         f"or pass --no-collaborator-coi to propose without it")
        collab_profiles = collaborator_coi.build_index(list(reviewers_by_email.values()), pcinfo_index)
        collab_derived = collaborator_coi.derive_conflicts(short_papers, collab_profiles, pcinfo_index)
        collab_hard = collaborator_coi.hard_conflicts(collab_derived)

    countries = []
    if not args.no_same_country_cap:
        try:
            country_pcinfo = pc_membership.load_pc_accounts(args.pcinfo)
        except (FileNotFoundError, ValueError):
            country_pcinfo = None
        layers = affiliation_country.load_layers(args.affiliation_countries, pcinfo_index=country_pcinfo)
        all_relevant = sorted(set(candidate_emails_all) | {e for pid in short_pids for e in current_pairs[pid]})
        countries, _reviewer_country, _coverage, _thin = ar.build_country_caps(
            short_papers, all_relevant, reviewers_by_email, args.same_country_cap, layers,
            majority=args.region_majority, min_resolved=args.region_min_resolved,
        )
    country_cap_by_pid = {pid: c for c in countries for pid in c.papers}

    pools, missing = ar.seniority_pools(
        set(reviewers_by_email), seniority, almost_senior_window=10, almost_junior_pubs=15, almost_out_of_area_career=5,
    )
    if missing:
        print(f"{len(missing)} reviewer(s) not in {args.seniority} — treated as neither senior, junior, nor out-of-area.", file=sys.stderr)

    feasible_by_pid: dict[int, list[Candidate]] = {}
    zero_feasible = []
    for pid in short_pids:
        paper = papers_by_pid[pid]
        remaining = current_pairs[pid] - {departed_email}
        movers_here = [(src, email) for src, email in movable if email not in remaining]
        cand_emails = sorted({email for _src, email in movers_here if email in email_index})
        if not cand_emails:
            zero_feasible.append(pid)
            continue
        idx = [email_index[e] for e in cand_emails]
        matrix = candidate_matrix_all[idx]
        paper_vec = np.array(paper_cache[str(pid)]["vector"], dtype=np.float32)
        extra = set(derived.get(pid, ())) | collab_hard.get(pid, set())
        gated_scores = dict(eligible_scores(paper, cand_emails, matrix, paper_vec, reviewers_by_email,
                                             area_gate=not args.no_area_gate, extra_conflicts=extra))
        released_scores = dict(eligible_scores(paper, cand_emails, matrix, paper_vec, reviewers_by_email,
                                                area_gate=False, extra_conflicts=extra))

        cap = country_cap_by_pid.get(pid)
        target_seniors_now = sum(1 for e in remaining if e in pools.seniors)
        target_juniors_now = sum(1 for e in remaining if e in pools.juniors)
        target_oob_now = sum(1 for e in remaining if e in pools.out_of_area)

        feasible: list[Candidate] = []
        for src, email in movers_here:
            if email not in released_scores:
                continue
            if cap is not None and email in cap.members:
                current_country_count = sum(1 for e in remaining if e in cap.members)
                if current_country_count >= cap.cap:
                    continue
            new_seniors = target_seniors_now + (1 if email in pools.seniors else 0)
            if new_seniors < args.min_seniors:
                continue
            new_juniors = target_juniors_now + (1 if email in pools.juniors else 0)
            if new_juniors > args.max_juniors:
                continue
            new_oob = target_oob_now + (1 if email in pools.out_of_area else 0)
            if new_oob > args.max_out_of_area:
                continue
            source_slate = current_pairs[src]
            source_seniors_after = sum(1 for e in source_slate if e in pools.seniors and e != email)
            if source_seniors_after < args.min_seniors:
                continue
            feasible.append(Candidate(email, src, released_scores[email], email in gated_scores))
        if not feasible:
            zero_feasible.append(pid)
        print(f"[{pid}] {len(feasible)} feasible candidate(s) across {len(movers_here)} donor pair(s) considered.",
              file=sys.stderr)
        feasible_by_pid[pid] = dedupe_by_email(feasible)

    if zero_feasible:
        print(f"WARNING: {len(zero_feasible)} short paper(s) have zero feasible candidates: {zero_feasible}", file=sys.stderr)

    # Global, not per-paper: the same person must never be handed to two
    # papers as a pick, or a chair who gets two "yes"es back overloads them.
    movable_by_target = assign_unique_picks(feasible_by_pid, TOP_N)
    short_of_top_n = [
        pid for pid in short_pids
        if pid not in zero_feasible and len(movable_by_target[pid]) < TOP_N
    ]
    if short_of_top_n:
        print(f"WARNING: {len(short_of_top_n)} short paper(s) got fewer than {TOP_N} unique picks "
              f"(their best matches were claimed by another short paper first): {short_of_top_n}", file=sys.stderr)

    all_rows: list[tuple[int, int, str, int, str, float]] = []
    for pid in short_pids:
        paper = papers_by_pid[pid]
        print(f"\n=== [{pid}] {paper['title']}")
        picks = movable_by_target[pid]
        if not picks:
            print("    *** NO FEASIBLE CANDIDATE FOUND ***")
            continue
        for rank, c in enumerate(picks, start=1):
            label = "primary" if rank == 1 else "backup"
            r = reviewers_by_email[c.email]
            src_title = papers_by_pid.get(c.source_pid, {}).get("title", "")
            area = "gated" if c.gated else "released"
            print(f"    {label:7} {c.affinity:.3f}  {r.name} <{c.email}>  [{r.tier}]  ({area})  "
                  f"-- move off [{c.source_pid}] {src_title}")
            all_rows.append((pid, rank, c.email, c.source_pid, c.email, c.affinity))

    total_papers = len(short_pids)
    total_with_pick = sum(1 for pid in short_pids if movable_by_target[pid])
    total_picks = sum(len(v) for v in movable_by_target.values())
    print(
        f"\nDone. {total_picks} proposed pick(s) across {total_with_pick} of {total_papers} short paper(s) "
        f"({len(zero_feasible)} with zero feasible candidates — read above). Nothing here is applied; "
        f"this is a proposal to ask reviewers about and act on by hand.",
        file=sys.stderr,
    )

    if args.pairs_csv:
        try:
            with open(args.pairs_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["target_pid", "rank", "add_email", "source_pid", "remove_email", "affinity"])
                for target_pid, rank, add_email, source_pid, remove_email, affinity in all_rows:
                    writer.writerow([target_pid, rank, add_email, source_pid, remove_email, f"{affinity:.6f}"])
        except OSError as exc:
            print(f"ERROR: could not write pairs CSV {args.pairs_csv}: {exc}", file=sys.stderr)
            return 1

    return 1 if zero_feasible else 0


if __name__ == "__main__":
    sys.exit(main())
