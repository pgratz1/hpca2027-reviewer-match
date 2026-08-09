"""Assign each accepted TRC member their top-N most similar eligible papers.

The TRC (Training Review Committee) is a cohort of PhD students reviewing
alongside the PC. Unlike the main reviewer assignment (scripts/assign_reviewers.py),
this is not a load-capped deferred-acceptance match: every TRC member is
guaranteed the same target (`--papers-per-student`, default 2), ranked purely
by SPECTER2 cosine similarity to their own declared area/keywords + DBLP
publications, with no seniority classes, no same-country cap, and no
relaxation ladder. The only cross-student constraint is `--paper-cap`
(default 1): once a paper has that many TRC reviewers, it is unavailable to
anyone else, so a single greedy pass over every (student, paper) pair by
descending score -- not a stable/optimal assignment, just documented as such
-- replaces the six-phase machinery the main assignment needs.

COI matches the main reviewer pipeline in full: declared `pc_conflicts`,
`own_paper_conflicts` (authorship), derived co-author COI, and derived
declared-collaborator COI. Two more layers apply here only:

  * a student inherits their *entire* advisor's conflict set (all four
    layers above, computed for the advisor as if they were a candidate
    reviewer) -- conservative by explicit design, not just the advisor's own
    authorship;
  * no paper the TRC chair (`--chair-email`) is conflicted with (by the same
    four layers) is ever assigned to anyone, computed once up front.

One more constraint sits outside COI entirely: a student is never assigned a
paper their advisor is *already reviewing* on the main PC/reserve slate, per
`--assignment-csv` (default `outputs/assignments/assignment.csv`, the
`--hotcrp-csv` `make` already produces -- **assumed already built**; a
missing file is a hard error, like every other required-upstream-artifact
check in this pipeline, unless `--no-advisor-review-check` is passed). Not a
conflict -- the advisor has no tie to the paper itself -- but a student's
review should be independent of whatever their advisor privately thinks of
it, which reviewing the same paper the advisor already reviews would risk.

Requires a pre-built TRC fingerprint cache (`make trc`, or
`build_fingerprints.py --role trc`) -- this script only builds paper
fingerprints inline, never reviewer ones, the same separation
`assign_reviewers.py` keeps.

Also writes `--missing-tag-csv`: a HotCRP Settings -> Accounts bulk-tag
upload (`email,remove_tags,add_tags`) for every accepted TRC member whose
HotCRP account isn't yet tagged `trc` -- the accepted-but-untagged half of
the cross-check `trc_reviewers.load_trc_members` already warns about on
every run. Uses `add_tags`, never a bare `tags` column, so it only grants
`trc` and never touches an account's other tags.

Writes three HotCRP-uploadable CSVs and a human-readable report to stdout.
`--review-csv` and `--tag-csv` are both wipe-then-reinstall, not purely
additive: each opens with a `clearreview`/`cleartag` row scoped to `all`
papers before re-adding the current slate, so a paper that drops out of the
TRC track on a rerun (roster change, updated fingerprints, a shifted main
assignment) actually loses its `TRC-track` tag and R1-style review row
instead of keeping a stale one forever.

    python -m scripts.assign_trc_reviews > outputs/assignments/trc_assignment.txt
    python -m scripts.assign_trc_reviews --paper-policy submitted --exclude-pids 1152
    python -m scripts.assign_trc_reviews --paper-cap 0   # no cap on TRC reviewers per paper
"""

from __future__ import annotations

from reviewer_match.paths import assignment_path, cache_path, input_path

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict

import numpy as np

from reviewer_match import coauthor_coi, collaborator_coi, fingerprint as fp, pc_membership
from reviewer_match.paper_matching import (
    build_paper_fingerprints,
    load_papers,
    own_paper_conflicts,
    parse_exclude_pids,
)
from reviewer_match.reviewers import Reviewer, load_reviewers
from reviewer_match.roster import DEFAULT_CSV as DEFAULT_PC_CSV
from reviewer_match.trc_reviewers import DEFAULT_TRC_CSV, load_trc_members, missing_tag_emails
from scripts.generate_clear_uploads import atomic_rows

DEFAULT_DATA = input_path("hpca2027-data.json")
DEFAULT_FINGERPRINT_CACHE = cache_path("trc_fingerprints.json")
DEFAULT_PAPER_CACHE = cache_path("paper_fingerprints.json")
DEFAULT_REVIEW_CSV = assignment_path("trc_review_upload.csv")
DEFAULT_TAG_CSV = assignment_path("trc_track_tags.csv")
DEFAULT_MISSING_TAG_CSV = assignment_path("trc_missing_tags.csv")
DEFAULT_MAIN_ASSIGNMENT_CSV = assignment_path("assignment.csv")

DEFAULT_PAPERS_PER_STUDENT = 2
DEFAULT_PAPER_CAP = 1
DEFAULT_CHAIR_EMAIL = "jsanmiguel@wisc.edu"
DEFAULT_ROUND = "TRC"
DEFAULT_ACTION = "externalreview"
DEFAULT_TRACK_TAG = "TRC-track"


def person_conflict_pids(
    emails: set[str],
    pids: list[int],
    own_by_pid: dict[int, set[str]],
    pcc_by_pid: dict[int, set[str]],
    coauthor_derived: dict[int, dict[str, object]],
    collab_hard: dict[int, set[str]],
) -> set[int]:
    """Every paper pid this identity (any of `emails`) is conflicted with.

    Shared by the TRC chair's global exclusion and each student's own and
    advisor-inherited exclusion sets -- one computation, fed by whichever
    address set the caller passes. `emails` covers every known address for
    one identity (see `resolve_advisor`), since authorship/declared
    conflicts can land under either one.
    """
    emails = {e.lower() for e in emails}
    if not emails:
        return set()
    empty: set[str] = set()
    return {
        pid for pid in pids
        if own_by_pid.get(pid, empty) & emails
        or pcc_by_pid.get(pid, empty) & emails
        or coauthor_derived.get(pid, {}).keys() & emails
        or collab_hard.get(pid, empty) & emails
    }


def load_reviewer_assignments(path: str) -> dict[str, set[int]]:
    """{lowercased hotcrp email: {assigned paper pid, ...}} from a `--hotcrp-csv` upload.

    Reads the `paper,action,email,round` shape `assign_reviewers.write_hotcrp_csv`
    writes (and `make`'s `assignment.csv`/`ASSIGNMENT_CSV` is exactly that).
    The leading wipe row (`paper == "all"`) is skipped; every other row is one
    real "this email reviews this paper" pair, matched on `paper`/`email`
    alone so the exact `action` spelling (always `primaryreview` today) is
    never a second thing this has to stay in sync with.
    """
    assigned: dict[str, set[int]] = defaultdict(set)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid_cell = (row.get("paper") or "").strip()
            email = (row.get("email") or "").strip().lower()
            if not pid_cell or pid_cell == "all" or not email or email == "all":
                continue
            try:
                pid = int(pid_cell)
            except ValueError:
                continue
            assigned[email].add(pid)
    return assigned


def eligible_pids_for_student(
    student_email: str,
    advisor_addrs: set[str],
    usable_pids: list[int],
    own_by_pid: dict[int, set[str]],
    pcc_by_pid: dict[int, set[str]],
    coauthor_derived: dict[int, dict[str, object]],
    collab_hard: dict[int, set[str]],
    topics_by_pid: dict[int, set[str]],
    primary: str,
    secondary: str,
    *,
    area_gate: bool = True,
    advisor_coi: bool = True,
    target: int = DEFAULT_PAPERS_PER_STUDENT,
    advisor_assigned_pids: frozenset[int] = frozenset(),
) -> tuple[list[int], bool]:
    """(candidate pids for this student, whether the area gate was relaxed).

    Excludes the student's own conflicts and -- unless `advisor_coi` is
    False -- the union of every conflict any resolved advisor address
    carries (see `person_conflict_pids`). Also excludes `advisor_assigned_pids`
    unconditionally: papers the advisor is already reviewing on the main
    slate are not a conflict, but a TRC review should stand on its own,
    independent of whatever the advisor already thinks of that paper. Both
    exclusions are hard, the same as COI everywhere else in this pipeline --
    never restored by the area-gate relaxation below. Area-gates on
    `primary`/`secondary` against each paper's topics the same way
    `paper_matching.eligible_scores` does; if that strict pool has fewer
    than `target` candidates, the gate is dropped for this student only and
    `relaxed=True` is returned, so a narrow declared interest doesn't leave
    them under-filled when the COI-eligible pool could still cover them.
    """
    excluded = person_conflict_pids(
        {student_email}, usable_pids, own_by_pid, pcc_by_pid, coauthor_derived, collab_hard
    )
    if advisor_coi and advisor_addrs:
        excluded |= person_conflict_pids(
            advisor_addrs, usable_pids, own_by_pid, pcc_by_pid, coauthor_derived, collab_hard
        )
    excluded |= advisor_assigned_pids

    candidate_pids = [pid for pid in usable_pids if pid not in excluded]
    if not area_gate:
        return candidate_pids, False

    areas = {a.lower() for a in (primary, secondary) if a}
    strict_pids = [pid for pid in candidate_pids if areas & topics_by_pid.get(pid, set())]
    if len(strict_pids) < target:
        return candidate_pids, True
    return strict_pids, False


def greedy_assign(
    triples: list[tuple[float, str, int]], papers_per_student: int, paper_cap: int
) -> tuple[dict[str, list[tuple[int, float]]], "Counter[int]"]:
    """Greedily match (score, email, pid) triples under a per-paper cap.

    Not a stable or optimal assignment -- a simple, deterministic pass over
    every candidate pair sorted by descending score (ties broken on pid then
    email, so the result never depends on dict/set iteration order): a pair
    is taken whenever the student still has room (`papers_per_student`) and
    the paper hasn't hit its cap (`paper_cap`, 0 = unlimited). Appropriate at
    this scale (tens of students, a paper cap of ~1) where a full
    deferred-acceptance match would be overkill.
    """
    ordered = sorted(triples, key=lambda t: (-t[0], t[2], t[1]))
    picks: dict[str, list[tuple[int, float]]] = defaultdict(list)
    paper_load: Counter = Counter()
    for score, email, pid in ordered:
        if len(picks[email]) >= papers_per_student:
            continue
        if paper_cap and paper_load[pid] >= paper_cap:
            continue
        picks[email].append((pid, score))
        paper_load[pid] += 1
    return picks, paper_load


def resolve_advisor(
    declared_email: str,
    pc_reviewers_by_email: dict[str, Reviewer],
    pc_index: pc_membership.PcIndex | None,
) -> tuple[Reviewer | None, set[str]]:
    """(this cycle's PC Reviewer for this advisor, if any; every known address).

    Advisors are almost always PC members/faculty, so the PC roster is where
    a DBLP pid comes from for the derived-COI layers. The declared TRC-form
    address does not have to be the address they hold their PC account
    under -- accepting from one and authoring under another is ordinary
    here (see pc_membership's module docstring) -- so a `pc_index` match is
    tried too. When no PC Reviewer is found at all, the advisor still
    contributes their authorship/declared-conflict layers under whichever
    address(es) were found; only the derived layers are unavailable.
    """
    declared_email = declared_email.lower()
    addrs = {declared_email}
    reviewer = pc_reviewers_by_email.get(declared_email)
    if reviewer is None and pc_index is not None:
        acct, _how = pc_index.match(declared_email)
        if acct is not None:
            addrs.add(acct.email.lower())
            reviewer = pc_reviewers_by_email.get(acct.email.lower())
    if reviewer is not None:
        addrs.add(reviewer.email.lower())
        addrs.add(reviewer.hotcrp_email.lower())
    return reviewer, addrs


def build_tag_csv_rows(distinct_pids: list[int], track_tag: str) -> list[list]:
    """Rows for the `--tag-csv` upload: wipe `track_tag` off every paper first.

    Rerun-safe, not purely additive -- a plain list of `tag` rows for the
    current `distinct_pids` would leave `track_tag` on a paper that dropped
    out of the TRC track on a later run (roster change, updated
    fingerprints, a shifted main assignment) forever, since nothing would
    ever ask HotCRP to remove it. The leading `["all", "cleartag", ...]` row
    is the same wipe-then-reinstall idiom `--review-csv`'s `clearreview` row
    already uses.
    """
    return [["all", "cleartag", "", track_tag, ""]] + [
        [pid, "tag", "", track_tag, ""] for pid in distinct_pids
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--trc-csv", default=DEFAULT_TRC_CSV, help="TRC acceptance-form CSV")
    parser.add_argument(
        "--pc-csv", default=DEFAULT_PC_CSV,
        help="PC acceptance-form CSV, for advisor pid/address resolution only",
    )
    parser.add_argument("--data", default=DEFAULT_DATA, help="HotCRP paper export")
    parser.add_argument(
        "--paper-policy", choices=("registered", "submitted", "complete"), default="registered"
    )
    parser.add_argument("--exclude-pids", default="", help="comma-separated paper IDs to skip")
    parser.add_argument("--fingerprint-cache", default=DEFAULT_FINGERPRINT_CACHE)
    parser.add_argument("--paper-cache", default=DEFAULT_PAPER_CACHE)
    parser.add_argument("--pcinfo", default=pc_membership.DEFAULT_PCINFO)
    parser.add_argument(
        "--papers-per-student", type=int, default=DEFAULT_PAPERS_PER_STUDENT,
        help=f"papers guaranteed to each TRC member (default: {DEFAULT_PAPERS_PER_STUDENT})",
    )
    parser.add_argument(
        "--paper-cap", type=int, default=DEFAULT_PAPER_CAP,
        help=f"max TRC reviewers per paper, 0 = unlimited (default: {DEFAULT_PAPER_CAP})",
    )
    parser.add_argument("--area-gate", dest="area_gate", action="store_true", default=True)
    parser.add_argument("--no-area-gate", dest="area_gate", action="store_false")
    parser.add_argument("--area-weight", type=float, default=1.0)
    parser.add_argument("--coauthor-years", type=int, default=coauthor_coi.DEFAULT_COAUTHOR_YEARS)
    parser.add_argument("--coauthor-cache", default=coauthor_coi.DEFAULT_COAUTHORS)
    parser.add_argument("--author-names-cache", default=coauthor_coi.DEFAULT_AUTHOR_NAMES)
    parser.add_argument("--no-coauthor-coi", action="store_true")
    parser.add_argument("--no-collaborator-coi", action="store_true")
    parser.add_argument(
        "--no-advisor-coi", action="store_true",
        help="diagnostic only: do not exclude papers via an advisor's conflicts",
    )
    parser.add_argument(
        "--assignment-csv", default=DEFAULT_MAIN_ASSIGNMENT_CSV,
        help="the main PC/reserve --hotcrp-csv assignment, assumed already built by `make`",
    )
    parser.add_argument(
        "--no-advisor-review-check", action="store_true",
        help="do not exclude papers an advisor is already reviewing on the main slate",
    )
    parser.add_argument(
        "--chair-email", default=DEFAULT_CHAIR_EMAIL,
        help="no paper this person is conflicted with ever enters the TRC track; empty disables",
    )
    parser.add_argument("--action", default=DEFAULT_ACTION, help="HotCRP review-assignment action")
    parser.add_argument("--round", dest="review_round", default=DEFAULT_ROUND)
    parser.add_argument("--track-tag", default=DEFAULT_TRACK_TAG)
    parser.add_argument("--review-csv", default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--tag-csv", default=DEFAULT_TAG_CSV)
    parser.add_argument(
        "--missing-tag-csv", default=DEFAULT_MISSING_TAG_CSV,
        help="HotCRP account bulk-tag upload for accepted TRC members not yet tagged `trc`",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.papers_per_student <= 0:
        parser.error("--papers-per-student must be greater than 0")
    if args.paper_cap < 0:
        parser.error("--paper-cap must be non-negative")
    if args.coauthor_years <= 0:
        parser.error("--coauthor-years must be greater than 0")

    exclude_pids = parse_exclude_pids(args.exclude_pids)
    papers = load_papers(args.data, paper_policy=args.paper_policy, exclude_pids=exclude_pids)
    if not papers:
        print(f"No eligible papers found in {args.data}", file=sys.stderr)
        return 1
    pids_all = [p["pid"] for p in papers]
    title_by_pid = {p["pid"]: p.get("title") or "" for p in papers}
    topics_by_pid = {p["pid"]: {t.lower() for t in p.get("topics", [])} for p in papers}

    paper_cache = fp.load_fingerprint_cache(args.paper_cache)
    build_paper_fingerprints(
        papers, paper_cache, args.paper_cache, area_weight=args.area_weight, device=args.device
    )
    paper_matrix = np.array(
        [paper_cache[str(pid)]["vector"] for pid in pids_all], dtype=np.float32
    )
    vector_by_pid = {pid: paper_matrix[i] for i, pid in enumerate(pids_all)}

    students = load_trc_members(args.trc_csv, pcinfo_path=args.pcinfo)
    if not students:
        print(f"No accepted TRC members found in {args.trc_csv}", file=sys.stderr)
        return 1

    # Re-derives the same cross-check load_trc_members already warned about
    # (pc_membership.load_pc_accounts is cached by path/mtime/size, so this
    # costs a re-scan of the already-parsed export, not a re-read of it) --
    # simpler than threading the lists back out of the loader for the one
    # caller that wants to act on them rather than just read the warning.
    not_yet_tagged, _not_yet_accepted = missing_tag_emails(students, args.pcinfo)
    atomic_rows(
        args.missing_tag_csv,
        ["email", "remove_tags", "add_tags"],
        [[email, "", "trc"] for email in not_yet_tagged],
    )

    student_fp = fp.load_fingerprint_cache(args.fingerprint_cache)
    missing_fp = sorted(s.email for s in students if s.email not in student_fp)
    if missing_fp:
        print(
            f"WARNING: {len(missing_fp)} TRC member(s) have no fingerprint in "
            f"{args.fingerprint_cache} and cannot be assigned: "
            f"{', '.join(missing_fp[:5])}{' ...' if len(missing_fp) > 5 else ''} — "
            f"run `make trc`",
            file=sys.stderr,
        )
    scoreable = [s for s in students if s.email not in set(missing_fp)]

    # --- Advisor identity resolution -----------------------------------
    pc_index = None
    if args.pcinfo:
        try:
            pc_index = pc_membership.load_pc_accounts(args.pcinfo)
        except (FileNotFoundError, ValueError) as exc:
            if not args.no_collaborator_coi:
                parser.error(
                    f"{exc}; needed for the declared-collaborator COI check and "
                    f"advisor-address resolution, or pass --no-collaborator-coi"
                )
            print(
                f"WARNING: {exc}; advisor-address resolution degrades to the "
                f"declared address only",
                file=sys.stderr,
            )
    # Address resolution serves both the advisor-COI-inheritance layer and
    # the advisor-already-reviewing check below, so it runs whenever either
    # is wanted -- not just when the COI layer specifically is.
    need_advisor_addrs = not args.no_advisor_coi or not args.no_advisor_review_check
    pc_reviewers_by_email = {}
    if need_advisor_addrs:
        pc_reviewers_by_email = {
            r.email: r for r in load_reviewers(args.pc_csv, pcinfo_path=args.pcinfo)
        }

    advisor_addrs_by_student: dict[str, set[str]] = {}
    advisor_reviewers_by_email: dict[str, Reviewer] = {}
    advisor_notes: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
    if need_advisor_addrs:
        for s in scoreable:
            addrs: set[str] = set()
            for declared in s.advisor_emails:
                reviewer, found = resolve_advisor(declared, pc_reviewers_by_email, pc_index)
                addrs |= found
                advisor_notes[s.email].append((declared, reviewer.pid if reviewer else None))
                if reviewer is not None:
                    advisor_reviewers_by_email[reviewer.email] = reviewer
            advisor_addrs_by_student[s.email] = addrs

    # --- Advisor's own review load, from the already-built main slate ---
    main_assignments: dict[str, set[int]] = {}
    if not args.no_advisor_review_check:
        try:
            main_assignments = load_reviewer_assignments(args.assignment_csv)
        except FileNotFoundError:
            parser.error(
                f"{args.assignment_csv}: not found -- this check assumes `make` has "
                f"already been run to produce it, or pass --no-advisor-review-check "
                f"to assign without it"
            )

    # --- Combined identity list for the derived-COI layers --------------
    # One pass over the paper corpus for every identity that needs a
    # derived-conflict computation: TRC students, resolved advisors, and
    # the TRC chair -- not one pass per identity.
    combined_people: list[Reviewer] = list(scoreable) + list(advisor_reviewers_by_email.values())
    chair_email = (args.chair_email or "").strip().lower()
    if chair_email:
        combined_people.append(
            Reviewer(
                email=chair_email, first="", last="", dblp_url="", pid=None,
                affiliation="", primary="", secondary="", tertiary="", keywords="",
                tier="chair", override_cap=None, hotcrp_email=chair_email,
            )
        )

    coauthor_derived: dict[int, dict[str, object]] = {}
    if not args.no_coauthor_coi:
        try:
            coauthors = coauthor_coi.load_coauthors(args.coauthor_cache)
            author_names = coauthor_coi.load_author_names(args.author_names_cache)
        except FileNotFoundError as exc:
            parser.error(
                f"{exc.filename}: not found; run `make dblp-snapshot`, or pass "
                f"--no-coauthor-coi to assign without this check"
            )
        with open(args.data, encoding="utf-8") as f:
            all_papers = json.load(f)
        coauthor_index = coauthor_coi.build_index(
            combined_people, coauthors, years=args.coauthor_years
        )
        coauthor_derived = coauthor_coi.derive_conflicts(
            papers, coauthor_index, author_names, all_papers
        )

    collab_hard: dict[int, set[str]] = {}
    if not args.no_collaborator_coi and pc_index is not None:
        collab_profiles = collaborator_coi.build_index(combined_people, pc_index)
        collab_full = collaborator_coi.derive_conflicts(papers, collab_profiles, pc_index)
        collab_hard = collaborator_coi.hard_conflicts(collab_full)

    own_by_pid = {p["pid"]: own_paper_conflicts(p) for p in papers}
    pcc_by_pid = {p["pid"]: {e.lower() for e in p.get("pc_conflicts", {})} for p in papers}

    # --- TRC chair's global exclusion, before any student is scored -----
    chair_pids: set[int] = set()
    if chair_email:
        chair_pids = person_conflict_pids(
            {chair_email}, pids_all, own_by_pid, pcc_by_pid, coauthor_derived, collab_hard
        )
    usable_pids = [pid for pid in pids_all if pid not in chair_pids]

    # --- Per-student eligible (paper, score) candidates ------------------
    triples: list[tuple[float, str, int]] = []
    relaxed_students: set[str] = set()
    advisor_review_overlap: dict[str, set[int]] = {}
    for s in scoreable:
        svec = np.array(student_fp[s.email]["vector"], dtype=np.float32)
        advisor_addrs = advisor_addrs_by_student.get(s.email, set())
        advisor_assigned_pids = frozenset().union(
            *(main_assignments.get(addr, set()) for addr in advisor_addrs)
        ) if advisor_addrs else frozenset()
        if advisor_assigned_pids:
            advisor_review_overlap[s.email] = advisor_assigned_pids & set(usable_pids)
        use_pids, relaxed = eligible_pids_for_student(
            s.email, advisor_addrs, usable_pids, own_by_pid, pcc_by_pid,
            coauthor_derived, collab_hard, topics_by_pid, s.primary, s.secondary,
            area_gate=args.area_gate, advisor_coi=not args.no_advisor_coi,
            target=args.papers_per_student, advisor_assigned_pids=advisor_assigned_pids,
        )
        if relaxed:
            relaxed_students.add(s.email)
        for pid in use_pids:
            score = float(vector_by_pid[pid] @ svec)
            triples.append((score, s.email, pid))

    # --- Global greedy matching, capped per paper ------------------------
    picks, _paper_load = greedy_assign(triples, args.papers_per_student, args.paper_cap)

    # --- Outputs -----------------------------------------------------
    distinct_pids = sorted({pid for pairs in picks.values() for pid, _ in pairs})

    review_rows = []
    for s in scoreable:
        for pid, score in sorted(picks.get(s.email, ()), key=lambda t: -t[1]):
            review_rows.append((pid, s.hotcrp_email))
    review_rows.sort(key=lambda r: (r[0], r[1]))
    atomic_rows(
        args.review_csv,
        ["paper", "action", "email", "round"],
        [["all", "clearreview", "all", args.review_round]]
        + [[pid, args.action, email, args.review_round] for pid, email in review_rows],
    )
    atomic_rows(
        args.tag_csv,
        ["paper", "action", "email", "tag", "round"],
        build_tag_csv_rows(distinct_pids, args.track_tag),
    )

    print(
        f"TRC review assignment: {len(students)} accepted TRC member(s), "
        f"{len(papers)} eligible paper(s) under --paper-policy {args.paper_policy}",
    )
    if chair_pids:
        print(
            f"{len(chair_pids)} paper(s) excluded from the TRC track entirely: "
            f"{chair_email} is conflicted with them"
        )
    if not_yet_tagged:
        print(
            f"{len(not_yet_tagged)} accepted TRC member(s) still need the `trc` "
            f"HotCRP tag — wrote {args.missing_tag_csv}"
        )

    for s in sorted(students, key=lambda s: s.email):
        print(f"\n=== {s.name} <{s.email}>")
        print(f"    areas: {s.primary} / {s.secondary} / {s.tertiary}")
        if s.advisor_emails:
            notes = advisor_notes.get(s.email, [(a, None) for a in s.advisor_emails])
            shown = ", ".join(
                f"{addr}{' (pid ' + pid + ')' if pid else ' (no PC-roster pid — authorship/declared conflicts only)'}"
                for addr, pid in notes
            )
            print(f"    advisor(s): {shown}")
        overlap = advisor_review_overlap.get(s.email)
        if overlap:
            print(
                f"    {len(overlap)} paper(s) excluded: advisor already reviewing "
                f"them on the main slate"
            )
        if s.email in missing_fp:
            print("    NO FINGERPRINT — run `make trc` first")
            continue
        if s.email in relaxed_students:
            print("    area gate relaxed: declared area/secondary too narrow against the COI-eligible pool")
        assigned = sorted(picks.get(s.email, ()), key=lambda t: -t[1])
        if len(assigned) < args.papers_per_student:
            print(f"    *** UNDER-FILLED: {len(assigned)} of {args.papers_per_student} ***")
        for rank, (pid, score) in enumerate(assigned, 1):
            print(f"  {rank}. {score:.3f}  [{pid}] {title_by_pid.get(pid, '')}")

    placed_full = sum(
        1 for s in scoreable if len(picks.get(s.email, ())) >= args.papers_per_student
    )
    overlap_students = len(advisor_review_overlap)
    print(
        f"\nDone. {placed_full} of {len(scoreable)} scoreable TRC member(s) placed "
        f"{args.papers_per_student} papers each; {len(missing_fp)} had no fingerprint; "
        f"{len(distinct_pids)} distinct paper(s) touched; {len(not_yet_tagged)} need "
        f"the `trc` tag; {overlap_students} student(s) had a paper excluded because "
        f"their advisor is already reviewing it; wrote {args.review_csv}, "
        f"{args.tag_csv} and {args.missing_tag_csv}.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
