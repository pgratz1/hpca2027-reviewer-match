"""Assign reviewer-assigned papers evenly among accepted area chairs.

    python -m scripts.assign_area_chairs > outputs/assignments/area_chair_assignment.txt

Every selected paper in assignment.txt receives exactly one non-conflicted
area chair. The default selection is the "registered" policy (non-withdrawn
papers with real content); use --paper-policy submitted once HotCRP has
submitted papers, or complete for the older completeness filter. Pass the
same policy assign_reviewers.py used. The assignment maximizes total SPECTER2
cosine affinity under balanced loads.

Conflicts are the same three layers the reviewer matcher applies — declared,
own-paper, and derived co-authorship — and each one is hard here, because a
missing edge is a route the flow simply cannot take. With only fifteen chairs
that can make the load bounds infeasible where the reviewer matcher would just
under-fill; the ValueError says so rather than dropping a conflict to fit.
"""

from __future__ import annotations

from reviewer_match.paths import assignment_path, cache_path, curated_path, input_path, report_path

import argparse
import csv
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from reviewer_match import coauthor_coi
from reviewer_match import collaborator_coi
from reviewer_match import fingerprint as fp
from reviewer_match import pc_membership
from reviewer_match.area_chairs import AreaChair
from reviewer_match.roster import load_roster
from reviewer_match.paper_matching import (
    PAPER_POLICIES,
    build_paper_fingerprints,
    load_papers,
    own_paper_conflicts,
    parse_exclude_pids,
)

DEFAULT_CSV = input_path("Area Chair Acceptance Form (Responses) - Form Responses 1.csv")
DEFAULT_ASSIGNMENT = assignment_path("assignment.txt")
DEFAULT_DATA = input_path("hpca2027-data.json")
DEFAULT_FINGERPRINT_CACHE = cache_path("area_chair_fingerprints.json")
DEFAULT_PAPER_CACHE = cache_path("paper_fingerprints.json")

# Reruns reassign papers as the matching stabilizes, so write_paper_tag_csv
# clears every track tag up to this ceiling before reapplying the current
# set -- the same "clear then reinstall" idiom assign_reviewers.py already
# uses for R1 reviews. Cleared unconditionally, not just this run's chair
# count, so a shrunk chair roster doesn't strand a previous run's
# higher-numbered tags on papers that moved away from them.
DEFAULT_TRACK_CLEAR_CEILING = 50

_PAPER_HEADER_RE = re.compile(r"^=== \[(\d+)\]")
_ASSIGNED_RE = re.compile(r"^    assigned (\d+) of")


def chair_conflicts(papers, chairs, args, parser) -> dict[int, set[str]]:
    """{paper pid: conflicted chair emails}, from all three COI layers.

    `pc_conflicts` alone is not enough and never was: it is what the authors
    declared, and a paper nobody swept declares nothing — so a chair who wrote
    the paper could be handed it to chair. `own_paper_conflicts` closes that,
    exactly as it does for reviewers, and the derived co-author layer closes
    the case where a chair wrote a *different* paper with one of its authors.

    Fifteen chairs is a thin pool, so the load bounds can become infeasible
    here where the reviewer matcher would merely under-fill. That surfaces as
    the ValueError from `maximize_balanced_affinity`, which is the honest
    outcome: a chair assignment that quietly kept a conflict would be worse.
    """
    conflicts = {
        paper["pid"]: (
            {email.lower() for email in (paper.get("pc_conflicts") or {})}
            | own_paper_conflicts(paper)
        )
        for paper in papers
    }
    if args.no_coauthor_coi:
        return conflicts

    try:
        coauthors = coauthor_coi.load_coauthors(args.coauthor_cache)
        author_names = coauthor_coi.load_author_names(args.author_names_cache)
    except FileNotFoundError as exc:
        parser.error(f"{exc.filename}: not found; run `make dblp-snapshot`, "
                     f"or pass --no-coauthor-coi to assign without this check")
    with open(args.data, encoding="utf-8") as f:
        all_papers = json.load(f)

    index = coauthor_coi.build_index(chairs, coauthors, years=args.coauthor_years)
    derived = coauthor_coi.derive_conflicts(
        papers, index, author_names, all_papers,
        use_identity=not args.no_coauthor_identity,
    )
    total = sum(len(found) for found in derived.values())
    new = sum(1 for found in derived.values() for c in found.values() if not c.declared)
    print(f"Co-author COI: {total} chair-paper pair(s) excluded over "
          f"{len(derived)} paper(s), {new} of them declared nowhere",
          file=sys.stderr)
    if index.uncovered:
        print(f"  {len(index.uncovered)} chair(s) have no co-author data and "
              f"pass this layer silently", file=sys.stderr)
    for pid, found in derived.items():
        conflicts[pid] |= set(found)

    if not args.no_collaborator_coi:
        try:
            pcinfo_index = pc_membership.load_pc_accounts(args.pcinfo)
        except (FileNotFoundError, ValueError) as exc:
            parser.error(f"{exc}; needed for the declared-collaborator COI check, "
                         f"or pass --no-collaborator-coi to assign without it")
        collab_profiles = collaborator_coi.build_index(chairs, pcinfo_index)
        collab_derived = collaborator_coi.derive_conflicts(papers, collab_profiles, pcinfo_index)
        # Only the name-matched subset excludes; affiliation overlap is
        # reported (make collaborator-coi) but too coarse to hard-block on
        # -- see collaborator_coi's module docstring.
        collab_hard = collaborator_coi.hard_conflicts(collab_derived)
        collab_total = sum(len(found) for found in collab_hard.values())
        print(f"Declared-collaborator COI: {collab_total} chair-paper pair(s) "
              f"excluded (name match) over {len(collab_hard)} paper(s)", file=sys.stderr)
        for pid, found in collab_hard.items():
            conflicts[pid] |= found

    return conflicts


def load_reviewer_assigned_pids(path: str) -> list[int]:
    """Paper IDs whose assignment.txt section reports at least one reviewer."""
    sections: dict[int, int | None] = {}
    current: int | None = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            header = _PAPER_HEADER_RE.match(line)
            if header:
                current = int(header.group(1))
                if current in sections:
                    raise ValueError(f"{path}: duplicate paper section [{current}]")
                sections[current] = None
                continue
            assigned = _ASSIGNED_RE.match(line)
            if assigned and current is not None:
                if sections[current] is not None:
                    raise ValueError(f"{path}: duplicate assigned count for paper [{current}]")
                sections[current] = int(assigned.group(1))
    missing = [pid for pid, count in sections.items() if count is None]
    if missing:
        raise ValueError(f"{path}: paper [{missing[0]}] has no assigned-count line")
    return [pid for pid, count in sections.items() if count and count > 0]


def load_bounds(n_papers: int, n_chairs: int, tolerance: float) -> tuple[int, int]:
    if n_chairs <= 0:
        raise ValueError("at least one accepted area chair is required")
    mean = n_papers / n_chairs
    lower = math.ceil((1.0 - tolerance) * mean)
    upper = math.floor((1.0 + tolerance) * mean)
    if lower * n_chairs > n_papers or upper * n_chairs < n_papers:
        lower, upper = math.floor(mean), math.ceil(mean)
    return lower, upper


@dataclass
class _Edge:
    to: int
    rev: int
    cap: int
    cost: float
    original_cap: int


def _add_edge(graph: list[list[_Edge]], src: int, dst: int, cap: int, cost: float) -> _Edge:
    forward = _Edge(dst, len(graph[dst]), cap, cost, cap)
    reverse = _Edge(src, len(graph[src]), 0, -cost, 0)
    graph[src].append(forward)
    graph[dst].append(reverse)
    return forward


def maximize_balanced_affinity(
    pids: list[int],
    chair_emails: list[str],
    scores: dict[tuple[int, str], float],
    lower: int,
    upper: int,
) -> dict[int, str]:
    """Exact maximum-sum assignment with uniform lower/upper chair loads."""
    if lower < 0 or upper < lower:
        raise ValueError("invalid load bounds")
    pids = sorted(pids)
    chair_emails = sorted(chair_emails)
    n_papers, n_chairs = len(pids), len(chair_emails)
    if lower * n_chairs > n_papers or upper * n_chairs < n_papers:
        raise ValueError("load bounds cannot cover all papers")

    source = 0
    paper_start = 1
    chair_start = paper_start + n_papers
    sink = chair_start + n_chairs
    graph: list[list[_Edge]] = [[] for _ in range(sink + 1)]
    assignment_edges: dict[tuple[int, str], _Edge] = {}

    values = list(scores.values())
    score_span = (max(values) - min(values)) if values else 0.0
    required_bonus = (score_span + 1.0) * (n_papers + 1)

    for i, pid in enumerate(pids):
        paper_node = paper_start + i
        _add_edge(graph, source, paper_node, 1, 0.0)
        for j, email in enumerate(chair_emails):
            score = scores.get((pid, email))
            if score is None:
                continue
            edge = _add_edge(graph, paper_node, chair_start + j, 1, -score)
            assignment_edges[(pid, email)] = edge

    for j in range(n_chairs):
        chair_node = chair_start + j
        if lower:
            _add_edge(graph, chair_node, sink, lower, -required_bonus)
        if upper > lower:
            _add_edge(graph, chair_node, sink, upper - lower, 0.0)

    flow = 0
    node_count = len(graph)
    while flow < n_papers:
        dist = [math.inf] * node_count
        previous: list[tuple[int, int] | None] = [None] * node_count
        in_queue = [False] * node_count
        dist[source] = 0.0
        queue = deque([source])
        in_queue[source] = True

        while queue:
            node = queue.popleft()
            in_queue[node] = False
            for edge_index, edge in enumerate(graph[node]):
                if edge.cap <= 0:
                    continue
                candidate = dist[node] + edge.cost
                if candidate < dist[edge.to] - 1e-12:
                    dist[edge.to] = candidate
                    previous[edge.to] = (node, edge_index)
                    if not in_queue[edge.to]:
                        queue.append(edge.to)
                        in_queue[edge.to] = True

        if previous[sink] is None:
            raise ValueError("conflicts and load bounds make a complete assignment impossible")
        node = sink
        while node != source:
            parent, edge_index = previous[node]
            edge = graph[parent][edge_index]
            edge.cap -= 1
            graph[node][edge.rev].cap += 1
            node = parent
        flow += 1

    result: dict[int, str] = {}
    for (pid, email), edge in assignment_edges.items():
        if edge.original_cap == 1 and edge.cap == 0:
            if pid in result:
                raise RuntimeError(f"optimizer assigned paper {pid} more than once")
            result[pid] = email
    loads = Counter(result.values())
    if len(result) != n_papers:
        raise RuntimeError("optimizer returned an incomplete assignment")
    if any(loads[email] < lower or loads[email] > upper for email in chair_emails):
        raise ValueError("conflicts make the minimum chair loads infeasible")
    return result


def track_tag(n: int) -> str:
    """Chair index n's track tag -- the same `~~track_N` spelling in both

    of the two HotCRP namespaces this repo writes it into: the paper tag that
    *names* the track, and the chair's own PC-account grant that the track's
    permissions name. `~~`-prefixed, so hidden from ordinary PC in both
    namespaces -- including from the chair it names: HotCRP's
    track-membership check (`Conf::check_tracks`) is a raw, viewer-independent
    string test on the tag, so a hidden paper tag still grants the chair's
    reviewer-identity and comment permissions correctly. All that is lost is
    the chair's own `#track_N` search shortcut in the HotCRP UI;
    `outputs/assignments/area_chair_assignment.txt` already lists each
    chair's papers by ID and title, so that report is the replacement, not a
    gap. Paper tags and account tags are separate HotCRP namespaces, so the
    identical spelling never collides -- only the object it tags, and thus
    which side of Settings -> Tags & tracks reads it, differs.
    """
    return f"~~track_{n}"


def write_account_tag_csv(
    path: str,
    chairs_by_email: dict[str, AreaChair],
    track_number: dict[str, int],
    clear_through: int = DEFAULT_TRACK_CLEAR_CEILING,
) -> None:
    """Atomically write a HotCRP Settings -> Accounts bulk-tag CSV.

    Uses `remove_tags`/`add_tags`, never the plain `tags` column: HotCRP's
    bulk user importer treats a bare `tags` value as a full replacement of
    the account's entire tag set (wiping `~~area-chairs` and anything else
    unrelated), while `remove_tags`/`add_tags` touch only the named tags.
    `remove_tags` clears every track tag up to `clear_through` before
    `add_tags` grants the current one, so a chair whose track number shifted
    since the last run doesn't keep the stale tag too.

    Keyed on `chair.hotcrp_email`, not the roster's `email`: a chair matched
    under a different address than their acceptance-form row would otherwise
    get tagged on an account HotCRP doesn't consider `pc` at all.
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
            writer.writerow(["email", "remove_tags", "add_tags"])
            remove_tags = " ".join(track_tag(n) for n in range(clear_through))
            for email in sorted(track_number):
                hotcrp_email = chairs_by_email[email].hotcrp_email
                writer.writerow(
                    [hotcrp_email, remove_tags, track_tag(track_number[email])]
                )
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_paper_tag_csv(
    path: str,
    by_chair: dict[str, list[int]],
    track_number: dict[str, int],
    clear_through: int = DEFAULT_TRACK_CLEAR_CEILING,
) -> None:
    """Atomically write a HotCRP Assignments bulk-update CSV clearing every

    track tag up to `clear_through`, then tagging every paper into its
    chair's track -- a rerun-safe replacement, not an addition, so a paper
    that moved to a different chair since the last run doesn't keep both tags.
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
            writer.writerow(["paper", "action", "email", "tag", "round"])
            for n in range(clear_through):
                writer.writerow(["all", "cleartag", "", track_tag(n), ""])
            for email in sorted(by_chair):
                tag = track_tag(track_number[email])
                for pid in sorted(by_chair[email]):
                    writer.writerow([pid, "tag", "", tag, ""])
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=DEFAULT_CSV, help="area-chair acceptance CSV")
    parser.add_argument("--reviewer-assignment", default=DEFAULT_ASSIGNMENT)
    parser.add_argument("--data", default=DEFAULT_DATA, help="HotCRP paper JSON")
    parser.add_argument(
        "--paper-policy", choices=PAPER_POLICIES, default="registered",
        help="paper selection policy (default: registered)",
    )
    parser.add_argument(
        "--exclude-pids", type=parse_exclude_pids, default=frozenset(),
        help="comma-separated paper IDs to exclude regardless of policy, "
             "e.g. a known test submission (default: none); must match "
             "assign_reviewers.py's setting or the reviewer-assignment "
             "cross-check fails",
    )
    parser.add_argument("--fingerprint-cache", default=DEFAULT_FINGERPRINT_CACHE)
    parser.add_argument("--paper-cache", default=DEFAULT_PAPER_CACHE)
    parser.add_argument(
        "--load-tolerance", type=float, default=0.0,
        help="fractional chair-load tolerance (default: 0, closest even split)",
    )
    parser.add_argument("--area-weight", type=float, default=1.0)
    parser.add_argument(
        "--no-coauthor-coi", action="store_true",
        help="disable the derived co-author COI (on by default, as for reviewers)"
    )
    parser.add_argument(
        "--no-collaborator-coi", action="store_true",
        help="disable the derived declared-collaborator COI (on by default, as for reviewers)"
    )
    parser.add_argument(
        "--coauthor-years", type=int, default=coauthor_coi.DEFAULT_COAUTHOR_YEARS,
        help="calendar years of co-authorship that conflict (default: %(default)s)"
    )
    parser.add_argument("--coauthor-cache", default=coauthor_coi.DEFAULT_COAUTHORS)
    parser.add_argument("--author-names-cache", default=coauthor_coi.DEFAULT_AUTHOR_NAMES)
    parser.add_argument(
        "--no-coauthor-identity", action="store_true",
        help="ignore DBLP's homonym numbering (see assign_reviewers.py)"
    )
    parser.add_argument(
        "--pcinfo", default=pc_membership.DEFAULT_PCINFO,
        help="HotCRP user export; also the source of the `~~area-chairs` tag, "
             "which adds chairs the acceptance form never collected "
             "(default: %(default)s)"
    )
    parser.add_argument(
        "--no-pc-check", action="store_true",
        help="skip the HotCRP membership check, for an export staler than the "
             "rosters. Also disables tag-sourced chairs, since both read the export"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--account-tag-csv",
        help="write a HotCRP Settings->Accounts bulk-tag CSV granting each chair their ~~track_N",
    )
    parser.add_argument(
        "--paper-tag-csv",
        help="write a HotCRP Assignments bulk-update CSV tagging each paper into its chair's ~~track_N",
    )
    parser.add_argument(
        "--track-clear-ceiling", type=int, default=DEFAULT_TRACK_CLEAR_CEILING,
        help="clear ~~track_0..~~track_(N-1) from papers and from chair accounts, "
             "before reapplying; covers a shrunk chair roster too (default: %(default)s)",
    )
    args = parser.parse_args()
    if not 0 <= args.load_tolerance < 1:
        parser.error("--load-tolerance must be at least 0 and less than 1")
    if args.area_weight <= 0:
        parser.error("--area-weight must be greater than 0")
    if args.coauthor_years <= 0:
        parser.error("--coauthor-years must be greater than 0")

    try:
        assigned_pids = load_reviewer_assigned_pids(args.reviewer_assignment)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if not assigned_pids:
        print(f"ERROR: no reviewer-assigned papers found in {args.reviewer_assignment}", file=sys.stderr)
        return 1

    selected_papers = load_papers(
        args.data, paper_policy=args.paper_policy, exclude_pids=args.exclude_pids
    )
    papers_by_pid = {p["pid"]: p for p in selected_papers}
    unknown = [pid for pid in assigned_pids if pid not in papers_by_pid]
    if unknown:
        print(
            f"ERROR: reviewer-assigned paper [{unknown[0]}] is excluded by "
            f"the {args.paper_policy} policy in {args.data}",
            file=sys.stderr,
        )
        return 1
    if args.paper_policy == "submitted":
        missing = sorted(set(papers_by_pid) - set(assigned_pids))
        if missing:
            print(
                f"ERROR: submitted paper [{missing[0]}] has no assigned reviewers in "
                f"{args.reviewer_assignment}",
                file=sys.stderr,
            )
            return 1
    papers = [papers_by_pid[pid] for pid in assigned_pids]

    # Through load_roster, not load_area_chairs, so the chair pool here is the
    # same one enrichment and fingerprinting built: the form plus any account
    # HotCRP tags `~~area-chairs` whose profile another roster can supply.
    try:
        chairs = load_roster(
            "area-chair", args.csv, args.data,
            pcinfo_path=None if args.no_pc_check else args.pcinfo,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    chairs_by_email: dict[str, AreaChair] = {chair.email: chair for chair in chairs}
    chair_cache = fp.load_fingerprint_cache(args.fingerprint_cache)
    chair_emails = sorted(email for email in chairs_by_email if email in chair_cache)
    missing_chairs = [chair.email for chair in chairs if chair.email not in chair_cache]
    if missing_chairs:
        print(
            f"ERROR: {len(missing_chairs)} accepted chair(s) lack fingerprints; "
            "run make area-chairs so enrichment and fingerprinting happen first",
            file=sys.stderr,
        )
        return 1
    if not chair_emails:
        print("ERROR: no accepted area chairs with fingerprints", file=sys.stderr)
        return 1

    paper_cache = fp.load_fingerprint_cache(args.paper_cache)
    build_paper_fingerprints(
        papers, paper_cache, args.paper_cache,
        area_weight=args.area_weight, device=args.device,
    )
    chair_matrix = np.array(
        [chair_cache[email]["vector"] for email in chair_emails], dtype=np.float32
    )
    conflicts_by_pid = chair_conflicts(papers, chairs, args, parser)
    scores: dict[tuple[int, str], float] = {}
    conflict_pairs = 0
    for paper in papers:
        pid = paper["pid"]
        paper_vec = np.array(paper_cache[str(pid)]["vector"], dtype=np.float32)
        similarities = chair_matrix @ paper_vec
        conflicts = conflicts_by_pid[pid]
        for email, score in zip(chair_emails, similarities):
            # pc_conflicts is declared against the chair's HotCRP account
            # address, which may differ from the roster key -- see the same
            # check in paper_matching.eligible_scores.
            if email in conflicts or chairs_by_email[email].hotcrp_email.lower() in conflicts:
                conflict_pairs += 1
                continue
            scores[(pid, email)] = float(score)

    try:
        lower, upper = load_bounds(len(papers), len(chair_emails), args.load_tolerance)
        assignment = maximize_balanced_affinity(
            assigned_pids, chair_emails, scores, lower, upper
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    loads = Counter(assignment.values())
    by_chair: dict[str, list[int]] = defaultdict(list)
    assigned_scores = []
    papers_by_pid = {paper["pid"]: paper for paper in papers}
    print("AREA CHAIR ASSIGNMENT")
    mean_load = len(papers) / len(chair_emails)
    strict_lower = math.ceil((1.0 - args.load_tolerance) * mean_load)
    strict_upper = math.floor((1.0 + args.load_tolerance) * mean_load)
    closest_balance = (lower, upper) != (strict_lower, strict_upper)
    balance_label = (
        f"closest integer balance; requested ±{args.load_tolerance:.0%}"
        if closest_balance else f"±{args.load_tolerance:.0%}"
    )
    print(
        f"Papers: {len(papers)}  Chairs: {len(chair_emails)}  "
        f"Load bounds: {lower}..{upper} ({balance_label})"
    )
    for paper in papers:
        pid = paper["pid"]
        email = assignment[pid]
        score = scores[(pid, email)]
        assigned_scores.append(score)
        by_chair[email].append(pid)

    ordered_chairs = sorted(chair_emails, key=lambda e: (chairs_by_email[e].name.lower(), e))
    track_number = {email: i for i, email in enumerate(ordered_chairs)}

    for email in ordered_chairs:
        chair = chairs_by_email[email]
        pids = sorted(by_chair[email])
        mean_score = (
            sum(scores[(pid, email)] for pid in pids) / len(pids)
            if pids else None
        )
        mean_label = "n/a" if mean_score is None else f"{mean_score:.3f}"
        print(f"\n=== {chair.name} <{email}>")
        print(
            f"    primary area: {chair.primary}\n"
            f"    track: {track_tag(track_number[email])}\n"
            f"    assigned {loads[email]} papers; mean affinity {mean_label}"
        )
        for pid in pids:
            paper = papers_by_pid[pid]
            print(f"  [{pid}] {scores[(pid, email)]:.3f}  {paper['title']}")
            print(f"        topics: {', '.join(paper.get('topics', []))}")

    assigned_conflicts = sum(
        assignment[p["pid"]] in conflicts_by_pid[p["pid"]] for p in papers
    )
    duplicate_or_missing = len(papers) - len(assignment)
    out_of_bounds = sum(
        not lower <= loads[email] <= upper for email in chair_emails
    )
    print("\n=== SELF-CHECK")
    print(f"Total affinity: {sum(assigned_scores):.3f}")
    print(f"Mean affinity: {sum(assigned_scores) / len(assigned_scores):.3f}")
    print(f"Eligible conflict edges excluded: {conflict_pairs}")
    print(f"Conflicted assignments: {assigned_conflicts} (should be 0)")
    print(f"Duplicate or missing paper assignments: {duplicate_or_missing} (should be 0)")
    print(f"Chairs outside load bounds: {out_of_bounds} (should be 0)")

    print(
        f"Done. Assigned {len(papers)} papers to {len(chair_emails)} area chairs; "
        f"loads {min(loads[email] for email in chair_emails)}.."
        f"{max(loads[email] for email in chair_emails)}, "
        f"mean affinity {sum(assigned_scores) / len(assigned_scores):.3f}.",
        file=sys.stderr,
    )

    clear_through = max(len(ordered_chairs), args.track_clear_ceiling)
    if args.account_tag_csv:
        try:
            write_account_tag_csv(args.account_tag_csv, chairs_by_email, track_number, clear_through)
        except OSError as exc:
            print(f"ERROR: could not write account tag CSV {args.account_tag_csv}: {exc}", file=sys.stderr)
            return 1
    if args.paper_tag_csv:
        try:
            write_paper_tag_csv(args.paper_tag_csv, by_chair, track_number, clear_through)
        except OSError as exc:
            print(f"ERROR: could not write paper tag CSV {args.paper_tag_csv}: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
