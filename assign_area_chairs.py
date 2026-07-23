"""Assign reviewer-assigned papers evenly among accepted area chairs.

    python assign_area_chairs.py > area_chair_assignment.txt

Every paper that has at least one reviewer in assignment.txt receives exactly
one non-conflicted area chair. The assignment maximizes total SPECTER2 cosine
affinity subject to a per-chair load within --load-tolerance of the mean.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass

import numpy as np

import fingerprint as fp
from area_chairs import AreaChair, load_area_chairs
from paper_matching import build_paper_fingerprints, load_papers

DEFAULT_CSV = "Area Chair Acceptance Form (Responses) - Form Responses 1.csv"
DEFAULT_ASSIGNMENT = "assignment.txt"
DEFAULT_DATA = "hpca2027-data.json"
DEFAULT_FINGERPRINT_CACHE = "area_chair_fingerprints.json"
DEFAULT_PAPER_CACHE = "paper_fingerprints.json"

_PAPER_HEADER_RE = re.compile(r"^=== \[(\d+)\]")
_ASSIGNED_RE = re.compile(r"^    assigned (\d+) of")


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
        raise ValueError(
            f"load tolerance {tolerance:.3f} gives infeasible integer bounds "
            f"{lower}..{upper} for {n_papers} papers and {n_chairs} chairs"
        )
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=DEFAULT_CSV, help="area-chair acceptance CSV")
    parser.add_argument("--reviewer-assignment", default=DEFAULT_ASSIGNMENT)
    parser.add_argument("--data", default=DEFAULT_DATA, help="HotCRP paper JSON")
    parser.add_argument("--fingerprint-cache", default=DEFAULT_FINGERPRINT_CACHE)
    parser.add_argument("--paper-cache", default=DEFAULT_PAPER_CACHE)
    parser.add_argument("--load-tolerance", type=float, default=0.10)
    parser.add_argument("--area-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 <= args.load_tolerance < 1:
        parser.error("--load-tolerance must be at least 0 and less than 1")
    if args.area_weight <= 0:
        parser.error("--area-weight must be greater than 0")

    try:
        assigned_pids = load_reviewer_assigned_pids(args.reviewer_assignment)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if not assigned_pids:
        print(f"ERROR: no reviewer-assigned papers found in {args.reviewer_assignment}", file=sys.stderr)
        return 1

    papers_by_pid = {p["pid"]: p for p in load_papers(args.data)}
    unknown = [pid for pid in assigned_pids if pid not in papers_by_pid]
    if unknown:
        print(
            f"ERROR: reviewer-assigned paper [{unknown[0]}] is missing, incomplete, or withdrawn in {args.data}",
            file=sys.stderr,
        )
        return 1
    papers = [papers_by_pid[pid] for pid in assigned_pids]

    chairs = load_area_chairs(args.csv)
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
    scores: dict[tuple[int, str], float] = {}
    conflict_pairs = 0
    for paper in papers:
        pid = paper["pid"]
        paper_vec = np.array(paper_cache[str(pid)]["vector"], dtype=np.float32)
        similarities = chair_matrix @ paper_vec
        conflicts = {email.lower() for email in paper.get("pc_conflicts", {})}
        for email, score in zip(chair_emails, similarities):
            if email in conflicts:
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
    print(
        f"Papers: {len(papers)}  Chairs: {len(chair_emails)}  "
        f"Load bounds: {lower}..{upper} (±{args.load_tolerance:.0%})"
    )
    for paper in papers:
        pid = paper["pid"]
        email = assignment[pid]
        score = scores[(pid, email)]
        assigned_scores.append(score)
        by_chair[email].append(pid)

    for email in sorted(chair_emails, key=lambda e: (chairs_by_email[e].name.lower(), e)):
        chair = chairs_by_email[email]
        pids = sorted(by_chair[email])
        mean_score = sum(scores[(pid, email)] for pid in pids) / len(pids)
        print(f"\n=== {chair.name} <{email}>")
        print(
            f"    primary area: {chair.primary}\n"
            f"    assigned {loads[email]} papers; mean affinity {mean_score:.3f}"
        )
        for pid in pids:
            paper = papers_by_pid[pid]
            print(f"  [{pid}] {scores[(pid, email)]:.3f}  {paper['title']}")
            print(f"        topics: {', '.join(paper.get('topics', []))}")

    assigned_conflicts = sum(
        assignment[p["pid"]] in {email.lower() for email in p.get("pc_conflicts", {})}
        for p in papers
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
        f"loads {min(loads.values())}..{max(loads.values())}, "
        f"mean affinity {sum(assigned_scores) / len(assigned_scores):.3f}.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
