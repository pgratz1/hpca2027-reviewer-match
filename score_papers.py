"""Fingerprint each paper and rank reviewers against it by SPECTER2 similarity.

    python score_papers.py                # every paper in hpca2027-data.json
    python score_papers.py --pid 8         # just one paper
    python score_papers.py --top 15

Each paper is fingerprinted the same way reviewers are (fingerprint.py,
build_fingerprints.py): one SPECTER2 document for title+abstract, one for
its declared topics, pooled and L2-normalized (fingerprint.pool). Results
are cached in paper_fingerprints.json, keyed by pid, so a paper already
cached isn't re-encoded. Reviewer candidates are then ranked by cosine
similarity against the reviewer fingerprints in fingerprints.json.

Two filters apply before ranking (not blended into the score):
  - COI: any reviewer whose email is a key in the paper's pc_conflicts is
    excluded outright (e.g. the paper's own authors).
  - Area gate: reviewer's primary/secondary area must overlap the paper's
    topics (case-insensitive — HotCRP's topic strings don't always match
    the CSV's casing, e.g. "Memory Systems" vs "Memory systems"), per the
    project brief's hard-gate design. Use --no-area-gate to rank the full
    non-conflicted pool instead.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import fingerprint as fp
from paper_matching import build_paper_fingerprints, eligible_scores, load_papers
from reviewers import load_reviewers

DEFAULT_CSV = "HPCA'27 PC Member Acceptance Form (Responses) - Form Responses 1.csv"
DEFAULT_DATA = "hpca2027-data.json"
DEFAULT_FINGERPRINT_CACHE = "fingerprints.json"
DEFAULT_PAPER_CACHE = "paper_fingerprints.json"


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
    parser.add_argument("--pid", type=int, default=None, help="only score this one paper (default: all)")
    parser.add_argument("--top", type=int, default=10, help="number of reviewers to print per paper (default: 10)")
    parser.add_argument(
        "--area-weight", type=float, default=1.0,
        help="weight of the topics document relative to the title+abstract document (default: 1.0). "
        "Note: with only 2 pooled documents per paper this gives topics roughly half the paper's "
        "fingerprint, proportionally much stronger than on the reviewer side where it's diluted "
        "among many titles."
    )
    parser.add_argument(
        "--no-area-gate", action="store_true",
        help="skip the hard area-eligibility gate; rank all non-conflicted reviewers"
    )
    parser.add_argument("--device", default="cuda", help="torch device for SPECTER2 (default: cuda)")
    args = parser.parse_args()

    if args.top < 0:
        parser.error("--top must be non-negative")
    if args.area_weight <= 0:
        parser.error("--area-weight must be greater than 0")

    papers = load_papers(args.data)
    if args.pid is not None:
        papers = [p for p in papers if p["pid"] == args.pid]
        if not papers:
            print(f"No paper with pid={args.pid} in {args.data}", file=sys.stderr)
            return 1

    # --- 1. Fingerprint any papers not already cached -----------------------
    paper_cache = fp.load_fingerprint_cache(args.paper_cache)
    build_paper_fingerprints(
        papers, paper_cache, args.paper_cache, area_weight=args.area_weight, device=args.device
    )

    # --- 2. Rank reviewers per paper -----------------------------------------
    reviewer_fp = fp.load_fingerprint_cache(args.fingerprint_cache)
    reviewers_by_email = {r.email: r for r in load_reviewers(args.csv)}

    candidate_emails = [e for e in reviewer_fp if e in reviewers_by_email]
    candidate_matrix = np.array([reviewer_fp[e]["vector"] for e in candidate_emails], dtype=np.float32)

    for p in papers:
        pid = p["pid"]
        paper_vec = np.array(paper_cache[str(pid)]["vector"], dtype=np.float32)
        scores = eligible_scores(
            p, candidate_emails, candidate_matrix, paper_vec, reviewers_by_email,
            area_gate=not args.no_area_gate,
        )

        print(f"\n=== [{pid}] {p['title']}")
        print(f"    topics: {', '.join(p.get('topics', []))}")
        if not scores:
            print("  (no eligible reviewers)")
            continue

        scores.sort(key=lambda es: -es[1])
        for rank, (email, score) in enumerate(scores[: args.top], 1):
            r = reviewers_by_email[email]
            print(f"  {rank:2d}. {score:.3f}  {r.name} <{email}>  [{r.tier}]  ({r.primary})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
