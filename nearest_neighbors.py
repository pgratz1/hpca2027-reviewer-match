"""Print each reviewer's top-N most similar other reviewers by fingerprint.

    python nearest_neighbors.py                          # every reviewer
    python nearest_neighbors.py --email someone@example.com # just one
    python nearest_neighbors.py --top 10

Similarity is the dot product between L2-normalized SPECTER2 fingerprints
(fingerprints.json, built by build_fingerprints.py), i.e. plain cosine
similarity. Candidates are the unique emails in the fingerprint cache, which
sidesteps the CSV's occasional duplicate rows (some reviewers resubmitted
the acceptance form and appear 2-3x under the same email).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import fingerprint as fp
from reviewers import load_reviewers

DEFAULT_CSV = "HPCA'27 PC Member Acceptance Form (Responses) - Form Responses 1.csv"
DEFAULT_FINGERPRINT_CACHE = "fingerprints.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=DEFAULT_CSV, help="path to the reviewer CSV")
    parser.add_argument(
        "--fingerprint-cache", default=DEFAULT_FINGERPRINT_CACHE, help="path to the fingerprint cache"
    )
    parser.add_argument("--top", type=int, default=5, help="number of neighbors to print per reviewer (default: 5)")
    parser.add_argument("--email", default=None, help="only print neighbors for this one reviewer (default: all)")
    args = parser.parse_args()

    if args.top < 0:
        parser.error("--top must be non-negative")
    if args.email is not None:
        args.email = args.email.strip().lower()

    fp_cache = fp.load_fingerprint_cache(args.fingerprint_cache)
    if not fp_cache:
        print(f"No fingerprints found in {args.fingerprint_cache}", file=sys.stderr)
        return 1

    reviewers_by_email = {r.email: r for r in load_reviewers(args.csv)}

    emails = list(fp_cache.keys())
    index_by_email = {e: i for i, e in enumerate(emails)}
    matrix = np.array([fp_cache[e]["vector"] for e in emails], dtype=np.float32)

    if args.email is not None:
        if args.email not in index_by_email:
            print(f"No fingerprint for {args.email!r}", file=sys.stderr)
            return 1
        targets = [args.email]
    else:
        targets = emails

    def label(email: str) -> str:
        r = reviewers_by_email.get(email)
        return f"{r.name} <{email}> ({r.primary})" if r else email

    for email in targets:
        idx = index_by_email[email]
        sims = matrix @ matrix[idx]
        order = [i for i in np.argsort(-sims) if i != idx][: args.top]
        print(f"\n=== {label(email)}")
        for rank, i in enumerate(order, 1):
            print(f"  {rank}. {sims[i]:.3f}  {label(emails[i])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
