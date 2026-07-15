"""Build a SPECTER2 fingerprint (768-dim vector) for each accepted reviewer.

    python build_fingerprints.py --limit 10   # validate on a handful first
    python build_fingerprints.py              # full run

Each reviewer's fingerprint pools SPECTER2 embeddings of: one document per
DBLP title from the most recent --years calendar years (default: 4, no count
cap unless --max-titles is set) and one document summarizing their declared
areas + keywords (weighted --area-weight relative to a single title).
Reviewers with no DBLP PID, or no titles in the --years window, fall back to
an area-profile-only fingerprint.

Fingerprints are cached in fingerprints.json (keyed by email). A versioned
content key covers the reviewer metadata, selected titles, model, and policy
flags, so unchanged entries avoid model work while changes rebuild
automatically. A transient DBLP failure creates or retains an area-only
fallback marked for retry; it cannot become permanently "complete". DBLP
titles reuse the same caches as main.py (dblp_cache.json / dblp_pubs_cache.json).

As a sanity check, the script prints each spot-check reviewer's top-5 nearest
neighbors by fingerprint cosine similarity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

import numpy as np
import requests
import torch

import fingerprint as fp
import specter2_model
from dblp import fetch_titles_for_pids, load_cache, load_colleague_cache
from reviewers import load_reviewers

DEFAULT_CSV = "HPCA'27 PC Member Acceptance Form (Responses) - Form Responses 1.csv"
DEFAULT_CACHE = "dblp_cache.json"
DEFAULT_COLLEAGUE_CACHE = "dblp_pubs_cache.json"
DEFAULT_FINGERPRINT_CACHE = "fingerprints.json"
FINGERPRINT_SCHEMA_VERSION = 2

# Always spot-checked if present in the current run; falls back to the first
# few reviewers processed when empty.
SPOT_CHECK_EMAILS: list[str] = []


def fingerprint_key(r, titles, *, years, max_titles, area_weight) -> str:
    """Content and policy key for a reviewer fingerprint."""
    payload = {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "model": specter2_model.BASE_MODEL,
        "adapter": specter2_model.PROXIMITY_ADAPTER,
        "email": r.email,
        "pid": r.pid,
        "primary": r.primary,
        "secondary": r.secondary,
        "tertiary": r.tertiary,
        "keywords": r.keywords,
        "years": years,
        "max_titles": max_titles,
        "area_weight": area_weight,
        "titles": titles,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=DEFAULT_CSV, help="path to the reviewer CSV")
    parser.add_argument("--cache", default=DEFAULT_CACHE, help="path to the writable DBLP title cache")
    parser.add_argument(
        "--colleague-cache", default=DEFAULT_COLLEAGUE_CACHE,
        help="path to the colleague's read-only pre-built DBLP cache"
    )
    parser.add_argument(
        "--fingerprint-cache", default=DEFAULT_FINGERPRINT_CACHE,
        help="path to the writable fingerprint cache"
    )
    parser.add_argument(
        "--years", type=int, default=4,
        help="DBLP titles from the most recent N calendar years (default: 4)"
    )
    parser.add_argument(
        "--max-titles", type=int, default=None,
        help="optional cap on most-recent titles pooled per reviewer (default: no cap, use --years alone)"
    )
    parser.add_argument(
        "--area-weight", type=float, default=1.0,
        help="weight of the area/keyword document relative to one title document (default: 1.0)"
    )
    parser.add_argument("--limit", type=int, default=None, help="only process the first N reviewers")
    parser.add_argument(
        "--delay", type=float, default=3.0,
        help="base seconds between live DBLP fetches; actual delay is delay ± 50%% jitter"
    )
    parser.add_argument("--device", default="cuda", help="torch device for SPECTER2 (default: cuda)")
    args = parser.parse_args()

    if args.years <= 0:
        parser.error("--years must be greater than 0")
    if args.max_titles is not None and args.max_titles < 0:
        parser.error("--max-titles must be 0 or greater")
    if args.area_weight <= 0:
        parser.error("--area-weight must be greater than 0")
    if args.delay < 0:
        parser.error("--delay must be non-negative")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARN: cuda requested but not available, falling back to cpu", file=sys.stderr)
        args.device = "cpu"

    reviewers = load_reviewers(args.csv)
    if args.limit is not None:
        reviewers = reviewers[: args.limit]

    fp_cache = fp.load_fingerprint_cache(args.fingerprint_cache)

    # Resolve titles for every PID-backed reviewer. Cache hits are cheap, and
    # doing this before the fingerprint freshness check lets DBLP cache edits
    # participate in the content key.
    write_cache = load_cache(args.cache)
    readonly_cache = load_colleague_cache(args.colleague_cache)
    titles_by_pid: dict[str, list[tuple[int, str]]] = {}
    failed_pids: set[str] = set()
    fetch_errors = 0

    with_pid = [r for r in reviewers if r.pid]
    if with_pid:
        def on_result(pid: str, titles: list[tuple[int, str]], source: str) -> None:
            titles_by_pid[pid] = titles

        def on_error(pid: str, exc: Exception) -> None:
            nonlocal fetch_errors
            fetch_errors += 1
            failed_pids.add(pid)
            print(f"  WARN: DBLP fetch failed for pid={pid}: {exc}", file=sys.stderr)

        session = requests.Session()
        fetch_titles_for_pids(
            list(dict.fromkeys(r.pid for r in with_pid)),
            years=args.years,
            session=session,
            write_cache=write_cache,
            readonly_cache=readonly_cache,
            cache_path=args.cache,
            delay=args.delay,
            on_result=on_result,
            on_error=on_error,
        )

    selected_by_email = {
        r.email: fp.select_titles(titles_by_pid.get(r.pid, []), args.max_titles)
        for r in reviewers
    }
    keys = {
        r.email: fingerprint_key(
            r, selected_by_email[r.email], years=args.years,
            max_titles=args.max_titles, area_weight=args.area_weight,
        )
        for r in reviewers
    }

    pending = []
    for r in reviewers:
        entry = fp_cache.get(r.email)
        fetch_failed = bool(r.pid and r.pid in failed_pids)
        if fetch_failed and entry is not None:
            entry["dblp_fetch_complete"] = False
            continue
        if entry is None or entry.get("fingerprint_key") != keys[r.email] or not entry.get("dblp_fetch_complete", True):
            pending.append(r)

    print(
        f"Loaded {len(reviewers)} accepted reviewers; {len(pending)} need fingerprints "
        f"({len(reviewers) - len(pending)} already cached).",
        file=sys.stderr,
    )

    # --- 2. Load SPECTER2 once, build every pending reviewer's input docs,
    #        encode in one batched pass, then pool per reviewer ---------------
    area_only = 0
    if pending:
        print(f"Loading SPECTER2 on {args.device}...", file=sys.stderr)
        tokenizer, model = specter2_model.load_model(device=args.device)

        flat_texts: list[str] = []
        flat_owner: list[int] = []  # index into `pending`
        per_reviewer_weights: list[list[float]] = [[] for _ in pending]
        n_titles_used = [0] * len(pending)

        for i, r in enumerate(pending):
            selected = selected_by_email[r.email]
            n_titles_used[i] = len(selected)

            for _, title in selected:
                flat_texts.append(fp.title_doc_text(tokenizer, title))
                flat_owner.append(i)
                per_reviewer_weights[i].append(1.0)

            flat_texts.append(fp.area_profile_text(tokenizer, r))
            flat_owner.append(i)
            per_reviewer_weights[i].append(args.area_weight)

        print(f"Encoding {len(flat_texts)} documents for {len(pending)} reviewers...", file=sys.stderr)
        vectors = specter2_model.encode_texts(flat_texts, tokenizer, model)

        offsets = [0] * (len(pending) + 1)
        for owner in flat_owner:
            offsets[owner + 1] += 1
        for i in range(len(pending)):
            offsets[i + 1] += offsets[i]

        for i, r in enumerate(pending):
            reviewer_vectors = vectors[offsets[i] : offsets[i + 1]]
            reviewer_weights = per_reviewer_weights[i]
            fingerprint_vec = fp.pool(reviewer_vectors, reviewer_weights)
            if n_titles_used[i] == 0:
                area_only += 1
            fp_cache[r.email] = {
                "vector": fingerprint_vec.tolist(),
                "n_titles": n_titles_used[i],
                "has_pid": bool(r.pid),
                "fingerprint_key": keys[r.email],
                "schema_version": FINGERPRINT_SCHEMA_VERSION,
                "dblp_fetch_complete": not bool(r.pid and r.pid in failed_pids),
            }

        fp.save_fingerprint_cache(fp_cache, args.fingerprint_cache)
    elif failed_pids:
        # Persist retry state without changing the last known-good vectors.
        fp.save_fingerprint_cache(fp_cache, args.fingerprint_cache)

    print(
        f"\nDone. computed={len(pending)} area_only_fallback={area_only} "
        f"dblp_fetch_errors={fetch_errors}",
        file=sys.stderr,
    )

    # --- 4. Sanity check: nearest neighbors for spot-check reviewers --------
    email_to_reviewer = {r.email: r for r in reviewers}
    emails_with_vectors = [r.email for r in reviewers if r.email in fp_cache]
    if len(emails_with_vectors) < 2:
        return 0

    matrix = np.array([fp_cache[e]["vector"] for e in emails_with_vectors], dtype=np.float32)
    index_by_email = {e: i for i, e in enumerate(emails_with_vectors)}

    spot_checks = [e for e in SPOT_CHECK_EMAILS if e in index_by_email]
    for e in emails_with_vectors:
        if len(spot_checks) >= 3:
            break
        if e not in spot_checks:
            spot_checks.append(e)

    for email in spot_checks:
        r = email_to_reviewer[email]
        idx = index_by_email[email]
        sims = matrix @ matrix[idx]
        order = np.argsort(-sims)
        top = [i for i in order if i != idx][:5]
        print(f"\n=== Nearest neighbors for {r.name} <{email}> ({r.primary})")
        for rank, i in enumerate(top, 1):
            other = email_to_reviewer[emails_with_vectors[i]]
            print(f"  {rank}. {sims[i]:.3f}  {other.name} <{other.email}>  ({other.primary})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
