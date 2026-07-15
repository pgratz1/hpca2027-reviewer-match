"""Reviewer DBLP title fetcher — prints each reviewer's recent paper titles.

    python main.py --limit 5         # validate on a handful first
    python main.py                   # full run
    python main.py --years 2         # narrower window

Titles are drawn from (in order): colleague's pre-built cache, our own cache,
then live DBLP. Results are cached in dblp_cache.json so each PID is fetched
at most once from the network.
"""

from __future__ import annotations

import argparse
import datetime
import sys

import requests

from dblp import fetch_titles_for_pids, load_cache, load_colleague_cache
from reviewers import load_reviewers

DEFAULT_CSV = "HPCA'27 PC Member Acceptance Form (Responses) - Form Responses 1.csv"
DEFAULT_CACHE = "dblp_cache.json"
DEFAULT_COLLEAGUE_CACHE = "dblp_pubs_cache.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV, help="path to the reviewer CSV")
    parser.add_argument("--cache", default=DEFAULT_CACHE, help="path to our writable JSON title cache")
    parser.add_argument(
        "--colleague-cache", default=DEFAULT_COLLEAGUE_CACHE,
        help="path to the colleague's read-only pre-built cache"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="only process the first N reviewers with a PID"
    )
    parser.add_argument(
        "--delay", type=float, default=3.0,
        help="base seconds between live DBLP fetches; actual delay is delay ± 50%% jitter"
    )
    parser.add_argument(
        "--years", type=int, default=4,
        help="include publications from the most recent N calendar years (default: 4)"
    )
    args = parser.parse_args()

    if args.years <= 0:
        parser.error("--years must be greater than 0")
    if args.delay < 0:
        parser.error("--delay must be non-negative")

    current_year = datetime.date.today().year
    year_window = f"{current_year - args.years + 1}–{current_year}"

    write_cache = load_cache(args.cache)
    readonly_cache = load_colleague_cache(args.colleague_cache)

    reviewers = load_reviewers(args.csv)
    with_pid = [r for r in reviewers if r.pid]
    skipped_no_pid = len(reviewers) - len(with_pid)

    in_colleague = sum(1 for r in with_pid if r.pid in readonly_cache)
    in_write = sum(1 for r in with_pid if r.pid not in readonly_cache and r.pid in write_cache)

    if args.limit is not None:
        with_pid = with_pid[: args.limit]

    print(
        f"Loaded {len(reviewers)} accepted reviewers; {len(with_pid)} to process "
        f"({in_colleague} in colleague cache, {in_write} in our cache, "
        f"{skipped_no_pid} skipped — no DBLP PID).",
        file=sys.stderr,
    )

    pid_to_reviewer = {r.pid: r for r in with_pid}
    counts = {"colleague": 0, "cache": 0, "live": 0, "errors": 0}

    def on_result(pid: str, titles: list[tuple[int, str]], source: str) -> None:
        counts[source] += 1
        r = pid_to_reviewer[pid]
        src_tag = f"[{source}]" if source != "colleague" else ""
        print(f"\n=== {r.name} <{r.email}>  [pid {r.pid}, {r.tier} PC] {src_tag}")
        if not titles:
            print(f"  (no publications found in {year_window})")
        else:
            print(f"  Papers in {year_window} ({len(titles)} total):")
            for j, (year, title) in enumerate(titles, 1):
                print(f"  {j:3d}. [{year}] {title}")

    def on_error(pid: str, exc: Exception) -> None:
        counts["errors"] += 1
        r = pid_to_reviewer[pid]
        print(f"  WARN: fetch failed for {r.name} pid={r.pid}: {exc}", file=sys.stderr)

    session = requests.Session()
    fetch_titles_for_pids(
        [r.pid for r in with_pid],
        years=args.years,
        session=session,
        write_cache=write_cache,
        readonly_cache=readonly_cache,
        cache_path=args.cache,
        delay=args.delay,
        on_result=on_result,
        on_error=on_error,
    )

    print(
        f"\nDone. colleague_cache={counts['colleague']} our_cache={counts['cache']} "
        f"live_fetched={counts['live']} errors={counts['errors']} "
        f"skipped_no_pid={skipped_no_pid}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
