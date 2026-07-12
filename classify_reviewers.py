"""Classify accepted PC members by publication seniority from DBLP history.

    python classify_reviewers.py

Three classes, based on publication counts in the four top architecture
venues (ISCA, MICRO, HPCA, ASPLOS):

  senior  — averaged at least --senior-rate (default 0.8) target-venue papers
            per year over the last --window (default 15) years, i.e. 12+
            papers in the window at the defaults;
  junior  — fewer than --junior-total (default 7) target-venue papers over
            the whole career;
  typical — everyone else. Senior is checked first.

Reviewers with no resolvable DBLP identity (no usable link in the acceptance
CSV and no dblp_overrides.csv entry) get class 'unknown' with a reason
column instead. Every unknown is also appended to dblp_overrides.csv as a
stub row (blank dblp cell; name/affiliation/reason in the note) unless
already listed, so that file doubles as the to-do list: fill in the dblp
cell (any link shape or bare PID) and rerun. Overrides are keyed by email so
they survive acceptance-CSV re-exports, and they win over the form's own
DBLP link (see reviewers.py).

Writes one row per accepted reviewer to reviewer_seniority.csv with the
per-venue career and window counts backing the classification, so the
assignment step can also spot near-threshold reviewers ("almost senior",
"almost not junior") when a paper can't otherwise satisfy its seniority
constraints. Progress and a class summary go to stderr. Uses the rich DBLP
caches (venue kept per publication); PIDs not covered by either cache are
fetched live once and cached in dblp_venue_cache.json.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import io
import re
import sys
from collections import Counter
from dataclasses import dataclass

import dblp
from reviewers import DEFAULT_OVERRIDES, Reviewer, load_reviewers

DEFAULT_CSV = "HPCA'27 PC Member Acceptance Form (Responses) - Form Responses 1.csv"
DEFAULT_COLLEAGUE_CACHE = "dblp_pubs_cache.json"
DEFAULT_VENUE_CACHE = "dblp_venue_cache.json"
DEFAULT_OUT = "reviewer_seniority.csv"
DEFAULT_WINDOW = 15
DEFAULT_SENIOR_RATE = 0.8
DEFAULT_JUNIOR_TOTAL = 7

TARGET_VENUES = ("ISCA", "MICRO", "HPCA", "ASPLOS")
# Anchored so near-misses don't count: matches the plain venue string and
# DBLP's multi-volume style ('ASPLOS (2)'), but not ISCAS, IEEE Micro,
# EUROMICRO, 'ISCA Workshops', or co-located workshops ('HASP@ISCA',
# 'NoCArc@MICRO', 'EMC2@HPCA/CVPR/ISCA').
TARGET_VENUE_RE = re.compile(r"^(ISCA|MICRO|HPCA|ASPLOS)( \(\d+\))?$")

OUT_FIELDS = [
    "email", "name", "affiliation", "tier",
    "pid", "pid_source", "class",
    "isca_career", "micro_career", "hpca_career", "asplos_career", "career_total",
    "isca_window", "micro_window", "hpca_window", "asplos_window", "window_total",
    "window_rate", "first_target_year", "last_target_year", "total_pubs",
    "pub_source", "reason",
]


@dataclass
class Classification:
    label: str  # 'senior' | 'junior' | 'typical'
    career_counts: dict[str, int]  # base venue -> career count
    career_total: int
    window_counts: dict[str, int]  # base venue -> count within the window
    window_total: int
    window_rate: float  # window_total / window
    first_target_year: int | None
    last_target_year: int | None
    total_pubs: int  # all deduped records, any venue


def target_venue(record: dict) -> str | None:
    """Base venue name if `record` is a countable target-venue paper, else None.

    Editorship records (type == 'proceedings') don't count; records with an
    empty/missing type do — thousands of real papers in the colleague cache
    have no type. Fields are read with .get() because some colleague-cache
    entries are person-shaped rather than publications.
    """
    if (record.get("type") or "") == "proceedings":
        return None
    m = TARGET_VENUE_RE.match((record.get("venue") or "").strip())
    return m.group(1) if m else None


def classify(
    records: list[dict],
    *,
    window: int,
    current_year: int,
    senior_rate: float,
    junior_total: int,
) -> Classification:
    """Classify one reviewer from their rich DBLP publication records.

    Records are deduped by (normalised title, year) first — the colleague
    cache merges sources and can list the same paper twice. The window is
    the `window` years ending at `current_year` inclusive.
    """
    seen: set[tuple[str, int]] = set()
    deduped: list[tuple[int, str | None]] = []  # (year, target venue or None)
    for rec in records:
        title = (rec.get("title") or "").strip()
        if not title:
            continue
        try:
            year = int(rec.get("year"))
        except (TypeError, ValueError):
            continue
        key = (title.lower().rstrip(". "), year)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((year, target_venue(rec)))

    window_start = current_year - window + 1
    career_counts = {v: 0 for v in TARGET_VENUES}
    window_counts = {v: 0 for v in TARGET_VENUES}
    target_years: list[int] = []
    for year, venue in deduped:
        if venue is None:
            continue
        career_counts[venue] += 1
        target_years.append(year)
        if window_start <= year <= current_year:
            window_counts[venue] += 1

    career_total = sum(career_counts.values())
    window_total = sum(window_counts.values())

    # Epsilon guards float products: 0.8 * 15 evaluates just above 12.0, which
    # would otherwise misclassify an exactly-at-threshold reviewer.
    if window_total >= senior_rate * window - 1e-9:
        label = "senior"
    elif career_total < junior_total:
        label = "junior"
    else:
        label = "typical"

    return Classification(
        label=label,
        career_counts=career_counts,
        career_total=career_total,
        window_counts=window_counts,
        window_total=window_total,
        window_rate=window_total / window,
        first_target_year=min(target_years) if target_years else None,
        last_target_year=max(target_years) if target_years else None,
        total_pubs=len(deduped),
    )


def load_seniority(path: str = DEFAULT_OUT) -> dict[str, dict]:
    """Load this script's output CSV: email -> {class, window_total, career_total}.

    The consumer is assign_reviewers.py's seniority constraints, which need
    the class plus the two near-threshold totals ("almost senior" is judged
    on window_total, "almost not junior" on career_total). Unknown-class rows
    have blank totals, loaded as None. Raises FileNotFoundError if the file
    doesn't exist — the caller owns the "run classify_reviewers.py first"
    message.
    """
    out: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["email"].strip().lower()] = {
                "class": (row.get("class") or "").strip(),
                "window_total": int(row["window_total"]) if (row.get("window_total") or "").strip() else None,
                "career_total": int(row["career_total"]) if (row.get("career_total") or "").strip() else None,
            }
    return out


def resolve_pid(reviewer: Reviewer) -> tuple[str | None, str, str]:
    """(pid, pid_source, reason) for one reviewer.

    A hand-maintained dblp_overrides.csv entry wins (applied inside
    load_reviewers), then the acceptance CSV's own DBLP link; otherwise the
    reviewer is unclassifiable until their stub row in the overrides file
    gets its dblp cell filled in.
    """
    if reviewer.pid:
        return reviewer.pid, "override" if reviewer.pid_from_override else "csv", ""
    return None, "none", "no DBLP link in the form or dblp_overrides.csv — fill in the stub's dblp cell"


def populate_override_stubs(path: str, unknowns: list[dict]) -> int:
    """Append a stub row to the overrides file for each unknown reviewer.

    The stub has a blank dblp cell (which load_dblp_overrides skips) and the
    reviewer's name, affiliation, and unknown-reason in the note column, so
    the file doubles as the to-do list of identities to hunt down. Reviewers
    already listed — filled in or still blank — are never touched, and
    existing rows are never rewritten, only appended after. Returns the
    number of stubs added.
    """
    try:
        with open(path, newline="", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        text = ""
    existing = {
        (row.get("email") or "").strip().lower()
        for row in csv.DictReader(io.StringIO(text))
    }
    new = [r for r in unknowns if r["email"] not in existing]
    if not new:
        return 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        if not text:
            f.write("email,dblp,note\n")
        elif not text.endswith("\n"):
            f.write("\n")
        writer = csv.writer(f)
        for r in new:
            note = f"{r['name']} ({r['affiliation']})"
            if r["reason"]:
                note += f" — {r['reason']}"
            writer.writerow([r["email"], "", note])
    return len(new)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--csv", default=DEFAULT_CSV, help="acceptance-form CSV (default: %(default)s)")
    parser.add_argument("--overrides", default=DEFAULT_OVERRIDES, help="hand-maintained DBLP override CSV; unknowns are appended here as blank stubs (default: %(default)s)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output CSV (default: %(default)s)")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="length of the senior window in years (default: %(default)s)")
    parser.add_argument("--senior-rate", type=float, default=DEFAULT_SENIOR_RATE, help="target-venue papers per year over the window to classify senior (default: %(default)s)")
    parser.add_argument("--junior-total", type=int, default=DEFAULT_JUNIOR_TOTAL, help="career target-venue papers below which a reviewer is junior (default: %(default)s)")
    parser.add_argument("--current-year", type=int, default=datetime.date.today().year, help="last year of the window, inclusive (default: this year)")
    parser.add_argument("--venue-cache", default=DEFAULT_VENUE_CACHE, help="writable rich DBLP cache (default: %(default)s)")
    parser.add_argument("--colleague-cache", default=DEFAULT_COLLEAGUE_CACHE, help="read-only rich DBLP cache (default: %(default)s)")
    parser.add_argument("--delay", type=float, default=3.0, help="seconds between live DBLP fetches, jittered ±50%% (default: %(default)s)")
    args = parser.parse_args()

    reviewers = load_reviewers(args.csv, overrides_path=args.overrides)
    venue_cache = dblp.load_rich_cache(args.venue_cache)
    colleague_cache = dblp.load_rich_cache(args.colleague_cache)

    resolved = [(r, *resolve_pid(r)) for r in reviewers]

    pids: list[str] = []
    for _, pid, _, _ in resolved:
        if pid and pid not in pids:
            pids.append(pid)

    n_live = 0

    def on_result(pid: str, records: list[dict], source: str) -> None:
        nonlocal n_live
        if source == "live":
            n_live += 1
            print(f"  fetched {pid}: {len(records)} records", file=sys.stderr)

    def on_error(pid: str, exc: Exception) -> None:
        print(f"  FAILED {pid}: {exc}", file=sys.stderr)

    print(f"Fetching DBLP records for {len(pids)} PIDs...", file=sys.stderr)
    results = dblp.fetch_records_for_pids(
        pids,
        write_cache=venue_cache,
        readonly_cache=colleague_cache,
        cache_path=args.venue_cache,
        delay=args.delay,
        on_result=on_result,
        on_error=on_error,
    )

    rows: list[dict] = []
    class_counts: Counter[str] = Counter()
    unknown_reasons: Counter[str] = Counter()
    for r, pid, pid_source, reason in resolved:
        row = {f: "" for f in OUT_FIELDS}
        row.update(
            email=r.email, name=r.name, affiliation=r.affiliation, tier=r.tier,
            pid=pid or "", pid_source=pid_source, reason=reason,
        )
        if pid is None:
            row["class"] = "unknown"
        elif pid not in results:
            row["class"] = "unknown"
            row["reason"] = f"DBLP fetch failed for {pid} — rerun to retry"
        else:
            records, pub_source = results[pid]
            c = classify(
                records,
                window=args.window,
                current_year=args.current_year,
                senior_rate=args.senior_rate,
                junior_total=args.junior_total,
            )
            row["class"] = c.label
            for venue in TARGET_VENUES:
                row[f"{venue.lower()}_career"] = c.career_counts[venue]
                row[f"{venue.lower()}_window"] = c.window_counts[venue]
            row["career_total"] = c.career_total
            row["window_total"] = c.window_total
            row["window_rate"] = f"{c.window_rate:.3f}"
            row["first_target_year"] = c.first_target_year if c.first_target_year is not None else ""
            row["last_target_year"] = c.last_target_year if c.last_target_year is not None else ""
            row["total_pubs"] = c.total_pubs
            row["pub_source"] = pub_source
        class_counts[row["class"]] += 1
        if row["class"] == "unknown":
            unknown_reasons[row["reason"]] += 1
        rows.append(row)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    window_start = args.current_year - args.window + 1
    print(
        f"\nWindow: {window_start}–{args.current_year}; senior needs "
        f">= {args.senior_rate * args.window:g} window papers, junior has "
        f"< {args.junior_total} career papers (in {'/'.join(TARGET_VENUES)})",
        file=sys.stderr,
    )
    print(
        "Classes: " + ", ".join(
            f"{label}={class_counts[label]}"
            for label in ("senior", "typical", "junior", "unknown")
        ),
        file=sys.stderr,
    )
    if unknown_reasons:
        print("Unknown reasons:", file=sys.stderr)
        for why, n in unknown_reasons.most_common():
            print(f"  {n:3d}  {why}", file=sys.stderr)
    n_stubs = populate_override_stubs(
        args.overrides, [r for r in rows if r["class"] == "unknown"]
    )
    if n_stubs:
        print(
            f"Added {n_stubs} stub rows to {args.overrides} — fill in the dblp "
            f"column and rerun to classify them",
            file=sys.stderr,
        )
    print(f"{n_live} live DBLP fetches; wrote {len(rows)} rows to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
