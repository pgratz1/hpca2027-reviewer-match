"""List submitted reviews with little or no text behind their scores.

    python -m scripts.short_reviews
    python -m scripts.short_reviews --short-words 200 --thin-fields 1

A review is flagged when any of these holds:

- its author-visible text (summary, strengths, weaknesses, comments,
  questions) runs to `--short-words` words or fewer;
- `--thin-fields` or more of the four core fields (summary, strengths,
  weaknesses, comments for authors) are empty or a placeholder ("n/a",
  "none", "-", "see above", ...);
- it is under `--paper-ratio` of the median length of the same paper's other
  reviews, which catches a review that is thin *for its paper*.

Reviews on desk-rejected papers (and withdrawn ones, which the paper export
omits) are skipped and stay out of each paper's median; papers rejected in
round 1 are kept, since their authors got those reviews. stdout
lists R1 reviews; TRC reviews appear in the CSV only. Writes
`outputs/reports/short_reviews.csv`, which names reviewers and is gitignored by
name; prints a per-reviewer summary to stdout. Offline, read-only.
"""

from __future__ import annotations

from reviewer_match.paths import input_path, report_path

import argparse
import csv
import json
import os
import re
import statistics
import sys
from collections import defaultdict

from reviewer_match import hotcrp_log, review_scores

DEFAULT_REVIEWS = input_path("hpca2027-reviews.csv")
DEFAULT_LOG = input_path("hpca2027-log.csv")
DEFAULT_DATA = input_path("hpca2027-data.json")
DEFAULT_OUT = report_path("short_reviews.csv")
DEFAULT_SHORT_WORDS = 150
DEFAULT_THIN_FIELDS = 2
DEFAULT_PAPER_RATIO = 0.25

AUTHOR_FIELDS = (
    "Paper summary", "Strengths", "Weaknesses", "Comments for authors",
    "Questions for revision/rebuttal",
)
CORE_FIELDS = AUTHOR_FIELDS[:4]
PC_FIELD = "Comments for PC"
# A field holding only a pointer elsewhere or a shrug. Short fields only: a
# long answer that happens to start "See ..." is real text.
PLACEHOLDER_MAX_WORDS = 5
PLACEHOLDER_RE = re.compile(
    r"^(?:(?:please\s+)?(?:see|refer|check)\b|(?:same\s+)?as\s+above|included\s+above|"
    r"n/?a\b|na\b|none\b|nil\b|nothing\b|no\b|-+$|\.+$)",
    re.IGNORECASE,
)
WORDS_RE = re.compile(r"(\d+) words$")
SAVE_RE = re.compile(r"^Review (\d+) (?:submitted|edited)[^:]*:")


def words(text: str | None) -> int:
    return len((text or "").split())


def thin(text: str | None) -> bool:
    """True for an empty field, one with no letters, or a short placeholder."""
    text = (text or "").strip()
    if not re.search(r"[A-Za-z]", text):
        return True
    return words(text) <= PLACEHOLDER_MAX_WORDS and bool(PLACEHOLDER_RE.match(text))


def log_word_counts(rows: list[dict[str, str]]) -> dict[int, int]:
    """{review id: word count at its latest save}, from the chronological log."""
    out: dict[int, int] = {}
    for row in rows:
        action = row["action"]
        m = SAVE_RE.match(action)
        w = WORDS_RE.search(action) if m else None
        if w:
            out[int(m.group(1))] = int(w.group(1))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reviews", default=DEFAULT_REVIEWS, help="HotCRP review CSV export")
    parser.add_argument("--log", default=DEFAULT_LOG, help="HotCRP action log CSV")
    parser.add_argument("--data", default=DEFAULT_DATA, help="HotCRP paper JSON export")
    parser.add_argument("--out", default=DEFAULT_OUT, help="per-review CSV of flagged reviews")
    parser.add_argument("--short-words", type=int, default=DEFAULT_SHORT_WORDS,
                        help="flag at this many author-visible words or fewer (default %(default)s)")
    parser.add_argument("--thin-fields", type=int, default=DEFAULT_THIN_FIELDS,
                        help="flag at this many empty/placeholder core fields (default %(default)s)")
    parser.add_argument("--paper-ratio", type=float, default=DEFAULT_PAPER_RATIO,
                        help="flag under this share of the paper's other reviews' median "
                             "length; 0 turns it off (default %(default)s)")
    parser.add_argument("--exclude-pids", default="",
                        help="comma-separated paper IDs treated as not under review")
    args = parser.parse_args()

    log_rows = hotcrp_log.load_log(args.log)
    live, _ = hotcrp_log.replay_assignments(log_rows)
    log_words = log_word_counts(log_rows)
    rid_by_pair = {(r.pid, r.email): rid for rid, r in live.items()}
    exclude = frozenset(int(p) for p in args.exclude_pids.split(",") if p.strip())
    # Desk rejections only: a paper rejected in round 1 got real reviews, and
    # `eligible_pids`' under-review set would drop it with them.
    _, desk_rejected = review_scores.eligible_pids(args.data, exclude)
    with open(args.data, encoding="utf-8") as f:
        kept = {p["pid"] for p in json.load(f)} - desk_rejected - exclude

    reviews: list[dict] = []
    skipped = 0
    with open(args.reviews, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid, email = int(row["paper"]), row["email"].strip().lower()
            if pid not in kept:
                skipped += 1
                continue
            per_field = {name: words(row.get(name)) for name in AUTHOR_FIELDS}
            reviews.append({
                "paper": pid, "review": row["review"], "rid": rid_by_pair.get((pid, email)),
                "reviewer": row["reviewername"], "email": email,
                "merit": row.get(review_scores.SCORE_FIELD) or "",
                "words": sum(per_field.values()),
                "words_pc": words(row.get(PC_FIELD)),
                "thin": [name for name in CORE_FIELDS if thin(row.get(name))],
                "per_field": per_field,
            })
    # A reviewer whose address changed since assignment is in the export under
    # the new one, so the log knows the review by the old one. Pair each such
    # row with the same paper's unclaimed review whose logged word count (which
    # includes Comments for PC) is closest; the log is the only record of round.
    claimed = {rv["rid"] for rv in reviews if rv["rid"] is not None}
    for rv in reviews:
        if rv["rid"] is None:
            candidates = [rid for rid, r in live.items() if r.pid == rv["paper"] and rid not in claimed]
            if candidates:
                target = rv["words"] + rv["words_pc"]
                rv["rid"] = min(candidates, key=lambda c: abs((log_words.get(c) or 0) - target))
                claimed.add(rv["rid"])
        rv["round"] = live[rv["rid"]].round if rv["rid"] in live else "?"

    by_paper: dict[int, list[dict]] = defaultdict(list)
    for rv in reviews:
        by_paper[rv["paper"]].append(rv)

    flagged = []
    for rv in reviews:
        others = [o["words"] for o in by_paper[rv["paper"]] if o is not rv]
        rv["paper_median"] = statistics.median(others) if others else None
        reasons = []
        if rv["words"] <= args.short_words:
            reasons.append(f"<= {args.short_words} words")
        if len(rv["thin"]) >= args.thin_fields:
            reasons.append(f"{len(rv['thin'])} empty/placeholder fields")
        if (args.paper_ratio and rv["paper_median"]
                and rv["words"] < args.paper_ratio * rv["paper_median"]):
            reasons.append(f"< {args.paper_ratio:g} x paper median")
        if reasons:
            rv["reasons"] = reasons
            flagged.append(rv)
    flagged.sort(key=lambda rv: (rv["words"], rv["paper"]))

    fields = ["paper", "review", "reviewer", "email", "round", "merit", "words",
              "paper_median", "words_pc", *(f"words_{n.split()[0].lower()}" for n in AUTHOR_FIELDS),
              "thin_fields", "reasons"]
    tmp = args.out + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for rv in flagged:
            writer.writerow([
                rv["paper"], rv["review"], rv["reviewer"], rv["email"], rv["round"],
                rv["merit"], rv["words"],
                "" if rv["paper_median"] is None else f"{rv['paper_median']:g}",
                rv["words_pc"], *(rv["per_field"][n] for n in AUTHOR_FIELDS),
                "; ".join(rv["thin"]), "; ".join(rv["reasons"]),
            ])
    os.replace(tmp, args.out)

    lengths = sorted(rv["words"] for rv in reviews)
    main_set = [rv for rv in flagged if rv["round"] == "R1"]
    print(f"{len(reviews)} submitted reviews ({skipped} on desk-rejected, withdrawn "
          f"or excluded papers skipped)", file=sys.stderr)
    print(f"median {statistics.median(lengths):g} words; shortest {lengths[0]}; "
          f"zero-text reviews: {sum(1 for w in lengths if w == 0)}", file=sys.stderr)
    print(f"wrote {args.out}", file=sys.stderr)

    print(f"Flagged: {len(flagged)} reviews; {len(main_set)} are R1 "
          f"({len(flagged) - len(main_set)} TRC, in the CSV only).")
    reason_counts: dict[str, int] = defaultdict(int)
    for rv in main_set:
        for reason in rv["reasons"]:
            reason_counts[reason] += 1
    for reason, n in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {n:4d}  {reason}")
    print()
    print("R1 reviews, shortest first:")
    print(f"  {'paper':>5} {'review':<7} {'words':>5} {'paper med':>9} {'merit':>5}  reviewer / reasons")
    for rv in main_set:
        med = "" if rv["paper_median"] is None else f"{rv['paper_median']:g}"
        print(f"  {rv['paper']:>5} {rv['review']:<7} {rv['words']:>5} "
              f"{med:>9} {rv['merit']:>5}  {rv['reviewer']} -- {'; '.join(rv['reasons'])}")

    by_reviewer: dict[str, list[dict]] = defaultdict(list)
    for rv in reviews:
        if rv["round"] == "R1":
            by_reviewer[rv["email"]].append(rv)
    repeat = []
    for email, rvs in by_reviewer.items():
        hits = [rv for rv in rvs if rv in main_set]
        if len(hits) >= 2:
            repeat.append((len(hits), len(rvs), statistics.median(rv["words"] for rv in rvs),
                           rvs[0]["reviewer"] or email))
    repeat.sort(key=lambda t: (-t[0], t[2]))
    print()
    print("Reviewers with 2+ flagged R1 reviews (flagged / submitted, their median words):")
    for hits, total, med, who in repeat:
        print(f"  {hits:2d} / {total:2d}  median {med:5g}  {who}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
