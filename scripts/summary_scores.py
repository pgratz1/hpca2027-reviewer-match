"""Do papers without a discussion summary score worse, and does summary length track score?

    python -m scripts.summary_scores
    python -m scripts.summary_scores --all-decided
    python -m scripts.summary_scores --permutations 20000 --seed 7

Each paper advancing to revision was meant to get an author-visible summary
from its discussion lead. This compares the papers that got one with those that
did not, by average pre-rebuttal overall merit, and -- among the papers that
did -- correlates the summary's length with that average.

**A summary** is any author-visible comment on the paper: a `Comment @A<n>`
block in `--reviews-txt`, HotCRP's "Download reviews (text)" export. Whether one
of them is by the paper's current lead (`--leads`, HotCRP's "Discussion leads
(CSV)", with `hotcrp_log.replay_leads` filling the papers that download hides)
is reported as a second split. Its **length** is the words in all of the
paper's author-visible comments together.

**The population** is every paper still under review (`review_scores.
eligible_pids`: `submitted`, not desk-rejected, minus `--exclude-pids`) tagged
`RevisionAdvance` -- the papers a summary was asked for. `--all-decided` widens
it to every paper with `--min-decided` or more scored reviews, tagged or not.

**The score** is the paper's average pre-rebuttal overall merit in
`--reviews`, HotCRP's review CSV, with TRC reviews left out as
`revision_cutoffs` leaves them out (identified from `--log`).

**A paper the text export cannot see is left out, not counted as unsummarised.**
The text download hides reviews and comments on the downloader's own
conflicts; a population paper whose block shows no review while the CSV holds
scores is reported as "not visible" and dropped from both groups, since
counting it as "no summary" would bias exactly the comparison this makes.

Significance is by permutation (difference in means; Pearson and Spearman
correlation), plus a Mann-Whitney U with a normal approximation -- stdlib only.
Offline, instant, read-only apart from `--papers-csv`, which carries
confidential review data and is gitignored.
"""

from __future__ import annotations

from reviewer_match.paths import input_path, report_path

import argparse
import csv
import math
import os
import random
import re
import statistics
import sys
import tempfile
from collections import Counter

from reviewer_match import hotcrp_log
from reviewer_match import paper_matching
from reviewer_match.review_scores import (
    ADVANCE_TAG, DEFAULT_MIN_DECIDED, average, eligible_pids, load_review_scores,
    review_rounds, tagged_pids,
)

DEFAULT_REVIEWS = input_path("hpca2027-reviews.csv")
DEFAULT_REVIEWS_TXT = input_path("hpca2027-reviews.txt")
DEFAULT_LEADS = input_path("hpca2027-leads.csv")
DEFAULT_LOG = input_path("hpca2027-log.csv")
DEFAULT_DATA = input_path("hpca2027-data.json")
DEFAULT_PAPERS_CSV = report_path("summary_scores_papers.csv")
DEFAULT_PERMUTATIONS = 10000
DEFAULT_SEED = 1
SCORE_BINS = (2.0, 2.5, 3.0, 3.5)  # upper edges, inclusive; a last open bin above
TOP_LEADS = 15

SUBMISSION_RE = re.compile(r"^Submission #(\d+): ", re.M)
REVIEW_RE = re.compile(r"^Review #\d+[A-Z]+ ", re.M)
# Every block that can follow a comment body starts on one of these lines.
BLOCK_RE = re.compile(r"^(?:Comment @\S+ by |Review #|\* \* \* )", re.M)
AUTHOR_COMMENT_RE = re.compile(r"^Comment @A\d+ by (.*)$", re.M)
EMAIL_RE = re.compile(r"<([^<>@\s]+@[^<>\s]+)>")

PAPERS_FIELDS = [
    "paper", "title", "lead", "n_reviews", "average", "visible",
    "has_summary", "summary_by_lead", "summary_words",
]


def parse_txt(text: str) -> dict[int, dict]:
    """{pid: {"reviews": n, "comments": [(email or "", words)]}} from the text export.

    `comments` holds the author-visible (`@A`) comments only. A body runs from
    the line after the header's rule to the next comment, review or `* * *`
    separator, whichever comes first.
    """
    parts = SUBMISSION_RE.split(text)
    papers: dict[int, dict] = {}
    for i in range(1, len(parts), 2):
        pid, block = int(parts[i]), parts[i + 1]
        comments = []
        for match in AUTHOR_COMMENT_RE.finditer(block):
            email = EMAIL_RE.search(match.group(1))
            rest = block[match.end():].split("\n", 2)
            body = rest[2] if len(rest) > 2 else ""
            end = BLOCK_RE.search(body)
            if end:
                body = body[:end.start()]
            comments.append(((email.group(1).lower() if email else ""), len(body.split())))
        papers[pid] = {"reviews": len(REVIEW_RE.findall(block)), "comments": comments}
    return papers


def load_leads(path: str, log_rows: list[dict[str, str]]) -> dict[int, str]:
    """{pid: lead email}: the leads download, with the log filling what it hides."""
    leads = {pid: email for pid, email in hotcrp_log.replay_leads(log_rows).items()}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                email = (row.get("leademail") or "").strip().lower()
                if email:
                    leads[int(row["paper"])] = email
    return {pid: email.lower() for pid, email in leads.items()}


def ranks(values: list[float]) -> list[float]:
    """1-based ranks, ties sharing their mean rank."""
    order = sorted(range(len(values)), key=values.__getitem__)
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def pearson(x: list[float], y: list[float]) -> float:
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy) if sxx and syy else float("nan")


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(ranks(x), ranks(y))


def permutation_p_corr(x: list[float], y: list[float], stat, n: int, rng: random.Random) -> float:
    """Two-sided p of |stat(x, y)| under shuffling y."""
    observed = abs(stat(x, y))
    shuffled = list(y)
    hits = 0
    for _ in range(n):
        rng.shuffle(shuffled)
        if abs(stat(x, shuffled)) >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (n + 1)


def permutation_p_means(a: list[float], b: list[float], n: int, rng: random.Random) -> float:
    """Two-sided p of |mean(a) - mean(b)| under shuffling the group labels."""
    observed = abs(statistics.fmean(a) - statistics.fmean(b))
    pooled, total, k = a + b, sum(a) + sum(b), len(a)
    hits = 0
    for _ in range(n):
        rng.shuffle(pooled)
        sa = sum(pooled[:k])
        if abs(sa / k - (total - sa) / (len(pooled) - k)) >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (n + 1)


def mann_whitney(a: list[float], b: list[float]) -> tuple[float, float]:
    """(U for `a`, two-sided p) by the normal approximation with tie correction."""
    n1, n2 = len(a), len(b)
    r = ranks(a + b)
    u = sum(r[:n1]) - n1 * (n1 + 1) / 2
    n = n1 + n2
    ties = sum(t ** 3 - t for t in Counter(a + b).values())
    var = n1 * n2 / 12 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0:
        return u, float("nan")
    z = (u - n1 * n2 / 2) / math.sqrt(var)
    return u, math.erfc(abs(z) / math.sqrt(2))


def score_bin(avg: float) -> str:
    low = 1.0
    for edge in SCORE_BINS:
        if avg <= edge:
            return f"{low:.1f}-{edge:.1f}" if low > 1.0 else f"<= {edge:.1f}"
        low = edge
    return f"> {SCORE_BINS[-1]:.1f}"


def bin_labels() -> list[str]:
    labels, low = [], 1.0
    for edge in SCORE_BINS:
        labels.append(f"{low:.1f}-{edge:.1f}" if low > 1.0 else f"<= {edge:.1f}")
        low = edge
    return labels + [f"> {SCORE_BINS[-1]:.1f}"]


def describe(values: list[float]) -> str:
    if not values:
        return "n=0"
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return (f"n={len(values):4d}  mean {statistics.fmean(values):.3f}  "
            f"median {statistics.median(values):.3f}  sd {sd:.3f}  "
            f"range {min(values):.2f}-{max(values):.2f}")


def compare(label: str, yes: list[float], no: list[float], args, rng: random.Random) -> None:
    print(f"\n{label}")
    print(f"  with    {describe(yes)}")
    print(f"  without {describe(no)}")
    if len(yes) < 2 or len(no) < 2:
        print("  (too few papers in a group to test)")
        return
    diff = statistics.fmean(no) - statistics.fmean(yes)
    p_perm = permutation_p_means(yes, no, args.permutations, rng)
    u, p_mw = mann_whitney(no, yes)
    print(f"  without - with: {diff:+.3f} score points; permutation p = {p_perm:.4f}; "
          f"Mann-Whitney U = {u:.0f}, p = {p_mw:.4f}")


def correlate(label: str, rows: list[dict], args, rng: random.Random) -> None:
    rows = [r for r in rows if r["words"] > 0]
    print(f"\n{label} (n={len(rows)})")
    if len(rows) < 3:
        print("  (too few papers to correlate)")
        return
    words = [float(r["words"]) for r in rows]
    avgs = [r["avg"] for r in rows]
    r_p, r_s = pearson(words, avgs), spearman(words, avgs)
    p_p = permutation_p_corr(words, avgs, pearson, args.permutations, rng)
    p_s = permutation_p_corr(words, avgs, spearman, args.permutations, rng)
    print(f"  words: median {statistics.median(words):.0f}, mean {statistics.fmean(words):.0f}, "
          f"range {min(words):.0f}-{max(words):.0f}")
    print(f"  Pearson  r   = {r_p:+.3f}  (permutation p = {p_p:.4f})")
    print(f"  Spearman rho = {r_s:+.3f}  (permutation p = {p_s:.4f})")
    print("  by average score:      papers  median words  mean words")
    for b in bin_labels():
        w = [x for x, a in zip(words, avgs) if score_bin(a) == b]
        if w:
            print(f"    {b:>9}             {len(w):5d}  {statistics.median(w):12.0f}  "
                  f"{statistics.fmean(w):10.0f}")
    print("  by length quartile:    papers  words         mean score")
    order = sorted(range(len(rows)), key=lambda i: (words[i], i))
    for q in range(4):
        chunk = order[q * len(order) // 4:(q + 1) * len(order) // 4]
        if chunk:
            lo, hi = words[chunk[0]], words[chunk[-1]]
            mean_avg = statistics.fmean(avgs[i] for i in chunk)
            print(f"    Q{q + 1}                  {len(chunk):5d}  {lo:5.0f}-{hi:<6.0f}  {mean_avg:10.3f}")


def write_papers_csv(path: str, rows: list[dict]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(PAPERS_FIELDS)
        for r in sorted(rows, key=lambda r: (r["avg"], r["pid"])):
            writer.writerow([
                r["pid"], r["title"], r["lead"], r["n"], f"{r['avg']:.3f}",
                int(r["visible"]), int(r["has"]), int(r["by_lead"]), r["words"],
            ])
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--reviews", default=DEFAULT_REVIEWS, help="HotCRP review CSV export")
    parser.add_argument("--reviews-txt", default=DEFAULT_REVIEWS_TXT,
                        help="HotCRP review text export (carries the comments)")
    parser.add_argument("--leads", default=DEFAULT_LEADS, help="HotCRP discussion leads CSV")
    parser.add_argument("--log", default=DEFAULT_LOG, help="HotCRP action log (TRC reviews, hidden leads)")
    parser.add_argument("--data", default=DEFAULT_DATA, help="HotCRP paper export JSON")
    parser.add_argument("--exclude-pids", default="", help="comma-separated paper IDs to leave out")
    parser.add_argument("--all-decided", action="store_true",
                        help=f"every paper with --min-decided reviews, not just {ADVANCE_TAG}")
    parser.add_argument("--min-decided", type=int, default=DEFAULT_MIN_DECIDED,
                        help=f"reviews needed under --all-decided (default {DEFAULT_MIN_DECIDED})")
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS,
                        help=f"shuffles per permutation test (default {DEFAULT_PERMUTATIONS})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"permutation seed (default {DEFAULT_SEED})")
    parser.add_argument("--papers-csv", default=DEFAULT_PAPERS_CSV, help="per-paper CSV to write")
    args = parser.parse_args()

    exclude = paper_matching.parse_exclude_pids(args.exclude_pids)
    for path in (args.reviews, args.reviews_txt, args.log, args.data):
        if not os.path.exists(path):
            print(f"ERROR: {path} not found", file=sys.stderr)
            return 1

    log_rows = hotcrp_log.load_log(args.log)
    training, _ = review_rounds(log_rows)
    scores, titles, skipped = load_review_scores(args.reviews, training)
    eligible, _ = eligible_pids(args.data, exclude)
    with open(args.reviews_txt, encoding="utf-8") as f:
        txt = parse_txt(f.read())
    leads = load_leads(args.leads, log_rows)

    if args.all_decided:
        population = {pid for pid in eligible if len(scores.get(pid, ())) >= args.min_decided}
        what = f"papers with >= {args.min_decided} PC reviews"
    else:
        population = eligible & tagged_pids(args.data, ADVANCE_TAG)
        what = f"{ADVANCE_TAG} papers"
    no_scores = sorted(pid for pid in population if not scores.get(pid))
    population -= set(no_scores)
    if not population:
        print("ERROR: no paper meets the population criteria", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    rows = []
    for pid in sorted(population):
        block = txt.get(pid, {"reviews": 0, "comments": []})
        lead = leads.get(pid, "")
        comments = block["comments"]
        rows.append({
            "pid": pid, "title": titles.get(pid, ""), "lead": lead,
            "n": len(scores[pid]), "avg": float(average(scores[pid])),
            "visible": block["reviews"] > 0,
            "has": bool(comments),
            "by_lead": bool(lead) and any(email == lead for email, _ in comments),
            "words": sum(words for _, words in comments),
        })
    visible = [r for r in rows if r["visible"]]
    hidden = [r for r in rows if not r["visible"]]

    print(f"{len(skipped)} TRC reviews left out.", file=sys.stderr)
    if no_scores:
        print(f"{len(no_scores)} population paper(s) have no PC review score and are left out: "
              + ", ".join(f"#{p}" for p in no_scores), file=sys.stderr)
    if hidden:
        print(f"{len(hidden)} paper(s) show no review in {args.reviews_txt} (the downloader's "
              "conflicts?) and are left out: " + ", ".join(f"#{r['pid']}" for r in hidden),
              file=sys.stderr)

    has = [r for r in visible if r["has"]]
    print(f"Population: {len(rows)} {what}; {len(visible)} visible in the text export.")
    print(f"  {len(has)} have an author-visible comment ({len(has) / len(visible):.1%}); "
          f"{sum(r['by_lead'] for r in visible)} by the paper's lead; "
          f"{len(visible) - len(has)} have none.")

    print("\nAverage pre-rebuttal overall merit")
    compare("Any author-visible summary:",
            [r["avg"] for r in has], [r["avg"] for r in visible if not r["has"]], args, rng)
    compare("Summary by the paper's lead:",
            [r["avg"] for r in visible if r["by_lead"]],
            [r["avg"] for r in visible if not r["by_lead"]], args, rng)

    print("\nShare with a summary by average score")
    print("    average       papers  with summary   by lead")
    for b in bin_labels():
        group = [r for r in visible if score_bin(r["avg"]) == b]
        if group:
            n = len(group)
            print(f"    {b:>9}     {n:6d}  {sum(r['has'] for r in group) / n:12.1%}  "
                  f"{sum(r['by_lead'] for r in group) / n:8.1%}")

    missing_by_lead = Counter(r["lead"] or "(no lead)" for r in visible if not r["has"])
    held_by_lead = Counter(r["lead"] or "(no lead)" for r in visible)
    if missing_by_lead:
        print(f"\nLeads with the most papers lacking a summary (top {TOP_LEADS})")
        for lead, n in missing_by_lead.most_common(TOP_LEADS):
            print(f"    {n:3d} of {held_by_lead[lead]:3d}  {lead}")

    print("\nSummary length (words) vs average score")
    correlate("Papers with any author-visible summary", has, args, rng)
    correlate("Papers with a summary by their lead", [r for r in visible if r["by_lead"]], args, rng)

    write_papers_csv(args.papers_csv, rows)
    print(f"\nWrote {args.papers_csv} ({len(rows)} papers).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
