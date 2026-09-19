"""What share of reviewed papers each revision-eligibility cutoff would catch.

    python -m scripts.revision_cutoffs
    python -m scripts.revision_cutoffs --thresholds 1.5,1.75,2,2.25,2.5
    python -m scripts.revision_cutoffs --min-reviews 5 --complete-only
    python -m scripts.revision_cutoffs --include-trc

Decision support for the revision cutoff: a paper whose average pre-rebuttal
overall merit falls at or below a cutoff is not invited to revise. For each
cutoff this reports what share of the reviewed papers each *net* would catch.
A net is the average test, optionally with an extra condition that guards
papers that have someone arguing for them:

  average only             average <= cutoff
  + no score >= 4          ... and no reviewer scored it 4 or 5
  + at most one score >= 3 ... and at most one reviewer scored it 3 or better
  + no score >= 3          ... and every reviewer scored it 1 or 2

`NETS` in `reviewer_match.review_scores` is the one place to add another
variation; the bar `scripts/assign_paper_leads.py` applies is defined there too.

**Every figure is reported for both "average <= cutoff" and "average <
cutoff".** Scores are integers and most papers have four to six of them, so
many averages land exactly on a cutoff: at 2.0 the two readings differ by
about 18 points of the population. The policy has to say which it means.

The population is every paper still `submitted` in `--data` (desk-rejected
and `--exclude-pids` papers dropped) that holds at least `--min-reviews`
submitted reviews in `--reviews`, HotCRP's review CSV export, which carries
submitted reviews only. **TRC reviews are left out by default** -- the
Training Review Committee are PhD students reviewing alongside the PC -- both
from the review count and from the scores; `--include-trc` counts them. The
export does not name a review's round, so TRC reviews are identified from the
action log (`--log`), which also supplies how many assigned R1 reviews each
paper is still waiting on: while reviews are coming in, a population of
"papers with 4+ reviews" is mostly papers with one still to arrive, and
`--complete-only` restricts it to papers with none outstanding.

Offline, instant, read-only apart from its two reports: `--html`, a
self-contained chart page, and `--papers-csv`, one row per paper in the
population. Both carry confidential review data and are gitignored.
"""

from __future__ import annotations

from reviewer_match.paths import input_path, report_path

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime
from fractions import Fraction

from reviewer_match import hotcrp_log
from reviewer_match import paper_matching
from reviewer_match.review_scores import (
    COMPARATOR_SYMBOLS, COMPARATORS, DEFAULT_MIN_REVIEWS, NETS, average, caught,
    eligible_pids, load_review_scores, parse_thresholds, review_rounds,
)

DEFAULT_REVIEWS = input_path("hpca2027-reviews.csv")
DEFAULT_LOG = input_path("hpca2027-log.csv")
DEFAULT_DATA = input_path("hpca2027-data.json")
DEFAULT_HTML = report_path("revision_cutoffs.html")
DEFAULT_PAPERS_CSV = report_path("revision_cutoffs_papers.csv")
DEFAULT_THRESHOLDS = "1,1.5,2,2.5,3"
DEFAULT_BIN_WIDTH = 0.25

PAPERS_FIELDS = [
    "pid", "title", "reviews", "outstanding", "scores", "average",
    "max_score", "scores_at_least_3",
]


def tabulate(
    population: dict[int, list[int]], thresholds: list[float]
) -> dict[str, list[list[int]]]:
    """{comparator: [[papers caught at each cutoff] for each net in NETS]}."""
    return {
        comparator: [
            [
                sum(caught(s, t, comparator, condition) for s in population.values())
                for t in thresholds
            ]
            for _, _, _, condition in NETS
        ]
        for comparator in COMPARATORS
    }


def histogram(population: dict[int, list[int]], width: float) -> list[dict[str, float]]:
    """[{lo, hi, n}] over (lo, hi] bins of paper averages, the first ending at 1.0.

    Upper-inclusive to match the "<=" reading of a cutoff, so the bin ending at
    2.0 holds exactly the papers that "<= 2.0" gains over "<= 1.75".
    """
    step = Fraction(str(width))
    top = max(average(s) for s in population.values())
    count = max(1, math.ceil((top - 1) / step)) + 1
    bins = [{"lo": float(1 - step + i * step), "hi": float(1 + i * step), "n": 0} for i in range(count)]
    for scores in population.values():
        index = max(0, math.ceil((average(scores) - 1) / step))
        bins[index]["n"] += 1
    return bins


def format_cutoff(t: float) -> str:
    return f"{t:.1f}" if float(t * 2).is_integer() else f"{t:.2f}"


def format_table(counts: dict[str, list[list[int]]], thresholds: list[float], total: int) -> str:
    """The stdout report: one block per comparator, cutoffs down, nets across."""
    width = max(13, *(len(short) for _, _, short, _ in NETS)) + 2
    lines = []
    for comparator in COMPARATORS:
        symbol = COMPARATOR_SYMBOLS[comparator]
        lines.append(f"Caught at average {symbol} cutoff (share of {total} papers)")
        lines.append("  " + "cutoff".ljust(8) + "".join(short.rjust(width) for _, _, short, _ in NETS))
        for j, t in enumerate(thresholds):
            cells = "".join(
                f"{100 * counts[comparator][i][j] / total:5.1f}% ({counts[comparator][i][j]:>4})".rjust(width)
                for i in range(len(NETS))
            )
            lines.append("  " + f"{symbol} {format_cutoff(t)}".ljust(8) + cells)
        lines.append("")
    return "\n".join(lines)


def write_papers_csv(
    path: str, population: dict[int, list[int]], titles: dict[int, str], outstanding: Counter
) -> None:
    """One row per paper in the population, lowest average first."""
    ordered = sorted(population, key=lambda pid: (average(population[pid]), pid))
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(PAPERS_FIELDS)
        for pid in ordered:
            s = population[pid]
            writer.writerow([
                pid, titles.get(pid, ""), len(s), outstanding[pid],
                " ".join(str(x) for x in sorted(s)), f"{float(average(s)):.3f}",
                max(s), sum(x >= 3 for x in s),
            ])
    os.replace(tmp, path)


def write_text(path: str, text: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def render_html(payload: dict) -> str:
    """The chart page: HTML_TEMPLATE with the payload embedded as inert JSON."""
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__PAYLOAD__", data)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--reviews", default=DEFAULT_REVIEWS, help="HotCRP review CSV export")
    parser.add_argument("--log", default=DEFAULT_LOG, help="HotCRP action log (review rounds, outstanding reviews)")
    parser.add_argument("--data", default=DEFAULT_DATA, help="HotCRP paper export JSON")
    parser.add_argument("--exclude-pids", default="", help="comma-separated paper IDs to leave out")
    parser.add_argument(
        "--thresholds", default=DEFAULT_THRESHOLDS,
        help=f"comma-separated average-score cutoffs (default {DEFAULT_THRESHOLDS})"
    )
    parser.add_argument(
        "--min-reviews", type=int, default=DEFAULT_MIN_REVIEWS,
        help=f"papers need at least this many submitted reviews (default {DEFAULT_MIN_REVIEWS})"
    )
    parser.add_argument(
        "--complete-only", action="store_true",
        help="only papers with no assigned R1 review still outstanding"
    )
    parser.add_argument(
        "--include-trc", action="store_true",
        help="count Training Review Committee reviews in the count and the average"
    )
    parser.add_argument(
        "--bin-width", type=float, default=DEFAULT_BIN_WIDTH,
        help=f"histogram bin width in score points (default {DEFAULT_BIN_WIDTH})"
    )
    parser.add_argument("--html", default=DEFAULT_HTML, help="chart page to write")
    parser.add_argument("--papers-csv", default=DEFAULT_PAPERS_CSV, help="per-paper CSV to write")
    args = parser.parse_args()

    thresholds = parse_thresholds(args.thresholds)
    exclude = paper_matching.parse_exclude_pids(args.exclude_pids)
    for path in (args.reviews, args.log, args.data):
        if not os.path.exists(path):
            print(f"ERROR: {path} not found", file=sys.stderr)
            return 1

    log_rows = hotcrp_log.load_log(args.log)
    training, outstanding = review_rounds(log_rows)
    skip = set() if args.include_trc else training
    scores, titles, skipped = load_review_scores(args.reviews, skip)
    eligible, desk_rejected = eligible_pids(args.data, exclude)

    # A TRC review the export holds under a different address than the log
    # would be counted as a PC review; say so rather than absorb it.
    in_export = set(scores) | {pid for pid, _ in skipped}
    unmatched = sorted(pair for pair in training - skipped if pair[0] in in_export)
    if not args.include_trc and unmatched:
        print(f"WARNING: {len(unmatched)} submitted TRC review(s) in the log were not found in "
              f"{args.reviews} and may be counted as PC reviews: "
              + ", ".join(f"#{pid}" for pid, _ in unmatched), file=sys.stderr)

    population = {
        pid: s for pid, s in scores.items()
        if pid in eligible and len(s) >= args.min_reviews
        and not (args.complete_only and outstanding[pid])
    }
    if not population:
        print("ERROR: no paper meets the population criteria", file=sys.stderr)
        return 1
    total = len(population)
    waiting = sum(1 for pid in population if outstanding[pid])
    dropped_desk = sum(1 for pid in desk_rejected if len(scores.get(pid, ())) >= args.min_reviews)
    trc_note = "TRC reviews counted" if args.include_trc else f"{len(skipped)} TRC reviews left out"

    print(f"{len(scores)} papers hold at least one review ({trc_note}).", file=sys.stderr)
    print(f"{total} papers have >= {args.min_reviews} submitted reviews"
          + (" and none outstanding." if args.complete_only
             else f"; {waiting} of them still have an assigned review outstanding."), file=sys.stderr)
    if dropped_desk:
        print(f"{dropped_desk} desk-rejected paper(s) with enough reviews left out.", file=sys.stderr)

    counts = tabulate(population, thresholds)
    print(format_table(counts, thresholds, total))

    log_through = log_rows[-1]["date"] if log_rows else ""
    reviews_at = datetime.fromtimestamp(os.path.getmtime(args.reviews)).strftime("%Y-%m-%d %H:%M")
    payload = {
        "reviewsAt": reviews_at,
        "logThrough": log_through[:16],
        "population": {
            "papers": total,
            "withReviews": len(scores),
            "outstanding": waiting,
            "minReviews": args.min_reviews,
            "completeOnly": args.complete_only,
            "trcReviews": len(skipped),
            "includeTrc": args.include_trc,
        },
        "thresholds": thresholds,
        "nets": [{"key": key, "label": label, "short": short} for key, label, short, _ in NETS],
        "counts": counts,
        "histogram": histogram(population, args.bin_width),
    }
    write_text(args.html, render_html(payload))
    write_papers_csv(args.papers_csv, population, titles, outstanding)
    print(f"Wrote {args.html} and {args.papers_csv}", file=sys.stderr)
    return 0


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Revision Cutoff Analysis</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --hairline: rgba(11,11,11,0.10); --wash: rgba(11,11,11,0.04);
  --net-average: #0d366b; --net-no4: #1c5cab; --net-one3: #2a78d6; --net-no3: #6da7ec;
  --bar: #2a78d6; --bar-hover: #1c5cab;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --hairline: rgba(255,255,255,0.10); --wash: rgba(255,255,255,0.05);
    --net-average: #b7d3f6; --net-no4: #6da7ec; --net-one3: #2a78d6; --net-no3: #184f95;
    --bar: #3987e5; --bar-hover: #6da7ec;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --hairline: rgba(255,255,255,0.10); --wash: rgba(255,255,255,0.05);
  --net-average: #b7d3f6; --net-no4: #6da7ec; --net-one3: #2a78d6; --net-no3: #184f95;
  --bar: #3987e5; --bar-hover: #6da7ec;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--page); color: var(--ink);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
}
main { max-width: 980px; margin: 0 auto; padding: 32px 16px 48px; }
h1 { font-size: 24px; line-height: 1.25; margin: 0 0 6px; font-weight: 650; }
h2 { font-size: 16px; margin: 0 0 2px; font-weight: 600; }
p { margin: 0; }
.lede { color: var(--ink-2); max-width: 72ch; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 24px 0; }
.tile { background: var(--surface); border: 1px solid var(--hairline); border-radius: 12px; padding: 14px 16px; }
.tile .label { color: var(--ink-2); font-size: 13px; }
.tile .value { font-size: 28px; font-weight: 600; line-height: 1.2; margin-top: 2px; }
.tile .sub { color: var(--muted); font-size: 12px; }
.controls { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin: 8px 0 12px; }
.controls .label { color: var(--ink-2); font-size: 13px; }
.seg { display: inline-flex; border: 1px solid var(--hairline); border-radius: 9px; padding: 2px; background: var(--surface); }
.seg button {
  font: inherit; font-size: 13px; color: var(--ink-2); background: none; border: 0;
  padding: 5px 12px; border-radius: 7px; cursor: pointer;
}
.seg button[aria-pressed="true"] { background: var(--wash); color: var(--ink); font-weight: 600; }
.seg button:focus-visible { outline: 2px solid var(--net-one3); outline-offset: 1px; }
.card { background: var(--surface); border: 1px solid var(--hairline); border-radius: 12px; padding: 18px 16px 12px; margin-bottom: 16px; position: relative; }
.card .sub { color: var(--ink-2); font-size: 13px; margin-bottom: 10px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; margin: 4px 0 6px; font-size: 13px; color: var(--ink-2); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.key { display: inline-block; width: 16px; height: 2px; border-radius: 1px; }
.chart { width: 100%; }
.chart svg { display: block; overflow: visible; }
.chart svg:focus-visible { outline: 2px solid var(--net-one3); outline-offset: 4px; border-radius: 4px; }
svg text { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; fill: var(--muted); font-variant-numeric: tabular-nums; }
svg .endlabel { fill: var(--ink-2); }
svg .endvalue { fill: var(--ink); font-weight: 600; }
svg .grid { stroke: var(--grid); stroke-width: 1; }
svg .axis { stroke: var(--axis); stroke-width: 1; }
svg .cross { stroke: var(--axis); stroke-width: 1; }
.tip {
  position: absolute; pointer-events: none; z-index: 2; min-width: 180px;
  background: var(--surface); border: 1px solid var(--hairline); border-radius: 10px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.12); padding: 8px 10px; font-size: 13px; display: none;
}
.tip .head { color: var(--ink-2); margin-bottom: 4px; }
.tip .row { display: flex; align-items: center; gap: 8px; line-height: 1.7; }
.tip .row strong { font-variant-numeric: tabular-nums; min-width: 48px; }
.tip .row .n { color: var(--muted); font-variant-numeric: tabular-nums; min-width: 34px; }
.tip .row .name { color: var(--ink-2); }
.tablewrap { overflow-x: auto; margin-top: 8px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; font-variant-numeric: tabular-nums; }
th, td { padding: 7px 10px; text-align: right; border-bottom: 1px solid var(--grid); white-space: nowrap; }
th { color: var(--ink-2); font-weight: 600; }
th:first-child, td:first-child { text-align: left; }
td .n { color: var(--muted); margin-left: 6px; }
th .key { vertical-align: middle; margin-right: 6px; }
details { margin-top: 8px; font-size: 13px; }
summary { color: var(--ink-2); cursor: pointer; }
.notes { color: var(--ink-2); font-size: 13px; max-width: 76ch; }
.notes h2 { color: var(--ink); margin-top: 20px; }
.notes ul { padding-left: 18px; margin: 6px 0; }
.notes li { margin: 3px 0; }
</style>
</head>
<body>
<main>
  <h1>Revision cutoff analysis</h1>
  <p class="lede" id="lede"></p>
  <div class="tiles" id="tiles"></div>

  <div class="controls">
    <span class="label">Paper is caught when its average is</span>
    <div class="seg" role="group" aria-label="Cutoff comparison">
      <button type="button" data-cmp="le" aria-pressed="true">at or below the cutoff (≤)</button>
      <button type="button" data-cmp="lt" aria-pressed="false">strictly below it (&lt;)</button>
    </div>
  </div>

  <section class="card">
    <h2>Share of papers each net would catch</h2>
    <p class="sub" id="lines-sub"></p>
    <div class="legend" id="legend"></div>
    <div class="chart" id="lines"></div>
    <div class="tip" id="lines-tip"></div>
    <div class="tablewrap"><table id="nets-table"></table></div>
  </section>

  <section class="card">
    <h2>Distribution of paper averages</h2>
    <p class="sub">Papers per 0.25-point band of average pre-rebuttal overall merit. Each column covers averages above its left edge up to and including its right edge, so the column ending at 2.0 is exactly what “≤ 2.0” adds over “≤ 1.75”. Not affected by the ≤ / &lt; switch.</p>
    <div class="chart" id="hist"></div>
    <div class="tip" id="hist-tip"></div>
    <details><summary>Show as table</summary><div class="tablewrap"><table id="hist-table"></table></div></details>
  </section>

  <section class="notes">
    <h2>How to read this</h2>
    <ul id="net-notes"></ul>
    <ul id="pop-notes"></ul>
  </section>
</main>

<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
(function () {
  "use strict";
  var D = JSON.parse(document.getElementById("payload").textContent);
  var P = D.population;
  var N = P.papers;
  var NS = "http://www.w3.org/2000/svg";
  var cmp = "le";
  var SYM = { le: "≤", lt: "<" };

  function cut(t) { return (t * 2) % 1 === 0 ? t.toFixed(1) : t.toFixed(2); }
  function pct(n) { return (100 * n / N).toFixed(1) + "%"; }
  function netColor(net) { return "var(--net-" + net.key + ")"; }
  function svgEl(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function htmlEl(tag, cls, text, parent) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function key(color, parent) {
    var k = htmlEl("span", "key", null, parent);
    k.style.background = color;
    return k;
  }
  function placeTip(tip, card, x, y) {
    var cw = card.clientWidth, tw = tip.offsetWidth;
    var left = x + 14;
    if (left + tw > cw - 8) left = Math.max(8, x - tw - 14);
    tip.style.left = left + "px";
    tip.style.top = Math.max(8, y - 12) + "px";
  }

  // ---- text ----------------------------------------------------------------
  var reviewsWord = P.minReviews + (P.completeOnly ? " or more submitted reviews and none outstanding" : " or more submitted reviews");
  document.getElementById("lede").textContent =
    "What share of the " + N + " papers with " + reviewsWord +
    " each revision-eligibility net would catch, by average pre-rebuttal overall merit. " +
    (P.includeTrc ? "Training Review Committee reviews are counted." : "Training Review Committee (TRC) reviews are left out of both the count and the average.") +
    " Reviews export " + D.reviewsAt + "; HotCRP log through " + D.logThrough + ".";

  var tiles = document.getElementById("tiles");
  [
    ["Papers analysed", N, "with " + P.minReviews + "+ reviews"],
    ["Still awaiting a review", P.outstanding, (100 * P.outstanding / N).toFixed(0) + "% of those analysed"],
    ["Papers with any review", P.withReviews, "before the " + P.minReviews + "-review floor"],
  ].forEach(function (t) {
    var d = htmlEl("div", "tile", null, tiles);
    htmlEl("div", "label", t[0], d);
    htmlEl("div", "value", t[1].toLocaleString(), d);
    htmlEl("div", "sub", t[2], d);
  });

  var legend = document.getElementById("legend");
  D.nets.forEach(function (net) {
    var s = htmlEl("span", null, null, legend);
    key(netColor(net), s);
    s.appendChild(document.createTextNode(net.label));
  });

  var notes = document.getElementById("net-notes");
  [
    "Every net starts from the same test: the paper's average pre-rebuttal overall merit against the cutoff. The three “+” nets add a condition, so they catch fewer papers and spare those with a reviewer arguing for them.",
    "“+ no score of 4 or better”: no reviewer gave it a 4 or 5.",
    "“+ at most one score of 3 or better”: no more than one reviewer gave it 3 or higher.",
    "“+ no score of 3 or better”: every reviewer gave it 1 or 2. Averages of such papers never exceed 2, so this net stops growing at a 2.0 cutoff.",
  ].forEach(function (t) { htmlEl("li", null, t, notes); });
  var popNotes = document.getElementById("pop-notes");
  [
    "Scores are integers and most papers have four to six of them, so many averages land exactly on a cutoff. Use the switch above to see how much “≤” and “<” differ; the policy needs to say which it means.",
    P.outstanding + " of the " + N + " papers analysed still have an assigned review outstanding, so their averages can still move. Re-run with --complete-only to see papers whose reviews are all in.",
    "Desk-rejected papers are left out.",
  ].forEach(function (t) { htmlEl("li", null, t, popNotes); });

  // ---- line chart -------------------------------------------------------------
  var linesBox = document.getElementById("lines");
  var linesTip = document.getElementById("lines-tip");
  var linesCard = linesBox.parentNode;

  function renderLines() {
    linesBox.textContent = "";
    linesTip.style.display = "none";
    document.getElementById("lines-sub").textContent =
      "Share of the " + N + " papers whose average is " + (cmp === "le" ? "at or below" : "strictly below") +
      " each cutoff, with each net's extra condition applied.";
    var counts = D.counts[cmp];
    var th = D.thresholds;
    var W = Math.max(280, linesBox.clientWidth);
    var narrow = W < 600;
    var H = 320;
    var m = { t: 14, r: narrow ? 18 : 176, b: 44, l: 46 };
    var pw = W - m.l - m.r, ph = H - m.t - m.b;
    var x0 = th[0], x1 = th[th.length - 1];
    function X(v) { return m.l + (x1 === x0 ? pw / 2 : (v - x0) / (x1 - x0) * pw); }
    function Y(n) { return m.t + (1 - n / N) * ph; }
    var svg = svgEl("svg", { width: W, height: H, viewBox: "0 0 " + W + " " + H, tabindex: "0", role: "img",
      "aria-label": "Line chart of the share of papers caught at each cutoff by each net. The table below lists every value." }, linesBox);

    [0, 0.25, 0.5, 0.75, 1].forEach(function (f) {
      var y = m.t + (1 - f) * ph;
      svgEl("line", { x1: m.l, x2: m.l + pw, y1: y, y2: y, "class": f === 0 ? "axis" : "grid" }, svg);
      var t = svgEl("text", { x: m.l - 8, y: y + 4, "text-anchor": "end" }, svg);
      t.textContent = (f * 100) + "%";
    });
    th.forEach(function (t) {
      var lab = svgEl("text", { x: X(t), y: m.t + ph + 18, "text-anchor": "middle" }, svg);
      lab.textContent = SYM[cmp] + " " + cut(t);
    });
    var xt = svgEl("text", { x: m.l + pw / 2, y: H - 6, "text-anchor": "middle" }, svg);
    xt.textContent = "Average overall-merit cutoff";

    var cross = svgEl("line", { "class": "cross", y1: m.t, y2: m.t + ph, visibility: "hidden" }, svg);

    D.nets.forEach(function (net, i) {
      var d = th.map(function (t, j) { return (j ? "L" : "M") + X(t).toFixed(1) + " " + Y(counts[i][j]).toFixed(1); }).join(" ");
      var path = svgEl("path", { d: d, fill: "none", "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
      path.style.stroke = netColor(net);
    });
    D.nets.forEach(function (net, i) {
      th.forEach(function (t, j) {
        var c = svgEl("circle", { cx: X(t), cy: Y(counts[i][j]), r: 4, "stroke-width": 2 }, svg);
        c.style.fill = netColor(net);
        c.style.stroke = "var(--surface)";
      });
    });

    // Direct end labels, only when they cannot collide; the legend and table carry them otherwise.
    if (!narrow) {
      var last = th.length - 1;
      var ends = D.nets.map(function (net, i) { return { net: net, y: Y(counts[i][last]), n: counts[i][last] }; });
      var ys = ends.map(function (e) { return e.y; }).sort(function (a, b) { return a - b; });
      var clear = ys.every(function (y, k) { return k === 0 || y - ys[k - 1] >= 16; });
      if (clear) {
        ends.forEach(function (e) {
          var t = svgEl("text", { x: X(x1) + 12, y: e.y + 4 }, svg);
          var v = svgEl("tspan", { "class": "endvalue" }, t);
          v.textContent = pct(e.n);
          var l = svgEl("tspan", { "class": "endlabel", dx: 6 }, t);
          l.textContent = e.net.short;
        });
      }
    }

    var hit = svgEl("rect", { x: m.l - 20, y: m.t, width: pw + 40, height: ph, fill: "transparent" }, svg);
    var current = -1;
    function show(j) {
      current = j;
      var x = X(th[j]);
      cross.setAttribute("x1", x); cross.setAttribute("x2", x);
      cross.setAttribute("visibility", "visible");
      linesTip.textContent = "";
      htmlEl("div", "head", "Average " + SYM[cmp] + " " + cut(th[j]), linesTip);
      D.nets.forEach(function (net, i) {
        var row = htmlEl("div", "row", null, linesTip);
        key(netColor(net), row);
        htmlEl("strong", null, pct(counts[i][j]), row);
        htmlEl("span", "n", String(counts[i][j]), row);
        htmlEl("span", "name", net.short, row);
      });
      linesTip.style.display = "block";
      var top = Math.min.apply(null, counts.map(function (c) { return Y(c[j]); }));
      placeTip(linesTip, linesCard, linesBox.offsetLeft + x, linesBox.offsetTop + top);
    }
    function hide() { current = -1; cross.setAttribute("visibility", "hidden"); linesTip.style.display = "none"; }
    function nearest(px) {
      var best = 0;
      th.forEach(function (t, j) { if (Math.abs(X(t) - px) < Math.abs(X(th[best]) - px)) best = j; });
      return best;
    }
    hit.addEventListener("pointermove", function (ev) {
      var r = svg.getBoundingClientRect();
      show(nearest(ev.clientX - r.left));
    });
    hit.addEventListener("pointerleave", hide);
    svg.addEventListener("focus", function () { show(current < 0 ? 0 : current); });
    svg.addEventListener("blur", hide);
    svg.addEventListener("keydown", function (ev) {
      if (ev.key === "ArrowRight") { show(Math.min(th.length - 1, current + 1)); ev.preventDefault(); }
      if (ev.key === "ArrowLeft") { show(Math.max(0, current - 1)); ev.preventDefault(); }
      if (ev.key === "Escape") hide();
    });
  }

  function renderNetsTable() {
    var table = document.getElementById("nets-table");
    table.textContent = "";
    var head = htmlEl("tr", null, null, htmlEl("thead", null, null, table));
    htmlEl("th", null, "Cutoff", head);
    D.nets.forEach(function (net) {
      var th = htmlEl("th", null, null, head);
      key(netColor(net), th);
      th.appendChild(document.createTextNode(net.short));
    });
    var body = htmlEl("tbody", null, null, table);
    D.thresholds.forEach(function (t, j) {
      var tr = htmlEl("tr", null, null, body);
      htmlEl("td", null, "Average " + SYM[cmp] + " " + cut(t), tr);
      D.counts[cmp].forEach(function (c) {
        var td = htmlEl("td", null, pct(c[j]), tr);
        htmlEl("span", "n", "(" + c[j] + ")", td);
      });
    });
  }

  // ---- histogram --------------------------------------------------------------
  var histBox = document.getElementById("hist");
  var histTip = document.getElementById("hist-tip");
  var histCard = histBox.parentNode;
  var bins = D.histogram;

  function niceStep(max) {
    var raw = max / 4, p = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / p;
    return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * p;
  }

  function renderHist() {
    histBox.textContent = "";
    histTip.style.display = "none";
    var W = Math.max(280, histBox.clientWidth), H = 240;
    var m = { t: 12, r: 12, b: 40, l: 46 };
    var pw = W - m.l - m.r, ph = H - m.t - m.b;
    var lo = bins[0].lo, hi = bins[bins.length - 1].hi;
    function X(v) { return m.l + (v - lo) / (hi - lo) * pw; }
    var maxN = Math.max.apply(null, bins.map(function (b) { return b.n; }));
    var step = niceStep(maxN), top = Math.ceil(maxN / step) * step;
    function Y(n) { return m.t + (1 - n / top) * ph; }
    var svg = svgEl("svg", { width: W, height: H, viewBox: "0 0 " + W + " " + H, role: "img",
      "aria-label": "Column chart of papers per band of average score. The table under 'Show as table' lists every value." }, histBox);
    for (var g = 0; g <= top; g += step) {
      svgEl("line", { x1: m.l, x2: m.l + pw, y1: Y(g), y2: Y(g), "class": g === 0 ? "axis" : "grid" }, svg);
      var yl = svgEl("text", { x: m.l - 8, y: Y(g) + 4, "text-anchor": "end" }, svg);
      yl.textContent = g.toLocaleString();
    }
    for (var v = Math.ceil(lo * 2) / 2; v <= hi + 1e-9; v += 0.5) {
      var xl = svgEl("text", { x: X(v), y: m.t + ph + 18, "text-anchor": "middle" }, svg);
      xl.textContent = v.toFixed(1);
    }
    var xt = svgEl("text", { x: m.l + pw / 2, y: H - 6, "text-anchor": "middle" }, svg);
    xt.textContent = "Average pre-rebuttal overall merit";
    var band = X(bins[0].hi) - X(bins[0].lo);
    var bw = Math.max(3, Math.min(24, band - 2));
    bins.forEach(function (b) {
      var cx = (X(b.lo) + X(b.hi)) / 2;
      var g = svgEl("g", { tabindex: "0" }, svg);
      svgEl("rect", { x: X(b.lo), y: m.t, width: band, height: ph, fill: "transparent" }, g);
      var bar = null;
      if (b.n > 0) {
        var y = Y(b.n), h = m.t + ph - y, r = Math.min(4, h, bw / 2), x = cx - bw / 2;
        var d = "M" + x + " " + (m.t + ph) + " V" + (y + r) + " Q" + x + " " + y + " " + (x + r) + " " + y +
                " H" + (x + bw - r) + " Q" + (x + bw) + " " + y + " " + (x + bw) + " " + (y + r) + " V" + (m.t + ph) + " Z";
        bar = svgEl("path", { d: d }, g);
        bar.style.fill = "var(--bar)";
      }
      function show() {
        if (bar) bar.style.fill = "var(--bar-hover)";
        histTip.textContent = "";
        htmlEl("div", "head", "Average above " + cut(b.lo) + ", up to " + cut(b.hi), histTip);
        var row = htmlEl("div", "row", null, histTip);
        htmlEl("strong", null, b.n + " papers", row);
        htmlEl("span", "name", pct(b.n) + " of those analysed", row);
        histTip.style.display = "block";
        placeTip(histTip, histCard, histBox.offsetLeft + cx, histBox.offsetTop + Y(b.n) - 30);
      }
      function hide() { if (bar) bar.style.fill = "var(--bar)"; histTip.style.display = "none"; }
      g.addEventListener("pointerenter", show);
      g.addEventListener("pointerleave", hide);
      g.addEventListener("focus", show);
      g.addEventListener("blur", hide);
    });
  }

  (function histTable() {
    var table = document.getElementById("hist-table");
    var head = htmlEl("tr", null, null, htmlEl("thead", null, null, table));
    ["Average above", "Up to and including", "Papers", "Share"].forEach(function (h) { htmlEl("th", null, h, head); });
    var body = htmlEl("tbody", null, null, table);
    bins.forEach(function (b) {
      var tr = htmlEl("tr", null, null, body);
      htmlEl("td", null, cut(b.lo), tr);
      htmlEl("td", null, cut(b.hi), tr);
      htmlEl("td", null, String(b.n), tr);
      htmlEl("td", null, pct(b.n), tr);
    });
  })();

  // ---- wiring -----------------------------------------------------------------
  var buttons = document.querySelectorAll(".seg button");
  Array.prototype.forEach.call(buttons, function (b) {
    b.addEventListener("click", function () {
      cmp = b.getAttribute("data-cmp");
      Array.prototype.forEach.call(buttons, function (o) { o.setAttribute("aria-pressed", String(o === b)); });
      renderLines();
      renderNetsTable();
    });
  });
  function renderAll() { renderLines(); renderNetsTable(); renderHist(); }
  renderAll();
  var lastW = linesBox.clientWidth;
  new ResizeObserver(function () {
    var w = linesBox.clientWidth;
    if (w !== lastW) { lastW = w; renderLines(); renderHist(); }
  }).observe(linesBox);
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
