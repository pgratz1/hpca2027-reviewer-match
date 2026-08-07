"""Compare the true SPECTER2 affinity of assignments from several arms.

    make baselines
    python -m scripts.compare_baselines outputs/evaluations/pairs-arm*.csv
    python -m scripts.compare_baselines a.csv b.csv --baseline b --worst 100

Reads the --pairs-csv files assign_reviewers.py writes, one per arm, and
reports how much match quality each arm reached. The arm label is the file's
stem with a leading "pairs-" stripped, so pairs-armB-s1.csv is "armB-s1".

The point is the randomized baselines: an arm run with --score-mode random
ranks on noise while every COI layer, the senior anchor, the junior and
out-of-area caps, the same-country cap and every load cap still bind, so the
gap between it and the production arm is what the embedding is worth.

Two things this does that reading the transcripts' "=== Match goodness ==="
blocks cannot. It restricts every mean to the papers scored in EVERY arm --
paper_goodness is None for a paper with no reviewers and the report drops those
from its mean, so arms that fill different numbers of papers otherwise average
different paper sets. And it prints a pair-weighted mean beside the
mean-of-paper-means, because a partly filled slate has fewer marginal picks and
so a higher mean: under-filling flatters a random arm and understates the drop.

Compare arms run at --surplus-per-paper 0. Full-slate figures are not
comparable across arms; see the banner a randomized run prints.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from reviewer_match.paths import evaluation_path

# Papers in the worst tail to summarize separately. The tail is where a matcher
# is actually under stress -- means hide it -- and 50 is the width the
# same-country/junior sweep in the README already reports.
DEFAULT_WORST = 50


def load_arm(path: str) -> dict[int, list[float]]:
    """{pid: [affinity of each assigned reviewer]} from one --pairs-csv."""
    by_pid: dict[int, list[float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = {"pid", "affinity"} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{path}: not a --pairs-csv; missing column(s) {', '.join(sorted(missing))}"
            )
        for row in reader:
            by_pid.setdefault(int(row["pid"]), []).append(float(row["affinity"]))
    return by_pid


def arm_label(path: str) -> str:
    stem = Path(path).stem
    return stem[len("pairs-"):] if stem.startswith("pairs-") else stem


def mean(values) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile of an already-sorted-able list, q in [0, 1]."""
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]


def fmt(value: float | None, places: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{places}f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "pairs_csv", nargs="+",
        help=f"--pairs-csv files, one per arm (typically under {evaluation_path('')})"
    )
    parser.add_argument(
        "--baseline", metavar="LABEL",
        help="arm every delta is measured against (default: the first file given)"
    )
    parser.add_argument(
        "--worst", type=int, default=DEFAULT_WORST, metavar="N",
        help="size of the worst-matched tail summarized separately (default: %(default)s)"
    )
    args = parser.parse_args()

    if args.worst < 1:
        parser.error("--worst must be at least 1")

    arms: dict[str, dict[int, list[float]]] = {}
    for path in args.pairs_csv:
        label = arm_label(path)
        if label in arms:
            parser.error(f"two files share the arm label {label!r}: rename one")
        try:
            arms[label] = load_arm(path)
        except (OSError, ValueError, KeyError) as exc:
            parser.error(str(exc))

    labels = list(arms)
    baseline = args.baseline or labels[0]
    if baseline not in arms:
        parser.error(f"--baseline {baseline!r} is not one of {', '.join(labels)}")

    # Every mean below is over this set and no other. An arm that filled fewer
    # papers would otherwise be averaging a different, easier population.
    shared = set.intersection(*(set(a) for a in arms.values()))
    if not shared:
        parser.error("no paper is scored in every arm; nothing to compare")
    goodness = {
        label: {pid: mean(arms[label][pid]) for pid in shared} for label in labels
    }

    print("=== Arms ===")
    for label in labels:
        by_pid = arms[label]
        pairs = sum(len(v) for v in by_pid.values())
        widest = max(len(v) for v in by_pid.values())
        full = sum(1 for v in by_pid.values() if len(v) >= widest)
        print(f"  {label:16s} {pairs:6d} pair(s), {len(by_pid):5d} paper(s) with a "
              f"reviewer, {full:5d} at {widest}")
    print(f"\n{len(shared)} paper(s) scored in every arm; every figure below is over "
          f"those and no others.")

    print("\n=== True SPECTER2 affinity ===")
    print(f"  {'arm':16s} {'paper mean':>11s} {'pair mean':>10s} "
          f"{'worst-' + str(args.worst):>10s} {'std':>7s}")
    for label in labels:
        values = [goodness[label][pid] for pid in shared]
        paper_mean = mean(values)
        pair_mean = mean(v for pid in shared for v in arms[label][pid])
        tail = mean(sorted(values)[:args.worst])
        std = (sum((v - paper_mean) ** 2 for v in values) / len(values)) ** 0.5
        print(f"  {label:16s} {fmt(paper_mean):>11s} {fmt(pair_mean):>10s} "
              f"{fmt(tail):>10s} {std:>7.4f}")

    print(f"\n=== Per-paper delta vs {baseline} ===")
    print(f"  {'arm':16s} {'mean':>9s} {'median':>9s} {'p5':>9s} {'p95':>9s} {'higher':>8s}")
    for label in labels:
        if label == baseline:
            continue
        deltas = [goodness[label][pid] - goodness[baseline][pid] for pid in shared]
        higher = sum(1 for d in deltas if d > 0) / len(deltas)
        print(f"  {label:16s} {mean(deltas):>+9.4f} {percentile(deltas, 0.5):>+9.4f} "
              f"{percentile(deltas, 0.05):>+9.4f} {percentile(deltas, 0.95):>+9.4f} "
              f"{higher:>7.1%}")
    print(f"\n'higher' is the fraction of papers the arm matched BETTER than "
          f"{baseline}; for a randomized arm it should be small, and a large value "
          f"means the signal is not doing what the production run assumes.")

    # Seeds of one arm differ only in the draw, so their spread is the sampling
    # error on every number above -- the thing a single-seed run cannot show.
    families: dict[str, list[str]] = {}
    for label in labels:
        families.setdefault(label.rsplit("-s", 1)[0], []).append(label)
    repeated = {f: ls for f, ls in families.items() if len(ls) > 1}
    if repeated:
        print("\n=== Spread across seeds ===")
        for family, group in sorted(repeated.items()):
            means = [mean(goodness[label][pid] for pid in shared) for label in group]
            print(f"  {family:16s} {len(group)} seed(s): mean {fmt(mean(means))}, "
                  f"range {fmt(min(means))}–{fmt(max(means))} "
                  f"(spread {fmt(max(means) - min(means))})")
    else:
        print("\nOne seed per arm: the spread above cannot be estimated from this run. "
              "Re-run with more seeds before quoting a number that matters.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
