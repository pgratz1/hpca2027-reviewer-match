"""Score completed blinded ratings for the abstract-enrichment experiment.

    python score_abstract_evaluation.py \
      --ratings abstract-evaluation-ratings.csv \
      --key abstract-evaluation-key.csv

Reports nDCG@10, capable-or-expert fraction in the top six, and unsuitable
reviewers in the top six for title-only and abstract-enriched rankings.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict


def dcg(relevances: list[int], k: int = 10) -> float:
    return sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ratings", default="abstract-evaluation-ratings.csv")
    parser.add_argument("--key", default="abstract-evaluation-key.csv")
    args = parser.parse_args()

    ratings = {}
    with open(args.ratings, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = row.get("expertise_rating_0_to_3", "").strip()
            if raw not in {"0", "1", "2", "3"}:
                parser.error(f"missing/invalid rating for paper {row['paper_pid']} / {row['reviewer_email']}")
            ratings[(row["paper_pid"], row["reviewer_email"].lower())] = int(raw)

    ranked = {"baseline": defaultdict(list), "enriched": defaultdict(list)}
    with open(args.key, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["paper_pid"], row["reviewer_email"].lower())
            if key not in ratings:
                parser.error(f"no rating for paper {key[0]} / {key[1]}")
            for model in ranked:
                raw_rank = row[f"{model}_rank"].strip()
                if raw_rank:
                    ranked[model][row["paper_pid"]].append((int(raw_rank), ratings[key]))

    for model in ("baseline", "enriched"):
        ndcgs, capable, unsuitable = [], [], []
        for pid, ranked_ratings in ranked[model].items():
            rels = [rating for _, rating in sorted(ranked_ratings)]
            ideal = sorted(
                [rating for (paper_pid, _), rating in ratings.items() if paper_pid == pid],
                reverse=True,
            )
            ideal_dcg = dcg(ideal)
            ndcgs.append(dcg(rels) / ideal_dcg if ideal_dcg else 1.0)
            top_six = rels[:6]
            capable.append(sum(r >= 2 for r in top_six) / len(top_six) if top_six else 0.0)
            unsuitable.append(sum(r == 0 for r in top_six))
        if not ndcgs:
            parser.error(f"no ranked papers found for {model}")
        print(
            f"{model}: papers={len(ndcgs)} mean_nDCG@10={sum(ndcgs)/len(ndcgs):.4f} "
            f"capable_or_expert@6={sum(capable)/len(capable):.4f} "
            f"unsuitable@6={sum(unsuitable)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
