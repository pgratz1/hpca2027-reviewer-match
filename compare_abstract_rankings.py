"""Build a blinded chair-rating sample for title-only vs abstract rankings.

    python compare_abstract_rankings.py \
      --baseline-fingerprints fingerprints-title-only.json \
      --enriched-fingerprints fingerprints.json

Writes a randomized rating CSV with no model/rank labels and a separate
comparison CSV that reveals baseline and enriched ranks. Papers are selected
round-robin across their first HotCRP topic, prioritizing those whose top-N
candidate sets disagree most.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict, deque

import numpy as np

import fingerprint as fp
from paper_matching import eligible_scores, load_papers
from reviewers import load_reviewers

DEFAULT_CSV = "HPCA'27 PC Member Acceptance Form (Responses) - Form Responses 1.csv"
DEFAULT_DATA = "hpca2027-data.json"
DEFAULT_PAPER_CACHE = "paper_fingerprints.json"


def ranked_emails(paper, emails, matrix, paper_vec, reviewers_by_email, top):
    pairs = eligible_scores(paper, emails, matrix, paper_vec, reviewers_by_email)
    return [email for email, _ in sorted(pairs, key=lambda pair: -pair[1])[:top]]


def choose_stratified(rows: list[dict], sample_size: int) -> list[dict]:
    groups: dict[str, deque] = {}
    collected = defaultdict(list)
    for row in rows:
        topic = (row["paper"].get("topics") or ["Unspecified"])[0]
        collected[topic].append(row)
    for topic, topic_rows in collected.items():
        topic_rows.sort(key=lambda r: (-r["disagreement"], r["paper"]["pid"]))
        groups[topic] = deque(topic_rows)
    selected = []
    while len(selected) < sample_size and groups:
        for topic in sorted(list(groups)):
            if len(selected) >= sample_size:
                break
            selected.append(groups[topic].popleft())
            if not groups[topic]:
                del groups[topic]
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--paper-cache", default=DEFAULT_PAPER_CACHE)
    parser.add_argument("--baseline-fingerprints", required=True)
    parser.add_argument("--enriched-fingerprints", required=True)
    parser.add_argument("--rating-out", default="abstract-evaluation-ratings.csv")
    parser.add_argument("--comparison-out", default="abstract-evaluation-key.csv")
    parser.add_argument("--sample-size", type=int, default=24)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    if args.sample_size <= 0 or args.top <= 0:
        parser.error("--sample-size and --top must be greater than 0")

    papers = load_papers(args.data)
    paper_cache = fp.load_fingerprint_cache(args.paper_cache)
    missing = [str(p["pid"]) for p in papers if str(p["pid"]) not in paper_cache]
    if missing:
        parser.error("paper fingerprint cache is incomplete; run score_papers.py or assign_reviewers.py first")
    baseline = fp.load_fingerprint_cache(args.baseline_fingerprints)
    enriched = fp.load_fingerprint_cache(args.enriched_fingerprints)
    reviewers_by_email = {r.email: r for r in load_reviewers(args.csv)}
    emails = sorted(set(baseline) & set(enriched) & set(reviewers_by_email))
    if not emails:
        parser.error("baseline and enriched caches have no reviewers in common")
    base_matrix = np.asarray([baseline[e]["vector"] for e in emails], dtype=np.float32)
    enriched_matrix = np.asarray([enriched[e]["vector"] for e in emails], dtype=np.float32)

    comparisons = []
    for paper in papers:
        paper_vec = np.asarray(paper_cache[str(paper["pid"])]["vector"], dtype=np.float32)
        base_rank = ranked_emails(
            paper, emails, base_matrix, paper_vec, reviewers_by_email, args.top
        )
        enriched_rank = ranked_emails(
            paper, emails, enriched_matrix, paper_vec, reviewers_by_email, args.top
        )
        union = set(base_rank) | set(enriched_rank)
        disagreement = 1.0 - (len(set(base_rank) & set(enriched_rank)) / len(union) if union else 1.0)
        comparisons.append({
            "paper": paper, "baseline": base_rank, "enriched": enriched_rank,
            "disagreement": disagreement,
        })

    selected = choose_stratified(comparisons, min(args.sample_size, len(comparisons)))
    rng = random.Random(args.seed)
    rating_rows = []
    key_rows = []
    for item in selected:
        paper = item["paper"]
        candidates = sorted(set(item["baseline"]) | set(item["enriched"]))
        rng.shuffle(candidates)
        for order, email in enumerate(candidates, 1):
            reviewer = reviewers_by_email[email]
            rating_rows.append({
                "paper_pid": paper["pid"], "paper_title": paper["title"],
                "paper_abstract": paper.get("abstract", ""),
                "paper_topics": "; ".join(paper.get("topics", [])),
                "candidate_order": order, "reviewer_name": reviewer.name,
                "reviewer_email": email, "reviewer_primary": reviewer.primary,
                "expertise_rating_0_to_3": "", "notes": "",
            })
            key_rows.append({
                "paper_pid": paper["pid"], "reviewer_email": email,
                "baseline_rank": item["baseline"].index(email) + 1 if email in item["baseline"] else "",
                "enriched_rank": item["enriched"].index(email) + 1 if email in item["enriched"] else "",
                "top_set_disagreement": f"{item['disagreement']:.6f}",
            })

    if not rating_rows:
        parser.error("the selected papers have no eligible candidates")

    for path, rows in ((args.rating_out, rating_rows), (args.comparison_out, key_rows)):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(
        f"Wrote {len(selected)} papers / {len(rating_rows)} candidate ratings to "
        f"{args.rating_out}; model ranks are in {args.comparison_out}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
