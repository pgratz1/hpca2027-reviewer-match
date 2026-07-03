"""Shared paper-fingerprinting and eligibility logic for the paper-scoring scripts.

Used by both score_papers.py (independent per-paper top-N ranking) and
assign_reviewers.py (global load-capped assignment across all papers), so
the paper fingerprinting, COI exclusion, and area gate aren't duplicated.
"""

from __future__ import annotations

import json
import sys

import numpy as np

import fingerprint as fp
import specter2_model


def load_papers(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_paper_fingerprints(
    papers: list[dict],
    paper_cache: dict,
    cache_path: str,
    *,
    area_weight: float = 1.0,
    device: str = "cuda",
) -> None:
    """Fingerprint any of `papers` not already in `paper_cache`, in place.

    Each paper pools two SPECTER2 documents (fingerprint.pool): title+abstract
    (weight 1.0) and its declared topics (weight `area_weight`). Saves
    `paper_cache` to `cache_path` if anything was added; a no-op (no model
    load) if every paper is already cached.
    """
    pending = [p for p in papers if str(p["pid"]) not in paper_cache]
    if not pending:
        return

    print(f"Loading SPECTER2 on {device}...", file=sys.stderr)
    tokenizer, model = specter2_model.load_model(device=device)

    flat_texts: list[str] = []
    flat_owner: list[int] = []  # index into `pending`
    weights: list[list[float]] = [[] for _ in pending]

    for i, p in enumerate(pending):
        flat_texts.append(fp.doc_text(tokenizer, p["title"], p.get("abstract", "")))
        flat_owner.append(i)
        weights[i].append(1.0)

        flat_texts.append(fp.doc_text(tokenizer, ", ".join(p.get("topics", []))))
        flat_owner.append(i)
        weights[i].append(area_weight)

    vectors = specter2_model.encode_texts(flat_texts, tokenizer, model)

    offsets = [0] * (len(pending) + 1)
    for owner in flat_owner:
        offsets[owner + 1] += 1
    for i in range(len(pending)):
        offsets[i + 1] += offsets[i]

    for i, p in enumerate(pending):
        paper_vectors = vectors[offsets[i] : offsets[i + 1]]
        fingerprint_vec = fp.pool(paper_vectors, weights[i])
        paper_cache[str(p["pid"])] = {
            "vector": fingerprint_vec.tolist(),
            "n_topics": len(p.get("topics", [])),
        }

    fp.save_fingerprint_cache(paper_cache, cache_path)


def eligible_scores(
    paper: dict,
    candidate_emails: list[str],
    candidate_matrix: np.ndarray,
    paper_vec: np.ndarray,
    reviewers_by_email: dict,
    *,
    area_gate: bool = True,
) -> list[tuple[str, float]]:
    """(email, cosine-similarity) pairs for reviewers eligible for `paper`.

    Excludes reviewers whose email is a key in the paper's pc_conflicts.
    Unless `area_gate` is False, also excludes reviewers whose primary/
    secondary area doesn't overlap the paper's topics (case-insensitive).
    """
    conflicted = {e.lower() for e in paper.get("pc_conflicts", {})}
    topic_set = {t.lower() for t in paper.get("topics", [])}

    eligible_idx = []
    for i, email in enumerate(candidate_emails):
        if email in conflicted:
            continue
        if area_gate:
            r = reviewers_by_email[email]
            areas = {a.lower() for a in (r.primary, r.secondary) if a}
            if not (areas & topic_set):
                continue
        eligible_idx.append(i)

    if not eligible_idx:
        return []

    sims = candidate_matrix[eligible_idx] @ paper_vec
    return [(candidate_emails[i], float(s)) for i, s in zip(eligible_idx, sims)]
