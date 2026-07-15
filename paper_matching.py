"""Shared paper-fingerprinting and eligibility logic for the paper-scoring scripts.

Used by both score_papers.py (independent per-paper top-N ranking) and
assign_reviewers.py (global load-capped assignment across all papers), so
the paper fingerprinting, COI exclusion, and area gate aren't duplicated.
"""

from __future__ import annotations

import hashlib
import json
import sys

import numpy as np

import fingerprint as fp
import specter2_model

PAPER_FINGERPRINT_SCHEMA_VERSION = 2


def completeness_gaps(paper: dict) -> list[str]:
    """Reasons a paper must be ignored by every tool; empty = assignable.

    Until the registration deadline, entries missing a real title, abstract,
    topics, or authors are tentative placeholders (policy), and withdrawn
    papers need no reviewers.
    """
    gaps = []
    if len((paper.get("title") or "").split()) < 3:
        gaps.append("title under 3 words")
    if not (paper.get("abstract") or "").strip():
        gaps.append("no abstract")
    if not paper.get("topics"):
        gaps.append("no topics")
    if not paper.get("authors"):
        gaps.append("no authors")
    if paper.get("withdrawn"):
        gaps.append("withdrawn")
    return gaps


def load_papers(path: str, *, with_skipped: bool = False):
    """Assignable papers from the HotCRP export, `completeness_gaps` applied.

    With `with_skipped=True` returns `(papers, skipped)` instead, where
    `skipped` is `[{pid, title, missing}, ...]` for the exclusion report.
    """
    with open(path, encoding="utf-8") as f:
        papers = json.load(f)
    complete, skipped = [], []
    for p in papers:
        gaps = completeness_gaps(p)
        if gaps:
            skipped.append({"pid": p["pid"], "title": p.get("title") or "", "missing": gaps})
        else:
            complete.append(p)
    if skipped:
        print(
            f"Skipping {len(skipped)} papers (incomplete or withdrawn); "
            f"{len(complete)} of {len(papers)} remain",
            file=sys.stderr,
        )
    if with_skipped:
        return complete, skipped
    return complete


def _doc_key(paper: dict, area_weight: float = 1.0) -> str:
    """Hash of the paper fields that feed its SPECTER2 documents.

    Papers keep being edited until the registration deadline, so cache
    entries are keyed by content, not just pid — a changed title, abstract,
    or topic list re-encodes that paper on the next run.
    """
    parts = [
        str(PAPER_FINGERPRINT_SCHEMA_VERSION),
        specter2_model.BASE_MODEL,
        specter2_model.PROXIMITY_ADAPTER,
        repr(float(area_weight)),
        paper.get("title") or "",
        paper.get("abstract") or "",
    ] + list(paper.get("topics", []))
    return hashlib.sha1("\x00".join(parts).encode("utf-8")).hexdigest()


def build_paper_fingerprints(
    papers: list[dict],
    paper_cache: dict,
    cache_path: str,
    *,
    area_weight: float = 1.0,
    device: str = "cuda",
) -> None:
    """Fingerprint any of `papers` missing or stale in `paper_cache`, in place.

    Each paper pools two SPECTER2 documents (fingerprint.pool): title+abstract
    (weight 1.0) and its declared topics (weight `area_weight`). A cached
    entry is stale when its stored `doc_key` no longer matches the paper's
    current content (see `_doc_key`). Saves `paper_cache` to `cache_path` if
    anything was (re)computed; a no-op (no model load) otherwise.
    """
    pending = [
        p for p in papers
        if paper_cache.get(str(p["pid"]), {}).get("doc_key") != _doc_key(p, area_weight)
    ]
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
            "doc_key": _doc_key(p, area_weight),
            "schema_version": PAPER_FINGERPRINT_SCHEMA_VERSION,
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
