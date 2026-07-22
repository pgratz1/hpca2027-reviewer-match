"""Build a per-reviewer SPECTER2 fingerprint from declared areas + DBLP titles.

Each reviewer is represented as a small set of SPECTER2 "documents" — one per
recent DBLP title, plus one summarizing their declared areas and free-text
keywords — embedded individually and then weighted-mean-pooled into a single
L2-normalized vector. Pooling documents separately (rather than flattening
everything into one string) keeps each input in the title+SEP+abstract shape
SPECTER2 was trained on, and avoids truncating a prolific author's papers
into one over-length string.

Reviewers with no DBLP PID, or with zero titles in the fetch window, fall
back to an area-profile-only fingerprint (just the one pooled document).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from transformers import PreTrainedTokenizerBase

from reviewers import Reviewer


def doc_text(tokenizer: PreTrainedTokenizerBase, primary: str, secondary: str = "") -> str:
    """A SPECTER2 input document: `primary SEP secondary` (secondary optional).

    This is the model's trained format (title SEP abstract); `secondary`
    defaults to empty for inputs that have no natural second half, e.g. a
    bare DBLP title or a reviewer's area list.
    """
    return f"{primary}{tokenizer.sep_token}{secondary}"


def area_profile_text(tokenizer: PreTrainedTokenizerBase, reviewer: Reviewer) -> str:
    """The reviewer's declared areas + keywords as one SPECTER2 document."""
    areas = ", ".join(a for a in (reviewer.primary, reviewer.secondary, reviewer.tertiary) if a)
    return doc_text(tokenizer, areas, reviewer.keywords)


def title_doc_text(tokenizer: PreTrainedTokenizerBase, title: str) -> str:
    """A DBLP title as one SPECTER2 document (no abstract available)."""
    return doc_text(tokenizer, title)


def publication_doc_text(
    tokenizer: PreTrainedTokenizerBase, title: str, abstract: str = ""
) -> str:
    """A publication in SPECTER2's native title-plus-abstract shape."""
    return doc_text(tokenizer, title, abstract)


def select_titles(
    titles: list[tuple[int, str]], max_titles: int | None
) -> list[tuple[int, str]]:
    """Cap to the `max_titles` most recent titles (input assumed year-descending)."""
    if max_titles is None:
        return list(titles)
    return titles[:max_titles]


def pool(vectors: np.ndarray, weights: list[float]) -> np.ndarray:
    """Weighted mean of `vectors`, L2-normalized."""
    w = np.asarray(weights, dtype=np.float64).reshape(-1, 1)
    mean = (vectors * w).sum(axis=0) / w.sum()
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 0 else mean


# ---------------------------------------------------------------------------
# Cache I/O (mirrors dblp.py's load_cache/save_cache pattern)
# ---------------------------------------------------------------------------

def load_fingerprint_cache(path: str) -> dict:
    """Load the fingerprint cache from disk (returns {} if not found).

    Format: {email: {"vector": [...], "n_titles": int, "has_pid": bool}, ...}
    """
    p = Path(path)
    if p.exists():
        with p.open(encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_fingerprint_cache(cache: dict, path: str) -> None:
    """Write the fingerprint cache to disk atomically."""
    p = Path(path)
    tmp = p.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    tmp.replace(p)
