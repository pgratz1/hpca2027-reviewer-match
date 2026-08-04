"""Shared paper selection, fingerprinting, and eligibility logic.

Used by both score_papers.py (independent per-paper top-N ranking) and
assign_reviewers.py (global load-capped assignment across all papers), so
the paper fingerprinting, COI exclusion, and area gate aren't duplicated.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys

import numpy as np

from . import fingerprint as fp
from . import specter2_model

PAPER_FINGERPRINT_SCHEMA_VERSION = 2
PAPER_POLICIES = ("registered", "submitted", "complete")

# A registered paper needs an abstract of more than one sentence and a title
# that isn't one of these stand-ins for an unwritten one (compared lowercased).
MIN_ABSTRACT_SENTENCES = 2
PLACEHOLDER_TITLES = {"test"}

_SENTENCE_RE = re.compile(r"[.!?]+")


def _is_withdrawn(paper: dict) -> bool:
    return bool(paper.get("withdrawn")) or str(paper.get("status") or "") == "withdrawn"


def _sentence_count(text: str) -> int:
    """Sentences in `text`: `.!?`-separated chunks that contain a letter.

    Deliberately crude — it only has to tell a real abstract from a
    placeholder like "TBD", "1. 2. 3." or "Abstract goes here."
    """
    return sum(1 for chunk in _SENTENCE_RE.split(text) if any(c.isalpha() for c in chunk))


def registration_gaps(paper: dict) -> list[str]:
    """Reasons a registered paper isn't real enough to assign reviewers to.

    HotCRP status stays "draft" until the submission deadline, so before it a
    paper counts as registered on its content: at least one author, a title of
    at least one non-placeholder word, and an abstract of more than one
    sentence. Withdrawn papers need no reviewers.
    """
    gaps = []
    title = (paper.get("title") or "").strip()
    if not title.split():
        gaps.append("no title")
    elif title.lower() in PLACEHOLDER_TITLES:
        gaps.append("placeholder title")
    if _sentence_count(paper.get("abstract") or "") < MIN_ABSTRACT_SENTENCES:
        gaps.append(f"abstract under {MIN_ABSTRACT_SENTENCES} sentences")
    if not paper.get("authors"):
        gaps.append("no authors")
    if _is_withdrawn(paper):
        gaps.append("withdrawn")
    return gaps


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
    if _is_withdrawn(paper):
        gaps.append("withdrawn")
    return gaps


def selection_gaps(paper: dict, paper_policy: str) -> list[str]:
    """Reasons `paper` is excluded under the requested selection policy."""
    if paper_policy == "registered":
        return registration_gaps(paper)
    if paper_policy == "submitted":
        status = str(paper.get("status") or "missing")
        return [] if status == "submitted" else [f"status {status}"]
    if paper_policy == "complete":
        return completeness_gaps(paper)
    raise ValueError(
        f"unknown paper policy {paper_policy!r}; expected one of {', '.join(PAPER_POLICIES)}"
    )


def load_papers(
    path: str, *, paper_policy: str = "registered", with_skipped: bool = False
):
    """Select assignable papers from a HotCRP export.

    With `with_skipped=True` returns `(papers, skipped)` instead, where
    `skipped` is `[{pid, title, missing}, ...]` for the exclusion report.
    The default ``registered`` policy takes every non-withdrawn paper whose
    content is real (see `registration_gaps`); ``submitted`` trusts HotCRP
    status alone, and ``complete`` preserves the older completeness checks.
    """
    with open(path, encoding="utf-8") as f:
        papers = json.load(f)
    selected, skipped = [], []
    for p in papers:
        gaps = selection_gaps(p, paper_policy)
        if gaps:
            skipped.append({"pid": p["pid"], "title": p.get("title") or "", "missing": gaps})
        else:
            selected.append(p)
    if skipped:
        print(
            f"Skipping {len(skipped)} papers under the {paper_policy} policy; "
            f"{len(selected)} of {len(papers)} remain",
            file=sys.stderr,
        )
    if with_skipped:
        return selected, skipped
    return selected


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


def own_paper_conflicts(paper: dict) -> set[str]:
    """Emails conflicted with `paper` by their own involvement in it.

    A floor under HotCRP's `pc_conflicts`, which is only as complete as the
    author COI sweep: authors have not yet been asked to declare conflicts
    against recently promoted PC and reserve reviewers, so a reviewer can be an
    author of a paper and not appear in its conflict list. Three reserve
    reviewers currently have no conflict recorded anywhere while authoring 18
    submissions between them.

    Three sources, none of which needs anyone to have declared anything:
    authors, contacts, and the `reserve_reviewer` nomination — HPCA's nomination
    field names a senior author of that same paper, which the data bears out
    (227 of 228 nominations are also authorships).

    This closes only the self-authorship hole. It cannot supply the conflicts a
    sweep would — a reviewer's ties to a paper they did not write stay invisible
    while nobody on that paper has been asked.
    """
    conflicted: set[str] = set()
    for person in (paper.get("authors") or []) + (paper.get("contacts") or []):
        email = (person.get("email") or "").strip().lower()
        if email:
            conflicted.add(email)
    for line in (paper.get("reserve_reviewer") or "").splitlines():
        parts = [part.strip() for part in line.split("/")]
        if len(parts) >= 3 and "@" in parts[-1]:
            conflicted.add(parts[-1].lower())
    return conflicted


def eligible_scores(
    paper: dict,
    candidate_emails: list[str],
    candidate_matrix: np.ndarray,
    paper_vec: np.ndarray,
    reviewers_by_email: dict,
    *,
    area_gate: bool = True,
    extra_conflicts=frozenset(),
) -> list[tuple[str, float]]:
    """(email, cosine-similarity) pairs for reviewers eligible for `paper`.

    Excludes reviewers conflicted with the paper — those listed in its
    pc_conflicts, those involved in it themselves (see `own_paper_conflicts`),
    and any email in `extra_conflicts`, which is how the derived co-author
    layer (`coauthor_coi`) gets in. Unless `area_gate` is False, also excludes
    reviewers whose primary/secondary area doesn't overlap the paper's topics
    (case-insensitive).

    This is the only place COI is applied. `assign_reviewers` builds its
    preference lists from what this returns and no later phase re-checks them,
    so a reviewer excluded here is unreachable even after every relaxation.

    `pc_conflicts` is declared against whatever address HotCRP has the
    reviewer's account under -- their `hotcrp_email`, not necessarily the
    `candidate_emails` roster key -- so both are checked. Someone
    `pc_membership` matched under a second address would otherwise have a
    real declared conflict invisible here and only caught by HotCRP itself,
    at upload, as a hard error.
    """
    conflicted = {e.lower() for e in paper.get("pc_conflicts", {})}
    conflicted |= own_paper_conflicts(paper)
    conflicted |= {e.lower() for e in extra_conflicts}
    topic_set = {t.lower() for t in paper.get("topics", [])}

    eligible_idx = []
    for i, email in enumerate(candidate_emails):
        r = reviewers_by_email[email]
        if email in conflicted or r.hotcrp_email.lower() in conflicted:
            continue
        if area_gate:
            areas = {a.lower() for a in (r.primary, r.secondary) if a}
            if not (areas & topic_set):
                continue
        eligible_idx.append(i)

    if not eligible_idx:
        return []

    sims = candidate_matrix[eligible_idx] @ paper_vec
    return [(candidate_emails[i], float(s)) for i, s in zip(eligible_idx, sims)]
