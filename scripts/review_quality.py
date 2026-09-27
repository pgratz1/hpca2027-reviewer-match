"""Compare reviews submitted before the on-time cutoff with those submitted after it.

    python -m scripts.review_quality
    python -m scripts.review_quality --genericness --sensitivity
    python -m scripts.review_quality --binoculars      # experimental; see below

Decision support for the timeliness policy (`scripts/timeliness_tags.py`): are
late reviews worse? Three axes, each measured per review and reported **only
as group statistics** -- nothing here is a verdict on an individual review.

**AI signals.** The form's own `LLM Usage In This Review` answer, where 2 is
"I did not use any AI/LLM tools" and 1 is "I confirm the above" (any use stayed
within policy -- which non-users tick too, so it bounds use from above and does
not measure it). Then the rate of words LLMs overuse (`MARKER_WORDS`, after
Liang et al. 2024), em-dashes and bold `**Heading:**` structure. None of these
detects AI text: two reviews known to be written entirely by frontier models
had no marker words and unremarkable style.

**Text-based detection is experimental and off by default.**
`--binoculars` adds the Binoculars score (Hans et al. 2024: perplexity over
cross-perplexity) from AllenAI's OLMo-2 1B base/instruct pair. **Lower
Binoculars means more machine-like.** `--reference-size N` has the instruct
model polish N early reviews to show where LLM-polished text lands; the
"below reference" rate uses the score that flags 90% of that reference. That
reference is a 1B model's output, so it shows direction and scale, not a
calibrated detector. It was dropped from the report because it failed the one
test with a known answer: OLMo-2 1B and, in a pilot, Falcon-7B (4-bit) both
scored the two frontier-model reviews at or above the median real review, i.e.
as human. Detectors also over-flag non-native English writing (Liang et al.
2023).

**Detail.** Words now and at first submission (from the log), words per
field, rebuttal questions, weakness items, references to figures/tables/
sections and numbers per 100 words, drafting effort from the log (draft saves,
hours from first draft to submission, largest single-save share of the text,
which is high when a review is pasted in whole), and, under `--genericness`,
how much closer the review's SPECTER2 embedding is to its own paper than to
the average paper.

**Bias.** Merit, Soundness, Novelty, and the headline number: merit minus the
mean of the paper's other reviews (the leave-one-out residual), which takes
paper quality out. Its absolute value measures disagreement. The share of
Strengths in Strengths + Weaknesses words is a cheap text sentiment.

**Groups**, by each review's *first* submission (`hotcrp_log.first_submitted_at`):
early (by `--announced`, the deadline the committee was told), grace (up to
`--cutoff`, the one the tags use), late (after it; split by day too), and
exempt: an R1 review assigned after `--exempt-after`, or any review by a
reviewer the chair gave an `extension` in `--extensions` -- they had less time
by design. TRC reviews and papers no longer under review are left out.
On time means early + grace.

**Statistics.** Reviews by one person are not independent, so every interval
is a cluster bootstrap over reviewers. Three contrasts per metric: late minus
on time across everyone; the same within the reviewers who have both (sign-flip
permutation p) -- this holds style, language and seniority fixed; and each
reviewer's last-submitted review minus their others, whatever the deadline.
`--sensitivity` repeats the first two without reviews edited after first
submission, since the export holds the current text, not what was submitted.
Many metrics are tested, so read a lone p near 0.05 as noise.

Writes `--pdf` (group tables only, printed by headless `--chrome`) and `--csv` (one row per review, for the
chair's own use -- per-review AI scores can be misread as an accusation, so it
never leaves the machine; both are gitignored by name). Review text is
confidential: every model runs locally and nothing is sent anywhere. Binoculars
scores are cached in `--ai-cache`, keyed on a hash of the text and the models.
"""

from __future__ import annotations

from reviewer_match.paths import cache_path, input_path, report_path

import argparse
import csv
import hashlib
import html
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np

from reviewer_match import hotcrp_log
from reviewer_match import paper_matching
from reviewer_match import review_scores
from scripts import timeliness_tags
from scripts.assign_paper_leads import write_csv

DEFAULT_REVIEWS = input_path("hpca2027-reviews.csv")
DEFAULT_PDF = report_path("review_quality.pdf")
# Headless Chrome prints the report; nothing else here renders a PDF.
DEFAULT_CHROME = "google-chrome"
DEFAULT_CSV = report_path("review_quality_reviews.csv")
DEFAULT_AI_CACHE = cache_path("review_ai_scores.json")
DEFAULT_PAPER_CACHE = cache_path("paper_fingerprints.json")
# "Monday, September 21, 2026 8:00am" Eastern, as the committee was told.
DEFAULT_ANNOUNCED = "2026-09-21 08:00:00 -0400"
DEFAULT_BOOTSTRAP = 2000
DEFAULT_PERMUTATIONS = 10000
DEFAULT_SEED = 1
DEFAULT_OBSERVER = "allenai/OLMo-2-0425-1B"
DEFAULT_PERFORMER = "allenai/OLMo-2-0425-1B-Instruct"
DEFAULT_MAX_TOKENS = 512
DEFAULT_MIN_TOKENS = 64
DEFAULT_BATCH = 4
DEFAULT_REFERENCE_SIZE = 100
# The reference score that flags this share of LLM-polished text.
REFERENCE_QUANTILE = 0.9
GENERIC_BATCH = 16

EARLY, GRACE, LATE, EXEMPT = "early", "grace", "late", "exempt"
GROUPS = (EARLY, GRACE, LATE, EXEMPT)
ON_TIME = (EARLY, GRACE)

LLM_FIELD = "LLM Usage In This Review"
LLM_CONFIRMED = "1"  # "I confirm the above"
LLM_NONE = "2"       # "I did not use any AI/LLM tools"

AUTHOR_FIELDS = (
    "Paper summary", "Strengths", "Weaknesses", "Comments for authors",
    "Questions for revision/rebuttal",
)
PC_FIELD = "Comments for PC"
SCORE_FIELDS = {
    "merit": review_scores.SCORE_FIELD, "soundness": "Soundness",
    "novelty": "Novelty and Expected Impact", "expertise": "Reviewer expertise",
}

# Words LLM-written or -polished review text overuses relative to human
# reviews (Liang et al. 2024, "Monitoring AI-Modified Content at Scale"), minus
# ones this field uses anyway ("novel", "leverage", "robust", "comprehensive",
# "overall", "furthermore", "compelling").
MARKER_WORDS = (
    "commendable", "meticulous", "meticulously", "intricate", "intricacies",
    "delve", "delves", "delving", "showcase", "showcases", "showcasing",
    "underscore", "underscores", "underscoring", "noteworthy", "invaluable",
    "pivotal", "laudable", "lucid", "lucidly", "ingenious", "cogent",
    "methodical", "methodically", "admirable", "admirably", "aptly",
    "compellingly", "seamless", "seamlessly", "nuanced",
    "multifaceted", "holistic", "realm", "tapestry", "elucidate", "elucidates",
    "bolster", "bolsters", "unwavering", "adeptly", "insightful", "thoughtfully",
    "undeniably", "notably",
)
MARKER_RE = re.compile(r"\b(?:" + "|".join(MARKER_WORDS) + r")\b", re.IGNORECASE)
EM_DASH_RE = re.compile("—")
BOLD_HEADING_RE = re.compile(r"\*\*[^*\n]{1,80}\*\*")
REFERENCE_RE = re.compile(
    r"(?:\b(?:fig(?:ure)?s?|tables?|tab|sec(?:tion)?s?|eq(?:uation)?s?|alg(?:orithm)?s?|"
    r"lines?|pages?|appendix|listing)\.?|§)\s*\(?[0-9IVX]",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?:%|x\b|×)?")
WORDS_RE = re.compile(r"(\d+) words$")
# Every log row that carries the review's running word count.
SAVE_RE = re.compile(r"^Review (\d+) (?:submitted|edited)[^:]*:")


# ---------------------------------------------------------------------------
# Text measures
# ---------------------------------------------------------------------------

def word_count(text: str) -> int:
    return len(text.split())


def per(count: int, words: int, scale: int) -> float | None:
    return count * scale / words if words else None


def text_measures(fields: dict[str, str]) -> dict[str, float | int | None]:
    """The text-only measures of one review, from its field values."""
    author = "\n\n".join(fields.get(f) or "" for f in AUTHOR_FIELDS)
    words = word_count(author)
    strengths = word_count(fields.get("Strengths") or "")
    weaknesses_text = fields.get("Weaknesses") or ""
    weaknesses = word_count(weaknesses_text)
    questions = fields.get("Questions for revision/rebuttal") or ""
    return {
        "words": words,
        "words_pc": word_count(fields.get(PC_FIELD) or ""),
        "words_summary": word_count(fields.get("Paper summary") or ""),
        "words_strengths": strengths,
        "words_weaknesses": weaknesses,
        "words_comments": word_count(fields.get("Comments for authors") or ""),
        "words_questions": word_count(questions),
        "has_questions": int(bool(questions.strip())),
        "n_questions": questions.count("?"),
        "weakness_items": sum(1 for line in weaknesses_text.splitlines() if line.strip()),
        "refs_per_100": per(len(REFERENCE_RE.findall(author)), words, 100),
        "numbers_per_100": per(len(NUMBER_RE.findall(author)), words, 100),
        "markers_per_1000": per(len(MARKER_RE.findall(author)), words, 1000),
        "em_dashes_per_1000": per(len(EM_DASH_RE.findall(author)), words, 1000),
        "bold_headings": len(BOLD_HEADING_RE.findall(author)),
        "strength_share": strengths / (strengths + weaknesses) if strengths + weaknesses else None,
    }


def author_text(fields: dict[str, str]) -> str:
    """What the authors will read, as the detector sees it."""
    return "\n\n".join((fields.get(f) or "").strip() for f in AUTHOR_FIELDS if (fields.get(f) or "").strip())


# ---------------------------------------------------------------------------
# Log measures
# ---------------------------------------------------------------------------

@dataclass
class ReviewHistory:
    """What the log says about how one review was written."""

    first_submitted: datetime | None = None
    words_at_submit: int | None = None
    draft_saves: int = 0
    first_draft: datetime | None = None
    max_jump: int = 0
    # Saves after the first submission. HotCRP logs an edit to a submitted
    # review as "edited, updated: ...", never as a second "submitted".
    edits_after: int = 0


def review_histories(rows: list[dict[str, str]]) -> dict[int, ReviewHistory]:
    """{review id: ReviewHistory} from the chronological log.

    Every save carries the review's running word count; the saves before the
    first submission are its drafting. `max_jump` is the largest increase any
    single save made, counting from an empty review.
    """
    out: dict[int, ReviewHistory] = defaultdict(ReviewHistory)
    last_words: dict[int, int] = {}
    for row in rows:
        action = row["action"]
        m = SAVE_RE.match(action)
        if not m:
            continue
        rid = int(m.group(1))
        h = out[rid]
        w = WORDS_RE.search(action)
        words = int(w.group(1)) if w else None
        if h.first_submitted is not None:
            h.edits_after += 1
            continue
        submitted = bool(hotcrp_log.SUBMIT_RE.match(action))
        when = hotcrp_log.parse_date(row["date"])
        if words is not None:
            h.max_jump = max(h.max_jump, words - last_words.get(rid, 0))
            last_words[rid] = words
        if submitted:
            h.first_submitted = when
            h.words_at_submit = words
        else:
            h.draft_saves += 1
            if h.first_draft is None:
                h.first_draft = when
    return dict(out)


def history_measures(h: ReviewHistory) -> dict[str, float | int | None]:
    span = None
    if h.first_submitted is not None:
        start = h.first_draft or h.first_submitted
        span = (h.first_submitted - start).total_seconds() / 3600
    return {
        "words_at_submit": h.words_at_submit,
        "draft_saves": h.draft_saves,
        "draft_hours": span,
        "largest_save_share": (min(1.0, h.max_jump / h.words_at_submit)
                               if h.words_at_submit else None),
        "edited_after_submit": int(h.edits_after > 0),
    }


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------

def review_group(
    submitted_at: datetime,
    assigned_at: datetime,
    extension: bool,
    *,
    announced: datetime,
    cutoff: datetime,
    exempt_after: date,
) -> tuple[str, int]:
    """(group, days late) for one review; days late is 0 outside the late group."""
    if extension or assigned_at.date() > exempt_after:
        return EXEMPT, 0
    if submitted_at <= announced:
        return EARLY, 0
    if submitted_at <= cutoff:
        return GRACE, 0
    return LATE, timeliness_tags.days_late(submitted_at, cutoff)


def loo_residuals(merits: dict[int, list[tuple[str, int]]]) -> dict[str, float]:
    """{review key: merit minus the mean of its paper's other reviews}.

    `merits` is {pid: [(review key, merit)]}. A paper's only review has no
    residual.
    """
    out: dict[str, float] = {}
    for reviews in merits.values():
        if len(reviews) < 2:
            continue
        total = sum(m for _, m in reviews)
        for key, m in reviews:
            out[key] = m - (total - m) / (len(reviews) - 1)
    return out


# ---------------------------------------------------------------------------
# Binoculars
# ---------------------------------------------------------------------------

def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def load_ai_cache(path: str) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_ai_cache(cache: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, sort_keys=True, indent=1)
    os.replace(tmp, path)


class Binoculars:
    """Hans et al. 2024: log-perplexity under the performer over the observer/performer cross-entropy."""

    def __init__(self, observer: str, performer: str, max_tokens: int, device: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(observer)
        if self.tokenizer.get_vocab() != AutoTokenizer.from_pretrained(performer).get_vocab():
            raise ValueError(f"{observer} and {performer} do not share a tokenizer")
        self.chat_tokenizer = AutoTokenizer.from_pretrained(performer)
        self.observer = AutoModelForCausalLM.from_pretrained(observer, dtype=torch.bfloat16).to(device).eval()
        self.performer = AutoModelForCausalLM.from_pretrained(performer, dtype=torch.bfloat16).to(device).eval()
        self.max_tokens = max_tokens
        self.device = device

    def token_count(self, text: str) -> int:
        return len(self.tokenizer(text, truncation=True, max_length=self.max_tokens).input_ids)

    def score(self, texts: list[str]) -> list[float]:
        """One score per text. The EOS token stands in for the BOS OLMo does not add."""
        torch = self.torch
        tok = self.tokenizer
        tok.padding_side = "right"
        enc = tok([tok.eos_token + t for t in texts], return_tensors="pt", padding=True,
                  truncation=True, max_length=self.max_tokens + 1).to(self.device)
        with torch.no_grad():
            obs = self.observer(**enc).logits[:, :-1]
            perf = self.performer(**enc).logits[:, :-1]
            scores = []
            # One sequence at a time: float32 over a 100k vocabulary is what fills the GPU.
            for i in range(len(texts)):
                n = int(enc.attention_mask[i, 1:].sum())
                log_perf = torch.log_softmax(perf[i, :n].float(), dim=-1)
                target = enc.input_ids[i, 1:n + 1]
                ppl = -log_perf.gather(-1, target.unsqueeze(-1)).mean()
                x_ppl = -(torch.softmax(obs[i, :n].float(), dim=-1) * log_perf).sum(-1).mean()
                scores.append((ppl / x_ppl).item())
        return scores

    def polish(self, texts: list[str], seed: int, max_new_tokens: int = 900) -> list[str]:
        """The instruct model's polished rewrite of each review: the LLM-modified reference."""
        torch = self.torch
        tok = self.chat_tokenizer
        tok.padding_side = "left"
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content":
                  "Polish the following conference paper review for clarity, grammar and a "
                  "professional tone. Keep every technical point. Reply with the revised "
                  "review only.\n\n" + t}],
                tokenize=False, add_generation_prompt=True)
            for t in texts
        ]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        torch.manual_seed(seed)
        with torch.no_grad():
            out = self.performer.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9,
                pad_token_id=tok.pad_token_id)
        return [tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True).strip() for o in out]


def binoculars_scores(
    texts: dict[str, str], cache: dict, model: Binoculars, *, min_tokens: int, batch: int
) -> dict[str, float | None]:
    """{review key: score or None when too short}, filling `cache` ({text hash: score})."""
    out: dict[str, float | None] = {}
    todo: list[tuple[str, str]] = []
    for key, text in texts.items():
        h = text_key(text)
        if h in cache:
            out[key] = cache[h]
        elif model.token_count(text) < min_tokens:
            cache[h] = out[key] = None
        else:
            todo.append((key, text))
    todo.sort(key=lambda kt: len(kt[1]))  # similar lengths pad less
    for start in range(0, len(todo), batch):
        chunk = todo[start:start + batch]
        for (key, text), score in zip(chunk, model.score([t for _, t in chunk])):
            cache[text_key(text)] = out[key] = round(score, 6)
        if start // batch % 50 == 0:
            print(f"  binoculars {start + len(chunk)}/{len(todo)}", file=sys.stderr)
    return out


def reference_scores(
    sources: list[str], cache: dict, model: Binoculars, *, seed: int, batch: int
) -> list[float]:
    """Binoculars scores of the instruct model's polish of each source ({source hash: score})."""
    todo = [t for t in sources if text_key(t) not in cache]
    for start in range(0, len(todo), batch):
        chunk = todo[start:start + batch]
        # Cap the source so the rewrite fits in the generation budget.
        polished = model.polish([" ".join(t.split()[:600]) for t in chunk], seed + start)
        for source, text in zip(chunk, polished):
            cache[text_key(source)] = round(model.score([text])[0], 6)
        print(f"  reference {start + len(chunk)}/{len(todo)}", file=sys.stderr)
    return [cache[text_key(t)] for t in sources]


# ---------------------------------------------------------------------------
# Genericness
# ---------------------------------------------------------------------------

def paper_margins(
    texts: dict[str, tuple[int, str]], paper_cache: dict, device: str
) -> dict[str, float | None]:
    """{review key: cos(review, own paper) - mean cos(review, every paper)} by SPECTER2."""
    from reviewer_match import specter2_model
    from reviewer_match.fingerprint import doc_text

    pids = sorted(int(p) for p in paper_cache)
    index = {pid: i for i, pid in enumerate(pids)}
    papers = np.array([paper_cache[str(p)]["vector"] for p in pids], dtype=np.float32)
    papers /= np.linalg.norm(papers, axis=1, keepdims=True)
    tokenizer, model = specter2_model.load_model(device)
    keys = [k for k, (pid, _) in texts.items() if pid in index]
    docs = [doc_text(tokenizer, *texts[k][1].split("\n\n", 1)) if "\n\n" in texts[k][1]
            else doc_text(tokenizer, texts[k][1]) for k in keys]
    vecs = specter2_model.encode_texts(docs, tokenizer, model, batch_size=GENERIC_BATCH)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    sims = vecs @ papers.T
    out: dict[str, float | None] = {k: None for k in texts}
    for row, key in enumerate(keys):
        own = sims[row, index[texts[key][0]]]
        out[key] = float(own - sims[row].mean())
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def cluster_mean_ci(
    values: list[tuple[str, float]], rng: np.random.Generator, n_boot: int
) -> tuple[float, float, float] | None:
    """(mean, lo, hi) of the pooled values, resampling the clusters (reviewers)."""
    if not values:
        return None
    by: dict[str, list[float]] = defaultdict(list)
    for c, v in values:
        by[c].append(v)
    sums = np.array([sum(v) for v in by.values()])
    counts = np.array([len(v) for v in by.values()])
    mean = sums.sum() / counts.sum()
    if len(by) < 2:
        return float(mean), float("nan"), float("nan")
    idx = rng.integers(0, len(by), size=(n_boot, len(by)))
    boot = sums[idx].sum(1) / counts[idx].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(mean), float(lo), float(hi)


def cluster_diff_ci(
    a: list[tuple[str, float]], b: list[tuple[str, float]], rng: np.random.Generator, n_boot: int
) -> tuple[float, float, float] | None:
    """(mean(b) - mean(a), lo, hi), resampling reviewers jointly across both groups."""
    if not a or not b:
        return None
    clusters = sorted({c for c, _ in a} | {c for c, _ in b})
    pos = {c: i for i, c in enumerate(clusters)}
    sa, na, sb, nb = (np.zeros(len(clusters)) for _ in range(4))
    for c, v in a:
        sa[pos[c]] += v
        na[pos[c]] += 1
    for c, v in b:
        sb[pos[c]] += v
        nb[pos[c]] += 1
    diff = sb.sum() / nb.sum() - sa.sum() / na.sum()
    idx = rng.integers(0, len(clusters), size=(n_boot, len(clusters)))
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = sb[idx].sum(1) / nb[idx].sum(1) - sa[idx].sum(1) / na[idx].sum(1)
    boot = boot[np.isfinite(boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(diff), float(lo), float(hi)


def paired_test(
    diffs: list[float], rng: np.random.Generator, n_boot: int, n_perm: int
) -> tuple[float, float, float, float, int] | None:
    """(mean, lo, hi, sign-flip p, n) of per-reviewer differences."""
    if len(diffs) < 2:
        return None
    d = np.array(diffs, dtype=float)
    mean = d.mean()
    boot = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    flips = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = (flips * d).mean(1)
    p = (np.sum(np.abs(null) >= abs(mean) - 1e-12) + 1) / (n_perm + 1)
    return float(mean), float(lo), float(hi), float(p), len(d)


# (key, label, axis, decimals). Binary keys read as rates.
METRICS = (
    ("llm_none", "Declared no AI use (share)", "AI", 3),
    ("llm_confirmed", "Declared 'within policy' (share)", "AI", 3),
    ("markers_per_1000", "LLM marker words per 1000", "AI", 2),
    ("em_dashes_per_1000", "Em-dashes per 1000 words", "AI", 2),
    ("bold_headings", "Bold **headings**", "AI", 2),
    ("binoculars", "Binoculars score (lower = machine-like)", "AI", 4),
    ("binoculars_none", "Binoculars, declared no AI", "AI", 4),
    ("below_reference", "Below LLM-polish reference (share)", "AI", 3),
    ("words", "Words, author-facing (now)", "Detail", 0),
    ("words_at_submit", "Words at first submission (log)", "Detail", 0),
    ("words_pc", "Words, comments for PC", "Detail", 0),
    ("words_summary", "Words, summary", "Detail", 0),
    ("words_strengths", "Words, strengths", "Detail", 0),
    ("words_weaknesses", "Words, weaknesses", "Detail", 0),
    ("words_comments", "Words, comments for authors", "Detail", 0),
    ("words_questions", "Words, rebuttal questions", "Detail", 0),
    ("has_questions", "Has rebuttal questions (share)", "Detail", 3),
    ("n_questions", "Question marks in rebuttal questions", "Detail", 2),
    ("weakness_items", "Weakness items (lines)", "Detail", 2),
    ("refs_per_100", "Fig/Table/Section refs per 100 words", "Detail", 3),
    ("numbers_per_100", "Numbers per 100 words", "Detail", 2),
    ("draft_saves", "Draft saves before submitting", "Detail", 2),
    ("draft_hours", "Hours, first draft to submission", "Detail", 1),
    ("largest_save_share", "Largest single save, share of text", "Detail", 3),
    ("paper_margin", "Paper-specificity margin (SPECTER2)", "Detail", 4),
    ("expertise", "Self-rated expertise (1-4)", "Detail", 3),
    ("merit", "Overall merit (1-5)", "Bias", 3),
    ("soundness", "Soundness (1-4)", "Bias", 3),
    ("novelty", "Novelty (1-4)", "Bias", 3),
    ("residual", "Merit minus paper's other reviews", "Bias", 3),
    ("abs_residual", "|Merit residual| (disagreement)", "Bias", 3),
    ("strength_share", "Strengths share of S+W words", "Bias", 3),
)
AXES = (("AI", "AI signals (self-report and style)"), ("Detail", "Detail"), ("Bias", "Bias"))


@dataclass
class Review:
    key: str           # HotCRP's "12A"
    pid: int
    email: str
    group: str
    days_late: int
    submitted_at: datetime
    values: dict[str, float | int | None] = field(default_factory=dict)


def values_of(reviews: list[Review], metric: str) -> list[tuple[str, float]]:
    return [(r.email, float(r.values[metric])) for r in reviews if r.values.get(metric) is not None]


def within_diffs(reviews: list[Review], metric: str) -> list[float]:
    """Per reviewer holding both: mean of their late reviews minus mean of their on-time ones."""
    by: dict[str, dict[bool, list[float]]] = defaultdict(lambda: {True: [], False: []})
    for r in reviews:
        v = r.values.get(metric)
        if v is not None and r.group != EXEMPT:
            by[r.email][r.group == LATE].append(float(v))
    return [float(np.mean(s[True]) - np.mean(s[False])) for s in by.values() if s[True] and s[False]]


def last_review_diffs(reviews: list[Review], metric: str) -> list[float]:
    """Per reviewer with 2+ reviews: their last-submitted review minus the mean of the rest."""
    by: dict[str, list[tuple[datetime, str, float]]] = defaultdict(list)
    for r in reviews:
        v = r.values.get(metric)
        if v is not None and r.group != EXEMPT:
            by[r.email].append((r.submitted_at, r.key, float(v)))
    out = []
    for items in by.values():
        if len(items) >= 2:
            items.sort()
            out.append(items[-1][2] - float(np.mean([v for _, _, v in items[:-1]])))
    return out


def analyse(reviews: list[Review], *, seed: int, n_boot: int, n_perm: int) -> list[dict]:
    """One row per metric: group means with CIs and the three contrasts."""
    rows = []
    for i, (metric, label, axis, decimals) in enumerate(METRICS):
        rng = np.random.default_rng([seed, i])
        by_group = {g: values_of([r for r in reviews if r.group == g], metric) for g in GROUPS}
        if not any(by_group.values()):
            continue
        on_time = by_group[EARLY] + by_group[GRACE]
        rows.append({
            "metric": metric, "label": label, "axis": axis, "decimals": decimals,
            "groups": {g: cluster_mean_ci(v, rng, n_boot) for g, v in by_group.items()},
            "n": {g: len(v) for g, v in by_group.items()},
            "on_time": cluster_mean_ci(on_time, rng, n_boot),
            "late_minus_on_time": cluster_diff_ci(on_time, by_group[LATE], rng, n_boot),
            "within": paired_test(within_diffs(reviews, metric), rng, n_boot, n_perm),
            "last": paired_test(last_review_diffs(reviews, metric), rng, n_boot, n_perm),
            "by_day": {
                d: cluster_mean_ci(values_of([r for r in reviews if r.group == LATE
                                              and min(r.days_late, 3) == d], metric), rng, n_boot)
                for d in (1, 2, 3)
            },
        })
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def fmt(x: float | None, decimals: int) -> str:
    if x is None or x != x:
        return "-"
    return f"{x:.{decimals}f}"


def fmt_ci(t, decimals: int) -> str:
    if not t:
        return "-"
    return f"{fmt(t[0], decimals)} [{fmt(t[1], decimals)}, {fmt(t[2], decimals)}]"


def fmt_paired(t, decimals: int) -> str:
    if not t:
        return "-"
    return f"{fmt(t[0], decimals)} [{fmt(t[1], decimals)}, {fmt(t[2], decimals)}] p={t[3]:.3f} n={t[4]}"


def excludes_zero(t) -> bool:
    return bool(t) and t[1] == t[1] and (t[1] > 0 or t[2] < 0)


def text_report(rows: list[dict], title: str) -> str:
    out = [title]
    for axis, name in AXES:
        axis_rows = [r for r in rows if r["axis"] == axis]
        if not axis_rows:
            continue
        out.append(f"\n{name}")
        out.append(f"  {'metric':40} {'early':>9} {'grace':>9} {'late':>9} {'exempt':>9}"
                   f"  {'late - on time [95% CI]':30} within-reviewer late - on time")
        for r in axis_rows:
            d = r["decimals"]
            means = [fmt(r["groups"][g][0] if r["groups"][g] else None, d) for g in GROUPS]
            mark = " *" if excludes_zero(r["late_minus_on_time"]) else "  "
            out.append(f"  {r['label'][:40]:40} " + " ".join(f"{m:>9}" for m in means)
                       + f"  {fmt_ci(r['late_minus_on_time'], d):30}{mark} {fmt_paired(r['within'], d)}")
    return "\n".join(out)


HTML_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Review Quality by Timeliness</title>
<style>
@page { size: letter landscape; margin: 12mm 10mm; }
:root { --ink: #0b0b0b; --ink-2: #52514e; --muted: #6f6d67; --hairline: #dcdad3; --wash: #f1f0ec; --flag: #b4460f; }
* { box-sizing: border-box; }
body { margin: 0; background: #ffffff; color: var(--ink);
  font: 11px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif;
  -webkit-print-color-adjust: exact; print-color-adjust: exact; }
h1 { font-size: 19px; line-height: 1.25; margin: 0 0 4px; font-weight: 650; }
h2 { font-size: 14px; margin: 16px 0 6px; font-weight: 600; break-after: avoid; }
h3 { font-size: 12px; margin: 10px 0 6px; font-weight: 600; color: var(--ink-2); break-after: avoid; }
.lede, .note { color: var(--ink-2); }
ul.note { padding-left: 18px; margin: 0; }
ul.note li { margin-bottom: 3px; }
.tiles { display: flex; gap: 10px; margin: 12px 0; }
.tile { flex: 1; border: 1px solid var(--hairline); border-radius: 8px; padding: 8px 12px; }
.tile .label { color: var(--ink-2); font-size: 11px; }
.tile .value { font-size: 20px; font-weight: 600; line-height: 1.2; }
.tile .sub { color: var(--muted); font-size: 10px; }
.scroll { border: 1px solid var(--hairline); border-radius: 8px; overflow: hidden; }
table { border-collapse: collapse; width: 100%; font-size: 9.5px; font-variant-numeric: tabular-nums; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
th, td { padding: 3px 6px; text-align: right; border-bottom: 1px solid var(--hairline); white-space: nowrap; }
th:first-child, td:first-child { text-align: left; white-space: normal; min-width: 150px; }
th { color: var(--ink-2); font-weight: 600; background: var(--wash); white-space: normal; }
td .ci { color: var(--muted); font-size: 8.5px; }
td.flag { color: var(--flag); font-weight: 600; }
</style>
</head>
<body>
<main>
"""


def cell(t, decimals: int, flag: bool = False) -> str:
    if not t:
        return "<td>-</td>"
    cls = ' class="flag"' if flag and excludes_zero(t) else ""
    ci = f'<br><span class="ci">[{fmt(t[1], decimals)}, {fmt(t[2], decimals)}]</span>' if t[1] == t[1] else ""
    return f"<td{cls}>{fmt(t[0], decimals)}{ci}</td>"


def paired_cell(t, decimals: int) -> str:
    if not t:
        return "<td>-</td>"
    cls = ' class="flag"' if excludes_zero(t) else ""
    return (f"<td{cls}>{fmt(t[0], decimals)}<br><span class=\"ci\">[{fmt(t[1], decimals)}, "
            f"{fmt(t[2], decimals)}] p={t[3]:.3f}, n={t[4]}</span></td>")


def table(head: list[str], body: list[list[str]]) -> str:
    return ('<div class="scroll"><table><thead><tr>'
            + "".join(f"<th>{html.escape(h)}</th>" for h in head) + "</tr></thead><tbody>"
            + "".join("<tr>" + "".join(tds) + "</tr>" for tds in body)
            + "</tbody></table></div>")


def html_tables(rows: list[dict], axis: str, sensitivity: list[dict] | None) -> str:
    """One axis as two tables that each fit a landscape page: group means, then contrasts."""
    rows = [r for r in rows if r["axis"] == axis]
    sens = {r["metric"]: r for r in sensitivity or []}
    label = lambda r: f"<td>{html.escape(r['label'])}</td>"
    means = [
        [label(r)] + [cell(r["groups"][g], r["decimals"]) for g in (EARLY, GRACE, LATE)]
        + [cell(r["by_day"][k], r["decimals"]) for k in (1, 2, 3)]
        + [cell(r["groups"][EXEMPT], r["decimals"])]
        for r in rows
    ]
    head = ["Metric", "Late − on time", "Within reviewer (late − on time)", "Last review − others"]
    if sensitivity is not None:
        head.insert(2, "Late − on time, unedited")
    contrasts = []
    for r in rows:
        d = r["decimals"]
        tds = [label(r), cell(r["late_minus_on_time"], d, flag=True)]
        if sensitivity is not None:
            s = sens.get(r["metric"])
            tds.append(cell(s["late_minus_on_time"] if s else None, d, flag=True))
        tds += [paired_cell(r["within"], d), paired_cell(r["last"], d)]
        contrasts.append(tds)
    return (table(["Metric", "Early", "Grace", "Late", "Late day 1", "Late day 2", "Late day 3+",
                   "Exempt"], means)
            + '<h3>Contrasts</h3>' + table(head, contrasts))


def render_html(
    rows: list[dict], sensitivity: list[dict] | None, counts: Counter, reviewers: Counter,
    llm: dict[str, Counter], notes: list[str], dates: dict[str, str],
) -> str:
    parts = [HTML_HEAD, "<h1>Review quality by timeliness</h1>"]
    parts.append(
        '<p class="lede">Reviews grouped by when each was <em>first</em> submitted: early (by the '
        f"announced deadline, {html.escape(dates['announced'])}), grace (up to the tag cutoff, "
        f"{html.escape(dates['cutoff'])}), late (after it), and exempt (assigned after "
        f"{html.escape(dates['exempt_after'])}, or under an agreed extension). On time means early + "
        "grace. Group totals only: nothing here judges an individual review.</p>")
    parts.append('<div class="tiles">')
    for g in GROUPS:
        parts.append(f'<div class="tile"><div class="label">{g.capitalize()}</div>'
                     f'<div class="value">{counts[g]:,}</div>'
                     f'<div class="sub">reviews by {reviewers[g]:,} reviewers</div></div>')
    parts.append("</div>")
    parts.append("<h2>Self-declared LLM use</h2>")
    parts.append('<div class="scroll"><table><thead><tr><th>Answer</th>'
                 + "".join(f"<th>{g.capitalize()}</th>" for g in GROUPS) + "</tr></thead><tbody>")
    for value, label in ((LLM_NONE, "2: did not use any AI/LLM tools"),
                         (LLM_CONFIRMED, "1: confirm use (if any) was within policy"),
                         ("", "blank")):
        tds = []
        for g in GROUPS:
            n = sum(llm[g].values())
            k = llm[g][value]
            tds.append(f"<td>{k:,}<br><span class=\"ci\">{(k / n if n else 0):.1%}</span></td>")
        parts.append(f"<tr><td>{label}</td>{''.join(tds)}</tr>")
    parts.append("</tbody></table></div>")
    for axis, name in AXES:
        if any(r["axis"] == axis for r in rows):
            parts.append(f"<h2>{name}</h2>")
            parts.append(html_tables(rows, axis, sensitivity))
    parts.append("<h2>How to read this</h2><ul class=\"note\">")
    parts += [f"<li>{html.escape(n)}</li>" for n in notes]
    parts.append("</ul></main></body></html>\n")
    return "\n".join(parts)


def write_pdf(page: str, path: str, chrome: str) -> None:
    """Print `page` to `path` with headless Chrome, atomically.

    The HTML only ever exists in a private temporary directory, and so does the
    browser profile, so a running Chrome session is never touched.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        source = os.path.join(tmp, "report.html")
        with open(source, "w", encoding="utf-8") as f:
            f.write(page)
        out = os.path.join(tmp, "report.pdf")
        result = subprocess.run(
            [chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
             f"--user-data-dir={os.path.join(tmp, 'profile')}", f"--print-to-pdf={out}",
             "file://" + source],
            capture_output=True, text=True, timeout=120)
        if result.returncode or not os.path.exists(out):
            raise RuntimeError(f"{chrome} could not print the report: {result.stderr.strip()[-400:]}")
        staged = os.path.join(directory, f".{os.path.basename(path)}.tmp")
        shutil.copyfile(out, staged)
        os.replace(staged, path)


NOTES = [
    "Each cell is a mean with a 95% interval from a bootstrap over reviewers, because one "
    "person's reviews are not independent. A highlighted contrast is one whose interval excludes zero.",
    "Within reviewer: for reviewers holding both on-time and late reviews, the mean of "
    "(their late mean - their on-time mean), with a sign-flip permutation p. This holds the "
    "person fixed - style, first language, seniority - so it is the number to trust.",
    "Last review - others: each reviewer's last-submitted review against their earlier ones, "
    "whatever the deadline. Rushing shows up here even among on-time reviewers.",
    "Merit residual: a review's merit minus the mean of the paper's other reviews. It takes "
    "paper quality out of the bias comparison.",
    "No text-based AI detector is reported. Binoculars (OLMo-2 1B, and Falcon-7B in a pilot) "
    "scored two reviews known to be written entirely by frontier models as human - at or above "
    "the median real review - so it cannot tell whether late reviews used AI. The marker-word "
    "and style rows cannot either: both known AI reviews had no marker words and unremarkable "
    "style. Self-report is the only AI evidence here.",
    "Answer 1 on the LLM question ('I confirm the above') is ticked by non-users too; it bounds "
    "use from above. Answer 2 is an explicit claim of no use.",
    "The export holds each review's current text, while groups use the first submission. "
    "The 'unedited' column, when present, drops reviews edited after first submission.",
    "About 30 metrics are tested at once; expect one or two intervals to exclude zero by chance.",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    timeliness_tags.add_common_arguments(parser)
    parser.add_argument("--reviews", default=DEFAULT_REVIEWS, help="HotCRP review CSV export")
    parser.add_argument("--announced", default=DEFAULT_ANNOUNCED,
                        help=f"deadline the committee was told, with offset (default: {DEFAULT_ANNOUNCED!r})")
    parser.add_argument("--pdf", default=DEFAULT_PDF, help="group report to write")
    parser.add_argument("--chrome", default=DEFAULT_CHROME, help="Chrome/Chromium binary that prints the PDF")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="per-review measures to write")
    parser.add_argument("--binoculars", action="store_true", help="score reviews with Binoculars (GPU)")
    parser.add_argument("--observer", default=DEFAULT_OBSERVER, help="Binoculars observer (base) model")
    parser.add_argument("--performer", default=DEFAULT_PERFORMER, help="Binoculars performer (instruct) model")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help=f"tokens of each review scored (default {DEFAULT_MAX_TOKENS})")
    parser.add_argument("--min-tokens", type=int, default=DEFAULT_MIN_TOKENS,
                        help=f"shorter reviews get no score (default {DEFAULT_MIN_TOKENS})")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="Binoculars batch size")
    parser.add_argument("--reference-size", type=int, default=DEFAULT_REFERENCE_SIZE,
                        help="early reviews the instruct model polishes as the LLM reference; 0 skips "
                             f"(default {DEFAULT_REFERENCE_SIZE})")
    parser.add_argument("--ai-cache", default=DEFAULT_AI_CACHE, help="Binoculars score cache")
    parser.add_argument("--genericness", action="store_true",
                        help="SPECTER2 paper-specificity margin (GPU)")
    parser.add_argument("--paper-cache", default=DEFAULT_PAPER_CACHE, help="paper fingerprint cache")
    parser.add_argument("--device", default="cuda", help="torch device for the models")
    parser.add_argument("--sensitivity", action="store_true",
                        help="add the late - on time contrast without reviews edited after submission")
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP, help="bootstrap resamples")
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS, help="sign-flip draws")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="seed for resampling and the reference")
    args = parser.parse_args()

    try:
        announced = hotcrp_log.parse_date(args.announced)
        ev = timeliness_tags.evaluate_from_args(args)
        if not os.path.exists(args.reviews):
            raise FileNotFoundError(f"{args.reviews} not found")
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if announced > ev.cutoff:
        print("ERROR: --announced is after --cutoff", file=sys.stderr)
        return 1

    rows = hotcrp_log.load_log(args.log)
    live, _ = hotcrp_log.replay_assignments(rows)
    histories = review_histories(rows)
    rid_by_pair = {(r.pid, r.email): rid for rid, r in live.items()}
    extension = {email for email, s in ev.states.items() if s.reason == "extension"}

    skipped: Counter = Counter()
    reviews: list[Review] = []
    fields_by_key: dict[str, dict[str, str]] = {}
    with open(args.reviews, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid, email = int(row["paper"]), row["email"].strip().lower()
            rid = rid_by_pair.get((pid, email))
            if pid not in ev.papers:
                skipped["paper no longer under review"] += 1
                continue
            if rid is None:
                skipped["no live assignment in the log"] += 1
                continue
            if live[rid].round != review_scores.REVIEW_ROUND:
                skipped[f"{live[rid].round} round"] += 1
                continue
            h = histories.get(rid)
            if h is None or h.first_submitted is None:
                skipped["no submission in the log"] += 1
                continue
            group, days = review_group(
                h.first_submitted, hotcrp_log.parse_date(live[rid].assigned_at), email in extension,
                announced=announced, cutoff=ev.cutoff, exempt_after=ev.exempt_after)
            review = Review(row["review"], pid, email, group, days, h.first_submitted)
            review.values.update(text_measures(row))
            review.values.update(history_measures(h))
            for name, column in SCORE_FIELDS.items():
                raw = (row.get(column) or "").strip()
                review.values[name] = int(raw) if raw.isdigit() else None
            llm = (row.get(LLM_FIELD) or "").strip()
            review.values["llm_answer"] = llm
            review.values["llm_none"] = int(llm == LLM_NONE) if llm else None
            review.values["llm_confirmed"] = int(llm == LLM_CONFIRMED) if llm else None
            reviews.append(review)
            fields_by_key[review.key] = row
    for reason, n in sorted(skipped.items()):
        print(f"  skipped {n} reviews: {reason}", file=sys.stderr)

    merits: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for r in reviews:
        if r.values["merit"] is not None:
            merits[r.pid].append((r.key, r.values["merit"]))
    residuals = loo_residuals(merits)
    for r in reviews:
        res = residuals.get(r.key)
        r.values["residual"] = res
        r.values["abs_residual"] = abs(res) if res is not None else None

    notes = list(NOTES)
    if args.binoculars:
        notes.append(
            "Binoculars (experimental, --binoculars): lower means more machine-like. It failed "
            "to flag two known frontier-model reviews, so read its rows as style, not AI use.")
        cache = load_ai_cache(args.ai_cache)
        model_key = f"{args.observer}|{args.performer}|{args.max_tokens}"
        scores_cache = cache.setdefault("scores", {}).setdefault(model_key, {})
        print(f"Loading {args.observer} + {args.performer}", file=sys.stderr)
        model = Binoculars(args.observer, args.performer, args.max_tokens, args.device)
        texts = {r.key: author_text(fields_by_key[r.key]) for r in reviews}
        scores = binoculars_scores(texts, scores_cache, model, min_tokens=args.min_tokens, batch=args.batch)
        save_ai_cache(cache, args.ai_cache)
        threshold = None
        if args.reference_size:
            early = sorted((r for r in reviews if r.group == EARLY and scores.get(r.key) is not None),
                           key=lambda r: r.key)
            sample = random.Random(args.seed).sample(early, min(args.reference_size, len(early)))
            ref_cache = cache.setdefault("reference", {}).setdefault(f"{model_key}|seed{args.seed}", {})
            ref = reference_scores([texts[r.key] for r in sample], ref_cache, model,
                                   seed=args.seed, batch=args.batch)
            save_ai_cache(cache, args.ai_cache)
            threshold = float(np.quantile(ref, REFERENCE_QUANTILE))
            human = [scores[r.key] for r in sample]
            notes.append(
                f"LLM-polish reference: {len(ref)} early reviews rewritten by {args.performer}; "
                f"their Binoculars median {np.median(ref):.4f} against {np.median(human):.4f} for "
                f"the same reviews as written. 'Below reference' uses {threshold:.4f}, the score "
                f"that flags {REFERENCE_QUANTILE:.0%} of the rewrites.")
            print(f"Reference: polished median {np.median(ref):.4f}, originals {np.median(human):.4f}, "
                  f"threshold {threshold:.4f}", file=sys.stderr)
        for r in reviews:
            s = scores.get(r.key)
            r.values["binoculars"] = s
            r.values["binoculars_none"] = s if r.values["llm_answer"] == LLM_NONE else None
            r.values["below_reference"] = (int(s < threshold)
                                           if s is not None and threshold is not None else None)
        del model

    if args.genericness:
        if not os.path.exists(args.paper_cache):
            print(f"ERROR: {args.paper_cache} not found; run make first", file=sys.stderr)
            return 1
        with open(args.paper_cache, encoding="utf-8") as f:
            paper_cache = json.load(f)
        print("Encoding reviews with SPECTER2", file=sys.stderr)
        texts = {r.key: (r.pid, author_text(fields_by_key[r.key])) for r in reviews}
        margins = paper_margins(texts, paper_cache, args.device)
        for r in reviews:
            r.values["paper_margin"] = margins[r.key]

    results = analyse(reviews, seed=args.seed, n_boot=args.bootstrap, n_perm=args.permutations)
    sensitivity = None
    if args.sensitivity:
        unedited = [r for r in reviews if not r.values["edited_after_submit"]]
        sensitivity = analyse(unedited, seed=args.seed, n_boot=args.bootstrap, n_perm=args.permutations)

    counts = Counter(r.group for r in reviews)
    reviewers = Counter()
    for g in GROUPS:
        reviewers[g] = len({r.email for r in reviews if r.group == g})
    llm = {g: Counter(r.values["llm_answer"] for r in reviews if r.group == g) for g in GROUPS}
    dates = {"announced": args.announced, "cutoff": args.cutoff, "exempt_after": args.exempt_after}

    columns = ["paper", "review", "email", "group", "days_late", "first_submitted", "llm_answer"] + [
        m for m, *_ in METRICS if m not in ("llm_none", "llm_confirmed")
        # Columns only an optional pass fills (--binoculars, --genericness) appear only when run.
        and any(r.values.get(m) is not None for r in reviews)] + ["edited_after_submit"]
    csv_rows = []
    for r in sorted(reviews, key=lambda r: (r.pid, r.key)):
        base = [r.pid, r.key, r.email, r.group, r.days_late,
                r.submitted_at.strftime(hotcrp_log.DATE_FORMAT), r.values["llm_answer"]]
        vals = [r.values.get(c) for c in columns[len(base):]]
        csv_rows.append(base + ["" if v is None else (round(v, 6) if isinstance(v, float) else v)
                                for v in vals])
    write_csv(args.csv, columns, csv_rows)
    try:
        write_pdf(render_html(results, sensitivity, counts, reviewers, llm, notes, dates), args.pdf, args.chrome)
    except (OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {args.pdf}\nWrote {args.csv}", file=sys.stderr)

    summary = ", ".join(f"{g} {counts[g]} ({reviewers[g]} reviewers)" for g in GROUPS)
    print(text_report(results, f"{len(reviews)} reviews: {summary}. * = interval excludes zero."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
