# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

The **HPCA 2027 reviewer–paper matching pipeline** — fully implemented and in
active use by the PC chair during the submission window (paper registration
deadline: July 25, 2026, so `hpca2027-data.json` is a moving snapshot).
It classifies PC members by seniority from DBLP history and assigns reviewers
to papers by SPECTER2 embedding similarity under COI, area, load, and
seniority constraints.

**Read `README.md` first** — it documents every script, the start-to-finish
workflow, and the data files. `hpca2027-matching-brief.md` is the original
design brief, kept for history; where they disagree, the README and code win.
`/home/pgratz/reviewer_match` is a symlink to this directory (the real path
contains spaces and parentheses — always quote it in shell commands).

## Running things

- Python env: `~/envs/hpca-matching/bin/python3` (torch+CUDA, transformers,
  adapters, numpy). `make` defaults to it; `requirements.txt` is
  documentation, not a reproducible installer.
- `make` runs the whole pipeline (classify → abstract enrichment →
  fingerprints → `assignment.txt`)
  and rebuilds only what's stale. Prefer it over invoking scripts by hand.
- `make area-chairs` is deliberately separate: it requires `assignment.txt`,
  builds 10-year chair fingerprints, and writes a chair-grouped
  `area_chair_assignment.txt` under hard COIs and ±10% loads.
- Library modules (imported, never run): `reviewers.py`, `dblp.py`,
  `paper_matching.py`, `fingerprint.py`, `specter2_model.py`. Runnable
  scripts: `classify_reviewers.py`, `build_fingerprints.py`,
  `enrich_publications.py`, `assign_reviewers.py`, `score_papers.py`,
  `nearest_neighbors.py`, `compare_abstract_rankings.py`,
  `score_abstract_evaluation.py`, `assign_area_chairs.py`, `main.py`.

## Architecture (filter-then-rank, then constrained assignment)

1. **Identity**: `dblp_overrides.csv` (email-keyed, hand-maintained) is the
   single identity layer; a filled `dblp` cell wins over the form's DBLP
   column. `classify_reviewers.py` auto-appends blank stub rows for reviewers
   it can't resolve — the file doubles as the to-do list.
2. **Seniority**: `classify_reviewers.py` → `reviewer_seniority.csv`
   (senior ≥0.8 papers/yr over 15y in ISCA/MICRO/HPCA/ASPLOS; junior <20
   pubs overall; out-of-area ≥20 pubs but <5 target-venue career; typical
   otherwise — all flag-tunable, checked in that order). PC-service
   overrides from `PCDB_with_emails.csv` (email-matched, promote-only)
   then make chairs, TopPicks PC/ERC members, and PC/ERC score ≥6
   senior, and juniors with score ≥2 typical.
3. **Affinity**: SPECTER2 fingerprints for reviewers (recent DBLP
   publications, using IEEE/ACM abstracts where cached, plus declared areas)
   and papers (title+abstract+topics); cosine similarity.
   COI is a hard filter and the area gate (reviewer primary/secondary ∩
   paper topics) governs the normal phases — neither is ever blended into
   the score, but the gate is released per-paper by the relaxation ladder.
4. **Assignment**: `assign_reviewers.py` — phased paper-proposing deferred
   acceptance aiming for a full slate plus ≥1 senior, ≤1 junior, and ≤1
   out-of-area per paper. Papers that can't fill release constraints in
   order: area gate → junior/out-of-area caps (almost-nots only) → senior
   requirement (almost-senior), every pool still ranked by fingerprint
   similarity. Prints a criteria report, per-paper match goodness (mean
   assigned-reviewer similarity, worst-first summary), a relaxation &
   exclusion report, and self-checks (over-cap, blocking pairs,
   junior/out-of-area policy) that must all be 0.

**Policy:** every paper-side tool ignores incomplete or withdrawn papers
(title under 3 words; missing abstract, topics, or authors; withdrawn flag) —
enforced centrally in `paper_matching.load_papers` / `completeness_gaps`.

## Data, caches, and PII

- The acceptance CSV filename contains an apostrophe and spaces — quote it.
  Parse it (and any CSV here) with Python's `csv` module, never
  `awk`/`cut`/line counting: fields contain embedded commas, quoted `""`
  escapes, and embedded newlines. `reviewers.load_reviewers` collapses
  duplicate form submissions to the latest per email and applies overrides.
- **Never commit data.** Everything derived from real PC members (both form
  CSVs, all caches, `dblp_overrides.csv`, `publication_exclusions.csv`,
  classifications, assignments, and retired report CSVs) is gitignored; only
  code and docs go to the GitHub remote. Check `git status` before committing.
- Caches are incremental, versioned, and **content/policy-aware**: paper
  fingerprints include title/abstract/topics, model, and area weight;
  reviewer fingerprints include metadata, PID, selected publications and
  abstracts, model, and embedding flags. Transient DBLP/API failures remain
  retryable. The DBLP caches (`dblp_cache.json`, `dblp_venue_cache.json`,
  `reviewer_publications.json`, read-only `dblp_pubs_cache.json`) are expensive to refill — live DBLP fetches are
  rate-limited (~3s jittered delay, 429 backoff ≥15s). Never delete them;
  `make clean-fingerprints` deliberately spares them.
- `.env` may hold an optional `S2_API_KEY`; it and editor backup
  variants are ignored. Never print, inspect, or commit secret values.
- `publication_exclusions.csv` is the hand-maintained, per-email DOI exclusion
  layer for reviewer and area-chair fingerprints; exclusions never delete
  entries from the shared publication or abstract caches.
- A title-only comparison cache must use a distinct path such as
  `fingerprints-title-only.json`; all `fingerprints-*.json` files are ignored.

## Conventions

- Simple, literal scripts — stdlib `csv`, no pandas/openpyxl, no speculative
  features or extra output formats.
- Each script: module docstring with usage examples, reused as the argparse
  description (`RawDescriptionHelpFormatter`); tunables as module-level
  `DEFAULT_*` constants exposed as flags.
- stdout is for results; progress, warnings, and summaries go to stderr.
- Cache writes are atomic (tmp + replace); reruns must be idempotent —
  verify with `cmp` after changes.
- When experimenting, work on scratch copies (`--out`, `--fingerprint-cache`,
  `--data`, `--paper-cache` flags exist for this); never mutate the real
  caches or `dblp_overrides.csv` in a test.
- Run `~/envs/hpca-matching/bin/python3 -m unittest tests.test_regressions`
  after changes, then run `make` for an end-to-end and self-check validation.
