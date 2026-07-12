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
- `make` runs the whole pipeline (classify → fingerprints → `assignment.txt`)
  and rebuilds only what's stale. Prefer it over invoking scripts by hand.
- Library modules (imported, never run): `reviewers.py`, `dblp.py`,
  `paper_matching.py`, `fingerprint.py`, `specter2_model.py`. Runnable
  scripts: `classify_reviewers.py`, `build_fingerprints.py`,
  `assign_reviewers.py`, `score_papers.py`, `nearest_neighbors.py`, `main.py`.

## Architecture (filter-then-rank, then constrained assignment)

1. **Identity**: `dblp_overrides.csv` (email-keyed, hand-maintained) is the
   single identity layer; a filled `dblp` cell wins over the form's DBLP
   column. `classify_reviewers.py` auto-appends blank stub rows for reviewers
   it can't resolve — the file doubles as the to-do list.
2. **Seniority**: `classify_reviewers.py` → `reviewer_seniority.csv`
   (senior ≥0.8 papers/yr over 15y in ISCA/MICRO/HPCA/ASPLOS; junior <7
   career; typical otherwise — all flag-tunable).
3. **Affinity**: SPECTER2 fingerprints for reviewers (recent DBLP titles +
   declared areas) and papers (title+abstract+topics); cosine similarity.
   Area gate (reviewer primary/secondary ∩ paper topics) and COI are hard
   filters, never blended into the score.
4. **Assignment**: `assign_reviewers.py` — phased paper-proposing deferred
   acceptance aiming for ≥1 senior and ≤1 junior per paper, degrading via
   almost-senior / almost-not-junior fallbacks; prints a criteria report and
   self-checks (over-cap, blocking pairs, junior policy) that must all be 0.

**Policy:** every paper-side tool ignores incomplete papers (no abstract or a
title under 3 words) — enforced centrally in `paper_matching.load_papers`.

## Data, caches, and PII

- The acceptance CSV filename contains an apostrophe and spaces — quote it.
  Parse it (and any CSV here) with Python's `csv` module, never
  `awk`/`cut`/line counting: fields contain embedded commas, quoted `""`
  escapes, and embedded newlines. `reviewers.load_reviewers` collapses
  duplicate form submissions to the latest per email and applies overrides.
- **Never commit data.** Everything derived from real PC members (the CSV,
  all caches, `dblp_overrides.csv`, `reviewer_seniority.csv`,
  `assignment.txt`, retired report CSVs) is gitignored; only code and docs go
  to the GitHub remote. Check `git status` before committing.
- Caches are incremental **and content-aware**: paper fingerprints re-encode
  when title/abstract/topics change (`doc_key`); reviewer fingerprints
  re-encode when a PID appears for a previously PID-less reviewer. The DBLP
  caches (`dblp_cache.json`, `dblp_venue_cache.json`, read-only
  `dblp_pubs_cache.json`) are expensive to refill — live DBLP fetches are
  rate-limited (~3s jittered delay, 429 backoff ≥15s). Never delete them;
  `make clean-fingerprints` deliberately spares them.

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
