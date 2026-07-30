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
  fingerprints → submitted-only `assignment.txt`)
  and rebuilds only what's stale. Prefer it over invoking scripts by hand.
- `make area-chairs` is deliberately separate: it requires `assignment.txt`,
  builds 10-year chair fingerprints, and writes a chair-grouped
  `area_chair_assignment.txt` under hard COIs and the closest feasible loads.
- `make reserve-need` is also separate and independent of the assignment: it
  sizes the review-slot shortfall, i.e. how many reserve reviewers have to be
  recruited. No GPU, no fingerprints, no network — pure arithmetic. Recruiting
  the reserves themselves happens outside this repo.
- `make reserve-info` joins the HotCRP reserve upload
  (`reserve_reviewer_upload.csv`) to the DBLP links in
  `reserve_reviewers_vetting_final.xlsx` and writes `reserve_reviewer_info.csv`
  — the reserve-side identity layer. Also independent of the assignment. Rows
  that fail a check are held back in `reserve_reviewer_unresolved.csv` rather
  than fingerprinting the wrong person. **Always run it as
  `make reserve-info VERIFY=--verify`**: offline, a PID is only name-checked
  when the PID itself spells a name, and verifying the rest against DBLP caught
  40 links naming the wrong person out of 243 (roster 225 → 186). A plain run
  overwrites that verified roster with the weaker offline result. The profile
  cache is warm, so a verified re-run costs 0 fetches.
- `make dblp-snapshot` is the **preferred source of publications**: it reads a
  local DBLP dump (`DBLP_SNAPSHOT`, default `dblp-2026-07-01.xml`, 5.2 GB,
  gitignored) and writes `dblp_snapshot_cache.json` for every roster PID —
  offline, ~4 min, no rate limit. Run it before `make`, `make area-chairs` or
  `make reserves` and those need no network for DBLP at all. The dump has no
  `pid` attributes, so `build_dblp_snapshot_cache.py` joins PID → name strings
  (from `<www key="homepages/PID">` records, aliases included) → publications by
  exact name match; DBLP guarantees a name string identifies one person. The
  cache is consulted **only for PIDs the existing caches lack**
  (`dblp.snapshot_gaps`) — re-sourcing a cached person would change their
  publication list, which is part of the fingerprint key, and re-embed them for
  nothing. PIDs absent from the snapshot (anyone added after it was taken) are
  reported and still use the live path.
- `make affiliation-countries` enumerates every affiliation string across the
  submissions and all three rosters and resolves which country each institution
  is in, writing `affiliation_countries.csv` with a **blank `country` column as
  the to-do list** (the `dblp_overrides.csv` idiom — the generator writes only
  `suggested`, never `country`, or the hand layer would outrank DBLP with a
  machine guess). Needed only for `--region-cap`; run `make dblp-snapshot` first,
  since its `dblp_affiliations.json` is the strongest layer. `--validate
  CN=china_faculty.csv '!CN=nonchina_faculty.csv'` checks it against the old
  hand split — which turns out to be exactly the `.cn` email test (100%/0%), so
  it misfiles Chinese-institution reviewers who use gmail/qq addresses.
- `make reserves` puts the reserve roster through the same three stages as the
  PC (enrich → fingerprints → classify) via `--role reserve`, writing
  `reserve_fingerprints.json` and `reserve_seniority.csv`. Reserves never filled
  in a form, so `reserve_reviewers.py` derives their areas from the HotCRP
  topics of the submissions they authored — without areas the area gate matches
  them to nothing. `roster.py` maps role → loader for all three scripts.
- `make smoke` rehearses a full assignment over **both** rosters:
  `make_smoke_dataset.py` writes `hpca2027-data-smoke.json` (a seeded 30% of the
  registered papers *marked withdrawn*, standing in for those never submitted),
  then `assign_reviewers.py --include-reserves --reserve-cap 6` runs over it. The
  self-checks — over cap, blocking pairs, junior/out-of-area — are the pass/fail;
  the shortage count is a capacity statement, not a verdict.
- **COI is a floor, not a full picture.** `paper_matching.own_paper_conflicts`
  treats authors, contacts, and `reserve_reviewer` nominees as conflicted, because
  authors have not yet been asked to declare conflicts against recently promoted
  PC and reserve reviewers. Coverage is uneven (reserves: median 1 declared
  conflict per paper vs 7–8 for the PC), which `assign_reviewers.py` now prints,
  and it makes any reserve-heavy result optimistic until the sweep runs.
- `make complete-papers` and `make area-chairs-complete` retain the former
  completeness filter in separate `*-complete.txt` artifacts.
- Library modules (imported, never run): `reviewers.py`, `dblp.py`,
  `paper_matching.py`, `fingerprint.py`, `specter2_model.py`,
  `reserve_reviewers.py`, `roster.py`, `affiliation_country.py`. Runnable
  scripts: `classify_reviewers.py`, `build_fingerprints.py`,
  `enrich_publications.py`, `assign_reviewers.py`, `score_papers.py`,
  `nearest_neighbors.py`, `compare_abstract_rankings.py`,
  `score_abstract_evaluation.py`, `assign_area_chairs.py`,
  `estimate_reserve_need.py`, `build_reserve_reviewer_info.py`, `make_smoke_dataset.py`,
  `resolve_reserve_pids.py`, `build_dblp_snapshot_cache.py`,
  `build_affiliation_countries.py`,
  `resolve_trc_members.py`, `main.py`.

## Architecture (filter-then-rank, then constrained assignment)

1. **Identity**: `dblp_overrides.csv` (email-keyed, hand-maintained) is the
   single identity layer for the PC. Reserve reviewers, who never filled in the
   acceptance form, get theirs from `reserve_reviewer_info.csv`, built from the
   vetting workbook but overridden per-email by `reserve_dblp_overrides.csv`
   (written by `resolve_reserve_pids.py`, finished by hand; a blank `dblp` cell
   is a to-do marker, not a decision, and never masks the workbook).
   A filled `dblp` cell wins over the form's DBLP
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
3b. **Region caps** (optional, `--region-cap CN=2`): a paper whose authors are
   majority-CC holds at most N reviewers affiliated in CC. Affiliation country,
   **never nationality**; HK/MO/TW/SG are separate ISO codes and are never
   folded into CN. Hard in all six phases including F3, so a paper under-fills
   rather than exceed it. Because a region class *crosses* the seniority
   classes, greedy choice stops being substitutable and stability is no longer
   guaranteed for capped papers — `Done.` splits the blocking-pair count, and
   the hard invariant that replaces it is `region_over == 0`. Coverage is
   printed because an unplaced reviewer can never consume a cap.
4. **Assignment**: `assign_reviewers.py` — phased paper-proposing deferred
   acceptance aiming for a full slate plus ≥1 senior, ≤1 junior, and ≤1
   out-of-area per paper. Papers that can't fill release constraints in
   order: area gate → junior/out-of-area caps (almost-nots only) → senior
   requirement (almost-senior), every pool still ranked by fingerprint
   similarity. Prints a criteria report, per-paper match goodness (mean
   assigned-reviewer similarity, worst-first summary), a relaxation &
   exclusion report, and self-checks (over-cap, blocking pairs,
   junior/out-of-area policy) that must all be 0.

**Policy:** every paper-side tool defaults to `--paper-policy registered`:
every non-withdrawn record with ≥1 author, a title of ≥1 word that isn't just
"test", and an abstract of more than one sentence (topics not required).
HotCRP `status` stays `draft` until the submission deadline, so content, not
status, distinguishes a real registration from a placeholder. `--paper-policy
submitted` selects exactly `status == "submitted"` regardless of any other
field — switch to it (`make PAPER_POLICY=submitted`) once submissions are in;
only under that policy must reviewer assignments be full and area-chair
assignments cover the same paper set. `--paper-policy complete` retains the
former pre-registration completeness checks in `*-complete.txt` artifacts.
The registered set (~1400 papers) far exceeds total PC review capacity, so
large shortage/relaxation reports are expected, not a bug.

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
  rate-limited (~3s jittered delay; on a 429 or a dropped connection,
  `dblp.get_with_retry` backs off from ≥15s, doubling with jitter, capped at
  `MAX_BACKOFF`, honouring `Retry-After` up to the same cap). The backoff index
  is per-call, so it never ratchets up across a run. **DBLP resets the
  connection outright once it has blocked an IP** — retrying cannot clear that,
  only waiting can — so the batch loops stop after `MAX_CONSECUTIVE_FAILURES`
  and keep what they cached instead of grinding for hours and leaving reviewers
  looking publication-less. If a run dies that way, wait and re-run; the caches
  resume. Prefer a larger `--delay` for big first-time fetches. Never delete them;
  `make clean-fingerprints` deliberately spares them. `dblp_profile_cache.json`
  and `dblp_author_search_cache.json`, filled by `resolve_trc_members.py`, are
  the same kind of cache.
- `.env` may hold an optional `S2_API_KEY`; it and editor backup
  variants are ignored. Never print, inspect, or commit secret values.
- `publication_exclusions.csv` is the hand-maintained, per-email DOI exclusion
  layer for reviewer and area-chair fingerprints; exclusions never delete
  entries from the shared publication or abstract caches.
- `affiliation_countries.csv` is the hand-maintained affiliation → ISO country
  layer for the region cap, and `dblp_affiliations.json` the DBLP notes under
  it; both are gitignored, both derive from real identities.
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
