# HPCA 2027 Reviewer–Paper Matching

Tooling for the HPCA 2027 program committee: reviewer identity resolution
against DBLP, publication-based seniority classification, and SPECTER2
embedding-based reviewer-to-paper affinity scoring and assignment. See
`hpca2027-matching-brief.md` for the original design brief.

## Setup

The working Python environment is a venv at `~/envs/hpca-matching` with
`torch` + CUDA, `transformers`, `adapters`, and `numpy` (`requirements.txt`
lists packages but isn't a reproducible install — the CUDA torch build came
from elsewhere). Run everything as:

```bash
~/envs/hpca-matching/bin/python3 <script>.py [args]
```

Scripts that don't touch SPECTER2 (`main.py`, `classify_reviewers.py`) also
run under plain `python3` — they only need `requests`.

## Pipeline

Both workflows share the reviewer loader (`reviewers.py`) and DBLP caches:

```
                       ┌─▶ classify_reviewers.py ──▶ reviewer_seniority.csv ──▶ (assign_reviewers.py)
acceptance CSV ──▶ reviewers.py (+ dblp_overrides.csv)
                       └─▶ build_fingerprints.py ──▶ fingerprints.json ─┐
                                                                        ├─▶ score_papers.py
paper JSON ──▶ paper_matching.py ──▶ paper_fingerprints.json ───────────┘    assign_reviewers.py
```

**Paper-completeness policy:** until the registration deadline, any paper in
`hpca2027-data.json` with no abstract or a title under 3 words is a
placeholder. `paper_matching.load_papers` drops them, so every paper-side
tool sees only complete papers (skip count reported to stderr).

## Scripts

### `main.py` — DBLP title fetcher (diagnostic)
Prints each reviewer's recent DBLP titles; validates the fetch/cache path.
```bash
python3 main.py --limit 5 --years 2
```

### `classify_reviewers.py` — seniority classification
Classifies every accepted reviewer from DBLP publication counts in ISCA,
MICRO, HPCA, and ASPLOS:
- **senior** — ≥ `--senior-rate` (0.8) papers/year over the last `--window`
  (15) years, i.e. 12+ in-window papers at the defaults;
- **junior** — career total < `--junior-total` (7);
- **typical** — neither (senior checked first).

Writes `reviewer_seniority.csv`: one row per reviewer with per-venue career
and window counts backing the classification (enough for the assignment step
to spot "almost senior" / "almost not junior" reviewers later). PIDs come
from `dblp_overrides.csv` (wins) or the acceptance CSV; anyone left is class
**unknown** with a reason, and gets a stub row appended to
`dblp_overrides.csv` (see below). Uncached PIDs are fetched live once into
`dblp_venue_cache.json`.
```bash
python3 classify_reviewers.py
python3 classify_reviewers.py --window 10 --senior-rate 1.0
```

### `build_fingerprints.py` — reviewer SPECTER2 fingerprints
Embeds each reviewer (recent DBLP titles + declared areas/keywords) into a
768-dim vector, cached in `fingerprints.json` by email. Incremental: cached
reviewers aren't recomputed. Reviewers with no PID get an area-only
fingerprint.
```bash
~/envs/hpca-matching/bin/python3 build_fingerprints.py --limit 10   # validate first
~/envs/hpca-matching/bin/python3 build_fingerprints.py
```
Key flags: `--years` (4), `--max-titles` (uncapped), `--area-weight` (1.0).

### `nearest_neighbors.py` — reviewer/reviewer similarity (diagnostic)
Prints each reviewer's most similar other reviewers by fingerprint cosine —
a sanity check that fingerprints cluster by topic.
```bash
~/envs/hpca-matching/bin/python3 nearest_neighbors.py --email someone@example.com
```

### `score_papers.py` — rank reviewers per paper (unconstrained)
Fingerprints each complete paper and prints its top-N reviewers by cosine
similarity, after excluding COI (`pc_conflicts`) and applying the area gate
(reviewer primary/secondary ∩ paper topics; `--no-area-gate` disables).
Per-paper and independent — no load awareness.
```bash
~/envs/hpca-matching/bin/python3 score_papers.py --pid 8 --top 10
```

### `assign_reviewers.py` — global load-capped assignment
One assignment across all papers at once, respecting COI, the area gate,
per-reviewer caps (`--light-cap` / `--full-cap`, or the CSV's per-reviewer
override column) and `--reviewers-per-paper`. Solved by paper-proposing
deferred acceptance (Hospital/Residents stable matching), run in phases that
enforce **seniority constraints** from `reviewer_seniority.csv`: each paper
should get ≥ `--min-seniors` (1) senior reviewers and ≤ `--max-juniors` (1)
juniors. Degradation when the pool can't satisfy that: a paper with no
eligible senior takes an *almost-senior* (typical with ≥
`--almost-senior-window` (10) window papers); a paper still under-filled at
the junior cap may take extra *almost-not-junior* juniors (≥
`--almost-junior-career` (5) career papers). Within the cap, juniors compete
on match score like everyone else. A **criteria report** prints which papers
are OK, degraded, or breaking the rules; under-filled papers are also flagged
and a per-area shortage report prints. `--no-seniority` skips all of it
(plain single-pass assignment).
```bash
~/envs/hpca-matching/bin/python3 assign_reviewers.py --light-cap 1 --full-cap 2 --reviewers-per-paper 6
```

## The DBLP override file

`dblp_overrides.csv` (columns `email,dblp,note`) is the **single
hand-maintained identity layer**, keyed by email so it survives
acceptance-CSV re-exports. A filled-in `dblp` cell (any link shape or bare
PID) **wins over the form's own DBLP column** — use it to fill in missing
links or correct wrong ones (e.g. a namesake's page). Rows with a blank
`dblp` cell are ignored, and `classify_reviewers.py` appends one such stub
per still-unknown reviewer (name/affiliation/reason in the note), so the
file doubles as the to-do list: fill in the blank cells and rerun.

It absorbed the output of a retired semi-automated lookup chain
(`lookup_no_dblp_reviewers.py` / `apply_human_guesses.py`, removed July
2026) that bulk-resolved the original ~57 no-DBLP reviewers; rows noting
"migrated from final_identity_resolution.csv" came from there.

## Support modules (not standalone scripts)

`reviewers.py` (acceptance-CSV parsing, duplicate-submission collapsing,
override application) · `dblp.py` (DBLP fetch, caching, rate limiting) ·
`paper_matching.py` (paper loading + completeness filter, eligibility) ·
`fingerprint.py` / `specter2_model.py` (embedding plumbing).

## Data files

**Inputs:** the acceptance-form CSV (Google Forms export — real names and
emails, treat as sensitive), `hpca2027-data.json` (HotCRP paper export),
`dblp_pubs_cache.json` (colleague's read-only rich DBLP cache).

**Hand-maintained:** `dblp_overrides.csv`.

**Generated** (safe to delete; rebuilt incrementally): `dblp_cache.json`,
`dblp_venue_cache.json`, `fingerprints.json`, `paper_fingerprints.json`,
`reviewer_seniority.csv`.

**Retired** (left over from the removed lookup chain; kept only as
historical reference, nothing reads them): `no_dblp_lookup_report.csv`,
`manual_review_report*.csv`, `final_identity_resolution.csv`,
`openalex_cache.json`.

All PII-bearing files above are gitignored; only code and docs are
committed.
