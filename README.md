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

or just use `make` (see the workflow below), which defaults to that
interpreter. Scripts that don't touch SPECTER2 (`main.py`,
`classify_reviewers.py`) also run under plain `python3` — they only need
`requests`.

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
`hpca2027-data.json` with a title under 3 words or missing its abstract,
topics, or authors is a placeholder; withdrawn papers need no reviewers.
`paper_matching.load_papers` (via `completeness_gaps`) drops them all, so
every paper-side tool sees only assignable papers (skip count reported to
stderr; `assign_reviewers.py` itemizes them in its relaxation & exclusion
report).

## Start-to-finish workflow

1. **Drop the inputs in place**: the latest acceptance-form CSV export (keep
   the exact filename) and a fresh `hpca2027-data.json` from HotCRP.
2. **`make`** — rebuilds whatever is stale, in order: reviewer seniority
   classification, reviewer fingerprints, then the assignment. The final
   output lands in **`assignment.txt`**: per-paper reviewer slates, the
   per-area shortage report, and the seniority criteria report.
3. **If classify reported reviewers with missing DBLP identities**, it
   appended blank stub rows for them to `dblp_overrides.csv` — fill in their
   `dblp` cells and `make` again. Unknowns caused by transient DBLP fetch
   failures are retried and do not create identity stubs.
4. **Ad-hoc follow-ups**: `score_papers.py --pid N` for one paper's full
   ranking, `nearest_neighbors.py --email X` to eyeball a reviewer's profile.

The equivalent manual commands, in dependency order:

```bash
~/envs/hpca-matching/bin/python3 classify_reviewers.py
~/envs/hpca-matching/bin/python3 build_fingerprints.py
~/envs/hpca-matching/bin/python3 assign_reviewers.py > assignment.txt
```

Make notes: `make PYTHON=python3` overrides the interpreter (the default is
the venv above); `make clean-fingerprints` forces a full re-embed but never
touches the rate-limited DBLP caches, so it costs GPU seconds, not network
time. Fingerprint caches are content- and policy-aware: paper content or
`--area-weight` changes and reviewer metadata, PID, selected-title, model, or
embedding-policy changes rebuild only affected entries. Legacy cache entries
without provenance metadata are rebuilt once. A transient DBLP error remains
marked for retry rather than permanently turning a PID-backed reviewer into
an area-only profile.

## Scripts

### `main.py` — DBLP title fetcher (diagnostic)
Prints each reviewer's recent DBLP titles; validates the fetch/cache path.
```bash
python3 main.py --limit 5 --years 2
```

### `classify_reviewers.py` — seniority classification
Classifies every accepted reviewer from DBLP publication counts in ISCA,
MICRO, HPCA, and ASPLOS (the target venues) and overall:
- **senior** — ≥ `--senior-rate` (0.8) target-venue papers/year over the last
  `--window` (15) years, i.e. 12+ in-window papers at the defaults;
- **junior** — < `--junior-pubs` (20) publications overall (any venue);
- **out-of-area** — ≥ `--junior-pubs` publications overall but
  < `--out-of-area-career` (5) career target-venue papers;
- **typical** — none of the above (checked in that order, senior first).

Then applies PC-service overrides from `PCDB_with_emails.csv` (`--pcdb`;
`--no-pcdb` skips) to reviewers whose email matches a PCDB row. With
score = `#PC` + 0.5 × `#ERC`, and only ever promoting:
- **senior** — past PC chair (`#Chair` > 0), any TopPicks PC/ERC membership,
  or score ≥ `--pcdb-senior-score` (6);
- **typical** — a junior with score ≥ `--pcdb-typical-score` (2).

A fired override is recorded in the `pcdb_override` column; duplicate PCDB
rows for one email (name variants) merge by summing the counts.

Writes `reviewer_seniority.csv`: one row per reviewer with per-venue career
and window counts backing the classification (enough for the assignment step
to spot "almost senior" / "almost not junior" / "almost not out-of-area"
reviewers later). PIDs come
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
fingerprint. Cached entries are automatically refreshed when their reviewer
metadata, PID, selected publications, model, or embedding flags change.
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
enforce **seniority constraints** from `reviewer_seniority.csv` — each paper
should get ≥ `--min-seniors` (1) senior reviewers, ≤ `--max-juniors` (1)
juniors, and ≤ `--max-out-of-area` (1) out-of-area reviewers — and a **full
slate**. When the normal constraints can't fill a paper's slate or senior
slot, they are released per-paper in a fixed order, each relaxed pool still
ranked by fingerprint similarity so match goodness holds up:

1. **area gate** — take the closest-fingerprint reviewers from any area
   (COI and reviewer capacity are never released);
2. **junior / out-of-area caps** — exceeded only by *almost-not-junior*
   juniors (≥ `--almost-junior-pubs` (15) pubs overall) and
   *almost-not-out-of-area* reviewers (≥ `--almost-out-of-area-career` (5)
   career target-venue papers);
3. **senior requirement** — filled by an *almost-senior* (typical with ≥
   `--almost-senior-window` (10) window papers) only when no true senior is
   available even from other areas.

Within their caps, juniors and out-of-area reviewers compete on match score
like everyone else. A **criteria report** prints which papers are OK,
degraded, or breaking the rules; a per-area **shortage report** covers slots
that stay unfilled even after relaxation (papers without topics appear under
`Unspecified/no matching topic`); the **match goodness** section ranks all
papers worst-first by the mean similarity of their assigned reviewers; and a
**relaxation & exclusion report** itemizes every skipped paper (what's
missing, or withdrawn) and every relaxed paper, reviewer by reviewer with
the released constraint and score. `--no-seniority` skips the seniority
constraints and criteria report (single-pass assignment; the area release
for under-filled papers still applies).
```bash
~/envs/hpca-matching/bin/python3 assign_reviewers.py --light-cap 7 --full-cap 15 --reviewers-per-paper 6
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
`dblp_pubs_cache.json` (colleague's read-only rich DBLP cache),
`PCDB_with_emails.csv` (PC-service history with emails — also sensitive).

**Hand-maintained:** `dblp_overrides.csv`.

**Generated** (safe to delete; rebuilt incrementally): `dblp_cache.json`,
`dblp_venue_cache.json`, `fingerprints.json`, `paper_fingerprints.json`,
`reviewer_seniority.csv`, `assignment.txt`.

**Retired** (left over from the removed lookup chain; kept only as
historical reference, nothing reads them): `no_dblp_lookup_report.csv`,
`manual_review_report*.csv`, `final_identity_resolution.csv`,
`openalex_cache.json`.

Unsafe numeric inputs (negative capacities or targets, nonpositive embedding
weights/windows) fail before network or model work. Executable but
contradictory policy combinations print a warning and continue so the
criteria report can show their consequences.

All PII-bearing files above are gitignored; only code and docs are
committed.
