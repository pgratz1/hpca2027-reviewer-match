# HPCA 2027 Reviewer–Paper Matching

Tooling for the HPCA 2027 program committee: reviewer identity resolution
against DBLP, publication-based seniority classification, and SPECTER2
embedding-based reviewer-to-paper affinity scoring and assignment. See
`docs/hpca2027-matching-brief.md` for the original design brief.

## Setup

The working Python environment is a venv at `~/envs/hpca-matching` with
`torch` + CUDA, `transformers`, `adapters`, and `numpy` (`requirements.txt`
lists packages but isn't a reproducible install — the CUDA torch build came
from elsewhere). From the repository root, set the import path once per shell:

```bash
export PYTHONPATH="$PWD/src:$PWD"
~/envs/hpca-matching/bin/python3 -m scripts.<command> [args]
```

or just use `make` (see the workflow below), which defaults to that
interpreter. Scripts that don't touch SPECTER2 (`scripts/main.py`,
`scripts/classify_reviewers.py`) also run under plain `python3` — they only need
`requests`.

## Repository layout

```text
src/reviewer_match/       reusable matching and data-loading modules
scripts/                  runnable pipeline, audit, and diagnostic commands
tests/                    regression tests
data/inputs/              externally supplied exports, rosters, and snapshots
data/curated/             hand-maintained overrides and policy decisions
data/cache/               expensive reproducible caches and intermediates
outputs/assignments/      final and experimental assignments
outputs/reports/          audits, classifications, and derived rosters
outputs/evaluations/      blinded-ranking evaluation files
archive/                  retired lookup data and historical snapshots
docs/                     design and codebase documentation
```

The data, cache, output, and archive contents are ignored because they may
contain PII. Their `.gitkeep` files preserve the layout in a fresh clone.

## Pipeline

Both workflows share the reviewer loader (`src/reviewer_match/reviewers.py`) and DBLP caches:

```
                       ┌─▶ scripts/classify_reviewers.py ──▶ outputs/reports/reviewer_seniority.csv ──▶ (scripts/assign_reviewers.py)
acceptance CSV ──▶ src/reviewer_match/reviewers.py (+ data/curated/dblp_overrides.csv, + data/inputs/hpca2027-pcinfo.csv)
                       └─▶ scripts/build_fingerprints.py ──▶ data/cache/fingerprints.json ─┐
                               ▲                                      │
              scripts/enrich_publications.py (DBLP DOI + S2 abstracts) ──┘
                                                                        ├─▶ scripts/score_papers.py
paper JSON ──▶ src/reviewer_match/paper_matching.py ──▶ data/cache/paper_fingerprints.json ───────────┘    scripts/assign_reviewers.py
```

`data/curated/dblp_overrides.csv` and `data/inputs/hpca2027-pcinfo.csv` answer two different questions.
The overrides file is the **identity** layer — which DBLP page is this person.
The HotCRP user export is the **membership** layer — is this person still on the
committee. Accepting an invitation is not the same as being on the PC: people
get removed afterwards, and the acceptance form has no way to know. Every roster
loader therefore drops anyone the export does not have on the PC. See
`scripts/audit_pc_roster.py` below, and `make PC_CHECK=--no-pc-check` to run without it.

**Paper-selection policy:** by default (`--paper-policy registered`), every
paper-side tool processes the *registered* papers — every record that has not
been withdrawn, has at least one author, a title of at least one word that
isn't just "test", and an abstract of more than one sentence. HotCRP `status`
stays `draft` until the submission deadline, so before it the content is what
distinguishes a real registration from a placeholder; topics are not
required (a topicless paper simply relies on the area-gate release).

Two alternate policies stay available: `--paper-policy submitted` selects
exactly the records whose HotCRP `status` is `submitted`, ignoring all other
metadata, and requires that every such paper receive a full reviewer slate
and one area chair. `--paper-policy complete` retains the older
title-≥3-words/abstract/topics/authors and withdrawn checks.

**Submissions are in, so `make` now passes `--paper-policy submitted`** —
`PAPER_POLICY` defaults to `submitted` and feeds every paper-side target. The
scripts themselves still default to `registered` when run by hand; `make
PAPER_POLICY=registered` restores the pre-deadline view.

## Start-to-finish workflow

1. **Drop the inputs in place**: the latest acceptance-form CSV export (keep
   the exact filename), a fresh `data/inputs/hpca2027-data.json`, and a fresh
   `data/inputs/hpca2027-pcinfo.csv` user export (HotCRP → Users → download → user
   information), which decides who is still on the PC.
2. **Run `make pc-roster`** and read the two reports it writes. It is offline
   and instant. `outputs/reports/pc_roster_pruned.csv` is who the pipeline is about to stop
   using because HotCRP no longer marks them `pc`; `outputs/reports/pc_roster_missing.csv` is
   the PC accounts no roster explains, most of which are people holding a
   second HotCRP account that wants merging. Both reports are worth acting on
   before an assignment, not after.
3. **Run `make dblp-snapshot`** if the DBLP dump is in place. It is offline and
   takes about four minutes, and it is what the derived co-author COI reads —
   `make` will now stop rather than assign without that layer. Follow it with
   **`make coauthor-coi`** and read
   `outputs/reports/coauthor_conflicts.csv`: the rows with an empty `declared`
   column are conflicts nobody recorded, and they are about to start excluding
   reviewers.
4. **Optionally fill in `S2_API_KEY` in the gitignored `.env` file**,
   then run **`make`** — rebuilds
   whatever is stale, in order: reviewer seniority classification, cached
   IEEE/ACM abstract enrichment, reviewer fingerprints, then the assignment. The final
   outputs land in **`outputs/assignments/assignment.txt`** and
   **`outputs/assignments/assignment.csv`**. The text file contains the per-paper
   reviewer slates, per-area shortage report, and seniority criteria report. The
   CSV is ready for HotCRP's Assignments → Bulk update page: it first clears all
   existing R1 review assignments, then installs the new slate as mandatory
   primary R1 reviews. Always use HotCRP's preview before approving the upload.
   Each row uses the reviewer's **`hotcrp_email`**, not their roster `email`:
   someone `pc_membership` matched under a second address (accepting the
   invitation from one email while HotCRP has their `pc` role under another —
   `outputs/reports/pc_roster_missing.csv`'s `alternate_account` rows) would
   otherwise write an address the bulk upload rejects, since HotCRP checks the
   exact address, not the person behind it. The durable fix is still to merge
   each such pair in HotCRP, as that report says; `hotcrp_email` just keeps a
   merge-pending duplicate from blocking every upload in the meantime.
   Two things this identity resolution also has to get right, both caught by
   HotCRP itself at upload before they were fixed here:
   - **COI is declared against `hotcrp_email`, not the roster key.** A real
     conflict recorded against someone's actual HotCRP account was invisible
     to `paper_matching.eligible_scores` (and `assign_area_chairs.py`'s own
     conflict check) when the roster still keyed them by their acceptance-form
     address — HotCRP's own "cannot review, conflicted" error at upload was
     the only thing that ever caught it. Both now check `pc_conflicts` against
     a candidate's `email` *and* `hotcrp_email`.
   - **Resubmitting the form under a second real address is two roster rows
     for one person**, not caught by `_latest_rows_by_email`'s same-email
     dedup. Left alone, the assignment stage sees two distinct candidates who
     happen to resolve to the same `hotcrp_email`, and can hand the same real
     person two reviews on one paper. `load_reviewers`, `load_area_chairs`
     and `load_reserve_reviewers` now collapse any such pair via
     `pc_membership.collapse_by_hotcrp_email` — latest form timestamp wins,
     the same policy `_latest_rows_by_email` already uses for same-email
     duplicates (first-seen wins for the reserve roster, which has no
     timestamp) — and report it to stderr.
5. **If classify reported reviewers with missing DBLP identities**, it
   appended blank stub rows for them to `data/curated/dblp_overrides.csv` — fill in their
   `dblp` cells and `make` again. Unknowns caused by transient DBLP fetch
   failures are retried and do not create identity stubs.
6. **Ad-hoc follow-ups**: `scripts/score_papers.py --pid N` for one paper's full
   ranking, `scripts/nearest_neighbors.py --email X` to eyeball a reviewer's profile.

The equivalent manual commands, in dependency order:

```bash
~/envs/hpca-matching/bin/python3 -m scripts.classify_reviewers
~/envs/hpca-matching/bin/python3 -m scripts.build_fingerprints
~/envs/hpca-matching/bin/python3 -m scripts.assign_reviewers \
    --paper-policy submitted --include-reserves --reserve-cap 6 \
    --hotcrp-csv outputs/assignments/assignment.csv \
    > outputs/assignments/assignment.txt
```

`make` passes those last two flags for you: the assignment runs over the
submitted set with **both** rosters, reserves capped at `RESERVE_CAP` (default
6), which means `make reserves` has to have run first. `make RESERVE_CAP=off`
assigns from the PC alone. To reproduce the former completeness-based selection
in its own artifacts, without overwriting `outputs/assignments/assignment.txt`:

```bash
make complete-papers          # assignment-complete.txt and assignment-complete.csv
make area-chairs-complete     # outputs/assignments/area_chair_assignment-complete.txt
```

The PC is smaller than the submission volume needs, so `make reserve-need`
sizes the shortfall: how many reserve reviewers have to be recruited to cover
it. It touches neither the assignment nor the fingerprint caches and can be run
at any time. Recruiting the reserves themselves is done outside this repo; once
they are added to HotCRP, `make reserve-info` resolves their DBLP identities
into `outputs/reports/reserve_reviewer_info.csv`.

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

### `scripts/main.py` — DBLP title fetcher (diagnostic)
Prints each reviewer's recent DBLP titles; validates the fetch/cache path.
```bash
python3 -m scripts.main --limit 5 --years 2
```

### `scripts/classify_reviewers.py` — seniority classification
Classifies every accepted reviewer from DBLP publication counts in ISCA,
MICRO, HPCA, and ASPLOS (the target venues) and overall:
- **senior** — ≥ `--senior-rate` (0.8) target-venue papers/year over the last
  `--window` (15) years, i.e. 12+ in-window papers at the defaults;
- **junior** — < `--junior-pubs` (20) publications overall (any venue);
- **out-of-area** — ≥ `--junior-pubs` publications overall but
  < `--out-of-area-career` (5) career target-venue papers;
- **typical** — none of the above (checked in that order, senior first).

Then applies PC-service overrides from `data/inputs/PCDB_with_emails.csv` (`--pcdb`;
`--no-pcdb` skips) to reviewers whose email matches a PCDB row. With
score = `#PC` + 0.5 × `#ERC`, and only ever promoting:
- **senior** — past PC chair (`#Chair` > 0), any TopPicks PC/ERC membership,
  or score ≥ `--pcdb-senior-score` (6);
- **typical** — a junior with score ≥ `--pcdb-typical-score` (2).

A fired override is recorded in the `pcdb_override` column; duplicate PCDB
rows for one email (name variants) merge by summing the counts.

Reviewers HotCRP no longer marks `pc` are dropped before any of this, so they
never reach `outputs/reports/reviewer_seniority.csv` (`--pcinfo` points at a different export;
`--no-pc-check` skips the check). See `scripts/audit_pc_roster.py`.

Writes `outputs/reports/reviewer_seniority.csv`: one row per reviewer with per-venue career
and window counts backing the classification (enough for the assignment step
to spot "almost senior" / "almost not junior" / "almost not out-of-area"
reviewers later). PIDs come
from `data/curated/dblp_overrides.csv` (wins) or the acceptance CSV; anyone left is class
**unknown** with a reason, and gets a stub row appended to
`data/curated/dblp_overrides.csv` (see below). Uncached PIDs are fetched live once into
`data/cache/dblp_venue_cache.json`.
```bash
python3 -m scripts.classify_reviewers
python3 -m scripts.classify_reviewers --window 10 --senior-rate 1.0
```

### `scripts/build_fingerprints.py` — reviewer SPECTER2 fingerprints
Embeds each reviewer (recent DBLP publications + declared areas/keywords) into a
768-dim vector, cached in `data/cache/fingerprints.json` by email. Incremental: cached
reviewers aren't recomputed. Reviewers with no PID get an area-only
fingerprint. Cached entries are automatically refreshed when their reviewer
metadata, PID, selected publications or abstracts, model, or embedding flags
change. Publications use SPECTER2's native `title [SEP] abstract` shape when
an enriched IEEE/ACM abstract is available and fall back to title-only.
```bash
~/envs/hpca-matching/bin/python3 -m scripts.build_fingerprints --limit 10   # validate first
~/envs/hpca-matching/bin/python3 -m scripts.build_fingerprints
```
Key flags: `--years` (4), `--max-titles` (uncapped), `--area-weight` (1.0).
Use `--no-abstracts` with a separate `--fingerprint-cache` to build the
comparison baseline.

### `scripts/enrich_publications.py` — reviewer-publication abstracts
Fetches DOI-bearing DBLP records into `data/cache/reviewer_publications.json`, then
retrieves IEEE and ACM paper abstracts from Semantic Scholar's DOI batch API.
Only DOI prefixes
`10.1109` and `10.1145` are eligible; no publisher pages are scraped.
Successful and confirmed-missing results persist in
`data/cache/publication_abstracts.json`, while DBLP/API failures remain retryable. The
`S2_API_KEY` is optional: without it, requests use Semantic Scholar's shared
unauthenticated rate limit with 429 backoff.

```bash
# .env (gitignored; make loads and exports it automatically)
S2_API_KEY=...

~/envs/hpca-matching/bin/python3 -m scripts.enrich_publications --limit 10
~/envs/hpca-matching/bin/python3 -m scripts.enrich_publications
```

The `.env` file is created with owner-only permissions. Do not put keys in
tracked repository files. Commands invoked directly rather than through
`make` still require loading the file first, for example `set -a; source
.env; set +a`.

### Abstract accuracy experiment
Build a title-only cache, then create the blinded, topic-stratified chair
rating sheet and its separately held rank key:

```bash
~/envs/hpca-matching/bin/python3 -m scripts.build_fingerprints \
  --no-abstracts --fingerprint-cache data/cache/fingerprints-title-only.json
~/envs/hpca-matching/bin/python3 -m scripts.compare_abstract_rankings \
  --baseline-fingerprints data/cache/fingerprints-title-only.json \
  --enriched-fingerprints data/cache/fingerprints.json
# Fill expertise_rating_0_to_3 in outputs/evaluations/abstract-evaluation-ratings.csv, then:
~/envs/hpca-matching/bin/python3 -m scripts.score_abstract_evaluation
```

The scorer reports mean nDCG@10, capable-or-expert fraction in the top six,
and unsuitable candidates in the top six. Adopt enriched fingerprints only
if the first two improve without increasing the third.

#### July 2026 operational comparison

The completed cache contains abstracts for 4,900 of 5,120 eligible IEEE/ACM
DOIs. Abstract-aware profiles used 3,814 abstract-bearing publications across
418 of 454 reviewers; all 454 current fingerprints use schema version 3.

Against a freshly built title-only baseline on the 486 currently complete
papers, abstracts materially changed the result:

- 462 papers (95.1%) had at least one different reviewer in the constrained
  six-person slate;
- 1,085 of 2,916 assignment slots (37.2%) changed, with a mean 3.77 of six
  reviewers retained per paper;
- the highest-ranked eligible reviewer changed for 239 papers (49.2%);
- mean overlap was 66.6% for the top six and 72.1% for the top ten.

These figures establish that enrichment has a large operational effect, not
that it improves ground-truth accuracy. Mean cosine goodness (0.968
title-only versus 0.970 enriched) is not a valid significance test because
the reviewer representation—and therefore the score scale—changes. Complete
the blinded ratings above before claiming a statistically significant quality
improvement.

### `scripts/nearest_neighbors.py` — reviewer/reviewer similarity (diagnostic)
Prints each reviewer's most similar other reviewers by fingerprint cosine —
a sanity check that fingerprints cluster by topic.
```bash
~/envs/hpca-matching/bin/python3 -m scripts.nearest_neighbors --email someone@example.com
```

### `scripts/score_papers.py` — rank reviewers per paper (unconstrained)
Fingerprints each selected paper and prints its top-N reviewers by cosine
similarity, after excluding COI (`pc_conflicts`) and applying the area gate
(reviewer primary/secondary ∩ paper topics; `--no-area-gate` disables).
Per-paper and independent — no load awareness.
```bash
~/envs/hpca-matching/bin/python3 -m scripts.score_papers --pid 8 --top 10
```

### `scripts/assign_reviewers.py` — global load-capped assignment
One assignment across all papers at once, over the reviewers HotCRP still marks
`pc` (`--pcinfo` / `--no-pc-check`), respecting COI, the area gate,
per-reviewer caps (`--light-cap` / `--full-cap`, or the CSV's per-reviewer
override column) and `--reviewers-per-paper`. Solved by paper-proposing
deferred acceptance (Hospital/Residents stable matching), run in phases that
enforce **seniority constraints** from `outputs/reports/reviewer_seniority.csv` — each paper
should get ≥ `--min-seniors` (1) senior reviewers, ≤ `--max-juniors` (1)
juniors, and ≤ `--max-out-of-area` (3) out-of-area reviewers — and a **full
slate** of `--reviewers-per-paper` (5). When the normal constraints can't fill a paper's slate or senior
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
missing, or withdrawn) under the registered and complete policies, and every
relaxed paper, reviewer by reviewer with the released constraint and score.
Under `--paper-policy submitted`, non-submitted records are summarized by
status instead and any unfilled submitted paper makes the command fail; the
registered policy expects shortages, since the registered set is far larger
than the PC's total review capacity. `--no-seniority` skips the seniority
constraints and criteria report (single-pass assignment; the area release
for under-filled papers still applies).
```bash
~/envs/hpca-matching/bin/python3 -m scripts.assign_reviewers --light-cap 7 --full-cap 15 --reviewers-per-paper 5
```

#### Surplus distribution

**On by default at 1.** `--reviewers-per-paper` is the slate every paper is
*guaranteed*; whatever reviewer capacity the six phases leave unspent is then
handed out, one extra reviewer at a time, to the papers whose slates match them
worst. `--surplus-per-paper N` sets how far above the target a single paper may
go (a ceiling, not a second target); `0` leaves the capacity unspent.

Rounds repeat until capacity runs out or nobody is left to help. Each one ranks
the eligible papers by their *current* match goodness, offers the worst `spare`
of them one reviewer each, and draws from the gated pool first and the
area-released pool second — F1 then F2, on a one-slot target. Only papers that
**reached** their base target are eligible: one still short failed the fill
phases minutes earlier under the same rules and a strictly larger pool, so an
offer could not place anything.

A paper that places nothing is dropped from later rounds. That is what makes the
slot **flow on to the next-worst paper**, and it is safe because reviewer
capacity only shrinks while a failed paper's slate stands still — it would fail
identically forever. It also means a round that places nothing is still
productive, so the loop deliberately does *not* stop there; it stops when spare
capacity is gone or no paper is eligible.

What stays hard: COI (all three layers — the stage draws from the same
`eligible_scores` preference lists, so it inherits this for free), the
same-country cap, and the junior/out-of-area caps, whose counts keep including
everything the earlier phases froze. The F3 *almost-not* pools are deliberately
**not** reused: an extra reviewer nobody was owed is not worth breaking
composition policy for.

The stage is **purely additive** — every phase above it is frozen, so no
existing assignment moves. `--reviewers-per-paper` therefore remains the only
number the shortage report, the relaxation report, the criteria report and the
`--paper-policy submitted` exit code are judged against, and no paper is ever
reported short because a surplus slot went to another paper. Running with
`--surplus-per-paper 0` reproduces the pre-surplus output line for line.

One thing to read carefully: **match goodness is a mean**, and a surplus
reviewer almost always scores below the slate that outbid it, so a paper that
*gained* a review shows a *lower* full-slate goodness. The surplus report prints
both figures per paper — goodness over the base slate, which is what compares
against a run with the stage off, and over the full slate — precisely so this
does not read as the assignment getting worse.

#### Same-country cap

**On by default at 2.** A paper whose authors are mostly from country C gets at
most `--same-country-cap N` reviewers affiliated in C. The same rule applies to
every country: a US paper is capped on US reviewers exactly as a Chinese paper
is capped on Chinese ones, and **no country is named in the policy** — the set
of capped countries is whatever the submissions happen to contain (31 in the
current data).

It keys on **where the institution is, never anyone's nationality**: HotCRP
records affiliations and email addresses, not citizenship, and institutional
proximity is what the concern is actually about. **Hong Kong, Macao, Taiwan and
Singapore are their own ISO codes and are never counted as CN**; both DBLP and
HotCRP do write "Hong Kong …, China", so a region name always outranks the
sovereign state it sits in. See "The affiliation-country file" for how an
institution is placed.

`--same-country-cap 0` is a real setting, not the off switch: it admits **no**
reviewer from the paper's own country, which this roster can actually satisfy.
Use `--no-same-country-cap` to disable the policy.

A paper is majority-C when more than `--region-majority` (default 0.5) of the
authors whose country could be **placed** are in C. The denominator is the
placed authors, not all of them — counting unplaced authors against the country
would make thin data look like a non-majority everywhere. To stop that reading
the other way (one placed author out of ten reading as 100%), a paper with fewer
than `--region-min-resolved` (default 0.5) of its authors placed is **not capped
at all** and is listed by name in the report.

The cap is **hard in every phase**, including the senior anchors, the
`fill (cap relaxed)` phase where the junior and out-of-area caps break, and the
surplus distribution that spends what is left over afterwards. A paper
under-fills rather than exceed it, and the shortfall shows up in the shortage
report. `country_cap_report` prints, per country, how many papers were capped,
how many sat at the cap, how many were **left short** by it, and how many
**traded** a better-matched same-country reviewer for another — the second is
the usual cost and the first is the serious one.

Two coverage lines lead the report and matter more here than under a single
named region. A reviewer whose country could not be placed is in no class and
can never consume a cap; a paper whose authors could not be placed is never
judged. Uneven coverage therefore does **not** weaken the rule evenly — it
exempts whoever the resolver is worst at placing. That was not hypothetical:
before `data/curated/affiliation_countries.csv` was filled in, `.cn` and `.kr` were ccTLDs
while `.edu` and `.com` were not, so US institutions were essentially invisible
and **US did not appear once among the majority countries**. Read the coverage
numbers before trusting a run.

**One caveat on stability.** `{juniors, out-of-area, country}` is a *crossing*
family — a reviewer from the paper's country can also be junior or out-of-area —
and greedy-by-score choice over a crossing family is not substitutable, so
paper-proposing deferred acceptance no longer guarantees a stable outcome for
capped papers. The `Done.` line therefore reports `F1 blocking pairs` over
papers with no cap (where the family is laminar and 0 is still guaranteed) and a
separate count over the capped ones. What stays hard everywhere is the thing the
rule needs: no cap is ever exceeded, which must always be 0.

```bash
~/envs/hpca-matching/bin/python3 -m scripts.assign_reviewers                  # cap 2
~/envs/hpca-matching/bin/python3 -m scripts.assign_reviewers --same-country-cap 3
~/envs/hpca-matching/bin/python3 -m scripts.assign_reviewers --no-same-country-cap
make SAME_COUNTRY_CAP=3
```

#### Area chairs are not reviewers

**On by default** (`--no-area-chair-exclusion` to disable, or
`make AREA_CHAIR_CHECK=--no-area-chair-exclusion`). An area chair chairs papers
and reviews none, so every area chair is removed from the reviewer pool before
any phase runs.

Membership is the **union of two sources**, because each one has been observed to
catch people the other misses:

- HotCRP's `~~area-chairs` tag in `data/inputs/hpca2027-pcinfo.csv`
- the area-chair acceptance form

On the current export those are 21 and 18 people and the union is 21; three
tagged accounts never returned the form. The gap has run the other way too — a
signatory the tag did not yet cover — which is why neither source alone is
trusted. `make pc-roster` reports both directions, and the count of area chairs
also sitting on a reviewer roster is the tripwire: if it rises between exports,
someone was made a chair after being staffed with reviews and the assignment is
stale.

Two details worth knowing. The filter runs **after** reserves are merged into the
pool, not on the PC roster alone — one of today's six excluded chairs is a
reserve reviewer, and filtering earlier would be undone by the merge. And
membership is resolved through the same email → name → local-part ladder the PC
check uses (`src/reviewer_match/pc_membership.py`), so a chair who accepted from
one address and reviews under another is still caught.

A missing acceptance form is a **hard error**, for the same reason a missing user
export is: an empty exclusion set is indistinguishable from "no area chairs" and
would silently reinstate every one of them.

The chair roster follows the tag too. An account tagged `~~area-chairs` that
never returned the form still gets a chair assignment, with its DBLP page and
areas borrowed from whichever roster does have it — the PC acceptance form or the
reserve roster. Anyone tagged who is on no roster at all has no identity to work
from, and is reported rather than silently skipped.

#### Derived co-author COI

**On by default, over the last 5 years** (`--coauthor-years`,
`--no-coauthor-coi` to disable). A reviewer who has co-authored a publication
with one of a paper's authors is conflicted with that paper, whether or not
anyone declared it. It reads `data/cache/dblp_coauthors.json`, so `make
dblp-snapshot` has to have run; if the cache is missing the run **stops** rather
than quietly dropping a COI layer.

This is a third layer beside the declared `pc_conflicts` and the
`own_paper_conflicts` floor, and it is meant to be largely redundant with them.
Where it is not, the sweep missed something. On the current data it excludes
22,044 reviewer-paper pairs, **9,508 of which are already declared** — and
12,536 of which are not. The confirmation rate is the reassuring part: it means
the layer mostly agrees with the humans, so the disagreements are worth reading.

Matching is on names and deliberately errs towards firing, because a withheld
reviewer costs one slot out of hundreds while a missed conflict costs the
review. `match` is `exact` when the name-token sets are equal and `partial` when
one is a strict subset of the other sharing ≥2 tokens, which is what lets DBLP's
"David Albert Wood" meet HotCRP's "David Wood". Names with fewer than two usable
tokens never match at all; a lone surname is not evidence.

##### DBLP homonym identity

`dblp.name_tokens` drops DBLP's homonym suffix, which is what makes names
comparable at all — but taken alone it also collapses people DBLP has already
told apart. The most collision-prone names in this data resolve to **more than
twenty distinct DBLP identities each** — the worst reaches 24. Treating those as
one person conflicts a third of the committee with any paper one of them wrote.

So where **both** sides are identified, the numbering is honoured: a reviewer
who wrote with `Wei Zhang 0001` is not conflicted with a paper whose author
declared `Wei Zhang 0012`. `coauthor_coi.identity` compares the suffix, **not
the raw string**, because two spellings of one person differ routinely — "José
García"/"Jose Garcia", "David A. Wood"/"David Wood" — and reading string
inequality as two people would drop real conflicts. It applies to `exact`
matches only; across a `partial` match the two strings are different names and
their numbers are not comparable.

Three limits, all of which keep it from over-reaching:

* It needs the paper's author to have declared a DBLP page. Only **52%** have,
  and only **529 of 5,162** author accounts resolve to a *numbered* homonym, so
  that is the ceiling on how often this can decide anything.
* An author who declared nothing keeps the permissive reading. Not knowing which
  Wei Zhang someone is cannot be evidence that they are not this one.
* An author who declared the **wrong** DBLP page would be compared against the
  wrong identity, which is the one path by which this drops a real conflict.
  Self-declared links are more trustworthy than third-party guesses, but see
  "A wrong PID is worse than no PID" elsewhere in this README.

Measured against the same run with the numbering ignored: 3,473 pairs removed
and **none added**, the declared-confirmation rate up from 38% to 43%, and the
suspected name collisions below down from 20 names / 1,522 conflicts to 13 /
678. Of the 3,473, only 280 were ever declared — and those stay blocked by
`pc_conflicts` regardless, so the layer's decision actually changes for 3,193
pairs. `--no-coauthor-identity` restores the blunt reading for a comparison run.

The window follows `dblp.filter_by_years`: the cutoff is
`current_year - years + 1`, so five years in 2026 means 2022 onwards.

Two things it cannot see, both worth stating rather than papering over. A
reviewer whose PID is missing from the snapshot has no co-author data and passes
the layer **silently** — that is why both the assignment and the audit count
them (currently 0 of 695, so the roster is fully covered). And the dump is a
fixed point in time, so co-authorships newer than it are invisible.

Measured cost on the registered set with reserves, against the same run with the
layer off — taken before the target moved to 5 and before surplus distribution
existed, so read the figures as a comparison between the two runs, not as
today's numbers: identical capacity (6,036 pairs assigned, 2,448 slots unfilled
either way), mean match goodness unchanged at 0.965, and seniority marginally
better (540 papers OK against 536). All the existing self-checks stay 0, and
`co-authored assignments` joins them.

```bash
make coauthor-coi                      # the itemised report, offline, ~6s
make COAUTHOR_COI=--no-coauthor-coi    # assign without it
~/envs/hpca-matching/bin/python3 -m scripts.assign_reviewers --coauthor-years 3
```

#### Derived declared-collaborator COI

**On by default** (`--no-collaborator-coi` to disable). HotCRP's own
Assignments-upload page warns "Proposed reviewer X may conflict with #N"
using two signals `pc_conflicts` never captured — this reproduces them from
`data/inputs/hpca2027-pcinfo.csv`'s `collaborators` column (every HotCRP
account has one, not just PC members; the export is ~5,600 rows, all of
them) rather than leaving them as a warning to notice on a 6,000-row bulk
upload:

* **Name.** A reviewer's declared collaborator is a paper's author, or a
  paper's author (if they hold any HotCRP account) declared the reviewer as
  theirs — checked both ways, since the declaration is one-sided by
  construction. `collaborators` also accepts HotCRP's `All (Institution)`
  shorthand for "everyone there," which never matches by name alone.
* **Affiliation.** The reviewer's own affiliation shares a significant word
  with an author's.

Only the **name** signal excludes. The affiliation signal is also HotCRP's,
and was built the same way at first — but measured on this data it touched
556 of 688 reviewers and 963 of 1156 papers, dominated by generic words
(`hong`, `california`, `chinese`, `computing`, `china`, `shanghai`,
`computer`, `tech`, `georgia`). A corpus document-frequency cutoff cannot
separate that from a genuinely specific match either: `shenzhen`, the
original example that motivated this layer, sits at 1.1% of accounts,
`california` at 2.3%, `shanghai` at 3.5% — the same range, not a different
one. HotCRP's own wording agrees it is a hint, not a rule: "you may want to
**confirm** all potential conflicts." So affiliation overlap is written to
the report and left there; `collaborator_coi.hard_conflicts` is the
name-only subset `assign_reviewers.py` and `assign_area_chairs.py` actually
exclude on.

Like the co-author layer this is meant to be mostly redundant with
`pc_conflicts` — on the current data the name signal excludes ~1,000
reviewer-paper pairs and ~500 chair-paper pairs, most of them already
declared. Where it is not, `scripts/audit_collaborator_conflicts.py` is the
itemised report, `declared` column and all, same convention as the co-author
one.

```bash
make collaborator-coi                            # the itemised report, offline
make COLLABORATOR_COI=--no-collaborator-coi       # assign without it
```

### `scripts/audit_coauthor_conflicts.py` — conflicts nobody declared

Writes `outputs/reports/coauthor_conflicts.csv`, one row per reviewer-paper
conflict the co-author layer finds, always written even when empty. The
`declared` column is the point of the file: `pc_conflicts` means the sweep
already had it, `own_paper` means the person is on the paper, and **empty means
nobody recorded it**. Sorted by `(pid, reviewer_email)` so a re-run against a
fresh export diffs cleanly.

Read the confirmed rows too — they are the control group. If the rows marked
`pc_conflicts` look like real co-authorships, the matching is working and the
empty ones deserve belief. On the current data the declared rows carry visibly
stronger evidence (median 3 shared papers, mean 8.4) than the undeclared ones
(median 2, mean 3.0), which is the expected shape: people declare the
collaborators they think of, and forget the one-off co-author on a large paper.

`reviewers_matched` is the handle on false positives — how many distinct
reviewers that author's name conflicts across every submission. A name reaching
dozens of reviewers is either someone everybody has written with or two people
sharing a name, and the declared rate separates them. The summary calls out
names that reach 20+ reviewers with under 10% ever declared; there are currently
13 such names, accounting for 678 of the undeclared conflicts. They stay blocked
either way — this is the report telling you where it is probably wrong, not a
filter.

What survives here is the residue homonym identity cannot reach: papers where
**no** author declared a DBLP page, so there is nothing to disambiguate against.
The remaining bias is worth stating plainly — the flagged names are almost
entirely ones that romanise into a small space, so the papers losing reviewers
to a collision are disproportionately those with Chinese-affiliated authors.
Getting those authors to fill in the DBLP field is what shrinks it further; no
amount of name matching will.

The flagged names themselves are deliberately **not** reproduced here: naming
them would disclose who submitted, which this file cannot carry. They are in
`outputs/reports/coauthor_conflicts.csv`, which is gitignored.

```bash
make coauthor-coi
~/envs/hpca-matching/bin/python3 -m scripts.audit_coauthor_conflicts --role reviewer
```

### `scripts/audit_collaborator_conflicts.py` — conflicts from declared collaborators

Writes `outputs/reports/collaborator_conflicts.csv`, one row per reviewer-paper
conflict `reviewer_match.collaborator_coi` finds, `declared` column and `kind`
column both — same `pc_conflicts`/`own_paper`/empty convention as the
co-author report, plus `kind` (`name` or `affiliation`) since only `name` is
actually excluded. The `direction` column says which side declared it:
`reviewer_declared` (a reviewer's own `collaborators` field named the
author), `author_declared` (an author's account named the reviewer), or
`institution` (affiliation overlap, either side).

Read it the same way as the co-author report: the declared rows are the
control group, the undeclared `name` rows are what the layer newly excludes,
and the `affiliation` rows — the large majority of the file — are exactly
what did **not** get excluded and are worth spot-checking the `evidence`
column for, since that is where a shared word being too generic to mean
anything would show up.

```bash
make collaborator-coi
~/envs/hpca-matching/bin/python3 -m scripts.audit_collaborator_conflicts --role reviewer
```

### `scripts/build_affiliation_countries.py` — the affiliation-country to-do list

Collects every distinct affiliation string across the submissions and all three
rosters (~1,300), runs the automatic layers over each, and writes
`data/curated/affiliation_countries.csv` with the machine's answer in `suggested` and an
**empty `country` column for a human to fill in**. Only `country` is read back.
The generator never writes it: that column is the hand-decided layer that
outranks DBLP and everything below, so filling it in automatically would collapse
the waterfall into a machine guess. A blank cell is a to-do marker, not a
decision — the same contract as `data/curated/dblp_overrides.csv`.

Reruns are safe. Hand values are carried over verbatim, rows whose affiliation
has left the data are kept with `people = 0` (the export is a moving snapshot and
a withdrawn paper must not delete a decision), and an unchanged rerun is
byte-identical.

`--validate CC=PATH` / `!CC=PATH` checks the resolver against any hand-labelled
CSV with an `email` column and lists the disagreements, which are the highest-value
rows to curate.
```bash
make affiliation-countries
~/envs/hpca-matching/bin/python3 -m scripts.build_affiliation_countries \
    --validate CN=china_faculty.csv --validate '!CN=nonchina_faculty.csv'
```

### `scripts/estimate_reserve_need.py` — size the reserve-reviewer cohort
Pure arithmetic on the selected papers and the PC's per-member caps: how many
review slots the papers need, how many the PC supplies, and how many reserve
reviewers close the gap at `--reviews-per-reserve` (default 4) reviews each. No
embeddings, no network, no GPU. It ignores COI and the area gate, so the count
is a *floor*; pass `--unfilled-slots N` to size the cohort against
`scripts/assign_reviewers.py`'s shortage-report total instead. Prints the cohort size at
several per-reserve loads so the one driving assumption is visible.
```bash
~/envs/hpca-matching/bin/python3 -m scripts.estimate_reserve_need
~/envs/hpca-matching/bin/python3 -m scripts.estimate_reserve_need --reviews-per-reserve 3
```

### `scripts/build_reserve_reviewer_info.py` — reserve-reviewer identities

Joins the two half-rosters the recruited reserves arrive as: HotCRP's upload
(`data/inputs/reserve_reviewer_upload.csv` — account email and name, no DBLP column) and the
vetting workbook (`data/inputs/reserve_reviewers_vetting_final.xlsx` — DBLP links, but also
covering candidates who were never added). The join is on email, and the result
is `outputs/reports/reserve_reviewer_info.csv` (`email,name,dblp`): the reserve-side identity
layer, the counterpart to `data/curated/dblp_overrides.csv` for the PC. The workbook is read
with a small stdlib unzip-and-parse (there is no openpyxl in the venv), so no
conversion step is needed.

A DBLP link belonging to the wrong person is worse than no link — it silently
fingerprints a stranger — so only rows that survive every check reach the
roster. The rest go to `outputs/reports/reserve_reviewer_unresolved.csv` with the reason, which
is the to-do list for hand resolution: `no_dblp_url`, `not_in_vetting`,
`unparseable`, `annotated` (a note trailing the link, e.g. "（disambiguation
page）", which `parse_pid` would otherwise swallow), `name_mismatch` (the PID's
own name is somebody else's), and `shared_pid` (two uploaded emails claim one
PID — either one is wrong, or one person holds two HotCRP accounts and would
draw double the reviews). Both files are sorted by email, so re-running against
a grown HotCRP export gives a readable diff.

Offline, `name_mismatch` can only be checked for the named PID form
(`z/HaoZhang2`); a numeric PID (`26/1737`) carries no name, and numeric is what
most of them are. **`--verify` is what makes the roster trustworthy**: it reads
each PID's own DBLP record, reusing `scripts/resolve_trc_members.py`'s cached,
rate-limited client. On the first run over the 243 recruited reserves it moved
the roster from 225 to 186 — 40 links named somebody else, a 17% error rate that
no offline check could see. Any row left without a usable PID is then searched
for by name and the hits go in the `detail` column as candidates — proposed,
never adopted.

Two practical notes. A plain run after a verified one **overwrites the verified
roster with the weaker offline result**, so prefer `VERIFY=--verify` once the
cache is warm (it is now: a re-run costs 0 fetches and finishes instantly). And
DBLP throttles the *author-search* endpoint far harder than person records —
`get_with_retry` backs off 15→30→60→120→240s per 429, so the search pass over a
few dozen unresolved names can take an hour even when every profile is cached.
```bash
~/envs/hpca-matching/bin/python3 -m scripts.build_reserve_reviewer_info
~/envs/hpca-matching/bin/python3 -m scripts.build_reserve_reviewer_info --verify
make reserve-info VERIFY=--verify
```

### `scripts/resolve_reserve_pids.py` — repair the reserves' DBLP identities

`--verify` proves the workbook's links are wrong but cannot say what is right.
This proposes replacements from the place the reserve reviewers came from: the
submissions. Three routes propose a PID — **self-declared** (the person is an
author, and that submission's positionally-aligned `dblp` field names their
page), **coauthor** (a page listing one of their own submission's co-authors),
and **search** (DBLP author search) — and every surviving candidate is fetched
and checked against its own record before acceptance.

Self-declared is *not* independent evidence: the workbook was largely built from
that same field, so where the alignment slipped the two are wrong together and
agreement proves nothing. That is why a name check is applied to every route.

Output is `data/curated/reserve_dblp_overrides.csv` (`email,dblp,note`) — the hand-maintained
identity layer for reserves, which `scripts/build_reserve_reviewer_info.py` reads *ahead
of* the workbook, exactly as `data/curated/dblp_overrides.csv` outranks the acceptance form.
Rows it could not resolve are written with an **empty `dblp` cell**, so the file
doubles as the to-do list: paste a link in, re-run `make reserve-info
VERIFY=--verify`, and that person joins the roster. A blank cell never masks the
workbook's own value. The `note` column records which pages were considered and
why each was rejected, so a human can finish by eye.
```bash
~/envs/hpca-matching/bin/python3 -m scripts.resolve_reserve_pids
~/envs/hpca-matching/bin/python3 -m scripts.resolve_reserve_pids --no-network
```

### `scripts/build_dblp_snapshot_cache.py` — publications from a local DBLP dump

Asking dblp.org for ~700 person records is more than it will serve politely:
doing it has produced an outright IP block (connections reset), read timeouts,
and 503s. DBLP publishes the whole database as one XML file, and everything this
pipeline asks the network for is in it. This reads that dump once, offline, and
writes the answers to a cache the existing loaders already understand.

The dump has no `pid` attribute anywhere — publications name their authors as
strings. The link comes from the person records, `<www key="homepages/PID">`,
which list the name strings belonging to each PID. So it runs two streaming
passes: the first learns which names belong to the PIDs the rosters want, the
second collects every publication written under one of those names. Matching is
exact rather than fuzzy, because DBLP guarantees a name string identifies one
person — that is what the `0001`/`0049` suffixes are for. Aliases are honoured,
so a paper filed under "F. Alpha" still reaches "François Alpha".

The dump declares a DTD it does not ship and uses named HTML entities, so a
stock parse dies on the first `&ccedil;`; the parser's entity table is seeded
from `html.entities` instead.

Output is `data/cache/dblp_snapshot_cache.json` in the same rich format the colleague cache
uses, so `dblp.load_rich_cache` reads it directly and `dblp.load_colleague_cache`
normalises the very same file to the `[[year, title]]` form
`scripts/build_fingerprints.py` wants — no consumer needs to know where it came from.

It is consulted **only for PIDs the existing caches lack** (`dblp.snapshot_gaps`).
That is deliberate: a fingerprint's cache key includes its publication list, so
re-sourcing an already-cached person would invalidate their fingerprint and
re-embed them for nothing. Gap-filling keeps existing artifacts byte-identical.
```bash
make dblp-snapshot                    # or: --snapshot <dump.xml>
make dblp-snapshot DBLP_SNAPSHOT=dblp-2027-01-01.xml
```
A snapshot is a fixed point in time: anyone added to a roster after it was taken
is absent and is **reported by PID at the end**, not silently left without
publications. Those still need the live path, which remains the fallback.

Pass 1 also writes `data/cache/dblp_affiliations.json`, `{pid: ["Institution, City,
Country", …]}`, from the `<note type="affiliation">` records sitting beside the
name strings. It is a separate file because `--out` has a shape the publication
loaders already read. This is the only place in the pipeline where a country is
stated outright rather than inferred, and it is what lets the region cap place
the roster offline.

Two more files fall out of the same two passes and back the derived co-author
COI. Both are separate from `--out` for the same reason the affiliations are:

* `data/cache/dblp_coauthors.json`, `{pid: {"Co Author": [[year, title], …]}}`.
  Pass 2 already reads every author of every record to decide who owns it, so
  keeping the names costs one dict write. Built for a wider window
  (`--coauthor-years`, default 10) than the COI check enforces, so narrowing
  that check never means re-reading 5 GB.
* `data/cache/dblp_author_names.json`, `{pid: ["Spelling", …]}` — every DBLP
  spelling of the PIDs **submission authors** declare for themselves. Pass 1 is
  one filtered scan and does not care how large its wanted set is, so covering
  the ~2,600 authors who supplied a DBLP link is free. 2,569 of 2,583 resolve.

The publication records themselves are untouched, and deliberately so: they are
built to be indistinguishable from a live fetch, and the fingerprint cache keys
off their contents. Adding a sixth field would re-embed every snapshot-sourced
reviewer for nothing.

All three publication-side files are written with their PIDs **sorted**. They are
keyed in whatever order `owners` — a set of PID strings — happened to iterate,
and Python randomises string hashing per process, so before this the same dump
produced the same data in a different order on every run and `cmp` could not
tell a real change from none.

### `src/reviewer_match/reserve_reviewers.py` — the reserve roster as Reviewer records

Reserve reviewers are recruited from the submissions' `reserve_reviewer`
nominations and added to HotCRP directly, so unlike the PC and the area chairs
they never filled in an acceptance form: `outputs/reports/reserve_reviewer_info.csv` gives an
email, a name and a DBLP page, and nothing else.

Areas matter anyway, because the area gate intersects a reviewer's
primary/secondary with a paper's topics and someone holding neither matches no
paper at all. They are derived here from the HotCRP topics of the submissions
the person authored (falling back to the papers that nominated them), taking the
three most frequent with ties broken alphabetically so the result — and every
fingerprint built from it — is reproducible. Those topics come from HotCRP's own
topic list, so they are already in the gate's vocabulary, with none of the
free-text drift the acceptance form's areas need `build_canonical_area_map` for.
In practice this gives reserves the same reach as the PC: a median of 421
gate-eligible papers against the PC's 450, with nobody reaching zero.

Records come back as real `Reviewer` objects carrying `tier="reserve"`, so
`scripts/enrich_publications.py`, `scripts/build_fingerprints.py` and `scripts/classify_reviewers.py`
all take a reserve unchanged via `--role reserve` (see `src/reviewer_match/roster.py`, which holds
the role-to-loader mapping). `make reserves` runs all three:
```bash
make reserves          # enrich -> fingerprints -> classify
```
It writes `data/cache/reserve_fingerprints.json` and `outputs/reports/reserve_seniority.csv` and leaves the
PC's own `data/cache/fingerprints.json` and `outputs/reports/reviewer_seniority.csv` untouched — deliberate,
since `scripts/classify_reviewers.py` rewrites its whole output CSV and sharing one would
drop every reserve row on any run that didn't also load them.

Seniority uses the identical thresholds and the identical `classify()` as the
PC, including the promote-only PCDB service overrides, so a reserve's
senior/typical/junior/out-of-area class means exactly what a PC member's does.
`scripts/assign_reviewers.py` loads them under `--include-reserves`, where
`tier == "reserve"` takes `--reserve-cap` papers (default 4, matching
`scripts/estimate_reserve_need.py`). The real assignment runs with them at a cap
of 6; `make reserves` has to have run first, since the pool needs
`data/cache/reserve_fingerprints.json` and `outputs/reports/reserve_seniority.csv`.
```bash
~/envs/hpca-matching/bin/python3 -m scripts.assign_reviewers \
    --paper-policy submitted --include-reserves --reserve-cap 6
```

### `scripts/audit_reserve_identities.py` — cross-check the reserves' DBLP pages

A name check cannot tell two people apart when they share a name — a computer
architect and a queuing theorist can be spelled identically — so both `--verify` and the recruiting workbook can be
confidently wrong about the same person. This asks questions a namesake fails,
from four sources that know nothing about each other:

- **co-authors** — the people they wrote HotCRP submissions with should appear
  on their DBLP page. This is the only signal that *positively identifies*
  rather than merely failing to contradict, and two researchers publishing
  together is not something a shared name can fake.
- **affiliation** — HotCRP's institution vs DBLP's recorded one
- **declared** — the DBLP link their own submission gives for them
- **volume** — a page with a handful of papers is unlikely to belong to someone
  invited onto a review committee

A signal only counts when it can speak (a page with no recorded affiliation
neither confirms nor denies). Output is `outputs/reports/reserve_identity_audit.csv`, ranked
most-doubtful first; `confirmed` means a co-author corroborated it. Nothing here
is a verdict — it is a list of people worth opening by hand.
```bash
~/envs/hpca-matching/bin/python3 -m scripts.audit_reserve_identities
```
Beware DBLP's **undisambiguated bucket pages** when acting on it: a page with no
homonym suffix and hundreds of papers (`26/6190` "Hui Yu", 363 papers) is a
holding pen for everyone of that name, not a person, and will score well on
co-author overlap for the wrong reason.

### `scripts/audit_pc_roster.py` — the HotCRP roster cross-check

Two lists have to agree and constantly drift apart: who accepted an invitation
(the acceptance forms and the reserve upload) and who is on the committee in
HotCRP today (`data/inputs/hpca2027-pcinfo.csv`). This reports both directions. It decides
nothing — the loaders in `src/reviewer_match/pc_membership.py` apply the rule; this explains what
the rule did and what it could not account for.

```bash
make pc-roster          # or: scripts/audit_pc_roster.py
```

**The one number that shaped the design:** of the roster rows whose own email
address is not marked `pc`, only a quarter are genuine removals. The rest hold a
*second* HotCRP account that is on the PC — people accept from an institutional
address and keep their account under a personal one, or move institution
mid-cycle. Keying the check on email alone would have removed twelve sitting PC
members on the export this was built against. So a roster row is matched to an
account by email, then by exact name tokens, then by email local part, and only
a row that fails all three is treated as removed. The matching is exact
throughout: a false match merely keeps someone already on the roster, while a
false miss silently removes a real reviewer, so the two error directions are not
equally bad.

`outputs/reports/pc_roster_pruned.csv` — who the loaders drop, with `problem`:

- `no_account` — no HotCRP account under this address and no PC account is the
  same person. The real "removed from HotCRP" case.
- `role_removed` — the account is still there but the `pc` role is gone.
  `detail` keeps the surviving tags, which is what separates a deliberate
  stand-down (a stood-down reserve keeps their `reserve-reviewer` tag) from a
  wiped account.
- `disabled` — on the PC but unable to log in, so unable to review.

`outputs/reports/pc_roster_missing.csv` — PC accounts that appear in neither acceptance form nor
the reserve upload, with `category`. `chair`, `trc`, `sysadmin`, `disabled` and
`area-chair` are settled and need nothing. The rest are work: `alternate_account`
(the person is on a roster under another address — merge the two in HotCRP, and
`make duplicates` lists the pairs), `declined` (their latest form response says
they cannot serve and they are still marked `pc` — the sharpest anomaly here),
and `no_roster_row` (on the PC with no acceptance and no upload row at all).

Both files are always written, even when empty, and sorted by email, so a re-run
against a fresh export diffs cleanly rather than going stale. A useful sanity
signal: `no_roster_row` counts drop to zero as invitees answer the form, so a
number that *stays* up is a real question, not a backlog.

If the export is missing, or is a truncated download in which nothing is marked
`pc`, every script refuses to run rather than pruning the entire roster and
reporting every paper unstaffed. `make PC_CHECK=--no-pc-check` is the deliberate
override, for when the export is staler than the rosters.

### `scripts/find_duplicate_accounts.py` — one person, two HotCRP accounts

Registering twice — once institutionally, once with gmail — is routine, and it
splits that person's conflicts, topics and review load across two accounts. This
lists candidate pairs ranked by confidence (shared ORCID, matching affiliation
or email, name variants), so a human can merge them. It decides nothing.

```bash
make duplicates         # pairs with a PC member on both sides
~/envs/hpca-matching/bin/python3 -m scripts.find_duplicate_accounts --pc-only
```

`--both-pc` is the remediation list for `scripts/audit_pc_roster.py`'s
`alternate_account` rows: two PC accounts mean two review loads and two
half-populated conflict sets for one human, so the matcher can hand them a paper
they are conflicted with under their other address. `--pc-only` widens this to
pairs with a PC member on either side, which also catches an author account
shadowing a PC member. Conflicting ORCIDs *demote* a name match rather than
promoting it: on a roster this size an exact name collision is more often two
people than one.

### `scripts/resolve_trc_members.py` — Training Review Committee roster

Fills two columns into the TRC roster CSV (`data/inputs/hpca2027-trc - hpca2027-trc.csv`),
the cohort of PhD students reviewing alongside the PC: `DBLP` (the student's
page, the identity every fingerprint here is built from) and `Advisor HotCRP
Email` (their advisor's account — a student inherits their advisor's conflicts).
Existing values in either column are left alone unless `--overwrite`; the file
is both the input and the output, so hand edits survive a rerun.

Advisor emails are resolved offline: the PC acceptance form first (its first
question *is* "confirm your HotCRP email address", so it is authoritative even
for those who declined — and those rows, which leave every other column blank,
are still matched when the address's local part names them and its domain is
their institution), then the author and contact lists of the submissions. Where
one name matches several accounts, the roster's advisor-affiliation cell breaks
the tie; where it can't, the cell stays blank and is reported. Advisors
reachable at more than one HotCRP address are listed separately, since the
conflict may need all of them.

Student DBLP identities are proposed by three independent routes — the `dblp`
field of a submission the student is an author on, the co-author list of their
advisor's DBLP record, and DBLP's author search — and then verified against the
candidate's own record. A page is only accepted if its name matches and its
publication count is plausible for a student: DBLP keeps *undisambiguated*
pages under bare common names (its "Cheng Chen" holds 725 papers by many
people), and fingerprinting one of those would poison the matching. Names that
are one person spelt two ways ("Maryam"/"Mariam") are matched only when
co-authorship with the advisor confirms them; same-named co-authors of one
advisor are separated by DBLP's recorded affiliation. Anything still ambiguous
is left blank and reported with the pages that were considered. Two caches,
`data/cache/dblp_profile_cache.json` and `data/cache/dblp_author_search_cache.json`, make reruns free.
```bash
~/envs/hpca-matching/bin/python3 -m scripts.resolve_trc_members
~/envs/hpca-matching/bin/python3 -m scripts.resolve_trc_members --out /tmp/dry-run.csv
~/envs/hpca-matching/bin/python3 -m scripts.resolve_trc_members --no-network   # caches only
```

### `scripts/assign_area_chairs.py` — balanced area-chair assignment

This is an independent workflow layered on the completed reviewer assignment.
It loads the chair roster, builds research fingerprints with the same
DBLP-publication, Semantic Scholar abstract, and declared-area policy used for
reviewers, and assigns every paper with at least one reviewer to one area chair.
Pass it the same `--paper-policy` the reviewer assignment used; under
`submitted`, that assignment must cover the complete submitted-paper set.

The roster is the acceptance form **plus** any account HotCRP tags
`~~area-chairs` that never returned it, whose DBLP page and areas are taken from
the PC form or reserve roster instead — the same union the reviewer-pool
exclusion uses, so the two can never disagree about who is a chair. That comes
from the export, so `--no-pc-check` disables it along with the membership check.

Conflicts are the same three layers the reviewer matcher applies — declared
`pc_conflicts`, the `own_paper_conflicts` floor, and the derived co-author layer
— and every one of them is hard here, because a missing edge is a route the
min-cost flow simply cannot take. Until recently only the declared layer was
applied, which meant a chair could in principle be handed a paper they wrote;
on the current export nothing actually slips through that gap (every chair
authorship is declared), but the floor should not depend on the sweep being
complete. The co-author layer excludes a further 389 chair-paper pairs, 131 of
them declared nowhere.

Twenty-odd chairs is a thin pool, so tightening COI can make the load bounds
infeasible where the reviewer matcher would merely under-fill. That surfaces as
a `ValueError` rather than a silently dropped conflict.

Area-chair profiles use a 10-year publication window (reviewer profiles retain
their four-year default), giving the smaller chair pool a deeper research
history. The optimizer maximizes total SPECTER2 cosine affinity globally while
using the closest possible floor/ceiling load balance; for 56 papers and 15
chairs, loads are 3–4. A wider experimental band can be requested with
`--load-tolerance`. The report is grouped by area chair, with
assigned paper IDs, titles, topics, and scores beneath each chair, followed by
affinity, conflict, coverage, and load-bound checks.

```bash
make area-chairs
# output: outputs/assignments/area_chair_assignment.txt
```

`make area-chairs` also writes two HotCRP bulk-upload CSVs implementing the
per-area Tracks feature, so an area chair can see reviewer identities and full
review content — hidden from non-conflicted PC by default — and post
discussion comments on their own papers without being expected to write a
review themselves. Each chair is numbered 0-indexed in the same order the
`.txt` report already prints them (`sorted(chair_emails, key=lambda e:
(chairs_by_email[e].name.lower(), e))`), and track N is written with **two
different tags in two separate HotCRP namespaces**; the report prints both
under each chair's `track:` line for a human-auditable record:

- the **papers** of track N are tagged plain `track_N`. Not `~~track_N`: a
  `~~` tag is chair-hidden, so the tag would be invisible — and unsearchable —
  to the one person it exists for, the chair who wants `#track_N` to pull up
  their pile. Nothing is lost by making it plain, because HotCRP registers any
  tag that *names* a track as chair-readonly (`TF_TRACK | TFM_ADMIN_PUBLIC |
  TF_CHAIR_READONLY`, `lib/tagger.php` upstream), so an ordinary PC member
  still cannot tag a paper into a track. Track membership itself is a raw
  `$prow->has_tag()` string test (`Conf::check_tracks`), viewer-independent, so
  the spelling changes visibility and nothing else.
- the **chair's PC account** is tagged `~~track_N`, and stays chair-only:
  it is the chair-to-track mapping, which the PC has no reason to see. This is
  the tag the track's permissions name, under Settings → Tags & tracks (`Who
  can see reviewer names?` / `Which non-reviewers can add comments?` → "PC
  members with tag `~~track_N`").

The tracks themselves are **not** created by this repo — they're set up by
hand in HotCRP first, one per chair, named `track_0` .. `track_23` for 24
chairs, and these CSVs only bulk-assign existing tags:

- `outputs/assignments/area_chair_account_tags.csv` — upload via **Settings →
  Accounts** bulk update, header `email,remove_tags,add_tags`, one row per
  chair. **Never use the plain `tags` column here**: HotCRP's bulk user
  importer (`src/userstatus.php`'s `$csv_keys`/`parse_json` in upstream
  HotCRP) treats a bare `tags` value as a full replacement of that account's
  entire tag set, silently wiping unrelated tags like `~~area-chairs` or `pc`.
  `remove_tags`/`add_tags` are additive/subtractive only, touching nothing
  else.
- `outputs/assignments/area_chair_paper_tags.csv` — upload via **Assignments →
  Bulk update**, header `paper,action,email,tag,round` with `action=tag` rows,
  one per paper, tagging it into its chair's track. This is a separate upload
  from `outputs/assignments/assignment.csv`: that file starts with
  `all,clearreview,all,R1` and owns the reviewer-review lifecycle, so track
  tagging is kept as its own independent bulk-update pass rather than coupled
  to reviewer-assignment re-uploads. As always, use HotCRP's preview before
  approving either upload.

Both CSVs are **replacements, not additions**, for the same reason the
reviewer-assignment CSV opens with `all,clearreview,all,R1`: as the matching
stabilizes and papers move between chairs (or a chair's 0-indexed track
number shifts because the sorted chair order changed), a stale tag from the
previous run has to go, not just accumulate alongside the new one.
`--track-clear-ceiling` (default 50, comfortably above any realistic chair
count so a shrunk roster's stale higher-numbered tags still get cleared)
governs how many track numbers get cleared before reapplying, in each
namespace's own spelling: `area_chair_paper_tags.csv` opens with one
`all,cleartag,,track_N,` row per track (paper tags), and each row of
`area_chair_account_tags.csv` carries the full `~~track_N` range in
`remove_tags` before `add_tags` grants the current one (account tags).
Re-running `make area-chairs` and re-uploading both files is therefore
enough to keep tags in sync for anyone still on the chair roster — **only**
gap is a chair who leaves the roster entirely between runs: with no row for
them in either CSV, their stale tags (both the paper tags on their old papers
and their own account's `~~track_N`) are untouched. `make clear-uploads`
closes that gap; see below.

The workflow reuses the DBLP, publication metadata, abstract, and paper
fingerprint caches, but writes chair vectors to
`data/cache/area_chair_fingerprints.json`. It is not part of the default `make` target.

### `scripts/generate_clear_uploads.py` — wipe assignments and track tags

**`make clear-uploads`** writes the undo half of both uploads — the same
`clearreview`, `cleartag` and `remove_tags` operations they open with, with
nothing reinstalled afterwards. Offline, instant, read-only with respect to
every input. Three files, because HotCRP takes them on two pages in two
formats:

- `outputs/assignments/clear_assignment.csv` — **Assignments → Bulk update**,
  a single `all,clearreview,all,R1` row removing every R1 review assignment.
  R1 is the only round `assign_reviewers.py` ever installs into, so it is the
  only one cleared by default; `make clear-uploads CLEAR_ROUND=all` leaves the
  round cell empty, which HotCRP reads as every round.
- `outputs/assignments/clear_paper_tags.csv` — **Assignments → Bulk update**,
  one `all,cleartag,,track_N,` row per track number. `cleartag` takes the tag
  off every paper carrying it, so this is one row per track regardless of how
  many papers hold it.
- `outputs/assignments/clear_account_tags.csv` — **Settings → Accounts**, one
  `email,remove_tags,add_tags` row per account, `remove_tags` carrying the
  whole `~~track_N` range and `add_tags` left empty. The plain `tags` column
  stays off limits for the reason given above (HotCRP reads it as a full
  replacement of the account's tag set). The empty `add_tags` column has to be
  **present**, which is not obvious: `p_profile.php`'s `save_bulk` decides
  whether an upload is a CSV or a plain list of email addresses *before* it
  looks at the header, and one of the two tests is whether the first line holds
  at least two commas. A two-column `email,remove_tags` header holds one, and
  on a HotCRP without the newer `(?:user|email)` test beside it every line is
  re-quoted into a single field — the header stops being a header and each row
  is validated whole as an email address, failing with **"Invalid email
  address" on line 2** and every line after it. Three columns never reach that
  branch, and the empty cell is a verified no-op: `UserStatus::parse_csv_main`
  skips any column whose trimmed value is empty, so `add_tags` is never set on
  the update object at all.

Nothing else is touched: not `~~area-chairs`, not any other tag, not
conflicts, not review preferences. As always, preview each upload in HotCRP
before approving it.

Two things it does that re-running `make area-chairs` cannot:

- **The account rows come from the HotCRP user export, not the chair roster.**
  That is what reaches the chair who left the roster between runs — invisible
  to `area_chair_account_tags.csv`, but still carrying `~~track_N` in the
  export. The chair roster is unioned in as a second source, since an export
  taken *before* the last tag upload knows nothing about the tags that upload
  wrote. Every row is an address the export itself lists, and the account's own
  address rather than an acceptance-form one: HotCRP's bulk user importer
  **creates** an account for an unknown email, so a clearing file naming a
  form-only address would quietly add users.
- **Track tags outside the canonical range are cleared too**, read off the
  paper export and the user export rather than assumed. Setting the track
  mechanism up by hand left a `track1`/`~~track1` pair on the live data that
  clearing `track_0..track_49` would strand. A tag that merely contains the
  word (`TRC-track`) does not match — that is a separate track and is nobody's
  to clear here.

The tag spellings and the clear ceiling are imported from
`assign_area_chairs.py`, never restated: a clearing file that spelled a tag
differently from the file that installed it would report success and leave the
tag in place.

## Publication exclusions

`data/curated/publication_exclusions.csv` is an optional hand-maintained file with columns
`email,doi,note`. Each row removes that DOI only from the named researcher's
fingerprint; it does not alter the shared DBLP metadata or abstract caches.
Emails are case-insensitive, and DOI URLs or bare DOIs are accepted. Both
reviewer and area-chair fingerprint builds load the file automatically.

```csv
email,doi,note
person@example.com,10.1109/example.1,Incorrect or unrepresentative publication
```

An exclusion that names a researcher in the current build but does not match
their selected publication window is reported as a warning. Removing the row
and rebuilding restores the publication.

## The DBLP override file

`data/curated/dblp_overrides.csv` (columns `email,dblp,note`) is the **single
hand-maintained identity layer**, keyed by email so it survives
acceptance-CSV re-exports. A filled-in `dblp` cell (any link shape or bare
PID) **wins over the form's own DBLP column** — use it to fill in missing
links or correct wrong ones (e.g. a namesake's page). Rows with a blank
`dblp` cell are ignored, and `scripts/classify_reviewers.py` appends one such stub
per still-unknown reviewer (name/affiliation/reason in the note), so the
file doubles as the to-do list: fill in the blank cells and rerun.

It absorbed the output of a retired semi-automated lookup chain
(`lookup_no_dblp_src/reviewer_match/reviewers.py` / `apply_human_guesses.py`, removed July
2026) that bulk-resolved the original ~57 no-DBLP reviewers; rows noting
"migrated from final_identity_resolution.csv" came from there.

## The affiliation-country file

`data/curated/affiliation_countries.csv` (columns `affiliation,country,suggested,source,
people,note`) is the hand-maintained layer under the region cap, keyed by the
normalized affiliation string. Same contract as `data/curated/dblp_overrides.csv`: a filled
`country` cell **wins over every automatic layer**, a blank cell is a to-do
marker rather than a decision, and `scripts/build_affiliation_countries.py` regenerates
the file with blanks so it doubles as the to-do list. `suggested` and `source`
are the machine's independent opinion and are never read back — only `country`
is.

`src/reviewer_match/affiliation_country.py` resolves an institution in four layers, first hit wins,
and **nothing is ever guessed**:

1. **`data/curated/affiliation_countries.csv`** — the hand layer.
2. **DBLP's `<note type="affiliation">`.** A profile often carries several and
   *their order means nothing* (a Tsinghua professor's notes can list UC Santa
   Barbara first), so the note is chosen by how well it matches the affiliation
   the person gave HotCRP; if none matches, this layer declines rather than
   answering with a former employer's country. The whole note is scanned for a
   country name, not just its trailing field, because DBLP writes "Hong Kong
   University of Science and Technology, …, China".
3. **A country or region name in the affiliation string**, matched on whole
   tokens. Adjectives are deliberately not names — "Chinese" would place "The
   Chinese University of Hong Kong" in CN — and names that are also common
   institution or place words ("Georgia", "Jordan", "Turkey") are excluded
   rather than producing confident wrong answers.
4. **The email ccTLD**, ignoring the TLDs sold generically (`.com`, `.edu`,
   `.io`, `.ai`, `.co`, …), which say nothing about location.

Anything the four layers cannot place stays **unresolved** and is reported, never
assumed. That is why `scripts/assign_reviewers.py` prints reviewer and paper coverage
whenever a region cap is on: an unplaced reviewer can never consume a cap.

## Support modules (not standalone scripts)

`src/reviewer_match/reviewers.py` (acceptance-CSV parsing, duplicate-submission collapsing,
override application) · `src/reviewer_match/pc_membership.py` (the HotCRP account model, name
comparison, and the "is this person still on the PC" predicate every roster
loader applies) · `src/reviewer_match/dblp.py` (DBLP fetch, caching, rate limiting) ·
`src/reviewer_match/paper_matching.py` (paper selection, fingerprinting, and eligibility) ·
`src/reviewer_match/fingerprint.py` / `src/reviewer_match/specter2_model.py` (embedding plumbing).

## Data files

**Inputs:** the reviewer and area-chair acceptance-form CSVs under
`data/inputs/` (Google Forms exports — real names and emails, treat as
sensitive), `data/inputs/hpca2027-data.json`
(HotCRP paper export), `data/inputs/hpca2027-pcinfo.csv` (the HotCRP *user* export — names,
emails, ORCIDs, affiliations and declared collaborators, so among the most
sensitive files here; it is the authority on who is on the PC, and like
`data/inputs/hpca2027-data.json` it is a moving snapshot, so a stale one is exactly what the
`make pc-roster` reports exist to surface), `data/inputs/dblp_pubs_cache.json` (colleague's
read-only rich DBLP cache), and `data/inputs/PCDB_with_emails.csv` (PC-service history with
emails — also sensitive). `data/inputs/hpca2027-trc - hpca2027-trc.csv` is the Training Review Committee
roster: an input that `scripts/resolve_trc_members.py` writes its two resolved columns
back into, so it is also hand-maintained. `data/inputs/reserve_reviewer_upload.csv` (the
HotCRP reserve-reviewer upload) and `data/inputs/reserve_reviewers_vetting_final.xlsx` (the
recruiting workbook holding their DBLP links) are the inputs to
`scripts/build_reserve_reviewer_info.py` — also sensitive.

**Hand-maintained:** `data/curated/dblp_overrides.csv`, `data/curated/publication_exclusions.csv`, and
`data/curated/reserve_dblp_overrides.csv`, and
`data/curated/affiliation_countries.csv`. The reserve override is written by
`scripts/resolve_reserve_pids.py` and finished by hand. Filling in one of its blank
`dblp` cells and re-running `make reserve-info VERIFY=--verify` is how a row
leaves `outputs/reports/reserve_reviewer_unresolved.csv`; correcting
`data/inputs/reserve_reviewers_vetting_final.xlsx` works too, but the override file wins.

**Caches** (reproducible, but potentially expensive to rebuild): `data/cache/dblp_cache.json`,
`data/cache/dblp_venue_cache.json`, `data/cache/fingerprints.json`,
`data/cache/area_chair_fingerprints.json`, `data/cache/paper_fingerprints.json`,
`data/cache/reviewer_publications.json`, `data/cache/publication_abstracts.json`,
`data/cache/dblp_profile_cache.json`, `data/cache/dblp_author_search_cache.json`,
and experimental fingerprint caches such as `data/cache/fingerprints-title-only.json`.
Live DBLP and abstract retrieval is rate-limited, so prefer the targeted clean
targets over deleting this directory.

**Outputs** (human-facing and safe to regenerate):
`outputs/reports/reviewer_seniority.csv`, `outputs/assignments/assignment.txt`, `outputs/assignments/area_chair_assignment.txt`,
`outputs/reports/reserve_reviewer_info.csv`, `outputs/reports/reserve_reviewer_unresolved.csv`,
`outputs/reports/pc_roster_pruned.csv`, `outputs/reports/pc_roster_missing.csv`, and
`outputs/reports/duplicate_accounts.csv`.

**Retired** (left over from the removed lookup chain; kept only as
historical reference, nothing reads them):
`archive/retired-identity-resolution/no_dblp_lookup_report.csv`,
`archive/retired-identity-resolution/manual_review_report*.csv`,
`archive/retired-identity-resolution/final_identity_resolution.csv`, and the
contents of `archive/retired-identity-resolution/old_json_files/`.

Unsafe numeric inputs (negative capacities or targets, nonpositive embedding
weights/windows) fail before network or model work. Executable but
contradictory policy combinations print a warning and continue so the
criteria report can show their consequences.

All PII-bearing files above are gitignored; only code and docs are
committed.
