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
                               ▲                                      │
              enrich_publications.py (DBLP DOI + S2 abstracts) ──┘
                                                                        ├─▶ score_papers.py
paper JSON ──▶ paper_matching.py ──▶ paper_fingerprints.json ───────────┘    assign_reviewers.py
```

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
and one area chair — switch to it once submissions are in. `--paper-policy
complete` retains the older title-≥3-words/abstract/topics/authors and
withdrawn checks.

## Start-to-finish workflow

1. **Drop the inputs in place**: the latest acceptance-form CSV export (keep
   the exact filename) and a fresh `hpca2027-data.json` from HotCRP.
2. **Optionally fill in `S2_API_KEY` in the gitignored `.env` file**,
   then run **`make`** — rebuilds
   whatever is stale, in order: reviewer seniority classification, cached
   IEEE/ACM abstract enrichment, reviewer fingerprints, then the assignment. The final
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

Once HotCRP has real submissions, build the submitted-only assignment with
`make PAPER_POLICY=submitted` (and `make PAPER_POLICY=submitted area-chairs`);
`PAPER_POLICY` defaults to `registered` and feeds both targets. To reproduce
the former completeness-based selection in its own artifacts, without
overwriting `assignment.txt`:

```bash
make complete-papers          # assignment-complete.txt
make area-chairs-complete     # area_chair_assignment-complete.txt
```

The PC is smaller than the submission volume needs, so `make reserve-need`
sizes the shortfall: how many reserve reviewers have to be recruited to cover
it. It touches neither the assignment nor the fingerprint caches and can be run
at any time. Recruiting the reserves themselves is done outside this repo; once
they are added to HotCRP, `make reserve-info` resolves their DBLP identities
into `reserve_reviewer_info.csv`.

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
Embeds each reviewer (recent DBLP publications + declared areas/keywords) into a
768-dim vector, cached in `fingerprints.json` by email. Incremental: cached
reviewers aren't recomputed. Reviewers with no PID get an area-only
fingerprint. Cached entries are automatically refreshed when their reviewer
metadata, PID, selected publications or abstracts, model, or embedding flags
change. Publications use SPECTER2's native `title [SEP] abstract` shape when
an enriched IEEE/ACM abstract is available and fall back to title-only.
```bash
~/envs/hpca-matching/bin/python3 build_fingerprints.py --limit 10   # validate first
~/envs/hpca-matching/bin/python3 build_fingerprints.py
```
Key flags: `--years` (4), `--max-titles` (uncapped), `--area-weight` (1.0).
Use `--no-abstracts` with a separate `--fingerprint-cache` to build the
comparison baseline.

### `enrich_publications.py` — reviewer-publication abstracts
Fetches DOI-bearing DBLP records into `reviewer_publications.json`, then
retrieves IEEE and ACM paper abstracts from Semantic Scholar's DOI batch API.
Only DOI prefixes
`10.1109` and `10.1145` are eligible; no publisher pages are scraped.
Successful and confirmed-missing results persist in
`publication_abstracts.json`, while DBLP/API failures remain retryable. The
`S2_API_KEY` is optional: without it, requests use Semantic Scholar's shared
unauthenticated rate limit with 429 backoff.

```bash
# .env (gitignored; make loads and exports it automatically)
S2_API_KEY=...

~/envs/hpca-matching/bin/python3 enrich_publications.py --limit 10
~/envs/hpca-matching/bin/python3 enrich_publications.py
```

The `.env` file is created with owner-only permissions. Do not put keys in
tracked repository files. Commands invoked directly rather than through
`make` still require loading the file first, for example `set -a; source
.env; set +a`.

### Abstract accuracy experiment
Build a title-only cache, then create the blinded, topic-stratified chair
rating sheet and its separately held rank key:

```bash
~/envs/hpca-matching/bin/python3 build_fingerprints.py \
  --no-abstracts --fingerprint-cache fingerprints-title-only.json
~/envs/hpca-matching/bin/python3 compare_abstract_rankings.py \
  --baseline-fingerprints fingerprints-title-only.json \
  --enriched-fingerprints fingerprints.json
# Fill expertise_rating_0_to_3 in abstract-evaluation-ratings.csv, then:
~/envs/hpca-matching/bin/python3 score_abstract_evaluation.py
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

### `nearest_neighbors.py` — reviewer/reviewer similarity (diagnostic)
Prints each reviewer's most similar other reviewers by fingerprint cosine —
a sanity check that fingerprints cluster by topic.
```bash
~/envs/hpca-matching/bin/python3 nearest_neighbors.py --email someone@example.com
```

### `score_papers.py` — rank reviewers per paper (unconstrained)
Fingerprints each selected paper and prints its top-N reviewers by cosine
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
missing, or withdrawn) under the registered and complete policies, and every
relaxed paper, reviewer by reviewer with the released constraint and score.
Under `--paper-policy submitted`, non-submitted records are summarized by
status instead and any unfilled submitted paper makes the command fail; the
registered policy expects shortages, since the registered set is far larger
than the PC's total review capacity. `--no-seniority` skips the seniority
constraints and criteria report (single-pass assignment; the area release
for under-filled papers still applies).
```bash
~/envs/hpca-matching/bin/python3 assign_reviewers.py --light-cap 7 --full-cap 15 --reviewers-per-paper 6
```

### `estimate_reserve_need.py` — size the reserve-reviewer cohort
Pure arithmetic on the selected papers and the PC's per-member caps: how many
review slots the papers need, how many the PC supplies, and how many reserve
reviewers close the gap at `--reviews-per-reserve` (default 4) reviews each. No
embeddings, no network, no GPU. It ignores COI and the area gate, so the count
is a *floor*; pass `--unfilled-slots N` to size the cohort against
`assign_reviewers.py`'s shortage-report total instead. Prints the cohort size at
several per-reserve loads so the one driving assumption is visible.
```bash
~/envs/hpca-matching/bin/python3 estimate_reserve_need.py
~/envs/hpca-matching/bin/python3 estimate_reserve_need.py --reviews-per-reserve 3
```

### `build_reserve_reviewer_info.py` — reserve-reviewer identities

Joins the two half-rosters the recruited reserves arrive as: HotCRP's upload
(`reserve_reviewer_upload.csv` — account email and name, no DBLP column) and the
vetting workbook (`reserve_reviewers_vetting_final.xlsx` — DBLP links, but also
covering candidates who were never added). The join is on email, and the result
is `reserve_reviewer_info.csv` (`email,name,dblp`): the reserve-side identity
layer, the counterpart to `dblp_overrides.csv` for the PC. The workbook is read
with a small stdlib unzip-and-parse (there is no openpyxl in the venv), so no
conversion step is needed.

A DBLP link belonging to the wrong person is worse than no link — it silently
fingerprints a stranger — so only rows that survive every check reach the
roster. The rest go to `reserve_reviewer_unresolved.csv` with the reason, which
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
each PID's own DBLP record, reusing `resolve_trc_members.py`'s cached,
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
~/envs/hpca-matching/bin/python3 build_reserve_reviewer_info.py
~/envs/hpca-matching/bin/python3 build_reserve_reviewer_info.py --verify
make reserve-info VERIFY=--verify
```

### `resolve_reserve_pids.py` — repair the reserves' DBLP identities

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

Output is `reserve_dblp_overrides.csv` (`email,dblp,note`) — the hand-maintained
identity layer for reserves, which `build_reserve_reviewer_info.py` reads *ahead
of* the workbook, exactly as `dblp_overrides.csv` outranks the acceptance form.
Rows it could not resolve are written with an **empty `dblp` cell**, so the file
doubles as the to-do list: paste a link in, re-run `make reserve-info
VERIFY=--verify`, and that person joins the roster. A blank cell never masks the
workbook's own value. The `note` column records which pages were considered and
why each was rejected, so a human can finish by eye.
```bash
~/envs/hpca-matching/bin/python3 resolve_reserve_pids.py
~/envs/hpca-matching/bin/python3 resolve_reserve_pids.py --no-network
```

### `build_dblp_snapshot_cache.py` — publications from a local DBLP dump

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

Output is `dblp_snapshot_cache.json` in the same rich format the colleague cache
uses, so `dblp.load_rich_cache` reads it directly and `dblp.load_colleague_cache`
normalises the very same file to the `[[year, title]]` form
`build_fingerprints.py` wants — no consumer needs to know where it came from.

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

### `reserve_reviewers.py` — the reserve roster as Reviewer records

Reserve reviewers are recruited from the submissions' `reserve_reviewer`
nominations and added to HotCRP directly, so unlike the PC and the area chairs
they never filled in an acceptance form: `reserve_reviewer_info.csv` gives an
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
`enrich_publications.py`, `build_fingerprints.py` and `classify_reviewers.py`
all take a reserve unchanged via `--role reserve` (see `roster.py`, which holds
the role-to-loader mapping). `make reserves` runs all three:
```bash
make reserves          # enrich -> fingerprints -> classify
```
It writes `reserve_fingerprints.json` and `reserve_seniority.csv` and leaves the
PC's own `fingerprints.json` and `reviewer_seniority.csv` untouched — deliberate,
since `classify_reviewers.py` rewrites its whole output CSV and sharing one would
drop every reserve row on any run that didn't also load them.

Seniority uses the identical thresholds and the identical `classify()` as the
PC, including the promote-only PCDB service overrides, so a reserve's
senior/typical/junior/out-of-area class means exactly what a PC member's does.
`assign_reviewers.py` knows `tier == "reserve"` takes `--reserve-cap` papers
(default 4, matching `estimate_reserve_need.py`); reserves are not yet loaded
into the assignment itself.

### `make_smoke_dataset.py` + `make smoke` — rehearse a full assignment

Registration is open, so the export holds far more papers than will ever need
reviewing — about 1,414 pass the `registered` policy against a pool that can
cover roughly 1,070. Assigning against all of them only measures the shortfall.

`make_smoke_dataset.py` writes `hpca2027-data-smoke.json`, a copy with a seeded
random `--fraction` (default **0.30**) of the selectable papers *marked
withdrawn* — not deleted, so `paper_matching`'s own `_is_withdrawn` drops them
through the same path a real withdrawal takes. The seed is fixed, so two runs
compare against an unmoving paper set; change `--fraction` to 0.25 for a tighter
case.

`make smoke` then runs the assignment over that set with **both** rosters
(`--include-reserves`), reserves capped at 6 papers, writing
`assignment-smoke.txt`.
```bash
make smoke                       # 30% withdrawn, reserves at 6
make smoke SMOKE_WITHDRAWN=0.25  # tighter
```
**Read the self-checks, not the shortage count.** Over-cap, blocking pairs and
the junior/out-of-area policy counts must all be 0 — those are the pass/fail. A
shortage is a statement about capacity, not about the matcher.

### `audit_reserve_identities.py` — cross-check the reserves' DBLP pages

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
neither confirms nor denies). Output is `reserve_identity_audit.csv`, ranked
most-doubtful first; `confirmed` means a co-author corroborated it. Nothing here
is a verdict — it is a list of people worth opening by hand.
```bash
~/envs/hpca-matching/bin/python3 audit_reserve_identities.py
```
Beware DBLP's **undisambiguated bucket pages** when acting on it: a page with no
homonym suffix and hundreds of papers (`26/6190` "Hui Yu", 363 papers) is a
holding pen for everyone of that name, not a person, and will score well on
co-author overlap for the wrong reason.

### `resolve_trc_members.py` — Training Review Committee roster

Fills two columns into the TRC roster CSV (`hpca2027-trc - hpca2027-trc.csv`),
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
`dblp_profile_cache.json` and `dblp_author_search_cache.json`, make reruns free.
```bash
~/envs/hpca-matching/bin/python3 resolve_trc_members.py
~/envs/hpca-matching/bin/python3 resolve_trc_members.py --out /tmp/dry-run.csv
~/envs/hpca-matching/bin/python3 resolve_trc_members.py --no-network   # caches only
```

### `assign_area_chairs.py` — balanced area-chair assignment

This is an independent workflow layered on the completed reviewer assignment.
It loads accepted responses from the area-chair form, builds research
fingerprints with the same DBLP-publication, Semantic Scholar abstract, and
declared-area policy used for reviewers, and assigns every paper with at least
one reviewer to one area chair. Pass it the same `--paper-policy` the
reviewer assignment used; under `submitted`, that assignment must cover the
complete submitted-paper set. HotCRP conflicts are hard exclusions.

Area-chair profiles use a 10-year publication window (reviewer profiles retain
their four-year default), giving the smaller chair pool a deeper research
history. The optimizer maximizes total SPECTER2 cosine affinity globally while
keeping each chair within 10% of the mean load when integer bounds permit it.
Otherwise it uses the closest possible floor/ceiling balance; for 56 papers
and 15 chairs, loads are 3–4. The report is grouped by area chair, with
assigned paper IDs, titles, topics, and scores beneath each chair, followed by
affinity, conflict, coverage, and load-bound checks.

```bash
make area-chairs
# output: area_chair_assignment.txt
```

The workflow reuses the DBLP, publication metadata, abstract, and paper
fingerprint caches, but writes chair vectors to
`area_chair_fingerprints.json`. It is not part of the default `make` target.

## Publication exclusions

`publication_exclusions.csv` is an optional hand-maintained file with columns
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
`paper_matching.py` (paper selection, fingerprinting, and eligibility) ·
`fingerprint.py` / `specter2_model.py` (embedding plumbing).

## Data files

**Inputs:** the reviewer and area-chair acceptance-form CSVs (Google Forms
exports — real names and emails, treat as sensitive), `hpca2027-data.json`
(HotCRP paper export), `dblp_pubs_cache.json` (colleague's read-only rich DBLP
cache), and `PCDB_with_emails.csv` (PC-service history with emails — also
sensitive). `hpca2027-trc - hpca2027-trc.csv` is the Training Review Committee
roster: an input that `resolve_trc_members.py` writes its two resolved columns
back into, so it is also hand-maintained. `reserve_reviewer_upload.csv` (the
HotCRP reserve-reviewer upload) and `reserve_reviewers_vetting_final.xlsx` (the
recruiting workbook holding their DBLP links) are the inputs to
`build_reserve_reviewer_info.py` — also sensitive.

**Hand-maintained:** `dblp_overrides.csv`, `publication_exclusions.csv`, and
`reserve_dblp_overrides.csv` — the reserve identity layer written by
`resolve_reserve_pids.py` and finished by hand. Filling in one of its blank
`dblp` cells and re-running `make reserve-info VERIFY=--verify` is how a row
leaves `reserve_reviewer_unresolved.csv`; correcting
`reserve_reviewers_vetting_final.xlsx` works too, but the override file wins.

**Generated** (safe to delete; rebuilt incrementally): `dblp_cache.json`,
`dblp_venue_cache.json`, `fingerprints.json`,
`area_chair_fingerprints.json`, `paper_fingerprints.json`,
`reviewer_publications.json`, `publication_abstracts.json`,
`dblp_profile_cache.json`, `dblp_author_search_cache.json`,
`reviewer_seniority.csv`, `assignment.txt`, `area_chair_assignment.txt`,
`reserve_reviewer_info.csv`, `reserve_reviewer_unresolved.csv`, and
experimental fingerprint caches such as `fingerprints-title-only.json`. The
enrichment caches are rebuildable but expensive because live DBLP retrieval
is rate-limited.

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
