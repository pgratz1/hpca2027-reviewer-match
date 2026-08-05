# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

The **HPCA 2027 reviewer–paper matching pipeline** — fully implemented and in
active use by the PC chair during the submission window (paper registration
deadline: July 25, 2026, so `data/inputs/hpca2027-data.json` is a moving snapshot).
It classifies PC members by seniority from DBLP history and assigns reviewers
to papers by SPECTER2 embedding similarity under COI, area, load, and
seniority constraints.

**Read `README.md` first** — it documents every script, the start-to-finish
workflow, and the data files. `docs/hpca2027-matching-brief.md` is the original
design brief, kept for history; where they disagree, the README and code win.
`/home/pgratz/reviewer_match` is a symlink to this directory (the real path
contains spaces and parentheses — always quote it in shell commands).

## Running things

- Python env: `~/envs/hpca-matching/bin/python3` (torch+CUDA, transformers,
  adapters, numpy). `make` defaults to it; `requirements.txt` is
  documentation, not a reproducible installer.
- `make` runs the whole pipeline (classify → abstract enrichment →
  fingerprints → submitted-only `outputs/assignments/assignment.txt`)
  and rebuilds only what's stale. Prefer it over invoking scripts by hand.
- `make area-chairs` is deliberately separate: it requires `outputs/assignments/assignment.txt`,
  builds 10-year chair fingerprints, and writes a chair-grouped
  `outputs/assignments/area_chair_assignment.txt` under hard COIs and the closest feasible loads.
- **`make pc-roster` cross-checks both rosters against `data/inputs/hpca2027-pcinfo.csv`**,
  the HotCRP user export, and should be run after every fresh export. Offline,
  instant, read-only. It writes `outputs/reports/pc_roster_pruned.csv` (roster rows the loaders
  now drop, because HotCRP no longer marks them `pc`) and
  `outputs/reports/pc_roster_missing.csv` (PC accounts no roster explains). The membership check
  itself is **on by default in every loader** — `make PC_CHECK=--no-pc-check`,
  or `--no-pc-check` on `scripts/classify_reviewers.py`/`scripts/assign_reviewers.py`, turns it
  off for when the export is staler than the rosters. A **missing export, or a
  truncated one in which nothing is marked `pc`, is a hard error**, not a
  silent skip: the alternative is pruning all ~460 reviewers and reporting every
  paper unstaffed. `make duplicates` is the companion remediation tool.
- `make reserve-need` is also separate and independent of the assignment: it
  sizes the review-slot shortfall, i.e. how many reserve reviewers have to be
  recruited. No GPU, no fingerprints, no network — pure arithmetic. Recruiting
  the reserves themselves happens outside this repo.
- `make reserve-info` joins the HotCRP reserve upload
  (`data/inputs/reserve_reviewer_upload.csv`) to the DBLP links in
  `data/inputs/reserve_reviewers_vetting_final.xlsx` and writes `outputs/reports/reserve_reviewer_info.csv`
  — the reserve-side identity layer. Also independent of the assignment. Rows
  that fail a check are held back in `outputs/reports/reserve_reviewer_unresolved.csv` rather
  than fingerprinting the wrong person. **Always run it as
  `make reserve-info VERIFY=--verify`**: offline, a PID is only name-checked
  when the PID itself spells a name, and verifying the rest against DBLP caught
  40 links naming the wrong person out of 243 (roster 225 → 186). A plain run
  overwrites that verified roster with the weaker offline result. The profile
  cache is warm, so a verified re-run costs 0 fetches.
- `make dblp-snapshot` is the **preferred source of publications**: it reads a
  local DBLP dump (`DBLP_SNAPSHOT`, default `data/inputs/dblp-2026-07-01.xml`, 5.2 GB,
  gitignored) and writes `data/cache/dblp_snapshot_cache.json` for every roster PID —
  offline, ~4 min, no rate limit. Run it before `make`, `make area-chairs` or
  `make reserves` and those need no network for DBLP at all. The dump has no
  `pid` attributes, so `scripts/build_dblp_snapshot_cache.py` joins PID → name strings
  (from `<www key="homepages/PID">` records, aliases included) → publications by
  exact name match; DBLP guarantees a name string identifies one person. The
  cache is consulted **only for PIDs the existing caches lack**
  (`dblp.snapshot_gaps`) — re-sourcing a cached person would change their
  publication list, which is part of the fingerprint key, and re-embed them for
  nothing. PIDs absent from the snapshot (anyone added after it was taken) are
  reported and still use the live path. The same two passes also write
  `data/cache/dblp_coauthors.json` (`{pid: {name: [[year, title]]}}`, 10-year
  build window so the 5-year COI check can narrow without re-reading 5 GB) and
  `data/cache/dblp_author_names.json` (`{pid: [spelling]}` for the PIDs
  submission authors declare, 2,569 of 2,583 resolved). Both are separate files
  for the reason `dblp_affiliations.json` is: **nothing may be added to the
  publication record dict**, which is a contract with the live fetch and with
  the fingerprint key — `tests/test_regressions.py` asserts its exact 5 keys.
- `make affiliation-countries` enumerates every affiliation string across the
  submissions and all three rosters and resolves which country each institution
  is in, writing `data/curated/affiliation_countries.csv` with a **blank `country` column as
  the to-do list** (the `data/curated/dblp_overrides.csv` idiom — the generator writes only
  `suggested`, never `country`, or the hand layer would outrank DBLP with a
  machine guess). Feeds the same-country cap; run `make dblp-snapshot` first,
  since its `data/cache/dblp_affiliations.json` is the strongest layer. `--validate
  CN=china_faculty.csv '!CN=nonchina_faculty.csv'` checks it against the old
  hand split — which turns out to be exactly the `.cn` email test (100%/0%), so
  it misfiles Chinese-institution reviewers who use gmail/qq addresses.
- `make reserves` puts the reserve roster through the same three stages as the
  PC (enrich → fingerprints → classify) via `--role reserve`, writing
  `data/cache/reserve_fingerprints.json` and `outputs/reports/reserve_seniority.csv`. Reserves never filled
  in a form, so `src/reviewer_match/reserve_reviewers.py` derives their areas from the HotCRP
  topics of the submissions they authored — without areas the area gate matches
  them to nothing. `src/reviewer_match/roster.py` maps role → loader for all three scripts.
  Both artifacts are now **prerequisites of the main assignment**, which runs
  `--include-reserves --reserve-cap $(RESERVE_CAP)` (default 6) — so `make
  reserves` has to have run first, and a missing one is a named error pointing
  at it rather than make's "No rule to make target". `make RESERVE_CAP=off`
  assigns from the PC alone — but **still needs those artifacts**, because the
  ex-reserves now on the PC are assigned from them either way (§3d).
- **COI is a floor, not a full picture.** `paper_matching.own_paper_conflicts`
  treats authors, contacts, and `reserve_reviewer` nominees as conflicted, because
  authors have not yet been asked to declare conflicts against recently promoted
  PC and reserve reviewers. Coverage is uneven (reserves: median 1 declared
  conflict per paper vs 7–8 for the PC), which `scripts/assign_reviewers.py` now prints,
  and it makes any reserve-heavy result optimistic until the sweep runs. The
  derived co-author layer (step 3a) partly closes this — and closes it hardest
  for reserves, who account for 8,234 of the 15,729 undeclared conflicts — but
  it only sees co-authorship. Advisor/advisee, same-institution and funding ties
  stay invisible; **it does not replace the sweep**.
- `make coauthor-coi` writes `outputs/reports/coauthor_conflicts.csv`, one row
  per conflict the co-author layer finds, with a `declared` column where an
  **empty value means nobody recorded it**. Offline, ~6s, read-only. Needs
  `data/cache/dblp_coauthors.json`, so run `make dblp-snapshot` first — and note
  that a missing co-author cache is a **hard error** in `assign_reviewers.py`
  and `assign_area_chairs.py`, not a silent skip, for the same reason the PC
  check is: quietly dropping a COI layer is worse than failing. The
  `reviewers_matched` column is the false-positive handle; the summary flags
  names reaching 20+ reviewers that were almost never declared (13 today,
  covering 678 undeclared conflicts) — reported, not filtered. What is left is
  the residue homonym identity cannot reach: names whose paper-side authors
  declared no DBLP page at all, overwhelmingly ones that romanise into a small
  space. **The residual bias is directional** — papers with Chinese-affiliated
  authors lose more reviewers to name collisions than others. Getting authors to
  fill in the DBLP field shrinks it; more name matching will not.
- **`assign_area_chairs.py` now applies all three COI layers**, not just
  `pc_conflicts`. The `own_paper_conflicts` floor it had been missing changes
  nothing on today's export (every chair authorship is already declared) but
  should not depend on that. With only 15 chairs, tightening COI can make the
  load bounds infeasible where the reviewer matcher would under-fill; that
  surfaces as `ValueError`, which is the honest outcome.
- `make clear-uploads` writes the **undo half** of the two HotCRP uploads —
  `outputs/assignments/clear_assignment.csv` (one `all,clearreview,all,R1` row;
  `CLEAR_ROUND=all` widens it to every round), `clear_paper_tags.csv`
  (`all,cleartag,,track_N,`) and `clear_account_tags.csv`
  (`email,remove_tags,add_tags`, the last cell empty — **the empty column has
  to be there**: `p_profile.php`'s `save_bulk` reads a first line holding under
  two commas as a plain list of emails unless it also passes the newer
  `(?:user|email)` test, re-quotes every row into a single field and validates
  it whole as an address. That is the "Invalid email address, line 2" a
  two-column header produced on the live instance. The empty cell itself is a
  no-op: `UserStatus::parse_csv_main` skips any column whose trimmed value is
  empty). Offline, instant, read-only. Two things `make area-chairs`
  cannot do: account rows come from the **user export**, not the chair roster,
  which is what reaches a chair who left the roster still carrying `~~track_N`
  (the roster is unioned in as a second source, since an export predating the
  last upload cannot know what it wrote); and **non-canonical spellings are
  cleared too** (`track1`/`~~track1` are live on this data), while a tag merely
  containing the word — `TRC-track` — is not. Every row names an address the
  export lists, because HotCRP's bulk user importer **creates** an account for
  an unknown email. Tag spellings and the ceiling are imported from
  `assign_area_chairs.py`, never restated: a clearing file that spells a tag
  differently reports success and leaves it in place. `~~area-chairs`,
  conflicts and preferences are never touched.
- `make complete-papers` and `make area-chairs-complete` retain the former
  completeness filter in separate `*-complete.txt` artifacts.
- Library modules (imported, never run): `src/reviewer_match/reviewers.py`, `src/reviewer_match/dblp.py`,
  `src/reviewer_match/paper_matching.py`, `src/reviewer_match/fingerprint.py`, `src/reviewer_match/specter2_model.py`,
  `src/reviewer_match/reserve_reviewers.py`, `src/reviewer_match/roster.py`, `src/reviewer_match/affiliation_country.py`,
  `src/reviewer_match/pc_membership.py`, `src/reviewer_match/coauthor_coi.py`,
  `src/reviewer_match/collaborator_coi.py`. Runnable
  scripts: `scripts/audit_pc_roster.py`, `scripts/find_duplicate_accounts.py`,
  `scripts/audit_coauthor_conflicts.py`, `scripts/audit_collaborator_conflicts.py`,
  `scripts/audit_reserve_identities.py`, `scripts/classify_reviewers.py`, `scripts/build_fingerprints.py`,
  `scripts/enrich_publications.py`, `scripts/assign_reviewers.py`, `scripts/score_papers.py`,
  `scripts/nearest_neighbors.py`, `scripts/compare_abstract_rankings.py`,
  `scripts/score_abstract_evaluation.py`, `scripts/assign_area_chairs.py`,
  `scripts/estimate_reserve_need.py`, `scripts/build_reserve_reviewer_info.py`,
  `scripts/resolve_reserve_pids.py`, `scripts/build_dblp_snapshot_cache.py`,
  `scripts/build_affiliation_countries.py`, `scripts/generate_clear_uploads.py`,
  `scripts/resolve_trc_members.py`, `scripts/main.py`.

## Architecture (filter-then-rank, then constrained assignment)

1. **Identity**: `data/curated/dblp_overrides.csv` (email-keyed, hand-maintained) is the
   single identity layer for the PC. Reserve reviewers, who never filled in the
   acceptance form, get theirs from `outputs/reports/reserve_reviewer_info.csv`, built from the
   vetting workbook but overridden per-email by `data/curated/reserve_dblp_overrides.csv`
   (written by `scripts/resolve_reserve_pids.py`, finished by hand; a blank `dblp` cell
   is a to-do marker, not a decision, and never masks the workbook).
   A filled `dblp` cell wins over the form's DBLP
   column. `scripts/classify_reviewers.py` auto-appends blank stub rows for reviewers
   it can't resolve — the file doubles as the to-do list.
1b. **Membership** is a *separate* authority from identity: `data/curated/dblp_overrides.csv`
   says which DBLP page a person is, `data/inputs/hpca2027-pcinfo.csv` (the HotCRP user
   export) says whether they are still on the committee. Accepting an
   invitation is not the same as being on the PC — people are removed
   afterwards and the form cannot know. `src/reviewer_match/pc_membership.py` applies the check
   inside `load_reviewers`, `load_reserve_reviewers` and `load_area_chairs`
   (not in `roster.load_roster`, which 7 of the 11 roster consumers bypass), so
   no two scripts can disagree about the committee. Two rules make it safe:
   **prune-only, never add** — the same shape as the promote-only PCDB rule —
   and **never prune someone holding a PC-marked account under another
   address**. That second rule is the whole difficulty: matching on email alone
   would have dropped 12 sitting PC members out of 16 candidates, because
   accepting from one address and holding the HotCRP account under another is
   ordinary. Matching is exact (email → name tokens → email local part) and
   never fuzzy; a false match keeps someone already on the roster, a false miss
   silently removes a real reviewer. The export is also the only record of a
   **reserve elevated to the PC** (`~~ex-rr` plus a tier tag, §3d) — a promotion
   no form can know about either.
2. **Seniority**: `scripts/classify_reviewers.py` → `outputs/reports/reviewer_seniority.csv`
   (senior ≥0.8 papers/yr over 15y in ISCA/MICRO/HPCA/ASPLOS; junior <20
   pubs overall; out-of-area ≥20 pubs but <5 target-venue career; typical
   otherwise — all flag-tunable, checked in that order). PC-service
   overrides from `data/inputs/PCDB_with_emails.csv` (email-matched, promote-only)
   then make chairs, TopPicks PC/ERC members, and PC/ERC score ≥6
   senior, and juniors with score ≥2 typical.
3. **Affinity**: SPECTER2 fingerprints for reviewers (recent DBLP
   publications, using IEEE/ACM abstracts where cached, plus declared areas)
   and papers (title+abstract+topics); cosine similarity.
   COI is a hard filter and the area gate (reviewer primary/secondary ∩
   paper topics) governs the normal phases — neither is ever blended into
   the score, but the gate is released per-paper by the relaxation ladder.
3a. **Derived co-author COI** (`--coauthor-years N`, **on by default at 5**;
   `--no-coauthor-coi` is the off switch): a reviewer who co-authored a
   publication with one of a paper's authors inside the window is conflicted
   with it, declared or not. A third layer beside `pc_conflicts` and
   `own_paper_conflicts`, and mostly redundant with them by design — 9,508 of
   the 22,044 pairs it excludes are already declared, and the other 12,536 are
   what `make coauthor-coi` itemises. Hard in every phase, like COI itself —
   including the surplus stage, which inherits it for free because
   `eligible_scores` is the only place COI is applied and no phase re-checks it. **Matching is on names and errs towards firing** at the user's
   explicit direction: a withheld reviewer costs one slot out of hundreds, a
   missed conflict costs the review. `exact` = equal token sets; `partial` =
   strict subset sharing ≥2 tokens; <2 usable tokens never matches. Window
   convention is `dblp.filter_by_years` (`current_year - years + 1`).
   **The silent failure is a missing PID**: no co-author data means every check
   passes, which is not the same as clean, so both the assignment and the audit
   count the uncovered (0 of 695 today). Measured cost vs the layer off — taken
   at 6 reviewers/paper with no surplus stage, so it is a comparison between two
   runs, not today's numbers: capacity identical (6,036 pairs, 2,448 unfilled),
   goodness unchanged at 0.965, seniority marginally better.
   **Homonym identity** (`--no-coauthor-identity` restores the blunt reading):
   `name_tokens` strips DBLP's "0012" suffix, which is what makes names
   comparable but also collapses people DBLP already told apart — the most
   collision-prone name in this data is 24 distinct identities. Where **both**
   sides are identified, `coauthor_coi.identity` compares the suffix and lets
   a different homonym through. It compares the **suffix, never the raw
   string**: one person's spellings differ freely ("José García"/"Jose Garcia",
   "David A. Wood"/"David Wood") and string inequality would read as two people
   and drop a real conflict. `exact` matches only — across a `partial` match the
   numbers are not comparable. Ceiling: only 529 of 5,162 author accounts
   resolve to a numbered homonym, and an author who declared nothing keeps the
   permissive reading (not knowing which homonym someone is is not evidence they
   are not this one). Worth 3,473 pairs removed / 0 added, confirmation rate
   38% → 43%. **Never name the flagged names in a committed file** — the repo is
   public and naming them discloses who submitted.
3c. **Area chairs are not reviewers** (`--no-area-chair-exclusion` is the off
   switch; `make AREA_CHAIR_CHECK=--no-area-chair-exclusion`): every area chair
   is removed from the pool before any phase runs. Membership is the **union of
   HotCRP's `~~area-chairs` tag and the acceptance form** — 21 and 18 today,
   union 21 — because each source has been seen to catch people the other
   misses. Applied **after the reserve merge**, not to the PC roster alone: one
   of the six excluded is a reserve, and `reviewers_by_email.update()` would put
   them straight back. Resolved through `pc_membership`'s email → name →
   local-part ladder, so a second address does not evade it. A missing
   acceptance form is a **hard error** for the same reason a missing export is.
   `Account.is_area_chair` is the single definition of the tag test — `endswith`,
   never `==`, because HotCRP writes twiddle tags. The **chair roster follows the
   tag too**: `load_area_chairs` takes a `supplement` of PC/reserve records so a
   tagged account that never returned the form still chairs, borrowing its PID
   and areas; `roster.load_roster` composes that, so enrich, fingerprint and
   assign all see one roster. Anyone tagged and on no roster is reported, never
   silently emitted.
3d. **Reserves elevated to the PC** (`~~ex-rr`): a reserve promoted onto the
   committee never filled in an acceptance form, so the promotion exists only as
   HotCRP tags — `~~ex-rr` says it happened, `pc-light`/`pc-full` says to which
   tier. `load_reserve_reviewers` stamps that tier instead of `reserve`, so they
   take `--light-cap`/`--full-cap` and count as PC in every tier report. 5 today,
   all light. Three rules: **the tier tag is read only for an ex-reserve** — ~474
   accounts carry it, and for anyone who returned the form the form's `PC
   membership` column stays the authority (6 disagree today, deliberately left
   alone); **an unsettled tier promotes nobody** — neither tag or both leaves
   them a reserve at the reserve cap, named on stderr and by `make pc-roster`,
   because a withheld promotion costs one paper while an invented one hands
   somebody a 15-paper load they never agreed to; and **they stay on the reserve
   roster**, which is still the only source of their DBLP page and derived areas,
   so `make reserves` keeps building their fingerprint and seniority row and
   their class comes from the same `classify()` and PCDB overrides as everyone
   else's. `tier` is **not** in the fingerprint key, so promoting re-embeds
   nobody. Because they are PC members they are in the pool **regardless of
   `--include-reserves`**, which gates the bench and not them —
   `assign_reviewers.split_promoted_reserves` does the split, and a promoted
   person who later also returns the form is deduped on `hotcrp_email` with the
   **form record winning** (its areas are declared, not inferred). Tag spellings
   live once in `pc_membership` (`EX_RESERVE_TAG`, `TIER_TAGS`, `tag_names`);
   matching is on the **normalised exact name**, not `endswith` as `~~area-chairs`
   uses, because `pc-full` and `pc-light` are two answers to one question.
   Under `--no-pc-check` there are no tags, so nobody is promoted and the run
   warns.
3b. **Same-country cap** (`--same-country-cap N`, script default 2, **`make`
   default 1** via `SAME_COUNTRY_CAP` — see §4a for why): a
   paper whose authors are mostly from country C holds at most N reviewers
   affiliated in C. One rule for every country — **no country is named in the
   policy**; the capped set is whatever the submissions contain (31 today).
   Affiliation country, **never nationality**; HK/MO/TW/SG are separate ISO
   codes and are never folded into CN. `--same-country-cap 0` admits no
   same-country reviewer (a real setting this roster satisfies);
   `--no-same-country-cap` is the off switch. Hard in every phase including F3
   and the surplus stage, so a paper under-fills rather than exceed it. Because a country class
   *crosses* the seniority classes, greedy choice stops being substitutable and
   stability is no longer guaranteed for capped papers — `Done.` splits the
   blocking-pair count, and the hard invariant that replaces it is
   `country_over == 0`. **Coverage is the thing to read**: an unplaced reviewer
   can never consume a cap, so uneven coverage exempts whoever the resolver
   places worst. Before `data/curated/affiliation_countries.csv` was filled, `.edu`/`.com`
   were generic while `.cn`/`.kr` were not, and US did not appear once among
   the majority countries.
4. **Assignment**: `scripts/assign_reviewers.py` — phased paper-proposing deferred
   acceptance over the PC **and, under `--include-reserves`, the reserve roster**
   (the pool `make` now assigns from: 235 full + 221 light + 202 reserve, where
   5 of the light are ex-reserves promoted under §3d and are in the pool with or
   without `--include-reserves`),
   aiming for a full slate of `--reviewers-per-paper`
   (`DEFAULT_REVIEWERS_PER_PAPER`, **5** since submissions closed) plus ≥1
   senior, ≤`--max-juniors` junior (script default 1, **`make` default 2** via
   `MAX_JUNIORS` — §4a), and ≤3 out-of-area per paper. Papers that can't fill
   release constraints in order: area gate → junior/out-of-area caps
   (almost-nots only) → senior requirement (almost-senior), every pool still
   ranked by fingerprint similarity. Prints a criteria report, per-paper match
   goodness (mean assigned-reviewer similarity, worst-first summary), a
   relaxation & exclusion report, and self-checks (over-cap, blocking pairs,
   junior/out-of-area policy, slate ceiling) that must all be 0.
4b. **The `make` policy pair: `SAME_COUNTRY_CAP=1`, `MAX_JUNIORS=2`.** Chosen
   together off a 7-cell sweep of (cap 3/2/1/off) × (juniors 1/2) run at 99.1%
   affiliation coverage, and they beat the former 2/1 on every quality measure:
   mean goodness 0.9650 vs 0.9647, worst-50 tail 0.9363 vs 0.9345, 6,283 pairs
   placed vs 6,172, 47 papers needing a relaxed constraint vs 60 — while papers
   that trade away a better-matched same-country reviewer go 400 → 631. Two
   things the sweep settled: **the country cap is nearly free** (off → 1 costs
   0.0006 of a mean against a 0.011 std, and never leaves a paper short), and
   **the junior lever is worth ~10× the country lever**, so loosening juniors
   more than pays for tightening the cap. The cost is real but is the junior
   half: **409 of 1,157 papers carry two juniors.** These are *operational*
   defaults in the Makefile, the same shape as `PAPER_POLICY` — the scripts' own
   defaults stay 2 and 1, so a bare `python -m scripts.assign_reviewers` is
   unchanged. **At cap 1 the split blocking-pair count is not 0** (1 today among
   capped papers); that is the documented consequence of a country class
   crossing the seniority classes, and `country_over == 0` is the invariant that
   replaces it — do not read it as a regression.
4a. **Surplus distribution** (`--surplus-per-paper N`, **on by default at 1**;
   `0` is the off switch): `--reviewers-per-paper` is what every paper is
   *guaranteed*, and whatever capacity the six phases leave unspent then goes,
   one reviewer at a time, to the papers with the lowest match goodness. Rounds
   re-rank by current goodness, offer the worst `spare` papers one slot each,
   and take the gated pool then the area-released pool — F1 then F2 on a
   one-slot target. Only papers that **reached** the base target are eligible; a
   paper that places nothing is dropped so its slot flows to the next-worst one,
   which is why a zero-placement round is productive and is deliberately *not* a
   stop condition. COI, the same-country cap and the junior/out-of-area caps all
   still bind; the F3 almost-not pools are **not** reused. **Purely additive** —
   earlier phases are frozen — so `paper_target` stays the base target and no
   report or exit code is affected: `--surplus-per-paper 0` reproduces the
   pre-surplus output line for line, which is the regression check worth
   keeping. Two things to expect in the output: goodness is a **mean**, so a
   boosted paper's full-slate figure *drops* (the report prints base and full
   side by side for exactly this reason); and the worst-matched papers are often
   the ones that *cannot* take a surplus slot, because thin eligibility is what
   made them worst-matched — on the 1192-paper submitted set, only 52 of the
   worst 200 could be helped, while 427 of 530 spare slots were placed overall.

**Policy:** every paper-side tool defaults to `--paper-policy registered`:
every non-withdrawn record with ≥1 author, a title of ≥1 word that isn't just
"test", and an abstract of more than one sentence (topics not required).
HotCRP `status` stays `draft` until the submission deadline, so content, not
status, distinguishes a real registration from a placeholder. `--paper-policy
submitted` selects exactly `status == "submitted"` regardless of any other
field; only under that policy must reviewer assignments be full and area-chair
assignments cover the same paper set. **Submissions are in (1,192 papers), so
the Makefile's `PAPER_POLICY` now defaults to `submitted`** and feeds every
paper-side target — the scripts' own default stays `registered`, and `make
PAPER_POLICY=registered` restores the pre-deadline view. `--paper-policy
complete` retains the former pre-registration completeness checks in
`*-complete.txt` artifacts. The registered set (~1400 papers) far exceeds total
PC review capacity, so large shortage/relaxation reports are expected under
that policy, not a bug.

## Data, caches, and PII

- The acceptance CSV filename contains an apostrophe and spaces — quote it.
  Parse it (and any CSV here) with Python's `csv` module, never
  `awk`/`cut`/line counting: fields contain embedded commas, quoted `""`
  escapes, and embedded newlines. `reviewers.load_reviewers` collapses
  duplicate form submissions to the latest per email and applies overrides.
- **Never commit data.** Everything derived from real PC members (both form
  CSVs, all caches, `data/curated/dblp_overrides.csv`, `data/curated/publication_exclusions.csv`,
  classifications, assignments, and retired report CSVs) is gitignored; only
  code and docs go to the GitHub remote. Check `git status` before committing.
- Caches are incremental, versioned, and **content/policy-aware**: paper
  fingerprints include title/abstract/topics, model, and area weight;
  reviewer fingerprints include metadata, PID, selected publications and
  abstracts, model, and embedding flags. Transient DBLP/API failures remain
  retryable. The DBLP caches (`data/cache/dblp_cache.json`, `data/cache/dblp_venue_cache.json`,
  `data/cache/reviewer_publications.json`, read-only `data/inputs/dblp_pubs_cache.json`) are expensive to refill — live DBLP fetches are
  rate-limited (~3s jittered delay; on a 429 or a dropped connection,
  `dblp.get_with_retry` backs off from ≥15s, doubling with jitter, capped at
  `MAX_BACKOFF`, honouring `Retry-After` up to the same cap). The backoff index
  is per-call, so it never ratchets up across a run. **DBLP resets the
  connection outright once it has blocked an IP** — retrying cannot clear that,
  only waiting can — so the batch loops stop after `MAX_CONSECUTIVE_FAILURES`
  and keep what they cached instead of grinding for hours and leaving reviewers
  looking publication-less. If a run dies that way, wait and re-run; the caches
  resume. Prefer a larger `--delay` for big first-time fetches. Never delete them;
  `make clean-fingerprints` deliberately spares them. `data/cache/dblp_profile_cache.json`
  and `data/cache/dblp_author_search_cache.json`, filled by `scripts/resolve_trc_members.py`, are
  the same kind of cache.
- `.env` may hold an optional `S2_API_KEY`; it and editor backup
  variants are ignored. Never print, inspect, or commit secret values.
- `data/curated/publication_exclusions.csv` is the hand-maintained, per-email DOI exclusion
  layer for reviewer and area-chair fingerprints; exclusions never delete
  entries from the shared publication or abstract caches.
- `data/curated/affiliation_countries.csv` is the hand-maintained affiliation → ISO country
  layer for the region cap, and `data/cache/dblp_affiliations.json` the DBLP notes under
  it; both are gitignored, both derive from real identities.
- `data/inputs/hpca2027-pcinfo.csv` is the HotCRP **user** export — names, emails, ORCIDs,
  affiliations and declared collaborators, so among the most sensitive files
  here — and, like `data/inputs/hpca2027-data.json`, a moving snapshot. It and the three
  reports derived from it (`outputs/reports/pc_roster_pruned.csv`, `outputs/reports/pc_roster_missing.csv`,
  `outputs/reports/duplicate_accounts.csv`) are gitignored. A stale export is the failure mode
  those reports exist to surface; note it can easily be *older* than the
  acceptance form, since form responses keep arriving.
- A title-only comparison cache must use a distinct path such as
  `data/cache/fingerprints-title-only.json`; all `fingerprints-*.json` files are ignored.

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
  caches or `data/curated/dblp_overrides.csv` in a test.
- Run `~/envs/hpca-matching/bin/python3 -m unittest tests.test_regressions`
  after changes, then run `make` for an end-to-end and self-check validation.
