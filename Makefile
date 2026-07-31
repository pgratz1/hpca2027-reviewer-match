# HPCA 2027 reviewer-paper matching pipeline.
#
#   make                  rebuild whatever is stale; final output: assignment.txt
#   make reserve-need     size the reserve-reviewer shortfall
#   make reserve-info     resolve the recruited reserves' DBLP identities
#   make reserve-pids     propose DBLP pages for the ones it held back
#   make pc-roster        cross-check both rosters against the HotCRP export
#   make duplicates       list people holding two HotCRP accounts
#   make dblp-snapshot    cache publications from the local DBLP dump (offline)
#   make affiliation-countries  resolve which country each affiliation is in
#   make reserves         enrich + fingerprint + classify the reserve reviewers
#   make smoke            rehearse a full assignment: PC + reserves, 30%% withdrawn
#   make complete-papers   retain the pre-registration completeness filter
#   make clean             remove assignment outputs
#   make clean-fingerprints  force a full re-embed (e.g. after changing
#                            --years / --area-weight policy); never touches
#                            the rate-limited DBLP caches
#
# Override the interpreter with `make PYTHON=python3` if not using the venv.
# The same-country cap is ON by default at 2: a paper whose authors are mostly
# from one country gets at most that many reviewers from it, the same rule for
# every country. `make SAME_COUNTRY_CAP=3` loosens it, `=0` admits no
# same-country reviewer at all, and `=off` disables the policy.
#
# assignment.txt and area_chair_assignment.txt cover the registered papers;
# once HotCRP has real submissions, switch with
# `make PAPER_POLICY=submitted` (and the same for `make area-chairs`).
#
# Every roster is checked against the HotCRP user export and anyone no longer
# marked pc is dropped. `make PC_CHECK=--no-pc-check` runs unchecked, for when
# the export is staler than the rosters and pruning someone would be the worse
# error. Run `make pc-roster` after each fresh export.

PYTHON ?= $(HOME)/envs/hpca-matching/bin/python3
PAPER_POLICY ?= registered
# Max reviewers from a paper's own majority-author country. A number, or `off`.
SAME_COUNTRY_CAP ?= 2
REGION_FLAG = $(if $(filter off,$(SAME_COUNTRY_CAP)),--no-same-country-cap,\
                   --same-country-cap $(SAME_COUNTRY_CAP))

# Optional local secrets. This file is gitignored; variables must be exported
# so enrich_publications.py can read them from its environment.
-include .env
export S2_API_KEY

CSV = HPCA'27 PC Member Acceptance Form (Responses) - Form Responses 1.csv
DATA = hpca2027-data.json
RESERVE_INFO = reserve_reviewer_info.csv
# HotCRP user export: the authority on who is on the PC right now, as opposed
# to who accepted an invitation. Blank PC_CHECK means the check is on.
PCINFO = hpca2027-pcinfo.csv
PC_CHECK ?=
DBLP_SNAPSHOT = dblp-2026-07-01.xml
AREA_CHAIR_CSV = Area Chair Acceptance Form (Responses) - Form Responses 1.csv
AREA_CHAIR_YEARS = 10
# make splits prerequisite lists on spaces, so dependencies use this
# backslash-escaped copy; recipes use the plain "$(CSV)" in shell quotes.
CSV_DEP = HPCA'27\ PC\ Member\ Acceptance\ Form\ (Responses)\ -\ Form\ Responses\ 1.csv

REVIEWER_LIBS = reviewers.py dblp.py pc_membership.py
EMBED_LIBS = fingerprint.py specter2_model.py

.DELETE_ON_ERROR:
.PHONY: all enrich area-chairs reserve-need reserve-info reserve-pids reserves dblp-snapshot affiliation-countries pc-roster duplicates smoke complete-papers area-chairs-complete clean clean-fingerprints

all: reviewer_seniority.csv enrich fingerprints.json
	$(PYTHON) build_fingerprints.py --csv "$(CSV)" --fingerprint-cache fingerprints.json
	$(MAKE) assignment.txt

enrich: enrich_publications.py dblp.py reviewers.py pc_membership.py $(CSV_DEP) dblp_overrides.csv $(PCINFO)
	$(PYTHON) enrich_publications.py --csv "$(CSV)"

area-chairs:
	@test -f assignment.txt || { echo "ERROR: assignment.txt not found; run make first" >&2; exit 1; }
	$(PYTHON) enrich_publications.py --role area-chair --csv "$(AREA_CHAIR_CSV)" \
		--years $(AREA_CHAIR_YEARS)
	$(PYTHON) build_fingerprints.py --role area-chair --csv "$(AREA_CHAIR_CSV)" \
		--fingerprint-cache area_chair_fingerprints.json --years $(AREA_CHAIR_YEARS)
	$(PYTHON) assign_area_chairs.py --paper-policy $(PAPER_POLICY) \
		--csv "$(AREA_CHAIR_CSV)" > area_chair_assignment.txt

# How many reserve reviewers the review-slot shortfall needs. Independent of
# the assignment and of the fingerprint caches: pure arithmetic over the
# selected papers and the PC's caps, no network and no GPU, so it is safe to
# run at any point.
reserve-need:
	$(PYTHON) estimate_reserve_need.py --paper-policy $(PAPER_POLICY) --csv "$(CSV)"

# Join the HotCRP reserve-reviewer upload to the vetting workbook's DBLP links,
# giving the reserves the identity everything downstream is built from. Also
# separate from all: it depends on neither the papers nor the fingerprints, and
# `make reserve-info VERIFY=--verify` checks every PID against DBLP over the
# network (rate-limited, cached in dblp_profile_cache.json).
reserve-info:
	$(PYTHON) build_reserve_reviewer_info.py $(VERIFY)

# Propose DBLP pages for the reserves build_reserve_reviewer_info.py had to hold
# back, from the submissions they authored. Writes reserve_dblp_overrides.csv,
# whose blank rows are the chair's to-do list; re-run reserve-info afterwards.
reserve-pids:
	$(PYTHON) resolve_reserve_pids.py

# Cross-check both rosters against the HotCRP user export, in both directions:
# who the pipeline still counts as a reviewer but HotCRP no longer marks pc, and
# which pc accounts have neither an acceptance nor a reserve-upload row. Offline
# and instant. Run it after every fresh export, and act on both reports before
# trusting an assignment.
pc-roster:
	@test -f $(PCINFO) || { echo "ERROR: $(PCINFO) not found; download it from HotCRP (Users -> download -> user information)" >&2; exit 1; }
	$(PYTHON) audit_pc_roster.py --pcinfo $(PCINFO) --csv "$(CSV)" \
		--area-chair-csv "$(AREA_CHAIR_CSV)" --reserve-info $(RESERVE_INFO) --data $(DATA)

# People holding two HotCRP accounts, which splits their conflicts and review
# load. The remediation tool for pc-roster's alternate_account rows.
duplicates:
	@test -f $(PCINFO) || { echo "ERROR: $(PCINFO) not found; download it from HotCRP (Users -> download -> user information)" >&2; exit 1; }
	$(PYTHON) find_duplicate_accounts.py --pcinfo $(PCINFO) --both-pc

# Publications for every roster PID, read out of the local DBLP dump instead of
# asking dblp.org ~700 times (which has produced IP blocks, timeouts and 503s).
# Two streaming passes over 5.2 GB: minutes, offline, and run once. The cache it
# writes is consulted only for PIDs the existing caches lack, so no fingerprint
# already computed is invalidated.
dblp-snapshot:
	@test -f $(DBLP_SNAPSHOT) || { echo "ERROR: $(DBLP_SNAPSHOT) not found; set DBLP_SNAPSHOT=<dump.xml>" >&2; exit 1; }
	$(PYTHON) build_dblp_snapshot_cache.py --snapshot $(DBLP_SNAPSHOT)

# Put the reserve reviewers through the same three stages a PC member goes
# through: publication metadata, SPECTER2 fingerprints, seniority class. Their
# areas are derived from the topics of the submissions they authored, since they
# never filled in an acceptance form. Separate from all, like area-chairs: it
# needs the GPU and does not feed assignment.txt yet. Run dblp-snapshot first
# and it needs no network either.
reserves:
	@test -f $(RESERVE_INFO) || { echo "ERROR: $(RESERVE_INFO) not found; run make reserve-info first" >&2; exit 1; }
	@test -f $(PCINFO) || { echo "ERROR: $(PCINFO) not found; download it from HotCRP, or pass PC_CHECK=--no-pc-check" >&2; exit 1; }
	$(PYTHON) enrich_publications.py --role reserve --csv $(RESERVE_INFO) --data $(DATA)
	$(PYTHON) build_fingerprints.py --role reserve --csv $(RESERVE_INFO) --data $(DATA) \
		--fingerprint-cache reserve_fingerprints.json
	$(PYTHON) classify_reviewers.py --role reserve --csv $(RESERVE_INFO) --data $(DATA) \
		$(PC_CHECK) --out reserve_seniority.csv

# End-to-end rehearsal: every paper against the PC *and* the reserves. Uses a
# copy of the export with SMOKE_WITHDRAWN of the registered papers marked
# withdrawn, standing in for the ones that never get submitted — assigning
# against all 1,414 only ever measures the shortfall. The self-checks in the
# output (over cap, blocking pairs, junior/out-of-area) are the pass/fail; the
# shortage count is a result, not a verdict.
SMOKE_WITHDRAWN = 0.30
SMOKE_RESERVE_CAP = 6
# The withdrawal fraction is part of the filename, not just of the recipe: make
# compares timestamps, so a shared name would let `make smoke
# SMOKE_WITHDRAWN=0.25` quietly reuse the 0.30 draw and report it as a 25% run.
# Tagging also lets two fractions sit side by side for comparison.
SMOKE_DATA = hpca2027-data-smoke-$(SMOKE_WITHDRAWN).json
SMOKE_OUT = assignment-smoke-$(SMOKE_WITHDRAWN).txt

$(SMOKE_DATA): make_smoke_dataset.py paper_matching.py $(DATA)
	$(PYTHON) make_smoke_dataset.py --data $(DATA) --out $@ --fraction $(SMOKE_WITHDRAWN)

# Enumerate every affiliation across the submissions and all three rosters and
# resolve which country each institution is in where the automatic layers can.
# Idempotent: hand-entered country cells are carried over, never rewritten.
affiliation-countries: build_affiliation_countries.py affiliation_country.py
	$(PYTHON) build_affiliation_countries.py --data $(DATA)

smoke: $(SMOKE_DATA)
	@test -f reserve_fingerprints.json || { echo "ERROR: reserve_fingerprints.json not found; run make reserves first" >&2; exit 1; }
	@test -f $(PCINFO) || { echo "ERROR: $(PCINFO) not found; download it from HotCRP, or pass PC_CHECK=--no-pc-check" >&2; exit 1; }
	$(PYTHON) assign_reviewers.py --data $(SMOKE_DATA) --csv "$(CSV)" \
		--include-reserves --reserve-cap $(SMOKE_RESERVE_CAP) \
		$(PC_CHECK) $(REGION_FLAG) > $(SMOKE_OUT)
	@echo "wrote $(SMOKE_OUT)" >&2

complete-papers: assignment-complete.txt

area-chairs-complete: assignment-complete.txt
	$(PYTHON) enrich_publications.py --role area-chair --csv "$(AREA_CHAIR_CSV)" \
		--years $(AREA_CHAIR_YEARS)
	$(PYTHON) build_fingerprints.py --role area-chair --csv "$(AREA_CHAIR_CSV)" \
		--fingerprint-cache area_chair_fingerprints.json --years $(AREA_CHAIR_YEARS)
	$(PYTHON) assign_area_chairs.py --paper-policy complete \
		--reviewer-assignment assignment-complete.txt --csv "$(AREA_CHAIR_CSV)" \
		> area_chair_assignment-complete.txt

reviewer_publications.json publication_abstracts.json &: enrich_publications.py dblp.py reviewers.py pc_membership.py $(CSV_DEP) dblp_overrides.csv $(PCINFO)
	$(PYTHON) enrich_publications.py --csv "$(CSV)"

# classify_reviewers.py may append stub rows for unknown reviewers to
# dblp_overrides.csv, leaving it newer than this target; the next make run
# reruns classify once (stub population is idempotent) and converges.
reviewer_seniority.csv: classify_reviewers.py $(REVIEWER_LIBS) $(CSV_DEP) dblp_overrides.csv PCDB_with_emails.csv $(PCINFO)
	$(PYTHON) classify_reviewers.py --csv "$(CSV)" $(PC_CHECK) --out $@

# build_fingerprints.py rewrites the cache only when content/policy changed or
# a DBLP retry state changed. The all recipe also runs its cheap freshness
# check so cache content, rather than timestamps alone, decides what is stale.
fingerprints.json: reviewer_publications.json publication_abstracts.json build_fingerprints.py $(REVIEWER_LIBS) $(EMBED_LIBS) $(CSV_DEP) dblp_overrides.csv dblp_pubs_cache.json $(PCINFO)
	$(PYTHON) build_fingerprints.py --csv "$(CSV)" --fingerprint-cache $@

# Stale paper fingerprints (edited titles/abstracts/topics) are detected and
# re-encoded inside this run, so paper_fingerprints.json needs no target.
assignment.txt: assign_reviewers.py paper_matching.py classify_reviewers.py \
		affiliation_country.py $(EMBED_LIBS) \
		fingerprints.json reviewer_seniority.csv hpca2027-data.json
	$(PYTHON) assign_reviewers.py --paper-policy $(PAPER_POLICY) --csv "$(CSV)" \
		$(PC_CHECK) $(REGION_FLAG) > $@

assignment-complete.txt: assign_reviewers.py paper_matching.py classify_reviewers.py \
		affiliation_country.py $(EMBED_LIBS) \
		fingerprints.json reviewer_seniority.csv hpca2027-data.json
	$(PYTHON) assign_reviewers.py --paper-policy complete --csv "$(CSV)" \
		$(PC_CHECK) $(REGION_FLAG) > $@

clean:
	rm -f assignment.txt area_chair_assignment.txt assignment-complete.txt \
		area_chair_assignment-complete.txt

clean-fingerprints:
	rm -f fingerprints.json paper_fingerprints.json area_chair_fingerprints.json
