# HPCA 2027 reviewer-paper matching pipeline.
#
#   make                  rebuild stale state; final output below ASSIGNMENT_DIR
#   make reserve-need     size the reserve-reviewer shortfall
#   make reserve-info     resolve recruited reserves' DBLP identities
#   make reserve-pids     propose DBLP pages for unresolved reserves
#   make pc-roster        cross-check rosters against the HotCRP user export
#   make duplicates       list people holding two HotCRP accounts
#   make dblp-snapshot    cache publications from the local DBLP dump
#   make coauthor-coi     report conflicts DBLP implies but nobody declared
#   make affiliation-countries  resolve affiliation countries
#   make reserves         enrich, fingerprint, and classify reserves
#   make smoke            rehearse a full assignment with withdrawn papers
#   make clean            remove assignment outputs only
#   make clean-fingerprints  remove embedding caches, never DBLP caches

PYTHON ?= $(HOME)/envs/hpca-matching/bin/python3
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
export PYTHONPATH := $(ROOT)/src:$(ROOT):$(PYTHONPATH)
RUN = $(PYTHON) -m

PAPER_POLICY ?= registered
SAME_COUNTRY_CAP ?= 2
REGION_FLAG = $(if $(filter off,$(SAME_COUNTRY_CAP)),--no-same-country-cap,\
                   --same-country-cap $(SAME_COUNTRY_CAP))

# Optional local secrets. Variables are exported for enrichment commands.
-include .env
export S2_API_KEY

INPUT_DIR = data/inputs
CURATED_DIR = data/curated
CACHE_DIR = data/cache
REPORT_DIR = outputs/reports
ASSIGNMENT_DIR = outputs/assignments

CSV = $(INPUT_DIR)/HPCA'27 PC Member Acceptance Form (Responses) - Form Responses 1.csv
CSV_DEP = data/inputs/HPCA'27\ PC\ Member\ Acceptance\ Form\ (Responses)\ -\ Form\ Responses\ 1.csv
AREA_CHAIR_CSV = $(INPUT_DIR)/Area Chair Acceptance Form (Responses) - Form Responses 1.csv
DATA = $(INPUT_DIR)/hpca2027-data.json
PCINFO = $(INPUT_DIR)/hpca2027-pcinfo.csv
PCDB = $(INPUT_DIR)/PCDB_with_emails.csv
DBLP_PUBS = $(INPUT_DIR)/dblp_pubs_cache.json
DBLP_SNAPSHOT = $(INPUT_DIR)/dblp-2026-07-01.xml

OVERRIDES = $(CURATED_DIR)/dblp_overrides.csv
RESERVE_OVERRIDES = $(CURATED_DIR)/reserve_dblp_overrides.csv
COUNTRIES = $(CURATED_DIR)/affiliation_countries.csv

COAUTHORS = $(CACHE_DIR)/dblp_coauthors.json
AUTHOR_NAMES = $(CACHE_DIR)/dblp_author_names.json

FINGERPRINTS = $(CACHE_DIR)/fingerprints.json
PAPER_FINGERPRINTS = $(CACHE_DIR)/paper_fingerprints.json
AREA_CHAIR_FINGERPRINTS = $(CACHE_DIR)/area_chair_fingerprints.json
RESERVE_FINGERPRINTS = $(CACHE_DIR)/reserve_fingerprints.json
PUBLICATIONS = $(CACHE_DIR)/reviewer_publications.json
ABSTRACTS = $(CACHE_DIR)/publication_abstracts.json

SENIORITY = $(REPORT_DIR)/reviewer_seniority.csv
RESERVE_SENIORITY = $(REPORT_DIR)/reserve_seniority.csv
RESERVE_INFO = $(REPORT_DIR)/reserve_reviewer_info.csv
ASSIGNMENT = $(ASSIGNMENT_DIR)/assignment.txt
COMPLETE_ASSIGNMENT = $(ASSIGNMENT_DIR)/assignment-complete.txt
AREA_CHAIR_ASSIGNMENT = $(ASSIGNMENT_DIR)/area_chair_assignment.txt
AREA_CHAIR_COMPLETE = $(ASSIGNMENT_DIR)/area_chair_assignment-complete.txt

PC_CHECK ?=
AREA_CHAIR_YEARS = 10

# The derived co-author COI is on by default, like the same-country cap. Set
# COAUTHOR_COI=--no-coauthor-coi to assign without it.
COAUTHOR_COI ?=

REVIEWER_LIBS = src/reviewer_match/reviewers.py src/reviewer_match/dblp.py \
	src/reviewer_match/pc_membership.py src/reviewer_match/paths.py
EMBED_LIBS = src/reviewer_match/fingerprint.py src/reviewer_match/specter2_model.py

.DELETE_ON_ERROR:
.PHONY: all enrich area-chairs reserve-need reserve-info reserve-pids reserves \
	dblp-snapshot coauthor-coi affiliation-countries pc-roster duplicates smoke \
	complete-papers area-chairs-complete clean clean-fingerprints

all: $(SENIORITY) enrich $(FINGERPRINTS)
	$(RUN) scripts.build_fingerprints --csv "$(CSV)" --fingerprint-cache $(FINGERPRINTS)
	$(MAKE) $(ASSIGNMENT)

enrich: scripts/enrich_publications.py $(REVIEWER_LIBS) $(CSV_DEP) $(OVERRIDES) $(PCINFO)
	$(RUN) scripts.enrich_publications --csv "$(CSV)"

area-chairs:
	@test -f $(ASSIGNMENT) || { echo "ERROR: $(ASSIGNMENT) not found; run make first" >&2; exit 1; }
	$(RUN) scripts.enrich_publications --role area-chair --csv "$(AREA_CHAIR_CSV)" \
		--years $(AREA_CHAIR_YEARS)
	$(RUN) scripts.build_fingerprints --role area-chair --csv "$(AREA_CHAIR_CSV)" \
		--fingerprint-cache $(AREA_CHAIR_FINGERPRINTS) --years $(AREA_CHAIR_YEARS)
	$(RUN) scripts.assign_area_chairs --paper-policy $(PAPER_POLICY) \
		--csv "$(AREA_CHAIR_CSV)" $(COAUTHOR_COI) > $(AREA_CHAIR_ASSIGNMENT)

reserve-need:
	$(RUN) scripts.estimate_reserve_need --paper-policy $(PAPER_POLICY) --csv "$(CSV)"

reserve-info:
	$(RUN) scripts.build_reserve_reviewer_info $(VERIFY)

reserve-pids:
	$(RUN) scripts.resolve_reserve_pids

pc-roster:
	@test -f $(PCINFO) || { echo "ERROR: $(PCINFO) not found; download it from HotCRP" >&2; exit 1; }
	$(RUN) scripts.audit_pc_roster --pcinfo $(PCINFO) --csv "$(CSV)" \
		--area-chair-csv "$(AREA_CHAIR_CSV)" --reserve-info $(RESERVE_INFO) --data $(DATA)

duplicates:
	@test -f $(PCINFO) || { echo "ERROR: $(PCINFO) not found; download it from HotCRP" >&2; exit 1; }
	$(RUN) scripts.find_duplicate_accounts --pcinfo $(PCINFO) --both-pc

dblp-snapshot:
	@test -f $(DBLP_SNAPSHOT) || { echo "ERROR: $(DBLP_SNAPSHOT) not found; set DBLP_SNAPSHOT=<dump.xml>" >&2; exit 1; }
	$(RUN) scripts.build_dblp_snapshot_cache --snapshot $(DBLP_SNAPSHOT) --data $(DATA)

coauthor-coi:
	@test -f $(COAUTHORS) || { echo "ERROR: $(COAUTHORS) not found; run make dblp-snapshot first" >&2; exit 1; }
	$(RUN) scripts.audit_coauthor_conflicts --paper-policy $(PAPER_POLICY) --data $(DATA)

reserves:
	@test -f $(RESERVE_INFO) || { echo "ERROR: $(RESERVE_INFO) not found; run make reserve-info first" >&2; exit 1; }
	@test -f $(PCINFO) || { echo "ERROR: $(PCINFO) not found; download it from HotCRP, or pass PC_CHECK=--no-pc-check" >&2; exit 1; }
	$(RUN) scripts.enrich_publications --role reserve --csv $(RESERVE_INFO) --data $(DATA)
	$(RUN) scripts.build_fingerprints --role reserve --csv $(RESERVE_INFO) --data $(DATA) \
		--fingerprint-cache $(RESERVE_FINGERPRINTS)
	$(RUN) scripts.classify_reviewers --role reserve --csv $(RESERVE_INFO) --data $(DATA) \
		$(PC_CHECK) --out $(RESERVE_SENIORITY)

SMOKE_WITHDRAWN = 0.30
SMOKE_RESERVE_CAP = 6
SMOKE_DATA = $(CACHE_DIR)/smoke/hpca2027-data-smoke-$(SMOKE_WITHDRAWN).json
SMOKE_OUT = $(ASSIGNMENT_DIR)/assignment-smoke-$(SMOKE_WITHDRAWN).txt

$(SMOKE_DATA): scripts/make_smoke_dataset.py src/reviewer_match/paper_matching.py $(DATA)
	$(RUN) scripts.make_smoke_dataset --data $(DATA) --out $@ --fraction $(SMOKE_WITHDRAWN)

affiliation-countries: scripts/build_affiliation_countries.py src/reviewer_match/affiliation_country.py
	$(RUN) scripts.build_affiliation_countries --data $(DATA)

smoke: $(SMOKE_DATA)
	@test -f $(RESERVE_FINGERPRINTS) || { echo "ERROR: $(RESERVE_FINGERPRINTS) not found; run make reserves first" >&2; exit 1; }
	@test -f $(PCINFO) || { echo "ERROR: $(PCINFO) not found; download it from HotCRP, or pass PC_CHECK=--no-pc-check" >&2; exit 1; }
	$(RUN) scripts.assign_reviewers --data $(SMOKE_DATA) --csv "$(CSV)" \
		--include-reserves --reserve-cap $(SMOKE_RESERVE_CAP) \
		$(PC_CHECK) $(REGION_FLAG) $(COAUTHOR_COI) > $(SMOKE_OUT)
	@echo "wrote $(SMOKE_OUT)" >&2

complete-papers: $(COMPLETE_ASSIGNMENT)

area-chairs-complete: $(COMPLETE_ASSIGNMENT)
	$(RUN) scripts.enrich_publications --role area-chair --csv "$(AREA_CHAIR_CSV)" \
		--years $(AREA_CHAIR_YEARS)
	$(RUN) scripts.build_fingerprints --role area-chair --csv "$(AREA_CHAIR_CSV)" \
		--fingerprint-cache $(AREA_CHAIR_FINGERPRINTS) --years $(AREA_CHAIR_YEARS)
	$(RUN) scripts.assign_area_chairs --paper-policy complete \
		--reviewer-assignment $(COMPLETE_ASSIGNMENT) --csv "$(AREA_CHAIR_CSV)" \
		$(COAUTHOR_COI) > $(AREA_CHAIR_COMPLETE)

$(PUBLICATIONS) $(ABSTRACTS) &: scripts/enrich_publications.py $(REVIEWER_LIBS) $(CSV_DEP) $(OVERRIDES) $(PCINFO)
	$(RUN) scripts.enrich_publications --csv "$(CSV)"

# Only ever reached when the file is missing: with no prerequisites, make treats
# an existing one as up to date and never runs this. Without it, a fresh
# checkout fails with make's "No rule to make target", which names neither the
# cause nor the fix.
$(COAUTHORS):
	@echo "ERROR: $@ not found; run make dblp-snapshot (needs the DBLP dump), or" >&2
	@echo "       assign without the co-author COI: make COAUTHOR_COI=--no-coauthor-coi" >&2
	@exit 1

$(SENIORITY): scripts/classify_reviewers.py $(REVIEWER_LIBS) $(CSV_DEP) $(OVERRIDES) $(PCDB) $(PCINFO)
	$(RUN) scripts.classify_reviewers --csv "$(CSV)" $(PC_CHECK) --out $@

$(FINGERPRINTS): $(PUBLICATIONS) $(ABSTRACTS) scripts/build_fingerprints.py $(REVIEWER_LIBS) $(EMBED_LIBS) $(CSV_DEP) $(OVERRIDES) $(DBLP_PUBS) $(PCINFO)
	$(RUN) scripts.build_fingerprints --csv "$(CSV)" --fingerprint-cache $@

$(ASSIGNMENT): scripts/assign_reviewers.py src/reviewer_match/paper_matching.py \
	scripts/classify_reviewers.py src/reviewer_match/affiliation_country.py \
	src/reviewer_match/coauthor_coi.py \
	$(EMBED_LIBS) $(FINGERPRINTS) $(SENIORITY) $(DATA) $(COAUTHORS)
	$(RUN) scripts.assign_reviewers --paper-policy $(PAPER_POLICY) --csv "$(CSV)" \
		$(PC_CHECK) $(REGION_FLAG) $(COAUTHOR_COI) > $@

$(COMPLETE_ASSIGNMENT): scripts/assign_reviewers.py src/reviewer_match/paper_matching.py \
	scripts/classify_reviewers.py src/reviewer_match/affiliation_country.py \
	src/reviewer_match/coauthor_coi.py \
	$(EMBED_LIBS) $(FINGERPRINTS) $(SENIORITY) $(DATA) $(COAUTHORS)
	$(RUN) scripts.assign_reviewers --paper-policy complete --csv "$(CSV)" \
		$(PC_CHECK) $(REGION_FLAG) $(COAUTHOR_COI) > $@

clean:
	rm -f $(ASSIGNMENT) $(AREA_CHAIR_ASSIGNMENT) $(COMPLETE_ASSIGNMENT) $(AREA_CHAIR_COMPLETE)

clean-fingerprints:
	rm -f $(FINGERPRINTS) $(PAPER_FINGERPRINTS) $(AREA_CHAIR_FINGERPRINTS)
