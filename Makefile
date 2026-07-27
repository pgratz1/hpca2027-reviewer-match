# HPCA 2027 reviewer-paper matching pipeline.
#
#   make                  rebuild whatever is stale; final output: assignment.txt
#   make reserve-need     size the reserve-reviewer shortfall
#   make complete-papers   retain the pre-registration completeness filter
#   make clean             remove assignment outputs
#   make clean-fingerprints  force a full re-embed (e.g. after changing
#                            --years / --area-weight policy); never touches
#                            the rate-limited DBLP caches
#
# Override the interpreter with `make PYTHON=python3` if not using the venv.
# assignment.txt and area_chair_assignment.txt cover the registered papers;
# once HotCRP has real submissions, switch with
# `make PAPER_POLICY=submitted` (and the same for `make area-chairs`).

PYTHON ?= $(HOME)/envs/hpca-matching/bin/python3
PAPER_POLICY ?= registered

# Optional local secrets. This file is gitignored; variables must be exported
# so enrich_publications.py can read them from its environment.
-include .env
export S2_API_KEY

CSV = HPCA'27 PC Member Acceptance Form (Responses) - Form Responses 1.csv
AREA_CHAIR_CSV = Area Chair Acceptance Form (Responses) - Form Responses 1.csv
AREA_CHAIR_YEARS = 10
# make splits prerequisite lists on spaces, so dependencies use this
# backslash-escaped copy; recipes use the plain "$(CSV)" in shell quotes.
CSV_DEP = HPCA'27\ PC\ Member\ Acceptance\ Form\ (Responses)\ -\ Form\ Responses\ 1.csv

REVIEWER_LIBS = reviewers.py dblp.py
EMBED_LIBS = fingerprint.py specter2_model.py

.DELETE_ON_ERROR:
.PHONY: all enrich area-chairs reserve-need complete-papers area-chairs-complete clean clean-fingerprints

all: reviewer_seniority.csv enrich fingerprints.json
	$(PYTHON) build_fingerprints.py --csv "$(CSV)" --fingerprint-cache fingerprints.json
	$(MAKE) assignment.txt

enrich: enrich_publications.py dblp.py reviewers.py $(CSV_DEP) dblp_overrides.csv
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

complete-papers: assignment-complete.txt

area-chairs-complete: assignment-complete.txt
	$(PYTHON) enrich_publications.py --role area-chair --csv "$(AREA_CHAIR_CSV)" \
		--years $(AREA_CHAIR_YEARS)
	$(PYTHON) build_fingerprints.py --role area-chair --csv "$(AREA_CHAIR_CSV)" \
		--fingerprint-cache area_chair_fingerprints.json --years $(AREA_CHAIR_YEARS)
	$(PYTHON) assign_area_chairs.py --paper-policy complete \
		--reviewer-assignment assignment-complete.txt --csv "$(AREA_CHAIR_CSV)" \
		> area_chair_assignment-complete.txt

reviewer_publications.json publication_abstracts.json &: enrich_publications.py dblp.py reviewers.py $(CSV_DEP) dblp_overrides.csv
	$(PYTHON) enrich_publications.py --csv "$(CSV)"

# classify_reviewers.py may append stub rows for unknown reviewers to
# dblp_overrides.csv, leaving it newer than this target; the next make run
# reruns classify once (stub population is idempotent) and converges.
reviewer_seniority.csv: classify_reviewers.py $(REVIEWER_LIBS) $(CSV_DEP) dblp_overrides.csv PCDB_with_emails.csv
	$(PYTHON) classify_reviewers.py --csv "$(CSV)" --out $@

# build_fingerprints.py rewrites the cache only when content/policy changed or
# a DBLP retry state changed. The all recipe also runs its cheap freshness
# check so cache content, rather than timestamps alone, decides what is stale.
fingerprints.json: reviewer_publications.json publication_abstracts.json build_fingerprints.py $(REVIEWER_LIBS) $(EMBED_LIBS) $(CSV_DEP) dblp_overrides.csv dblp_pubs_cache.json
	$(PYTHON) build_fingerprints.py --csv "$(CSV)" --fingerprint-cache $@

# Stale paper fingerprints (edited titles/abstracts/topics) are detected and
# re-encoded inside this run, so paper_fingerprints.json needs no target.
assignment.txt: assign_reviewers.py paper_matching.py classify_reviewers.py $(EMBED_LIBS) \
		fingerprints.json reviewer_seniority.csv hpca2027-data.json
	$(PYTHON) assign_reviewers.py --paper-policy $(PAPER_POLICY) --csv "$(CSV)" > $@

assignment-complete.txt: assign_reviewers.py paper_matching.py classify_reviewers.py $(EMBED_LIBS) \
		fingerprints.json reviewer_seniority.csv hpca2027-data.json
	$(PYTHON) assign_reviewers.py --paper-policy complete --csv "$(CSV)" > $@

clean:
	rm -f assignment.txt area_chair_assignment.txt assignment-complete.txt \
		area_chair_assignment-complete.txt

clean-fingerprints:
	rm -f fingerprints.json paper_fingerprints.json area_chair_fingerprints.json
