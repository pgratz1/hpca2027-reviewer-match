# HPCA 2027 reviewer-paper matching pipeline.
#
#   make                  rebuild stale state; final output below ASSIGNMENT_DIR
#                         (submitted papers, PC + reserves — needs make reserves)
#   make reserve-need     size the reserve-reviewer shortfall
#   make reserve-info     resolve recruited reserves' DBLP identities
#   make reserve-pids     propose DBLP pages for unresolved reserves
#   make pc-roster        cross-check rosters against the HotCRP user export
#   make duplicates       list people holding two HotCRP accounts
#   make dblp-snapshot    cache publications from the local DBLP dump
#   make coauthor-coi     report conflicts DBLP implies but nobody declared
#   make collaborator-coi report conflicts declared collaborators/affiliation imply
#   make affiliation-countries  resolve affiliation countries
#   make reserves         enrich, fingerprint, and classify reserves
#   make trc              enrich, fingerprint, and assign TRC (PhD student) reviews
#   make clear-uploads    HotCRP CSVs that wipe R1 reviews and the track tags
#   make baselines        randomized arms: how much of the match is SPECTER2?
#   make clean            remove assignment outputs only
#   make clean-fingerprints  remove embedding caches, never DBLP caches

PYTHON ?= $(HOME)/envs/hpca-matching/bin/python3
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
export PYTHONPATH := $(ROOT)/src:$(ROOT):$(PYTHONPATH)
RUN = $(PYTHON) -m

# Submissions are in, so the paper set is the submitted one and the reserve
# roster is part of the pool. PAPER_POLICY=registered goes back to the
# pre-deadline view; RESERVE_CAP=off assigns from the PC alone.
PAPER_POLICY ?= submitted
RESERVE_CAP ?= 6
RESERVE_FLAG = $(if $(filter off,$(RESERVE_CAP)),,\
                    --include-reserves --reserve-cap $(RESERVE_CAP))
# One same-country reviewer per paper, and up to two juniors. Chosen together
# off a 7-cell sweep of (cap 3/2/1/off) x (juniors 1/2) at 99.1% affiliation
# coverage: this pair beats the former cap 2 / 1 junior on every quality measure
# -- mean goodness 0.9650 vs 0.9647, worst-50 tail 0.9365 vs 0.9346, 6275 pairs
# placed vs 6164, 47 papers needing a relaxed constraint vs 60 -- while taking
# papers that trade away a better-matched same-country reviewer from 400 to 629.
# The country cap is close to free (off -> 1 costs 0.0006 of a mean, against a
# 0.011 std) and never leaves a paper short; the junior change is what actually
# buys the quality, and it costs 409 of 1157 papers a second junior reviewer.
# These are the *operational* defaults, the same way PAPER_POLICY is: the
# scripts' own defaults stay at 2 and 1 so a bare `python -m scripts...` run is
# unchanged. SAME_COUNTRY_CAP=off disables the cap; =0 admits no same-country
# reviewer at all, which is a different setting.
SAME_COUNTRY_CAP ?= 1
REGION_FLAG = $(if $(filter off,$(SAME_COUNTRY_CAP)),--no-same-country-cap,\
                   --same-country-cap $(SAME_COUNTRY_CAP))
MAX_JUNIORS ?= 2
JUNIOR_FLAG = --max-juniors $(MAX_JUNIORS)

# #1152 "Test TRC" is a standing administrative test submission (used for
# exercising the track-tag setup in HotCRP), not a real paper, so it should
# never get real reviewers or an area chair -- an operational default the
# same way PAPER_POLICY is; the scripts' own default stays empty. Threaded
# through every assign_reviewers/assign_area_chairs invocation so it never
# reappears in a report or a baseline measurement. EXCLUDE_PIDS= (empty)
# turns it off.
EXCLUDE_PIDS ?= 1152
EXCLUDE_FLAG = $(if $(EXCLUDE_PIDS),--exclude-pids $(EXCLUDE_PIDS),)

# make trc's own operational defaults: at most one TRC reviewer per paper
# (maximizes how many distinct papers get TRC-track coverage, over letting
# students converge on the same few standout papers), and jsanmiguel@wisc.edu
# is the sitting TRC chair -- no paper he is conflicted with ever enters the
# TRC track. TRC_PAPER_CAP=0 removes the cap.
TRC_CHAIR_EMAIL ?= jsanmiguel@wisc.edu
TRC_ROUND ?= TRC
TRC_PAPER_CAP ?= 1

# Optional local secrets. Variables are exported for enrichment commands.
-include .env
export S2_API_KEY

INPUT_DIR = data/inputs
CURATED_DIR = data/curated
CACHE_DIR = data/cache
REPORT_DIR = outputs/reports
ASSIGNMENT_DIR = outputs/assignments
# Baselines land here and NOT in ASSIGNMENT_DIR: a randomized arm is a
# measurement, and nothing that looks like an assignment should sit beside one.
EVALUATION_DIR = outputs/evaluations

CSV = $(INPUT_DIR)/HPCA'27 PC Member Acceptance Form (Responses) - Form Responses 1.csv
CSV_DEP = data/inputs/HPCA'27\ PC\ Member\ Acceptance\ Form\ (Responses)\ -\ Form\ Responses\ 1.csv
AREA_CHAIR_CSV = $(INPUT_DIR)/Area Chair Acceptance Form (Responses) - Form Responses 1.csv
AREA_CHAIR_CSV_DEP = data/inputs/Area\ Chair\ Acceptance\ Form\ (Responses)\ -\ Form\ Responses\ 1.csv
TRC_CSV = $(INPUT_DIR)/HPCA'27 TRC Member Acceptance Form (Responses) - Form Responses 1.csv
TRC_CSV_DEP = data/inputs/HPCA'27\ TRC\ Member\ Acceptance\ Form\ (Responses)\ -\ Form\ Responses\ 1.csv
DATA = $(INPUT_DIR)/hpca2027-data.json
PCINFO = $(INPUT_DIR)/hpca2027-pcinfo.csv
PCDB = $(INPUT_DIR)/PCDB_with_emails.csv
DBLP_PUBS = $(INPUT_DIR)/dblp_pubs_cache.json
DBLP_SNAPSHOT = $(INPUT_DIR)/dblp-2026-07-01.xml

OVERRIDES = $(CURATED_DIR)/dblp_overrides.csv
RESERVE_OVERRIDES = $(CURATED_DIR)/reserve_dblp_overrides.csv
TRC_OVERRIDES = $(CURATED_DIR)/trc_dblp_overrides.csv
COUNTRIES = $(CURATED_DIR)/affiliation_countries.csv
# The hand-maintained country layer feeds the same-country cap, so editing it has
# to restage the assignment -- without this, filling in a blank `country` cell
# leaves a stale result that make reports as up to date. Guarded by `wildcard`
# because the file is optional: affiliation_country.load_affiliation_countries
# returns {} when it is absent, and a hard prerequisite would instead break a
# fresh checkout with "No rule to make target".
COUNTRIES_DEP = $(wildcard $(COUNTRIES))

COAUTHORS = $(CACHE_DIR)/dblp_coauthors.json
AUTHOR_NAMES = $(CACHE_DIR)/dblp_author_names.json

FINGERPRINTS = $(CACHE_DIR)/fingerprints.json
PAPER_FINGERPRINTS = $(CACHE_DIR)/paper_fingerprints.json
AREA_CHAIR_FINGERPRINTS = $(CACHE_DIR)/area_chair_fingerprints.json
RESERVE_FINGERPRINTS = $(CACHE_DIR)/reserve_fingerprints.json
TRC_FINGERPRINTS = $(CACHE_DIR)/trc_fingerprints.json
PUBLICATIONS = $(CACHE_DIR)/reviewer_publications.json
ABSTRACTS = $(CACHE_DIR)/publication_abstracts.json

SENIORITY = $(REPORT_DIR)/reviewer_seniority.csv
RESERVE_SENIORITY = $(REPORT_DIR)/reserve_seniority.csv
RESERVE_INFO = $(REPORT_DIR)/reserve_reviewer_info.csv
ASSIGNMENT = $(ASSIGNMENT_DIR)/assignment.txt
ASSIGNMENT_CSV = $(ASSIGNMENT_DIR)/assignment.csv
TRC_ASSIGNMENT = $(ASSIGNMENT_DIR)/trc_assignment.txt
TRC_REVIEW_CSV = $(ASSIGNMENT_DIR)/trc_review_upload.csv
TRC_TAG_CSV = $(ASSIGNMENT_DIR)/trc_track_tags.csv
TRC_MISSING_TAG_CSV = $(ASSIGNMENT_DIR)/trc_missing_tags.csv
COMPLETE_ASSIGNMENT = $(ASSIGNMENT_DIR)/assignment-complete.txt
COMPLETE_ASSIGNMENT_CSV = $(ASSIGNMENT_DIR)/assignment-complete.csv
AREA_CHAIR_ASSIGNMENT = $(ASSIGNMENT_DIR)/area_chair_assignment.txt
AREA_CHAIR_COMPLETE = $(ASSIGNMENT_DIR)/area_chair_assignment-complete.txt
AREA_CHAIR_ACCOUNT_TAGS = $(ASSIGNMENT_DIR)/area_chair_account_tags.csv
AREA_CHAIR_PAPER_TAGS = $(ASSIGNMENT_DIR)/area_chair_paper_tags.csv
AREA_CHAIR_ACCOUNT_TAGS_COMPLETE = $(ASSIGNMENT_DIR)/area_chair_account_tags-complete.csv
AREA_CHAIR_PAPER_TAGS_COMPLETE = $(ASSIGNMENT_DIR)/area_chair_paper_tags-complete.csv
# The live assignment replayed out of HotCRP's action log, and the incremental
# rerun built on top of it. Kept apart from $(ASSIGNMENT_CSV): that file is
# what the pipeline last proposed, this one is what HotCRP actually holds, and
# collapsing the two would lose the diff that is the whole point.
LOG = data/inputs/hpca2027-log.csv
CURRENT_ASSIGNMENT_CSV = $(ASSIGNMENT_DIR)/current_assignment.csv
REVIEWER_ACTIVITY = $(REPORT_DIR)/reviewer_activity.csv
PINNED_REVIEWERS = $(REPORT_DIR)/reviewers_pinned.txt
RERUN_ASSIGNMENT = $(ASSIGNMENT_DIR)/assignment-rerun.txt
RERUN_ASSIGNMENT_CSV = $(ASSIGNMENT_DIR)/assignment-rerun.csv
# Which activity proxy decides who is left alone. `any` is the safer default:
# it errs towards pinning, and a pin costs match quality where a wrong release
# costs somebody the work they already started.
ACTIVITY_SIGNAL ?= any

CLEAR_ASSIGNMENT = $(ASSIGNMENT_DIR)/clear_assignment.csv
CLEAR_PAPER_TAGS = $(ASSIGNMENT_DIR)/clear_paper_tags.csv
CLEAR_ACCOUNT_TAGS = $(ASSIGNMENT_DIR)/clear_account_tags.csv

# Round the clearing upload wipes. CLEAR_ROUND=all clears every round, not
# just the one this pipeline assigns into.
CLEAR_ROUND ?= R1

PC_CHECK ?=
AREA_CHAIR_YEARS = 10

# The derived co-author COI is on by default, like the same-country cap. Set
# COAUTHOR_COI=--no-coauthor-coi to assign without it.
COAUTHOR_COI ?=

# Same for the derived declared-collaborator COI (name-matched only; the
# affiliation-overlap signal is reported, not excluded -- see
# reviewer_match.collaborator_coi). Set COLLABORATOR_COI=--no-collaborator-coi
# to assign without it.
COLLABORATOR_COI ?=

# Area chairs are kept out of the reviewer pool by default. Set
# AREA_CHAIR_CHECK=--no-area-chair-exclusion to assign papers to them anyway.
AREA_CHAIR_CHECK ?=

REVIEWER_LIBS = src/reviewer_match/reviewers.py src/reviewer_match/dblp.py \
	src/reviewer_match/pc_membership.py src/reviewer_match/paths.py
EMBED_LIBS = src/reviewer_match/fingerprint.py src/reviewer_match/specter2_model.py

# Everything an assign_reviewers run reads. Shared by the submitted assignment,
# the complete-policy one and the baseline arms, which had drifted into three
# copies of the same list.
ASSIGN_DEPS = scripts/assign_reviewers.py src/reviewer_match/paper_matching.py \
	scripts/classify_reviewers.py src/reviewer_match/affiliation_country.py \
	src/reviewer_match/coauthor_coi.py src/reviewer_match/reserve_reviewers.py \
	src/reviewer_match/area_chairs.py src/reviewer_match/pc_membership.py \
	$(EMBED_LIBS) $(FINGERPRINTS) $(SENIORITY) $(DATA) $(COAUTHORS) \
	$(AREA_CHAIR_CSV_DEP) $(PCINFO) $(COUNTRIES_DEP) \
	$(RESERVE_INFO) $(RESERVE_FINGERPRINTS) $(RESERVE_SENIORITY)

.DELETE_ON_ERROR:
.PHONY: all enrich area-chairs reserve-need reserve-info reserve-pids reserves trc \
	dblp-snapshot coauthor-coi collaborator-coi affiliation-countries pc-roster duplicates \
	complete-papers area-chairs-complete clear-uploads baselines clean clean-fingerprints \
	log-assignments reviewer-activity rerun

all: $(SENIORITY) enrich $(FINGERPRINTS)
	$(RUN) scripts.build_fingerprints --csv "$(CSV)" --fingerprint-cache $(FINGERPRINTS)
	$(MAKE) $(ASSIGNMENT) $(ASSIGNMENT_CSV)

enrich: scripts/enrich_publications.py $(REVIEWER_LIBS) $(CSV_DEP) $(OVERRIDES) $(PCINFO)
	$(RUN) scripts.enrich_publications --csv "$(CSV)"

area-chairs:
	@test -f $(ASSIGNMENT) || { echo "ERROR: $(ASSIGNMENT) not found; run make first" >&2; exit 1; }
	$(RUN) scripts.enrich_publications --role area-chair --csv "$(AREA_CHAIR_CSV)" \
		--years $(AREA_CHAIR_YEARS)
	$(RUN) scripts.build_fingerprints --role area-chair --csv "$(AREA_CHAIR_CSV)" \
		--fingerprint-cache $(AREA_CHAIR_FINGERPRINTS) --years $(AREA_CHAIR_YEARS)
	$(RUN) scripts.assign_area_chairs --paper-policy $(PAPER_POLICY) \
		--csv "$(AREA_CHAIR_CSV)" $(COAUTHOR_COI) $(COLLABORATOR_COI) $(EXCLUDE_FLAG) \
		--account-tag-csv $(AREA_CHAIR_ACCOUNT_TAGS) --paper-tag-csv $(AREA_CHAIR_PAPER_TAGS) \
		> $(AREA_CHAIR_ASSIGNMENT)

clear-uploads:
	@test -f $(PCINFO) || { echo "ERROR: $(PCINFO) not found; download it from HotCRP" >&2; exit 1; }
	$(RUN) scripts.generate_clear_uploads --pcinfo $(PCINFO) --csv "$(AREA_CHAIR_CSV)" \
		--data $(DATA) --round $(CLEAR_ROUND) \
		--assignment-out $(CLEAR_ASSIGNMENT) --paper-tag-out $(CLEAR_PAPER_TAGS) \
		--account-tag-out $(CLEAR_ACCOUNT_TAGS)

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

collaborator-coi:
	@test -f $(PCINFO) || { echo "ERROR: $(PCINFO) not found; download it from HotCRP" >&2; exit 1; }
	$(RUN) scripts.audit_collaborator_conflicts --paper-policy $(PAPER_POLICY) --data $(DATA) --pcinfo $(PCINFO)

reserves:
	@test -f $(RESERVE_INFO) || { echo "ERROR: $(RESERVE_INFO) not found; run make reserve-info first" >&2; exit 1; }
	@test -f $(PCINFO) || { echo "ERROR: $(PCINFO) not found; download it from HotCRP, or pass PC_CHECK=--no-pc-check" >&2; exit 1; }
	$(RUN) scripts.enrich_publications --role reserve --csv $(RESERVE_INFO) --data $(DATA)
	$(RUN) scripts.build_fingerprints --role reserve --csv $(RESERVE_INFO) --data $(DATA) \
		--fingerprint-cache $(RESERVE_FINGERPRINTS)
	$(RUN) scripts.classify_reviewers --role reserve --csv $(RESERVE_INFO) --data $(DATA) \
		$(PC_CHECK) --out $(RESERVE_SENIORITY)

# Independent of `make`/`make area-chairs` (no shared prerequisites beyond
# the paper export), the same way `make reserves` stands alone. No seniority
# classification step -- TRC members are matched by nearest-neighbor
# similarity, not run through the PC's junior/senior deferred-acceptance
# constraints, so scripts.classify_reviewers has nothing to contribute here.
trc:
	@test -f "$(TRC_CSV)" || { echo "ERROR: $(TRC_CSV) not found" >&2; exit 1; }
	@test -f $(ASSIGNMENT_CSV) || { echo "ERROR: $(ASSIGNMENT_CSV) not found; run make first (a TRC member must never be assigned a paper their advisor is already reviewing, which this checks against the main slate)" >&2; exit 1; }
	$(RUN) scripts.enrich_publications --role trc --csv "$(TRC_CSV)"
	$(RUN) scripts.build_fingerprints --role trc --csv "$(TRC_CSV)" \
		--fingerprint-cache $(TRC_FINGERPRINTS)
	$(RUN) scripts.assign_trc_reviews --trc-csv "$(TRC_CSV)" --pc-csv "$(CSV)" \
		--data $(DATA) --paper-policy $(PAPER_POLICY) $(EXCLUDE_FLAG) \
		--fingerprint-cache $(TRC_FINGERPRINTS) --paper-cap $(TRC_PAPER_CAP) \
		--chair-email $(TRC_CHAIR_EMAIL) --round $(TRC_ROUND) \
		--assignment-csv $(ASSIGNMENT_CSV) \
		--review-csv $(TRC_REVIEW_CSV) --tag-csv $(TRC_TAG_CSV) \
		--missing-tag-csv $(TRC_MISSING_TAG_CSV) \
		> $(TRC_ASSIGNMENT)

affiliation-countries: scripts/build_affiliation_countries.py src/reviewer_match/affiliation_country.py
	$(RUN) scripts.build_affiliation_countries --data $(DATA)

complete-papers: $(COMPLETE_ASSIGNMENT) $(COMPLETE_ASSIGNMENT_CSV)

area-chairs-complete: $(COMPLETE_ASSIGNMENT)
	$(RUN) scripts.enrich_publications --role area-chair --csv "$(AREA_CHAIR_CSV)" \
		--years $(AREA_CHAIR_YEARS)
	$(RUN) scripts.build_fingerprints --role area-chair --csv "$(AREA_CHAIR_CSV)" \
		--fingerprint-cache $(AREA_CHAIR_FINGERPRINTS) --years $(AREA_CHAIR_YEARS)
	$(RUN) scripts.assign_area_chairs --paper-policy complete \
		--reviewer-assignment $(COMPLETE_ASSIGNMENT) --csv "$(AREA_CHAIR_CSV)" \
		$(COAUTHOR_COI) $(COLLABORATOR_COI) $(EXCLUDE_FLAG) \
		--account-tag-csv $(AREA_CHAIR_ACCOUNT_TAGS_COMPLETE) --paper-tag-csv $(AREA_CHAIR_PAPER_TAGS_COMPLETE) \
		> $(AREA_CHAIR_COMPLETE)

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

# Same idiom, for the reserve half of the pool: reached when missing, and now
# also when older than RESERVE_INFO -- fingerprint content is purely PID and
# publications, so it only actually depends on identity, not on PCINFO's
# tags. RESERVE_CAP=off is no escape either: reserves elevated to the PC
# (`~~ex-rr`) are assigned even when the reserve bench is not, and their
# fingerprint is only ever built here.
$(RESERVE_FINGERPRINTS): $(RESERVE_INFO)
	@echo "ERROR: $@ is missing or older than $(RESERVE_INFO); run make reserves" >&2
	@echo "       (needed even with RESERVE_CAP=off, for the ex-reserves now on the PC)" >&2
	@exit 1

# Seniority is different: `classify_reviewers --role reserve` bakes the
# `~~ex-rr` tier tag straight into the tier column, and that tag lives only in
# PCINFO -- so a fresh export can promote or re-tier a reserve without
# touching RESERVE_INFO at all. On 2026-08-06 that silently left two
# ex-reserves with no seniority row and no area, still assigned, just under a
# `[light/?]` unknown class nobody was told about. A named, loud error here
# beats a silent one there, the same tradeoff every other "missing export"
# check in this Makefile makes.
$(RESERVE_SENIORITY): $(RESERVE_INFO) $(PCINFO)
	@echo "ERROR: $@ is missing or older than $(RESERVE_INFO)/$(PCINFO); run make reserves" >&2
	@echo "       (needed even with RESERVE_CAP=off, for the ex-reserves now on the PC)" >&2
	@exit 1

# Same idiom again, one stage earlier: the reserve roster is what names the
# ex-reserves in the first place, so the assignment needs it however RESERVE_CAP
# is set.
$(RESERVE_INFO):
	@echo "ERROR: $@ not found; run make reserve-info VERIFY=--verify" >&2
	@exit 1

$(SENIORITY): scripts/classify_reviewers.py $(REVIEWER_LIBS) $(CSV_DEP) $(OVERRIDES) $(PCDB) $(PCINFO)
	$(RUN) scripts.classify_reviewers --csv "$(CSV)" $(PC_CHECK) --out $@

$(FINGERPRINTS): $(PUBLICATIONS) $(ABSTRACTS) scripts/build_fingerprints.py $(REVIEWER_LIBS) $(EMBED_LIBS) $(CSV_DEP) $(OVERRIDES) $(DBLP_PUBS) $(PCINFO)
	$(RUN) scripts.build_fingerprints --csv "$(CSV)" --fingerprint-cache $@

$(ASSIGNMENT) $(ASSIGNMENT_CSV) &: $(ASSIGN_DEPS)
	$(RUN) scripts.assign_reviewers --paper-policy $(PAPER_POLICY) --csv "$(CSV)" \
		--area-chair-csv "$(AREA_CHAIR_CSV)" \
		--hotcrp-csv $(ASSIGNMENT_CSV) \
		$(RESERVE_FLAG) $(PC_CHECK) $(AREA_CHAIR_CHECK) $(REGION_FLAG) $(JUNIOR_FLAG) $(COAUTHOR_COI) $(COLLABORATOR_COI) $(EXCLUDE_FLAG) \
		> $(ASSIGNMENT)

# --- Incremental rerun ------------------------------------------------------
# What HotCRP actually holds, replayed from its action log — manual UI edits
# included, which no artifact under $(ASSIGNMENT_DIR) knows about. Offline,
# instant, read-only. Prefer HotCRP's own "Download → Review assignments"
# export when you have it; this is the reconstruction, and diffing the two
# checks both.
log-assignments:
	@test -f $(LOG) || { echo "ERROR: $(LOG) not found; download the action log from HotCRP" >&2; exit 1; }
	$(RUN) scripts.extract_log_assignments --log $(LOG) --out $(CURRENT_ASSIGNMENT_CSV)

# Who has not touched their assignment since it landed. Writes the address
# list `rerun` pins on. Note HotCRP logs no login event: this is a proxy.
reviewer-activity:
	@test -f $(LOG) || { echo "ERROR: $(LOG) not found; download the action log from HotCRP" >&2; exit 1; }
	$(RUN) scripts.audit_reviewer_activity --log $(LOG) $(PC_CHECK) \
		--signal $(ACTIVITY_SIGNAL) --out $(REVIEWER_ACTIVITY) \
		--pinned-out $(PINNED_REVIEWERS)

# Re-solve around the reviewers who have already started work: their current
# pairs are frozen and they get nothing new, every other pair is released and
# rematched. Writes to assignment-rerun.* and never over $(ASSIGNMENT_CSV), so
# `make diff` against the live state stays possible. Nothing is uploaded here —
# read the churn in the diff first.
rerun: $(ASSIGN_DEPS) scripts/extract_log_assignments.py scripts/audit_reviewer_activity.py \
		src/reviewer_match/hotcrp_log.py src/reviewer_match/assignment_io.py
	$(MAKE) log-assignments reviewer-activity
	$(RUN) scripts.assign_reviewers --paper-policy $(PAPER_POLICY) --csv "$(CSV)" \
		--area-chair-csv "$(AREA_CHAIR_CSV)" \
		--pin-csv $(CURRENT_ASSIGNMENT_CSV) --pin-emails $(PINNED_REVIEWERS) \
		--hotcrp-csv $(RERUN_ASSIGNMENT_CSV) \
		$(RESERVE_FLAG) $(PC_CHECK) $(AREA_CHAIR_CHECK) $(REGION_FLAG) $(JUNIOR_FLAG) $(COAUTHOR_COI) $(COLLABORATOR_COI) $(EXCLUDE_FLAG) \
		> $(RERUN_ASSIGNMENT)
	@echo "" >&2
	@echo "Churn this rerun would cause, against what HotCRP holds today:" >&2
	@echo "Read the goodness delta before uploading: churn that buys nothing is churn." >&2
	@echo "Every address in $(PINNED_REVIEWERS) must show a load delta of 0 and no lineup change." >&2
	$(RUN) scripts.diff_assignments $(CURRENT_ASSIGNMENT_CSV) $(RERUN_ASSIGNMENT_CSV) \
		--old-label live --new-label rerun --score-affinity >&2

$(COMPLETE_ASSIGNMENT) $(COMPLETE_ASSIGNMENT_CSV) &: $(ASSIGN_DEPS)
	$(RUN) scripts.assign_reviewers --paper-policy complete --csv "$(CSV)" \
		--area-chair-csv "$(AREA_CHAIR_CSV)" \
		--hotcrp-csv $(COMPLETE_ASSIGNMENT_CSV) \
		$(PC_CHECK) $(AREA_CHAIR_CHECK) $(REGION_FLAG) $(JUNIOR_FLAG) $(COAUTHOR_COI) $(COLLABORATOR_COI) $(EXCLUDE_FLAG) \
		> $(COMPLETE_ASSIGNMENT)

# Randomized baselines: how much of the match quality is the SPECTER2 signal?
# Arm A is the production configuration, B drops SPECTER2 and the declared-area
# gate, C drops SPECTER2 only -- so A-C is what the embedding buys inside an
# area and C-B is what the area gate buys. Identical policy flags by
# construction, because they are the same variables the $(ASSIGNMENT) recipe
# uses: SAME_COUNTRY_CAP=1 and MAX_JUNIORS=2 differ from the script's own
# defaults, and a hand-typed command line gets them wrong.
#
# --surplus-per-paper 0 on every arm, and not negotiable: the surplus stage
# offers slots to the worst-matched papers, "worst-matched" is measured on the
# ranking score, and that means something different once the ranking is noise.
# Arm A is re-run here rather than reused from $(ASSIGNMENT) so all three share
# it. No --hotcrp-csv anywhere: a baseline slate must never be uploadable, which
# assign_reviewers.py also refuses on its own.
BASELINE_SEEDS ?= 1
BASELINE_FLAGS = --paper-policy $(PAPER_POLICY) --csv "$(CSV)" \
	--area-chair-csv "$(AREA_CHAIR_CSV)" --surplus-per-paper 0 \
	$(RESERVE_FLAG) $(PC_CHECK) $(AREA_CHAIR_CHECK) $(REGION_FLAG) $(JUNIOR_FLAG) \
	$(COAUTHOR_COI) $(COLLABORATOR_COI) $(EXCLUDE_FLAG)

# A random arm routinely leaves a paper short, so assign_reviewers exits 1 under
# --paper-policy submitted. That is a finding, not a build failure -- the report
# is wanted either way -- so the exit code is noted and the loop continues,
# which also keeps .DELETE_ON_ERROR from removing the transcript.
baselines: $(ASSIGN_DEPS) scripts/compare_baselines.py
	@mkdir -p $(EVALUATION_DIR)
	$(RUN) scripts.assign_reviewers $(BASELINE_FLAGS) \
		--pairs-csv $(EVALUATION_DIR)/pairs-armA.csv \
		> $(EVALUATION_DIR)/assignment-armA.txt
	@for s in $(BASELINE_SEEDS); do \
	  for arm in B:--no-area-gate C:; do \
	    a=$${arm%%:*}; extra=$${arm#*:}; \
	    echo "$(RUN) scripts.assign_reviewers ... --score-mode random --score-seed $$s $$extra"; \
	    $(RUN) scripts.assign_reviewers $(BASELINE_FLAGS) --score-mode random \
	      --score-seed $$s $$extra \
	      --pairs-csv $(EVALUATION_DIR)/pairs-arm$$a-s$$s.csv \
	      > $(EVALUATION_DIR)/assignment-arm$$a-s$$s.txt \
	      || echo "arm $$a seed $$s: incomplete slate, see its shortage report" >&2; \
	  done; \
	done
	$(RUN) scripts.compare_baselines $(EVALUATION_DIR)/pairs-armA.csv \
		$(EVALUATION_DIR)/pairs-armB-*.csv $(EVALUATION_DIR)/pairs-armC-*.csv

clean:
	rm -f $(ASSIGNMENT) $(ASSIGNMENT_CSV) $(AREA_CHAIR_ASSIGNMENT) \
		$(COMPLETE_ASSIGNMENT) $(COMPLETE_ASSIGNMENT_CSV) $(AREA_CHAIR_COMPLETE) \
		$(AREA_CHAIR_ACCOUNT_TAGS) $(AREA_CHAIR_PAPER_TAGS) \
		$(AREA_CHAIR_ACCOUNT_TAGS_COMPLETE) $(AREA_CHAIR_PAPER_TAGS_COMPLETE) \
		$(TRC_ASSIGNMENT) $(TRC_REVIEW_CSV) $(TRC_TAG_CSV) $(TRC_MISSING_TAG_CSV)

clean-fingerprints:
	rm -f $(FINGERPRINTS) $(PAPER_FINGERPRINTS) $(AREA_CHAIR_FINGERPRINTS) $(TRC_FINGERPRINTS)
