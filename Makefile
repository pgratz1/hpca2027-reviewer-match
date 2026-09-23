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
#   make revision-cutoffs share of reviewed papers each revision cutoff would catch
#   make paper-leads      random, load-balanced leads for papers that advance
#   make revision-tags    RevisionAdvance / NoRevision tags for decided papers
#   make timeliness-tags  ~~ontime / ~~onedaylate / ... by authors' own reviews
#   make timeliness-emails  draft author emails for papers held by a late author
#   make timeliness-apologies  corrections for emailed papers no longer held
#   make extra-reviewers  shortlist finished PC members to ask for one more review
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

# targeted-rerun: like `rerun`, but a named FORCE_RELEASE list of reviewers is
# released regardless of their HotCRP activity -- for a reviewer whose
# fingerprint was wrong (see dblp_identity_audit.md), not just one who hasn't
# logged in yet. A pair a PC chair hand-edited still wins over the force
# list; see build_targeted_rerun_pins.py. FORCE_RELEASE is required, so a
# bare `make targeted-rerun` fails loudly rather than silently no-op'ing.
FORCE_RELEASE ?=
FP_SOURCE_OVERRIDES = $(CURATED_DIR)/fingerprint_source_overrides.csv
TARGETED_PINNED_REVIEWERS = $(REPORT_DIR)/reviewers_pinned_targeted.txt
TARGETED_ASSIGNMENT = $(ASSIGNMENT_DIR)/assignment-targeted-rerun.txt
TARGETED_ASSIGNMENT_CSV = $(ASSIGNMENT_DIR)/assignment-targeted-rerun.csv

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
	log-assignments reviewer-activity rerun targeted-rerun swap-candidates swap-upload fill-slots \
	revision-cutoffs paper-leads revision-tags timeliness-tags timeliness-emails \
	timeliness-apologies extra-reviewers

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

# Re-solve around everyone with HotCRP activity or a manually-edited pair,
# EXCEPT the named FORCE_RELEASE reviewers, who are released regardless of
# activity -- for fixing specific known-bad matches, not a general rerun.
# Rebuilds fingerprints for just the FORCE_RELEASE set first (own submitted
# abstracts or area-only, per fingerprint_source_overrides.csv -- see
# outputs/reports/dblp_identity_audit.md), across both rosters since a
# reviewer's fingerprint role and their assignment tier can differ (an
# ~~ex-rr promoted to pc-light still fingerprints as a reserve). Writes to
# assignment-targeted-rerun.* and never over $(ASSIGNMENT_CSV) or
# $(RERUN_ASSIGNMENT_CSV). Nothing is uploaded here — read the churn in the
# diff first.
targeted-rerun: $(ASSIGN_DEPS) scripts/extract_log_assignments.py scripts/audit_reviewer_activity.py \
		scripts/build_targeted_rerun_pins.py scripts/build_fingerprints.py \
		src/reviewer_match/hotcrp_log.py src/reviewer_match/assignment_io.py
	@test -n "$(FORCE_RELEASE)" || { echo "ERROR: make targeted-rerun FORCE_RELEASE=<path, one address per line>" >&2; exit 1; }
	$(MAKE) log-assignments reviewer-activity
	$(RUN) scripts.build_fingerprints --csv "$(CSV)" --fingerprint-cache $(FINGERPRINTS) \
		--fingerprint-source-overrides $(FP_SOURCE_OVERRIDES) --emails $(FORCE_RELEASE)
	$(RUN) scripts.build_fingerprints --role reserve --csv $(RESERVE_INFO) --data $(DATA) \
		--fingerprint-cache $(RESERVE_FINGERPRINTS) \
		--fingerprint-source-overrides $(FP_SOURCE_OVERRIDES) --emails $(FORCE_RELEASE)
	$(RUN) scripts.build_targeted_rerun_pins \
		--activity-pinned $(PINNED_REVIEWERS) \
		--baseline-csv $(ASSIGNMENT_CSV) --current-csv $(CURRENT_ASSIGNMENT_CSV) \
		--force-release $(FORCE_RELEASE) --out $(TARGETED_PINNED_REVIEWERS)
	$(RUN) scripts.assign_reviewers --paper-policy $(PAPER_POLICY) --csv "$(CSV)" \
		--area-chair-csv "$(AREA_CHAIR_CSV)" \
		--pin-csv $(CURRENT_ASSIGNMENT_CSV) --pin-emails $(TARGETED_PINNED_REVIEWERS) \
		--hotcrp-csv $(TARGETED_ASSIGNMENT_CSV) \
		$(RESERVE_FLAG) $(PC_CHECK) $(AREA_CHAIR_CHECK) $(REGION_FLAG) $(JUNIOR_FLAG) $(COAUTHOR_COI) $(COLLABORATOR_COI) $(EXCLUDE_FLAG) \
		> $(TARGETED_ASSIGNMENT)
	@echo "" >&2
	@echo "Churn this targeted rerun would cause, against what HotCRP holds today:" >&2
	@echo "Read the goodness delta before uploading: churn that buys nothing is churn." >&2
	$(RUN) scripts.diff_assignments $(CURRENT_ASSIGNMENT_CSV) $(TARGETED_ASSIGNMENT_CSV) \
		--old-label live --new-label targeted-rerun --score-affinity >&2

# A reviewer has left the committee entirely (or, with SWAP_PIDS set, just a
# few named papers -- a late COI, an ability-to-review issue -- keeping the
# rest of their load untouched) and needs no re-solve, just a proposal: for
# each paper they leave short of the target slate size (the script's own
# --reviewers-per-paper default, 5), which already-assigned reviewer on some
# OTHER paper holding one more than that could move over without breaking any
# standing rule (COI, area, country cap, the senior floor) and without having
# already submitted the review they'd be leaving. Prints two ranked
# candidates per short paper (primary + backup); nothing is solved globally
# and nothing is uploaded -- see scripts/propose_reviewer_swaps.py.
# DEPARTED_EMAIL is required, so a bare `make swap-candidates` fails loudly
# rather than silently no-op'ing.
DEPARTED_EMAIL ?=
SWAP_PIDS ?=
SWAP_PIDS_FLAG = $(if $(SWAP_PIDS),--departed-pids $(SWAP_PIDS),)
SWAP_EXCLUDE ?=
SWAP_EXCLUDE_FLAG = $(if $(SWAP_EXCLUDE),--exclude-movers $(SWAP_EXCLUDE),)
SWAP_PAIRS_CSV = $(ASSIGNMENT_DIR)/proposed_swaps.csv

swap-candidates: scripts/extract_log_assignments.py scripts/propose_reviewer_swaps.py \
		src/reviewer_match/hotcrp_log.py src/reviewer_match/assignment_io.py \
		src/reviewer_match/paper_matching.py $(SENIORITY) $(FINGERPRINTS) $(PAPER_FINGERPRINTS)
	@test -n "$(DEPARTED_EMAIL)" || { echo "ERROR: make swap-candidates DEPARTED_EMAIL=<address>" >&2; exit 1; }
	$(MAKE) log-assignments
	$(RUN) scripts.propose_reviewer_swaps --departed-email "$(DEPARTED_EMAIL)" $(SWAP_PIDS_FLAG) $(SWAP_EXCLUDE_FLAG) \
		--paper-policy $(PAPER_POLICY) --csv "$(CSV)" --area-chair-csv "$(AREA_CHAIR_CSV)" \
		--current-csv $(CURRENT_ASSIGNMENT_CSV) --log $(LOG) \
		--fingerprint-cache $(FINGERPRINTS) --paper-cache $(PAPER_FINGERPRINTS) \
		--seniority $(SENIORITY) --pairs-csv $(SWAP_PAIRS_CSV) \
		$(RESERVE_FLAG) $(PC_CHECK) $(AREA_CHAIR_CHECK) $(REGION_FLAG) $(JUNIOR_FLAG) $(COAUTHOR_COI) $(COLLABORATOR_COI) $(EXCLUDE_FLAG)

# Once the chair has actually asked around and knows who said yes, this turns
# a hand-maintained CONFIRMED_SWAPS CSV (target_pid,add_email,source_pid,
# remove_source -- see scripts/generate_swap_upload.py) into the small HotCRP
# Assignments -> Bulk update delta that applies it: clear DEPARTED_EMAIL from
# each target paper, add the confirmed mover, and (unless a row's own
# remove_source says no -- an add, not a move, for a mover who is keeping the
# paper they already started) clear the mover from their source paper too.
# Deliberately NOT the all,clearreview,all,R1 shape swap-candidates and every
# other bulk upload here open with -- this is a delta, not a replacement, so
# only the named pairs are touched. Preview it in HotCRP before approving.
CONFIRMED_SWAPS ?= data/curated/confirmed_swaps.csv
SWAP_UPLOAD_OUT = $(ASSIGNMENT_DIR)/swap_upload.csv
# The papers a departing reviewer holds that no confirmed swap backfills --
# they still need the review cleared. Empty for a partial departure, where
# the reviewer keeps everything the swaps do not move.
SWAP_CLEAR_PIDS ?=
SWAP_CLEAR_FLAG = $(if $(SWAP_CLEAR_PIDS),--clear-pids $(SWAP_CLEAR_PIDS),)

swap-upload: scripts/generate_swap_upload.py src/reviewer_match/reviewers.py src/reviewer_match/reserve_reviewers.py
	@test -n "$(DEPARTED_EMAIL)" || { echo "ERROR: make swap-upload DEPARTED_EMAIL=<address>" >&2; exit 1; }
	@test -f "$(CONFIRMED_SWAPS)" || { echo "ERROR: $(CONFIRMED_SWAPS) not found -- see scripts/generate_swap_upload.py" >&2; exit 1; }
	$(RUN) scripts.generate_swap_upload --departed-email "$(DEPARTED_EMAIL)" \
		--confirmed-csv $(CONFIRMED_SWAPS) --out $(SWAP_UPLOAD_OUT) --csv "$(CSV)" \
		$(SWAP_CLEAR_FLAG) $(PC_CHECK)

# Reviewers being pulled off whatever they have not submitted yet, once it is
# too late to swap: every paper that drops below FILL_TO reviewers gets topped
# back up from spare capacity (desk rejections free some), under the same COI
# layers, area gate, seniority, junior/out-of-area and same-country rules as
# the main assignment, and nobody receives more than MAX_NEW_PER_REVIEWER new
# papers. Submitted reviews stay. The baseline is HotCRP's own
# Search -> Download -> "Review assignments" export, re-downloaded together
# with the action log. Writes FILL_SLOTS_UPLOAD, a delta (clear rows for the
# removed pairs, add rows for the new ones -- never all,clearreview), for
# HotCRP's Assignments -> Bulk update preview. Nothing is uploaded here.
# Seniority and fingerprints are checked for, not prerequisites: a patch has
# to score with what the live assignment was built from, so a fresh HotCRP
# export must not trigger a reclassification or a re-embed on the way through.
REMOVED_EMAILS ?=
FILL_TO ?= 5
MAX_NEW_PER_REVIEWER ?= 2
FILL_SLOTS_BASELINE ?= $(INPUT_DIR)/hpca2027-pcassignments.csv
# Optional file of addresses never to offer a paper. Anyone the log shows
# pulled off every review is excluded without it.
FILL_SLOTS_EXCLUDE ?=
# New papers go only to reviewers whose load a desk rejection lightened.
# FILL_SLOTS_POOL= (empty) opens the pool to anyone with spare capacity.
FILL_SLOTS_POOL ?= --only-dropped-paper-reviewers
FILL_SLOTS_EXCLUDE_FLAG = $(if $(FILL_SLOTS_EXCLUDE),--exclude-candidates $(FILL_SLOTS_EXCLUDE),)
FILL_SLOTS_PAIRS = $(ASSIGNMENT_DIR)/fill_slots_pairs.csv
FILL_SLOTS_UPLOAD = $(ASSIGNMENT_DIR)/fill_slots_upload.csv

fill-slots: scripts/fill_open_slots.py scripts/assign_reviewers.py src/reviewer_match/hotcrp_log.py \
		src/reviewer_match/assignment_io.py src/reviewer_match/paper_matching.py
	@test -n "$(REMOVED_EMAILS)" || { echo "ERROR: make fill-slots REMOVED_EMAILS=\"<address> [<address> ...]\"" >&2; exit 1; }
	@for f in $(SENIORITY) $(FINGERPRINTS) $(RESERVE_FINGERPRINTS) $(RESERVE_SENIORITY); do \
		test -f $$f || { echo "ERROR: $$f not found; run make and make reserves first" >&2; exit 1; }; done
	@test -f $(FILL_SLOTS_BASELINE) || { echo "ERROR: $(FILL_SLOTS_BASELINE) not found; download HotCRP's review assignments" >&2; exit 1; }
	@test -f $(LOG) || { echo "ERROR: $(LOG) not found; download the action log from HotCRP" >&2; exit 1; }
	$(RUN) scripts.fill_open_slots --baseline $(FILL_SLOTS_BASELINE) --log $(LOG) \
		$(foreach e,$(REMOVED_EMAILS),--removed-email $(e)) \
		--fill-to $(FILL_TO) --max-new-per-reviewer $(MAX_NEW_PER_REVIEWER) $(FILL_SLOTS_EXCLUDE_FLAG) $(FILL_SLOTS_POOL) \
		--paper-policy $(PAPER_POLICY) --csv "$(CSV)" --area-chair-csv "$(AREA_CHAIR_CSV)" \
		--fingerprint-cache $(FINGERPRINTS) --paper-cache $(PAPER_FINGERPRINTS) --seniority $(SENIORITY) \
		--pairs-csv $(FILL_SLOTS_PAIRS) --delta-hotcrp-csv $(FILL_SLOTS_UPLOAD) \
		$(RESERVE_FLAG) $(PC_CHECK) $(AREA_CHAIR_CHECK) $(REGION_FLAG) $(JUNIOR_FLAG) $(COAUTHOR_COI) $(COLLABORATOR_COI) $(EXCLUDE_FLAG)

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

# What share of the papers with REVISION_MIN_REVIEWS+ submitted reviews each
# revision-eligibility net would catch, by average pre-rebuttal overall merit,
# at "<=" and "<" alike. TRC reviews are left out (identified from the action
# log, which also counts each paper's outstanding reviews). Offline, instant,
# read-only apart from its two reports. REVISION_FLAGS passes more through,
# e.g. REVISION_FLAGS=--complete-only or REVISION_FLAGS=--include-trc.
REVIEWS = $(INPUT_DIR)/hpca2027-reviews.csv
REVISION_MIN_REVIEWS ?= 5
REVISION_FLAGS ?=
revision-cutoffs: scripts/revision_cutoffs.py src/reviewer_match/review_scores.py src/reviewer_match/hotcrp_log.py
	@test -f $(REVIEWS) || { echo "ERROR: $(REVIEWS) not found; download the reviews CSV from HotCRP" >&2; exit 1; }
	@test -f $(LOG) || { echo "ERROR: $(LOG) not found; download the action log from HotCRP" >&2; exit 1; }
	$(RUN) scripts.revision_cutoffs --reviews $(REVIEWS) --log $(LOG) --data $(DATA) \
		--min-reviews $(REVISION_MIN_REVIEWS) $(EXCLUDE_FLAG) $(REVISION_FLAGS)

# One discussion lead per paper that advances to revision: fewer than
# REVISION_MIN_REVIEWS submitted PC reviews, or over the bar (default: not
# "average <= 2.5 and at most one score of 3 or better"; LEAD_FLAGS="--bar-cutoff
# 2.25 --bar-net no4" changes it). Drawn at random from the paper's own submitted
# full/light PC reviewers, with lead load proportional to assigned review load.
# Existing leads in LEADS (search page > Download > Reviews > "Discussion leads
# (CSV)"; the review-assignments download carries none) are kept, with the ones
# it hides on the downloader's conflicts filled in from the log; the upload is
# a delta. Nothing is uploaded. LEAD_SEED changes the draw.
PCASSIGNMENTS = $(INPUT_DIR)/hpca2027-pcassignments.csv
LEADS = $(INPUT_DIR)/hpca2027-leads.csv
LEAD_SEED ?= 1
LEAD_FLAGS ?=
paper-leads: scripts/assign_paper_leads.py src/reviewer_match/review_scores.py \
		src/reviewer_match/assignment_io.py src/reviewer_match/hotcrp_log.py
	@for f in $(REVIEWS) $(LOG) $(PCASSIGNMENTS) $(LEADS); do \
	  test -f $$f || { echo "ERROR: $$f not found; download it from HotCRP" >&2; exit 1; }; \
	done
	$(RUN) scripts.assign_paper_leads --reviews $(REVIEWS) --log $(LOG) --data $(DATA) \
		--pcassignments $(PCASSIGNMENTS) --existing-leads $(LEADS) --min-reviews $(REVISION_MIN_REVIEWS) \
		--seed $(LEAD_SEED) $(EXCLUDE_FLAG) $(LEAD_FLAGS)

# RevisionAdvance on papers over the bar, NoRevision on those under it, for
# every paper with REVISION_MIN_REVIEWS+ submitted PC reviews; papers short of
# reviews stay untagged. Same bar and flags as paper-leads (LEAD_FLAGS), so the
# two agree. The upload is a delta that clears the opposite tag first, so a rerun
# after a paper crosses the bar is safe. Nothing is uploaded.
revision-tags: scripts/revision_tags.py src/reviewer_match/review_scores.py src/reviewer_match/hotcrp_log.py
	@for f in $(REVIEWS) $(LOG); do \
	  test -f $$f || { echo "ERROR: $$f not found; download it from HotCRP" >&2; exit 1; }; \
	done
	$(RUN) scripts.revision_tags --reviews $(REVIEWS) --log $(LOG) --data $(DATA) \
		--min-reviews $(REVISION_MIN_REVIEWS) $(EXCLUDE_FLAG) $(LEAD_FLAGS)

# ~~ontime on every paper whose PC/reserve authors had submitted all the R1
# reviews they hold by ONTIME_CUTOFF (or that has no such author), and
# ~~onedaylate, ~~twodayslate, ... by the 24-hour window the last of them first
# submitted in. A paper whose author still owes a review has no day yet and gets
# ~~delayedmissingreview until it does. TRC reviews are ignored. A reviewer given
# an R1 review after EXEMPT_AFTER, or listed as an extension in EXTENSIONS,
# counts as on time. Papers already carrying a day tag in DATA are never
# re-tagged; the placeholder is cleared off every paper no longer blocked, so the
# upload is a delta. Run it daily on a fresh log and paper export and upload the
# result. Nothing is uploaded.
ONTIME_CUTOFF ?= 2026-09-22 11:00:00 -0400
EXEMPT_AFTER ?= 2026-09-13
EXTENSIONS ?= $(CURATED_DIR)/review_extensions.csv
timeliness-tags: scripts/timeliness_tags.py src/reviewer_match/hotcrp_log.py src/reviewer_match/review_scores.py
	@for f in $(LOG) $(DATA); do \
	  test -f $$f || { echo "ERROR: $$f not found; download it from HotCRP" >&2; exit 1; }; \
	done
	$(RUN) scripts.timeliness_tags --log $(LOG) --data $(DATA) --cutoff "$(ONTIME_CUTOFF)" \
		--exempt-after $(EXEMPT_AFTER) --extensions "$(EXTENSIONS)" $(PC_CHECK) $(EXCLUDE_FLAG)

# One drafted email per paper held by ~~delayedmissingreview, addressed to that
# paper's authors and contacts, naming the committee author whose reviews are
# outstanding. Decided by the same timeliness_tags.evaluate() as the tags, so the
# two cannot name different people. ANNOUNCED_DEADLINE is what the email quotes
# and is deliberately not ONTIME_CUTOFF: the cutoff the tags use is later, so
# nobody is tagged for missing the announced time by a few hours. The output
# names individuals to their collaborators -- read it before sending any of it.
# Nothing is sent.
ANNOUNCED_DEADLINE ?= Monday, September 21, 2026 at 8:00am EST
SIGNATURE ?= -- The HPCA 2027 Program Chairs
timeliness-emails: scripts/timeliness_emails.py scripts/timeliness_tags.py
	@for f in $(LOG) $(DATA); do \
	  test -f $$f || { echo "ERROR: $$f not found; download it from HotCRP" >&2; exit 1; }; \
	done
	$(RUN) scripts.timeliness_emails --log $(LOG) --data $(DATA) --cutoff "$(ONTIME_CUTOFF)" \
		--exempt-after $(EXEMPT_AFTER) --extensions "$(EXTENSIONS)" \
		--announced-deadline "$(ANNOUNCED_DEADLINE)" --signature "$(SIGNATURE)" \
		$(PC_CHECK) $(EXCLUDE_FLAG)

# Corrections for papers in an already-sent drafts file (SENT_EMAILS) that are
# not held today -- the first batch counted reviews on desk-rejected papers as
# outstanding. Sent papers still held are listed on stdout, not emailed. Keep a
# copy of each sent file under its own name: timeliness-emails overwrites its
# output. Nothing is sent.
SENT_EMAILS ?= outputs/reports/timeliness_emails_sent_2026-09-22.txt
timeliness-apologies: scripts/timeliness_emails.py scripts/timeliness_tags.py
	@for f in $(LOG) $(DATA) "$(SENT_EMAILS)"; do \
	  test -f "$$f" || { echo "ERROR: $$f not found" >&2; exit 1; }; \
	done
	$(RUN) scripts.timeliness_emails --log $(LOG) --data $(DATA) --cutoff "$(ONTIME_CUTOFF)" \
		--exempt-after $(EXEMPT_AFTER) --extensions "$(EXTENSIONS)" \
		--signature "$(SIGNATURE)" --apology-for "$(SENT_EMAILS)" \
		$(PC_CHECK) $(EXCLUDE_FLAG)

# For each EXTRA_PIDS paper, a ranked shortlist of full/light PC members who
# first-submitted every R1 review they hold (papers still under review only)
# before COMPLETED_BY, to ask for one extra review -- plus one suggested first
# ask per paper, nobody suggested twice. Every COI layer, the area gate (in-area
# first, then released) and the junior/out-of-area/same-country caps against
# the paper's live R1 slate bind; the tier load cap does not, since everyone
# listed has met theirs. Offline, no GPU. Nothing is uploaded.
EXTRA_PIDS ?= 178,344,583,707,1723
COMPLETED_BY ?= 2026-09-19 08:00:00 -0400
EXTRA_SHORTLIST ?= 8
extra-reviewers: scripts/extra_reviewer_candidates.py
	@for f in $(LOG) $(DATA) $(PCINFO) $(SENIORITY) $(RESERVE_SENIORITY) $(FINGERPRINTS) $(RESERVE_FINGERPRINTS) $(COAUTHORS); do \
	  test -f $$f || { echo "ERROR: $$f not found" >&2; exit 1; }; \
	done
	$(RUN) scripts.extra_reviewer_candidates --pids $(EXTRA_PIDS) --completed-by "$(COMPLETED_BY)" \
		--shortlist $(EXTRA_SHORTLIST) --log $(LOG) --data $(DATA) $(REGION_FLAG) $(JUNIOR_FLAG) $(EXCLUDE_FLAG)

clean:
	rm -f $(ASSIGNMENT) $(ASSIGNMENT_CSV) $(AREA_CHAIR_ASSIGNMENT) \
		$(COMPLETE_ASSIGNMENT) $(COMPLETE_ASSIGNMENT_CSV) $(AREA_CHAIR_COMPLETE) \
		$(AREA_CHAIR_ACCOUNT_TAGS) $(AREA_CHAIR_PAPER_TAGS) \
		$(AREA_CHAIR_ACCOUNT_TAGS_COMPLETE) $(AREA_CHAIR_PAPER_TAGS_COMPLETE) \
		$(TRC_ASSIGNMENT) $(TRC_REVIEW_CSV) $(TRC_TAG_CSV) $(TRC_MISSING_TAG_CSV)

clean-fingerprints:
	rm -f $(FINGERPRINTS) $(PAPER_FINGERPRINTS) $(AREA_CHAIR_FINGERPRINTS) $(TRC_FINGERPRINTS)
