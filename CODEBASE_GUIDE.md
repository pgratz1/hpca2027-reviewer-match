# Codebase Guide

This repository implements the HPCA 2027 reviewer-to-paper matching workflow. It turns a PC acceptance-form export, DBLP publication histories, and a HotCRP paper export into a load-capped assignment whose candidates are filtered for conflicts and declared areas, ranked by SPECTER2 similarity, and shaped by seniority policy.

This guide is an implementation-oriented map of the repository. `README.md` remains the operator's manual, `CLAUDE.md` records maintenance conventions, and `hpca2027-matching-brief.md` is historical context. Where those differ, the current code and README describe the implemented behavior.

## Mental model

The system is best understood as four layers:

1. **Reviewer identity and metadata** — `reviewers.py` parses the acceptance CSV, keeps the latest submission per case-insensitive email, removes declines, applies hand-maintained DBLP overrides, and produces `Reviewer` records.
2. **Evidence and representations** — `dblp.py` obtains publication histories; `classify_reviewers.py` derives seniority; `build_fingerprints.py`, `fingerprint.py`, and `specter2_model.py` derive normalized reviewer vectors.
3. **Paper eligibility and affinity** — `paper_matching.py` filters incomplete papers, builds content-aware paper vectors, excludes conflicts, applies the area gate, and calculates cosine scores.
4. **Assignment and reporting** — `assign_reviewers.py` distributes reviewers globally under load and composition constraints. `score_papers.py`, `nearest_neighbors.py`, and `main.py` provide diagnostic views of intermediate results.

The normal data flow is:

```text
acceptance CSV + dblp_overrides.csv
        |
        +--> classify_reviewers.py --> reviewer_seniority.csv --------+
        |                                                             |
        +--> build_fingerprints.py --> fingerprints.json              |
                                                                      v
HotCRP JSON --> paper_matching.py --> paper_fingerprints.json --> assign_reviewers.py
                                                                      |
                                                                      v
                                                               assignment.txt
```

`make` encodes this dependency order and is the preferred entry point.

## Module map

### Input and identity

- `reviewers.py` is the canonical reviewer loader used throughout the project. CSV headers are found by case-insensitive substring rather than exact spelling. Missing expected columns fail loudly. Repeat form submissions are resolved by timestamp before acceptance status is evaluated.
- `dblp_overrides.csv` is the durable, email-keyed identity correction layer. A valid nonblank override wins over the form's DBLP value. The file is sensitive, hand-maintained, and gitignored.
- `dblp.py` has no CSV knowledge. It parses several PID formats, reads title-only and rich-publication caches, fetches DBLP XML from official mirrors when necessary, retries rate limits, and writes live results atomically after each successful fetch.

### Seniority

- `classify_reviewers.py` counts deduplicated papers at exactly ISCA, MICRO, HPCA, and ASPLOS. At the defaults, a reviewer is senior with at least 12 target papers in the 15-year window, junior with fewer than 20 publications overall, out-of-area with at least 20 overall but fewer than 5 career target-venue papers, and typical otherwise. The senior test is evaluated first; promote-only PC-service overrides are then applied.
- Reviewers without a PID are written as `unknown`. The script appends blank identity stubs to `dblp_overrides.csv`, making that file both the correction layer and the unresolved-identity queue.
- `reviewer_seniority.csv` stores both the class and its supporting counts. Assignment uses the counts for the near-threshold fallback groups, not just for auditing.

### Embeddings

- `specter2_model.py` is the model boundary: it loads `allenai/specter2_base` with the proximity adapter and returns CLS embeddings in batches.
- `enrich_publications.py` resolves DOI-bearing reviewer publications through DBLP and caches IEEE/ACM abstracts. IEEE Xplore is queried directly when available; Semantic Scholar handles ACM records and IEEE misses, with authenticated and rate-limited unauthenticated modes. Confirmed results and retryable failures have distinct cache states.
- `fingerprint.py` supplies SPECTER2's `title [SEP] abstract` document shape and normalized weighted pooling. A reviewer is represented by one document per recent DBLP publication plus one area/keyword document; publications without a cached abstract fall back to title-only.
- `build_fingerprints.py` orchestrates reviewer fetching and encoding. Reviewers without usable recent publications receive an area-only vector. Reviewer cache entries are keyed by normalized email and record schema/provenance, title count, abstract count, PID presence, and DBLP fetch completeness. `--no-abstracts` builds a controlled title-only baseline when paired with a separate cache path.
- `paper_matching.py` represents each paper with two documents: title plus abstract, and the topic list. Their pooled vector is normalized, so dot products against reviewer vectors are cosine similarities.

Both fingerprint caches store versioned provenance keys. Paper keys cover title, abstract, topics, area weight, and model identifiers. Reviewer keys cover identity, declared metadata, selected DBLP publications and abstracts, embedding policy, and model identifiers. Legacy entries rebuild once, and transient DBLP/API failures remain marked for retry. `make clean-fingerprints` remains available when an explicit full rebuild is desired.

### Eligibility and assignment

`paper_matching.eligible_scores` is the shared filter-and-rank boundary used by both scoring and assignment:

- an email present in `pc_conflicts` is ineligible;
- by default, at least one paper topic must match the reviewer's primary or secondary area, case-insensitively;
- tertiary area does not satisfy the gate;
- among eligible pairs, the score is the dot product of normalized vectors.

`assign_reviewers.py` performs paper-proposing deferred acceptance. Papers propose in descending affinity order, while each reviewer retains their best offers up to their cap. A CSV-level cap override takes precedence over the light/full tier defaults.

With seniority enabled, assignment is deliberately phased:

1. anchor each paper with true seniors;
2. use near-senior typical reviewers where a true senior cannot be placed;
3. fill remaining slots by score while enforcing the junior cap;
4. if necessary, admit extra near-threshold juniors to under-filled papers.

Earlier phases are frozen. This preserves the composition policy but means the final multi-phase result is not claimed to be globally stable in the classical sense. The script instead checks the relevant phase stability, reviewer caps, junior policy, and assignment consistency, then prints seniority and per-area shortage reports.

## Important invariants

- Email is the reviewer identity key and is normalized to lowercase.
- The latest Google Forms row per email is authoritative, including a later decline.
- Paper-side tools ignore entries without an abstract or with a title shorter than three words.
- COI is never released. Area overlap is a hard gate in the normal assignment phases, not a score component; it may be released per paper by the documented shortage ladder.
- Primary and secondary reviewer areas count for eligibility; tertiary does not.
- Fingerprint vectors are L2-normalized, so matrix dot products are cosine similarity.
- DBLP cache priority is colleague read-only cache, local writable cache, then live DBLP.
- Cache writes use temporary files plus replacement; reruns should be idempotent.
- Under-filled papers without topics appear in the shortage report under `Unspecified/no matching topic`.
- Results belong on stdout; progress, warnings, and summaries generally belong on stderr. `assignment.txt` is produced by redirecting assignment stdout.
- Real reviewer and paper data, overrides, caches, classifications, and assignments contain sensitive information and must not be committed.

## Operating the repository

The intended interpreter is `~/envs/hpca-matching/bin/python3`; the requirements file documents dependencies but does not reproduce the CUDA-enabled PyTorch environment.

```bash
make
```

This builds seniority, reviewer fingerprints, and the final assignment when their dependencies are stale. Useful focused checks are:

```bash
~/envs/hpca-matching/bin/python3 main.py --limit 5
~/envs/hpca-matching/bin/python3 build_fingerprints.py --limit 10
~/envs/hpca-matching/bin/python3 score_papers.py --pid 8 --top 10
~/envs/hpca-matching/bin/python3 nearest_neighbors.py --email someone@example.com
~/envs/hpca-matching/bin/python3 -m unittest tests.test_regressions
```

Use alternate `--out`, `--data`, `--fingerprint-cache`, and `--paper-cache` paths for experiments. Avoid deleting the rate-limited DBLP caches. `make clean-fingerprints` intentionally removes only embedding caches.

## Maintenance notes

The regression suite in `tests/test_regressions.py` covers enrichment fallback and cache behavior, fingerprint provenance, evaluation utilities, completeness filtering, policy validation, and assignment invariants. Validation also includes deterministic reruns and the self-check summary at the end of a full `make`; a successful process exit alone is insufficient for matching-logic changes.

For the abstract experiment, `compare_abstract_rankings.py` creates a blinded,
topic-stratified rating sheet and a separate rank key. After chair ratings are
entered, `score_abstract_evaluation.py` reports nDCG@10, capable/expert rate in
the top six, and unsuitable-reviewer count. Ranking or slate turnover proves
that abstracts affect selection, but only the blinded ratings support a claim
that accuracy improved.

Good extension points are intentionally narrow:

- CSV interpretation and reviewer fields belong in `reviewers.py`.
- DBLP parsing, retry, and cache behavior belong in `dblp.py`.
- embedding mechanics belong in `specter2_model.py` and `fingerprint.py`;
- shared paper completeness, fingerprinting, conflict, and area policy belong in `paper_matching.py`;
- global capacity and seniority policy belong in `assign_reviewers.py`.

When adding command-line behavior, follow the existing style: a useful module docstring reused by `argparse`, module-level default constants, explicit flags for tunable policy, and simple standard-library data handling rather than a framework. Before committing, inspect `git status` carefully because nearly all operational data is intentionally ignored and the only expected tracked changes are code and documentation.
