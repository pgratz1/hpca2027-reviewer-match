# HPCA 2027 Reviewer-Paper Matching Pipeline — Project Brief

> Historical design brief. The pipeline is now implemented; `README.md` and
> the code are authoritative where this document's original proposal differs.

## Goal
Automate and improve reviewer-paper assignment for HPCA 2027, which has the largest PC in the conference's history (including the most first-time reviewers). Replace/augment manual bidding with a data-driven affinity score, combining coarse-grained area selection with fine-grained textual similarity from each reviewer's recent publication history.

## Inputs (already collected)
- **Reviewers**: DBLP profile link (PID), primary area, secondary area — selected from ~12 predefined HPCA subject areas.
- **Papers**: abstract, author-selected keywords, primary area, secondary area — same 12-area taxonomy.

## Design decided so far
1. **Area filter (hard gate).** A reviewer is only eligible for a paper if there's overlap between {reviewer primary, reviewer secondary} and {paper primary, paper secondary}. This is a hard constraint, not a soft weight — reviewers should not be assigned outside their declared areas.
2. **DBLP fetch and enrichment.** Pull recent publications from each known DBLP PID, cache their DOIs, and enrich IEEE/ACM records with abstracts from IEEE Xplore and Semantic Scholar. DOI metadata and abstracts are persistent, resumable caches.
3. **Embedding model.** Use SPECTER2 (`allenai/specter2`) — purpose-built for scientific paper similarity (trained on citation-graph relatedness), used by OpenReview/ARR for the same reviewer-matching problem. Sentence-transformers (`all-mpnet-base-v2`) as a fallback if SPECTER2 setup is troublesome.
4. **Scoring.** Mean-pool normalized SPECTER2 documents for each reviewer's recent publications and declared areas. Use native `title [SEP] abstract` documents where an abstract is cached and title-only otherwise. Papers use title, abstract, and topics. Cosine similarity ranks eligible candidates.
5. **Output and assignment.** Run a phased, load-capped deferred-acceptance assignment with COI, area, seniority, junior, and out-of-area policy checks, plus explicit shortage and relaxation reports.

## Original open questions and current status
- **HotCRP integration**: resolved with a standalone assignment pipeline reading the HotCRP JSON export and enforcing reviewer loads and paper targets locally.
- **COI cross-check**: reuse the DBLP fetch step to pull coauthor lists and cross-reference against HotCRP's self-declared COI list, since self-declared COI lists are known to be incomplete.
- **Title-only vs. abstract-enriched reviewer profiles**: implemented through DOI enrichment and a blinded evaluation workflow. On the July 2026 snapshot, abstracts changed 37.2% of final assignment slots. That demonstrates a material effect, while a claim of improved accuracy remains pending chair ratings.
- **Weighting of primary vs. secondary area** if a soft weighting is wanted anywhere downstream (e.g., in tie-breaking within the optimizer).

## Suggested stack
- Python 3.11+
- `requests` for DBLP XML fetch
- `sentence-transformers` or HF `transformers` for SPECTER2
- `numpy`/`scipy` for cosine similarity at scale
- `networkx` or `pulp` for the optimizer, if not delegating to HotCRP's built-in assigner
- Runs comfortably on local hardware (RTX 4090) — full PC-scale embedding job is a few hundred reviewers × few hundred papers, seconds of GPU time, no cloud API needed.

## Non-goals
- No LLM-as-judge scoring for the core affinity calculation — this is an embeddings + filtering problem, not a generation problem. LLM (local Devstral via existing Ollama/opencode setup) is only useful for auxiliary tasks: edge-case triage, generating human-readable rationale for flagged assignments, or parsing irregular DBLP entries.

## Remaining validation

Complete the blinded chair-rating sample and compare title-only with
abstract-enriched nDCG@10, capable/expert rate in the top six, and unsuitable
reviewers. Continue refreshing the HotCRP snapshot and rerunning `make` as
registrations become complete.
