# HPCA 2027 Reviewer-Paper Matching Pipeline — Project Brief

## Goal
Automate and improve reviewer-paper assignment for HPCA 2027, which has the largest PC in the conference's history (including the most first-time reviewers). Replace/augment manual bidding with a data-driven affinity score, combining coarse-grained area selection with fine-grained textual similarity from each reviewer's recent publication history.

## Inputs (already collected)
- **Reviewers**: DBLP profile link (PID), primary area, secondary area — selected from ~12 predefined HPCA subject areas.
- **Papers**: abstract, author-selected keywords, primary area, secondary area — same 12-area taxonomy.

## Design decided so far
1. **Area filter (hard gate).** A reviewer is only eligible for a paper if there's overlap between {reviewer primary, reviewer secondary} and {paper primary, paper secondary}. This is a hard constraint, not a soft weight — reviewers should not be assigned outside their declared areas.
2. **DBLP fetch.** Pull each reviewer's last 10 publication titles from their DBLP PID (`https://dblp.org/pid/{PID}.xml`), sorted by year descending. No author disambiguation needed since PIDs are already known.
3. **Embedding model.** Use SPECTER2 (`allenai/specter2`) — purpose-built for scientific paper similarity (trained on citation-graph relatedness), used by OpenReview/ARR for the same reviewer-matching problem. Sentence-transformers (`all-mpnet-base-v2`) as a fallback if SPECTER2 setup is troublesome.
4. **Scoring.** Within each area-eligible reviewer-paper pair: embed reviewer's 10 titles (concatenated or mean-pooled) and embed paper's abstract + keywords, take cosine similarity. This is the ranking signal *within* the area-gated candidate set — area itself is not blended into the score, just used as the eligibility filter.
5. **Output.** A reviewer × paper affinity table (only for area-eligible pairs) to feed into the assignment step.

## Open questions to resolve during implementation
- **HotCRP integration**: check whether we're using HotCRP's native "Topics" feature for the area taxonomy already — if so, the area gate may already exist in HotCRP and we just need to inject the DBLP-cosine score as an external/topic score via HotCRP's bulk import, and let HotCRP's own globally-optimal autoassigner (min-cost max-flow) do the final assignment. If not, decide whether to build a standalone optimizer (e.g., min-cost flow via networkx/scipy, or ILP via PuLP) that respects per-reviewer load and per-paper reviewer-count constraints.
- **COI cross-check**: reuse the DBLP fetch step to pull coauthor lists and cross-reference against HotCRP's self-declared COI list, since self-declared COI lists are known to be incomplete.
- **Title-only vs. abstract-enriched reviewer profiles**: DBLP doesn't carry abstracts. Decide whether title-only SPECTER2 embeddings are sufficient, or whether it's worth cross-referencing Semantic Scholar (by DOI/title) to enrich reviewer profiles with abstracts for higher-quality embeddings.
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

## Next steps for this session
1. Confirm HotCRP setup: is the 12-area taxonomy already implemented as HotCRP "Topics"?
2. Get a sample export of reviewer DBLP PIDs + area selections, and paper abstracts/keywords/areas (from HotCRP export) to work against real data.
3. Build the DBLP fetch + SPECTER2 embedding pipeline first, validate on a handful of known reviewers/papers before running full-scale.
