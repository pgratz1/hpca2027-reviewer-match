# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

This is the **HPCA 2027 reviewer–paper matching pipeline** — a data project to automate reviewer-to-paper assignment for the HPCA 2027 conference PC. It is currently **pre-implementation**: the only contents are the input data and the design brief. There is no source code, no git repo, no build/test tooling yet. The first substantive work is building the pipeline described below.

Read `hpca2027-matching-brief.md` in full before starting implementation work — it is the authoritative spec and records design decisions and still-open questions.

## Files

- `hpca2027-matching-brief.md` — project brief / spec. The source of truth for goals and design.
- `HPCA'27 PC Member Acceptance Form (Responses) - Form Responses 1.csv` — Google Forms export of PC-member responses (**691 records**). Filename contains an apostrophe and spaces; quote it in shell commands. Parse with a real CSV reader (Python `csv`), not `awk`/`cut`/`wc -l` — fields contain embedded commas, quoted `""` escapes, **and embedded newlines** (the file is 767 physical lines but only 691 logical records).

### CSV shape

One row per invited PC member. The key column is **`PC membership`**, which has three values:
- `No, I am unable to accept` — 241 declines; all other fields blank. Filter these out.
- `Yes, I accept as a full PC member` — 237.
- `Yes, I accept as a light PC member` — 213.

So **450 accepted** members with area data, split into two load tiers: **light PC members should get a smaller paper load** than full members — carry this distinction through to the optimizer's per-reviewer load constraints.

Relevant columns for the pipeline: HotCRP email, DBLP link (or literal `none`), affiliation, and **primary / secondary / tertiary area** plus free-text primary-area keywords. Note: **48 of the 450 accepted members have no DBLP link** (blank or `none`) — the pipeline needs a fallback for building their reviewer profile (e.g., area gate only, or match by keywords/affiliation).

### Area taxonomy (13 areas)

Reviewers and papers both select from this fixed HPCA taxonomy; area overlap is the hard eligibility gate:

Compilers and programming models · Datacenters/parallel architectures/systems · Domain-Specific Accelerators / Reconfigurable Computing · GPUs · Interconnection networks · ML architectures · Memory systems · Microarchitecture · Near data processing · Quantum Computing · Reliability/fault tolerance/emerging technologies · Security · Tools and Analysis

## Intended architecture (from the brief)

The scoring pipeline is a **filter-then-rank** design, not a blended score:

1. **Area gate (hard constraint).** A reviewer is eligible for a paper only if `{reviewer primary, secondary} ∩ {paper primary, secondary} ≠ ∅`. Area is *not* mixed into the numeric score — only used to gate the candidate set.
2. **DBLP fetch.** For each reviewer, pull the last 10 publication titles from `https://dblp.org/pid/{PID}.xml`, year-descending. PIDs come from the CSV's DBLP link column; no author disambiguation needed. Reuse this fetch for a COI cross-check (coauthor lists vs. HotCRP's self-declared COIs).
3. **Embedding.** SPECTER2 (`allenai/specter2`), purpose-built for scientific-paper similarity; fall back to `all-mpnet-base-v2` if SPECTER2 setup is troublesome.
4. **Score.** Within each area-eligible pair, cosine similarity between the reviewer's embedded titles and the paper's embedded abstract+keywords.
5. **Output.** A reviewer × paper affinity table over eligible pairs only, fed to the assignment step (ideally HotCRP's native min-cost-max-flow autoassigner via bulk import; a standalone `networkx`/`pulp` optimizer only if HotCRP can't ingest the scores).

**Non-goal:** no LLM-as-judge for the core affinity score — it's an embeddings + filtering problem. A local LLM is reserved for auxiliary tasks only (edge-case triage, human-readable rationale, parsing irregular DBLP entries).

## Suggested stack (per brief)

Python 3.11+, `requests` (DBLP XML), `sentence-transformers` / HF `transformers` (SPECTER2), `numpy`/`scipy` (cosine at scale), `networkx`/`pulp` (optimizer if needed). Runs locally on an RTX 4090 — the full PC-scale job is a few hundred reviewers × few hundred papers, seconds of GPU time, no cloud API.

## Open questions to resolve during implementation

These are unresolved in the brief; confirm before committing to an approach:
- Is the 13-area taxonomy already implemented as HotCRP "Topics"? If so, the area gate and final assignment may already live in HotCRP and this project only injects the cosine score.
- Title-only reviewer embeddings (DBLP has no abstracts) vs. enriching via Semantic Scholar by DOI/title.
- Whether primary vs. secondary area gets any soft weighting downstream (e.g., optimizer tie-breaks).
