"""Build a SPECTER2 fingerprint (768-dim vector) for each accepted reviewer.

    python -m scripts.build_fingerprints --limit 10   # validate on a handful first
    python -m scripts.build_fingerprints              # full run

Each reviewer's fingerprint pools SPECTER2 embeddings of: one document per
DBLP publication from the most recent --years calendar years (default: 4, no
count cap unless --max-titles is set), using an IEEE/ACM abstract when cached,
and one document summarizing their declared
areas + keywords (weighted --area-weight relative to a single title).
Reviewers with no DBLP PID, or no titles in the --years window, fall back to
an area-profile-only fingerprint.

Fingerprints are cached in fingerprints.json (keyed by email). A versioned
content key covers the reviewer metadata, selected titles, model, and policy
flags, so unchanged entries avoid model work while changes rebuild
automatically. A transient DBLP failure creates or retains an area-only
fallback marked for retry; it cannot become permanently "complete". DBLP
titles reuse the same caches as main.py (dblp_cache.json / dblp_pubs_cache.json).

As a sanity check, the script prints each spot-check reviewer's top-5 nearest
neighbors by fingerprint cosine similarity.
"""

from __future__ import annotations

from reviewer_match.paths import assignment_path, cache_path, curated_path, input_path, report_path

import argparse
import csv
import hashlib
import json
import sys

import numpy as np
import requests
import torch

from reviewer_match import fingerprint as fp
from reviewer_match import paper_matching as pm
from reviewer_match import specter2_model
from reviewer_match.reserve_reviewers import DEFAULT_DATA
from reviewer_match.roster import load_roster, role_label
from reviewer_match.dblp import (
    fetch_titles_for_pids, load_cache, load_colleague_cache, normalise_doi,
    snapshot_gaps,
)

DEFAULT_CSV = input_path("HPCA'27 PC Member Acceptance Form (Responses) - Form Responses 1.csv")
DEFAULT_CACHE = cache_path("dblp_cache.json")
DEFAULT_COLLEAGUE_CACHE = input_path("dblp_pubs_cache.json")
DEFAULT_SNAPSHOT_CACHE = cache_path("dblp_snapshot_cache.json")
DEFAULT_FINGERPRINT_CACHE = cache_path("fingerprints.json")
DEFAULT_METADATA_CACHE = cache_path("reviewer_publications.json")
DEFAULT_ABSTRACT_CACHE = cache_path("publication_abstracts.json")
DEFAULT_PUBLICATION_EXCLUSIONS = curated_path("publication_exclusions.csv")
DEFAULT_FINGERPRINT_SOURCE_OVERRIDES = curated_path("fingerprint_source_overrides.csv")
FINGERPRINT_SCHEMA_VERSION = 3
FINGERPRINT_SOURCES = ("own-papers", "area-only")

# Always spot-checked if present in the current run; falls back to the first
# few reviewers processed when empty.
SPOT_CHECK_EMAILS: list[str] = []


def load_publication_exclusions(path: str) -> dict[str, set[str]]:
    """Load per-email DOI exclusions from an optional hand-maintained CSV."""
    try:
        f = open(path, newline="", encoding="utf-8")
    except FileNotFoundError:
        return {}
    exclusions: dict[str, set[str]] = {}
    with f:
        reader = csv.DictReader(f)
        required = {"email", "doi"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path}: missing required column(s): {', '.join(sorted(missing))}"
            )
        for line_number, row in enumerate(reader, 2):
            email = (row.get("email") or "").strip().lower()
            raw_doi = (row.get("doi") or "").strip()
            if not email and not raw_doi and not (row.get("note") or "").strip():
                continue
            if not email or not raw_doi:
                raise ValueError(f"{path}:{line_number}: email and doi are required")
            doi = normalise_doi(raw_doi)
            if doi is None:
                raise ValueError(f"{path}:{line_number}: invalid DOI {raw_doi!r}")
            exclusions.setdefault(email, set()).add(doi)
    return exclusions


def apply_publication_exclusions(
    email: str,
    publications: list[tuple[int, str, str, str, str]],
    exclusions: dict[str, set[str]],
) -> tuple[list[tuple[int, str, str, str, str]], set[str]]:
    """Remove publications whose DOI is excluded for `email`."""
    excluded_dois = exclusions.get(email.lower(), set())
    matched = {pub[2] for pub in publications if pub[2] in excluded_dois}
    return [pub for pub in publications if pub[2] not in excluded_dois], matched


def load_fingerprint_source_overrides(path: str) -> dict[str, str]:
    """Load the hand-maintained email -> {own-papers, area-only} override.

    A reviewer whose DBLP page is a merged/wrong identity (see
    outputs/reports/dblp_identity_audit.md) has no trustworthy DBLP
    publication list at all -- unlike publication_exclusions.csv, which
    prunes specific bad DOIs from an otherwise-kept page, this forces the
    reviewer's normal DBLP-derived titles out of their fingerprint entirely,
    in favor of either their own submitted-paper abstracts (`own-papers`,
    see paper_matching.authored_paper_documents) or the plain area-profile
    fallback that already exists for a reviewer with no DBLP page at all
    (`area-only`). Missing file is not an error -- most runs have nobody to
    override.
    """
    try:
        f = open(path, newline="", encoding="utf-8")
    except FileNotFoundError:
        return {}
    overrides: dict[str, str] = {}
    with f:
        reader = csv.DictReader(f)
        required = {"email", "source"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing required column(s): {', '.join(sorted(missing))}")
        for line_number, row in enumerate(reader, 2):
            email = (row.get("email") or "").strip().lower()
            source = (row.get("source") or "").strip().lower()
            if not email and not source and not (row.get("note") or "").strip():
                continue
            if not email or not source:
                raise ValueError(f"{path}:{line_number}: email and source are required")
            if source not in FINGERPRINT_SOURCES:
                raise ValueError(
                    f"{path}:{line_number}: source must be one of {FINGERPRINT_SOURCES}, got {source!r}"
                )
            overrides[email] = source
    return overrides


def fingerprint_key(r, publications, *, years, max_titles, area_weight, source="dblp") -> str:
    """Content and policy key for a reviewer fingerprint.

    `source` is omitted from the payload for the (overwhelming majority)
    default "dblp" case, so a plain full run's keys -- and therefore its
    cache -- stay byte-identical to before this parameter existed. It's only
    included for an overridden reviewer, both because `publications` itself
    already differs for them in the ordinary case and to correctly force a
    rebuild in the edge case where an area-only override happens to produce
    the same (empty) publications list DBLP already would have.
    """
    payload = {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "model": specter2_model.BASE_MODEL,
        "adapter": specter2_model.PROXIMITY_ADAPTER,
        "email": r.email,
        "pid": r.pid,
        "primary": r.primary,
        "secondary": r.secondary,
        "tertiary": r.tertiary,
        "keywords": r.keywords,
        "years": years,
        "max_titles": max_titles,
        "area_weight": area_weight,
        "publications": publications,
    }
    if source != "dblp":
        payload["fingerprint_source"] = source
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=None, help=f"path to the roster (default: {DEFAULT_CSV}; per --role otherwise)")
    parser.add_argument(
        "--data", default=DEFAULT_DATA,
        help=f"HotCRP paper export, for --role reserve area derivation (default: {DEFAULT_DATA})",
    )
    parser.add_argument(
        "--role", choices=("reviewer", "area-chair", "reserve", "trc"), default="reviewer",
        help="acceptance-form schema to load (default: reviewer)",
    )
    parser.add_argument("--cache", default=DEFAULT_CACHE, help="path to the writable DBLP title cache")
    parser.add_argument(
        "--colleague-cache", default=DEFAULT_COLLEAGUE_CACHE,
        help="path to the colleague's read-only pre-built DBLP cache"
    )
    parser.add_argument(
        "--snapshot-cache", default=DEFAULT_SNAPSHOT_CACHE,
        help=f"local DBLP dump cache, used only for PIDs the other caches lack "
             f"(default: {DEFAULT_SNAPSHOT_CACHE})"
    )
    parser.add_argument(
        "--fingerprint-cache", default=DEFAULT_FINGERPRINT_CACHE,
        help="path to the writable fingerprint cache"
    )
    parser.add_argument(
        "--publication-metadata-cache", default=DEFAULT_METADATA_CACHE,
        help="PID-keyed DBLP DOI metadata produced by enrich_publications.py"
    )
    parser.add_argument(
        "--abstract-cache", default=DEFAULT_ABSTRACT_CACHE,
        help="DOI-keyed abstract cache produced by enrich_publications.py"
    )
    parser.add_argument(
        "--publication-exclusions", default=DEFAULT_PUBLICATION_EXCLUSIONS,
        help="optional CSV of per-email publication DOIs to exclude"
    )
    parser.add_argument(
        "--fingerprint-source-overrides", default=DEFAULT_FINGERPRINT_SOURCE_OVERRIDES,
        help="optional CSV forcing specific reviewers off DBLP entirely, onto their own "
             "submitted-paper abstracts or an area-only fingerprint (see "
             "load_fingerprint_source_overrides)"
    )
    parser.add_argument(
        "--emails", default=None, metavar="PATH",
        help="restrict to one address per line (# comments allowed) instead of the whole "
             "roster; addresses not present in the current --role's roster are skipped. "
             "For cheaply re-embedding a handful of reviewers instead of the whole pool."
    )
    parser.add_argument(
        "--no-abstracts", action="store_true",
        help="build a reproducible title-only baseline even if abstracts are cached"
    )
    parser.add_argument(
        "--years", type=int, default=4,
        help="DBLP titles from the most recent N calendar years (default: 4)"
    )
    parser.add_argument(
        "--max-titles", type=int, default=None,
        help="optional cap on most-recent titles pooled per reviewer (default: no cap, use --years alone)"
    )
    parser.add_argument(
        "--area-weight", type=float, default=1.0,
        help="weight of the area/keyword document relative to one title document (default: 1.0)"
    )
    parser.add_argument("--limit", type=int, default=None, help="only process the first N reviewers")
    parser.add_argument(
        "--delay", type=float, default=3.0,
        help="base seconds between live DBLP fetches; actual delay is delay ± 50%% jitter"
    )
    parser.add_argument("--device", default="cuda", help="torch device for SPECTER2 (default: cuda)")
    args = parser.parse_args()

    if args.years <= 0:
        parser.error("--years must be greater than 0")
    if args.max_titles is not None and args.max_titles < 0:
        parser.error("--max-titles must be 0 or greater")
    if args.area_weight <= 0:
        parser.error("--area-weight must be greater than 0")
    if args.delay < 0:
        parser.error("--delay must be non-negative")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARN: cuda requested but not available, falling back to cpu", file=sys.stderr)
        args.device = "cpu"

    reviewers = load_roster(args.role, args.csv, args.data)
    role = role_label(args.role)
    if args.limit is not None:
        reviewers = reviewers[: args.limit]
    if args.emails is not None:
        with open(args.emails, encoding="utf-8") as f:
            wanted = {
                line.strip().lower() for line in f
                if line.strip() and not line.strip().startswith("#")
            }
        reviewers = [r for r in reviewers if r.email in wanted]
        missing = wanted - {r.email for r in reviewers}
        if missing:
            print(
                f"  {len(missing)} --emails address(es) not in this {role} roster, skipped: "
                f"{', '.join(sorted(missing))}",
                file=sys.stderr,
            )

    fp_cache = fp.load_fingerprint_cache(args.fingerprint_cache)
    publication_metadata = fp.load_fingerprint_cache(args.publication_metadata_cache)
    abstract_cache = fp.load_fingerprint_cache(args.abstract_cache)
    try:
        publication_exclusions = load_publication_exclusions(args.publication_exclusions)
        fingerprint_source_overrides = load_fingerprint_source_overrides(
            args.fingerprint_source_overrides
        )
    except ValueError as exc:
        parser.error(str(exc))

    # Resolve titles for every PID-backed reviewer. Cache hits are cheap, and
    # doing this before the fingerprint freshness check lets DBLP cache edits
    # participate in the content key.
    write_cache = load_cache(args.cache)
    readonly_cache = load_colleague_cache(args.colleague_cache)
    # The local DBLP dump backs up both caches for anyone they don't cover, so
    # a full roster no longer needs the network. load_colleague_cache reads the
    # snapshot's rich format and normalises it exactly as it does the
    # colleague's.
    snapshot = snapshot_gaps(
        load_colleague_cache(args.snapshot_cache), readonly_cache, write_cache
    )
    if snapshot:
        print(f"Snapshot supplies {len(snapshot)} PID(s) neither cache has "
              f"({args.snapshot_cache})", file=sys.stderr)
        readonly_cache = {**snapshot, **readonly_cache}
    titles_by_pid: dict[str, list[tuple[int, str]]] = {}
    failed_pids: set[str] = set()
    fetch_errors = 0

    with_pid = [r for r in reviewers if r.pid]
    if with_pid:
        def on_result(pid: str, titles: list[tuple[int, str]], source: str) -> None:
            titles_by_pid[pid] = titles

        def on_error(pid: str, exc: Exception) -> None:
            nonlocal fetch_errors
            fetch_errors += 1
            failed_pids.add(pid)
            print(f"  WARN: DBLP fetch failed for pid={pid}: {exc}", file=sys.stderr)

        session = requests.Session()
        fetch_titles_for_pids(
            list(dict.fromkeys(r.pid for r in with_pid)),
            years=args.years,
            session=session,
            write_cache=write_cache,
            readonly_cache=readonly_cache,
            cache_path=args.cache,
            delay=args.delay,
            on_result=on_result,
            on_error=on_error,
        )

    selected_by_email = {
        r.email: fp.select_titles(titles_by_pid.get(r.pid, []), args.max_titles)
        for r in reviewers
    }

    active_emails = {r.email for r in reviewers}
    active_overrides = {
        email: source for email, source in fingerprint_source_overrides.items()
        if email in active_emails
    }
    # area-only bypasses DBLP outright, even for a reviewer whose PID still
    # resolves to real titles -- their DBLP page is the thing distrusted, not
    # just its absence. Forcing the input empty here reuses the existing
    # no-titles fallback below rather than adding a second code path.
    for email, source in active_overrides.items():
        if source == "area-only":
            selected_by_email[email] = []

    own_paper_emails = {e for e, source in active_overrides.items() if source == "own-papers"}
    own_paper_docs: dict[str, list[tuple[int, str, str, str, str]]] = {}
    if own_paper_emails:
        with open(args.data, encoding="utf-8") as f:
            papers = json.load(f)
        own_paper_docs = pm.authored_paper_documents(papers, own_paper_emails)
        for email in sorted(own_paper_emails - set(own_paper_docs)):
            print(
                f"  WARN: {email} is overridden to own-papers but authored no submission "
                f"with an abstract; falling back to area-only",
                file=sys.stderr,
            )

    def title_key(year: int, title: str) -> tuple[int, str]:
        return int(year), title.lower().rstrip(". ")

    doi_by_pid_title: dict[str, dict[tuple[int, str], str]] = {}
    for pid, entry in publication_metadata.items():
        doi_by_pid_title[pid] = {
            title_key(record["year"], record["title"]): (record.get("doi") or "").lower()
            for record in entry.get("records", [])
            if record.get("year") is not None and record.get("title")
        }

    publications_by_email: dict[str, list[tuple[int, str, str, str, str]]] = {}
    matched_exclusions: dict[str, set[str]] = {}
    for r in reviewers:
        if r.email in own_paper_emails:
            publications = list(own_paper_docs.get(r.email, []))
        else:
            publications = []
            for year, title in selected_by_email[r.email]:
                doi = doi_by_pid_title.get(r.pid, {}).get(title_key(year, title), "")
                entry = abstract_cache.get(doi, {}) if doi and not args.no_abstracts else {}
                abstract = entry.get("abstract", "") if entry.get("status") == "found" else ""
                source = entry.get("source", "") if abstract else ""
                publications.append((year, title, doi, abstract, source))
        publications, matched = apply_publication_exclusions(
            r.email, publications, publication_exclusions
        )
        publications_by_email[r.email] = publications
        matched_exclusions[r.email] = matched

    for email in sorted(active_emails & set(publication_exclusions)):
        for doi in sorted(publication_exclusions[email] - matched_exclusions[email]):
            print(
                f"WARN: publication exclusion for {email} did not match a selected DOI: {doi}",
                file=sys.stderr,
            )

    keys = {
        r.email: fingerprint_key(
            r, publications_by_email[r.email], years=args.years,
            max_titles=args.max_titles, area_weight=args.area_weight,
            source=active_overrides.get(r.email, "dblp"),
        )
        for r in reviewers
    }

    pending = []
    for r in reviewers:
        entry = fp_cache.get(r.email)
        fetch_failed = bool(r.pid and r.pid in failed_pids)
        if fetch_failed and entry is not None:
            entry["dblp_fetch_complete"] = False
            continue
        if entry is None or entry.get("fingerprint_key") != keys[r.email] or not entry.get("dblp_fetch_complete", True):
            pending.append(r)

    print(
        f"Loaded {len(reviewers)} accepted {role}; {len(pending)} need fingerprints "
        f"({len(reviewers) - len(pending)} already cached).",
        file=sys.stderr,
    )

    # --- 2. Load SPECTER2 once, build every pending reviewer's input docs,
    #        encode in one batched pass, then pool per reviewer ---------------
    area_only = 0
    no_content: list = []
    if pending:
        print(f"Loading SPECTER2 on {args.device}...", file=sys.stderr)
        tokenizer, model = specter2_model.load_model(device=args.device)

        flat_texts: list[str] = []
        flat_owner: list[int] = []  # index into `pending`
        per_reviewer_weights: list[list[float]] = [[] for _ in pending]
        n_titles_used = [0] * len(pending)

        for i, r in enumerate(pending):
            selected = publications_by_email[r.email]
            n_titles_used[i] = len(selected)

            for _, title, _, abstract, _ in selected:
                flat_texts.append(fp.publication_doc_text(tokenizer, title, abstract))
                flat_owner.append(i)
                per_reviewer_weights[i].append(1.0)

            # Only when there is something to say. A reviewer with no declared
            # areas and no keywords would otherwise contribute a bare separator
            # at full --area-weight: the identical meaningless vector for every
            # such person, pulling their fingerprints toward a common point.
            if r.primary or r.secondary or r.tertiary or r.keywords:
                flat_texts.append(fp.area_profile_text(tokenizer, r))
                flat_owner.append(i)
                per_reviewer_weights[i].append(args.area_weight)

        print(f"Encoding {len(flat_texts)} documents for {len(pending)} reviewers...", file=sys.stderr)
        vectors = specter2_model.encode_texts(flat_texts, tokenizer, model)

        offsets = [0] * (len(pending) + 1)
        for owner in flat_owner:
            offsets[owner + 1] += 1
        for i in range(len(pending)):
            offsets[i + 1] += offsets[i]

        for i, r in enumerate(pending):
            reviewer_vectors = vectors[offsets[i] : offsets[i + 1]]
            reviewer_weights = per_reviewer_weights[i]
            # No publications and nothing declared: there is no such thing as a
            # fingerprint for this person. Leave them out rather than caching a
            # zero vector that would silently score against every paper.
            if not reviewer_weights:
                no_content.append(r)
                # A previous run may have cached a usable vector before the
                # roster or publication evidence changed. Confirmed absence of
                # all content invalidates that vector; leaving it in the cache
                # would make the warning below false and keep the researcher
                # eligible for assignment with stale evidence.
                fp_cache.pop(r.email, None)
                continue
            fingerprint_vec = fp.pool(reviewer_vectors, reviewer_weights)
            if n_titles_used[i] == 0:
                area_only += 1
            fp_cache[r.email] = {
                "vector": fingerprint_vec.tolist(),
                "n_titles": n_titles_used[i],
                "n_abstracts": sum(bool(pub[3]) for pub in publications_by_email[r.email]),
                "n_excluded": len(matched_exclusions[r.email]),
                "has_pid": bool(r.pid),
                "fingerprint_key": keys[r.email],
                "schema_version": FINGERPRINT_SCHEMA_VERSION,
                "dblp_fetch_complete": not bool(r.pid and r.pid in failed_pids),
                "fingerprint_source": active_overrides.get(r.email, "dblp"),
            }

        fp.save_fingerprint_cache(fp_cache, args.fingerprint_cache)
    elif failed_pids:
        # Persist retry state without changing the last known-good vectors.
        fp.save_fingerprint_cache(fp_cache, args.fingerprint_cache)

    print(
        f"\nDone. computed={len(pending)} area_only_fallback={area_only} "
        f"dblp_fetch_errors={fetch_errors} abstracts_used="
        f"{sum(sum(bool(pub[3]) for pub in publications_by_email[r.email]) for r in reviewers)} "
        f"publications_excluded={sum(len(matched_exclusions[r.email]) for r in reviewers)}",
        file=sys.stderr,
    )
    if no_content:
        names = ", ".join(f"{r.name} <{r.email}>" for r in no_content)
        print(
            f"WARNING: {len(no_content)} with neither publications nor declared "
            f"areas have no fingerprint and cannot be matched to any paper: {names}",
            file=sys.stderr,
        )

    # --- 4. Sanity check: nearest neighbors for spot-check reviewers --------
    email_to_reviewer = {r.email: r for r in reviewers}
    emails_with_vectors = [r.email for r in reviewers if r.email in fp_cache]
    if len(emails_with_vectors) < 2:
        return 0

    matrix = np.array([fp_cache[e]["vector"] for e in emails_with_vectors], dtype=np.float32)
    index_by_email = {e: i for i, e in enumerate(emails_with_vectors)}

    spot_checks = [e for e in SPOT_CHECK_EMAILS if e in index_by_email]
    for e in emails_with_vectors:
        if len(spot_checks) >= 3:
            break
        if e not in spot_checks:
            spot_checks.append(e)

    for email in spot_checks:
        r = email_to_reviewer[email]
        idx = index_by_email[email]
        sims = matrix @ matrix[idx]
        order = np.argsort(-sims)
        top = [i for i in order if i != idx][:5]
        print(f"\n=== Nearest neighbors for {r.name} <{email}> ({r.primary})")
        for rank, i in enumerate(top, 1):
            other = email_to_reviewer[emails_with_vectors[i]]
            print(f"  {rank}. {sims[i]:.3f}  {other.name} <{other.email}>  ({other.primary})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
