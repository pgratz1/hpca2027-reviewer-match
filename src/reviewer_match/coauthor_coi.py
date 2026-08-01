"""Conflicts of interest derived from DBLP co-authorship, not from declarations.

`pc_conflicts` is only as complete as the author COI sweep, and
`paper_matching.own_paper_conflicts` closes just the self-authorship hole. This
adds a third layer that needs nobody to have declared anything: a reviewer who
has co-authored a publication with one of a paper's authors inside the window is
conflicted with that paper.

It is meant to be *redundant* with the declared markings. Where it is not, that
is a conflict nobody recorded, which is the whole point of the report
`scripts/audit_coauthor_conflicts.py` writes.

Matching is on names, and deliberately errs towards firing. Not assigning a
paper over a name collision costs one reviewer slot out of a pool of hundreds;
assigning a real conflict costs the review. So:

  * `names_match` accepts an exact token-set match, and also a strict subset
    sharing at least MIN_MATCH_TOKENS tokens, so DBLP's "David Albert Wood"
    still meets HotCRP's "David Wood".
  * `dblp.name_tokens` has already dropped bare initials and DBLP's homonym
    suffix, so "Wei Zhang 0001" and "Wei Zhang 0025" are one name here. That
    over-matches on purpose.
  * A name with fewer than MIN_MATCH_TOKENS usable tokens is ignored outright.
    A lone surname is not evidence of anything, and matching on one would
    conflict a reviewer with every paper an unrelated namesake wrote.

What it cannot see is the more important half. A reviewer whose PID is missing
from the snapshot has no co-authors here and will silently pass every check, so
callers must report that coverage gap rather than read a clean result as clean —
see `coverage_gap`. Advisor/advisee, same-institution and funding ties stay
invisible either way; this does not replace the sweep.
"""

from __future__ import annotations

from .paths import assignment_path, cache_path, curated_path, input_path, report_path

import datetime
import json
import re
from collections import defaultdict
from dataclasses import dataclass

from .dblp import index_self_declared_pids
from .paper_matching import own_paper_conflicts
from .pc_membership import token_set

DEFAULT_COAUTHORS = cache_path("dblp_coauthors.json")
DEFAULT_AUTHOR_NAMES = cache_path("dblp_author_names.json")

# Calendar years of co-authorship that count as a conflict. The cache is built
# wider (build_dblp_snapshot_cache.DEFAULT_COAUTHOR_YEARS) so this can move
# without re-reading the 5 GB dump.
DEFAULT_COAUTHOR_YEARS = 5

# Below this many usable name tokens a match is a coincidence, not a person.
MIN_MATCH_TOKENS = 2

# DBLP's homonym discriminator: the trailing number in "Wei Zhang 0012" that
# says which Wei Zhang this is. `dblp.name_tokens` strips it, which is what lets
# a name match at all -- but thrown away entirely it collapses people DBLP has
# already told apart, and among these submissions "Wei Zhang" is 24 different
# researchers. `identity` keeps it so the two can be compared where both sides
# are known. Same pattern as dblp._HOMONYM_SUFFIX_RE, deliberately.
_SUFFIX_RE = re.compile(r"\s+(\d+)$")


def identity(name: str) -> str:
    """DBLP's homonym number for a name string, or "" for the unnumbered one.

    Compared instead of the raw string because two spellings of *one* person
    routinely differ -- "Jose Garcia"/"José García", "David A. Wood"/"David
    Wood" -- and string inequality there would read as two people. The suffix
    is the only part of the string that actually carries identity.
    """
    match = _SUFFIX_RE.search(name.strip())
    return match.group(1) if match else ""


@dataclass(frozen=True)
class CoauthorConflict:
    """One reviewer, one paper, and the co-authorship that conflicts them."""

    email: str            # the reviewer
    author_name: str      # the person on the paper they wrote with
    author_email: str     # that person's HotCRP address, "" if the paper lists none
    author_affiliation: str  # so a reader can judge whether it is the right person
    match: str            # 'exact' | 'partial'
    shared: int           # co-authored publications inside the window
    latest_year: int
    latest_title: str
    declared: str         # 'pc_conflicts' | 'own_paper' | '' — '' is newly uncovered


def load_coauthors(path: str = DEFAULT_COAUTHORS) -> dict[str, dict[str, list]]:
    """{pid: {co-author name: [[year, title], ...]}} as the snapshot build wrote it."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_author_names(path: str = DEFAULT_AUTHOR_NAMES) -> dict[str, list[str]]:
    """{pid: [DBLP name spelling, ...]} for PIDs submission authors declared."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def names_match(one: frozenset[str], other: frozenset[str]) -> str | None:
    """'exact', 'partial', or None for two comparable name-token sets.

    'partial' is a strict subset — one name written with a middle name the other
    spells out, or omits. Requiring the shared part to reach MIN_MATCH_TOKENS is
    what stops "Wei" alone from marrying two unrelated people.
    """
    if len(one) < MIN_MATCH_TOKENS or len(other) < MIN_MATCH_TOKENS:
        return None
    if one == other:
        return "exact"
    if (one < other or other < one) and len(one & other) >= MIN_MATCH_TOKENS:
        return "partial"
    return None


class CoauthorIndex:
    """Reviewers looked up by the names of the people they have written with.

    Inverted by token, because the direct comparison does not fit in the time
    available: ~40,000 co-author names against ~9,000 author slots is 360
    million subset tests. Indexing by token turns each lookup into the union of
    a few surname buckets, which is tens of candidates.
    """

    def __init__(self) -> None:
        # token -> {name tokens} -> {reviewer email: (shared, latest_year, latest_title)}
        self._by_token: dict[str, set[frozenset[str]]] = defaultdict(set)
        self._entries: dict[frozenset[str], dict[str, tuple[int, int, str]]] = defaultdict(dict)
        # Which DBLP homonyms of that name the reviewer actually wrote with.
        self._identities: dict[frozenset[str], dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self.covered: set[str] = set()      # reviewers with co-author data
        self.uncovered: set[str] = set()    # reviewers without: the blind spot

    def add(
        self, email: str, tokens: frozenset[str], evidence: tuple[int, int, str],
        identity_: str = "",
    ) -> None:
        if len(tokens) < MIN_MATCH_TOKENS:
            return
        held = self._entries[tokens].get(email)
        if held is not None:
            # Two DBLP spellings fold to one token set whenever they are the
            # same name under different homonym numbers. Add the counts and keep
            # the more recent paper: the count is evidence for a human to read,
            # and over-stating it errs the safe way.
            shared, year, title = evidence
            if held[1] > year:
                year, title = held[1], held[2]
            evidence = (held[0] + shared, year, title)
        self._entries[tokens][email] = evidence
        self._identities[tokens][email].add(identity_)
        for token in tokens:
            self._by_token[token].add(tokens)

    def lookup(
        self, tokens: frozenset[str], identities: set[str] | None = None
    ) -> dict[str, tuple[str, tuple[int, int, str]]]:
        """{reviewer email: (match, evidence)} for everyone who wrote with this name.

        `identities` are the DBLP homonym numbers of the person being looked up,
        known only when they declared a DBLP page for themselves. Given those, a
        reviewer who wrote with a *different* homonym of the same name is not
        conflicted — DBLP has already established they are different people, and
        collapsing them is what makes a common name conflict half the committee.

        Applied to exact name matches only. Across a partial match the two
        strings are different names, so their numbers are not comparable and the
        permissive reading stands.
        """
        if len(tokens) < MIN_MATCH_TOKENS:
            return {}
        found: dict[str, tuple[str, tuple[int, int, str]]] = {}
        seen: set[frozenset[str]] = set()
        for token in tokens:
            for candidate in self._by_token.get(token, ()):
                if candidate in seen:
                    continue
                seen.add(candidate)
                match = names_match(candidate, tokens)
                if match is None:
                    continue
                for email, evidence in self._entries[candidate].items():
                    if (
                        match == "exact"
                        and identities
                        and not (identities & self._identities[candidate][email])
                    ):
                        continue
                    # An exact match is better evidence than a partial one, and
                    # a reviewer can be reachable both ways at once.
                    if email not in found or (match == "exact" and found[email][0] == "partial"):
                        found[email] = (match, evidence)
        return found


def build_index(
    people,
    coauthors: dict[str, dict[str, list]],
    *,
    years: int = DEFAULT_COAUTHOR_YEARS,
    current_year: int | None = None,
) -> CoauthorIndex:
    """Index `people` (Reviewers or AreaChairs) by who they published with.

    Only co-authorships in the last `years` calendar years count, on
    `dblp.filter_by_years`' convention: the cutoff is
    `current_year - years + 1`, so five years in 2026 means 2022 onwards.
    """
    if years <= 0:
        raise ValueError("co-author window must be greater than 0 years")
    if current_year is None:
        current_year = datetime.date.today().year
    cutoff = current_year - years + 1

    index = CoauthorIndex()
    for person in people:
        shared = coauthors.get(person.pid) if person.pid else None
        if shared is None:
            index.uncovered.add(person.email)
            continue
        index.covered.add(person.email)
        own = token_set(person.name)
        for name, pubs in shared.items():
            windowed = [(int(y), t) for y, t in pubs if int(y) >= cutoff]
            if not windowed:
                continue
            tokens = token_set(name)
            # A reviewer is not their own co-author. The snapshot already drops
            # their known spellings; this catches one it did not know about.
            if not tokens or tokens == own:
                continue
            latest_year, latest_title = max(windowed)
            index.add(person.email, tokens, (len(windowed), latest_year, latest_title),
                      identity(name))
    return index


def paper_people(paper: dict, author_names: dict[str, list[str]], self_pids: dict[str, str]):
    """[(name tokens, display name, email, affiliation, identities)] for the paper's people.

    Authors and contacts, each under their HotCRP spelling *and* under every
    DBLP spelling of the PID they declared for themselves. Half the author slots
    carry such a link, and it is what lets a co-author DBLP knows as one name
    match an author HotCRP knows as another.

    `identities` are the DBLP homonym numbers of the spellings that declared
    link supplies, per name. It stays **empty when the author declared no DBLP
    page**, and an empty set means "unknown", never "the unnumbered one" — the
    caller must not narrow a match on it, because not knowing which Wei Zhang
    someone is cannot be evidence that they are not this one.
    """
    grouped: dict[tuple[frozenset[str], str], list] = {}
    for person in (paper.get("authors") or []) + (paper.get("contacts") or []):
        email = (person.get("email") or "").strip().lower()
        affiliation = (person.get("affiliation") or "").strip()
        display = f"{person.get('given_name', '')} {person.get('family_name', '')}".strip()
        pid = self_pids.get(email)
        declared = author_names.get(pid, []) if pid else []
        for spelling in ([display] if display else []) + list(declared):
            tokens = token_set(spelling)
            if not tokens:
                continue
            entry = grouped.setdefault(
                (tokens, email), [tokens, display or spelling, email, affiliation, set()]
            )
            # Only the declared DBLP spellings carry an identity; the HotCRP
            # name is just a name and says nothing about which homonym it is.
            if spelling in declared:
                entry[4].add(identity(spelling))
    return [tuple(v) for v in grouped.values()]


def derive_conflicts(
    papers: list[dict],
    index: CoauthorIndex,
    author_names: dict[str, list[str]],
    all_papers: list[dict] | None = None,
    *,
    use_identity: bool = True,
) -> dict[int, dict[str, CoauthorConflict]]:
    """{paper pid: {reviewer email: CoauthorConflict}} over `papers`.

    `all_papers` is the unfiltered export, used only to read authors' self-
    declared DBLP links: an author's link on a withdrawn paper still identifies
    the same person on the one under review. Defaults to `papers`.

    `use_identity=False` ignores DBLP's homonym numbering and treats every
    spelling of a name as one person — the older, blunter reading, kept so a run
    can be diffed against it.
    """
    self_pids = {
        email: counts.most_common(1)[0][0]
        for email, counts in index_self_declared_pids(all_papers or papers).items()
        if counts
    }

    derived: dict[int, dict[str, CoauthorConflict]] = {}
    for paper in papers:
        pc_declared = {e.lower() for e in (paper.get("pc_conflicts") or {})}
        own = own_paper_conflicts(paper)
        found: dict[str, CoauthorConflict] = {}
        for tokens, display, author_email, affil, ids in paper_people(
            paper, author_names, self_pids
        ):
            for email, (match, (shared, year, title)) in index.lookup(
                tokens, ids if use_identity else None
            ).items():
                held = found.get(email)
                # Keep the strongest single piece of evidence per reviewer: an
                # exact name match first, then the one with more shared papers.
                if held is not None and not (
                    (match == "exact" and held.match == "partial")
                    or (match == held.match and shared > held.shared)
                ):
                    continue
                found[email] = CoauthorConflict(
                    email=email,
                    author_name=display,
                    author_email=author_email,
                    author_affiliation=affil,
                    match=match,
                    shared=shared,
                    latest_year=year,
                    latest_title=title,
                    declared=("pc_conflicts" if email in pc_declared
                              else "own_paper" if email in own else ""),
                )
        if found:
            derived[paper["pid"]] = found
    return derived


def coverage_gap(index: CoauthorIndex, people) -> list:
    """The people this layer is blind to — no co-author data, so never conflicted.

    A false negative here is silent by construction: they simply pass every
    check. Callers report this; they do not treat an empty result as clean.
    """
    return [p for p in people if p.email in index.uncovered]
