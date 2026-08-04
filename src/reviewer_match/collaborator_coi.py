"""Conflicts of interest from declared "collaborators", not derived from DBLP.

HotCRP's own Assignments upload page warns "Proposed reviewer X may conflict
with #N" using two signals this pipeline did not check before this module —
so the only place they were ever caught was that warning, easy to miss on a
6,000-row bulk upload:

  * A reviewer's declared `collaborators` field (any HotCRP account has one,
    not just PC members — `data/inputs/hpca2027-pcinfo.csv` is the full
    ~5,600-row user export, not a PC-only one) names one of the paper's
    authors, or a paper's author (if they hold any HotCRP account) declared
    the reviewer as theirs. Checked both ways, because the declaration is
    one-sided by construction — one person updating their profile does not
    update the other's.
  * The reviewer's own affiliation shares a significant word with a paper
    author's, on either side's plain affiliation string or an "All
    (Institution)" collaborators line (HotCRP's own shorthand for "everyone
    at this institution").

Only the name signal is a hard exclusion, the same direction as
`coauthor_coi`: a withheld reviewer costs one slot out of hundreds, a missed
conflict costs the review. The affiliation signal turned out too coarse to
hard-block on in practice — measured on this data, a plain shared-word
overlap touches 556 of 688 reviewers and 963 of 1156 papers, dominated by
generic words ("hong", "california", "chinese", "computing"), and a
document-frequency cutoff cannot separate those from a genuinely specific
match ("shenzhen" sits at a similar frequency to "california"). HotCRP's own
warning treats it the same way — "you may want to confirm all potential
conflicts", not an automatic block. `derive_conflicts` still computes both
signals; `hard_conflicts` is the name-only subset callers should exclude on,
and `scripts/audit_collaborator_conflicts.py` reports everything, affiliation
included, for a human to skim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .coauthor_coi import MIN_MATCH_TOKENS, names_match
from .paper_matching import own_paper_conflicts
from .pc_membership import PcIndex, fold, token_set

# Generic words that would match almost any two institutions, not one
# specific place -- sharing only these is not evidence of anything.
INSTITUTION_STOPWORDS = {
    "university", "univ", "institute", "institution", "college", "school",
    "of", "the", "and", "for", "department", "dept", "laboratory", "lab",
    "labs", "center", "centre", "national", "state", "technology",
    "technological", "science", "sciences", "research", "group", "inc",
    "corp", "corporation", "co", "ltd", "llc", "company", "international",
}

_TRAILING_PARENS_RE = re.compile(r"^(.*?)\s*\(([^()]*)\)\s*$")


def parse_collaborator_lines(text: str) -> list[tuple[str, str]]:
    """[(name or 'All', institution)] for each non-blank line of a `collaborators` field.

    HotCRP's own format: one collaborator per line, optionally
    "Name (Institution)"; "All (Institution)" means everyone there.
    """
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = _TRAILING_PARENS_RE.match(line)
        if match:
            name, institution = match.group(1).strip(), match.group(2).strip()
        else:
            name, institution = line, ""
        if name:
            lines.append((name, institution))
    return lines


def is_institution_wide(name: str) -> bool:
    """Is this collaborator line HotCRP's "All (Institution)" shorthand?"""
    return name.strip().lower() == "all"


def institution_tokens(text: str) -> frozenset[str]:
    """Significant words of an affiliation/institution string.

    Folded (accent/case/punctuation-insensitive, `pc_membership.fold`) and
    filtered against `INSTITUTION_STOPWORDS`; words of 2 letters or fewer are
    dropped too, since an abbreviation that short is rarely one institution's
    alone. What survives is meant to be broad, not precise — see the module
    docstring.
    """
    return frozenset(
        t for t in fold(text).split()
        if len(t) > 2 and t not in INSTITUTION_STOPWORDS
    )


@dataclass(frozen=True)
class CollaboratorConflict:
    """One reviewer, one paper, and the declared tie that conflicts them."""

    email: str
    kind: str          # 'name' | 'affiliation'
    direction: str      # who declared it -- see derive_conflicts
    author_name: str
    author_email: str
    author_affiliation: str
    evidence: str        # the matched name, or the shared institution word
    declared: str        # 'pc_conflicts' | 'own_paper' | '' -- '' is newly found


@dataclass
class _Profile:
    email: str
    name_tokens: frozenset[str]
    # Collaborator names this person declared (excluding "All" lines), each
    # with its own token set, for the "reviewer declared author" direction.
    declared_names: list[frozenset[str]]
    # Own affiliation tokens plus every "All (Institution)" institution this
    # person declared -- everything that makes *them* conflicted by place.
    institutions: frozenset[str]


def _profile(email: str, name: str, acct) -> _Profile:
    """A person's collaborator-COI-relevant signals, from their HotCRP account.

    `acct` is `None` when the roster row's `hotcrp_email` has no match in the
    export at all (should not happen for anyone already kept by the
    membership check, but handled the same defensive way `pc_membership`
    treats any unmatched address: signals nothing rather than raising).
    """
    declared_names: list[frozenset[str]] = []
    institutions: set[str] = set(institution_tokens(acct.affiliation) if acct else "")
    for collab_name, institution in parse_collaborator_lines(acct.collaborators if acct else ""):
        if is_institution_wide(collab_name):
            institutions |= institution_tokens(institution)
        else:
            tokens = token_set(collab_name)
            if len(tokens) >= MIN_MATCH_TOKENS:
                declared_names.append(tokens)
    return _Profile(
        email=email, name_tokens=token_set(name),
        declared_names=declared_names, institutions=frozenset(institutions),
    )


def build_index(people, pcinfo_index: PcIndex) -> list[_Profile]:
    """One `_Profile` per candidate, keyed off their HotCRP account.

    Looked up by `hotcrp_email` (see `pc_membership.hotcrp_email_for`), not
    the roster `email`: HotCRP's `collaborators`/`affiliation` fields live on
    the account, which for someone matched under a second address is not the
    roster row's own address.
    """
    return [
        _profile(person.email, person.name, pcinfo_index.by_email.get(person.hotcrp_email))
        for person in people
    ]


# A name match is higher-precision evidence than a shared affiliation word,
# so when a reviewer is conflicted through more than one author on the same
# paper, the report should show the name match, not whichever author's
# affiliation happened to be checked first.
_KIND_RANK = {"affiliation": 0, "name": 1}


def derive_conflicts(
    papers: list[dict], profiles: list[_Profile], pcinfo_index: PcIndex,
) -> dict[int, dict[str, CollaboratorConflict]]:
    """{paper pid: {reviewer email: CollaboratorConflict}} over `papers`.

    Checked per paper, per author, against every candidate: papers rarely
    have more than a handful of authors, so this stays cheap despite not
    building an inverted index the way `coauthor_coi` does for DBLP names.
    """
    derived: dict[int, dict[str, CollaboratorConflict]] = {}
    for paper in papers:
        pc_declared = {e.lower() for e in (paper.get("pc_conflicts") or {})}
        own = own_paper_conflicts(paper)
        found: dict[str, CollaboratorConflict] = {}
        for person in (paper.get("authors") or []) + (paper.get("contacts") or []):
            author_email = (person.get("email") or "").strip().lower()
            author_name = f"{person.get('given_name', '')} {person.get('family_name', '')}".strip()
            author_affiliation = (person.get("affiliation") or "").strip()
            author_name_tokens = token_set(author_name)
            author_acct = pcinfo_index.by_email.get(author_email) if author_email else None
            author_declared_names: list[frozenset[str]] = []
            author_institutions: set[str] = set(institution_tokens(author_affiliation))
            if author_acct is not None:
                author_institutions |= institution_tokens(author_acct.affiliation)
                for collab_name, institution in parse_collaborator_lines(author_acct.collaborators):
                    if is_institution_wide(collab_name):
                        author_institutions |= institution_tokens(institution)
                    else:
                        tokens = token_set(collab_name)
                        if len(tokens) >= MIN_MATCH_TOKENS:
                            author_declared_names.append(tokens)

            for profile in profiles:
                conflict = _match(
                    profile, author_name_tokens, author_declared_names, author_institutions,
                )
                if conflict is not None:
                    kind, direction, evidence = conflict
                    held = found.get(profile.email)
                    if held is not None and _KIND_RANK[held.kind] >= _KIND_RANK[kind]:
                        continue
                    found[profile.email] = CollaboratorConflict(
                        email=profile.email, kind=kind, direction=direction,
                        author_name=author_name, author_email=author_email,
                        author_affiliation=author_affiliation, evidence=evidence,
                        declared=("pc_conflicts" if profile.email in pc_declared
                                  else "own_paper" if profile.email in own else ""),
                    )
        if found:
            derived[paper["pid"]] = found
    return derived


def _match(
    profile: _Profile,
    author_name_tokens: frozenset[str],
    author_declared_names: list[frozenset[str]],
    author_institutions: frozenset[str],
) -> tuple[str, str, str] | None:
    """(kind, direction, evidence) for the first signal that fires, or None."""
    if author_name_tokens:
        for declared in profile.declared_names:
            if names_match(declared, author_name_tokens):
                return "name", "reviewer_declared", " ".join(sorted(declared))
    if profile.name_tokens:
        for declared in author_declared_names:
            if names_match(declared, profile.name_tokens):
                return "name", "author_declared", " ".join(sorted(declared))
    shared = profile.institutions & author_institutions
    if shared:
        return "affiliation", "institution", sorted(shared)[0]
    return None


def hard_conflicts(derived: dict[int, dict[str, CollaboratorConflict]]) -> dict[int, set[str]]:
    """{paper pid: reviewer emails}, name-matched conflicts only.

    The subset of `derive_conflicts`'s result callers should actually
    exclude on. Affiliation-overlap conflicts stay in the full result for
    `scripts/audit_collaborator_conflicts.py` to report, but are informational
    only -- see the module docstring for why they are not excluded here.
    """
    return {
        pid: {email for email, conflict in found.items() if conflict.kind == "name"}
        for pid, found in derived.items()
    }
