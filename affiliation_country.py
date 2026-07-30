"""Resolve which country an institution sits in, for the per-paper region cap.

Country here means where the *institution* is, never anyone's nationality, and
Hong Kong, Macao, Taiwan and Singapore are their own ISO 3166-1 alpha-2 codes:
separate jurisdictions with separate communities, which is also how DBLP writes
its affiliation notes and how the ccTLDs in the HotCRP export fall. Nothing is
ever folded into another region by this module.

Four layers, first hit wins, and **nothing is guessed** — an affiliation none of
them place stays UNRESOLVED so the caller can report the coverage instead of
assuming:

1. `affiliation_countries.csv`, the hand-maintained layer, matched on the
   normalized affiliation string. A blank `country` cell is a to-do marker, not
   a decision, and never masks a later layer.
2. DBLP's `<note type="affiliation">`, which names the country outright ("Ant
   Research, Beijing, China"). A profile often carries several, and **their
   order means nothing** -- a Tsinghua professor's notes list UC Santa Barbara
   first -- so the note is chosen by how well it matches the affiliation the
   person gave HotCRP, and if none of them matches, this layer declines rather
   than picking a former employer's country.
3. A country or region *name* in the affiliation string. Adjectives are
   deliberately not names — "Chinese" would place "The Chinese University of
   Hong Kong" in CN, which is exactly the error a region cap must not make. Both
   DBLP and HotCRP do write "Hong Kong ..., China" and "University of Macau,
   China", so a region name always outranks the sovereign state it sits in.
4. The email's ccTLD, ignoring the TLDs sold generically (.com, .io, .ai, .co
   and friends), which say nothing about where anyone is.
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

DEFAULT_COUNTRIES = "affiliation_countries.csv"
DEFAULT_DBLP_AFFILIATIONS = "dblp_affiliations.json"
DEFAULT_PROFILE_CACHE = "dblp_profile_cache.json"

# Not a country. Distinct from None so a caller can tell "we looked and could
# not place it" from "we never asked".
UNRESOLVED = ""

# Country and region names as they appear in affiliation strings, normalized the
# same way the strings are. Adjectives ("chinese", "korean", "german") are
# excluded on purpose: they modify institution names far more often than they
# name a location.
#
# Names that are also common institution or place words are excluded too --
# "georgia" (Georgia Tech), "jordan", "turkey", "chad", "guinea". Layers 1 and 4
# still place those institutions; guessing from a name this ambiguous would
# produce confident wrong answers, which is worse than UNRESOLVED.
COUNTRY_NAMES: dict[str, str] = {
    "argentina": "AR",
    "australia": "AU",
    "austria": "AT",
    "bangladesh": "BD",
    "belgium": "BE",
    "brazil": "BR",
    "brasil": "BR",
    "canada": "CA",
    "chile": "CL",
    "china": "CN",
    "p r china": "CN",
    "pr china": "CN",
    "peoples republic of china": "CN",
    "colombia": "CO",
    "croatia": "HR",
    "cyprus": "CY",
    "czech republic": "CZ",
    "czechia": "CZ",
    "denmark": "DK",
    "egypt": "EG",
    "estonia": "EE",
    "finland": "FI",
    "france": "FR",
    "germany": "DE",
    "deutschland": "DE",
    "greece": "GR",
    "hong kong": "HK",
    "hong kong sar": "HK",
    "hungary": "HU",
    "iceland": "IS",
    "india": "IN",
    "indonesia": "ID",
    "iran": "IR",
    "ireland": "IE",
    "israel": "IL",
    "italy": "IT",
    "italia": "IT",
    "japan": "JP",
    "kenya": "KE",
    "korea": "KR",
    "south korea": "KR",
    "republic of korea": "KR",
    "north korea": "KP",
    "luxembourg": "LU",
    "macao": "MO",
    "macau": "MO",
    "malaysia": "MY",
    "mexico": "MX",
    "netherlands": "NL",
    "the netherlands": "NL",
    "holland": "NL",
    "new zealand": "NZ",
    "nigeria": "NG",
    "norway": "NO",
    "pakistan": "PK",
    "peru": "PE",
    "philippines": "PH",
    "poland": "PL",
    "portugal": "PT",
    "qatar": "QA",
    "romania": "RO",
    "russia": "RU",
    "russian federation": "RU",
    "saudi arabia": "SA",
    "serbia": "RS",
    "singapore": "SG",
    "slovakia": "SK",
    "slovenia": "SI",
    "south africa": "ZA",
    "spain": "ES",
    "espana": "ES",
    "sweden": "SE",
    "switzerland": "CH",
    "taiwan": "TW",
    "thailand": "TH",
    "tunisia": "TN",
    "ukraine": "UA",
    "united arab emirates": "AE",
    "uae": "AE",
    "united kingdom": "GB",
    "uk": "GB",
    "great britain": "GB",
    "england": "GB",
    "scotland": "GB",
    "wales": "GB",
    "northern ireland": "GB",
    "united states": "US",
    "united states of america": "US",
    "usa": "US",
    "u s a": "US",
    "vietnam": "VN",
    "viet nam": "VN",
}

# ccTLD -> ISO alpha-2. Identical apart from the handful where the two registries
# disagree; .uk is the one that actually matters here.
_CCTLD_EXCEPTIONS = {"uk": "GB"}

# TLDs that are sold to anyone anywhere. A .io or .ai address says nothing about
# where its owner is, and .com/.edu/.org are the largest ambiguous block in the
# HotCRP export, so none of them may resolve a country.
GENERIC_TLDS = frozenset(
    """
    com edu org net gov mil int info biz name pro app dev xyz online site tech
    io ai co me tv cc ly to gg sh st fm am
    """.split()
)

# Every code this module can produce. A region cap naming a code outside this set
# would silently be a class of zero reviewers, so callers validate against it.
ISO_ALPHA2: frozenset[str] = frozenset(COUNTRY_NAMES.values()) | frozenset(
    _CCTLD_EXCEPTIONS.values()
)

# A region that outranks the sovereign state it sits in. DBLP and HotCRP both
# write "Hong Kong University of Science and Technology, ..., China" and
# "University of Macau, China"; the region is the more specific statement of
# where the institution is, and this pipeline never folds these into CN.
_OUTRANKS: dict[str, frozenset[str]] = {
    "HK": frozenset({"CN"}),
    "MO": frozenset({"CN"}),
    "TW": frozenset({"CN"}),
}

# Words shared by half the world's institutions, dropped before a DBLP note is
# matched against the affiliation someone typed into HotCRP. Deliberately a
# separate list from resolve_trc_members._AFFILIATION_STOPWORDS, which serves
# identity tie-breaking: this one keeps place names, because "Hong Kong" and
# "Beijing" are exactly what tells two notes apart here.
_NOTE_STOPWORDS = frozenset(
    """
    university universite universitat universidad univ college institute institut
    school department dept faculty laboratory laboratories lab labs center centre
    research academy academic technology technological science sciences
    engineering computer computing national state of the and for at in a
    """.split()
)

_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class CountryLayers:
    """The two file-backed layers, loaded once and shared by every lookup."""

    overrides: dict[str, str] = field(default_factory=dict)  # normalized affil -> ISO
    dblp_by_pid: dict[str, list[str]] = field(default_factory=dict)  # PID -> notes


def normalize_affiliation(text: str) -> str:
    """Casefolded, accent-folded, punctuation-free form of an affiliation string.

    Only trivially-equivalent spellings are meant to collide: "The Hong Kong
    University of Science and Technology" and "Hong Kong Univ. of Science &
    Technology" stay distinct, because merging them would be a guess about two
    strings a human has not looked at yet.
    """
    folded = unicodedata.normalize("NFKD", text or "")
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    folded = _PUNCT_RE.sub(" ", folded.casefold())
    folded = _WS_RE.sub(" ", folded).strip()
    if folded.startswith("the "):
        folded = folded[4:]
    return folded


def is_country_code(code: str) -> bool:
    """True if `code` is an ISO alpha-2 code this module can actually resolve."""
    return code in ISO_ALPHA2


def load_affiliation_countries(path: str = DEFAULT_COUNTRIES) -> dict[str, str]:
    """Load the hand-maintained layer: normalized affiliation -> ISO alpha-2.

    Only the `country` column is read. The generator writes its guesses into
    `suggested`, never into `country`, so this layer stays purely what a human
    decided — otherwise a machine guess would outrank the DBLP note beneath it.

    A blank `country` cell is a to-do marker and is skipped. A non-blank value
    that isn't a code this module knows fails loudly with the offending row,
    since a silent skip would make the override mysteriously not take effect.

    Returns {} if the file doesn't exist.
    """
    try:
        f = open(path, newline="", encoding="utf-8")
    except FileNotFoundError:
        return {}
    out: dict[str, str] = {}
    with f:
        for row in csv.DictReader(f):
            affiliation = normalize_affiliation(row.get("affiliation") or "")
            code = (row.get("country") or "").strip().upper()
            if not affiliation or not code:
                continue
            if not is_country_code(code):
                raise ValueError(
                    f"{path}: {affiliation!r} has country {code!r}, "
                    f"which is not an ISO alpha-2 code this module knows"
                )
            out[affiliation] = code
    return out


def load_layers(
    countries_path: str = DEFAULT_COUNTRIES,
    dblp_path: str = DEFAULT_DBLP_AFFILIATIONS,
    profile_cache: str = DEFAULT_PROFILE_CACHE,
) -> CountryLayers:
    """Load the hand layer and the DBLP affiliation notes.

    The offline snapshot (`dblp_affiliations.json`, near-complete for the roster)
    is layered over the network profile cache rather than replacing it, so PIDs
    fetched live but absent from the dump are still covered.
    """
    dblp_by_pid: dict[str, list[str]] = {}
    for path, extract in ((profile_cache, _profile_affiliations), (dblp_path, _plain_affiliations)):
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
        except FileNotFoundError:
            continue
        for pid, value in raw.items():
            notes = extract(value)
            if notes:
                dblp_by_pid[pid] = notes
    return CountryLayers(load_affiliation_countries(countries_path), dblp_by_pid)


def _profile_affiliations(value) -> list[str]:
    """Affiliation notes out of a dblp_profile_cache.json entry."""
    if isinstance(value, dict):
        return [str(a) for a in (value.get("affiliations") or [])]
    return []


def _plain_affiliations(value) -> list[str]:
    """Affiliation notes out of a dblp_affiliations.json entry (a bare list)."""
    if isinstance(value, list):
        return [str(a) for a in value]
    return []


def _note_tokens(text: str) -> frozenset[str]:
    """Distinctive tokens of an affiliation, for matching one note to another."""
    return frozenset(
        t for t in normalize_affiliation(text).split() if t not in _NOTE_STOPWORDS
    )


def best_matching_note(affiliations: Sequence[str], affiliation: str) -> str:
    """The DBLP note that best matches `affiliation`, or "" if none does.

    DBLP's note order carries no meaning — a profile can list a former employer
    abroad ahead of the current institution — so the person's own HotCRP
    affiliation picks the note. No shared distinctive token means no match, and
    no match means this layer declines: a former employer's country is a wrong
    answer, which is worse than no answer.
    """
    wanted = _note_tokens(affiliation)
    if not wanted:
        return ""
    best, best_overlap = "", 0
    for note in affiliations:
        overlap = len(_note_tokens(note) & wanted)
        if overlap > best_overlap:
            best, best_overlap = note, overlap
    return best


def country_from_dblp(affiliations: Sequence[str], affiliation: str = "") -> str:
    """ISO code from the DBLP affiliation note matching `affiliation`.

    The whole note is scanned for a country or region name rather than just its
    trailing field: DBLP writes "Hong Kong University of Science and Technology,
    Department of Electronic and Computer Engineering, China", where the trailing
    field is the one thing that must not decide the answer.
    """
    note = best_matching_note(affiliations, affiliation)
    return country_from_text(note) if note else UNRESOLVED


def country_from_text(affiliation: str) -> str:
    """ISO code from a country or region name written in the affiliation string.

    Matching is on whole tokens, so "Indiana" is not "India" and "Chinese" is not
    "China". The longest name wins, and among equal-length names the rightmost
    one does: affiliations are written most-specific-last, so the trailing name
    is the location and an earlier one is usually part of the institution's title
    ("Korea Advanced Institute of Science and Technology, Daejeon, Korea").
    """
    tokens = normalize_affiliation(affiliation).split()
    if not tokens:
        return UNRESOLVED
    matches: dict[str, tuple[int, int]] = {}
    for name, code in COUNTRY_NAMES.items():
        parts = name.split()
        n = len(parts)
        for i in range(len(tokens) - n + 1):
            if tokens[i : i + n] == parts:
                key = (n, i)
                if code not in matches or key > matches[code]:
                    matches[code] = key
    if not matches:
        return UNRESOLVED
    # "Hong Kong ..., China" names both; the region is the specific answer, so
    # drop whatever it outranks before comparing on length and position.
    outranked = {c for code in matches for c in _OUTRANKS.get(code, ())}
    live = {code: key for code, key in matches.items() if code not in outranked} or matches
    return max(live, key=lambda code: live[code])


def country_from_email(email: str) -> str:
    """ISO code from the address's ccTLD, or UNRESOLVED for a generic TLD."""
    _, _, domain = (email or "").strip().lower().partition("@")
    if not domain:
        return UNRESOLVED
    tld = domain.rsplit(".", 1)[-1]
    if not tld or tld in GENERIC_TLDS or not tld.isalpha() or len(tld) != 2:
        return UNRESOLVED
    code = _CCTLD_EXCEPTIONS.get(tld, tld.upper())
    return code if is_country_code(code) else UNRESOLVED


def resolve_country(
    affiliation: str,
    email: str = "",
    dblp_affiliations: Sequence[str] = (),
    overrides: Mapping[str, str] | None = None,
) -> tuple[str, str]:
    """(ISO alpha-2 or UNRESOLVED, which layer answered).

    The layer name is returned so callers can report coverage per layer and so
    the generator can show its work; see the module docstring for the order.
    """
    if overrides:
        code = overrides.get(normalize_affiliation(affiliation))
        if code:
            return code, "hand"
    code = country_from_dblp(dblp_affiliations, affiliation)
    if code:
        return code, "dblp"
    code = country_from_text(affiliation)
    if code:
        return code, "affiliation"
    code = country_from_email(email)
    if code:
        return code, "email"
    return UNRESOLVED, "unresolved"


def author_country(author: Mapping, layers: CountryLayers) -> tuple[str, str]:
    """Country of a HotCRP author/contact record.

    Authors carry no DBLP PID, so only the hand, affiliation-text and ccTLD
    layers can answer for them.
    """
    return resolve_country(
        author.get("affiliation") or "",
        author.get("email") or "",
        (),
        layers.overrides,
    )


def reviewer_country(reviewer, layers: CountryLayers) -> tuple[str, str]:
    """Country of a Reviewer/AreaChair record, using its DBLP PID when it has one."""
    pid = getattr(reviewer, "pid", None)
    return resolve_country(
        getattr(reviewer, "affiliation", "") or "",
        getattr(reviewer, "email", "") or "",
        layers.dblp_by_pid.get(pid or "", ()),
        layers.overrides,
    )
