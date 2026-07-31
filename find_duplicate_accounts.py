"""Find HotCRP accounts that look like the same person under two addresses.

    python find_duplicate_accounts.py
    python find_duplicate_accounts.py --pcinfo hpca2027-pcinfo.csv --out duplicate_accounts.csv
    python find_duplicate_accounts.py --min-confidence medium
    python find_duplicate_accounts.py --pc-only
    python find_duplicate_accounts.py --both-pc

One person registering twice -- once with an institutional address, once with
gmail -- is routine in HotCRP, and it splits their conflicts, their topics and
their review load across two accounts. This lists the candidates so a human can
merge them.

`--pc-only` keeps the pairs with a PC member on at least one side. Those are the
ones that cost something before the deadline: a duplicated PC member reviews
under one account while their conflicts sit on the other, and an author account
shadowing a PC member hides that person's conflicts entirely. Duplicate author
accounts are worth cleaning up too, but they cannot misroute a review.

`--both-pc` narrows that to the pairs where the same person sits on the PC twice.
That is the case the assignment pipeline sees directly: two PC rows mean two
review loads and two half-populated conflict sets for one human, so the matcher
can hand them a paper they are conflicted with under their other address.

Output is one row per candidate *pair*, not per cluster, because the evidence
differs pair by pair: three accounts sharing one common two-token name are
usually two people, not one, and only a pair-level verdict can say which two of
the three to merge.

Nothing here decides anything. `confidence` ranks the pairs for a human
reviewing them in order:

  high    -- a shared ORCID, or an identical name backed by a matching
             affiliation or email
  review  -- a shared ORCID under two unrelated names: one of the two records
             has someone else's ORCID in it, which is worth knowing either way
  medium  -- an identical or reordered name with nothing contradicting it
  low     -- a near-miss spelling, or a name match whose two accounts claim
             *different* ORCIDs, which is the signature of two distinct people
             who happen to share a common name

The last case is why conflicting ORCIDs demote rather than promote: on a roster
this size an exact name collision is more often two people than one.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from itertools import combinations, permutations

from dblp import name_tokens

DEFAULT_PCINFO = "hpca2027-pcinfo.csv"
DEFAULT_OUT = "duplicate_accounts.csv"

# A token pair this close counts as one spelling of one name, but only for
# tokens long enough that a single edit is unlikely to land on a different name.
# "Zhang"/"Wang" and "Chen"/"Chan" are short and distinct; a long surname such
# as "Thornquist" mistyped is still Thornquist.
DEFAULT_TOKEN_RATIO = 0.85
MIN_FUZZY_TOKEN_LEN = 6

# Common English short forms. Deliberately small: a wrong entry here silently
# merges two people, and the ORCID and affiliation evidence carries most pairs
# without it.
NICKNAMES = {
    "rich": "richard", "rick": "richard", "dick": "richard",
    "bill": "william", "will": "william", "billy": "william",
    "bob": "robert", "rob": "robert", "bobby": "robert",
    "mike": "michael", "mick": "michael",
    "dave": "david", "tom": "thomas", "tommy": "thomas",
    "steve": "stephen", "chris": "christopher", "jim": "james",
    "joe": "joseph", "dan": "daniel", "danny": "daniel",
    "matt": "matthew", "nick": "nicholas", "alex": "alexander",
    "andy": "andrew", "drew": "andrew", "ben": "benjamin",
    "tony": "anthony", "ted": "theodore", "sam": "samuel",
    "ed": "edward", "eddie": "edward", "jeff": "jeffrey",
    "greg": "gregory", "ken": "kenneth", "ron": "ronald",
    "pete": "peter", "phil": "philip", "vince": "vincent",
    "zach": "zachary", "josh": "joshua", "nate": "nathaniel",
    "kate": "katherine", "cathy": "catherine", "liz": "elizabeth",
    "beth": "elizabeth", "sue": "susan", "jen": "jennifer",
    "maggie": "margaret", "peggy": "margaret", "abby": "abigail",
}

FIELDS = (
    "confidence", "reason", "evidence",
    "name_a", "email_a", "affiliation_a", "orcid_a", "country_a", "roles_a",
    "name_b", "email_b", "affiliation_b", "orcid_b", "country_b", "roles_b",
)

CONFIDENCE_ORDER = {"high": 0, "review": 1, "medium": 2, "low": 3}


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def fold(text: str) -> str:
    """Accent-folded, lowercased, punctuation-collapsed form of a free-text field."""
    folded = unicodedata.normalize("NFKD", text)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", folded.lower()).split())


def canonical(token: str) -> str:
    """A name token with its common short form expanded, so Rich meets Richard."""
    return NICKNAMES.get(token, token)


def ordered_tokens(text: str) -> tuple[str, ...]:
    """Name tokens in the order written, bare initials dropped.

    The order is what separates a genuinely reordered name from the same name
    plus a middle initial: "Solvig H. Hallberg" and "Solvig Hallberg" are one
    spelling of one name once the initial goes, and calling that a reordering
    would be wrong about which field moved.
    """
    return tuple(canonical(t) for t in fold(text).split() if len(t) > 1)


def on_pc(roles: str) -> bool:
    """Is this account on the PC? HotCRP writes a chair's roles as "chair pc"."""
    return "pc" in roles.split()


def registrable_domain(email: str) -> str:
    """The last two labels of the mail domain, so a.edu meets mail.a.edu."""
    _, _, domain = email.partition("@")
    labels = fold(domain).split()
    return ".".join(labels[-2:]) if len(labels) >= 2 else ""


class Account:
    """One row of the HotCRP user export, with everything a comparison needs."""

    def __init__(self, row: dict[str, str]) -> None:
        self.given = row.get("given_name", "").strip()
        self.family = row.get("family_name", "").strip()
        self.email = row.get("email", "").strip()
        self.affiliation = row.get("affiliation", "").strip()
        self.orcid = row.get("orcid", "").strip()
        self.country = row.get("country", "").strip()
        self.roles = row.get("roles", "").strip()

        self.name = " ".join(p for p in (self.given, self.family) if p)
        # name_tokens drops bare initials, which is what makes "J. Smith" and
        # "John Smith" comparable at all.
        self.tokens = frozenset(canonical(t) for t in name_tokens(self.name))
        self.ordered = (ordered_tokens(self.given), ordered_tokens(self.family))
        self.local = fold(self.email.partition("@")[0])
        self.domain = registrable_domain(self.email)
        self.affil = fold(self.affiliation)


# ---------------------------------------------------------------------------
# Name comparison
# ---------------------------------------------------------------------------

def near_token(a: str, b: str, ratio: float) -> bool:
    """Are these two tokens one name spelled two ways?

    Only tokens long enough to survive an edit and still be the same name are
    compared fuzzily. Short ones must match exactly: pairs like "Cen"/"Chen",
    "Xi"/"Xin" and "Kai"/"Kani" are one edit apart yet are different names, and
    every rule loose enough to join them buries the real duplicates in hundreds
    of pairs that only share a common family name.
    """
    if a == b:
        return True
    if len(a) < MIN_FUZZY_TOKEN_LEN or len(b) < MIN_FUZZY_TOKEN_LEN:
        return False
    return SequenceMatcher(None, a, b).ratio() >= ratio


def tokens_cover(small: frozenset[str], large: frozenset[str], ratio: float) -> bool:
    """Is every token of `small` present in `large`, allowing one spelling drift?

    True for a name that gained a middle name between registrations.
    """
    drift = 0
    for token in small:
        if token in large:
            continue
        if any(near_token(token, other, ratio) for other in large):
            drift += 1
            if drift > 1:
                return False
            continue
        return False
    return True


def glued(a: frozenset[str], b: frozenset[str]) -> bool:
    """Does one side write the whole name as a single token? ("meilingokonkwo")

    HotCRP's family_name field collects this often enough to be worth a rule.
    """
    for one, other in ((a, b), (b, a)):
        if len(one) == 1 and 1 < len(other) <= 3:
            solo = next(iter(one))
            if any("".join(order) == solo for order in permutations(other)):
                return True
    return False


def compare_names(a: Account, b: Account, ratio: float) -> tuple[str, int]:
    """(reason, strength) for two names; strength 0 means unrelated.

    Strength is 3 for an identical name, 2 for the same name written another
    way, 1 for a near miss -- the ladder `classify` reads.
    """
    if not a.tokens or not b.tokens:
        return "", 0
    if a.tokens == b.tokens:
        if a.ordered == b.ordered:
            return "exact_name", 3
        return "reordered_name", 2
    if glued(a.tokens, b.tokens):
        return "glued_name", 2
    small, large = sorted((a.tokens, b.tokens), key=len)
    # A one-token name is a subset of every name sharing that token, which would
    # make a doubled single-token name a variant of all forty people who share
    # that family name. Two tokens is the least that can distinguish a person,
    # so it is the least that can match one.
    if 2 <= len(small) < len(large) and tokens_cover(small, large, ratio):
        return "name_variant", 2
    if len(a.tokens) == len(b.tokens) and tokens_cover(a.tokens, b.tokens, ratio):
        return "similar_name", 1
    return "", 0


# ---------------------------------------------------------------------------
# Pair classification
# ---------------------------------------------------------------------------

def classify(a: Account, b: Account, ratio: float) -> tuple[str, str, str] | None:
    """(confidence, reason, evidence) for one candidate pair, or None."""
    reason, strength = compare_names(a, b, ratio)

    orcid_match = bool(a.orcid) and a.orcid == b.orcid
    orcid_conflict = bool(a.orcid) and bool(b.orcid) and a.orcid != b.orcid
    affil_match = bool(a.affil) and bool(b.affil) and (
        a.affil == b.affil or a.affil.startswith(b.affil) or b.affil.startswith(a.affil)
    )
    domain_match = bool(a.domain) and a.domain == b.domain
    local_match = bool(a.local) and bool(b.local) and (
        a.local == b.local
        or SequenceMatcher(None, a.local, b.local).ratio() >= 0.8
    )

    if not strength and not orcid_match:
        return None

    evidence = ",".join(
        label for label, present in (
            ("same_orcid", orcid_match),
            ("conflicting_orcid", orcid_conflict),
            ("same_affiliation", affil_match),
            ("same_email_domain", domain_match),
            ("similar_email_local", local_match),
        ) if present
    )

    if orcid_match:
        if strength:
            return "high", reason, evidence
        # Two unrelated names on one ORCID: not a duplicate, but one of the two
        # records is wrong, and a wrong ORCID misroutes conflicts just as badly.
        return "review", "shared_orcid_different_names", evidence
    if orcid_conflict:
        return "low", reason, evidence

    supported = affil_match or domain_match or local_match
    if strength == 3:
        return ("high" if supported else "medium"), reason, evidence
    if strength == 2:
        return ("medium" if supported else "low"), reason, evidence
    return "low", reason, evidence


def candidate_pairs(accounts: list[Account]) -> set[tuple[int, int]]:
    """Index-pairs worth comparing: those sharing a name token or an ORCID.

    Blocking, not a shortcut -- comparing all 15.8M pairs of a 5,600-row roster
    would spend nearly all of it on names with no letter in common.
    """
    by_token: dict[str, list[int]] = defaultdict(list)
    by_orcid: dict[str, list[int]] = defaultdict(list)
    for i, acct in enumerate(accounts):
        for token in acct.tokens:
            by_token[token].append(i)
        # A prefix block catches drift in the token itself, which a shared-token
        # block by definition cannot.
        for token in acct.tokens:
            if len(token) >= MIN_FUZZY_TOKEN_LEN:
                by_token["~" + token[:4]].append(i)
        if acct.orcid:
            by_orcid[acct.orcid].append(i)

    pairs: set[tuple[int, int]] = set()
    for block in list(by_token.values()) + list(by_orcid.values()):
        for i, j in combinations(sorted(set(block)), 2):
            pairs.add((i, j))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pcinfo", default=DEFAULT_PCINFO, help="HotCRP user export CSV")
    parser.add_argument("--out", default=DEFAULT_OUT, help="where to write the pairs")
    parser.add_argument(
        "--min-confidence", choices=sorted(CONFIDENCE_ORDER), default="low",
        help="drop pairs ranked below this (default: low, i.e. keep everything)",
    )
    parser.add_argument(
        "--token-ratio", type=float, default=DEFAULT_TOKEN_RATIO,
        help=f"similarity two long name tokens need (default: {DEFAULT_TOKEN_RATIO})",
    )
    parser.add_argument(
        "--pc-only", action="store_true",
        help="keep only pairs with a PC member on at least one side",
    )
    parser.add_argument(
        "--both-pc", action="store_true",
        help="keep only pairs with a PC member on both sides (implies --pc-only)",
    )
    args = parser.parse_args()

    with open(args.pcinfo, newline="", encoding="utf-8") as handle:
        accounts = [Account(row) for row in csv.DictReader(handle)]
    print(f"Read {len(accounts)} accounts from {args.pcinfo}", file=sys.stderr)

    pairs = candidate_pairs(accounts)
    print(f"Comparing {len(pairs)} candidate pairs", file=sys.stderr)

    cutoff = CONFIDENCE_ORDER[args.min_confidence]
    results = []
    for i, j in pairs:
        a, b = accounts[i], accounts[j]
        verdict = classify(a, b, args.token_ratio)
        if verdict is None:
            continue
        confidence, reason, evidence = verdict
        if CONFIDENCE_ORDER[confidence] > cutoff:
            continue
        pc_sides = on_pc(a.roles) + on_pc(b.roles)
        if args.both_pc and pc_sides < 2:
            continue
        if args.pc_only and not pc_sides:
            continue
        # Order the pair so the account carrying more information reads first;
        # that is usually the one to keep in a merge.
        if (bool(b.orcid), bool(b.affiliation), b.roles, b.email) > (
            bool(a.orcid), bool(a.affiliation), a.roles, a.email
        ):
            a, b = b, a
        results.append({
            "confidence": confidence, "reason": reason, "evidence": evidence,
            "name_a": a.name, "email_a": a.email, "affiliation_a": a.affiliation,
            "orcid_a": a.orcid, "country_a": a.country, "roles_a": a.roles,
            "name_b": b.name, "email_b": b.email, "affiliation_b": b.affiliation,
            "orcid_b": b.orcid, "country_b": b.country, "roles_b": b.roles,
        })

    results.sort(key=lambda r: (
        CONFIDENCE_ORDER[r["confidence"]], fold(r["name_a"]), r["email_a"], r["email_b"]
    ))

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDS))
        writer.writeheader()
        writer.writerows(results)

    counts: dict[str, int] = defaultdict(int)
    reasons: dict[str, int] = defaultdict(int)
    both_pc = one_pc = 0
    for row in results:
        counts[row["confidence"]] += 1
        reasons[row["reason"]] += 1
        pc_sides = on_pc(row["roles_a"]) + on_pc(row["roles_b"])
        both_pc += pc_sides == 2
        one_pc += pc_sides == 1
    print(f"Wrote {len(results)} candidate pairs to {args.out}", file=sys.stderr)
    for level in sorted(counts, key=lambda c: CONFIDENCE_ORDER[c]):
        print(f"  {level:7} {counts[level]}", file=sys.stderr)
    print("  by reason: " + ", ".join(
        f"{r}={n}" for r, n in sorted(reasons.items(), key=lambda kv: -kv[1])
    ), file=sys.stderr)
    # Two PC accounts split a review load; one PC account shadowed by an author
    # account splits that person's conflicts. Both need a merge, for different
    # reasons, so they are worth counting apart.
    print(f"  both sides PC: {both_pc}   one side PC: {one_pc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
