"""The HotCRP user export as the authority on who is currently on the PC.

People selected for the PC -- full, light or reserve -- are sometimes removed
again. The acceptance form and the reserve roster are both snapshots of who was
*invited*; only the HotCRP user export says who is on the committee now. This
module reads that export so the roster loaders can drop anyone HotCRP no longer
marks `pc`, and so `audit_pc_roster.py` can report both directions of the
disagreement.

Matching a roster row to an account is the whole difficulty. Keying on email
alone is wrong far more often than it is right: people accept the invitation
from one address and hold their HotCRP account under another, and on the current
export twelve of the sixteen roster rows whose own address is not `pc` turn out
to be sitting PC members reachable under a second address. So a row is matched
by email, then by exact name tokens, then by email local part, and only a row
that fails all three is treated as removed.

Every index here is exact. Fuzzy comparison is available (`compare_names`, used
by `find_duplicate_accounts.py`) but deliberately not used for the membership
test: a false match merely keeps someone who was already on the roster, while a
false miss silently removes a sitting PC member, so the two error directions are
not remotely equal in cost.
"""

from __future__ import annotations

from .paths import assignment_path, cache_path, curated_path, input_path, report_path

import csv
import os
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from itertools import permutations

from .dblp import name_tokens

DEFAULT_PCINFO = input_path("hpca2027-pcinfo.csv")

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


def token_set(text: str) -> frozenset[str]:
    """The comparable name tokens of a free-text name, initials dropped.

    `name_tokens` drops bare initials, which is what makes "J. Smith" and
    "John Smith" comparable at all.
    """
    return frozenset(canonical(t) for t in name_tokens(text))


def on_pc(roles: str) -> bool:
    """Is this account on the PC? HotCRP writes a chair's roles as "chair pc"."""
    return "pc" in roles.split()


def registrable_domain(email: str) -> str:
    """The last two labels of the mail domain, so a.edu meets mail.a.edu."""
    _, _, domain = email.partition("@")
    labels = fold(domain).split()
    return ".".join(labels[-2:]) if len(labels) >= 2 else ""


def local_part(email: str) -> str:
    """The folded local part of an address, or "" if there isn't one.

    A roster cell with no "@" is a name someone typed into an email box, not an
    address; returning "" keeps it out of the local-part index, where it would
    otherwise match on the whole cell.
    """
    if "@" not in email:
        return ""
    return fold(email.partition("@")[0])


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
        self.tags = row.get("tags", "").strip()
        self.collaborators = row.get("collaborators", "").strip()
        self.disabled = row.get("disabled", "").strip().lower() == "yes"

        self.name = " ".join(p for p in (self.given, self.family) if p)
        self.tokens = token_set(self.name)
        self.ordered = (ordered_tokens(self.given), ordered_tokens(self.family))
        self.local = fold(self.email.partition("@")[0])
        self.domain = registrable_domain(self.email)
        self.affil = fold(self.affiliation)

    @property
    def is_pc(self) -> bool:
        """On the PC and able to log in.

        A disabled account cannot review, so it does not keep its holder on the
        roster -- but it is still indexed by email, so the audit can tell
        "disabled" apart from "no account at all".
        """
        return on_pc(self.roles) and not self.disabled

    @property
    def is_area_chair(self) -> bool:
        """Tagged an area chair in HotCRP, and so barred from reviewing.

        The tag is written `~~area-chairs` -- HotCRP's twiddle form, for a tag
        only the chairs can see -- so the test is `endswith`, never `==`, which
        would match nobody. Other twiddle tags exist on this export
        (`~~rr-batch2`), so the leading marker cannot be assumed away.

        A hypothetical `~~sub-area-chairs` would match too. That errs towards
        excluding someone from the reviewer pool, which is the safe direction:
        a withheld reviewer costs one slot out of thousands, while a missed
        chair breaks the rule that an area chair reviews nothing.
        """
        return any(t.endswith("area-chairs") for t in self.tags.split())


# ---------------------------------------------------------------------------
# Name comparison (fuzzy; used by find_duplicate_accounts.py, not by the gate)
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
    way, 1 for a near miss -- the ladder `find_duplicate_accounts.classify`
    reads.
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
# The membership index
# ---------------------------------------------------------------------------

class PcIndex:
    """The HotCRP accounts of one export, indexed for exact membership lookup.

    `by_email` covers every account, so a caller can ask what became of a
    particular address. The other two indexes cover only accounts that are
    currently on the PC, since they exist to answer one question: does this
    roster row belong to somebody who is on the committee, under any address?
    """

    def __init__(self, accounts: list[Account], path: str) -> None:
        self.path = path
        self.accounts = accounts
        self.by_email: dict[str, Account] = {}
        self.by_tokens: dict[frozenset[str], list[Account]] = {}
        self.by_local: dict[str, list[Account]] = {}
        self.pc_accounts: list[Account] = [a for a in accounts if a.is_pc]
        for acct in accounts:
            email = acct.email.lower()
            if email:
                self.by_email.setdefault(email, acct)
            if not acct.is_pc:
                continue
            if acct.tokens:
                self.by_tokens.setdefault(acct.tokens, []).append(acct)
            if acct.local:
                self.by_local.setdefault(acct.local, []).append(acct)

    def _first(self, candidates: list[Account]) -> Account:
        """Pick deterministically when a name or local part covers two accounts.

        A collision resolves toward keeping the person, which is the harmless
        direction: the worst case is that the roster stays as it is today.
        """
        return min(candidates, key=lambda a: a.email.lower())

    def match(self, email: str, first: str = "", last: str = "") -> tuple[Account | None, str]:
        """The PC account behind this roster row, and how it was found.

        Returns (account, "email" | "name" | "local"), or (None, "") when the
        person is not on the PC under any address this can see.
        """
        email = (email or "").strip().lower()
        acct = self.by_email.get(email)
        if acct is not None and acct.is_pc:
            return acct, "email"

        # A roster row whose email cell holds a name rather than an address is a
        # form-entry slip, not a resignation -- read the cell as a name too.
        name = " ".join(p for p in (first, last) if p).strip()
        if "@" not in email:
            name = f"{name} {email}".strip()
        tokens = token_set(name)
        if tokens and tokens in self.by_tokens:
            return self._first(self.by_tokens[tokens]), "name"

        local = local_part(email)
        if local and local in self.by_local:
            return self._first(self.by_local[local]), "local"

        return None, ""


def hotcrp_email_for(email: str, acct: Account | None, how: str) -> str:
    """The address to write into a HotCRP bulk upload for this roster row.

    `match()` may find the account by name or local part rather than email --
    accepting from one address while HotCRP knows them under another is
    ordinary here (see module docstring). Writing the roster row's own
    address into a bulk-assignment CSV in that case fails on upload, since
    HotCRP checks the exact address: only the account's own email is one it
    will recognize as `pc`.
    """
    return acct.email.lower() if acct is not None and how != "email" else email


def collapse_by_hotcrp_email(records: list, ranks: dict[str, object] | None = None):
    """One record per HotCRP account: the highest-ranked among duplicates.

    Someone who resubmits a roster form under a second real address (not a
    typo of the first, an address `match()` resolves to a different account
    than their own) produces two roster rows for one person. Left alone, the
    assignment stage sees two distinct candidates who happen to share a
    `hotcrp_email`, and can hand the same real person two reviews on one
    paper -- invisible in the roster, since the two rows never compare equal
    on `email`.

    Every record must expose `.email` and `.hotcrp_email`. `ranks[record.email]`
    breaks ties -- for a roster with resubmission timestamps, pass those, so
    the same "latest wins" policy applies as for same-email duplicates
    (`_latest_rows_by_email`). Without `ranks`, the first record encountered
    for each account wins.

    Returns (survivors, collapsed), where `collapsed` is
    `[(kept_email, dropped_email), ...]` for callers that want to report it.
    """
    best: dict[str, tuple] = {}
    for i, r in enumerate(records):
        rank = ranks[r.email] if ranks is not None else 0
        prior = best.get(r.hotcrp_email)
        if prior is None or rank > prior[0]:
            best[r.hotcrp_email] = (rank, i, r)
    kept_by_hotcrp = {h: v[2] for h, v in best.items()}
    collapsed = [
        (kept_by_hotcrp[r.hotcrp_email].email, r.email)
        for r in records
        if kept_by_hotcrp[r.hotcrp_email] is not r
    ]
    survivors = sorted(kept_by_hotcrp.values(), key=lambda r: best[r.hotcrp_email][1])
    return survivors, collapsed


_CACHE: dict[tuple[str, int, int], PcIndex] = {}


def load_pc_accounts(path: str = DEFAULT_PCINFO) -> PcIndex:
    """Load the HotCRP user export and index its PC accounts.

    Raises FileNotFoundError naming the remedy if the export is absent, and
    ValueError if it contains no PC account at all. The second check is the
    important one: a truncated or half-downloaded export parses perfectly and
    marks nobody as `pc`, which would prune the entire roster and report every
    paper as unstaffed. Refusing is the only safe reading of an export that
    disagrees with reality that violently.

    Results are cached on (path, mtime, size) because a single `make` run loads
    three rosters in one process and the export is 5,600 rows.
    """
    try:
        stat = os.stat(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{path}: HotCRP user export not found -- download it from HotCRP "
            f"(Users -> select all -> download -> user information), or pass "
            f"--no-pc-check to skip the membership check"
        ) from None

    key = (os.path.abspath(path), stat.st_mtime_ns, stat.st_size)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    with open(path, newline="", encoding="utf-8-sig") as f:
        accounts = [Account(row) for row in csv.DictReader(f)]
    index = PcIndex(accounts, path)
    if not index.pc_accounts:
        raise ValueError(
            f"{path}: no account has the 'pc' role -- the export looks truncated "
            f"or is the wrong file; re-download it, or pass --no-pc-check to skip "
            f"the membership check"
        )
    _CACHE[key] = index
    return index


def report_pruned(dropped: list[str], kept: int, label: str, index: PcIndex,
                  stream=None) -> None:
    """One summary line for a roster the membership check shrank.

    One line, not one per person: every script in the pipeline loads a roster,
    so a per-person warning would print hundreds of times a run. The PC-account
    count rides along because it is the cheapest tripwire for a bad export --
    if that number collapses, the next line will be a mass removal.

    `stream` is resolved at call time, not bound as a default: a default would
    capture whatever sys.stderr was at import and quietly ignore any later
    redirect_stderr, including the ones the tests rely on.
    """
    stream = sys.stderr if stream is None else stream
    total = kept + len(dropped)
    if not dropped:
        print(
            f"HotCRP membership check: all {total} {label} are on the PC "
            f"({len(index.pc_accounts)} PC accounts in {index.path})",
            file=stream,
        )
        return
    shown = ", ".join(sorted(dropped)[:5])
    if len(dropped) > 5:
        shown += f", +{len(dropped) - 5} more"
    verb = "is" if len(dropped) == 1 else "are"
    print(
        f"Warning: {len(dropped)} of {total} {label} {verb} not on the HotCRP PC "
        f"and {'was' if len(dropped) == 1 else 'were'} dropped ({shown}); "
        f"run `make pc-roster` for the report",
        file=stream,
    )
