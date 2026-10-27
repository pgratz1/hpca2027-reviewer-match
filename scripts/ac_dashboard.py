"""Area-chair dashboards: one page per track of the papers still under review.

    # Every track, plus all.html for the PC chairs (needs HOTCRP_SITE and
    # HOTCRP_TOKEN in .env; the discussion opened on --since):
    python -m scripts.ac_dashboard --since 2026-10-10

    # Replay the last run's API snapshot without touching the site:
    python -m scripts.ac_dashboard --since 2026-10-10 --offline

    # Two tracks into a scratch directory, with a pinned timestamp:
    python -m scripts.ac_dashboard --since 2026-10-10 --tracks 5,7 \\
        --out-dir /tmp/x --generated-at '2026-10-16 09:00'

    # Portrait pages, through a different Chrome binary:
    python -m scripts.ac_dashboard --since 2026-10-10 --orientation portrait --chrome chromium

One PDF per track, each in its own directory --
`outputs/reports/ac_dashboard/track_05/track_05_dana-fox.pdf` -- holding
every paper tagged `~~track_NN` that is still under review (submitted, no
decision), and `all/all.pdf` with every track for the PC chairs. The chair
emails each track's PDF to that track's area chair; a track page carries only
its own papers, so no chair sees another track's reviewers or scores.

**The phase.** The papers left to settle are tagged `pc-discuss`
(`--discuss-tag`). One advances only with **more champions than
anti-champions**, so with no champion it is rejected and a tie rejects too.
A vote is the review form's Champion / Anti-champion checkboxes
(`ChaAntCha`; 1 = Champion, 2 = Anti-champion), counted over submitted PC
reviews; a review ticking both counts on neither side and is flagged. Papers
tagged `likely-accept` / `online-accept` or `likely-reject` / `online-reject`
are settled, and are listed only so that a **contrary** vote -- an
anti-champion on a settled accept, a champion on a settled reject -- marks
the paper "reopen?". Tags pointing two ways (pc-discuss with online-reject)
or no phase tag at all are flagged for the chair to fix.

**Per paper:** each reviewer's original and current score with their vote
marked, the averages, champions and anti-champions (counts and review
letters), the outcome the rule gives today (or the settled tag), whether the
discussion lead has posted since `--since`, how many reviewers have updated
their review or commented since then, the revision tag and the last activity
date. Papers are grouped: reopen? / tags to fix, then in discussion, then
settled accept, then settled reject. Attention flags: reopen, conflicting or
missing phase tags, both boxes ticked, and -- on papers in discussion only --
no champion, tied, no lead, lead silent, all reviewers silent; agreeing
votes and split scores are shown for information.

**Sources.** Everything live comes from HotCRP's API
(`reviewer_match.hotcrp_api`): the scope from a search (`--scope-search`,
`dec:none`), titles, tags, decisions and leads from the search's field
columns, scores and review status from `reviews`, discussion activity from
`comments`. A comment's creation date is its first log row
(`hotcrp_log.comment_created_at`) when `--log` exists, else the API's
last-modified date, which over-counts an old comment edited recently -- the
page says which. Chairs come from the user export's `~~track_NN` account tags
(`--pcinfo`, stamped "chairs as of" its download date). The API acts as the
token's owner, so a paper they are conflicted with returns no reviews and no
comments: such a paper is still listed, with its title and track, and flagged
hidden -- never silently dropped.

**Conflicts.** A page never shows a paper its readers are conflicted with.
`all.html` is read by the PC chairs together (every enabled account whose
roles include `chair`, plus any other account under the same name -- a
chair's second address carries conflicts too; `--pc-chairs` overrides the
list), `all_<name>.html` by one of them alone, omitting only that chair's own
conflicts, and each track page by that track's chairs; a paper that `conf:<address>` matches for
any reader of a page is left off it, and the page says how many were. The
area-chair assignment was made under hard COIs, so a track page should omit
nothing; the count is reported rather than assumed. The papers **every** PC chair
is conflicted with go to the **conflict chair** instead -- the intersection,
not the union: a paper only one chair is conflicted with is handled by the
other, on their `all_<name>.html` (the account tagged
`~~conflict-chair`; `--conflict-chair` overrides): they are added to the top
of that chair's own track page as a labelled group -- or to
`conflict_chair.html` when the conflict chair has no track -- minus any the
conflict chair is conflicted with too.

A lead is a display name in HotCRP's reply, not an address; it is resolved
to an account through the user export and the API's own author spellings,
and the lead's comments are matched by address or, failing that, by name.

**Who counts.** Submitted PC reviews only (review type >= 2, round not TRC):
a review never submitted is assumed not to be coming and is not shown, and
training reviews are left out as everywhere else here. A reviewer is active
since a date when their review was modified
on/after it (a post-rebuttal score counts as a modification) or they posted a
discussion comment -- not a draft, not a response, not an author's, not
admin-only -- on/after it. The nudges' definitions, imported from
`scripts/discussion_nudges.py` so the two tools cannot disagree.

**PDF.** Every page is built as HTML and printed to PDF through headless
Chrome (`--chrome`, default `google-chrome`): the page is wrapped in a print
document (light theme, A4 landscape, or `--orientation portrait`, the
controls hidden) and Chrome runs its script and prints. Only the PDF is kept,
in a directory named after the page (`track_05/`, `all/`, `all_<name>/`,
`conflict_chair/`); a track's file also names its chair, so the attachments
sort themselves. The pages below are named by the HTML they are printed from
(`track_NN.html` is `track_NN/track_NN_<chair>.pdf`). A page's directory is
created once and only ever refilled, so a Dropbox link to it
(`scripts/ac_dashboard_links.py`) keeps working across reruns -- deleting the
output directory breaks every such link.

No page carries email addresses, reviewer names or review text: a reviewer
appears only as a review letter with its scores, activity is summed per
paper, and the rows are built field by field, never from the API object. The pages name reviewers and show
scores, so the output directory is gitignored by name. `--generated-at` pins
the timestamp, so a rerun on the same snapshot is byte-identical.
stdout is one line, `pages papers hidden flagged`; everything else is stderr.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from reviewer_match import hotcrp_api, hotcrp_log, pc_membership, review_scores
from reviewer_match.paths import cache_path, input_path, report_path
from scripts.discussion_nudges import (
    Comment, ReviewState, commented_since, fetch_state, fold_name, parse_comments, parse_pids,
    parse_reviews, parse_since,
)
from scripts.discussion_nudges import parse_emails as parse_emails_list

DEFAULT_OUT_DIR = report_path("ac_dashboard")
DEFAULT_PCINFO = pc_membership.DEFAULT_PCINFO
DEFAULT_LOG = input_path("hpca2027-log.csv")
DEFAULT_SNAPSHOT = cache_path("hotcrp_api")
DEFAULT_SCOPE_SEARCH = "dec:none"
DEFAULT_ACCEPT_TAG = "online-accept"
DEFAULT_REJECT_TAG = "online-reject"
DEFAULT_DISCUSS_TAG = "pc-discuss"
DEFAULT_LIKELY_ACCEPT_TAG = "likely-accept"
DEFAULT_LIKELY_REJECT_TAG = "likely-reject"
DEFAULT_RESPONSE_NAME = "Revision"
DEFAULT_SPLIT_GAP = 2
# The search columns one `papers` call answers with, per paper.
PAPER_FIELDS = "title tags decision lead"
# Over `pc_membership.tag_names` output, so `~~track-discussion` and
# `TRC-track` cannot match.
TRACK_TAG_RE = re.compile(r"^track_(\d+)$")
ALL_PAGE = "all.html"
PAGE_NAME = "track_{:02d}.html"
CONFLICT_PAGE = "conflict_chair.html"
CHAIR_PAGE = "all_{}.html"             # one per PC chair, omitting only their own conflicts
# A page's directory, and the flat .html/.pdf files written before there were any.
PAGE_DIR_RE = re.compile(r"^(track_\d+|conflict_chair|all(_[a-z0-9-]+)?)$")
FLAT_RE = re.compile(r"^(track_\d+(_[a-z0-9-]+)?|conflict_chair|all(_[a-z0-9-]+)?)\.(html|pdf)$")
DEFAULT_CHROME = "google-chrome"
# The page, wrapped for Chrome's printer: a real document, light theme, and
# the data-print switch the page's script reads (expand every row, no controls).
PRINT_DOCUMENT = ('<!doctype html><html lang="en" data-theme="light" data-print="1"><head>'
                  '<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
                  '<style>@page { size: A4 {orientation}; margin: 10mm 10mm 12mm; }</style></head><body>'
                  '{page}</body></html>')
DEFAULT_ORIENTATION = "landscape"
CHROME_FLAGS = ["--headless=new", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                "--virtual-time-budget=10000", "--run-all-compositor-stages-before-draw"]
CONFLICT_CHAIR_TAG = "conflict-chair"
CONFLICT_GROUP_LABEL = "Conflicted with every PC chair"
DATE_FORMAT = "%Y-%m-%d %H:%M"

# Phases, from the paper's tags. `discuss` is live: it advances only with
# more champions than anti-champions. The settled ones were decided online
# (likely-* or online-*) and reopen only on a contrary vote.
DISCUSS, SETTLED_ACCEPT, SETTLED_REJECT = "discuss", "settled-accept", "settled-reject"
CONFLICTING, UNTAGGED = "conflicting", "untagged"
# What the champion rule gives today.
ADVANCES, NO_CHAMPION, OUTVOTED = "advances", "no-champion", "outvoted"
# The page's groups, top to bottom: what needs a decision first.
GROUP_ORDER = ["reopen", DISCUSS, SETTLED_ACCEPT, SETTLED_REJECT]

# Attention flags, in the order they are reported. `act` flags tint the row
# and count towards "needs attention"; `info` flags are shown, not counted.
ACT, INFO = "act", "info"
FLAGS = [
    ("hidden", "hidden from the chair's account", ACT),
    ("reopen", "contrary vote on a settled paper", ACT),
    ("conflicting-tags", "conflicting phase tags", ACT),
    ("untagged", "no phase tag", ACT),
    ("both-boxes", "a review ticks both champion and anti-champion", ACT),
    ("anti-only", "anti-champion, no champion", ACT),
    ("no-champion", "no votes yet", ACT),
    ("tied", "champions and anti-champions tied", ACT),
    ("no-lead", "no lead", ACT),
    ("lead-silent", "lead silent", ACT),
    ("all-reviewers-silent", "all reviewers silent", ACT),
    ("agreeing-vote", "vote agrees with the settled outcome", INFO),
    ("split", "scores split", INFO),
]
FLAG_LEVEL = {key: level for key, _, level in FLAGS}
FLAG_LABEL = {key: label for key, label, _ in FLAGS}


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrackChair:
    track: int
    name: str
    email: str


@dataclass
class ReviewerRow:
    letter: str
    name: str
    expertise: int | None
    status: str
    submitted: bool
    pre: int | None
    post: int | None
    modified_at: int
    updated: bool                      # review modified since the discussion opened
    comments: int                      # discussion comments since then
    last_comment_at: int | None
    since_response: bool | None        # active after the response; None without one
    champion: bool = False
    anti: bool = False

    @property
    def vote(self) -> str:
        """champion | anti | both | "" -- `both` counts on neither side."""
        return "both" if self.champion and self.anti else "champion" if self.champion else "anti" if self.anti else ""

    @property
    def current(self) -> int | None:
        return self.post if self.post is not None else self.pre

    @property
    def active(self) -> bool:
        return self.updated or self.comments > 0


@dataclass
class PaperRow:
    pid: int
    title: str
    tracks: list[int]
    lead_name: str
    lead_email: str
    lead_posted_at: int | None         # first discussion comment since the cutoff
    lead_comments: int
    phase: str                         # DISCUSS | SETTLED_* | CONFLICTING | UNTAGGED
    phase_tags: list[str]              # the phase tags the paper holds, as spelled
    revision: str                      # advance | none | ""
    response_at: int | None
    reviewers: list[ReviewerRow] = field(default_factory=list)   # submitted PC reviews only
    hidden: bool = False
    flags: list[str] = field(default_factory=list)

    @property
    def submitted(self) -> list[ReviewerRow]:
        return self.reviewers

    @property
    def not_rescored(self) -> int:
        """Reviews without a post-rebuttal score yet."""
        return sum(1 for r in self.reviewers if r.post is None)

    @property
    def avg_pre(self) -> float | None:
        scores = [r.pre for r in self.submitted if r.pre is not None]
        return float(review_scores.average(scores)) if scores else None

    @property
    def avg_current(self) -> float | None:
        scores = [r.current for r in self.submitted if r.current is not None]
        return float(review_scores.average(scores)) if scores else None

    @property
    def last_activity(self) -> int | None:
        stamps = [r.modified_at for r in self.reviewers if r.modified_at]
        stamps += [r.last_comment_at for r in self.reviewers if r.last_comment_at]
        stamps += [s for s in (self.lead_posted_at, self.response_at) if s]
        return max(stamps) if stamps else None

    @property
    def attention(self) -> int:
        return sum(1 for f in self.flags if FLAG_LEVEL[f] == ACT)

    def letters(self, vote: str) -> list[str]:
        return [r.letter for r in self.submitted if r.vote == vote]

    @property
    def champions(self) -> int:
        return len(self.letters("champion"))

    @property
    def antis(self) -> int:
        return len(self.letters("anti"))

    @property
    def projection(self) -> str:
        """What the champion rule gives today; "" when the votes are hidden."""
        return "" if self.hidden else projected(self.champions, self.antis)

    @property
    def reopen(self) -> bool:
        """A settled paper with a vote against its outcome."""
        return ((self.phase == SETTLED_ACCEPT and self.antis > 0)
                or (self.phase == SETTLED_REJECT and self.champions > 0))

    @property
    def group(self) -> str:
        if self.reopen or self.phase in (CONFLICTING, UNTAGGED):
            return "reopen"
        return self.phase


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

# An address someone typed into their name field, as HotCRP renders it
# ("Ann Author <ann@x.edu>", HTML-escaped in a review's `reviewer`).
_ADDRESS_RE = re.compile(r"(?:&lt;|<)[^<>]*(?:&gt;|>)|\S+@\S+")


def clean_name(name: str) -> str:
    """A display name with any email address dropped: no page carries one."""
    return " ".join(_ADDRESS_RE.sub(" ", name or "").split())


def tag_set(tags: str) -> set[str]:
    """The bare tag names in a search column's `#a #~~b` list (or an
    export's `a#0 ~~b#1` cell: the `#` prefix is dropped either way)."""
    return pc_membership.tag_names(" ".join(t.lstrip("#") for t in (tags or "").split()))


def track_numbers(tags: str) -> list[int]:
    return sorted(int(m.group(1)) for t in tag_set(tags) for m in [TRACK_TAG_RE.match(t)] if m)


def track_chairs(pcinfo_path: str) -> dict[int, list[TrackChair]]:
    """{track: chairs} from the `~~track_NN` tags on area-chair accounts."""
    index = pc_membership.load_pc_accounts(pcinfo_path)
    out: dict[int, list[TrackChair]] = defaultdict(list)
    for email, acct in sorted(index.by_email.items(), key=lambda kv: (kv[1].name, kv[0])):
        if not acct.is_area_chair or acct.disabled:
            continue
        tracks = track_numbers(acct.tags)
        if not tracks:
            print(f"WARNING: area chair {acct.name or email} holds no ~~track_N tag", file=sys.stderr)
        for n in tracks:
            out[n].append(TrackChair(n, acct.name or email, email))
    return dict(out)


def pc_chairs(pcinfo_path: str) -> dict[str, str]:
    """{address: name} of the PC chairs: enabled accounts with `chair` among
    their roles, widened to every enabled account under the same name."""
    index = pc_membership.load_pc_accounts(pcinfo_path)
    names = {fold_name(clean_name(a.name)) for a in index.by_email.values()
             if "chair" in a.roles.split() and not a.disabled and a.name}
    return {email: acct.name for email, acct in sorted(index.by_email.items())
            if acct.name and not acct.disabled and fold_name(clean_name(acct.name)) in names}


def conflict_chairs(pcinfo_path: str) -> list[str]:
    """Addresses of the enabled accounts tagged `~~conflict-chair`."""
    index = pc_membership.load_pc_accounts(pcinfo_path)
    return sorted(e for e, a in index.by_email.items()
                  if not a.disabled and CONFLICT_CHAIR_TAG in pc_membership.tag_names(a.tags))


def same_name_accounts(pcinfo_path: str, emails: list[str]) -> list[str]:
    """`emails` plus every other enabled account under the same display name."""
    index = pc_membership.load_pc_accounts(pcinfo_path)
    wanted = {fold_name(clean_name(index.by_email[e].name)) for e in emails if e in index.by_email}
    extra = [e for e, a in sorted(index.by_email.items())
             if a.name and not a.disabled and fold_name(clean_name(a.name)) in wanted]
    return sorted(set(emails) | set(extra))


def conflicts_of(site: hotcrp_api.Site, emails: list[str], pids: set[int]) -> dict[str, set[int]]:
    """{address: the papers in `pids` HotCRP's `conf:` search says it conflicts with}."""
    return {email: set(site.search(f"conf:{email}")) & pids for email in sorted(set(emails))}


def all_chairs_conflicted(each_chair: dict[str, list[str]], conflicts: dict[str, set[int]]) -> set[int]:
    """The conflict chair's charge: papers every PC chair is conflicted with,
    a chair counting as conflicted when any of their addresses is. A paper only
    some chairs are conflicted with stays with the others."""
    per_chair = [set().union(*(conflicts[e] for e in who)) for who in each_chair.values()]
    return set.intersection(*per_chair) if per_chair else set()


def name_to_emails(pcinfo_path: str | None, reviews: list[ReviewState], comments: list[Comment]) -> dict[str, set[str]]:
    """{folded display name: addresses seen under it}, from the user export
    and the API's own spellings. A name under two addresses is ambiguous."""
    out: dict[str, set[str]] = defaultdict(set)
    if pcinfo_path and os.path.exists(pcinfo_path):
        for email, acct in pc_membership.load_pc_accounts(pcinfo_path).by_email.items():
            if acct.name and not acct.disabled:
                out[fold_name(clean_name(acct.name))].add(email)
    for r in reviews:
        if r.reviewer and r.email:
            out[fold_name(clean_name(r.reviewer))].add(r.email)
    for c in comments:
        if c.author and c.author_email:
            out[fold_name(clean_name(c.author))].add(c.author_email)
    return out


def resolve_lead(name: str, names: dict[str, set[str]]) -> str:
    """The lead's address when the name belongs to exactly one, else ""."""
    emails = names.get(fold_name(name), set())
    return next(iter(emails)) if len(emails) == 1 else ""


def by_lead(c: Comment, lead_email: str, lead_name: str) -> bool:
    if lead_email and c.author_email == lead_email:
        return True
    return bool(lead_name) and fold_name(clean_name(c.author)) == fold_name(lead_name)


def paper_facts(site: hotcrp_api.Site, pids: set[int]) -> dict[int, dict]:
    """{pid: {title, tags, decision, lead}} from the search columns."""
    facts: dict[int, dict] = {}
    for q in hotcrp_api.pid_query(pids):
        for row in site.papers(q, PAPER_FIELDS):
            try:
                pid = int(row["pid"])
            except (KeyError, TypeError, ValueError):
                continue
            facts[pid] = {
                "title": str(row.get("title") or ""),
                "tags": str(row.get("tags") or ""),
                "decision": str(row.get("decision") or ""),
                "lead": clean_name(str(row.get("lead") or "")),
            }
    return facts


def phase_tag_map(discuss: str = DEFAULT_DISCUSS_TAG, likely_accept: str = DEFAULT_LIKELY_ACCEPT_TAG,
                  likely_reject: str = DEFAULT_LIKELY_REJECT_TAG, online_accept: str = DEFAULT_ACCEPT_TAG,
                  online_reject: str = DEFAULT_REJECT_TAG) -> dict[str, str]:
    """{tag as spelled: DISCUSS | SETTLED_ACCEPT | SETTLED_REJECT}."""
    return {discuss: DISCUSS, likely_accept: SETTLED_ACCEPT, online_accept: SETTLED_ACCEPT,
            likely_reject: SETTLED_REJECT, online_reject: SETTLED_REJECT}


DEFAULT_PHASE_TAGS = phase_tag_map()


def phase_of(tags: set[str], phase_tags: dict[str, str] = DEFAULT_PHASE_TAGS) -> tuple[str, list[str]]:
    """(phase, the phase tags held). Tags pointing different ways -- pc-discuss
    with online-reject, likely-accept with likely-reject -- are CONFLICTING."""
    held = [t for t in phase_tags if t.lower() in tags]
    phases = {phase_tags[t] for t in held}
    if not phases:
        return UNTAGGED, held
    return (phases.pop() if len(phases) == 1 else CONFLICTING), held


def projected(champions: int, antis: int) -> str:
    """The rule: a paper advances only with more champions than
    anti-champions, so with none it is rejected, and a tie rejects too."""
    if champions == 0:
        return NO_CHAMPION
    return ADVANCES if champions > antis else OUTVOTED


def revision_of(tags: set[str]) -> str:
    if review_scores.ADVANCE_TAG.lower() in tags:
        return "advance"
    if review_scores.NO_REVISION_TAG.lower() in tags:
        return "none"
    return ""


# ---------------------------------------------------------------------------
# The rows
# ---------------------------------------------------------------------------

def build_rows(
    pids: set[int], facts: dict[int, dict], reviews: list[ReviewState], comments: list[Comment],
    since: datetime, names: dict[str, set[str]], *,
    phase_tags: dict[str, str] = DEFAULT_PHASE_TAGS,
    response_name: str = DEFAULT_RESPONSE_NAME, split_gap: int = DEFAULT_SPLIT_GAP,
) -> list[PaperRow]:
    """One PaperRow per paper in `pids`, flags attached, sorted by pid."""
    reviews_by_pid: dict[int, list[ReviewState]] = defaultdict(list)
    for r in reviews:
        if r.pid in pids:
            reviews_by_pid[r.pid].append(r)
    comments_by_pid: dict[int, list[Comment]] = defaultdict(list)
    for c in comments:
        if c.pid in pids:
            comments_by_pid[c.pid].append(c)
    cutoff = int(since.timestamp())

    rows = []
    for pid in sorted(pids):
        fact = facts.get(pid, {})
        tags = tag_set(fact.get("tags", ""))
        lead_name = clean_name(fact.get("lead", ""))
        lead_email = resolve_lead(lead_name, names) if lead_name else ""
        cs = comments_by_pid.get(pid, [])
        rs = reviews_by_pid.get(pid, [])
        phase, held = phase_of(tags, phase_tags)

        responses = [c.modified_at for c in cs if c.response == response_name and not c.draft and c.modified_at]
        response_at = max(responses) if responses else None
        lead_posts = sorted(c.created_at for c in cs if c.discussion and by_lead(c, lead_email, lead_name)
                            and c.created_at >= since)
        row = PaperRow(
            pid=pid, title=fact.get("title", ""), tracks=track_numbers(fact.get("tags", "")),
            lead_name=lead_name, lead_email=lead_email,
            lead_posted_at=int(lead_posts[0].timestamp()) if lead_posts else None,
            lead_comments=len(lead_posts),
            phase=phase, phase_tags=held, revision=revision_of(tags),
            response_at=response_at,
            hidden=not rs and not cs,
        )
        for r in sorted((r for r in rs if r.pc and r.submitted), key=lambda r: (r.ordinal or "~", r.rid)):
            mine = [c for c in cs if c.discussion and r.email and c.author_email == r.email]
            since_start = [c for c in mine if c.created_at >= since]
            last = max((int(c.created_at.timestamp()) for c in mine), default=None)
            if response_at is None:
                since_response = None
            else:
                since_response = r.modified_at >= response_at or any(
                    int(c.created_at.timestamp()) >= response_at for c in mine)
            row.reviewers.append(ReviewerRow(
                letter=r.ordinal, name=clean_name(r.reviewer), expertise=r.expertise, status=r.status,
                submitted=r.submitted, pre=r.pre, post=r.post, modified_at=r.modified_at,
                updated=r.post is not None or r.modified_at >= cutoff,
                comments=len(since_start), last_comment_at=last, since_response=since_response,
                champion=r.champion, anti=r.anti,
            ))
        row.flags = attention_flags(row, split_gap)
        rows.append(row)
    return rows


def attention_flags(row: PaperRow, split_gap: int = DEFAULT_SPLIT_GAP) -> list[str]:
    """The discussion flags (lead, silence, champions) apply to pc-discuss
    papers only: a settled paper needs nothing until a contrary vote arrives."""
    flags = []
    submitted = row.submitted
    if row.hidden:
        flags.append("hidden")
    if row.reopen:
        flags.append("reopen")
    if row.phase == CONFLICTING:
        flags.append("conflicting-tags")
    if row.phase == UNTAGGED:
        flags.append("untagged")
    if row.letters("both"):
        flags.append("both-boxes")
    if row.phase == DISCUSS:
        if row.projection == NO_CHAMPION:
            flags.append("anti-only" if row.antis else "no-champion")
        if row.projection == OUTVOTED and row.champions == row.antis:
            flags.append("tied")
        if not row.lead_name:
            flags.append("no-lead")
        elif row.lead_posted_at is None and not row.hidden:
            flags.append("lead-silent")
        if submitted and not any(r.active for r in submitted):
            flags.append("all-reviewers-silent")
    if ((row.phase == SETTLED_ACCEPT and row.champions) or (row.phase == SETTLED_REJECT and row.antis)):
        flags.append("agreeing-vote")
    current = [r.current for r in submitted if r.current is not None]
    if len(current) >= 2 and max(current) - min(current) >= split_gap:
        flags.append("split")
    return flags


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def fmt_day(ts: int | None) -> str:
    return datetime.fromtimestamp(ts).astimezone().strftime("%b %-d") if ts else ""


def fmt_stamp(ts: int | None) -> str:
    return datetime.fromtimestamp(ts).astimezone().strftime(DATE_FORMAT) if ts else ""


def reviewer_dict(r: ReviewerRow) -> dict:
    """One score chip: the review letter and its scores. No name and no
    per-reviewer activity reach the page; those are summed per paper."""
    return {
        "letter": r.letter, "pre": r.pre, "post": r.post, "current": r.current,
        "rescored": r.post is not None,
        "changed": r.post is not None and r.pre is not None and r.post != r.pre,
        "vote": r.vote,
    }


def paper_dict(row: PaperRow, site_url: str) -> dict:
    """The page's view of one paper: named fields only, no addresses, no text."""
    submitted = row.submitted
    return {
        "pid": row.pid, "title": row.title, "url": f"{site_url}/paper/{row.pid}",
        "tracks": row.tracks, "hidden": row.hidden,
        "lead": row.lead_name, "leadPosted": fmt_day(row.lead_posted_at), "leadComments": row.lead_comments,
        "reviewers": [reviewer_dict(r) for r in row.reviewers],
        "submitted": len(submitted), "notRescored": row.not_rescored,
        "activeSinceStart": sum(1 for r in submitted if r.active),
        "activeSinceResponse": (sum(1 for r in submitted if r.since_response)
                                if row.response_at is not None else None),
        "avgPre": row.avg_pre, "avgCurrent": row.avg_current,
        "phase": row.phase, "phaseTags": row.phase_tags, "phaseGroup": row.group,
        "champions": row.champions, "antis": row.antis,
        "championLetters": row.letters("champion"), "antiLetters": row.letters("anti"),
        "bothLetters": row.letters("both"), "projection": row.projection,
        "revision": row.revision,
        "response": fmt_day(row.response_at), "responseAt": row.response_at or 0,
        "lastActivity": fmt_day(row.last_activity), "lastActivityAt": row.last_activity or 0,
        "flags": row.flags, "attention": row.attention,
    }


def page_payload(
    rows: list[PaperRow], chairs: dict[int, list[TrackChair]], track: int | None, *,
    site_url: str, since: datetime, generated_at: str, notes: list[str],
    phase_tags: dict[str, str] = DEFAULT_PHASE_TAGS, conflicted: set[int] = frozenset(),
    extra: set[int] = frozenset(), extra_label: str = CONFLICT_GROUP_LABEL, reader: str = "",
) -> dict:
    """The JSON one page embeds. A track page gets only that track's papers
    (filtered here, not in the browser); `track=None` is every track; `track`
    -1 is a page with no track of its own. Papers in `conflicted` -- the
    page's readers' conflicts -- are left off and counted. Papers in `extra`
    (the conflict chair's charge) are added as a labelled group ahead of the
    track's own, each paper once."""
    def in_track(r: PaperRow) -> bool:
        return track is None or track in r.tracks

    mine = [r for r in rows if (in_track(r) or r.pid in extra) and r.pid not in conflicted]
    omitted = sum(1 for r in rows if (in_track(r) or r.pid in extra) and r.pid in conflicted)
    tracks = sorted({t for r in rows for t in r.tracks} | set(chairs)) if track is None else [track] if track >= 0 else []
    papers = []
    for r in mine:
        d = paper_dict(r, site_url)
        d["group"] = "extra" if r.pid in extra else ""
        papers.append(d)
    return {
        "track": track,
        "reader": reader,
        "omitted": omitted,
        "extraLabel": extra_label,
        "extraCount": sum(1 for p in papers if p["group"]),
        "tracks": [{"n": n, "chairs": [c.name for c in chairs.get(n, [])],
                    "papers": sum(1 for r in rows if n in r.tracks)} for n in tracks],
        "papers": papers,
        "flags": [{"key": key, "label": label, "level": level} for key, label, level in FLAGS],
        "groups": GROUP_ORDER,
        "tags": {"discuss": [t for t, ph in phase_tags.items() if ph == DISCUSS],
                 "accept": [t for t, ph in phase_tags.items() if ph == SETTLED_ACCEPT],
                 "reject": [t for t, ph in phase_tags.items() if ph == SETTLED_REJECT],
                 "advance": review_scores.ADVANCE_TAG, "none": review_scores.NO_REVISION_TAG},
        "since": since.strftime("%b %-d"),
        "sinceLong": since.strftime("%b %-d, %Y"),
        "generatedAt": generated_at,
        "notes": notes,
    }


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", fold_name(clean_name(name))).strip("-")


def page_title(track: int | None, reader: str = "") -> str:
    if track is None:
        return f"HPCA 2027 All Tracks ({reader})" if reader else "HPCA 2027 All Tracks"
    return "HPCA 2027 Conflict Chair" if track < 0 else f"HPCA 2027 Track {track}"


def render_html(payload: dict, title: str) -> str:
    """HTML_TEMPLATE with the payload embedded as inert JSON."""
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True).replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__TITLE__", title).replace("__PAYLOAD__", data)


def page_dir(page: str) -> str:
    """The directory a page's PDF goes in: `track_05.html` -> `track_05`."""
    return re.sub(r"\.html$", "", page)


def pdf_name(page: str, chairs: dict[int, list[TrackChair]]) -> str:
    """A page's PDF: track pages carry the chair's name."""
    m = re.match(r"^track_(\d+)\.html$", page)
    if m and chairs.get(int(m.group(1))):
        names = "-".join(slug(c.name) for c in chairs[int(m.group(1))])
        return f"track_{int(m.group(1)):02d}_{names}.pdf"
    return re.sub(r"\.html$", ".pdf", page)


def print_document(page_html: str, orientation: str = DEFAULT_ORIENTATION) -> str:
    return PRINT_DOCUMENT.replace("{orientation}", orientation).replace("{page}", page_html)


def write_pdfs(out_dir: str, pages: dict[str, str], names: dict[str, str], chrome: str, *,
               sweep: bool = True, orientation: str = DEFAULT_ORIENTATION) -> list[str]:
    """Print every page to `<page_dir>/<names[page]>` under `out_dir` with
    headless Chrome, replacing any other PDF in that directory (a chair since
    renamed). On a full run, also remove page directories no longer generated
    and the flat files of the old layout. Returns the paths written, relative
    to `out_dir`."""
    binary = shutil.which(chrome)
    if not binary:
        raise SystemExit(f"ERROR: {chrome!r} not found; pass --chrome or install Google Chrome / Chromium")
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for page, html in pages.items():
        sub = os.path.join(out_dir, page_dir(page))
        os.makedirs(sub, exist_ok=True)
        # Temp files in out_dir, never in the page's own (possibly shared)
        # directory: same filesystem, so the replace below stays atomic.
        fd, src = tempfile.mkstemp(dir=out_dir, suffix=".print.html")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(print_document(html, orientation))
        tmp_pdf = src[:-len(".print.html")] + ".pdf.tmp"
        try:
            result = subprocess.run([binary, *CHROME_FLAGS, f"--print-to-pdf={tmp_pdf}", "file://" + os.path.abspath(src)],
                                    capture_output=True, text=True, timeout=120)
            if result.returncode != 0 or not os.path.exists(tmp_pdf):
                raise SystemExit(f"ERROR: Chrome failed to print {page}: {result.stderr.strip()[-400:]}")
            os.replace(tmp_pdf, os.path.join(sub, names[page]))
            written.append(os.path.join(page_dir(page), names[page]))
        finally:
            for path in (src, tmp_pdf):
                if os.path.exists(path):
                    os.remove(path)
        for name in sorted(os.listdir(sub)):
            if name.endswith(".pdf") and name != names[page]:
                os.remove(os.path.join(sub, name))
                print(f"removed stale {os.path.join(page_dir(page), name)}", file=sys.stderr)
    dirs = {page_dir(page) for page in pages}
    for name in sorted(os.listdir(out_dir)) if sweep else []:
        path = os.path.join(out_dir, name)
        if os.path.isdir(path) and PAGE_DIR_RE.match(name) and name not in dirs:
            for inner in os.listdir(path):
                if inner.endswith(".pdf"):
                    os.remove(os.path.join(path, inner))
            if not os.listdir(path):
                os.rmdir(path)
            print(f"removed stale {name}/", file=sys.stderr)
        elif os.path.isfile(path) and FLAT_RE.match(name):
            os.remove(path)
            print(f"removed stale {name}", file=sys.stderr)
    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--since", required=True, help="when the discussion opened (date, or a log-style timestamp)")
    p.add_argument("--scope-search", default=DEFAULT_SCOPE_SEARCH,
                   help=f"HotCRP search selecting the papers (default {DEFAULT_SCOPE_SEARCH!r})")
    p.add_argument("--exclude-pids", default="", help="comma-separated paper IDs to leave out")
    p.add_argument("--tracks", default="", help="comma-separated track numbers to write (default: all)")
    p.add_argument("--pcinfo", default=DEFAULT_PCINFO, help="HotCRP user export (chairs, names)")
    p.add_argument("--log", default=DEFAULT_LOG, help="HotCRP action log (comment creation dates)")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--snapshot", default=DEFAULT_SNAPSHOT, help="directory of API snapshots")
    p.add_argument("--offline", action="store_true", help="answer every API call from --snapshot")
    p.add_argument("--discuss-tag", default=DEFAULT_DISCUSS_TAG,
                   help=f"tag of the papers still in discussion (default {DEFAULT_DISCUSS_TAG})")
    p.add_argument("--likely-accept-tag", default=DEFAULT_LIKELY_ACCEPT_TAG)
    p.add_argument("--likely-reject-tag", default=DEFAULT_LIKELY_REJECT_TAG)
    p.add_argument("--accept-tag", default=DEFAULT_ACCEPT_TAG, help="the online-accept tag (settled accept)")
    p.add_argument("--reject-tag", default=DEFAULT_REJECT_TAG, help="the online-reject tag (settled reject)")
    p.add_argument("--response-name", default=DEFAULT_RESPONSE_NAME,
                   help=f"the response round whose submission counts (default {DEFAULT_RESPONSE_NAME})")
    p.add_argument("--split-gap", type=int, default=DEFAULT_SPLIT_GAP,
                   help=f"flag papers whose current scores differ by this much (default {DEFAULT_SPLIT_GAP})")
    p.add_argument("--pc-chairs", default="",
                   help="comma-separated addresses whose conflicts all.html omits (default: the user "
                        "export's chair accounts, plus their other addresses)")
    p.add_argument("--conflict-chair", default="",
                   help="address of the conflict chair, who gets the papers every PC chair is conflicted with "
                        "(default: the account tagged ~~conflict-chair; 'none' for nobody)")
    p.add_argument("--site-url", default="", help="HotCRP base URL for paper links (default: the site)")
    p.add_argument("--generated-at", default="", help="timestamp to print (YYYY-MM-DD HH:MM; default: now)")
    p.add_argument("--chrome", default=DEFAULT_CHROME, help=f"Chrome/Chromium binary that prints the PDFs (default {DEFAULT_CHROME})")
    p.add_argument("--orientation", choices=("landscape", "portrait"), default=DEFAULT_ORIENTATION,
                   help=f"PDF page orientation (default {DEFAULT_ORIENTATION})")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not shutil.which(args.chrome):   # before any API call, not after
        raise SystemExit(f"ERROR: {args.chrome!r} not found; pass --chrome or install Google Chrome / Chromium")
    since = parse_since(args.since)
    generated_at = args.generated_at or datetime.now().astimezone().strftime(DATE_FORMAT)
    exclude = frozenset(parse_pids(args.exclude_pids))
    only_tracks = {int(t) for t in re.split(r"[,\s]+", args.tracks) if t.strip()}

    site = hotcrp_api.Site.from_env(snapshot_dir=args.snapshot, offline=args.offline)
    site_url = (args.site_url or site.base_url).rstrip("/")
    chairs = track_chairs(args.pcinfo)
    chairs_as_of = datetime.fromtimestamp(os.path.getmtime(args.pcinfo)).strftime(DATE_FORMAT)

    pids = set(site.search(args.scope_search)) - exclude
    print(f"{len(pids)} papers in scope (search {args.scope_search!r})", file=sys.stderr)
    if not pids:
        raise SystemExit("ERROR: no papers in scope")
    facts = paper_facts(site, pids)
    missing = sorted(pids - set(facts))
    if missing:
        print(f"WARNING: {len(missing)} paper(s) answered no search row: {', '.join(map(str, missing))}",
              file=sys.stderr)

    notes = [f"Discussion activity is counted from {since.strftime('%b %-d, %Y')}.",
             f"Chairs and names as of the user export of {chairs_as_of}."]
    log_rows = hotcrp_log.load_log(args.log) if os.path.exists(args.log) else []
    if log_rows:
        notes.append(f"Comment dates are creation dates from the action log through {log_rows[-1]['date'][:16]}; "
                     "a comment created after that is dated by its last edit.")
    else:
        print(f"WARNING: {args.log} missing; comment creation dates fall back to last-modified", file=sys.stderr)
        notes.append("No action log: a comment is dated by its last edit, so an old comment edited recently "
                     "counts as recent.")
    created = hotcrp_log.comment_created_at(log_rows)

    raw_reviews, raw_comments = fetch_state(site, pids)
    reviews = parse_reviews(raw_reviews)
    comments = [c for c in parse_comments(raw_comments, created) if c.pid in pids]
    names = name_to_emails(args.pcinfo, reviews, comments)
    phase_tags = phase_tag_map(args.discuss_tag, args.likely_accept_tag, args.likely_reject_tag,
                               args.accept_tag, args.reject_tag)
    rows = build_rows(pids, facts, reviews, comments, since, names, phase_tags=phase_tags,
                      response_name=args.response_name, split_gap=args.split_gap)
    print(f"API: {len(reviews)} reviews, {len(comments)} comments", file=sys.stderr)

    # Who reads each page, and what they must not see.
    if args.pc_chairs:
        chair_emails = same_name_accounts(args.pcinfo, sorted(parse_emails_list(args.pc_chairs)))
    else:
        chair_emails = sorted(pc_chairs(args.pcinfo))
    # Each chair by name, with every address under it: their own page's readers.
    index = pc_membership.load_pc_accounts(args.pcinfo).by_email
    each_chair: dict[str, list[str]] = defaultdict(list)
    for email in chair_emails:
        each_chair[clean_name(index[email].name) if email in index else email].append(email)
    if not chair_emails:
        print("WARNING: no PC chair account found; all.html omits nothing", file=sys.stderr)
    readers = {None: chair_emails}
    for n, cs in chairs.items():
        readers[n] = same_name_accounts(args.pcinfo, [c.email for c in cs])
    if args.conflict_chair.lower() == "none":
        conflict_chair = []
    elif args.conflict_chair:
        conflict_chair = same_name_accounts(args.pcinfo, sorted(parse_emails_list(args.conflict_chair)))
    else:
        conflict_chair = same_name_accounts(args.pcinfo, conflict_chairs(args.pcinfo))
    # The conflict chair's page: their own track's, or a page of its own.
    conflict_track = next((n for n, who in readers.items() if n is not None and set(who) & set(conflict_chair)), None)
    if conflict_chair and conflict_track is None:
        readers[-1] = conflict_chair
    conflicts = conflicts_of(site, [e for who in readers.values() for e in who], pids)
    for email in chair_emails:
        print(f"PC chair {email}: {len(conflicts[email])} conflicted paper(s) left off {ALL_PAGE}", file=sys.stderr)
    pc_conflicted = all_chairs_conflicted(each_chair, conflicts)
    if conflict_chair:
        own = set().union(*(conflicts[e] for e in conflict_chair)) & pc_conflicted
        where = PAGE_NAME.format(conflict_track) if conflict_track is not None else CONFLICT_PAGE
        print(f"conflict chair {', '.join(conflict_chair)}: {len(pc_conflicted - own)} of the "
              f"{len(pc_conflicted)} paper(s) every PC chair is conflicted with added to {where}"
              + (f" ({len(own)} they are conflicted with too, left off)" if own else ""), file=sys.stderr)
    elif pc_conflicted:
        print(f"WARNING: no conflict chair (tag ~~{CONFLICT_CHAIR_TAG}); the {len(pc_conflicted)} paper(s) "
              f"every PC chair is conflicted with appear on no page but their tracks'", file=sys.stderr)
    for n in sorted(k for k in readers if k is not None):
        in_track = {r.pid for r in rows if n in r.tracks}
        hit = set().union(*(conflicts[e] for e in readers[n])) & in_track
        if hit:
            print(f"WARNING: track {n}'s chair is conflicted with {len(hit)} paper(s) in the track, left off "
                  f"its page: {', '.join(map(str, sorted(hit)))}", file=sys.stderr)

    hidden = [r.pid for r in rows if r.hidden]
    if hidden:
        print(f"HIDDEN: {len(hidden)} paper(s) returned no reviews and no comments -- the token's "
              f"owner is conflicted: {', '.join(map(str, hidden))}", file=sys.stderr)
    unresolved = sorted({r.lead_name for r in rows if r.lead_name and not r.lead_email})
    if unresolved:
        print(f"{len(unresolved)} lead name(s) match no single account (comments matched by name): "
              + ", ".join(unresolved), file=sys.stderr)
    untracked = [r.pid for r in rows if not r.tracks]
    if untracked:
        print(f"WARNING: {len(untracked)} paper(s) carry no ~~track_N tag and appear only in {ALL_PAGE}: "
              + ", ".join(map(str, untracked)), file=sys.stderr)
    phases = Counter(r.phase for r in rows)
    print("phases: " + ", ".join(f"{phases[k]} {k}" for k in (DISCUSS, SETTLED_ACCEPT, SETTLED_REJECT, CONFLICTING, UNTAGGED)),
          file=sys.stderr)
    proj = Counter(r.projection or "hidden" for r in rows if r.phase == DISCUSS)
    print(f"{args.discuss_tag} today: " + ", ".join(f"{proj[k]} {k}" for k in (ADVANCES, NO_CHAMPION, OUTVOTED, "hidden") if proj[k]),
          file=sys.stderr)
    voted = [f"{r.pid} (▲{r.champions} ▼{r.antis}, {r.phase})" for r in rows if r.champions or r.antis or r.letters("both")]
    print(f"{len(voted)} paper(s) with a vote: " + (", ".join(voted) or "none"), file=sys.stderr)
    for key, what in (("reopen", "REOPEN? contrary vote on a settled paper"),
                      ("both-boxes", "a review ticks both champion and anti-champion"),
                      ("conflicting-tags", "conflicting phase tags"), ("untagged", "no phase tag")):
        hit = [r.pid for r in rows if key in r.flags]
        if hit:
            print(f"{what}: {', '.join(map(str, hit))}", file=sys.stderr)
    counts = Counter(f for r in rows for f in r.flags)
    for key, label, level in FLAGS:
        if counts[key]:
            print(f"  {counts[key]:4d}  {label}" + ("" if level == ACT else " (info)"), file=sys.stderr)

    tracks = sorted({t for r in rows for t in r.tracks} | set(chairs))
    for n in tracks:
        if n not in chairs:
            print(f"WARNING: track {n} has papers but no chair account tagged ~~track_{n}", file=sys.stderr)
    pages = {}
    common = dict(site_url=site_url, since=since, generated_at=generated_at, notes=notes,
                  phase_tags=phase_tags)
    def conflicted(key) -> set[int]:
        return set().union(*(conflicts[e] for e in readers.get(key, [])))

    def extra(key) -> set[int]:
        return pc_conflicted if conflict_chair and key == (conflict_track if conflict_track is not None else -1) else set()

    for n in tracks:
        if only_tracks and n not in only_tracks:
            continue
        pages[PAGE_NAME.format(n)] = render_html(
            page_payload(rows, chairs, n, conflicted=conflicted(n), extra=extra(n), **common), page_title(n))
    if not only_tracks:
        pages[ALL_PAGE] = render_html(
            page_payload(rows, chairs, None, conflicted=conflicted(None), **common), page_title(None))
        for name, who in sorted(each_chair.items()):
            own = set().union(*(conflicts[e] for e in who))
            pages[CHAIR_PAGE.format(slug(name))] = render_html(
                page_payload(rows, chairs, None, conflicted=own, reader=name, **common), page_title(None, name))
        if -1 in readers:
            pages[CONFLICT_PAGE] = render_html(
                page_payload(rows, chairs, -1, conflicted=conflicted(-1), extra=extra(-1), **common), page_title(-1))
    pdfs = write_pdfs(args.out_dir, pages, {page: pdf_name(page, chairs) for page in pages}, args.chrome,
                      sweep=not only_tracks, orientation=args.orientation)
    for n in tracks:
        if not only_tracks or n in only_tracks:
            print(f"  track {n:2d}: {sum(1 for r in rows if n in r.tracks):3d} papers  "
                  + ", ".join(c.name for c in chairs.get(n, [])), file=sys.stderr)
    flagged = sum(1 for r in rows if r.attention)
    print(f"Wrote {len(pdfs)} PDF(s) to {args.out_dir}", file=sys.stderr)
    print(f"{len(pages)}\t{len(rows)}\t{len(hidden)}\t{flagged}")
    return 0


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<title>__TITLE__</title>
<meta name="claude:viewport-fit-sides" content="cover">
<style>
/* Layout: a summary strip, then one dense sortable table, one row per paper. */
:root {
  --bg: #f6f5f2; --surface: #fdfdfb; --ink: #1c1b18; --ink-2: #5a574f; --muted: #8b8880;
  --line: #e3e1da; --wash: rgba(28,27,24,0.045); --accent: #2f5d8a; --accent-ink: #ffffff;
  --good: #2d7a4f; --good-bg: #e3f2e8; --warn: #9a6a07; --warn-bg: #fbf1d6; --bad: #a83a30; --bad-bg: #f9e4e1;
  --info-bg: #e8eef5; --chip: #ecebe6; --tint-act: #fdf6ec; --tint-hidden: #f2f0ec;
  --font-ui: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --font-mono: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg: #15161a; --surface: #1d1f24; --ink: #ecebe6; --ink-2: #b4b1a8; --muted: #7d7b74;
    --line: #2f3238; --wash: rgba(236,235,230,0.06); --accent: #8ab4e0; --accent-ink: #15161a;
    --good: #7dc99a; --good-bg: #1f3a2b; --warn: #e4b75a; --warn-bg: #3d3116; --bad: #ee8f84; --bad-bg: #452422;
    --info-bg: #26303c; --chip: #2a2d33; --tint-act: #262219; --tint-hidden: #1a1b1f;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #15161a; --surface: #1d1f24; --ink: #ecebe6; --ink-2: #b4b1a8; --muted: #7d7b74;
  --line: #2f3238; --wash: rgba(236,235,230,0.06); --accent: #8ab4e0; --accent-ink: #15161a;
  --good: #7dc99a; --good-bg: #1f3a2b; --warn: #e4b75a; --warn-bg: #3d3116; --bad: #ee8f84; --bad-bg: #452422;
  --info-bg: #26303c; --chip: #2a2d33; --tint-act: #262219; --tint-hidden: #1a1b1f;
}
* { box-sizing: border-box; }
[hidden] { display: none !important; }
body {
  margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.45 var(--font-ui);
  padding-block: 24px 48px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
}
main { margin: 0 auto; min-width: 0; }
h1 { font-size: 22px; line-height: 1.25; margin: 0 0 4px; font-weight: 650; text-wrap: balance; }
h2 { font-size: 15px; margin: 0 0 6px; font-weight: 600; }
p { margin: 0; }
.lede { color: var(--ink-2); max-width: 80ch; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 18px 0 14px; }
.tile { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 10px 14px; }
.tile .label { color: var(--ink-2); font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }
.tile .value { font-size: 26px; font-weight: 600; line-height: 1.2; font-variant-numeric: tabular-nums; }
.tile.act .value { color: var(--warn); }
.controls { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 16px; margin: 0 0 10px; color: var(--ink-2); font-size: 13px; }
.controls label { display: inline-flex; align-items: center; gap: 6px; }
.controls input[type="search"], .controls select {
  font: inherit; color: var(--ink); background: var(--surface); border: 1px solid var(--line); border-radius: 7px; padding: 5px 8px;
}
.controls input[type="search"] { min-width: 200px; }
.tablewrap { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; font-variant-numeric: tabular-nums; }
th, td { padding: 7px 9px; text-align: left; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--ink); font-weight: 650; cursor: pointer; user-select: none; position: sticky; top: env(safe-area-inset-top, 0px); background: var(--chip); z-index: 1; border-bottom: 2px solid var(--ink-2); vertical-align: bottom; }
th .sub { font-weight: 400; color: var(--ink-2); font-size: 11px; line-height: 1.3; }
th[aria-sort="ascending"]::after { content: " \25B4"; }
th[aria-sort="descending"]::after { content: " \25BE"; }
th.num, td.num { text-align: right; }
tr.paper:hover { background: var(--wash); }
tr.paper.act { background: var(--tint-act); }
tr.paper.hidden-paper { background: var(--tint-hidden); color: var(--ink-2); }
td.title { min-width: 220px; max-width: 480px; white-space: normal; }
td.title a { color: var(--accent); text-decoration: none; }
td.title a:hover, td.title a:focus-visible { text-decoration: underline; }
td.title .pid { color: var(--muted); font-family: var(--font-mono); font-size: 12px; margin-right: 6px; }
.scores { display: inline-flex; flex-wrap: wrap; gap: 3px; }
.score {
  display: inline-block; min-width: 22px; text-align: center; padding: 1px 5px; border-radius: 5px;
  background: var(--chip); font-family: var(--font-mono); font-size: 12px;
}
.score.rescored { background: var(--info-bg); font-weight: 600; }
.score.open { color: var(--muted); border: 1px dashed var(--line); background: transparent; }
.avg { white-space: nowrap; }
.avg .arrow { color: var(--muted); margin: 0 3px; }
.sub { display: block; color: var(--muted); font-size: 12px; }
.ok { color: var(--good); }
.warn { color: var(--warn); }
.bad { color: var(--bad); }
.pill { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 12px; font-weight: 600; }
.pill.accept { background: var(--good-bg); color: var(--good); }
.pill.reject { background: var(--bad-bg); color: var(--bad); }
.pill.both { background: var(--bad-bg); color: var(--bad); }
.pill.none { background: var(--chip); color: var(--muted); font-weight: 500; }
.pill.advances { background: var(--good-bg); color: var(--good); }
.pill.no-champion, .pill.outvoted { background: var(--bad-bg); color: var(--bad); }
.pill.reopen { background: var(--warn-bg); color: var(--warn); }
.pill.settled-accept { background: var(--good-bg); color: var(--good); font-weight: 500; }
.pill.settled-reject { background: var(--chip); color: var(--ink-2); font-weight: 500; }
.votes { white-space: nowrap; font-weight: 600; }
.votes .up { color: var(--good); } .votes .down { color: var(--bad); } .votes .sep { color: var(--muted); margin: 0 4px; font-weight: 400; }
.score .mark { margin-left: 2px; font-family: var(--font-ui); }
.score .mark.up { color: var(--good); } .score .mark.down { color: var(--bad); } .score .mark.both { color: var(--warn); }
.flags { display: flex; flex-wrap: wrap; gap: 3px; }
.flag { display: inline-block; padding: 0 7px; border-radius: 5px; font-size: 12px; white-space: nowrap; background: var(--chip); color: var(--ink-2); }
.flag.act { background: var(--warn-bg); color: var(--warn); }
.flag.hidden { background: var(--bad-bg); color: var(--bad); }
tr.group td { background: var(--wash); color: var(--ink-2); font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; padding: 6px 9px; }
.empty { padding: 24px; color: var(--muted); text-align: center; }
.notes { color: var(--ink-2); font-size: 13px; max-width: 80ch; margin-top: 22px; }
.notes ul { padding-left: 18px; margin: 6px 0; }
.notes li { margin: 3px 0; }
.notes code { font-family: var(--font-mono); font-size: 12px; }
th:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
@media (prefers-reduced-motion: no-preference) { tr.paper { transition: background 120ms; } }
@media (max-width: 720px) { td.title { min-width: 0; } th, td { padding: 6px 5px; } }
.print .controls { display: none; }
@media print {
  body { padding: 0; font-size: 11px; background: #fff; }
  table { font-size: 10.5px; }
  th { position: static; cursor: default; }
  thead { display: table-header-group; }
  th .sub { font-size: 9.5px; }
  th, td { padding: 4px 5px; }
  .scores { min-width: 118px; }
  td.title { min-width: 150px; }
  .tiles { margin: 10px 0 8px; grid-template-columns: repeat(auto-fit, minmax(92px, 1fr)); gap: 6px; }
  .tile { padding: 6px 10px; } .tile .value { font-size: 18px; }
  .tablewrap { border: 0; border-radius: 0; }
  tr.paper, tr.group { break-inside: avoid; }
  td.title { max-width: 300px; }
  .pill { white-space: nowrap; }
  .notes { break-before: auto; }
}
</style>
<main>
  <h1 id="heading"></h1>
  <p class="lede" id="lede"></p>
  <div class="tiles" id="tiles"></div>
  <div class="controls" id="controls">
    <label id="track-label" hidden>Track <select id="track"></select></label>
    <label>Phase <select id="phase"></select></label>
    <label><input type="checkbox" id="only-act"> needs attention only</label>
    <label>Find <input type="search" id="q" placeholder="paper # or title"></label>
    <span id="shown"></span>
  </div>
  <div class="tablewrap"><table id="papers"><thead></thead><tbody></tbody></table></div>
  <section class="notes" id="notes">
    <h2>How to read this</h2>
    <ul id="flag-notes"></ul>
    <ul id="data-notes"></ul>
  </section>
</main>

<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
(function () {
  "use strict";
  var D = JSON.parse(document.getElementById("payload").textContent);
  var ALL = D.track === null;
  var LABEL = {}, LEVEL = {};
  D.flags.forEach(function (f) { LABEL[f.key] = f.label; LEVEL[f.key] = f.level; });
  var chairsOf = {};
  D.tracks.forEach(function (t) { chairsOf[t.n] = t.chairs; });

  function el(tag, cls, text, parent) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function avg(x) { return x == null ? "–" : x.toFixed(2); }
  function cmpText(a, b) { return a < b ? -1 : a > b ? 1 : 0; }

  // ---- header ---------------------------------------------------------------
  var heading = ALL ? "All tracks" + (D.reader ? " · " + D.reader : "") : D.track < 0 ? "Conflict chair" : "Track " + D.track + (chairsOf[D.track] && chairsOf[D.track].length ? " · " + chairsOf[D.track].join(", ") : "");
  document.getElementById("heading").textContent = heading;
  document.title = ALL ? "HPCA 2027 All Tracks" + (D.reader ? " (" + D.reader + ")" : "") : D.track < 0 ? "HPCA 2027 Conflict Chair" : "HPCA 2027 Track " + D.track;

  // ---- state ----------------------------------------------------------------
  var PRINT = document.documentElement.getAttribute("data-print") === "1";
  var state = { track: "", phase: "", onlyAct: false, q: "", sort: "attention", dir: "desc" };
  if (PRINT) {
    document.querySelector("main").className = "print";
  } else {
    try {
      var saved = JSON.parse(localStorage.getItem("ac-dashboard") || "{}");
      ["track", "phase", "onlyAct", "q", "sort", "dir"].forEach(function (k) { if (k in saved) state[k] = saved[k]; });
    } catch (e) {}
  }
  function persist() {
    if (PRINT) return;
    try { localStorage.setItem("ac-dashboard", JSON.stringify({ track: state.track, phase: state.phase, onlyAct: state.onlyAct, q: state.q, sort: state.sort, dir: state.dir })); } catch (e) {}
  }

  // ---- controls -------------------------------------------------------------
  var trackSel = document.getElementById("track");
  if (ALL) {
    document.getElementById("track-label").hidden = false;
    var o = el("option", null, "all", trackSel); o.value = "";
    D.tracks.forEach(function (t) {
      var opt = el("option", null, "Track " + t.n + (t.chairs.length ? " · " + t.chairs.join(", ") : "") + " (" + t.papers + ")", trackSel);
      opt.value = String(t.n);
    });
    trackSel.value = state.track;
    if (trackSel.value !== state.track) state.track = "";
    trackSel.addEventListener("change", function () { state.track = trackSel.value; persist(); render(); });
  } else {
    state.track = "";
  }
  function tagList(xs) { return xs.join(" / "); }
  var GROUP_LABEL = {
    reopen: "Reopen? or tags to fix",
    discuss: "Still in discussion (" + tagList(D.tags.discuss) + ")",
    "settled-accept": "Settled accept (" + tagList(D.tags.accept) + ")",
    "settled-reject": "Settled reject (" + tagList(D.tags.reject) + ")"
  };
  var phaseSel = document.getElementById("phase");
  [["", "all"], ["reopen", "reopen? / tags to fix"], ["discuss", tagList(D.tags.discuss)], ["settled-accept", "settled accept"], ["settled-reject", "settled reject"]].forEach(function (o) {
    var opt = el("option", null, o[1], phaseSel); opt.value = o[0];
  });
  phaseSel.value = state.phase || "";
  if (phaseSel.value !== (state.phase || "")) state.phase = "";
  phaseSel.addEventListener("change", function () { state.phase = phaseSel.value; persist(); render(); });
  var onlyAct = document.getElementById("only-act");
  onlyAct.checked = !!state.onlyAct;
  onlyAct.addEventListener("change", function () { state.onlyAct = onlyAct.checked; persist(); render(); });
  var qBox = document.getElementById("q");
  qBox.value = state.q || "";
  qBox.addEventListener("input", function () { state.q = qBox.value; persist(); render(); });

  // ---- columns --------------------------------------------------------------
  var COLS = [
    { key: "pid", title: "Paper", sub: "# and title (links to HotCRP)", num: false, sort: function (p) { return p.pid; } },
    ALL ? { key: "track", title: "Track", sub: "", num: true, sort: function (p) { return p.tracks[0] == null ? 999 : p.tracks[0]; } } : null,
    { key: "scores", title: "Scores", sub: "per review: original → post-rebuttal; ▲ champion, ▼ anti", sort: function (p) { return p.notRescored; } },
    { key: "avg", title: "Average", sub: "original → current", num: true, sort: function (p) { return p.avgCurrent == null ? 99 : p.avgCurrent; } },
    { key: "votes", title: "Votes", sub: "▲ champions · ▼ anti-champions (review letters)", sort: function (p) { return p.champions - p.antis; } },
    { key: "outcome", title: "Outcome", sub: "rule today, or settled tag; tags beneath", sort: function (p) { return OUTCOME_RANK[outcomeKey(p)] * 10 + (p.champions - p.antis); } },
    { key: "lead", title: "Discussion lead", sub: "posted since " + D.since + "?", sort: function (p) { return (p.leadPosted ? "0" : "1") + (p.lead || "~"); } },
    { key: "active", title: "Reviewers active", sub: "updated or commented since " + D.since + " / of submitted", num: true, sort: function (p) { return p.submitted ? p.activeSinceStart / p.submitted : -1; } },
    { key: "last", title: "Last activity", sub: "review, comment or response", sort: function (p) { return p.lastActivityAt; } },
    { key: "attention", title: "Needs attention", sub: "flags (see notes)", num: true, sort: function (p) { return p.attention * 100 + (p.champions + p.antis + p.bothLetters.length ? 50 : 0) + p.flags.length; } }
  ].filter(Boolean);

  var thead = document.querySelector("#papers thead");
  var tbody = document.querySelector("#papers tbody");
  function renderHead() {
    thead.textContent = "";
    var tr = el("tr", null, null, thead);
    COLS.forEach(function (c) {
      var th = el("th", c.num ? "num" : null, c.title, tr);
      if (c.sub) el("span", "sub", c.sub, th);
      th.tabIndex = 0;
      th.setAttribute("aria-sort", state.sort === c.key ? (state.dir === "asc" ? "ascending" : "descending") : "none");
      function toggle() {
        if (state.sort === c.key) state.dir = state.dir === "asc" ? "desc" : "asc";
        else { state.sort = c.key; state.dir = c.key === "attention" || c.key === "last" || c.key === "votes" ? "desc" : "asc"; }
        persist(); renderHead(); render();
      }
      th.addEventListener("click", toggle);
      th.addEventListener("keydown", function (ev) { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); toggle(); } });
    });
  }

  // ---- cells ----------------------------------------------------------------
  function scoresCell(td, p) {
    if (p.hidden) { el("span", "sub", "not visible to this account", td); return; }
    var box = el("span", "scores", null, td);
    p.reviewers.forEach(function (r) {
      var s;
      var who = r.letter ? "Review " + r.letter : "Review";
      // Blue = a post-rebuttal score exists (changed or not); grey = not re-scored yet.
      var cls = r.rescored ? "score rescored" : "score";
      if (r.changed) { s = el("span", cls, r.pre + "→" + r.post, box); s.title = who + ": " + r.pre + " originally, now " + r.post; }
      else { s = el("span", cls, String(r.current == null ? "?" : r.current), box); s.title = who + (r.rescored ? ": unchanged after the rebuttal" : ": original score; post-rebuttal score not entered"); }
      if (r.vote) {
        el("span", "mark " + (r.vote === "champion" ? "up" : r.vote === "anti" ? "down" : "both"), r.vote === "champion" ? "▲" : r.vote === "anti" ? "▼" : "▲▼", s);
        s.title += r.vote === "champion" ? "; champion" : r.vote === "anti" ? "; anti-champion" : "; ticked both champion and anti-champion (counted on neither side)";
      }
    });
    if (p.notRescored) el("span", "sub", p.notRescored + " of " + p.submitted + " not re-scored", td);
    else if (p.submitted) el("span", "sub ok", "all re-scored", td);
  }
  function avgCell(td, p) {
    if (p.hidden) { el("span", null, "–", td); return; }
    var span = el("span", "avg", null, td);
    var moved = p.avgPre != null && p.avgCurrent != null && Math.abs(p.avgPre - p.avgCurrent) > 1e-9;
    if (moved) { el("span", null, avg(p.avgPre), span); el("span", "arrow", "→", span); el("strong", null, avg(p.avgCurrent), span); }
    else el("span", null, avg(p.avgCurrent), span);
    el("span", "sub", p.submitted + " review" + (p.submitted === 1 ? "" : "s"), td);
  }
  function leadCell(td, p) {
    if (!p.lead) { el("span", "warn", "none", td); return; }
    el("span", null, p.lead, td);
    if (p.hidden) el("span", "sub", "activity not visible", td);
    else if (p.leadPosted) el("span", "sub ok", "posted " + p.leadPosted + (p.leadComments > 1 ? " (" + p.leadComments + " comments)" : ""), td);
    else el("span", "sub warn", "no comment since " + D.since, td);
  }
  function activeCell(td, p) {
    if (p.hidden) { el("span", null, "–", td); return; }
    var cls = p.submitted && p.activeSinceStart === p.submitted ? "ok" : p.activeSinceStart ? "" : "warn";
    el("span", cls, p.activeSinceStart + " / " + p.submitted, td);
    if (p.activeSinceResponse != null) el("span", "sub", p.activeSinceResponse + " / " + p.submitted + " since response", td);
    else el("span", "sub", "since " + D.since, td);
  }
  function votesCell(td, p) {
    if (p.hidden) { el("span", "sub", "not visible", td); return; }
    var span = el("span", "votes", null, td);
    el("span", "up", "▲ " + p.champions, span);
    el("span", "sep", "·", span);
    el("span", "down", "▼ " + p.antis, span);
    var who = [];
    if (p.championLetters.length) who.push("▲ " + p.championLetters.join(", "));
    if (p.antiLetters.length) who.push("▼ " + p.antiLetters.join(", "));
    if (p.bothLetters.length) who.push("both: " + p.bothLetters.join(", "));
    el("span", "sub", who.length ? who.join("  ") : "no votes yet", td);
  }
  // What the Outcome column shows: the rule's verdict on a paper in
  // discussion, the settled tag otherwise, "reopen?" over both.
  var OUTCOME_RANK = { reopen: 0, fix: 0, "no-champion": 1, outvoted: 2, hidden: 3, advances: 4, "settled-accept": 5, "settled-reject": 6 };
  function outcomeKey(p) {
    if (p.flags.indexOf("reopen") >= 0) return "reopen";
    if (p.phase === "conflicting" || p.phase === "untagged") return "fix";
    if (p.phase === "discuss") return p.projection || "hidden";
    return p.phase;
  }
  function outcomeCell(td, p) {
    var k = outcomeKey(p);
    var text = { reopen: "reopen?", fix: p.phase === "untagged" ? "no phase tag" : "conflicting tags", "no-champion": p.antis ? "reject: anti-champion only" : "likely reject: no champion yet",
                 outvoted: p.champions === p.antis ? "reject: tied" : "reject: outvoted", hidden: "votes not visible", advances: "advances",
                 "settled-accept": "accept", "settled-reject": "reject" }[k];
    el("span", "pill " + (k === "fix" ? "reopen" : k === "hidden" ? "none" : k), text, td);
    if (p.phaseTags.length) el("span", "sub", p.phaseTags.join(" + "), td);
    if (p.revision) el("span", "sub", p.revision === "advance" ? D.tags.advance : D.tags.none, td);
  }
  function flagsCell(td, p) {
    var box = el("div", "flags", null, td);
    p.flags.forEach(function (f) { if (LEVEL[f] === "act") el("span", "flag " + (f === "hidden" ? "hidden" : "act"), LABEL[f], box); });
  }

  // ---- table ----------------------------------------------------------------
  function visible() {
    var q = (state.q || "").trim().toLowerCase();
    return D.papers.filter(function (p) {
      if (ALL && state.track !== "" && p.tracks.indexOf(Number(state.track)) < 0) return false;
      if (state.phase && p.phaseGroup !== state.phase) return false;
      if (state.onlyAct && !p.attention) return false;
      if (q && String(p.pid).indexOf(q) < 0 && p.title.toLowerCase().indexOf(q) < 0 && (p.lead || "").toLowerCase().indexOf(q) < 0) return false;
      return true;
    });
  }
  function sorted(list) {
    var col = COLS.filter(function (c) { return c.key === state.sort; })[0] || COLS[COLS.length - 1];
    var sign = state.dir === "asc" ? 1 : -1;
    return list.slice().sort(function (a, b) {
      if (a.group !== b.group) return a.group ? -1 : 1;   // the conflict chair's group stays on top
      if (a.phaseGroup !== b.phaseGroup) return D.groups.indexOf(a.phaseGroup) - D.groups.indexOf(b.phaseGroup);
      var ka = col.sort(a), kb = col.sort(b);
      var c = typeof ka === "number" && typeof kb === "number" ? ka - kb : cmpText(String(ka), String(kb));
      if (c !== 0) return sign * c;
      if (col.key !== "attention" && a.attention !== b.attention) return b.attention - a.attention;
      if (a.avgCurrent != null && b.avgCurrent != null && a.avgCurrent !== b.avgCurrent) return a.avgCurrent - b.avgCurrent;
      return a.pid - b.pid;
    });
  }
  function render() {
    tbody.textContent = "";
    var list = sorted(visible());
    document.getElementById("shown").textContent = list.length === D.papers.length ? D.papers.length + " papers" : list.length + " of " + D.papers.length + " papers";
    if (!list.length) {
      var tr0 = el("tr", null, null, tbody);
      var td0 = el("td", "empty", "No paper matches.", tr0);
      td0.colSpan = COLS.length;
      return;
    }
    var lastGroup = null;
    var groupSize = {};
    list.forEach(function (p) { var k = p.group + "|" + p.phaseGroup; groupSize[k] = (groupSize[k] || 0) + 1; });
    list.forEach(function (p) {
      var key = p.group + "|" + p.phaseGroup;
      if (key !== lastGroup) {
        lastGroup = key;
        var gtr = el("tr", "group", null, tbody);
        var gtd = el("td", null, (p.group ? D.extraLabel + " · " : "") + GROUP_LABEL[p.phaseGroup] + " (" + groupSize[key] + ")", gtr);
        gtd.colSpan = COLS.length;
      }
      var tr = el("tr", "paper" + (p.attention ? " act" : "") + (p.hidden ? " hidden-paper" : "") + (p.group ? " extra" : ""), null, tbody);
      COLS.forEach(function (c) {
        var td = el("td", c.num ? "num" : (c.key === "pid" ? "title" : null), null, tr);
        switch (c.key) {
          case "pid":
            el("span", "pid", "#" + p.pid, td);
            var a = el("a", null, p.title || "(untitled)", td);
            a.href = p.url; a.target = "_blank"; a.rel = "noopener";
            break;
          case "track": td.textContent = p.tracks.length ? p.tracks.join(", ") : "–"; break;
          case "scores": scoresCell(td, p); break;
          case "avg": avgCell(td, p); break;
          case "lead": leadCell(td, p); break;
          case "active": activeCell(td, p); break;
          case "votes": votesCell(td, p); break;
          case "outcome": outcomeCell(td, p); break;
          case "last": td.textContent = p.lastActivity || "–"; break;
          case "attention": flagsCell(td, p); break;
        }
      });
    });
  }

  // ---- summary --------------------------------------------------------------
  var papers = D.papers;
  var count = function (key) { return papers.filter(function (p) { return p.flags.indexOf(key) >= 0; }).length; };
  var act = papers.filter(function (p) { return p.attention > 0; }).length;
  var own = papers.length - (D.extraCount || 0);
  document.getElementById("lede").textContent =
    (D.extraCount ? (own ? own + " paper" + (own === 1 ? "" : "s") + " without a decision in the track, plus " : "") + D.extraCount + " paper" + (D.extraCount === 1 ? "" : "s") + " the PC chairs are conflicted with" + (own ? "" : ", without a decision") : papers.length + " paper" + (papers.length === 1 ? "" : "s") + " without a decision") +
    (ALL ? " across " + D.tracks.length + " tracks" : "") +
    (D.omitted ? ", not counting " + D.omitted + " that " + (ALL ? (D.reader ? "you are" : "a PC chair is") : "the track chair is") + " conflicted with" : "") +
    ". Generated " + D.generatedAt + (PRINT ? "" : "") +
    "; activity counted since " + D.sinceLong + (PRINT ? "." : ". Click a column to sort.");
  var tiles = document.getElementById("tiles");
  var discuss = papers.filter(function (p) { return p.phase === "discuss"; });
  var proj = function (k) { return discuss.filter(function (p) { return p.projection === k; }).length; };
  [
    ["In discussion", discuss.length, ""],
    ["Advancing", proj("advances"), ""],
    ["Anti-champion, no champion", discuss.filter(function (p) { return p.projection === "no-champion" && p.antis > 0; }).length, "act"],
    ["No votes yet", discuss.filter(function (p) { return p.projection === "no-champion" && !p.antis; }).length, "act"],
    ["Outvoted / tied", proj("outvoted"), "act"],
    ["Reopen?", count("reopen"), "act"],
    ["Settled accept", papers.filter(function (p) { return p.phase === "settled-accept"; }).length, ""],
    ["Settled reject", papers.filter(function (p) { return p.phase === "settled-reject"; }).length, ""],
    ["Needs attention", act, "act"]
  ].forEach(function (t) {
    var d = el("div", "tile" + (t[2] && t[1] ? " act" : ""), null, tiles);
    el("div", "label", t[0], d);
    el("div", "value", String(t[1]), d);
  });

  var flagNotes = document.getElementById("flag-notes");
  [
    ["The rule", "a paper tagged " + tagList(D.tags.discuss) + " advances only if it has more champions than anti-champions. With no champion it is rejected, and so it is when anti-champions equal or outnumber champions: a tie rejects. Votes are the Champion / Anti-champion boxes on the review form, counted over submitted PC reviews only; a review that ticks both counts on neither side and is flagged."],
    ["Settled papers", "papers tagged " + tagList(D.tags.accept) + " or " + tagList(D.tags.reject) + " are already decided and are listed only so a contrary vote is noticed: an anti-champion on a settled accept, or a champion on a settled reject, marks the paper reopen? and moves it to the top. A vote that agrees with the settled outcome is shown, not flagged. The lead and silence flags apply only to papers still in discussion."],
    ["Champions", "▲ champions · ▼ anti-champions, with the review letters beneath; the same marks appear on the score chips."],
    ["Outcome", "for a paper in discussion, what the rule gives with the votes as they stand -- \"reject\" once an anti-champion equals or outnumbers its champions, \"likely reject\" while nobody has voted; for a settled paper, its tag. The tags themselves are beneath."],
    ["Scores", "one chip per submitted PC review: the original pre-rebuttal overall merit, or original→current once the reviewer has entered a post-rebuttal score. Blue chips have a post-rebuttal score (changed or not); grey chips are still waiting for one. Beneath, how many reviews have no post-rebuttal score yet. Reviews never submitted are assumed not to be coming and are not shown."],
    ["Average", "mean of the current scores over submitted PC reviews; original → current when a score changed."],
    ["Lead", "whether the discussion lead has posted a comment on the paper since " + D.since + " (not a draft, not an author response, not admin-only)."],
    ["Reviewers active", "submitted reviewers who have updated their review or commented since " + D.since + "; and, once the authors responded, since that response."],
    ["Needs attention", D.flags.filter(function (f) { return f.level === "act"; }).map(function (f) { return f.label; }).join(", ") + ": each tints the row and is listed in the last column. " + D.flags.filter(function (f) { return f.level === "info"; }).map(function (f) { return f.label; }).join(", ") + " are visible in the Scores, Champions and Outcome columns and do not count."]
  ].forEach(function (n) {
    var li = el("li", null, null, flagNotes);
    el("strong", null, n[0] + ": ", li);
    li.appendChild(document.createTextNode(n[1]));
  });
  var dataNotes = document.getElementById("data-notes");
  D.notes.forEach(function (n) { el("li", null, n, dataNotes); });
  if (D.extraCount) {
    el("li", null, "The group \"" + D.extraLabel + "\" holds the papers under review that every PC chair is conflicted with, from every track; they are in your charge as conflict chair and stay at the top under any sort.", dataNotes);
  }
  if (D.omitted) {
    el("li", null, D.omitted + " paper" + (D.omitted === 1 ? "" : "s") + " under review " + (ALL ? "in these tracks are" : "in this track is") + " left off this page because " + (ALL ? (D.reader ? D.reader : "one of the PC chairs") : "the track chair") + " is conflicted with " + (D.omitted === 1 ? "it" : "them") + ", as HotCRP records it.", dataNotes);
  }
  if (papers.some(function (p) { return p.hidden; })) {
    el("li", null, "A paper marked hidden is one the account generating this page is conflicted with: HotCRP shows it no reviews and no comments, so its scores and activity are unknown here, not absent.", dataNotes);
  }

  renderHead();
  render();
})();
</script>
"""


if __name__ == "__main__":
    sys.exit(main())
