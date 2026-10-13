"""Nudge discussion leads and reviewers who are behind, as HotCRP comments.

    # Who is the token?  (needs HOTCRP_SITE and HOTCRP_TOKEN in .env)
    python -m scripts.discussion_nudges --whoami

    # Leads who have posted no starter since the response deadline:
    python -m scripts.discussion_nudges --leads-without-starter --since 2026-10-10 \\
        --due 'Thursday, October 15'

    # Reviewers who have neither updated their review nor commented since:
    python -m scripts.discussion_nudges --reviewers-silent --since 2026-10-10 \\
        --due 'Friday'

    # Then, once the preview reads right:
    python -m scripts.discussion_nudges --send outputs/assignments/nudges.json

    # A test comment to yourself on the standing test paper, and its removal:
    python -m scripts.discussion_nudges --papers 1152 --mention-emails chair@x.edu \\
        --text 'API test: {mentions}'
    python -m scripts.discussion_nudges --delete-comment 1152:4321

Two stages, deliberately separate. **Generate** reads the live discussion
state through HotCRP's API (`reviewer_match.hotcrp_api`) and writes a batch:
`outputs/assignments/nudges.json` (what would be posted) and a preview,
`outputs/reports/nudges.txt`, in the `timeliness_emails.txt` block style.
Nothing is posted. **Send** (`--send BATCH`) posts that batch, one comment per
paper, first as a dry run of every item and then for real, and records every
(paper, person) pair in `outputs/reports/nudge_ledger.csv`. The chair reads
the preview between the two; the day's condition is a matter of flags, so the
same tool serves "leads with no starter" on Monday and "reviewers who have
not updated their scores or joined the discussion" on Wednesday.

**Why comments.** A reviewer-visible comment that @-mentions someone makes
HotCRP email them (mention notifications), and the paper's other reviewers
see it in the thread. No mail credentials, no browser, no bulk-mail page, and
HotCRP's own log records it (`Sent mail #N`).

**Scope.** Papers tagged `RevisionAdvance` in the export, still under review
(`review_scores.eligible_pids`), with no decision, minus `--exclude-pids`.
`--all-undecided` drops the tag test; `--papers` names papers outright;
`--scope-search` takes a HotCRP search. The API acts as the token's owner,
so papers they are conflicted with return nothing: a paper in scope with no
reviews in the reply is listed on stderr as **unseen**, never silently
skipped. The same hole the leads download has.

**Conditions** (each adds a group; every group's count is reported):

- `--leads-without-starter`: the paper's lead (leads download + log, as
  `assign_paper_leads` merges them) has no discussion comment on it created
  on/after `--since`. A discussion comment is one that is not a draft, not a
  response, not by an author and not admin-only. Creation is the comment's
  first log row (`hotcrp_log.comment_created_at`), falling back to the API's
  `modified_at`. A paper with no lead is reported, not nudged.
- `--reviewers-silent`: a PC reviewer (review type >= 2, round not TRC) with a
  submitted review whose review was last modified before `--since` **and**
  who has no discussion comment on the paper since then. `--require-both`
  widens it to anyone missing either.
- `--reviewers-missing-review`: PC reviewers whose review is not submitted.
- `--mention-emails`: named people on every paper in scope (for a test).

`--cooldown-days` skips anyone the ledger shows nudged for the same kind on
the same paper that recently; `--exclude-emails` excuses people outright, and
so do the `extension` rows of `--extensions`, but only when it is passed: an
R1 review extension does not excuse a lead from the post-revision discussion,
so nothing reads `review_extensions.csv` by default.

**Message.** One comment per paper and kind, all of that paper's targets
mentioned together. `--lead-template` / `--reviewer-template` /
`--missing-template` name files in `templates/nudges/` (see its README for
the placeholders); `--text` replaces them all. Comments are posted
`--visibility rev` and tagged `nudge`.

**Mentions.** HotCRP's `MentionParser` emits a mention only when the text
after `@` matches **exactly one** candidate; a name two accounts share
(the chairs' own test accounts, for one) matches two and is silently left as
plain text -- no email, no highlight. The parser also accepts the full
address, `@user@site.edu`, which can only match one account. So a person is
mentioned by name (`@Given Family`, as the API spells it, then the user
export) when that name belongs to one account in the user export, and by
address otherwise. Somebody with no known name at all is mentioned by
address too. On a real save HotCRP answers `Mentioned user X` once per
resolved mention, and the send stage counts those against what it sent:
a shortfall is printed as UNRESOLVED.

**Threads.** HotCRP emails a mention only to someone who can see the
comment, and the dry run says when that fails ("Some mentioned users cannot
see this comment now"). A `rev`-topic comment lives on the review thread,
which a reviewer with their own review outstanding cannot see, and which
HotCRP hid from a *completed* reviewer on the test paper too -- the rule is
HotCRP's, not ours, so the dry run's answer is what counts. Live on
2026-10-12, review-thread nudges to the submitted reviewers and leads of real
RevisionAdvance papers passed. `--reviewers-missing-review` nudges go on the
submission thread (`topic=paper`) by default; `--topic` overrides for every
kind. An item the dry run flags is not posted and is listed, so the batch can
be regenerated with `--topic paper` (or `--visibility au` for an author)
rather than silently emailing nobody.

**Send-stage guards.** A batch older than `--max-age-hours` is refused (the
state it was generated from has moved). So is one that would re-nudge
anyone inside the cooldown, unless `--force`. A nudge whose (batch, paper,
kind) is already in the ledger is skipped, so re-running a half-finished send
posts only what is left. Every item is dry-run first; one error anywhere and
nothing is posted.

stdout is for results; everything else goes to stderr.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import textwrap
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone

from reviewer_match import assignment_io, hotcrp_api, hotcrp_log, pc_membership, review_scores
from reviewer_match.paths import assignment_path, cache_path, curated_path, input_path, report_path, template_path
from scripts import assign_paper_leads, timeliness_tags

DEFAULT_DATA = input_path("hpca2027-data.json")
DEFAULT_LOG = input_path("hpca2027-log.csv")
DEFAULT_LEADS = input_path("hpca2027-leads.csv")
DEFAULT_PCINFO = pc_membership.DEFAULT_PCINFO
DEFAULT_EXTENSIONS = curated_path("review_extensions.csv")
DEFAULT_BATCH = assignment_path("nudges.json")
DEFAULT_PREVIEW = report_path("nudges.txt")
DEFAULT_LEDGER = report_path("nudge_ledger.csv")
DEFAULT_SNAPSHOT = cache_path("hotcrp_api")
DEFAULT_COOLDOWN_DAYS = 3
DEFAULT_MAX_AGE_HOURS = 6
DEFAULT_VISIBILITY = "rev"
DEFAULT_TOPIC = "rev"
# The submission thread, visible to a reviewer whose own review is outstanding
# (the review thread is not).
PAPER_TOPIC = "paper"
DEFAULT_TAG = "nudge"
CANNOT_SEE = "cannot see this comment"
MENTIONED_RE = re.compile(r"Mentioned user")

# API review `status` values that mean "turned in" (HotCRP has no `submitted`
# status: `delivered` is the submitted state, `complete` a delivered review
# whose form is complete).
SUBMITTED_STATUSES = frozenset({"delivered", "complete"})
# `rtype`: 1 external, 2 PC, 3 secondary, 4 primary, 5 metareview.
PC_RTYPE_MIN = 2

KIND_LEAD = "lead_starter"
KIND_SILENT = "reviewer_silent"
KIND_MISSING = "reviewer_missing"
KIND_MANUAL = "manual"
DEFAULT_TEMPLATES = {
    KIND_LEAD: "leads_starter",
    KIND_SILENT: "reviewers_discuss",
    KIND_MISSING: "reviewers_missing",
}
KIND_LABELS = {
    KIND_LEAD: "lead without a discussion starter",
    KIND_SILENT: "reviewer silent since the cutoff",
    KIND_MISSING: "reviewer with a review outstanding",
    KIND_MANUAL: "named on the command line",
}

LEDGER_HEADER = ("sent_at", "batch", "pid", "cid", "kind", "email", "template")
BATCH_VERSION = 1


# ---------------------------------------------------------------------------
# What the API returns, reduced to what a nudge needs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Comment:
    pid: int
    cid: int
    author_email: str
    author: str
    visibility: str
    topic: str
    response: str
    draft: bool
    by_author: bool
    modified_at: int
    created_at: datetime

    @property
    def discussion(self) -> bool:
        """A comment that counts as taking part: not a draft, not a response,
        not an author's, not admin-only, and from a known account."""
        return bool(self.author_email) and not self.draft and not self.response \
            and not self.by_author and self.visibility != "admin"


@dataclass(frozen=True)
class ReviewState:
    pid: int
    rid: int
    email: str
    reviewer: str
    round: str
    rtype: int
    status: str
    modified_at: int

    @property
    def submitted(self) -> bool:
        return self.status in SUBMITTED_STATUSES

    @property
    def pc(self) -> bool:
        return self.rtype >= PC_RTYPE_MIN and self.round != review_scores.TRAINING_ROUND


@dataclass(frozen=True)
class Target:
    pid: int
    kind: str
    email: str
    note: str = ""


@dataclass
class Nudge:
    pid: int
    kind: str
    template: str
    text: str
    visibility: str
    topic: str
    tags: str
    mentions: list[str]
    names: list[str]
    title: str


def _epoch_to_dt(value) -> datetime:
    try:
        return datetime.fromtimestamp(int(value or 0), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return datetime.fromtimestamp(0, tz=timezone.utc)


def parse_comments(objs: list[dict], created: dict[int, str] | None = None) -> list[Comment]:
    """API comment objects -> Comment, creation dated from the log where it can be."""
    out = []
    for o in objs:
        try:
            pid, cid = int(o["pid"]), int(o["cid"])
        except (KeyError, TypeError, ValueError):
            continue
        stamp = (created or {}).get(cid)
        created_at = hotcrp_log.parse_date(stamp) if stamp else _epoch_to_dt(o.get("modified_at"))
        out.append(Comment(
            pid=pid, cid=cid,
            author_email=(o.get("author_email") or "").strip().lower(),
            author=(o.get("author") or "").strip(),
            visibility=o.get("visibility") or "rev",
            topic=o.get("topic") or "rev",
            response=o.get("response") or "",
            draft=bool(o.get("draft")),
            by_author=bool(o.get("by_author")),
            modified_at=int(o.get("modified_at") or 0),
            created_at=created_at,
        ))
    return out


def parse_reviews(objs: list[dict]) -> list[ReviewState]:
    out = []
    for o in objs:
        try:
            pid, rid = int(o["pid"]), int(o["rid"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append(ReviewState(
            pid=pid, rid=rid,
            email=(o.get("reviewer_email") or "").strip().lower(),
            reviewer=(o.get("reviewer") or "").strip(),
            round=str(o.get("round") or ""),
            rtype=int(o.get("rtype") or 0),
            status=str(o.get("status") or ""),
            modified_at=int(o.get("modified_at") or 0),
        ))
    return out


# ---------------------------------------------------------------------------
# Scope and inputs
# ---------------------------------------------------------------------------

def load_papers(path: str) -> dict[int, dict]:
    with open(path, encoding="utf-8") as f:
        return {int(p["pid"]): p for p in json.load(f)}


def scope_pids(data_path: str, exclude: frozenset[int], *, all_undecided: bool = False) -> set[int]:
    """Papers on the discussion timeline: under review, undecided, and (by
    default) tagged RevisionAdvance."""
    under_review, _ = review_scores.eligible_pids(data_path, exclude)
    papers = load_papers(data_path)
    undecided = {pid for pid in under_review if not papers.get(pid, {}).get("decision")}
    if all_undecided:
        return undecided
    return undecided & review_scores.tagged_pids(data_path, review_scores.ADVANCE_TAG)


def parse_since(value: str) -> datetime:
    """`--since`: a date (midnight, local offset) or a full log-style timestamp."""
    value = value.strip()
    try:
        day = date.fromisoformat(value)
    except ValueError:
        return hotcrp_log.parse_date(value)
    local = datetime.now().astimezone().tzinfo
    return datetime.combine(day, datetime.min.time(), tzinfo=local)


def parse_pids(value: str) -> set[int]:
    return {int(p.strip().lstrip("#")) for p in re.split(r"[,\s]+", value or "") if p.strip()}


def parse_emails(value: str) -> set[str]:
    return {e.strip().lower() for e in re.split(r"[,\s]+", value or "") if e.strip()}


def load_email_file(path: str) -> set[str]:
    """One address per line (or the first CSV column); `#` lines are comments."""
    out = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            first = line.split(",")[0].strip().lower()
            if first and first != "email" and "@" in first:
                out.add(first)
    return out


def current_leads(leads_path: str | None, log_rows: list[dict[str, str]]) -> tuple[dict[int, str], list[int]]:
    """{pid: lead email} as `assign_paper_leads` sees them, and the pids the
    download hid (taken from the log)."""
    log_leads = hotcrp_log.replay_leads(log_rows)
    if leads_path and os.path.exists(leads_path):
        download = assignment_io.load_leads(leads_path)
    else:
        download = {}
    leads, hidden, _ = assign_paper_leads.merge_hidden_leads(download, log_leads)
    return leads, hidden


def fetch_state(site: hotcrp_api.Site, pids: set[int]) -> tuple[list[dict], list[dict]]:
    """Raw review and comment objects for every paper in `pids`, in chunks."""
    reviews: list[dict] = []
    comments: list[dict] = []
    for q in hotcrp_api.pid_query(pids):
        reviews += site.reviews(q)
        comments += site.comments(q)
    return reviews, comments


def fold_name(name: str) -> str:
    return " ".join(name.split()).casefold()


def name_index(reviews: list[ReviewState], pcinfo_path: str | None) -> tuple[dict[str, str], set[str]]:
    """({email: display name}, {folded name held by 2+ accounts}).

    Names: the user export first, overridden by the API's live spelling
    wherever a review carries one (a mention has to match the name HotCRP
    holds *now*, and the export can be a week old). The ambiguous set is
    every name that two or more enabled accounts in the export share -- a
    mention by that name matches both and so neither (see the docstring)."""
    names: dict[str, str] = {}
    held: Counter = Counter()
    if pcinfo_path:
        try:
            index = pc_membership.load_pc_accounts(pcinfo_path)
        except (FileNotFoundError, ValueError) as exc:
            print(f"WARNING: {exc}; names come from the API only, and no name is known to be "
                  f"ambiguous", file=sys.stderr)
        else:
            for email, acct in index.by_email.items():
                if acct.name:
                    names[email] = acct.name
                    if not acct.disabled:
                        held[fold_name(acct.name)] += 1
    for r in reviews:
        if r.email and r.reviewer:
            names[r.email] = r.reviewer
    return names, {name for name, n in held.items() if n > 1}


def mention(email: str, names: dict[str, str], ambiguous: set[str]) -> tuple[str, str]:
    """(what goes after the @, display name). By name when that name is one
    account's; by address when it is shared or unknown."""
    name = names.get(email, "")
    if name and fold_name(name) not in ambiguous:
        return name, name
    return email, name or email


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

def _by_pid(comments: list[Comment]) -> dict[int, list[Comment]]:
    out: dict[int, list[Comment]] = defaultdict(list)
    for c in comments:
        out[c.pid].append(c)
    return out


def commented_since(comments: list[Comment], email: str, since: datetime) -> bool:
    return any(c.discussion and c.author_email == email and c.created_at >= since for c in comments)


def leads_without_starter(
    pids: set[int], leads: dict[int, str], comments: list[Comment], since: datetime
) -> tuple[list[Target], list[int]]:
    """(targets, [pid with no lead])."""
    by_pid = _by_pid(comments)
    targets, no_lead = [], []
    for pid in sorted(pids):
        lead = leads.get(pid)
        if not lead:
            no_lead.append(pid)
            continue
        if commented_since(by_pid.get(pid, []), lead, since):
            continue
        targets.append(Target(pid, KIND_LEAD, lead, "no comment by the lead since the cutoff"))
    return targets, no_lead


def reviewers_silent(
    pids: set[int], reviews: list[ReviewState], comments: list[Comment], since: datetime,
    *, require_both: bool = False,
) -> list[Target]:
    by_pid = _by_pid(comments)
    cutoff = since.timestamp()
    targets = []
    for r in sorted(reviews, key=lambda r: (r.pid, r.email, r.rid)):
        if r.pid not in pids or not r.pc or not r.submitted or not r.email:
            continue
        updated = r.modified_at >= cutoff
        commented = commented_since(by_pid.get(r.pid, []), r.email, since)
        silent = not (updated and commented) if require_both else not (updated or commented)
        if not silent:
            continue
        missing = [what for what, done in (("review update", updated), ("comment", commented)) if not done]
        targets.append(Target(r.pid, KIND_SILENT, r.email, "no " + ", no ".join(missing)))
    return targets


def reviewers_missing_review(pids: set[int], reviews: list[ReviewState]) -> list[Target]:
    targets = []
    for r in sorted(reviews, key=lambda r: (r.pid, r.email, r.rid)):
        if r.pid not in pids or not r.pc or r.submitted or not r.email:
            continue
        targets.append(Target(r.pid, KIND_MISSING, r.email, f"review {r.status or 'not started'}"))
    return targets


def dedupe(targets: list[Target]) -> list[Target]:
    seen: set[tuple[int, str, str]] = set()
    out = []
    for t in targets:
        key = (t.pid, t.kind, t.email)
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

def load_ledger(path: str) -> list[dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = set(LEDGER_HEADER) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: ledger lacks {', '.join(sorted(missing))}")
        return list(reader)


def append_ledger(path: str, rows: list[dict[str, str]]) -> None:
    existing = load_ledger(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_HEADER, extrasaction="ignore")
        w.writeheader()
        for row in existing + rows:
            w.writerow(row)
    os.replace(tmp, path)


def recent_keys(ledger: list[dict[str, str]], now: datetime, days: int) -> set[tuple[int, str, str]]:
    """{(pid, kind, email)} nudged within the last `days` days."""
    if days <= 0:
        return set()
    floor = now - timedelta(days=days)
    out = set()
    for row in ledger:
        try:
            when = datetime.fromisoformat(row["sent_at"])
        except ValueError:
            continue
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        if when >= floor:
            out.add((int(row["pid"]), row["kind"], row["email"].lower()))
    return out


def sent_in_batch(ledger: list[dict[str, str]], batch_id: str) -> set[tuple[int, str]]:
    return {(int(r["pid"]), r["kind"]) for r in ledger if r["batch"] == batch_id}


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

class _Fields(dict):
    def __missing__(self, key):
        raise KeyError(key)


def load_template(spec: str) -> str:
    """`spec` is a path, or a name under templates/nudges/."""
    path = spec if os.path.exists(spec) else template_path(os.path.join("nudges", f"{spec}.txt"))
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"no nudge template {spec!r} (looked for {path})") from None


def render(template: str, **fields) -> str:
    try:
        return template.format_map(_Fields(fields))
    except KeyError as exc:
        raise ValueError(f"template uses unknown placeholder {{{exc.args[0]}}}; known: "
                         + ", ".join(sorted(fields))) from None


def build_nudges(
    targets: list[Target], names: dict[str, str], titles: dict[int, str], leads: dict[int, str],
    templates: dict[str, str], *, text: str | None = None, due: str = "",
    visibility: str = DEFAULT_VISIBILITY, topic: str | None = None, tag: str = DEFAULT_TAG,
    ambiguous: set[str] = frozenset(),
) -> tuple[list[Nudge], list[str]]:
    """One Nudge per (paper, kind), everyone of that kind mentioned together.
    Also returns the addresses mentioned by address rather than name (no name
    known, or a name two accounts share). `topic` None means the review
    thread, except the submission thread for the missing-review kind."""
    grouped: dict[tuple[int, str], list[Target]] = defaultdict(list)
    for t in targets:
        grouped[(t.pid, t.kind)].append(t)
    by_address: set[str] = set()
    nudges = []
    for (pid, kind), group in sorted(grouped.items()):
        emails = sorted({t.email for t in group}, key=lambda e: (names.get(e, "~" + e).lower(), e))
        handles, shown = [], []
        for e in emails:
            handle, name = mention(e, names, ambiguous)
            if handle == e:
                by_address.add(e)
            handles.append(handle)
            shown.append(name)
        kind_topic = topic or (PAPER_TOPIC if kind == KIND_MISSING else DEFAULT_TOPIC)
        if text is not None:
            template_name, body = "--text", text
        else:
            template_name = templates.get(kind) or DEFAULT_TEMPLATES.get(kind)
            if not template_name:
                raise ValueError(f"no template for kind {kind!r}; pass --text")
            body = load_template(template_name)
        lead_name = names.get(leads.get(pid, ""), "") or "the discussion lead"
        rendered = render(
            body,
            mentions=", ".join("@" + h for h in handles),
            names=", ".join(shown),
            n=len(shown),
            pid=pid,
            title=titles.get(pid, ""),
            lead=lead_name,
            due=due or "",
            due_clause=f" by {due}" if due else "",
        )
        nudges.append(Nudge(pid=pid, kind=kind, template=template_name, text=rendered,
                            visibility=visibility, topic=kind_topic, tags=tag, mentions=emails,
                            names=shown, title=titles.get(pid, "")))
    return nudges, sorted(by_address)


def preview_text(nudges: list[Nudge]) -> str:
    blocks = []
    for n in nudges:
        who = ", ".join(f"{name} <{email}>" for name, email in zip(n.names, n.mentions))
        thread = "submission thread" if n.topic == PAPER_TOPIC else "review thread"
        head = (f"===== PAPER {n.pid} =====\n{n.title}\nKind: {KIND_LABELS.get(n.kind, n.kind)} "
                f"({thread}, {n.visibility})\nTo: {who}\n")
        body = "\n".join(textwrap.fill(p, 78) if p else "" for p in n.text.split("\n"))
        blocks.append(head + "\n" + body)
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def write_batch(path: str, batch: dict) -> None:
    write_text(path, json.dumps(batch, indent=1, sort_keys=True) + "\n")


def load_batch(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        batch = json.load(f)
    if batch.get("version") != BATCH_VERSION or "nudges" not in batch:
        raise ValueError(f"{path}: not a nudge batch this script wrote")
    return batch


# ---------------------------------------------------------------------------
# Send
# ---------------------------------------------------------------------------

def send_batch(
    site: hotcrp_api.Site, batch: dict, ledger_path: str, *, now: datetime,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS, cooldown_days: int = DEFAULT_COOLDOWN_DAYS,
    force: bool = False, notify: bool = True, out=sys.stdout, err=sys.stderr,
) -> tuple[int, int]:
    """Post every nudge in `batch` not already in the ledger.
    Returns (posted, held back because a mentioned person cannot see the thread)."""
    batch_id = batch["generated_at"]
    generated = datetime.fromisoformat(batch_id)
    age = now - generated
    if age > timedelta(hours=max_age_hours) and not force:
        raise SystemExit(f"ERROR: batch generated {age} ago (over {max_age_hours}h); regenerate it, "
                         f"or --force to post it anyway")
    if batch.get("site") and batch["site"].rstrip("/") != site.base_url:
        raise SystemExit(f"ERROR: batch was generated against {batch['site']}, token is for {site.base_url}")

    ledger = load_ledger(ledger_path)
    done = sent_in_batch(ledger, batch_id)
    recent = recent_keys(ledger, now, cooldown_days)
    todo, skipped, blocked = [], [], []
    for item in batch["nudges"]:
        n = Nudge(**item)
        if (n.pid, n.kind) in done:
            skipped.append(n)
            continue
        hot = [e for e in n.mentions if (n.pid, n.kind, e) in recent]
        if hot and not force:
            blocked.append((n, hot))
            continue
        todo.append(n)
    if skipped:
        print(f"{len(skipped)} nudge(s) already posted from this batch; skipping them", file=err)
    if blocked:
        print(f"ERROR: {len(blocked)} nudge(s) would re-nudge someone within {cooldown_days} days:", file=err)
        for n, hot in blocked:
            print(f"  #{n.pid} {n.kind}: {', '.join(hot)}", file=err)
        raise SystemExit("ERROR: refusing; regenerate with a longer --cooldown-days, or --force")
    if not todo:
        print("Nothing to post.", file=err)
        return 0, 0

    # Dry-run everything before posting anything. An item some mentioned
    # person cannot see is held back (HotCRP would save it and email nobody),
    # not fatal; anything else wrong is fatal for the whole batch.
    problems = 0
    unseeable = []
    postable = []
    for n in todo:
        try:
            resp = site.post_comment(n.pid, n.text, visibility=n.visibility, topic=n.topic,
                                     tags=n.tags, notify=notify, dry_run=True)
        except hotcrp_api.HotcrpError as exc:
            print(f"  #{n.pid} {n.kind}: {exc}", file=err)
            problems += 1
            continue
        messages = resp.get("message_list") or []
        texts = [hotcrp_api.format_message(m) for m in messages]
        if any(CANNOT_SEE in t for t in texts):
            unseeable.append(n)
            continue
        for m in hotcrp_api.warnings(messages):
            print(f"  #{n.pid} {n.kind}: warning: {hotcrp_api.format_message(m)}", file=err)
        bad = hotcrp_api.errors(messages)
        if bad or resp.get("valid") is False:
            for m in bad:
                print(f"  #{n.pid} {n.kind}: {hotcrp_api.format_message(m)}", file=err)
            problems += 1
            continue
        postable.append(n)
    if problems:
        raise SystemExit(f"ERROR: {problems} of {len(todo)} nudges failed the dry run; nothing posted")
    if unseeable:
        print(f"NOT POSTED: {len(unseeable)} nudge(s) mention someone HotCRP says cannot see that "
              f"thread (an author, or a reviewer the review thread is hidden from); regenerate them "
              f"with --topic {PAPER_TOPIC}, or --visibility au for an author:", file=err)
        for n in unseeable:
            print(f"  #{n.pid} {n.kind}: {', '.join(n.mentions)}", file=err)
    if not postable:
        print("Nothing to post.", file=err)
        return 0, len(unseeable)
    print(f"Dry run passed for {len(postable)} nudge(s); posting", file=err)

    posted = 0
    unresolved = 0
    for n in postable:
        resp = site.post_comment(n.pid, n.text, visibility=n.visibility, topic=n.topic,
                                 tags=n.tags, notify=notify, dry_run=False)
        cid = resp.get("cid") or (resp.get("comment") or {}).get("cid") or ""
        confirmed = sum(1 for m in resp.get("message_list") or ()
                        if MENTIONED_RE.search(hotcrp_api.format_message(m)))
        if confirmed < len(n.mentions):
            unresolved += len(n.mentions) - confirmed
            print(f"  UNRESOLVED: comment {cid} on #{n.pid} mentions {len(n.mentions)}, HotCRP resolved "
                  f"{confirmed}: {', '.join(n.mentions)}", file=err)
        stamp = now.isoformat(timespec="seconds")
        append_ledger(ledger_path, [
            {"sent_at": stamp, "batch": batch_id, "pid": str(n.pid), "cid": str(cid),
             "kind": n.kind, "email": e, "template": n.template}
            for e in n.mentions
        ])
        posted += 1
        print(f"#{n.pid}\t{n.kind}\tcomment {cid}\t{', '.join(n.mentions)}", file=out)
    print(f"Posted {posted} comment(s); ledger {ledger_path}", file=err)
    if unresolved:
        print(f"WARNING: {unresolved} mention(s) did not resolve to an account; those people were "
              f"not emailed", file=err)
    return posted, len(unseeable)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    io = p.add_argument_group("inputs")
    io.add_argument("--data", default=DEFAULT_DATA, help="HotCRP paper export JSON (scope, titles)")
    io.add_argument("--log", default=DEFAULT_LOG, help="HotCRP action log (comment creation dates, leads)")
    io.add_argument("--leads", default=DEFAULT_LEADS, help="HotCRP discussion-leads download")
    io.add_argument("--pcinfo", default=DEFAULT_PCINFO, help="HotCRP user export (names)")
    io.add_argument("--exclude-pids", default="", help="comma-separated paper ids never in scope")
    io.add_argument("--snapshot", default=DEFAULT_SNAPSHOT, help="where API replies are saved")
    io.add_argument("--offline", action="store_true", help="answer every GET from --snapshot")

    sc = p.add_argument_group("scope")
    sc.add_argument("--papers", default="", help="comma-separated paper ids: exactly these")
    sc.add_argument("--scope-search", default="", help="a HotCRP search giving the papers")
    sc.add_argument("--all-undecided", action="store_true", help="every undecided paper, not just RevisionAdvance")

    co = p.add_argument_group("conditions")
    co.add_argument("--since", help="cutoff: YYYY-MM-DD (local midnight) or a log-style timestamp")
    co.add_argument("--leads-without-starter", action="store_true")
    co.add_argument("--reviewers-silent", action="store_true")
    co.add_argument("--require-both", action="store_true", help="silent = missing a review update OR a comment")
    co.add_argument("--reviewers-missing-review", action="store_true")
    co.add_argument("--mention-emails", default="", help="comma-separated addresses to mention on every scope paper")
    co.add_argument("--cooldown-days", type=int, default=DEFAULT_COOLDOWN_DAYS)
    co.add_argument("--exclude-emails", help="file of addresses never nudged")
    co.add_argument("--extensions", default=None,
                    help=f"review_extensions.csv (e.g. {DEFAULT_EXTENSIONS}); its `extension` rows are never "
                         f"nudged. Off unless passed")

    ms = p.add_argument_group("message")
    ms.add_argument("--lead-template", default=DEFAULT_TEMPLATES[KIND_LEAD])
    ms.add_argument("--reviewer-template", default=DEFAULT_TEMPLATES[KIND_SILENT])
    ms.add_argument("--missing-template", default=DEFAULT_TEMPLATES[KIND_MISSING])
    ms.add_argument("--text", help="the whole message (placeholders allowed); overrides every template")
    ms.add_argument("--due", default="", help="deadline wording for {due} / {due_clause}")
    ms.add_argument("--visibility", default=DEFAULT_VISIBILITY, choices=("admin", "pc", "rev", "au"))
    ms.add_argument("--topic", default=None, choices=(DEFAULT_TOPIC, PAPER_TOPIC),
                    help=f"thread: {DEFAULT_TOPIC} (review thread; default) or {PAPER_TOPIC} (submission "
                         f"thread, visible to reviewers with a review outstanding; default for "
                         f"--reviewers-missing-review)")
    ms.add_argument("--tag", default=DEFAULT_TAG, help="comment tag (empty for none)")

    ou = p.add_argument_group("outputs")
    ou.add_argument("--batch", default=DEFAULT_BATCH)
    ou.add_argument("--preview", default=DEFAULT_PREVIEW)
    ou.add_argument("--ledger", default=DEFAULT_LEDGER)

    ac = p.add_argument_group("actions (instead of generating)")
    ac.add_argument("--whoami", action="store_true", help="check the token and exit")
    ac.add_argument("--send", metavar="BATCH", help="post this batch")
    ac.add_argument("--max-age-hours", type=float, default=DEFAULT_MAX_AGE_HOURS)
    ac.add_argument("--force", action="store_true", help="post despite age or cooldown")
    ac.add_argument("--no-notify", action="store_true", help="post without HotCRP's emails (a test)")
    ac.add_argument("--delete-comment", metavar="PID:CID", help="remove one comment (a test's)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    now = datetime.now().astimezone()

    if args.whoami:
        site = hotcrp_api.Site.from_env()
        me = site.whoami()
        print(f"{site.base_url}: token belongs to {me.get('email', '?')}"
              + (" (disabled)" if me.get("disabled") else ""))
        return 0

    if args.delete_comment:
        pid, _, cid = args.delete_comment.partition(":")
        site = hotcrp_api.Site.from_env()
        site.delete_comment(int(pid), int(cid))
        print(f"Deleted comment {cid} on #{pid}", file=sys.stderr)
        return 0

    if args.send:
        site = hotcrp_api.Site.from_env()
        batch = load_batch(args.send)
        _, held = send_batch(site, batch, args.ledger, now=now, max_age_hours=args.max_age_hours,
                             cooldown_days=args.cooldown_days, force=args.force, notify=not args.no_notify)
        return 1 if held else 0

    conditions = [args.leads_without_starter, args.reviewers_silent, args.reviewers_missing_review,
                  bool(args.mention_emails)]
    if not any(conditions):
        raise SystemExit("ERROR: pass a condition: --leads-without-starter, --reviewers-silent, "
                         "--reviewers-missing-review or --mention-emails")
    if (args.leads_without_starter or args.reviewers_silent) and not args.since:
        raise SystemExit("ERROR: --since is required with --leads-without-starter / --reviewers-silent")
    since = parse_since(args.since) if args.since else None
    if args.mention_emails and args.text is None:
        raise SystemExit("ERROR: --mention-emails needs --text")

    exclude = frozenset(parse_pids(args.exclude_pids))
    papers = load_papers(args.data)
    titles = {pid: p.get("title", "") for pid, p in papers.items()}
    site = hotcrp_api.Site.from_env(snapshot_dir=args.snapshot, offline=args.offline)

    if args.papers:
        pids = parse_pids(args.papers)
        scope_note = "named on the command line"
    elif args.scope_search:
        pids = set(site.search(args.scope_search)) - exclude
        scope_note = f"search {args.scope_search!r}"
    else:
        pids = scope_pids(args.data, exclude, all_undecided=args.all_undecided)
        scope_note = "undecided, under review" + ("" if args.all_undecided else f", tagged {review_scores.ADVANCE_TAG}")
    print(f"{len(pids)} papers in scope ({scope_note})", file=sys.stderr)
    if not pids:
        raise SystemExit("ERROR: no papers in scope")

    log_rows = hotcrp_log.load_log(args.log) if os.path.exists(args.log) else []
    if not log_rows:
        print(f"WARNING: {args.log} missing; comment creation dates fall back to last-modified", file=sys.stderr)
    created = hotcrp_log.comment_created_at(log_rows)
    leads, hidden = current_leads(args.leads, log_rows)
    if hidden:
        print(f"{len(hidden)} lead(s) taken from the log (hidden in the download)", file=sys.stderr)

    raw_reviews, raw_comments = fetch_state(site, pids)
    reviews = parse_reviews(raw_reviews)
    comments = [c for c in parse_comments(raw_comments, created) if c.pid in pids]
    seen = {r.pid for r in reviews} | {c.pid for c in comments}
    unseen = sorted(pids - seen)
    print(f"API: {len(reviews)} reviews, {len(comments)} comments on {len(seen)} papers", file=sys.stderr)
    if unseen:
        print(f"UNSEEN: {len(unseen)} paper(s) in scope returned nothing -- the token's owner is "
              f"conflicted, or they have no reviews: {', '.join(map(str, unseen))}", file=sys.stderr)
    names, ambiguous = name_index(reviews, args.pcinfo)

    targets: list[Target] = []
    if args.leads_without_starter:
        found, no_lead = leads_without_starter(pids - set(unseen), leads, comments, since)
        print(f"{KIND_LABELS[KIND_LEAD]}: {len(found)}", file=sys.stderr)
        if no_lead:
            print(f"  {len(no_lead)} paper(s) with no lead, not nudged: {', '.join(map(str, no_lead))}",
                  file=sys.stderr)
        targets += found
    if args.reviewers_silent:
        found = reviewers_silent(pids, reviews, comments, since, require_both=args.require_both)
        print(f"{KIND_LABELS[KIND_SILENT]}: {len(found)} on {len({t.pid for t in found})} papers", file=sys.stderr)
        targets += found
    if args.reviewers_missing_review:
        found = reviewers_missing_review(pids, reviews)
        print(f"{KIND_LABELS[KIND_MISSING]}: {len(found)} on {len({t.pid for t in found})} papers", file=sys.stderr)
        targets += found
    if args.mention_emails:
        targets += [Target(pid, KIND_MANUAL, e) for pid in sorted(pids) for e in sorted(parse_emails(args.mention_emails))]
    targets = dedupe(targets)

    excused: set[str] = set()
    if args.exclude_emails:
        excused |= load_email_file(args.exclude_emails)
    ext_path = args.extensions
    if ext_path and since is not None:
        for email, (status, _) in timeliness_tags.load_extensions(ext_path, since).items():
            if status == timeliness_tags.EXTENSION:
                excused.add(email)
    ledger = load_ledger(args.ledger)
    recent = recent_keys(ledger, now, args.cooldown_days)
    kept = []
    dropped = Counter()
    for t in targets:
        if t.email in excused:
            dropped["excused"] += 1
        elif (t.pid, t.kind, t.email) in recent:
            dropped["cooldown"] += 1
        else:
            kept.append(t)
    for why, n in sorted(dropped.items()):
        print(f"  {n} target(s) dropped: {why}", file=sys.stderr)

    templates = {KIND_LEAD: args.lead_template, KIND_SILENT: args.reviewer_template,
                 KIND_MISSING: args.missing_template}
    nudges, by_address = build_nudges(kept, names, titles, leads, templates, text=args.text, due=args.due,
                                      visibility=args.visibility, topic=args.topic, tag=args.tag,
                                      ambiguous=ambiguous)
    if by_address:
        print(f"{len(by_address)} person(s) mentioned by address rather than name (name unknown, or "
              f"shared by two accounts): {', '.join(by_address)}", file=sys.stderr)

    batch = {
        "version": BATCH_VERSION,
        "generated_at": now.isoformat(timespec="seconds"),
        "site": site.base_url,
        "since": since.isoformat() if since else None,
        "scope": scope_note,
        "nudges": [asdict(n) for n in nudges],
    }
    write_batch(args.batch, batch)
    write_text(args.preview, preview_text(nudges))
    people = len({(n.pid, e) for n in nudges for e in n.mentions})
    print(f"{len(nudges)} comment(s) drafted, mentioning {people} (paper, person) pair(s)", file=sys.stderr)
    print(f"Wrote {args.batch} and {args.preview}; read the preview, then --send {args.batch}", file=sys.stderr)
    print(f"{len(nudges)}\t{people}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
