"""Area-chair accept reports: each track's `~~accept` papers by category, then its `~~reject` papers.

    # Every track, plus all/ and all_<chair>/ for the PC chairs (needs
    # HOTCRP_SITE and HOTCRP_TOKEN in .env):
    python -m scripts.ac_accept_report

    # Replay the last run's API snapshot, two tracks, into a scratch directory:
    python -m scripts.ac_accept_report --offline --tracks 5,7 --out-dir /tmp/x

One PDF per track, written into the same `track_NN/` directory
`scripts/ac_dashboard.py` uses (`track_05/track_05_accepts_dana-fox.pdf`), so
the Dropbox links `scripts/ac_dashboard_links.py` already sent show it. Any
other PDF in a page's directory -- the status dashboard -- is **moved** to
`<out-dir>/archive/<date>/<page dir>/`, never deleted, which keeps one PDF per
shared directory.

**The categories.** Every paper tagged `~~accept` (`--scope-search`) is
placed in one of three, by its tags and its submitted PC reviews:

1. `online-accept` (set A) or `likely-accept` (set B): accepted outright,
   some with shepherding (`shepherd` tag, shown).
2. `pc-discuss`, resolved to accept by reviewer consensus (no reject review),
   accept reviews outnumbering reject reviews by `--margin` (2) or more, or
   more champions than anti-champions.
3. `pc-discuss`, none of the above, but borderline -- accept reviews equal to
   or one more than reject reviews -- with more potential champions (current
   merit >= `--champion-min`, 4) than potential anti-champions (current merit
   <= `--anti-max`, 1). These are tentative accepts.

An accept review has current merit >= `--accept-min` (3), a reject review
less; current means post-rebuttal where given, else pre-rebuttal. Votes are
the review form's Champion / Anti-champion boxes, a review ticking both
counting on neither side (the dashboard's rule, imported). A tagged paper
that fits none -- no phase tag, or pc-discuss under every test -- or whose
reviews the token's owner cannot see is listed under "Check" and named on
stderr, never silently placed.

**Rejected papers** (`--reject-search`, `#~~reject`) follow the accepts on
every page, one table with the phase tag each was decided under
(online-reject, likely-reject or pc-discuss). A paper tagged both ways goes
to "Check".

**Awaiting reviews.** A paper still without a decision (`--pending-search`,
`dec:none`) that carries neither tag is one extra reviewers were assigned to:
it is listed after the accepts, with its outstanding PC reviews counted and
`--pending-note` saying when they are due.

**Conflicts and privacy** are the dashboard's own: the same readers per page
(`ac_dashboard.page_readers`), the same `conf:` omissions, the conflict
chair's group. No page carries a reviewer's name, an address or review text:
reviews are letters with their scores. The output is gitignored with the
dashboard's directory. stdout is one line, `pages papers cat1 cat2 cat3
check pending rejected`; everything else is stderr.
"""

from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from reviewer_match import hotcrp_api
from scripts import ac_dashboard as ad
from scripts.discussion_nudges import ReviewState, parse_pids, parse_reviews

DEFAULT_SCOPE_SEARCH = "#~~accept"
DEFAULT_REJECT_SEARCH = "#~~reject"
DEFAULT_PENDING_SEARCH = "dec:none"
DEFAULT_PENDING_NOTE = ("Additional reviewers were assigned to these papers. Their reviews are expected by "
                        "Friday, October 30, and the paper will be resolved once they arrive.")
DEFAULT_ACCEPT_MIN = 3          # merit 3-5 is an accept review, 1-2 a reject review
DEFAULT_CHAMPION_MIN = 4        # a potential champion: current merit 4 or 5
DEFAULT_ANTI_MAX = 1            # a potential anti-champion: current merit 1
DEFAULT_MARGIN = 2
DEFAULT_ORIENTATION = "portrait"
SHEPHERD_TAG = "shepherd"
PAGE_SUFFIX = "accepts"
PAPER_FIELDS = "title tags"

CAT1, CAT2, CAT3, PENDING, CHECK, REJECTED = "1", "2", "3", "pending", "check", "rejected"
CATEGORIES = [CAT1, CAT2, CAT3, PENDING, CHECK, REJECTED]
# Rejected papers are ordered by how they were decided.
REJECT_ORDER = ["online-reject", "likely-reject", "pc-discuss", ""]

TEXT = {
    CAT1: ("Online accept and likely accept (sets A and B)",
           "Online accept papers and likely-accept papers that remained in these sets will be accepted "
           "(some with shepherding) — these are sets A and B."),
    CAT2: ("PC discuss, resolved to accept",
           "PC discuss papers resolve to accept by one of 3 ways: (1) Reviewer consensus; (2) +2 accept to "
           "reject reviews; or (3) One more champion to anti-champion votes. A total of {n} papers were "
           "marked as accepted this way."),
    CAT3: ("PC discuss, tentatively accepted (borderline)",
           "The remaining papers were to be rejected by the original process. However, we identified papers "
           "that are borderline (+1 or equal accept to reject reviews) and that had more implied/potential "
           "champion reviews (4 or 5 post-rebuttal merit score) than implied/potential anti-champion (score "
           "of 1 post-rebuttal). For these papers, based on the discussion, if the potential champions do not "
           "explicitly say that they will not champion the paper we tentatively marked these papers as "
           "accept, and posted a message to the reviewers to give them a chance to disagree with the "
           "decision. These decisions may change if the reviewers prefer a different outcome."),
    CHECK: ("Check", "Tagged accept, but the tags and reviews fit none of the categories above, or the "
                     "reviews are hidden from the chairs' account."),
    PENDING: ("Awaiting additional reviews", DEFAULT_PENDING_NOTE),
    REJECTED: ("Rejected papers", "Papers marked reject, by the phase they were decided in."),
}


@dataclass(frozen=True)
class Rules:
    accept_min: int = DEFAULT_ACCEPT_MIN
    champion_min: int = DEFAULT_CHAMPION_MIN
    anti_max: int = DEFAULT_ANTI_MAX
    margin: int = DEFAULT_MARGIN


@dataclass
class Paper:
    pid: int
    title: str
    tracks: list[int]
    tags: set[str]
    reviews: list[ReviewState] = field(default_factory=list)   # submitted PC reviews, by letter
    category: str = ""
    accept_set: str = ""         # A | B, category 1 only
    rejected: bool = False       # tagged ~~reject
    accepted: bool = True        # tagged ~~accept
    outstanding: int = 0         # PC reviews assigned, not yet submitted
    reasons: list[str] = field(default_factory=list)

    @property
    def hidden(self) -> bool:
        return not self.reviews

    def current(self) -> list[int]:
        return [s for r in self.reviews for s in [r.post if r.post is not None else r.pre] if s is not None]


@dataclass(frozen=True)
class Tally:
    accepts: int
    rejects: int
    champions: int
    antis: int
    potential_champions: int
    potential_antis: int


def tally(paper: Paper, rules: Rules) -> Tally:
    cur = paper.current()
    return Tally(
        accepts=sum(1 for s in cur if s >= rules.accept_min),
        rejects=sum(1 for s in cur if s < rules.accept_min),
        champions=sum(1 for r in paper.reviews if r.champion and not r.anti),
        antis=sum(1 for r in paper.reviews if r.anti and not r.champion),
        potential_champions=sum(1 for s in cur if s >= rules.champion_min),
        potential_antis=sum(1 for s in cur if s <= rules.anti_max),
    )


def classify(paper: Paper, rules: Rules = Rules(), phase_tags: dict[str, str] = ad.DEFAULT_PHASE_TAGS) -> None:
    """Set the paper's category, accept set and reasons in place."""
    online = [t for t, ph in phase_tags.items() if ph == ad.SETTLED_ACCEPT and t.startswith("online")]
    likely = [t for t, ph in phase_tags.items() if ph == ad.SETTLED_ACCEPT and t not in online]
    discuss = [t for t, ph in phase_tags.items() if ph == ad.DISCUSS]
    has = lambda ts: any(t.lower() in paper.tags for t in ts)   # noqa: E731
    paper.reasons = []
    if not paper.accepted and not paper.rejected:
        paper.category = PENDING
        paper.reasons = [f"{paper.outstanding} review{'s' * (paper.outstanding != 1)} outstanding"]
        return
    if paper.rejected:
        if paper.accepted:
            paper.category, paper.reasons = CHECK, ["tagged both ~~accept and ~~reject"]
            return
        held = [t for t in REJECT_ORDER if t and t in paper.tags]
        paper.category, paper.reasons = REJECTED, held or ["no phase tag"]
        return
    if has(online) or has(likely):
        paper.category, paper.accept_set = CAT1, "A" if has(online) else "B"
        if has(discuss):
            paper.reasons.append("also tagged pc-discuss")
        return
    if not has(discuss):
        paper.category, paper.reasons = CHECK, ["no online/likely-accept or pc-discuss tag"]
        return
    if paper.hidden:
        paper.category, paper.reasons = CHECK, ["reviews hidden from the chairs' account"]
        return
    t = tally(paper, rules)
    if t.rejects == 0:
        paper.reasons.append("consensus")
    if t.accepts - t.rejects >= rules.margin:
        paper.reasons.append(f"+{t.accepts - t.rejects} accept to reject")
    if t.champions > t.antis:
        paper.reasons.append(f"{t.champions} champion{'s' * (t.champions != 1)} to {t.antis} anti")
    if paper.reasons:
        paper.category = CAT2
    elif t.accepts - t.rejects in (0, 1) and t.potential_champions > t.potential_antis:
        paper.category = CAT3
        paper.reasons = [f"{'+1' if t.accepts > t.rejects else 'equal'} accept to reject; "
                         f"{t.potential_champions} potential champion{'s' * (t.potential_champions != 1)} "
                         f"to {t.potential_antis} potential anti"]
    else:
        paper.category = CHECK
        paper.reasons = [f"pc-discuss, fits no category ({t.accepts} accept / {t.rejects} reject, "
                         f"{t.champions} champion / {t.antis} anti)"]


def build_papers(pids: set[int], facts: dict[int, dict], reviews: list[ReviewState], rules: Rules,
                 phase_tags: dict[str, str] = ad.DEFAULT_PHASE_TAGS, rejected: set[int] = frozenset(),
                 accepted: set[int] | None = None) -> list[Paper]:
    """`pids` is every paper; `rejected` the ~~reject ones, `accepted` the
    ~~accept ones (default: every paper not rejected)."""
    by_pid: dict[int, list[ReviewState]] = defaultdict(list)
    outstanding: Counter = Counter()
    for r in reviews:
        if r.pid in pids and r.pc and r.submitted:
            by_pid[r.pid].append(r)
        elif r.pid in pids and r.pc:
            outstanding[r.pid] += 1
    papers = []
    for pid in sorted(pids):
        fact = facts.get(pid, {})
        p = Paper(pid=pid, title=fact.get("title", ""), tracks=ad.track_numbers(fact.get("tags", "")),
                  tags=ad.tag_set(fact.get("tags", "")),
                  reviews=sorted(by_pid.get(pid, []), key=lambda r: (r.ordinal or "~", r.rid)),
                  rejected=pid in rejected,
                  accepted=pid in accepted if accepted is not None else pid not in rejected,
                  outstanding=outstanding[pid])
        classify(p, rules, phase_tags)
        papers.append(p)
    return papers


# ---------------------------------------------------------------------------
# The page: static HTML, printed by headless Chrome
# ---------------------------------------------------------------------------

CSS = """
:root { --ink:#1c1b18; --ink-2:#5a574f; --line:#e3e1da; --chip:#ecebe6; --accent:#2f5d8a;
        --good:#2d7a4f; --bad:#a83a30; --warn-bg:#fbf1d6; }
body { margin:0; background:#fff; color:var(--ink); font:10.5pt/1.4 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
h1 { font-size:16pt; margin:0 0 2px; }
.meta { color:var(--ink-2); margin:0 0 14px; }
h2 { font-size:12pt; margin:18px 0 4px; break-after:avoid; }
h2 .n { color:var(--ink-2); font-weight:400; }
p.def { margin:0 0 8px; color:var(--ink-2); }
table { width:100%; border-collapse:collapse; margin-bottom:6px; }
th, td { text-align:left; vertical-align:top; padding:4px 6px; border-bottom:1px solid var(--line); }
th { font-size:9pt; color:var(--ink-2); font-weight:600; }
tr { break-inside:avoid; }
td.pid { white-space:nowrap; font-variant-numeric:tabular-nums; }
td.pid a { color:var(--accent); text-decoration:none; }
.scores { white-space:nowrap; font-variant-numeric:tabular-nums; }
.chip { display:inline-block; background:var(--chip); border-radius:3px; padding:0 4px; margin:0 2px 2px 0; }
.up { color:var(--good); } .down { color:var(--bad); }
.tag { display:inline-block; font-size:8.5pt; border:1px solid var(--line); border-radius:3px; padding:0 4px; }
.check td { background:var(--warn-bg); }
.empty { color:var(--ink-2); font-style:italic; margin:0 0 8px; }
.foot { margin-top:18px; font-size:8.5pt; color:var(--ink-2); }
"""


def scores_html(p: Paper) -> str:
    chips = []
    for r in p.reviews:
        s = f"{r.pre}" if r.post is None or r.post == r.pre else f"{r.pre}→{r.post}"
        vote = (' <span class="up">▲</span>' if r.champion and not r.anti
                else ' <span class="down">▼</span>' if r.anti and not r.champion else "")
        chips.append(f'<span class="chip">{html.escape(r.ordinal or "?")} {s}{vote}</span>')
    return "".join(chips) or "—"


def avg_html(p: Paper) -> str:
    cur = p.current()
    return f"{sum(cur) / len(cur):.2f}" if cur else "—"


def table_html(papers: list[Paper], site_url: str, cat: str) -> str:
    if not papers:
        return '<p class="empty">None in this report.</p>'
    third = {CAT1: "Set", CAT2: "Resolved by", CAT3: "Why borderline", CHECK: "Why", REJECTED: "Decided as",
             PENDING: "Status"}[cat]
    if cat == REJECTED:
        rank = {t: i for i, t in enumerate(REJECT_ORDER)}
        papers = sorted(papers, key=lambda p: (min((rank.get(r, len(rank)) for r in p.reasons), default=len(rank)), p.pid))
    rows = []
    for p in papers:
        if cat == CAT1:
            info = p.accept_set + (' <span class="tag">shepherd</span>' if SHEPHERD_TAG in p.tags else "")
            info += "".join(f" <br>{html.escape(r)}" for r in p.reasons)
        else:
            info = "; ".join(html.escape(r) for r in p.reasons)
        track = ", ".join(str(t) for t in p.tracks) or "—"
        rows.append(
            f'<tr class="{"check" if cat == CHECK else ""}"><td class="pid"><a href="{html.escape(site_url)}/paper/{p.pid}">'
            f'#{p.pid}</a></td><td>{html.escape(p.title)}</td><td>{track}</td>'
            f'<td class="scores">{scores_html(p)}</td><td>{avg_html(p)}</td><td>{info}</td></tr>')
    return ('<table><thead><tr><th>Paper</th><th>Title</th><th>Track</th><th>Reviews (merit, ▲ champion ▼ anti)</th>'
            f'<th>Avg</th><th>{third}</th></tr></thead><tbody>{"".join(rows)}</tbody></table>')


def render_page(title: str, subtitle: str, papers: list[Paper], totals: Counter, site_url: str,
                notes: list[str], pending_note: str = DEFAULT_PENDING_NOTE) -> str:
    parts = [f"<title>{html.escape(title)}</title><style>{CSS}</style><main>",
             f"<h1>{html.escape(title)}</h1><p class=\"meta\">{html.escape(subtitle)}</p>"]
    for i, cat in enumerate(CATEGORIES):
        mine = [p for p in papers if p.category == cat]
        if cat in (CHECK, PENDING) and not mine:
            continue
        heading, text = TEXT[cat]
        if cat == PENDING:
            text = pending_note
        label = f"{i + 1}. {heading}" if cat in (CAT1, CAT2, CAT3) else heading
        parts.append(f'<h2>{html.escape(label)} <span class="n">({len(mine)} here)</span></h2>'
                     f'<p class="def">{html.escape(text.format(n=totals[CAT2]))}</p>')
        parts.append(table_html(mine, site_url, cat))
    parts.append('<div class="foot">' + "<br>".join(html.escape(n) for n in notes) + "</div></main>")
    return "".join(parts)


def write_pdfs(out_dir: str, pages: dict[str, str], names: dict[str, str], chrome: str, *,
               orientation: str, archive_dir: str) -> tuple[list[str], list[str]]:
    """Print each page to `<page_dir>/<names[page]>`, first moving any other
    PDF in that directory to `archive_dir/<page_dir>/`. (written, archived)."""
    binary = shutil.which(chrome)
    if not binary:
        raise SystemExit(f"ERROR: {chrome!r} not found; pass --chrome or install Google Chrome / Chromium")
    os.makedirs(out_dir, exist_ok=True)
    written, archived = [], []
    for page, body in pages.items():
        sub_name = ad.page_dir(page)
        sub = os.path.join(out_dir, sub_name)
        os.makedirs(sub, exist_ok=True)
        # Temp files in out_dir, never in the (shared) page directory.
        fd, src = tempfile.mkstemp(dir=out_dir, suffix=".print.html")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(ad.print_document(body, orientation))
        tmp_pdf = src[:-len(".print.html")] + ".pdf.tmp"
        try:
            result = subprocess.run([binary, *ad.CHROME_FLAGS, f"--print-to-pdf={tmp_pdf}", "file://" + os.path.abspath(src)],
                                    capture_output=True, text=True, timeout=120)
            if result.returncode != 0 or not os.path.exists(tmp_pdf):
                raise SystemExit(f"ERROR: Chrome failed to print {page}: {result.stderr.strip()[-400:]}")
            for name in sorted(os.listdir(sub)):
                if name.endswith(".pdf") and name != names[page]:
                    dest = os.path.join(archive_dir, sub_name)
                    os.makedirs(dest, exist_ok=True)
                    shutil.move(os.path.join(sub, name), os.path.join(dest, name))
                    archived.append(os.path.join(sub_name, name))
            os.replace(tmp_pdf, os.path.join(sub, names[page]))
            written.append(os.path.join(sub_name, names[page]))
        finally:
            for path in (src, tmp_pdf):
                if os.path.exists(path):
                    os.remove(path)
    return written, archived


def pdf_name(page: str, chairs: dict[int, list[ad.TrackChair]]) -> str:
    """The dashboard's name with `_accepts` after the page: track_05_accepts_dana-fox.pdf."""
    base = re.sub(r"\.html$", "", page)
    name = ad.pdf_name(page, chairs)
    return f"{base}_{PAGE_SUFFIX}{name[len(base):]}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scope-search", default=DEFAULT_SCOPE_SEARCH,
                   help=f"HotCRP search selecting the accepted papers (default {DEFAULT_SCOPE_SEARCH!r})")
    p.add_argument("--reject-search", default=DEFAULT_REJECT_SEARCH,
                   help=f"HotCRP search selecting the rejected papers (default {DEFAULT_REJECT_SEARCH!r})")
    p.add_argument("--pending-search", default=DEFAULT_PENDING_SEARCH,
                   help="HotCRP search for undecided papers; those with neither tag are awaiting reviews "
                        f"(default {DEFAULT_PENDING_SEARCH!r})")
    p.add_argument("--pending-note", default=DEFAULT_PENDING_NOTE, help="text above the awaiting-reviews table")
    p.add_argument("--exclude-pids", default="", help="comma-separated paper IDs to leave out")
    p.add_argument("--tracks", default="", help="comma-separated track numbers to write (default: all)")
    p.add_argument("--pcinfo", default=ad.DEFAULT_PCINFO, help="HotCRP user export (chairs)")
    p.add_argument("--out-dir", default=ad.DEFAULT_OUT_DIR)
    p.add_argument("--archive-dir", default="",
                   help="where displaced PDFs go (default <out-dir>/archive/<today>)")
    p.add_argument("--snapshot", default=ad.DEFAULT_SNAPSHOT, help="directory of API snapshots")
    p.add_argument("--offline", action="store_true", help="answer every API call from --snapshot")
    p.add_argument("--accept-min", type=int, default=DEFAULT_ACCEPT_MIN,
                   help=f"lowest merit counted as an accept review (default {DEFAULT_ACCEPT_MIN})")
    p.add_argument("--champion-min", type=int, default=DEFAULT_CHAMPION_MIN,
                   help=f"lowest merit counted as a potential champion (default {DEFAULT_CHAMPION_MIN})")
    p.add_argument("--anti-max", type=int, default=DEFAULT_ANTI_MAX,
                   help=f"highest merit counted as a potential anti-champion (default {DEFAULT_ANTI_MAX})")
    p.add_argument("--margin", type=int, default=DEFAULT_MARGIN,
                   help=f"accept-minus-reject margin that resolves to accept (default {DEFAULT_MARGIN})")
    p.add_argument("--pc-chairs", default="", help="as in scripts/ac_dashboard.py")
    p.add_argument("--conflict-chair", default="", help="as in scripts/ac_dashboard.py")
    p.add_argument("--site-url", default="", help="HotCRP base URL for paper links (default: the site)")
    p.add_argument("--generated-at", default="", help="timestamp to print (YYYY-MM-DD HH:MM; default: now)")
    p.add_argument("--chrome", default=ad.DEFAULT_CHROME)
    p.add_argument("--orientation", choices=("landscape", "portrait"), default=DEFAULT_ORIENTATION)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not shutil.which(args.chrome):
        raise SystemExit(f"ERROR: {args.chrome!r} not found; pass --chrome or install Google Chrome / Chromium")
    now = datetime.now().astimezone()
    generated_at = args.generated_at or now.strftime(ad.DATE_FORMAT)
    archive_dir = args.archive_dir or os.path.join(args.out_dir, "archive", now.strftime("%Y-%m-%d"))
    rules = Rules(args.accept_min, args.champion_min, args.anti_max, args.margin)
    exclude = frozenset(parse_pids(args.exclude_pids))
    only_tracks = {int(t) for t in re.split(r"[,\s]+", args.tracks) if t.strip()}

    site = hotcrp_api.Site.from_env(snapshot_dir=args.snapshot, offline=args.offline)
    site_url = (args.site_url or site.base_url).rstrip("/")
    chairs = ad.track_chairs(args.pcinfo)
    accepted = set(site.search(args.scope_search)) - exclude
    rejected = set(site.search(args.reject_search)) - exclude
    pending = set(site.search(args.pending_search)) - exclude - accepted - rejected
    pids = accepted | rejected | pending
    if pending:
        print(f"{len(pending)} undecided with neither tag, awaiting reviews: "
              + ", ".join(map(str, sorted(pending))), file=sys.stderr)
    print(f"{len(accepted)} accepted (search {args.scope_search!r}), {len(rejected)} rejected "
          f"(search {args.reject_search!r})", file=sys.stderr)
    if not accepted:
        raise SystemExit("ERROR: no accepted papers in scope")
    facts: dict[int, dict] = {}
    for q in hotcrp_api.pid_query(pids):
        for row in site.papers(q, PAPER_FIELDS):
            if str(row.get("pid", "")).isdigit():
                facts[int(row["pid"])] = {"title": str(row.get("title") or ""), "tags": str(row.get("tags") or "")}
    reviews: list = []
    for q in hotcrp_api.pid_query(pids):
        reviews += site.reviews(q)
    papers = build_papers(pids, facts, parse_reviews(reviews), rules, rejected=rejected, accepted=accepted)
    totals = Counter(p.category for p in papers)
    print("categories: " + ", ".join(f"{totals[c]} {c}" for c in CATEGORIES), file=sys.stderr)
    for p in papers:
        if p.category == CHECK:
            print(f"CHECK #{p.pid}: {'; '.join(p.reasons)}", file=sys.stderr)

    rd = ad.page_readers(site, args.pcinfo, chairs, pids, args.pc_chairs, args.conflict_chair)
    chairs_as_of = datetime.fromtimestamp(os.path.getmtime(args.pcinfo)).strftime(ad.DATE_FORMAT)
    notes = [f"Generated {generated_at} from HotCRP (papers tagged ~~accept or ~~reject). Chairs as of the user export of "
             f"{chairs_as_of}.",
             f"An accept review has current merit {rules.accept_min}-5, a reject review 1-{rules.accept_min - 1}; "
             f"current is post-rebuttal where given, else pre-rebuttal (shown pre→post). Potential champion: "
             f"merit ≥{rules.champion_min}; potential anti-champion: merit ≤{rules.anti_max}. ▲/▼ are the "
             "Champion / Anti-champion boxes; a review ticking both counts on neither side. Submitted PC "
             "reviews only."]

    def page(key, title: str, conflicted: set[int], extra: set[int] = frozenset(), reader: str = "") -> str:
        mine = [p for p in papers
                if (key is None or key in p.tracks or p.pid in extra) and p.pid not in conflicted]
        omitted = sum(1 for p in papers if (key is None or key in p.tracks or p.pid in extra) and p.pid in conflicted)
        who = (", ".join(c.name for c in chairs.get(key, [])) if isinstance(key, int) and key >= 0
               else reader or "PC chairs")
        n_rej = sum(1 for p in mine if p.category == REJECTED)
        n_acc = sum(1 for p in mine if p.category in (CAT1, CAT2, CAT3, CHECK))
        sub = f"{who} · {n_acc} accepted paper(s) · {n_rej} rejected"
        if extra:
            sub += f", including {len(extra - conflicted)} conflicted with every PC chair"
        if omitted:
            sub += f" · {omitted} omitted (conflict)"
        n_pend = sum(1 for p in mine if p.category == PENDING)
        if n_pend:
            sub += f" · {n_pend} awaiting reviews"
        return render_page(title, sub, mine, totals, site_url, notes, args.pending_note)

    tracks = sorted({t for p in papers for t in p.tracks} | set(chairs))
    pages = {}
    for n in tracks:
        if only_tracks and n not in only_tracks:
            continue
        pages[ad.PAGE_NAME.format(n)] = page(n, f"HPCA 2027 Track {n}: Accepted papers", rd.conflicted(n), rd.extra(n))
    if not only_tracks:
        pages[ad.ALL_PAGE] = page(None, "HPCA 2027 All Tracks: Accepted papers", rd.conflicted(None))
        for name in sorted(rd.each_chair):
            pages[ad.CHAIR_PAGE.format(ad.slug(name))] = page(
                None, f"HPCA 2027 All Tracks: Accepted papers ({name})", rd.own(name), reader=name)
        if -1 in rd.readers:
            pages[ad.CONFLICT_PAGE] = page(-1, "HPCA 2027 Conflict Chair: Accepted papers",
                                           rd.conflicted(-1), rd.extra(-1))
    written, archived = write_pdfs(args.out_dir, pages, {pg: pdf_name(pg, chairs) for pg in pages}, args.chrome,
                                   orientation=args.orientation, archive_dir=archive_dir)
    for n in tracks:
        if not only_tracks or n in only_tracks:
            c = Counter(p.category for p in papers if n in p.tracks)
            print(f"  track {n:2d}: {sum(c.values()) - c[REJECTED] - c[PENDING]:3d} accepted  ({c[CAT1]}/{c[CAT2]}/{c[CAT3]}"
                  + (f", {c[CHECK]} check" if c[CHECK] else "") + f"), {c[REJECTED]:3d} rejected"
                  + (f", {c[PENDING]} awaiting" if c[PENDING] else "") + "  "
                  + ", ".join(ch.name for ch in chairs.get(n, [])), file=sys.stderr)
    if archived:
        print(f"Moved {len(archived)} earlier PDF(s) to {archive_dir}", file=sys.stderr)
    print(f"Wrote {len(written)} PDF(s) to {args.out_dir}", file=sys.stderr)
    print(f"{len(pages)}\t{len(papers)}\t{totals[CAT1]}\t{totals[CAT2]}\t{totals[CAT3]}\t{totals[CHECK]}\t{totals[PENDING]}\t{totals[REJECTED]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
