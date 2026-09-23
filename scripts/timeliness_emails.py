"""Draft one email per paper held by ~~delayedmissingreview, for the chair to send.

    python -m scripts.timeliness_emails
    python -m scripts.timeliness_emails --out /tmp/emails.txt
    python -m scripts.timeliness_emails --announced-deadline 'Monday, September 21, 2026 at 8:00am EST'

A paper whose author on the review committee has not submitted the R1 reviews
they hold has its own reviews held back (`scripts/timeliness_tags.py` tags it
`~~delayedmissingreview`). Its authors have not been told, and the people best
placed to do anything about it are the co-authors sitting on the delayed paper.

This writes one drafted message per held paper into a single text file, to be
split and sent by hand. **Nothing is sent and no mail is configured.** The
same `timeliness_tags.evaluate()` decides which papers are held and who is
holding them, so the emails and the tags can never name different people.

**Which papers.** Exactly the ones `timeliness_tags` reports `blocked` or
`still blocked`. A paper already carrying a day tag, or with no committee
author, is never written.

**Who it goes to.** Every address in the paper's `authors` and `contacts`
lists, deduped, authors first -- the contact list is where the submitting
account lives, which on some papers is the only person who reads HotCRP.

**Who is named.** The committee author(s) with reviews outstanding, by the
name the HotCRP user export shows and the address the paper lists them under,
with the count of reviews they still owe. A blocker with nothing outstanding
is someone the chair marked `late` by hand rather than someone with missing
reviews, and gets wording that says so -- the count sentence would be false.

**Apologies.** `--apology-for SENT_FILE` reads a drafts file that was already
sent and, for every paper in it that is *not* held today, drafts a correction
to the same recipients instead: the hold was an error, the paper is not held
back, with an apology to the author it named. Written to `--apology-out`; the
normal drafts are not written in this mode. Sent papers that are still held are
listed on stdout, marked where a named author's count has changed, and get no
email. The case this exists for is reviews on desk-rejected papers counting as
outstanding; the email never says which other paper was involved.

The output names individuals as delinquent to their collaborators, so it is
the most socially sensitive file this repo writes. It is gitignored twice over,
by location and by name. Read it before you send any of it.
"""

from __future__ import annotations

from reviewer_match.paths import report_path

import argparse
import os
import re
import sys
import textwrap
from collections import Counter

from reviewer_match import pc_membership
from scripts import timeliness_tags

DEFAULT_OUT = report_path("timeliness_emails.txt")
DEFAULT_APOLOGY_OUT = report_path("timeliness_apology_emails.txt")
DEFAULT_SIGNATURE = "-- The HPCA 2027 Program Chairs"
# The deadline the *authors* were given, which is deliberately NOT
# `timeliness_tags.DEFAULT_CUTOFF` (2026-09-22 11:00 -0400). The cutoff this
# pipeline tags against is about 27 hours later, so nobody is tagged for
# missing the announced time by a few hours. An email has to quote what people
# were told, not the internal grace, so these two never derive from each other.
DEFAULT_DEADLINE = "Monday, September 21, 2026 at 8:00am EST"

SEPARATOR = "===== PAPER {pid} ====="
SEPARATOR_RE = re.compile(r"^===== PAPER (\d+) =====$", re.M)
WIDTH = 78

BODY = """\
Dear authors of HPCA 2027 submission #{pid},

    "{title}"

We are writing to let you know that the reviews for this submission will be \
released later than those for other papers, and that its response period will \
be shortened accordingly.

Reviews are held for any submission whose author on the review committee has \
not completed their own assigned reviews. That deadline was {deadline}.

{blockers}

The reviews for your submission will be released as soon as those reviews are \
in; how long the delay runs depends only on when that happens. If you believe \
this message is in error -- if an extension was agreed with the chairs, for \
instance -- please reply and let us know.

{signature}
"""


APOLOGY_BODY = """\
Dear authors of HPCA 2027 submission #{pid},

    "{title}"

Earlier today we wrote to tell you that the reviews for this submission would \
be released late because {named} had assigned reviews outstanding. That was our \
error, and we apologize to all of you, and to {named} in particular. The count \
wrongly included review assignments on submissions that are no longer under \
review, which should never have counted.

Your submission is not being held back. Its reviews will be released on the \
normal schedule, with the full response period.

{signature}
"""


def recipients(paper: dict) -> list[str]:
    """Every author and contact address, deduped, authors first."""
    out: list[str] = []
    for group in ("authors", "contacts"):
        for person in paper.get(group) or []:
            email = (person.get("email") or "").strip().lower()
            if email and email not in out:
                out.append(email)
    return out


def blocker_name(key: str, index: pc_membership.PcIndex | None, display: dict[str, str]) -> str:
    """What to call them: the name HotCRP shows, then the roster's, then the address."""
    account = index.by_email.get(key) if index else None
    if account and account.name:
        return account.name
    return display.get(key) or key


def blocker_clause(
    keys: list[str],
    states: dict[str, timeliness_tags.ReviewerState],
    source: dict[str, str],
    index: pc_membership.PcIndex | None,
    display: dict[str, str],
) -> str:
    """The sentence naming who is holding the paper up, singular or plural."""
    parts = []
    for key in keys:
        state = states[key]
        # A blocker with nothing outstanding was marked `late` by hand: saying
        # they have reviews missing would be false.
        tail = (f"who has {state.outstanding} of {state.held} assigned reviews still outstanding"
                if state.outstanding
                else "whose reviewing obligation for HPCA 2027 is not yet discharged")
        parts.append(f"{blocker_name(key, index, display)} <{source.get(key, key)}>, {tail}")
    if len(parts) == 1:
        return f"For this submission that author is {parts[0]}."
    return f"For this submission those authors are {', and '.join([', '.join(parts[:-1]), parts[-1]])}."


def render(
    paper: dict,
    keys: list[str],
    states: dict[str, timeliness_tags.ReviewerState],
    source: dict[str, str],
    index: pc_membership.PcIndex | None,
    display: dict[str, str],
    *,
    deadline: str,
    signature: str,
) -> str:
    """One complete message: separator, headers, blank line, wrapped body."""
    pid = paper["pid"]
    body = BODY.format(
        pid=pid,
        title=paper.get("title") or "",
        deadline=deadline,
        blockers=blocker_clause(keys, states, source, index, display),
        signature=signature,
    )
    return message(paper, f"HPCA 2027 submission #{pid}: review release delayed", body)


def message(paper: dict, subject: str, body: str) -> str:
    """Separator, headers, blank line, wrapped body."""
    # Wrap the prose, but never the quoted title or a line that is already short:
    # a split on the separator and then the first blank line has to recover the
    # headers and the body exactly as written.
    wrapped = []
    for para in body.split("\n"):
        if para.startswith("    ") or len(para) <= WIDTH:
            wrapped.append(para)
        else:
            wrapped += textwrap.wrap(para, width=WIDTH)
    return "\n".join([
        SEPARATOR.format(pid=paper["pid"]),
        f"To: {', '.join(recipients(paper))}",
        f"Subject: {subject}",
        "",
        *wrapped,
    ])


def join_names(names: list[str]) -> str:
    """"A", "A and B", "A, B, and C"."""
    if len(names) <= 2:
        return " and ".join(names)
    return f"{', '.join(names[:-1])}, and {names[-1]}"


def sent_papers(text: str) -> dict[int, dict[str, tuple[int, int] | None]]:
    """{pid: {address named: (outstanding, held) as stated, or None}} from a sent drafts file.

    Addresses rather than names: a name can carry periods and wrap across lines,
    while `blocker_clause` always writes the address as `<addr>`, and only that
    sentence of the body does. The counts are None for count-free wording.
    """
    out: dict[int, dict[str, tuple[int, int] | None]] = {}
    parts = SEPARATOR_RE.split(text)
    for pid, block in zip(parts[1::2], parts[2::2]):
        clause = block.split("For this submission", 1)[1] if "For this submission" in block else ""
        clause = " ".join(clause.split("\n\n", 1)[0].split())
        named: dict[str, tuple[int, int] | None] = {}
        for m in re.finditer(r"<([^>]+)>(?:, who has (\d+) of (\d+) assigned)?", clause):
            named[m.group(1).strip().lower()] = (int(m.group(2)), int(m.group(3))) if m.group(2) else None
        out[int(pid)] = named
    return out


def render_apology(paper: dict, names: list[str], *, signature: str) -> str:
    """One correction for a paper that was wrongly reported as held."""
    pid = paper["pid"]
    body = APOLOGY_BODY.format(
        pid=pid,
        title=paper.get("title") or "",
        named=join_names(names) or "one of its authors",
        signature=signature,
    )
    return message(paper, f"HPCA 2027 submission #{pid}: correction -- reviews are not delayed", body)


def drafts(ev: timeliness_tags.Evaluation, index, *, deadline: str, signature: str) -> list[str]:
    """One rendered message per held paper, in pid order."""
    out = []
    for row in ev.report:
        if row["status"] not in timeliness_tags.BLOCKED_STATUSES:
            continue
        pid = row["paper"]
        keys = [k for k in ev.author_keys[pid] if ev.states[k].days is None]
        if not keys:  # decide() cannot report blocked without one; belt and braces
            continue
        out.append(render(ev.papers[pid], keys, ev.states, ev.author_source[pid], index,
                          ev.display, deadline=deadline, signature=signature))
    return out


def write_text(path: str, messages: list[str]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n\n".join(messages) + ("\n" if messages else ""))
    os.replace(tmp, path)
    print(f"Wrote {path}", file=sys.stderr)


def apologies(
    ev: timeliness_tags.Evaluation,
    index,
    sent: dict[int, dict[str, tuple[int, int] | None]],
    *,
    signature: str,
) -> tuple[list[str], list[int], list[int]]:
    """(rendered apologies, sent pids still held, sent pids no longer in the export)."""
    held = {r["paper"] for r in ev.report if r["status"] in timeliness_tags.BLOCKED_STATUSES}
    out, still_held, missing = [], [], []
    for pid in sorted(sent):
        if pid not in ev.papers:
            missing.append(pid)
        elif pid in held:
            still_held.append(pid)
        else:
            key_of = {addr: key for key, addr in ev.author_source[pid].items()}
            names = [blocker_name(key_of.get(a, a), index, ev.display) for a in sent[pid]]
            out.append(render_apology(ev.papers[pid], names, signature=signature))
    return out, still_held, missing


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    timeliness_tags.add_common_arguments(parser)
    parser.add_argument("--out", default=DEFAULT_OUT, help="text file of drafted emails to write")
    parser.add_argument("--announced-deadline", default=DEFAULT_DEADLINE,
                        help="the review deadline the committee was given, as the email should "
                             f"word it (default: {DEFAULT_DEADLINE!r}); deliberately separate "
                             "from --cutoff, which is what the tags are decided against")
    parser.add_argument("--signature", default=DEFAULT_SIGNATURE, help="sign-off line")
    parser.add_argument("--apology-for", metavar="SENT_FILE",
                        help="a drafts file already sent: write corrections for its papers that "
                             "are no longer held, instead of the normal drafts")
    parser.add_argument("--apology-out", default=DEFAULT_APOLOGY_OUT,
                        help="text file of drafted corrections to write under --apology-for")
    args = parser.parse_args()

    try:
        ev = timeliness_tags.evaluate_from_args(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    index = None if args.no_pc_check else pc_membership.load_pc_accounts(args.pcinfo)
    if args.apology_for:
        return main_apologies(args, ev, index)
    messages = drafts(ev, index, deadline=args.announced_deadline, signature=args.signature)
    write_text(args.out, messages)

    named: Counter = Counter()
    addresses = 0
    for row in ev.report:
        if row["status"] in timeliness_tags.BLOCKED_STATUSES:
            addresses += len(recipients(ev.papers[row["paper"]]))
            named.update(k for k in ev.author_keys[row["paper"]] if ev.states[k].days is None)
    print(f"Deadline as the email states it: {args.announced_deadline}")
    print(f"{len(messages)} email(s) drafted, {addresses} recipient address(es), "
          f"{len(named)} author(s) named.")
    if named:
        print("\n# papers  author named")
        for key, n in sorted(named.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"{n:>7}   {key}")
    print("\nNothing has been sent. Read the file before sending any of it.", file=sys.stderr)
    return 0


def main_apologies(args, ev: timeliness_tags.Evaluation, index) -> int:
    try:
        with open(args.apology_for, encoding="utf-8") as f:
            sent = sent_papers(f.read())
    except FileNotFoundError:
        print(f"ERROR: {args.apology_for} not found", file=sys.stderr)
        return 1
    messages, still_held, missing = apologies(ev, index, sent, signature=args.signature)
    write_text(args.apology_out, messages)

    addresses = sum(len(recipients(ev.papers[pid])) for pid in sent
                    if pid in ev.papers and pid not in still_held)
    print(f"{len(sent)} paper(s) in {args.apology_for}: {len(messages)} correction(s) drafted "
          f"({addresses} recipient address(es)), {len(still_held)} still held, "
          f"{len(missing)} no longer in the export.")
    if still_held:
        print("\nStill held (no email drafted):")
        print("#  paper  outstanding/held now  (as emailed)  author")
        for pid in still_held:
            row = next(r for r in ev.report if r["paper"] == pid)
            for key in row["blocked_by"].split("; "):
                s = ev.states[key]
                listed = ev.author_source[pid].get(key, key)
                if listed not in sent[pid]:
                    was, mark = "", "  NOT NAMED in the sent email"
                else:
                    stated = sent[pid][listed]
                    was = f"{stated[0]}/{stated[1]}" if stated else "-"
                    mark = "  COUNT CHANGED" if stated and stated != (s.outstanding, s.held) else ""
                print(f"{pid:>8}  {s.outstanding:>8}/{s.held:<11} {was:>12}  {key}{mark}")
    if missing:
        print(f"\nNo longer in the export (skipped): {', '.join(map(str, missing))}")
    print("\nNothing has been sent. Read the file before sending any of it.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
