"""Move the live reviews of every paper that advances into the next review round.

    python -m scripts.advance_review_rounds
    python -m scripts.advance_review_rounds --from-round R1 --to-round R2 \\
        --response Revision --upload outputs/assignments/round_change_upload.csv

A HotCRP review belongs to exactly one round, fixed when it is assigned, and
review-form fields restricted to a round ("Present on: round:R2") appear only
on reviews whose own round that is. Opening R2 and reopening R1 for editing
change nothing about the reviews already filed under R1: a reviewer opens
their R1 review and sees the R1 form. So the reviews themselves have to move.

A paper advances when it has **no decision yet** in the paper export
(`hpca2027-data.json`); a decided paper never moves. Two opt-in gates narrow
that: `--require-response` also asks that the authors **submitted the
response round named `--response`** ("Revision"), whether as a revised paper
attached to it or as a rebuttal comment in its text, and
`--require-attachment` asks for the attachment too. The response state is
replayed from the action log, where HotCRP logs every response save as
`Response N (Revision) submitted: attachments`; a draft response is not a
submitted one, and "attached" means at least one save changed the
attachments (the log names the fields a save changed, never whether a file
was added or removed). Either way stdout lists the undecided papers with no
submitted response and the split between attached and text-only ones, so the
chair can see what the policy let through.

The PC reviews to move are read from HotCRP's own "Review assignments"
download (`hpca2027-pcassignments.csv`), the authoritative record of what
HotCRP holds under each account's current address; the log replay is diffed
against it and any disagreement is reported (an address a reviewer has since
changed is the usual one). That download carries no external reviews, so the
few external reviews assigned in `--from-round` come from the log replay and
are moved too, under the address the log recorded. External TRC reviews are
in their own round and are left alone, with a count.

The upload is `paper,action,email,reviewtype,round` with one row per review:

    1234,review,drew@example.edu,any,R1:R2

which is HotCRP's documented form for changing an existing review's round.
`any` matches only an existing review, so the row can never create one, and
`R1:R2` matches only a review currently in R1, so a re-upload after the move
is a no-op. No `clearreview` row is ever written: review text, scores and
submitted status all stay with the review. Preview the upload on HotCRP's
bulk-assignment page before committing it -- it lists every round change.

Offline, instant, read-only apart from --upload.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections import Counter

from reviewer_match import hotcrp_log
from reviewer_match.paths import assignment_path, input_path

DEFAULT_DATA = input_path("hpca2027-data.json")
DEFAULT_LOG = input_path("hpca2027-log.csv")
DEFAULT_PCASSIGNMENTS = input_path("hpca2027-pcassignments.csv")
DEFAULT_UPLOAD = assignment_path("round_change_upload.csv")
DEFAULT_FROM_ROUND = "R1"
DEFAULT_TO_ROUND = "R2"
DEFAULT_RESPONSE = "Revision"
DEFAULT_EXCLUDE_PIDS = ""

UPLOAD_HEADER = ("paper", "action", "email", "reviewtype", "round")
# The export's PC review actions. `clearreview` is its bookkeeping row and
# `externalreview` the TRC round's; neither moves.
PC_REVIEW_ACTIONS = frozenset({"primaryreview", "secondaryreview", "pcreview"})


def undecided_pids(data_path: str, exclude: frozenset[int]) -> set[int]:
    """Papers in the export with no decision set."""
    with open(data_path, encoding="utf-8") as f:
        papers = json.load(f)
    return {
        int(p["pid"]) for p in papers
        if not p.get("decision") and int(p["pid"]) not in exclude
    }


def load_export_reviews(path: str, round_name: str) -> dict[int, set[str]]:
    """{pid: {email}} of the PC reviews the export holds in `round_name`."""
    reviews: dict[int, set[str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["action"] in PC_REVIEW_ACTIONS and row["round"] == round_name:
                reviews.setdefault(int(row["paper"]), set()).add(row["email"].strip().lower())
    return reviews


def advancing_pids(
    undecided: set[int], responses: dict[int, hotcrp_log.Response], *,
    require_response: bool = False, require_attachment: bool = False,
) -> tuple[set[int], set[int], set[int]]:
    """(advance, held for lacking an attachment, no submitted response) among the undecided.

    By default every undecided paper advances and the other two sets only
    describe it. `require_response` holds back the third set;
    `require_attachment` implies it and holds back the second as well, so a
    submitted rebuttal comment no longer counts as a revision.
    """
    require_response = require_response or require_attachment
    advance, held, none = set(), set(), set()
    for pid in undecided:
        resp = responses.get(pid)
        if resp is None or not resp.submitted:
            none.add(pid)
            if not require_response:
                advance.add(pid)
        elif require_attachment and not resp.attached:
            held.add(pid)
        else:
            advance.add(pid)
    return advance, held, none


def upload_rows(
    advance: set[int], reviews: dict[int, set[str]], from_round: str, to_round: str
) -> list[list[object]]:
    return [
        [pid, "review", email, "any", f"{from_round}:{to_round}"]
        for pid in sorted(advance)
        for email in sorted(reviews.get(pid, ()))
    ]


def write_csv(path: str, header: tuple[str, ...], rows: list[list[object]]) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def parse_pids(value: str) -> frozenset[int]:
    return frozenset(int(p) for p in value.split(",") if p.strip())


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default=DEFAULT_DATA, help="HotCRP paper export JSON (decisions)")
    parser.add_argument("--log", default=DEFAULT_LOG, help="HotCRP action log (responses)")
    parser.add_argument("--pcassignments", default=DEFAULT_PCASSIGNMENTS,
                        help='HotCRP "Review assignments" download (the reviews to move)')
    parser.add_argument("--from-round", default=DEFAULT_FROM_ROUND)
    parser.add_argument("--to-round", default=DEFAULT_TO_ROUND)
    parser.add_argument("--response", default=DEFAULT_RESPONSE,
                        help="response round whose submission carries the revision")
    parser.add_argument("--require-response", action="store_true",
                        help="advance only papers whose authors submitted the --response round")
    parser.add_argument("--require-attachment", action="store_true",
                        help="... and only with an attachment (implies --require-response)")
    parser.add_argument("--exclude-pids", default=DEFAULT_EXCLUDE_PIDS,
                        help="comma-separated paper IDs to leave out")
    parser.add_argument("--upload", default=DEFAULT_UPLOAD, help="HotCRP bulk-assignment file to write")
    args = parser.parse_args()

    for path in (args.data, args.log, args.pcassignments):
        if not os.path.exists(path):
            print(f"ERROR: {path} not found; download it from HotCRP", file=sys.stderr)
            return 1
    if args.from_round == args.to_round:
        print("ERROR: --from-round and --to-round are the same", file=sys.stderr)
        return 1

    undecided = undecided_pids(args.data, parse_pids(args.exclude_pids))
    rows = hotcrp_log.load_log(args.log)
    responses = hotcrp_log.replay_responses(rows, args.response)
    reviews = load_export_reviews(args.pcassignments, args.from_round)
    advance, held, none = advancing_pids(
        undecided, responses,
        require_response=args.require_response, require_attachment=args.require_attachment,
    )
    rebuttal_only = sorted(pid for pid in advance - none if not responses[pid].attached)

    # The export carries the PC reviews; the log is the only source for the
    # external reviews assigned in this round (the TRC externals are in theirs).
    live, _ = hotcrp_log.replay_assignments(rows)
    log_reviews = hotcrp_log.live_pairs(live, kind="primary", round=args.from_round)
    external = hotcrp_log.live_pairs(live, kind="external", round=args.from_round)
    to_move = {pid: reviews.get(pid, set()) | external.get(pid, set()) for pid in advance}
    upload = upload_rows(advance, to_move, args.from_round, args.to_round)
    write_csv(args.upload, UPLOAD_HEADER, upload)
    print(f"Wrote {args.upload}", file=sys.stderr)

    # Cross-checks, reported rather than resolved: the export is authoritative.
    n_external = sum(len(external.get(pid, ())) for pid in advance)
    other_rounds = Counter(
        r.round for r in live.values() if r.pid in advance and r.round != args.from_round
    )
    disagreements = sorted(
        pid for pid in advance if reviews.get(pid, set()) != log_reviews.get(pid, set())
    )
    unstaffed = sorted(pid for pid in advance if not reviews.get(pid))
    re_edited = sorted(pid for pid in advance - none if responses[pid].attachment_saves_after_submit)
    responded_decided = sorted(
        pid for pid, r in responses.items() if r.submitted and pid not in undecided
    )

    print(f"{len(undecided)} papers without a decision; {len(responses)} papers with a live "
          f"'{args.response}' response in the log.")
    print(f"{len(advance)} papers advance {args.from_round} -> {args.to_round}: "
          f"{len(upload)} reviews in the upload, {n_external} of them external reviews "
          f"the log holds in {args.from_round}.")
    print(f"  {len(advance - none) - len(rebuttal_only)} with an attached revision, {len(rebuttal_only)} "
          f"with a rebuttal comment only: {' '.join(map(str, rebuttal_only))}")
    if held:
        print(f"{len(held)} undecided papers submitted a '{args.response}' response with no "
              f"attachment (held by --require-attachment): {' '.join(map(str, sorted(held)))}")
    fate = f"stay in {args.from_round}" if none.isdisjoint(advance) else "advance all the same"
    print(f"{len(none)} undecided papers submitted no '{args.response}' response "
          f"({fate}): {' '.join(map(str, sorted(none)))}")
    if args.require_attachment and re_edited:
        print(f"{len(re_edited)} advancing papers changed their attachments again after submitting "
              f"(the log cannot say whether a file was removed; check if in doubt): "
              f"{' '.join(map(str, re_edited))}")
    if other_rounds:
        left = ", ".join(f"{n} in {rnd}" for rnd, n in sorted(other_rounds.items()))
        print(f"Reviews on advancing papers left in other rounds: {left}.")
    if unstaffed:
        print(f"WARNING: {len(unstaffed)} advancing papers hold no {args.from_round} PC review in "
              f"{args.pcassignments}: {' '.join(map(str, unstaffed))}", file=sys.stderr)
    if disagreements:
        print(f"NOTE: export and log disagree on the {args.from_round} reviewers of "
              f"{len(disagreements)} advancing papers (the export's current addresses are used):",
              file=sys.stderr)
        for pid in disagreements:
            only_export = sorted(reviews.get(pid, set()) - log_reviews.get(pid, set()))
            only_log = sorted(log_reviews.get(pid, set()) - reviews.get(pid, set()))
            print(f"  {pid}: export only {only_export}; log only {only_log}", file=sys.stderr)
    if responded_decided:
        print(f"NOTE: {len(responded_decided)} decided or excluded papers also submitted a response; "
              f"not advanced.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
