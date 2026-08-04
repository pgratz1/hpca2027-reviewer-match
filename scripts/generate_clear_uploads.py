"""Generate the HotCRP bulk uploads that wipe reviewer assignments and track tags.

    python -m scripts.generate_clear_uploads
    python -m scripts.generate_clear_uploads --round all

Writes three uploads, because HotCRP takes them on two different pages in two
different formats:

    outputs/assignments/clear_assignment.csv    Assignments -> Bulk update
    outputs/assignments/clear_paper_tags.csv    Assignments -> Bulk update
    outputs/assignments/clear_account_tags.csv  Settings -> Accounts

Each is the undo half of what `assign_reviewers.py --hotcrp-csv` and
`assign_area_chairs.py --account-tag-csv/--paper-tag-csv` install: the same
`clearreview`, `cleartag` and `remove_tags` operations those files open with,
with nothing reinstalled afterwards. Nothing else is touched -- not
`~~area-chairs`, not any other tag, not conflicts, not review preferences.

The tag spellings and the clear ceiling are imported from assign_area_chairs,
never restated here: a clearing file that spelled a tag differently from the
file that installed it would report success and leave the tag in place.
"""

from __future__ import annotations

from reviewer_match.paths import assignment_path, curated_path, input_path

import argparse
import csv
import json
import os
import re
import sys
import tempfile
from pathlib import Path

from reviewer_match import pc_membership
from reviewer_match.area_chairs import area_chair_emails
from scripts.assign_area_chairs import (
    DEFAULT_TRACK_CLEAR_CEILING,
    account_track_tag,
    paper_track_tag,
)

DEFAULT_CSV = input_path("Area Chair Acceptance Form (Responses) - Form Responses 1.csv")
DEFAULT_OVERRIDES = curated_path("dblp_overrides.csv")
DEFAULT_DATA = input_path("hpca2027-data.json")
DEFAULT_ASSIGNMENT_OUT = assignment_path("clear_assignment.csv")
DEFAULT_PAPER_TAG_OUT = assignment_path("clear_paper_tags.csv")
DEFAULT_ACCOUNT_TAG_OUT = assignment_path("clear_account_tags.csv")

# The round assign_reviewers.py installs into, and so the one to clear.
DEFAULT_ROUND = "R1"

# A tag naming one of the per-chair tracks, in either namespace and in any of
# the spellings a hand-made tag takes: `track_7` is what this repo writes, but
# `track1` and `~~track1` are both live on the current export, left behind by
# setting the mechanism up by hand. A wipe that cleared only the canonical
# range would leave those exact tags behind, which is the whole failure it
# exists to prevent. `TRC-track` deliberately does not match -- it is a
# separate track, not a chair's, and is nobody's to clear here.
TRACK_TAG_RE = re.compile(r"^track[_-]?\d+$", re.IGNORECASE)


def is_track_tag(tag: str) -> bool:
    """Is `tag` one of the per-chair track tags, in either namespace?

    Takes the tag as HotCRP writes it -- `~~` prefix for the account/private
    namespace, `#value` suffix for a tag's order value -- and tests the name
    in between.
    """
    return bool(TRACK_TAG_RE.match(tag.split("#", 1)[0].lstrip("~")))


def atomic_rows(path: str, header: list[str], rows: list[list[str]]) -> None:
    """Write one CSV atomically, so a failed run leaves the old file intact."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temporary = Path(f.name)
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_clear_assignment_csv(path: str, review_round: str) -> None:
    """Write the Assignments bulk update that removes every review assignment.

    One row: the same `all,clearreview,all,R1` line the assignment upload opens
    with, which is what makes that upload a replacement rather than an
    addition. An empty `round` cell widens it from R1 to every round; the
    default stays R1, since R1 is the only round this pipeline ever assigns
    into and clearing a round nothing here created is not this repo's call.
    """
    atomic_rows(path, ["paper", "action", "email", "round"],
                [["all", "clearreview", "all", review_round]])


def stray_paper_tags(data_path: str, clear_through: int) -> list[str]:
    """Track tags on papers in `data_path` that the canonical range misses.

    Read from the paper export rather than assumed, because the ones that
    matter are precisely the ones no code here wrote.
    """
    with open(data_path, encoding="utf-8") as f:
        papers = json.load(f)
    canonical = {paper_track_tag(n).casefold() for n in range(clear_through)}
    strays: dict[str, None] = {}
    for paper in papers:
        for tag in paper.get("tags") or []:
            name = str(tag).split("#", 1)[0]
            if is_track_tag(name) and name.casefold() not in canonical:
                strays.setdefault(name, None)
    return sorted(strays)


def write_clear_paper_tag_csv(path: str, strays: list[str], clear_through: int) -> None:
    """Write the Assignments bulk update that untags every paper from its track.

    `cleartag` takes the tag off every paper that carries it, so the canonical
    range is one row per track number regardless of how many papers hold it.
    """
    rows = [["all", "cleartag", "", paper_track_tag(n), ""] for n in range(clear_through)]
    rows += [["all", "cleartag", "", tag, ""] for tag in strays]
    atomic_rows(path, ["paper", "action", "email", "tag", "round"], rows)


def track_tag_holders(index) -> dict[str, list[str]]:
    """{account email: the track tags it holds}, for every account in the export.

    The authoritative answer to who has to appear in the account upload: a
    chair who left the roster between runs is invisible to the chair roster but
    still carries the tag, and is exactly the case `assign_area_chairs.py`'s
    own clear-then-reinstall cannot reach.
    """
    holders: dict[str, list[str]] = {}
    for acct in index.accounts:
        held = [tag for tag in acct.tags.split() if is_track_tag(tag)]
        if held:
            holders[acct.email.lower()] = sorted(held)
    return holders


def account_rows(
    holders: dict[str, list[str]],
    chair_emails: set[str],
    index,
    clear_through: int,
) -> list[list[str]]:
    """One `email,remove_tags` row per account that could hold a track tag.

    Two sources, because neither alone is complete. The export says who holds
    one *today*; the chair roster says who was tagged by the last upload, which
    an export taken before that upload cannot know. Removing a tag an account
    does not have is a no-op, so over-including is free.

    What is not free is naming an address HotCRP has never seen: the bulk user
    importer *creates* an account for an unknown email, so a clearing file that
    listed a chair's acceptance-form address rather than their account address
    would quietly add users. Every row is therefore an address the export
    itself lists.
    """
    canonical = [account_track_tag(n) for n in range(clear_through)]
    known = set(canonical)
    rows = []
    for email in sorted(set(holders) | {e for e in chair_emails if e in index.by_email}):
        extra = [tag for tag in holders.get(email, []) if tag not in known]
        rows.append([index.by_email[email].email, " ".join(canonical + extra)])
    return rows


def write_clear_account_tag_csv(path: str, rows: list[list[str]]) -> None:
    """Write the Settings -> Accounts bulk update that strips the ~~track_N grants.

    Header is `email,remove_tags,add_tags` with every `add_tags` cell empty --
    the same three columns `assign_area_chairs.py` writes, and the empty cell
    is what makes this the clearing half. The plain `tags` column stays off
    limits for the reason that script gives: HotCRP's user importer reads it as
    a full replacement of the account's tag set, wiping `~~area-chairs` and
    `pc` along with it.

    The empty column has to be *present*, which is not obvious and is the whole
    reason this note exists. `p_profile.php`'s `save_bulk` decides whether an
    upload is a CSV or a plain list of email addresses before it ever looks at
    the header, and one of the two tests is whether the first line holds at
    least two commas. A two-column `email,remove_tags` header holds one, and on
    a HotCRP without the newer `(?:user|email)` test beside it every line is
    re-quoted into a single field: the header stops being a header and each row
    is validated whole as an email address, which fails with "Invalid email
    address" on line 2 and every line after it. Three columns never reach that
    branch.

    The empty cell is a no-op, not a guess: `UserStatus::parse_csv_main` skips
    any column whose trimmed value is `""`, so `add_tags` is never set on the
    update object and `normalize`'s `isset($cj->add_tags)` branch never runs.
    """
    atomic_rows(path, ["email", "remove_tags", "add_tags"], [row + [""] for row in rows])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--csv", default=DEFAULT_CSV, help="area-chair acceptance CSV")
    parser.add_argument("--overrides", default=DEFAULT_OVERRIDES, help="DBLP identity overrides")
    parser.add_argument("--data", default=DEFAULT_DATA, help="HotCRP paper export JSON")
    parser.add_argument(
        "--pcinfo", default=pc_membership.DEFAULT_PCINFO,
        help="HotCRP user export naming the accounts to clear (default: %(default)s)",
    )
    parser.add_argument(
        "--round", default=DEFAULT_ROUND, dest="review_round",
        help="review round to clear; 'all' clears every round (default: %(default)s)",
    )
    parser.add_argument(
        "--track-clear-ceiling", type=int, default=DEFAULT_TRACK_CLEAR_CEILING,
        help="clear track_0..track_(N-1) and ~~track_0..~~track_(N-1) (default: %(default)s)",
    )
    parser.add_argument("--assignment-out", default=DEFAULT_ASSIGNMENT_OUT)
    parser.add_argument("--paper-tag-out", default=DEFAULT_PAPER_TAG_OUT)
    parser.add_argument("--account-tag-out", default=DEFAULT_ACCOUNT_TAG_OUT)
    args = parser.parse_args()

    review_round = "" if args.review_round.lower() == "all" else args.review_round
    index = pc_membership.load_pc_accounts(args.pcinfo)
    holders = track_tag_holders(index)
    chair_emails = area_chair_emails(args.csv, args.overrides, pcinfo_path=args.pcinfo)
    rows = account_rows(holders, chair_emails, index, args.track_clear_ceiling)
    strays = stray_paper_tags(args.data, args.track_clear_ceiling)

    write_clear_assignment_csv(args.assignment_out, review_round)
    write_clear_paper_tag_csv(args.paper_tag_out, strays, args.track_clear_ceiling)
    write_clear_account_tag_csv(args.account_tag_out, rows)

    scope = "every round" if not review_round else f"round {review_round}"
    print(f"{args.assignment_out}: Assignments -> Bulk update; clears all reviews in {scope}")
    print(f"{args.paper_tag_out}: Assignments -> Bulk update; clears "
          f"track_0..track_{args.track_clear_ceiling - 1} from all papers"
          + (f", plus {', '.join(strays)}" if strays else ""))
    print(f"{args.account_tag_out}: Settings -> Accounts; removes "
          f"~~track_0..~~track_{args.track_clear_ceiling - 1} from {len(rows)} account(s) "
          f"({len(holders)} hold a track tag in {args.pcinfo} today)")
    print("Preview each upload in HotCRP before approving it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
