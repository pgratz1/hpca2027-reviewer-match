"""Fill only the reviewer slots a removed reviewer leaves open, freezing every other pair.

    python -m scripts.fill_open_slots --baseline outputs/assignments/assignment.csv \\
        --removed-email person@example.edu --paper-policy submitted \\
        --include-reserves --reserve-cap 6 --same-country-cap 1 --max-juniors 2 \\
        --exclude-pids 1152 --pairs-csv new-pairs.csv --merged-hotcrp-csv patched.csv

    python -m scripts.fill_open_slots --baseline outputs/assignments/assignment.csv \\
        --pids 247,1043 --pairs-csv new-pairs.csv

    # Mid-review: pull two reviewers off whatever they have not submitted,
    # top every paper that drops below 5 back up to 5 from spare capacity,
    # at most 2 new papers per reviewer, and write only the change:
    python -m scripts.fill_open_slots --baseline data/inputs/hpca2027-pcassignments.csv \\
        --log data/inputs/hpca2027-log.csv \\
        --removed-email a@example.edu --removed-email b@example.edu \\
        --fill-to 5 --max-new-per-reviewer 2 --paper-policy submitted \\
        --include-reserves --reserve-cap 6 --same-country-cap 1 --max-juniors 2 \\
        --exclude-pids 1152 --pairs-csv new-pairs.csv --delta-hotcrp-csv delta.csv

Unlike scripts.assign_reviewers, which is always a fresh global solve, this is
a *patch*: every pair in --baseline not touching an orphaned paper is frozen
exactly as it stood, and only the open slot(s) on the named paper(s) are
filled. Use this to see the smallest possible fix for a removed reviewer,
as opposed to `assign_reviewers.py`'s from-scratch rerun (which can ripple
into papers that had nothing to do with the removal, because reviewer
capacity is shared across the whole conference in a single solve).

--removed-email derives the orphaned papers from the baseline's own record
for that address (intersected with the current paper selection -- a baseline
pid no longer in today's selection needs no replacement and is reported, not
silently dropped) and refuses to run if that address is still on the current
roster, unless --force. --pids names papers explicitly instead, for
rechecking a slate or handling more than one removal at once. Either way, the
number of slots filled per paper is `len(baseline slate) - len(survivors)` --
never `--reviewers-per-paper - len(survivors)` -- so a paper that already
carried a surplus reviewer refills to its own original size, not back down to
the base target. --reviewers-per-paper here only clamps the senior-anchor
target (min(--min-seniors, --reviewers-per-paper)); it never sets the fill
target.

Every decision function (COI layers, the area-gate ladder, the seniority
anchor phases, the junior/out-of-area/same-country caps, deferred
acceptance itself) is imported from scripts.assign_reviewers and
reviewer_match.paper_matching and reused verbatim -- restricted to the
orphaned papers only, which is what keeps this a patch and not a rerun. No
surplus stage runs: freezing means freezing, and a bonus slot for an
already-full paper is out of scope for a targeted fix.

Once reviews are being written, four flags turn this into a live patch
(`make fill-slots`). `--log` reads HotCRP's action log: a removed
reviewer's *submitted* reviews stay (they still count toward the slate and
are never cleared), only unsubmitted ones are taken away, and a removed
reviewer still on the PC needs no --force -- they are never a candidate
either way. `--fill-to N` refills each paper to N reviewers rather than to
its own baseline size, so a paper that drops from 6 to 5 needs nothing.
`--max-new-per-reviewer N` bounds how many new papers any one reviewer
receives, on top of their tier cap. `--delta-hotcrp-csv` writes just the
change -- a clear row per removed pair, an add row per new one -- and never
the `all,clearreview,all,R1` full replacement --merged-hotcrp-csv writes,
which with reviews in progress is one upload from deleting finished work.
The baseline for this mode should be HotCRP's own Search -> Download ->
"Review assignments" export: it is what HotCRP holds now, and papers since
desk-rejected are simply absent from it, which is what frees their
reviewers' capacity.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from reviewer_match import affiliation_country
from reviewer_match import assignment_io
from reviewer_match import coauthor_coi
from reviewer_match import collaborator_coi
from reviewer_match import fingerprint as fp
from reviewer_match import hotcrp_log
from reviewer_match import pc_membership
from reviewer_match.area_chairs import area_chair_emails, drop_area_chairs
from reviewer_match.paper_matching import build_paper_fingerprints, load_papers, parse_exclude_pids, PAPER_POLICIES
from reviewer_match.paths import cache_path, report_path
from reviewer_match.reserve_reviewers import DEFAULT_INFO as DEFAULT_RESERVE_INFO
from reviewer_match.reserve_reviewers import load_reserve_reviewers
from reviewer_match.reviewers import DEFAULT_CAP_OVERRIDES, load_reviewers
from reviewer_match.roster import DEFAULT_AREA_CHAIR_CSV

from scripts import assign_reviewers as ar
from scripts.build_targeted_rerun_pins import load_email_list
from scripts.classify_reviewers import DEFAULT_OUT as DEFAULT_SENIORITY, load_seniority

# "Review 4861 edited, updated draft: ..." -- a saved but unsubmitted review.
DRAFT_RE = re.compile(r"^Review (\d+) edited, updated draft")


def draft_review_ids(rows: list[dict[str, str]]) -> set[int]:
    """Review ids with at least one draft save in the log -- work a clear would delete."""
    out = set()
    for row in rows:
        m = DRAFT_RE.match(row["action"])
        if m:
            out.add(int(m.group(1)))
    return out


def emptied_reviewers(rows: list[dict[str, str]], live: dict) -> set[str]:
    """HotCRP addresses once given R1 reviews, every one of which has since been unassigned.

    That is what a departure looks like in HotCRP: the account can keep its
    `pc` role and tier tag long after the chair has pulled all its reviews,
    so neither the roster nor pc_membership sees it, and its empty load
    reads as the most spare capacity on the committee. A desk rejection does
    not unassign anything, so a reviewer whose papers were all desk-rejected
    is not caught here.
    """
    ever = set()
    for row in rows:
        m = hotcrp_log.ASSIGN_RE.match(row["action"])
        if m and m.group(3) == "R1":
            ever.add(row["affected_email"])
    still = {review.email for review in live.values() if review.round == "R1"}
    return ever - still


def dropped_paper_holders(live: dict, data_pids: set[int]) -> dict[str, list[int]]:
    """{HotCRP address: [pids]} for live R1 reviews on papers the export no longer carries.

    A desk-rejected paper leaves the paper export but its reviews stay
    assigned in HotCRP, so the log still shows them live: these are exactly
    the reviewers whose load a desk rejection lightened.
    """
    out: dict[str, list[int]] = defaultdict(list)
    for review in live.values():
        if review.round == "R1" and review.pid not in data_pids:
            out[review.email].append(review.pid)
    return {email: sorted(pids) for email, pids in out.items()}


def still_on_roster(removed_email: str, reviewers_by_email: dict, force: bool) -> bool:
    """True when --removed-email needs --force: they're still a current candidate.

    Running this tool against someone still eligible is almost certainly a
    mistake -- they were never actually removed, or the wrong address was
    given -- so it refuses by default rather than silently "fixing" slots
    that were never broken.
    """
    return removed_email in reviewers_by_email and not force


def removed_emails_of(removed) -> frozenset[str]:
    """`removed` (None, one email, or a collection of emails) as a frozenset."""
    if removed is None:
        return frozenset()
    if isinstance(removed, str):
        return frozenset({removed})
    return frozenset(removed)


def is_removed(pid: int, email: str, removed, kept: frozenset = frozenset()) -> bool:
    """True when (pid, email) is a pair this run takes away.

    A removed reviewer's pair in `kept` (a review they already submitted,
    see `split_submitted`) stays where it is: it still counts toward the
    paper's slate and is never cleared.
    """
    return email in removed_emails_of(removed) and (pid, email) not in kept


def split_submitted(
    removed_pairs: list[tuple[int, str]],
    hotcrp_email: dict[str, str],
    rid_by_pair: dict[tuple[int, str], int],
    submitted: set[int],
) -> tuple[frozenset, list[tuple[int, str]]]:
    """(pairs to keep because the review is submitted, pairs the log cannot place).

    `removed_pairs` are roster-email keyed; the log is keyed on HotCRP's
    lower-cased address. A pair the log has no live review id for cannot be
    shown to be unsubmitted, so it is kept too and returned separately for a
    warning -- clearing a review that turns out to be finished is the one
    outcome here that cannot be undone.
    """
    kept, unplaced = set(), []
    for pid, email in removed_pairs:
        rid = rid_by_pair.get((pid, hotcrp_email.get(email, email).lower()))
        if rid is None:
            kept.add((pid, email))
            unplaced.append((pid, email))
        elif rid in submitted:
            kept.add((pid, email))
    return frozenset(kept), unplaced


def derive_orphaned_pids(
    baseline_pairs: dict[int, dict], removed, current_pids: set[int], kept: frozenset = frozenset()
) -> tuple[list[int], list[int]]:
    """(orphaned pids still in the current selection, baseline pids that dropped out).

    `removed` is one email or several; a pid is orphaned when it holds at
    least one pair `is_removed` takes away.
    """
    baseline_pids = sorted(
        pid for pid, emails in baseline_pairs.items()
        if any(is_removed(pid, email, removed, kept) for email in emails)
    )
    orphaned = [pid for pid in baseline_pids if pid in current_pids]
    dropped = [pid for pid in baseline_pids if pid not in current_pids]
    return orphaned, dropped


def seed_used(
    baseline_pairs: dict[int, dict], removed, current_pids: set[int], kept: frozenset = frozenset()
) -> dict[str, int]:
    """{email: total baseline pairs on papers still in the current selection}.

    Counts every pid in the baseline still selected, not just the orphaned
    ones -- a replacement candidate's remaining capacity has to reflect load
    frozen everywhere else in the conference, not only on the papers being
    patched. Excludes the removed reviewer's own rows, and excludes any
    baseline pid that has since dropped out of the current paper selection
    (withdrawn, desk-rejected, or otherwise removed): that reviewer's
    obligation on a paper that no longer exists is void, and counting it
    anyway understates their true remaining capacity -- exactly what would
    hide the free capacity a paper's own removal creates for its other
    reviewers.
    """
    used: dict[str, int] = defaultdict(int)
    for pid, emails in baseline_pairs.items():
        if pid not in current_pids:
            continue
        for email in emails:
            if not is_removed(pid, email, removed, kept):
                used[email] += 1
    return used


def seed_slates_and_targets(
    baseline_pairs: dict[int, dict],
    orphaned_pids: list[int],
    reviewers_by_email: dict,
    removed,
    kept: frozenset = frozenset(),
    fill_to: int | None = None,
    live: bool = False,
) -> tuple[dict[int, list[str]], dict[int, int]]:
    """{pid: surviving emails}, {pid: additional slots to fill}.

    By default the target is `len(baseline slate) - len(survivors)`, read from
    the baseline's own recorded slate size -- never `--reviewers-per-paper`,
    which would silently shrink a paper that already carried a surplus
    reviewer. `fill_to` replaces that with `max(0, fill_to - len(survivors))`:
    a paper left at or above `fill_to` needs nothing, whatever it held before.

    A survivor no longer resolvable on the current roster (a second
    departure, independent of `removed`) is dropped from the slate and its
    slot is opened too, the same "backfill what's now open" rule applied to
    the named removal -- unless `live`, where the baseline is what HotCRP
    holds right now and nothing this run uploads clears that survivor's
    review, so it still counts. A pair in `kept` always survives.
    """
    slates: dict[int, list[str]] = {}
    target: dict[int, int] = {}
    for pid in orphaned_pids:
        emails = list(baseline_pairs.get(pid, {}))
        baseline_size = len(emails)
        survivors: list[str] = []
        seen: set[str] = set()
        for email in emails:
            if email in seen or is_removed(pid, email, removed, kept):
                continue
            if email not in reviewers_by_email and not live and (pid, email) not in kept:
                continue
            seen.add(email)
            survivors.append(email)
        slates[pid] = survivors
        if fill_to is None:
            target[pid] = baseline_size - len(survivors)
        else:
            target[pid] = max(0, fill_to - len(survivors))
    return slates, target


def check_pid_containment(touched_pids, orphaned_set: set[int]) -> list[int]:
    """pids with a new pick outside the orphaned set -- should always be empty."""
    return sorted(pid for pid in touched_pids if pid not in orphaned_set)


def check_reviewer_caps(used: dict[str, int], reviewers_by_email: dict, light_cap: int, full_cap: int, reserve_cap: int) -> list[str]:
    """Reviewers whose final total load exceeds their own tier cap -- should always be empty."""
    over = []
    for email, n in used.items():
        if email not in reviewers_by_email:
            continue
        cap = ar.reviewer_paper_cap(reviewers_by_email[email], light_cap, full_cap, reserve_cap)
        if n > cap:
            over.append(email)
    return sorted(over)


def clamp_new_per_reviewer(reviewer_cap: dict[str, int], used: dict[str, int], max_new: int) -> None:
    """Cap every candidate at `max_new` papers beyond their seeded load, in place.

    The same one-line freeze `assign_reviewers --pin-emails` uses: capacity is
    the only thing `assignment_phase` reads, so lowering it here bounds every
    phase without any of them knowing why.
    """
    for email in reviewer_cap:
        reviewer_cap[email] = min(reviewer_cap[email], used.get(email, 0) + max_new)


def check_new_per_reviewer(new_slates: dict[int, list[str]], max_new: int | None) -> list[str]:
    """Reviewers handed more than `max_new` new papers -- should always be empty."""
    if max_new is None:
        return []
    counts: dict[str, int] = defaultdict(int)
    for emails in new_slates.values():
        for email in emails:
            counts[email] += 1
    return sorted(email for email, n in counts.items() if n > max_new)


def check_hard_class_caps(slates: dict[int, list[str]], pids, capped) -> dict[int, list[int]]:
    """{pid: [class index over its cap]} -- should always be empty for a hard-capped class."""
    if not capped:
        return {}
    counts = ar.class_counts_of(slates, pids, capped)
    limits = ar.resolve_caps(capped, pids)
    violations = {}
    for pid in pids:
        bad = [k for k in range(len(capped)) if counts[pid][k] > limits[k][pid]]
        if bad:
            violations[pid] = bad
    return violations


def write_merged_hotcrp_csv(
    path: str,
    baseline_pairs: dict[int, dict],
    orphaned_set: set[int],
    slates: dict[int, list[str]],
    reviewers_by_email: dict,
) -> None:
    """The baseline, with the orphaned papers' slates patched in, everything else verbatim.

    Every non-orphaned pid's rows are rebuilt from the baseline's own (pid,
    email) content untouched -- this is what makes "never touches anything
    outside the orphaned papers" true by construction rather than merely
    asserted. Row order within a pid is not preserved (the original
    affinity-descending order isn't recoverable for pairs this run never
    scored); HotCRP's bulk importer does not care about row order.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", dir=target.parent,
            prefix=f".{target.name}.", suffix=".tmp", delete=False,
        ) as f:
            temporary = Path(f.name)
            writer = csv.writer(f)
            writer.writerow(["paper", "action", "email", "round"])
            writer.writerow(["all", "clearreview", "all", "R1"])
            for pid in sorted(baseline_pairs):
                if pid in orphaned_set:
                    continue
                for email in sorted(baseline_pairs[pid]):
                    hotcrp_email = reviewers_by_email[email].hotcrp_email if email in reviewers_by_email else email
                    writer.writerow([pid, "primaryreview", hotcrp_email, "R1"])
            for pid in sorted(orphaned_set):
                for email in sorted(slates.get(pid, [])):
                    hotcrp_email = reviewers_by_email[email].hotcrp_email if email in reviewers_by_email else email
                    writer.writerow([pid, "primaryreview", hotcrp_email, "R1"])
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_delta_hotcrp_csv(
    path: str,
    cleared: list[tuple[int, str]],
    new_slates: dict[int, list[str]],
    reviewers_by_email: dict,
) -> None:
    """Just the change, as a HotCRP bulk-assignment upload: clear rows, then add rows.

    Never opens with `all,clearreview,all,R1`: once reviews are being
    written, a full replacement is one upload away from deleting finished
    work, and a delta touches nothing it does not name. One clear row per
    `cleared` (pid, email) pair -- including papers that needed no
    replacement -- then one `primaryreview` row per new pair. Addresses go
    out as each reviewer's `hotcrp_email`, the only one HotCRP will match.
    """
    def hotcrp(email: str) -> str:
        return reviewers_by_email[email].hotcrp_email if email in reviewers_by_email else email

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", dir=target.parent,
            prefix=f".{target.name}.", suffix=".tmp", delete=False,
        ) as f:
            temporary = Path(f.name)
            writer = csv.writer(f)
            writer.writerow(list(assignment_io.HOTCRP_CSV_HEADER))
            for pid, email in sorted(cleared):
                writer.writerow([pid, "clearreview", hotcrp(email), "R1"])
            for pid in sorted(new_slates):
                for email in sorted(new_slates[pid]):
                    writer.writerow([pid, "primaryreview", hotcrp(email), "R1"])
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", required=True, help="the assignment to patch (--hotcrp-csv or --pairs-csv)")
    parser.add_argument("--removed-email", action="append", help="reviewer whose baseline slots need a replacement (repeatable)")
    parser.add_argument("--log", metavar="PATH", help="HotCRP action log; a removed reviewer's already-submitted reviews are kept "
                                                    "(and still count toward the slate), and the baseline is read as live, so a "
                                                    "removed reviewer still on the PC needs no --force")
    parser.add_argument("--exclude-candidates", metavar="PATH", help="file of addresses, one per line, never to offer a new paper")
    parser.add_argument("--only-dropped-paper-reviewers", action="store_true",
                        help="with --log: offer new papers only to reviewers who were assigned a paper since "
                             "dropped from the export (desk-rejected or withdrawn) -- the capacity those freed")
    parser.add_argument("--fill-to", type=int, metavar="N", help="fill each orphaned paper back up to N reviewers, not to its own baseline size")
    parser.add_argument("--max-new-per-reviewer", type=int, metavar="N", help="hand no reviewer more than N new papers in this run")
    parser.add_argument("--delta-hotcrp-csv", metavar="PATH", help="write only the change as a HotCRP upload: clear each removed pair, add each new one")
    parser.add_argument("--pids", help="comma-separated pids to (re)fill explicitly, instead of --removed-email")
    parser.add_argument("--force", action="store_true", help="fill slots for --removed-email even if that address is still on the current roster")
    parser.add_argument("--pairs-csv", metavar="PATH", help="write just the new pairs (pid,email,phase,rank_score,affinity)")
    parser.add_argument("--merged-hotcrp-csv", metavar="PATH", help="write the baseline with the orphaned papers patched in; requires a --hotcrp-csv baseline")
    parser.add_argument("--data", default=ar.DEFAULT_DATA, help="path to the HotCRP paper export JSON")
    parser.add_argument("--paper-policy", choices=PAPER_POLICIES, default="registered", help="paper selection policy (default: registered)")
    parser.add_argument("--exclude-pids", type=parse_exclude_pids, default=frozenset(), help="comma-separated paper IDs to exclude regardless of policy (default: none)")
    parser.add_argument("--csv", default=ar.DEFAULT_CSV, help="path to the reviewer CSV")
    parser.add_argument("--cap-overrides", default=DEFAULT_CAP_OVERRIDES, help="hand-maintained per-reviewer paper-cap override CSV")
    parser.add_argument("--pcinfo", default=pc_membership.DEFAULT_PCINFO, help="HotCRP user export deciding who is still on the PC")
    parser.add_argument("--no-pc-check", action="store_true", help="keep everyone the roster lists, even if HotCRP no longer marks them pc")
    parser.add_argument("--fingerprint-cache", default=ar.DEFAULT_FINGERPRINT_CACHE, help="path to the reviewer fingerprint cache")
    parser.add_argument("--paper-cache", default=ar.DEFAULT_PAPER_CACHE, help="path to the writable paper fingerprint cache")
    parser.add_argument(
        "--reviewers-per-paper", type=int, default=ar.DEFAULT_REVIEWERS_PER_PAPER,
        help="clamps only the senior-anchor target (min(--min-seniors, this)); the "
             "fill target always comes from each orphaned paper's own baseline "
             "slate size, never this flag (default: %(default)s)"
    )
    parser.add_argument("--light-cap", type=int, default=7, help="max papers per light PC member (default: 7)")
    parser.add_argument("--full-cap", type=int, default=15, help="max papers per full PC member (default: 15)")
    parser.add_argument("--include-reserves", action="store_true", help="add the reserve reviewers to the pool")
    parser.add_argument("--reserve-cap", type=int, default=ar.DEFAULT_RESERVE_CAP, help=f"max papers per reserve reviewer (default: {ar.DEFAULT_RESERVE_CAP})")
    parser.add_argument("--reserve-info", default=DEFAULT_RESERVE_INFO, help="reserve roster for --include-reserves")
    parser.add_argument("--reserve-fingerprint-cache", default=cache_path("reserve_fingerprints.json"), help="reserve fingerprint cache for --include-reserves")
    parser.add_argument("--reserve-seniority", default=report_path("reserve_seniority.csv"), help="reserve seniority CSV for --include-reserves")
    parser.add_argument("--area-weight", type=float, default=1.0, help="weight of the topics document relative to title+abstract (default: 1.0)")
    parser.add_argument("--no-area-gate", action="store_true", help="skip the hard area-eligibility gate")
    parser.add_argument("--seniority", default=DEFAULT_SENIORITY, help="reviewer seniority CSV from classify_reviewers.py")
    parser.add_argument("--no-seniority", action="store_true", help="skip the seniority constraints; plain fill + area-released fill")
    parser.add_argument("--min-seniors", type=int, default=1, help="senior reviewers each paper should get (default: %(default)s)")
    parser.add_argument("--max-juniors", type=int, default=1, help="max junior reviewers per paper before relaxation (default: %(default)s)")
    parser.add_argument("--max-out-of-area", type=int, default=3, help="max out-of-area reviewers per paper before relaxation (default: %(default)s)")
    parser.add_argument("--same-country-cap", type=int, default=ar.DEFAULT_SAME_COUNTRY_CAP, metavar="N", help="max reviewers from a paper's majority-author country (default: %(default)s)")
    parser.add_argument("--no-same-country-cap", action="store_true", help="disable the same-country cap entirely")
    parser.add_argument("--region-majority", type=float, default=ar.DEFAULT_REGION_MAJORITY, help="share of placed authors that must share a country for its cap to bind (default: %(default)s)")
    parser.add_argument("--region-min-resolved", type=float, default=ar.DEFAULT_REGION_MIN_RESOLVED, help="share of authors whose country must be known before a paper is judged (default: %(default)s)")
    parser.add_argument("--affiliation-countries", default=affiliation_country.DEFAULT_COUNTRIES, help="hand-maintained affiliation -> country file")
    parser.add_argument("--no-coauthor-coi", action="store_true", help="disable the derived co-author COI")
    parser.add_argument("--coauthor-years", type=int, default=coauthor_coi.DEFAULT_COAUTHOR_YEARS, help="calendar years of co-authorship that conflict (default: %(default)s)")
    parser.add_argument("--no-collaborator-coi", action="store_true", help="disable the derived declared-collaborator COI")
    parser.add_argument("--area-chair-csv", default=DEFAULT_AREA_CHAIR_CSV, help="area-chair acceptance form")
    parser.add_argument("--no-area-chair-exclusion", action="store_true", help="do not exclude area chairs from the candidate pool")
    parser.add_argument("--coauthor-cache", default=coauthor_coi.DEFAULT_COAUTHORS, help="co-author cache from make dblp-snapshot")
    parser.add_argument("--author-names-cache", default=coauthor_coi.DEFAULT_AUTHOR_NAMES, help="DBLP name spellings of submission authors")
    parser.add_argument("--no-coauthor-identity", action="store_true", help="ignore DBLP's homonym numbering in the co-author COI")
    parser.add_argument("--almost-senior-window", type=int, default=10, help="window papers for a typical reviewer to count as almost-senior (default: %(default)s)")
    parser.add_argument("--almost-junior-pubs", type=int, default=15, help="overall pubs for a junior to count as almost-not-junior (default: %(default)s)")
    parser.add_argument("--almost-out-of-area-career", type=int, default=5, help="career target-venue papers for almost-not-out-of-area (default: %(default)s)")
    parser.add_argument("--device", default="cuda", help="torch device for SPECTER2 (default: cuda)")
    args = parser.parse_args()

    if bool(args.removed_email) == bool(args.pids):
        parser.error("exactly one of --removed-email or --pids is required")
    if args.reviewers_per_paper < 0:
        parser.error("--reviewers-per-paper must be non-negative")
    if args.light_cap < 0 or args.full_cap < 0 or args.reserve_cap < 0:
        parser.error("--light-cap, --full-cap and --reserve-cap must be non-negative")
    if args.min_seniors < 0 or args.max_juniors < 0 or args.max_out_of_area < 0:
        parser.error("--min-seniors, --max-juniors, and --max-out-of-area must be non-negative")
    if args.same_country_cap < 0:
        parser.error("--same-country-cap must be non-negative")
    if args.coauthor_years <= 0:
        parser.error("--coauthor-years must be greater than 0")
    if args.fill_to is not None and args.fill_to < 0:
        parser.error("--fill-to must be non-negative")
    if args.max_new_per_reviewer is not None and args.max_new_per_reviewer < 0:
        parser.error("--max-new-per-reviewer must be non-negative")
    if args.only_dropped_paper_reviewers and not args.log:
        parser.error("--only-dropped-paper-reviewers requires --log")
    if args.delta_hotcrp_csv and not args.removed_email:
        parser.error("--delta-hotcrp-csv requires --removed-email (it clears their reviews)")

    seniority = None
    if not args.no_seniority:
        try:
            seniority = load_seniority(args.seniority)
        except FileNotFoundError:
            print(f"{args.seniority} not found — run classify_reviewers.py first, "
                  f"or pass --no-seniority to fill without the seniority constraints",
                  file=sys.stderr)
            return 1

    old_fmt, baseline_pairs = assignment_io.load_assignment_pairs(args.baseline)
    if args.merged_hotcrp_csv and old_fmt != "hotcrp":
        parser.error("--merged-hotcrp-csv requires a --hotcrp-csv baseline (got a --pairs-csv one)")

    papers, skipped_papers = load_papers(args.data, paper_policy=args.paper_policy, exclude_pids=args.exclude_pids, with_skipped=True)
    if not papers:
        print(f"No papers found in {args.data}", file=sys.stderr)
        return 1
    papers_by_pid = {p["pid"]: p for p in papers}

    pcinfo = None if args.no_pc_check else args.pcinfo
    reviewers_by_email = {
        r.email: r for r in load_reviewers(args.csv, pcinfo_path=pcinfo, cap_overrides_path=args.cap_overrides)
    }

    reserves = []
    if pcinfo:
        reserves = load_reserve_reviewers(args.reserve_info, args.data, pcinfo_path=pcinfo, cap_overrides_path=args.cap_overrides)
    elif args.include_reserves:
        reserves = load_reserve_reviewers(args.reserve_info, args.data, pcinfo_path=None, cap_overrides_path=args.cap_overrides)

    promoted, already_pc, true_reserves = ar.split_promoted_reserves(reserves, reviewers_by_email)
    merged = promoted + (true_reserves if args.include_reserves else [])
    reserve_fp: dict = {}
    if merged:
        reserve_fp = fp.load_fingerprint_cache(args.reserve_fingerprint_cache)
        reserve_seniority = {}
        if seniority is not None:
            try:
                reserve_seniority = load_seniority(args.reserve_seniority)
            except FileNotFoundError:
                reserve_seniority = {}
        reviewers_by_email.update({r.email: r for r in merged})
        if seniority is not None:
            seniority.update(reserve_seniority)

    area_chairs_dropped: list[str] = []
    if not args.no_area_chair_exclusion:
        index = None
        try:
            index = pc_membership.load_pc_accounts(args.pcinfo)
        except (FileNotFoundError, ValueError):
            index = None
        try:
            chair_emails = area_chair_emails(args.area_chair_csv, pcinfo_path=args.pcinfo if index else None)
        except FileNotFoundError:
            parser.error(f"{args.area_chair_csv}: not found; pass --no-area-chair-exclusion to skip this check")
        area_chairs_dropped = drop_area_chairs(reviewers_by_email, chair_emails, index)

    # A --hotcrp-csv baseline is keyed on hotcrp_email; every COI/scoring
    # function below is keyed on the roster email. Normalize once, up front,
    # the same way diff_assignments.py does -- getting this wrong would make
    # a survivor pc_membership matched under a second address look like they
    # left the roster, opening a slot that was never actually empty.
    email_map = assignment_io.hotcrp_email_map(reviewers_by_email)
    hotcrp_to_roster = {hotcrp: roster for roster, hotcrp in email_map.items()}
    if old_fmt == "hotcrp":
        baseline_pairs = assignment_io.remap_pairs(baseline_pairs, hotcrp_to_roster)
    removed: frozenset[str] = frozenset()
    if args.removed_email:
        removed = frozenset(
            hotcrp_to_roster.get(e.strip().lower(), e.strip().lower()) for e in args.removed_email
        )
    removed_label = ", ".join(sorted(removed))

    current_pids = set(papers_by_pid)
    kept: frozenset = frozenset()
    not_candidates: set[str] = set()
    dropped_by: dict[str, list[int]] | None = None
    if args.exclude_candidates:
        not_candidates |= {hotcrp_to_roster.get(e, e) for e in load_email_list(args.exclude_candidates)}
    if args.removed_email:
        for email in sorted(removed):
            if args.log:
                if email in reviewers_by_email:
                    print(f"{email} is still on the roster: taking them off their unsubmitted reviews "
                          f"only, and never offering them a new one.", file=sys.stderr)
            elif still_on_roster(email, reviewers_by_email, args.force):
                parser.error(f"{email} is still on the current roster; pass --force to fill slots for them anyway, "
                             f"or --log to take them off their unsubmitted reviews only")
        if args.log:
            rows = hotcrp_log.load_log(args.log)
            live, _anomalies = hotcrp_log.replay_assignments(rows)
            emptied = sorted(hotcrp_to_roster.get(e, e) for e in emptied_reviewers(rows, live))
            emptied = [e for e in emptied if e in reviewers_by_email and e not in removed]
            if emptied:
                print(f"{len(emptied)} reviewer(s) on the roster have had every R1 review unassigned in "
                      f"{args.log} -- treated as departed, never offered a paper: {', '.join(emptied)}",
                      file=sys.stderr)
            not_candidates |= set(emptied)
            if args.only_dropped_paper_reviewers:
                with open(args.data, encoding="utf-8") as f:
                    data_pids = {int(paper["pid"]) for paper in json.load(f)}
                dropped_by = {
                    hotcrp_to_roster.get(e, e): pids for e, pids in dropped_paper_holders(live, data_pids).items()
                }
                outside = {e for e in reviewers_by_email if e not in dropped_by}
                print(f"Candidate pool limited to the {len(set(reviewers_by_email) - outside)} roster reviewer(s) "
                      f"assigned a paper no longer in {args.data} (desk-rejected or withdrawn).", file=sys.stderr)
                not_candidates |= outside
            rid_by_pair = {(r.pid, r.email): rid for rid, r in live.items() if r.kind == "primary" and r.round == "R1"}
            removed_pairs = [
                (pid, email) for pid, emails in baseline_pairs.items() if pid in current_pids
                for email in emails if email in removed
            ]
            kept, unplaced = split_submitted(removed_pairs, email_map, rid_by_pair, hotcrp_log.submitted_review_ids(rows))
            for pid, email in sorted(kept - set(unplaced)):
                print(f"Keeping [{pid}] {email}: review already submitted.", file=sys.stderr)
            if unplaced:
                print(f"WARNING: {len(unplaced)} removed pair(s) have no live review in {args.log}, so cannot be shown "
                      f"unsubmitted -- kept, not cleared: "
                      f"{', '.join(f'[{pid}] {email}' for pid, email in sorted(unplaced))}", file=sys.stderr)
            drafts = draft_review_ids(rows)
            started = sorted(pair for pair in removed_pairs if pair not in kept
                             and rid_by_pair.get((pair[0], email_map.get(pair[1], pair[1]).lower())) in drafts)
            if started:
                print(f"NOTE: {len(started)} review(s) being cleared have a saved draft, which clearing deletes: "
                      f"{', '.join(f'[{pid}] {email}' for pid, email in started)}", file=sys.stderr)
        orphaned_pids, dropped_pids = derive_orphaned_pids(baseline_pairs, removed, current_pids, kept)
        if dropped_pids:
            print(f"{len(dropped_pids)} baseline paper(s) for {removed_label} are no longer in the "
                  f"current paper selection, no replacement needed: {', '.join(map(str, dropped_pids))}",
                  file=sys.stderr)
        if not orphaned_pids:
            print(f"No open slots to fill for {removed_label}.", file=sys.stderr)
            return 0
    else:
        orphaned_pids = sorted({int(p) for p in args.pids.split(",")})
        missing_baseline = [pid for pid in orphaned_pids if pid not in baseline_pairs]
        if missing_baseline:
            parser.error(f"pid(s) not found in --baseline: {', '.join(map(str, missing_baseline))}")
        missing_current = [pid for pid in orphaned_pids if pid not in current_pids]
        if missing_current:
            parser.error(f"pid(s) not in the current paper selection: {', '.join(map(str, missing_current))}")

    orphaned_set = set(orphaned_pids)
    cleared = sorted(
        (pid, email) for pid in orphaned_pids for email in baseline_pairs.get(pid, {})
        if is_removed(pid, email, removed, kept)
    )

    used = seed_used(baseline_pairs, removed, current_pids, kept)
    slates, target = seed_slates_and_targets(
        baseline_pairs, orphaned_pids, reviewers_by_email, removed, kept, fill_to=args.fill_to, live=bool(args.log),
    )
    original_slate = {pid: list(slates[pid]) for pid in orphaned_pids}

    active_pids = [pid for pid in orphaned_pids if target[pid] > 0]
    skipped_pids = [pid for pid in orphaned_pids if target[pid] <= 0]
    if skipped_pids:
        print(f"{len(skipped_pids)} orphaned paper(s) already have a full slate, nothing to fill: "
              f"{', '.join(map(str, skipped_pids))}", file=sys.stderr)
    if not active_pids:
        print("Nothing to fill.", file=sys.stderr)
        if args.pairs_csv:
            ar.write_pairs_csv(args.pairs_csv, {}, {}, {}, {})
        if args.merged_hotcrp_csv:
            write_merged_hotcrp_csv(args.merged_hotcrp_csv, baseline_pairs, orphaned_set, slates, reviewers_by_email)
        if args.delta_hotcrp_csv:
            write_delta_hotcrp_csv(args.delta_hotcrp_csv, cleared, {}, reviewers_by_email)
        return 0
    active_papers = [papers_by_pid[pid] for pid in active_pids]

    paper_cache = fp.load_fingerprint_cache(args.paper_cache)
    build_paper_fingerprints(active_papers, paper_cache, args.paper_cache, area_weight=args.area_weight, device=args.device)

    reviewer_fp = fp.load_fingerprint_cache(args.fingerprint_cache)
    reviewer_fp.update(reserve_fp)

    # A removed reviewer may still be on the roster (pulled off their reviews,
    # not off the PC): they must never be offered one of these papers back,
    # or another one in their place.
    candidate_emails = [
        e for e in reviewer_fp if e in reviewers_by_email and e not in removed and e not in not_candidates
    ]
    candidate_matrix = np.array([reviewer_fp[e]["vector"] for e in candidate_emails], dtype=np.float32)

    derived: dict = {}
    if not args.no_coauthor_coi:
        try:
            coauthors = coauthor_coi.load_coauthors(args.coauthor_cache)
            author_names = coauthor_coi.load_author_names(args.author_names_cache)
        except FileNotFoundError as exc:
            parser.error(f"{exc.filename}: not found; run `make dblp-snapshot`, "
                         f"or pass --no-coauthor-coi to fill without this check")
        with open(args.data, encoding="utf-8") as f:
            all_papers = json.load(f)
        roster = list(reviewers_by_email.values())
        coauthor_index = coauthor_coi.build_index(roster, coauthors, years=args.coauthor_years)
        derived = coauthor_coi.derive_conflicts(active_papers, coauthor_index, author_names, all_papers, use_identity=not args.no_coauthor_identity)

    collab_hard: dict = {}
    if not args.no_collaborator_coi:
        try:
            pcinfo_index = pc_membership.load_pc_accounts(args.pcinfo)
        except (FileNotFoundError, ValueError) as exc:
            parser.error(f"{exc}; needed for the declared-collaborator COI check, "
                         f"or pass --no-collaborator-coi to fill without it")
        collab_profiles = collaborator_coi.build_index(list(reviewers_by_email.values()), pcinfo_index)
        collab_derived = collaborator_coi.derive_conflicts(active_papers, collab_profiles, pcinfo_index)
        collab_hard = collaborator_coi.hard_conflicts(collab_derived)

    pair_scores = ar.build_pair_scores(
        active_papers, paper_cache, candidate_emails, candidate_matrix, reviewers_by_email,
        {p["pid"]: set(derived.get(p["pid"], ())) | collab_hard.get(p["pid"], set()) for p in active_papers},
        area_gate=not args.no_area_gate,
        light_cap=args.light_cap, full_cap=args.full_cap, reserve_cap=args.reserve_cap,
    )
    eligible_by_pid = pair_scores.eligible
    released_by_pid = pair_scores.released
    score_lookup = pair_scores.rank
    affinity_lookup = pair_scores.affinity
    reviewer_cap = pair_scores.reviewer_cap
    load_before = {e: used.get(e, 0) for e in reviewer_cap}
    tier_cap = dict(reviewer_cap)
    if args.max_new_per_reviewer is not None:
        clamp_new_per_reviewer(reviewer_cap, used, args.max_new_per_reviewer)

    # The reviewers already on each paper have to be classed too, not just the
    # candidates: they count toward its senior floor and its junior,
    # out-of-area and same-country caps. Most survivors are candidates in an
    # open pool, which hid this; a restricted pool leaves them unclassed, so a
    # senior survivor went uncounted and a paper's caps started empty.
    survivor_emails = sorted({
        e for pid in active_pids for e in slates[pid] if e in reviewers_by_email and e not in set(candidate_emails)
    })

    countries = []
    if not args.no_same_country_cap:
        try:
            country_pcinfo = pc_membership.load_pc_accounts(args.pcinfo)
        except (FileNotFoundError, ValueError):
            country_pcinfo = None
        layers = affiliation_country.load_layers(args.affiliation_countries, pcinfo_index=country_pcinfo)
        countries, reviewer_country, paper_coverage, thin_papers = ar.build_country_caps(
            active_papers, candidate_emails + survivor_emails, reviewers_by_email, args.same_country_cap, layers,
            majority=args.region_majority, min_resolved=args.region_min_resolved,
        )
    country_capped = [(c.members, c.papers) for c in countries]
    capped_pids = {pid for c in countries for pid in c.papers}

    paper_prefs = {pid: [e for e, _ in sorted(eligible_by_pid[pid], key=lambda es: -es[1])] for pid in active_pids}
    released_prefs = {pid: [e for e, _ in sorted(released_by_pid[pid], key=lambda es: -es[1])] for pid in active_pids}

    assigned_via: dict[tuple[int, str], str] = {}

    def added_so_far(pid: int) -> int:
        return len(slates[pid]) - len(original_slate[pid])

    def run_phase(label, prefs, phase_target, candidates, capped=()):
        # Pools also class the survivors (for counting); only candidates are offered.
        held, phase_prefs, phase_cap = ar.assignment_phase(
            active_pids, prefs, phase_target, slates, used, reviewer_cap, score_lookup,
            set(candidates) & set(reviewer_cap), capped,
        )
        for pid, emails in held.items():
            for e in emails:
                assigned_via[(pid, e)] = label
        return held, phase_prefs, phase_cap

    blocking = capped_blocking = 0
    if args.no_seniority:
        # Judged in this phase's own terms, the same way the seniority branch
        # judges F1 alone below: `cap1` is each candidate's REMAINING capacity
        # at the start of this phase (already net of every baseline pair
        # elsewhere in the conference, via the seeded `used`), and `held1` is
        # this phase's own picks, so the two line up. Checking against the
        # raw tier cap and the full accumulated `slates` instead would
        # under-count a candidate's true conference-wide load, since this
        # script never builds a paper_held view of anything outside the
        # orphaned papers.
        held1, prefs1, cap1 = run_phase("fill", paper_prefs, dict(target), set(reviewer_cap), country_capped)
        remaining = {pid: target[pid] - added_so_far(pid) for pid in active_pids}
        run_phase("fill (area released)", released_prefs, remaining, set(reviewer_cap), country_capped)
        pairs1 = {pid: [(e, score_lookup[(e, pid)]) for e in prefs1[pid]] for pid in active_pids}
        laminar = {pid: v for pid, v in pairs1.items() if pid not in capped_pids}
        blocking = ar.count_blocking_pairs(laminar, held1, cap1, dict(target), score_lookup, country_capped)
        crossing = {pid: v for pid, v in pairs1.items() if pid in capped_pids}
        capped_blocking = ar.count_blocking_pairs(crossing, held1, cap1, dict(target), score_lookup, country_capped) if crossing else 0
    else:
        pools, missing = ar.seniority_pools(
            set(reviewer_cap) | set(survivor_emails), seniority, args.almost_senior_window, args.almost_junior_pubs, args.almost_out_of_area_career,
        )
        if missing:
            print(f"Warning: {len(missing)} candidate reviewer(s) not in {args.seniority} — "
                  f"treated as neither senior nor junior", file=sys.stderr)

        senior_have = {pid: sum(1 for e in slates[pid] if e in pools.seniors) for pid in active_pids}
        anchor_ceiling = min(args.min_seniors, args.reviewers_per_paper)
        anchor_need = {pid: min(target[pid], max(0, anchor_ceiling - senior_have[pid])) for pid in active_pids}

        run_phase("senior anchor", paper_prefs, anchor_need, pools.seniors, country_capped)
        a2_target = {pid: max(0, anchor_need[pid] - added_so_far(pid)) for pid in active_pids}
        run_phase("senior anchor (area released)", released_prefs, a2_target, pools.seniors, country_capped)
        a3_target = {pid: max(0, anchor_need[pid] - added_so_far(pid)) for pid in active_pids}
        run_phase("almost-senior anchor", released_prefs, a3_target, pools.almost_seniors, country_capped)

        capped = [(pools.juniors, args.max_juniors), (pools.out_of_area, args.max_out_of_area), *country_capped]
        f1_seed = ar.class_counts_of(slates, active_pids, capped)
        fill_target = {pid: target[pid] - added_so_far(pid) for pid in active_pids}
        held2, prefs2, cap2 = run_phase("fill", paper_prefs, fill_target, set(reviewer_cap), capped)

        f2_target = {pid: target[pid] - added_so_far(pid) for pid in active_pids}
        run_phase("fill (area released)", released_prefs, f2_target, set(reviewer_cap), capped)
        f3_target = {pid: target[pid] - added_so_far(pid) for pid in active_pids}
        run_phase("fill (cap relaxed)", released_prefs, f3_target, pools.almost_not_juniors | pools.almost_not_out_of_area, country_capped)

        pairs2 = {pid: [(e, score_lookup[(e, pid)]) for e in prefs2[pid]] for pid in active_pids}
        laminar = {pid: v for pid, v in pairs2.items() if pid not in capped_pids}
        blocking = ar.count_blocking_pairs(laminar, held2, cap2, fill_target, score_lookup, capped, f1_seed)
        crossing = {pid: v for pid, v in pairs2.items() if pid in capped_pids}
        capped_blocking = ar.count_blocking_pairs(crossing, held2, cap2, fill_target, score_lookup, capped, f1_seed) if crossing else 0

        deep_junior_over = sum(
            1 for pid in active_pids
            if sum(1 for e in slates[pid] if e in pools.juniors and e not in pools.almost_not_juniors) > args.max_juniors
        )
        deep_oob_over = sum(
            1 for pid in active_pids
            if sum(1 for e in slates[pid] if e in pools.out_of_area and e not in pools.almost_not_out_of_area) > args.max_out_of_area
        )

    # --- Self-checks --------------------------------------------------------
    new_slates = {pid: [e for e in slates[pid] if e not in original_slate[pid]] for pid in orphaned_pids}
    containment_violations = check_pid_containment(
        (pid for pid, emails in new_slates.items() if emails), orphaned_set
    )
    cap_violations = check_reviewer_caps(used, reviewers_by_email, args.light_cap, args.full_cap, args.reserve_cap)
    country_violations = check_hard_class_caps(slates, active_pids, country_capped) if not args.no_same_country_cap else {}
    new_per_reviewer_violations = check_new_per_reviewer(new_slates, args.max_new_per_reviewer)

    # --- Report --------------------------------------------------------------
    for pid in orphaned_pids:
        p = papers_by_pid[pid]
        print(f"\n=== [{pid}] {p['title']}")
        print(f"    baseline slate size: {len(baseline_pairs.get(pid, {}))}")
        if pid in skipped_pids:
            print(f"    nothing to fill ({len(original_slate[pid])} reviewer(s) remain"
                  + (")" if args.fill_to is not None else ", matching the baseline size)"))
            continue
        for email in sorted(original_slate[pid]):
            print(f"    kept:  {email}")
        for email in sorted(e for e in baseline_pairs.get(pid, {}) if (pid, e) in set(cleared)):
            print(f"    removed: {email}")
        for email in sorted(new_slates[pid]):
            r = reviewers_by_email[email]
            phase = assigned_via.get((pid, email), "?")
            klass = (seniority or {}).get(email, {}).get("class") or "unclassified"
            freed = "" if dropped_by is None else f"  freed by {','.join(map(str, dropped_by.get(email, [])))}"
            print(f"    added: {affinity_lookup[(email, pid)]:.3f}  {r.name} <{email}>  [{r.tier}, {klass}]  "
                  f"load {load_before[email]} -> {used[email]} of {tier_cap[email]}  ({phase}){freed}")
        if target[pid] > len(new_slates[pid]):
            print(f"    *** UNDER-FILLED *** {len(new_slates[pid])} of {target[pid]} open slot(s) filled")

    total_new = sum(len(v) for v in new_slates.values())
    print(
        f"\nDone. {total_new} new reviewer-paper pair(s) placed across {len(active_pids)} "
        f"paper(s) with an open slot ({len(skipped_pids)} orphaned paper(s) needed nothing); "
        f"{len(containment_violations)} pair(s) touched outside the orphaned set — should "
        f"always be 0; {len(cap_violations)} reviewer(s) over their own cap — should always "
        f"be 0; {blocking} blocking pair(s) — should always be 0; {capped_blocking} blocking "
        f"pair(s) among capped papers (country class crosses seniority classes, not "
        f"guaranteed stable — see README); {len(country_violations)} paper(s) over the "
        f"same-country cap — should always be 0"
        + ("" if args.max_new_per_reviewer is None else
           f"; {len(new_per_reviewer_violations)} reviewer(s) given more than "
           f"{args.max_new_per_reviewer} new paper(s) — should always be 0")
        + ("" if args.no_seniority else
           f"; {deep_junior_over} paper(s) over the junior policy and {deep_oob_over} over "
           f"the out-of-area policy — should always be 0")
        + ".",
        file=sys.stderr,
    )
    if containment_violations:
        print(f"ERROR: pair(s) touched outside the orphaned set: {containment_violations}", file=sys.stderr)
        return 1

    if args.pairs_csv:
        try:
            ar.write_pairs_csv(args.pairs_csv, new_slates, assigned_via, score_lookup, affinity_lookup)
        except OSError as exc:
            print(f"ERROR: could not write pairs CSV {args.pairs_csv}: {exc}", file=sys.stderr)
            return 1
    if args.merged_hotcrp_csv:
        write_merged_hotcrp_csv(args.merged_hotcrp_csv, baseline_pairs, orphaned_set, slates, reviewers_by_email)
    if args.delta_hotcrp_csv:
        write_delta_hotcrp_csv(args.delta_hotcrp_csv, cleared, new_slates, reviewers_by_email)
        print(f"{len(cleared)} clear row(s) and {total_new} add row(s) -> {args.delta_hotcrp_csv}. This is a "
              f"delta, not a replacement -- preview it in HotCRP before approving.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
