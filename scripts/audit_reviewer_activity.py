"""Which assigned reviewers have not touched their assignment since it landed.

    python -m scripts.audit_reviewer_activity
    python -m scripts.audit_reviewer_activity --since '2026-08-10'
    python -m scripts.audit_reviewer_activity --signal engaged

Reads HotCRP's action log, replays who currently holds which reviews, and
reports each reviewer's most recent action since their own assignment. The
address list it writes is what `assign_reviewers.py --pin-emails` consumes: the
reviewers who *have* engaged are the ones an incremental rerun must not
disturb. This script is the only thing that decides who that is.

Offline, instant, read-only apart from its two reports.

**A HotCRP action log records no login event and no "paper viewed" event.**
Nothing here can tell you who signed in. Two proxies are reported side by side
and both are floors, never proof of the negative:

  any_activity  the account did anything at all the log records -- opened a
                paper, reset a password, edited its topic interests. The
                default gate, and the safer one: it errs towards leaving
                somebody's slate alone.
  engaged       the account did something that can only follow from opening a
                paper or a review form (downloaded a submission or the review
                file, accepted, declined or edited a review). Stronger
                evidence, but it misses a reviewer who read the abstracts on
                their review list and downloaded nothing.

Activity is credited to the account that *performed* an action, never to
`affected_email` -- a chair assigning somebody a paper writes that person's
address into the log, and counting it would mark every reviewer active the
instant they were assigned.

The cutoff is per-reviewer by default: each address is judged from the earliest
of its own live assignments, which is what "since they were assigned" means for
anybody added in a later patch. `--since` replaces that with one global
timestamp.
"""

from __future__ import annotations

from reviewer_match.paths import input_path, report_path

import argparse
import csv
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

from reviewer_match import hotcrp_log
from reviewer_match import pc_membership
from reviewer_match.roster import load_roster

DEFAULT_LOG = input_path("hpca2027-log.csv")
DEFAULT_OUT = report_path("reviewer_activity.csv")
DEFAULT_PINNED = report_path("reviewers_pinned.txt")

SIGNALS = ("any", "engaged")

OUT_FIELDS = [
    "email", "hotcrp_email", "name", "tier", "roster",
    "papers", "first_assigned", "cutoff",
    "any_activity", "engaged", "last_activity", "last_action", "n_actions",
    "pinned",
]


def build_roster(pcinfo_path: str | None) -> dict[str, object]:
    """{hotcrp address: roster record} over the PC and reserve rosters.

    Keyed on the HotCRP address rather than the roster email because that is
    the key space the log speaks in. The PC roster is applied last so a
    promoted reserve is described by their PC record, matching what
    `assign_reviewers.split_promoted_reserves` does with the same two lists.
    """
    by_hotcrp: dict[str, object] = {}
    for role in ("reserve", "reviewer"):
        try:
            roster = load_roster(role, pcinfo_path=pcinfo_path)
        except (FileNotFoundError, ValueError) as exc:
            print(f"warning: could not load the {role} roster: {exc}", file=sys.stderr)
            continue
        for record in roster:
            by_hotcrp[record.hotcrp_email.lower()] = record
    return by_hotcrp


def write_report(path: str, rows: list[dict]) -> None:
    """The per-reviewer report, atomically, sorted by email so a re-run diffs cleanly."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", dir=target.parent,
            prefix=f".{target.name}.", suffix=".tmp", delete=False,
        ) as f:
            temporary = Path(f.name)
            writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_pinned(path: str, emails: list[str]) -> None:
    """One address per line -- the `--pin-emails` shape, atomically."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent,
            prefix=f".{target.name}.", suffix=".tmp", delete=False,
        ) as f:
            temporary = Path(f.name)
            for email in emails:
                f.write(f"{email}\n")
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--log", default=DEFAULT_LOG, help="HotCRP action log CSV export")
    parser.add_argument("--out", default=DEFAULT_OUT, help="per-reviewer activity report")
    parser.add_argument(
        "--pinned-out", default=DEFAULT_PINNED,
        help="address list of the reviewers to pin (--pin-emails shape)",
    )
    parser.add_argument(
        "--since", default=None,
        help="one global cutoff, e.g. '2026-08-10' (default: each reviewer's own "
             "earliest live assignment)",
    )
    parser.add_argument(
        "--signal", default="any", choices=SIGNALS,
        help="which proxy decides who gets pinned (default: any)",
    )
    parser.add_argument("--kind", default="primary", help="review kind to audit (default: primary)")
    parser.add_argument("--round", default="R1", help="review round to audit (default: R1)")
    parser.add_argument("--pcinfo", default=pc_membership.DEFAULT_PCINFO, help="HotCRP user export")
    parser.add_argument(
        "--no-pc-check", action="store_true",
        help="skip the HotCRP PC-membership check when loading the rosters",
    )
    args = parser.parse_args()

    try:
        rows = hotcrp_log.load_log(args.log)
    except OSError as exc:
        print(f"cannot read the action log: {exc}", file=sys.stderr)
        return 1

    live, anomalies = hotcrp_log.replay_assignments(rows)
    pairs = hotcrp_log.live_pairs(live, kind=args.kind, round=args.round)
    load = Counter(email for emails in pairs.values() for email in emails)
    if not load:
        print(f"no live {args.kind}/{args.round} reviews in {args.log}", file=sys.stderr)
        return 1

    first_assigned = hotcrp_log.first_assigned_at(live)
    cutoff = args.since if args.since else {e: first_assigned.get(e, "") for e in load}
    any_seen = hotcrp_log.last_activity(rows, since=cutoff)
    engaged_seen = hotcrp_log.engagement_actions(rows, since=cutoff)
    chosen = any_seen if args.signal == "any" else engaged_seen

    by_hotcrp = build_roster(None if args.no_pc_check else args.pcinfo)

    report: list[dict] = []
    for email in sorted(load):
        record = by_hotcrp.get(email)
        seen = any_seen.get(email)
        report.append({
            "email": getattr(record, "email", ""),
            "hotcrp_email": email,
            "name": getattr(record, "name", ""),
            "tier": getattr(record, "tier", ""),
            "roster": "yes" if record is not None else "NOT ON ROSTER",
            "papers": load[email],
            "first_assigned": first_assigned.get(email, ""),
            "cutoff": cutoff if isinstance(cutoff, str) else cutoff.get(email, ""),
            "any_activity": "yes" if email in any_seen else "",
            "engaged": "yes" if email in engaged_seen else "",
            "last_activity": seen[0] if seen else "",
            "last_action": seen[1] if seen else "",
            "n_actions": seen[2] if seen else 0,
            "pinned": "yes" if email in chosen else "",
        })

    silent = [r for r in report if not r["pinned"]]
    pinned = [r["hotcrp_email"] for r in report if r["pinned"]]

    try:
        write_report(args.out, report)
        write_pinned(args.pinned_out, pinned)
    except OSError as exc:
        print(f"cannot write the reports: {exc}", file=sys.stderr)
        return 1

    # stdout is the answer to the question asked: who has not looked.
    print(f"# {len(silent)} of {len(report)} assigned reviewers show no `{args.signal}` "
          f"activity since they were assigned")
    print(f"# papers  {'reviewer':38s}  name")
    for row in sorted(silent, key=lambda r: (-r["papers"], r["hotcrp_email"])):
        print(f"{row['papers']:>8}  {row['hotcrp_email']:38s}  {row['name']}")

    print(
        f"\n{len(report)} assigned reviewer(s): {len(any_seen.keys() & load.keys())} with any "
        f"activity, {len(engaged_seen.keys() & load.keys())} with paper engagement, "
        f"{len(silent)} silent under `{args.signal}`",
        file=sys.stderr,
    )
    print(f"    reviews held by the silent: {sum(r['papers'] for r in silent)} of {sum(load.values())}",
          file=sys.stderr)
    print(f"    {args.out}", file=sys.stderr)
    print(f"    {args.pinned_out}  ({len(pinned)} address(es) to pin)", file=sys.stderr)

    off_roster = [r["hotcrp_email"] for r in report if r["roster"] != "yes"]
    if off_roster:
        print(
            f"\n{len(off_roster)} assigned address(es) are on no roster and cannot be pinned "
            f"or rematched: {', '.join(off_roster)}",
            file=sys.stderr,
        )
    if anomalies:
        print(f"\n{len(anomalies)} log anomal(ies); run extract_log_assignments.py for the breakdown",
              file=sys.stderr)
    print(
        "\nNote: HotCRP logs no login event. `any` means the account acted in the log at "
        "all, `engaged` that it opened a paper or a review form. Neither proves someone "
        "has not read their assignment.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
