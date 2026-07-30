"""Enumerate every affiliation string and resolve which country it is in.

    python build_affiliation_countries.py
    python build_affiliation_countries.py --role reviewer --role reserve
    python build_affiliation_countries.py --validate CN=china_faculty.csv \
                                          --validate '!CN=nonchina_faculty.csv'

The per-paper region cap in assign_reviewers.py needs to know where an
institution is, on both sides: which reviewers are affiliated in a region, and
what share of a paper's authors are. `affiliation_country` can place many of
them automatically, but the HotCRP affiliation cell is free text and most of it
names no country at all, so the residue has to be decided by hand.

This writes that to-do list. It collects every distinct affiliation string
across the submissions and all three rosters, runs the automatic layers over
each, and writes `affiliation_countries.csv` with the machine's answer in
`suggested` and an **empty `country` column for a human to fill in**.

Only `country` is ever read back. The generator never writes it, because
`affiliation_country` treats that column as the hand-decided layer that outranks
DBLP and everything below it -- filling it in here would collapse the waterfall
into a machine guess wearing a human's hat. A blank cell is a to-do marker, not
a decision, exactly as in dblp_overrides.csv.

Reruns are safe: existing `country` values are carried over verbatim, and rows
whose affiliation has since left the data are kept with `people = 0` rather than
dropped, so hand work does not evaporate when a paper is withdrawn.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import affiliation_country as ac
from roster import ROLES, load_roster, role_label

DEFAULT_DATA = "hpca2027-data.json"
DEFAULT_OUT = ac.DEFAULT_COUNTRIES

FIELDS = ("affiliation", "country", "suggested", "source", "people", "note")


def collect_affiliations(data_path: str, roles: list[str]) -> dict[str, dict]:
    """{normalized affiliation: {raw, people, emails, pids, roster}} over the data.

    `people` counts distinct addresses rather than rows, so one prolific author
    does not outrank an institution that ten people share; the sort order that
    decides what a human curates first depends on it.

    `roster` marks a string that some reviewer, area chair or reserve reviewer
    uses. Those are the ones that must reach 100%: an unplaced reviewer can never
    consume a region cap, so an unplaced roster string silently weakens the rule,
    while an unplaced author string only makes one paper harder to judge.
    """
    entries: dict[str, dict] = {}

    def add(raw: str, email: str, pid: str | None, is_roster: bool) -> None:
        key = ac.normalize_affiliation(raw)
        if not key:
            return
        e = entries.setdefault(
            key, {"raw": raw.strip(), "emails": set(), "pids": set(), "roster": False}
        )
        if email:
            e["emails"].add(email.strip().lower())
        if pid:
            e["pids"].add(pid)
        e["roster"] = e["roster"] or is_roster

    with open(data_path, encoding="utf-8") as f:
        papers = json.load(f)
    for p in papers:
        for person in (p.get("authors") or []) + (p.get("contacts") or []):
            add(person.get("affiliation") or "", person.get("email") or "", None, False)

    for role in roles:
        for person in load_roster(role):
            add(getattr(person, "affiliation", "") or "",
                getattr(person, "email", "") or "",
                getattr(person, "pid", None), True)

    for e in entries.values():
        e["people"] = len(e["emails"])
    return entries


def suggest(entry: dict, layers: ac.CountryLayers) -> tuple[str, str]:
    """(ISO code or "", which layer answered) from the automatic layers alone.

    The hand layer is deliberately not consulted: this column is the machine's
    independent opinion, and showing it agreeing with a value a human just typed
    would make the file look corroborated when nothing corroborated it.
    """
    notes: list[str] = []
    for pid in sorted(entry["pids"]):
        notes = layers.dblp_by_pid.get(pid) or []
        if notes:
            break
    for email in sorted(entry["emails"]):
        code, layer = ac.resolve_country(entry["raw"], email, notes, None)
        if code:
            return code, layer
    code, layer = ac.resolve_country(entry["raw"], "", notes, None)
    return (code, layer) if code else ("", "unresolved")


def merge_rows(
    existing: list[dict], entries: dict[str, dict], layers: ac.CountryLayers
) -> list[dict]:
    """Fold this run's findings into whatever the file already held.

    Hand-entered `country` values win over everything and are never rewritten.
    Rows for affiliations no longer in the data are retained with people = 0 --
    the HotCRP export is a moving snapshot, and a withdrawn paper must not delete
    a decision someone already made.
    """
    kept: dict[str, dict] = {}
    for row in existing:
        key = ac.normalize_affiliation(row.get("affiliation") or "")
        if key:
            kept[key] = row

    rows = []
    for key in set(kept) | set(entries):
        entry = entries.get(key)
        old = kept.get(key, {})
        if entry is None:
            rows.append({
                "affiliation": old.get("affiliation", key),
                "country": (old.get("country") or "").strip().upper(),
                "suggested": old.get("suggested", ""),
                "source": old.get("source", ""),
                "people": "0",
                "note": old.get("note", ""),
            })
            continue
        code, layer = suggest(entry, layers)
        rows.append({
            "affiliation": entry["raw"],
            "country": (old.get("country") or "").strip().upper(),
            "suggested": code,
            "source": layer,
            "people": str(entry["people"]),
            "note": "roster" if entry["roster"] else "",
        })

    # Most-shared first so the hand work that buys the most coverage is at the
    # top; the name breaks ties so an unchanged rerun is byte-identical.
    rows.sort(key=lambda r: (-int(r["people"]), ac.normalize_affiliation(r["affiliation"])))
    return rows


def read_existing(path: str) -> list[dict]:
    """Rows already in the file, or [] if it doesn't exist yet."""
    try:
        f = open(path, newline="", encoding="utf-8")
    except FileNotFoundError:
        return []
    with f:
        return list(csv.DictReader(f))


def write_countries(path: str, rows: list[dict]) -> None:
    """Write the country file atomically, as every cache write here does."""
    target = Path(path)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(target)


def parse_validate(spec: str) -> tuple[str, bool, str]:
    """Parse `CC=PATH` or `!CC=PATH` into (code, expected_membership, path)."""
    negate = spec.startswith("!")
    body = spec[1:] if negate else spec
    code, sep, path = body.partition("=")
    code = code.strip().upper()
    if not sep or not code or not path.strip():
        raise argparse.ArgumentTypeError(f"expected CC=PATH or !CC=PATH, got {spec!r}")
    if not ac.is_country_code(code):
        raise argparse.ArgumentTypeError(f"{code!r} is not an ISO alpha-2 code we know")
    return code, not negate, path.strip()


def validate(specs, by_email: dict[str, tuple[str, str]]) -> None:
    """Check the resolver against hand-labelled rosters and report disagreements.

    Prints the disagreements by name: those are both the highest-value entries to
    curate and the only evidence available that a layer is wrong rather than
    merely silent.
    """
    for code, expected, path in specs:
        try:
            f = open(path, newline="", encoding="utf-8")
        except FileNotFoundError:
            print(f"  {path}: not found, skipped", file=sys.stderr)
            continue
        agree = disagree = unresolved = 0
        wrong: list[str] = []
        with f:
            for row in csv.DictReader(f):
                email = (row.get("email") or "").strip().lower()
                if not email or email not in by_email:
                    continue
                got, _ = by_email[email]
                if not got:
                    unresolved += 1
                elif (got == code) == expected:
                    agree += 1
                else:
                    disagree += 1
                    wrong.append(f"{email} -> {got}")
        label = f"{'' if expected else 'not '}{code}"
        print(f"  {path}: {agree} agree, {disagree} disagree, {unresolved} unresolved "
              f"(expected {label})", file=sys.stderr)
        for w in sorted(wrong)[:20]:
            print(f"      {w}", file=sys.stderr)
        if len(wrong) > 20:
            print(f"      ... and {len(wrong) - 20} more", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help=f"HotCRP paper export (default: {DEFAULT_DATA})")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help=f"country file to write (default: {DEFAULT_OUT})")
    parser.add_argument("--role", action="append", choices=ROLES, default=None,
                        help="roster to cover; repeatable (default: all three)")
    parser.add_argument("--dblp-affiliations", default=ac.DEFAULT_DBLP_AFFILIATIONS)
    parser.add_argument("--profile-cache", default=ac.DEFAULT_PROFILE_CACHE)
    parser.add_argument("--validate", action="append", type=parse_validate, default=[],
                        metavar="CC=PATH",
                        help="check the resolver against a hand-labelled CSV with an "
                             "email column; !CC=PATH asserts the rows are NOT CC")
    args = parser.parse_args()

    roles = args.role or list(ROLES)
    # The hand layer is loaded so existing decisions can be reported, but never
    # fed to suggest(): see its docstring.
    layers = ac.load_layers(args.out, args.dblp_affiliations, args.profile_cache)
    print(f"{len(layers.overrides)} hand-decided affiliation(s), "
          f"{len(layers.dblp_by_pid)} PID(s) with a DBLP affiliation note", file=sys.stderr)

    entries = collect_affiliations(args.data, roles)
    rows = merge_rows(read_existing(args.out), entries, layers)
    write_countries(args.out, rows)

    roster_rows = [r for r in rows if r["note"] == "roster"]
    def unplaced(rs):
        return [r for r in rs if not r["country"] and not r["suggested"]]
    print(f"\nWrote {len(rows)} affiliation(s) -> {args.out}", file=sys.stderr)
    print(f"  {len(roster_rows)} used by a roster member, "
          f"{len(unplaced(roster_rows))} of those unplaced", file=sys.stderr)
    print(f"  {len(rows)} total, {len(unplaced(rows))} unplaced", file=sys.stderr)
    sources = Counter(r["source"] for r in rows if r["suggested"])
    print(f"  suggested by: {', '.join(f'{n} {s}' for s, n in sources.most_common())}",
          file=sys.stderr)

    # Coverage the caps will actually see, counted over people rather than
    # strings -- one unplaced string a hundred authors share matters far more
    # than a hundred nobody uses.
    fresh = ac.load_layers(args.out, args.dblp_affiliations, args.profile_cache)
    by_email: dict[str, tuple[str, str]] = {}
    for role in roles:
        people = load_roster(role)
        placed = 0
        for person in people:
            code, layer = ac.reviewer_country(person, fresh)
            by_email[(person.email or "").strip().lower()] = (code, layer)
            placed += bool(code)
        print(f"  {role_label(role)}: {placed} of {len(people)} placed "
              f"({100 * placed / len(people):.1f}%)" if people else
              f"  {role_label(role)}: empty roster", file=sys.stderr)

    if args.validate:
        print("\nValidation:", file=sys.stderr)
        validate(args.validate, by_email)

    if unplaced(roster_rows):
        print(f"\nFill the blank country cells in {args.out}: "
              f"{len(unplaced(roster_rows))} roster affiliation(s) are still unplaced, "
              f"and a reviewer with no country can never consume a region cap.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
