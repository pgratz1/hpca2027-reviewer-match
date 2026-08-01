"""Build a publication cache for every roster PID from a local DBLP dump.

    python -m scripts.build_dblp_snapshot_cache
    python -m scripts.build_dblp_snapshot_cache --snapshot data/inputs/dblp-2026-07-01.xml
    python -m scripts.build_dblp_snapshot_cache --role reserve     # one roster only

Asking dblp.org for ~700 person records is more than it will serve politely:
doing it has produced an outright IP block (connections reset), read timeouts,
and 503s. DBLP publishes the whole database as one XML dump, and everything the
pipeline asks the network for is in it. This reads that dump once and writes the
answers to a cache the existing loaders already understand.

The dump does not link publications to PIDs. There is no `pid` attribute
anywhere in it; publications name their authors as strings. The link comes from
the person records — `<www key="homepages/PID">` — which list the name strings
belonging to each PID. So this runs two passes: the first learns which names
belong to the PIDs we want, the second collects every publication written under
one of those names. Matching is exact, not fuzzy: DBLP guarantees a name string
identifies one person, which is what the "0001"/"0049" suffixes are for.

Output is `{pid: [{title, year, venue, type, doi}, ...]}` — the same shape the
colleague cache uses, so `dblp.load_rich_cache` reads it directly and
`dblp.load_colleague_cache` normalises the very same file to the [[year, title]]
form build_fingerprints.py wants. No consumer needs to know where it came from.

Pass 1 also writes `dblp_affiliations.json`, `{pid: ["Institution, City,
Country", ...]}`, from the `<note type="affiliation">` records sitting alongside
the name strings. It is a separate file because `--out` has a shape the loaders
already read. This is the one place in the pipeline where a country is stated
outright instead of inferred, and it is what lets `affiliation_country` place
the roster offline.

Two more files fall out of the same two passes, and back the derived co-author
COI (`reviewer_match.coauthor_coi`):

  * `dblp_coauthors.json`, `{pid: {"Co Author": [[year, title], ...]}}` — pass 2
    already reads every author of every record to decide who owns it, so the
    names cost nothing beyond keeping them. Written for a wider window
    (`--coauthor-years`) than the COI check uses, so narrowing that check does
    not mean re-reading 5 GB.
  * `dblp_author_names.json`, `{pid: ["Spelling", ...]}` — every DBLP name
    spelling of the PIDs *submission authors* declare for themselves. Pass 1 is
    one filtered scan and does not care how large its `wanted` set is, so
    covering the ~2,700 authors who supplied a DBLP link is free. It is what
    lets a co-author named one way in DBLP match an author named another way in
    HotCRP.

Both are separate files for the same reason `dblp_affiliations.json` is: `--out`
has a shape the loaders already understand, and a second shape inside it would
break them. The publication records themselves are left untouched — they are
built to be indistinguishable from a live fetch, and the fingerprint cache keys
off their contents.

A snapshot is a fixed point in time: anyone added to a roster after it was taken
is absent, and is reported at the end rather than silently left with no
publications. Those still need the live path.
"""

from __future__ import annotations

from reviewer_match.paths import assignment_path, cache_path, curated_path, input_path, report_path

import argparse
import datetime
import html
import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from reviewer_match.dblp import index_self_declared_pids, normalise_doi, save_cache
from reviewer_match.roster import ROLES, load_roster

DEFAULT_SNAPSHOT = input_path("dblp-2026-07-01.xml")
DEFAULT_OUT = cache_path("dblp_snapshot_cache.json")
DEFAULT_DATA = input_path("hpca2027-data.json")

# Kept apart from DEFAULT_OUT deliberately: that file is {pid: [publication]} and
# dblp.load_rich_cache reads it directly, so a second shape in it would break the
# loaders that already understand it.
DEFAULT_AFFILIATIONS_OUT = cache_path("dblp_affiliations.json")
DEFAULT_COAUTHORS_OUT = cache_path("dblp_coauthors.json")
DEFAULT_AUTHOR_NAMES_OUT = cache_path("dblp_author_names.json")

# Wider than the COI window on purpose. Re-reading the dump costs four minutes,
# so the file is built once with room to spare and `coauthor_coi` narrows it at
# read time; see DEFAULT_COAUTHOR_YEARS there for the window actually enforced.
DEFAULT_COAUTHOR_YEARS = 10

# Person records live under this key prefix; everything else in <www> is a
# stray web page, not a human.
_HOMEPAGE_PREFIX = "homepages/"

# Publication elements carry the author's name as text. `www` is excluded
# because a person record lists its own owner as an author and would otherwise
# read as a publication of theirs, exactly as the live path excludes it.
_SKIP_TAGS = frozenset({"www"})


def entity_aware_parser() -> ET.XMLParser:
    """An XMLParser that knows DBLP's HTML entities.

    The dump declares a DTD that is not distributed with it and uses named HTML
    entities (`&ccedil;`, `&AElig;`, ...). Without the DTD a stock parse dies on
    the first one with "undefined entity", so the table is seeded from Python's
    own copy instead of shipping a DTD alongside the data.
    """
    parser = ET.XMLParser()
    for name, value in html.entities.entitydefs.items():
        parser.entity[name] = value
    return parser


def top_level_records(path: str):
    """Yield each direct child of <dblp>, complete, then release it.

    Depth has to be tracked rather than clearing on every `end` event: a record's
    own <author> and <title> children close before it does, so clearing eagerly
    empties the very element about to be handed over. Releasing only at depth 1,
    after the record is complete, is what keeps a 5 GB file from becoming a 5 GB
    tree in memory.
    """
    depth = 0
    root = None
    for event, elem in ET.iterparse(path, events=("start", "end"),
                                    parser=entity_aware_parser()):
        if event == "start":
            if root is None:
                root = elem
            depth += 1
            continue
        depth -= 1
        if depth != 1:
            continue
        yield elem
        elem.clear()
        while root is not None and len(root):
            del root[0]


def _title_text(pub) -> str | None:
    """Full title text, flattening the markup DBLP puts inside titles."""
    title_el = pub.find("title")
    if title_el is None:
        return None
    text = "".join(title_el.itertext()).strip()
    return text or None


def _publication_doi(pub) -> str:
    """Normalised DOI from the record's <ee> links, or "" if none is one."""
    for ee in pub.findall("ee"):
        doi = normalise_doi((ee.text or "").strip())
        if doi:
            return doi
    return ""


def collect_person_records(path: str, wanted: set[str]) -> dict[str, dict]:
    """Pass 1: {pid: {"names": [...], "affiliations": [...]}} for `wanted`.

    The affiliation notes sit in the very records the name strings come from, so
    they cost nothing to pick up here and save the region cap a second network
    source. DBLP writes them canonically as "Institution, City, Country", which
    is the only place in this pipeline a country is stated outright rather than
    inferred; `affiliation_country` reads them as its second layer.
    """
    people: dict[str, dict] = {}
    for elem in top_level_records(path):
        if elem.tag != "www":
            continue
        key = elem.get("key") or ""
        if not key.startswith(_HOMEPAGE_PREFIX):
            continue
        pid = key[len(_HOMEPAGE_PREFIX):]
        if pid in wanted:
            found = ["".join(a.itertext()).strip() for a in elem.findall("author")]
            notes = [
                "".join(n.itertext()).strip()
                for n in elem.findall("note")
                if n.get("type") == "affiliation"
            ]
            people[pid] = {
                "names": [n for n in found if n],
                "affiliations": [n for n in notes if n],
            }
    return people


def collect_names(path: str, wanted: set[str]) -> dict[str, list[str]]:
    """Pass 1, names only: {pid: [name strings]} for the PIDs in `wanted`."""
    return {pid: rec["names"] for pid, rec in collect_person_records(path, wanted).items()}


def collect_publications(
    path: str,
    pid_by_name: dict[str, str],
    *,
    coauthors: dict | None = None,
    coauthor_cutoff: int | None = None,
) -> dict[str, list[dict]]:
    """Pass 2: {pid: [record]} for every publication written under a known name.

    Records are built exactly as dblp._fetch_all_records_from_dblp builds them,
    so a cached entry is indistinguishable from a live one. Nothing may be added
    to that dict: the fingerprint cache keys off the publication list, so a sixth
    field would re-embed every snapshot-sourced reviewer for nothing.

    Pass `coauthors` to also collect who each PID wrote with, into
    `{pid: {name: [[year, title], ...]}}`. Publications older than
    `coauthor_cutoff` are still returned as records but contribute no
    co-authors. The names are read either way, to decide ownership, so keeping
    them costs one dict write per author.
    """
    records: dict[str, list[dict]] = defaultdict(list)
    for elem in top_level_records(path):
        if elem.tag in _SKIP_TAGS:
            continue

        names = [
            name for name in
            ("".join(a.itertext()).strip() for a in elem.findall("author"))
            if name
        ]
        owners = {pid_by_name[name] for name in names if name in pid_by_name}
        if not owners:
            continue

        title = _title_text(elem)
        year_el = elem.find("year")
        if title is None or year_el is None or not (year_el.text or "").strip():
            continue
        try:
            year = int(year_el.text.strip())
        except ValueError:
            continue

        venue_el = elem.find("booktitle")
        if venue_el is None:
            venue_el = elem.find("journal")
        venue = (venue_el.text or "").strip() if venue_el is not None else ""
        record = {
            "title": title, "year": year, "venue": venue,
            "type": elem.tag, "doi": _publication_doi(elem),
        }
        for pid in owners:
            records[pid].append(record)

        if coauthors is None or (coauthor_cutoff is not None and year < coauthor_cutoff):
            continue
        for pid in owners:
            for name in names:
                # Compare PIDs, not strings: a person writing under an alias is
                # still themselves, and pass 1 collected every spelling they use.
                if pid_by_name.get(name) == pid:
                    continue
                coauthors.setdefault(pid, {}).setdefault(name, []).append([year, title])

    for pubs in records.values():
        pubs.sort(key=lambda r: r["year"], reverse=True)
    if coauthors is not None:
        for shared in coauthors.values():
            for pubs in shared.values():
                pubs.sort(key=lambda yt: yt[0], reverse=True)
    return dict(records)


def wanted_pids(roles: list[str]) -> dict[str, str]:
    """{pid: display name} over the requested rosters."""
    wanted: dict[str, str] = {}
    for role in roles:
        for person in load_roster(role):
            if person.pid:
                wanted.setdefault(person.pid, person.name)
    return wanted


def author_pids(data_path: str) -> set[str]:
    """Every DBLP PID a submission author declared for themselves.

    Read from the whole export rather than through `load_papers`: which papers
    count as registered is a policy that moves, and a name spelling harvested
    for a paper that later withdraws costs one dict entry.
    """
    with open(data_path, encoding="utf-8") as f:
        papers = json.load(f)
    return {
        counts.most_common(1)[0][0]
        for counts in index_self_declared_pids(papers).values()
        if counts
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT,
                        help=f"DBLP XML dump (default: {DEFAULT_SNAPSHOT})")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help=f"publication cache to write (default: {DEFAULT_OUT})")
    parser.add_argument("--affiliations-out", default=DEFAULT_AFFILIATIONS_OUT,
                        help=f"DBLP affiliation notes to write, {{pid: [note]}} "
                             f"(default: {DEFAULT_AFFILIATIONS_OUT})")
    parser.add_argument("--coauthors-out", default=DEFAULT_COAUTHORS_OUT,
                        help=f"co-authors to write, {{pid: {{name: [[year, title]]}}}} "
                             f"(default: {DEFAULT_COAUTHORS_OUT})")
    parser.add_argument("--author-names-out", default=DEFAULT_AUTHOR_NAMES_OUT,
                        help=f"DBLP name spellings of submission authors to write, "
                             f"{{pid: [name]}} (default: {DEFAULT_AUTHOR_NAMES_OUT})")
    parser.add_argument("--coauthor-years", type=int, default=DEFAULT_COAUTHOR_YEARS,
                        help=f"calendar years of co-authorship to keep; the COI "
                             f"check narrows this at read time "
                             f"(default: {DEFAULT_COAUTHOR_YEARS})")
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help=f"HotCRP submissions, for the authors' own DBLP links "
                             f"(default: {DEFAULT_DATA})")
    parser.add_argument("--role", action="append", choices=ROLES, default=None,
                        help="roster to cover; repeatable (default: all three)")
    args = parser.parse_args()

    if not Path(args.snapshot).exists():
        raise SystemExit(f"{args.snapshot}: not found")
    if not Path(args.data).exists():
        raise SystemExit(f"{args.data}: not found")

    roles = args.role or list(ROLES)
    wanted = wanted_pids(roles)
    print(f"{len(wanted)} PID(s) wanted across {', '.join(roles)}", file=sys.stderr)

    authors = author_pids(args.data)
    print(f"{len(authors)} PID(s) declared by submission authors, for name "
          f"spellings only", file=sys.stderr)

    print(f"Pass 1/2: reading person records from {args.snapshot} ...", file=sys.stderr)
    people = collect_person_records(args.snapshot, set(wanted) | authors)
    names = {pid: rec["names"] for pid, rec in people.items() if pid in wanted}
    print(f"  matched {len(names)} of {len(wanted)} PID(s) to a person record",
          file=sys.stderr)

    author_names = {
        pid: rec["names"] for pid, rec in people.items()
        if pid in authors and rec["names"]
    }
    save_cache(author_names, args.author_names_out)
    print(f"  {len(author_names)} of {len(authors)} author PID(s) resolved to a "
          f"name -> {args.author_names_out}", file=sys.stderr)

    # Roster only: this file is the roster's affiliation layer, and an author
    # who never reviews would only add rows affiliation_country never reads.
    affiliations = {
        pid: rec["affiliations"] for pid, rec in people.items()
        if pid in wanted and rec["affiliations"]
    }
    save_cache(affiliations, args.affiliations_out)
    print(f"  {len(affiliations)} PID(s) carry a DBLP affiliation note -> "
          f"{args.affiliations_out}", file=sys.stderr)

    pid_by_name: dict[str, str] = {}
    for pid, spellings in names.items():
        for name in spellings:
            # A name string belongs to one person in DBLP; if the dump ever
            # disagrees, keep the first and say so rather than silently
            # reassigning someone's publications.
            if name in pid_by_name and pid_by_name[name] != pid:
                print(f"  WARNING: {name!r} claimed by {pid_by_name[name]} and "
                      f"{pid}; keeping the first", file=sys.stderr)
                continue
            pid_by_name[name] = pid
    print(f"  {len(pid_by_name)} name spelling(s) to match on", file=sys.stderr)

    print("Pass 2/2: collecting publications ...", file=sys.stderr)
    coauthors: dict[str, dict[str, list]] = {}
    cutoff = datetime.date.today().year - args.coauthor_years + 1
    records = collect_publications(
        args.snapshot, pid_by_name, coauthors=coauthors, coauthor_cutoff=cutoff,
    )

    # A PID with a person record but no publications is a real answer (an empty
    # list), not a gap; only PIDs the dump never mentioned are missing. The same
    # holds for someone who has published only alone.
    for pid in names:
        records.setdefault(pid, [])
        coauthors.setdefault(pid, {})

    # Sorted so a re-run is byte-identical. Both dicts are keyed in the order
    # `owners` -- a set of PID strings -- happened to iterate, and Python
    # randomises string hashing per process, so without this the same dump
    # produces the same data in a different order every time and `cmp` cannot
    # tell a real change from none.
    save_cache(dict(sorted(records.items())), args.out)
    save_cache(dict(sorted(coauthors.items())), args.coauthors_out)

    total = sum(len(v) for v in records.values())
    print(f"\nWrote {len(records)} PID(s), {total} publication(s) -> {args.out}",
          file=sys.stderr)
    distinct = sum(len(v) for v in coauthors.values())
    print(f"Wrote {distinct} co-author link(s) since {cutoff} -> "
          f"{args.coauthors_out}", file=sys.stderr)
    missing = sorted(set(wanted) - set(records))
    if missing:
        print(
            f"\n{len(missing)} PID(s) are not in this snapshot and still need a "
            f"live fetch — expected for anyone added to a roster after it was "
            f"taken:", file=sys.stderr,
        )
        for pid in missing:
            print(f"    {pid:<16} {wanted[pid]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
