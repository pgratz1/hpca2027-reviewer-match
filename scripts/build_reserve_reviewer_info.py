"""Join the HotCRP reserve-reviewer upload to the vetting workbook's DBLP links.

The reserve reviewers recruited to cover the review-slot shortfall (see
estimate_reserve_need.py) arrive as two half-rosters: HotCRP's upload CSV knows
each one's account email and name but has no DBLP column, and the vetting
workbook knows their DBLP page but covers candidates who were never added. DBLP
is the identity every fingerprint in this pipeline is built from, so neither
half is usable alone.

This joins them on email and writes `reserve_reviewer_info.csv`
(`email,name,dblp`) — the reserve-side identity layer, the counterpart to
dblp_overrides.csv for the PC.

A DBLP link that belongs to the wrong person is worse than no link at all: it
silently fingerprints a stranger and matches papers to them. So a row is only
written to the roster if it survives every check below; anything doubtful goes
to `reserve_reviewer_unresolved.csv` with the reason, which is the to-do list
for hand resolution. Both files are sorted by email, so re-running against a
grown HotCRP export produces a readable diff.

  no_dblp_url    the workbook's dblp_url cell is empty
  not_in_vetting the uploaded email has no row in the workbook at all
  unparseable    the cell holds something parse_pid cannot read as a PID
  annotated      text trails the PID ("... （disambiguation page）"); parse_pid
                 would swallow it, but it is the chair recording doubt
  name_mismatch  the PID's own name is somebody else's ("Tao Zhang" ->
                 z/HaoZhang2). Offline this can only be checked for the named
                 PID form; --verify extends it to numeric PIDs
  shared_pid     two uploaded emails claim one PID. Either one is wrong, or one
                 person holds two HotCRP accounts and would draw double the
                 reviews. Both claimants are held back; which it is, is a call
                 for the chair, not for this script

    python -m scripts.build_reserve_reviewer_info
    python -m scripts.build_reserve_reviewer_info --verify     # check PIDs against DBLP
    python -m scripts.build_reserve_reviewer_info --out /tmp/dry-run.csv
"""

from __future__ import annotations

from reviewer_match.paths import assignment_path, cache_path, curated_path, input_path, report_path

import argparse
import csv
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from reviewer_match.dblp import name_tokens, parse_pid

DEFAULT_UPLOAD = input_path("reserve_reviewer_upload.csv")
DEFAULT_VETTING = input_path("reserve_reviewers_vetting_final.xlsx")
DEFAULT_SHEET = "All"
DEFAULT_OUT = report_path("reserve_reviewer_info.csv")
DEFAULT_UNRESOLVED = report_path("reserve_reviewer_unresolved.csv")
DEFAULT_OVERRIDES = curated_path("reserve_dblp_overrides.csv")
DEFAULT_PROFILE_CACHE = cache_path("dblp_profile_cache.json")
DEFAULT_SEARCH_CACHE = cache_path("dblp_author_search_cache.json")

# Two name tokens have to agree before a PID is accepted as the right person.
# One is too weak: "Tao Zhang" and "Hao Zhang 0002" share "zhang", and that pair
# is exactly the error this catches. Names that only have one token to give are
# exempted rather than failed (see names_agree).
MIN_SHARED_NAME_TOKENS = 2

# Override cells that mean "the workbook is wrong and I have no replacement",
# as opposed to an empty cell, which only means "not looked at yet".
_WITHHELD = {"none", "-", "n/a", "na", "unknown"}

# And a decision that is finished rather than pending: this person is not a
# reserve reviewer at all. Both keep someone off the roster; they are separate
# so the report can distinguish work still outstanding from work concluded.
_EXCLUDED = {"excluded", "exclude", "drop", "remove", "not-a-reviewer"}

ROSTER_FIELDS = ["email", "name", "dblp"]
UNRESOLVED_FIELDS = ["email", "name", "dblp_url", "problem", "detail"]

_SPREADSHEET_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

# The leading letters of a cell reference ("BC12" -> column BC, row 12).
_CELL_REF_RE = re.compile(r"^([A-Z]+)", re.IGNORECASE)

# A named PID's second segment is the person's name in CamelCase with an
# optional homonym number: "o/KunleOlukotun", "z/HaoZhang2", "k/StefanosKaxiras".
# A numeric PID ("26/1737") carries no name and cannot be checked this way.
_NAMED_PID_RE = re.compile(r"^[A-Za-z]+/([A-Za-z]+?)\d*$")


# ---------------------------------------------------------------------------
# Minimal .xlsx reading
# ---------------------------------------------------------------------------
# openpyxl is not in the pipeline's environment and this is the only spreadsheet
# the project reads, so the sheet is unzipped directly. Both string encodings
# are handled: the current workbook inlines its text (t="inlineStr", what Google
# Sheets exports), while Excel and LibreOffice intern it in xl/sharedStrings.xml
# (t="s") -- reading only one would silently yield a workbook of empty cells.

def _column_index(ref: str) -> int:
    """Zero-based column number for a cell reference: A -> 0, Z -> 25, AA -> 26."""
    letters = _CELL_REF_RE.match(ref)
    index = 0
    for char in (letters.group(1) if letters else "").upper():
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


def _cell_text(cell: ET.Element, shared: list[str]) -> str:
    """The displayed text of one cell, whichever way its string is stored."""
    kind = cell.get("t")
    if kind == "inlineStr":
        inline = cell.find(_SPREADSHEET_NS + "is")
        if inline is None:
            return ""
        return "".join(t.text or "" for t in inline.iter(_SPREADSHEET_NS + "t"))
    value = cell.find(_SPREADSHEET_NS + "v")
    text = (value.text or "") if value is not None else ""
    if kind == "s":
        try:
            return shared[int(text)]
        except (ValueError, IndexError):
            return ""
    return text


def read_sheet(path: str, sheet_name: str) -> list[dict[str, str]]:
    """Rows of one worksheet as dicts keyed by the header row's column names.

    Cells the writer skipped are absent from the XML, not blank, so each row is
    placed by column index and gaps are filled with "".
    """
    with zipfile.ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {
            rel.get("Id"): rel.get("Target").lstrip("/")
            for rel in rels
            if rel.get("Id") and rel.get("Target")
        }

        target = None
        names = []
        for sheet in workbook.iter(_SPREADSHEET_NS + "sheet"):
            names.append(sheet.get("name"))
            if sheet.get("name") == sheet_name:
                rel_id = next(
                    (v for k, v in sheet.attrib.items() if k.endswith("}id")), None
                )
                target = targets.get(rel_id)
        if target is None:
            raise SystemExit(
                f"{path}: no sheet named {sheet_name!r} (has: {', '.join(names)})"
            )
        if not target.startswith("xl/"):
            target = "xl/" + target

        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            table = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = [
                "".join(t.text or "" for t in item.iter(_SPREADSHEET_NS + "t"))
                for item in table.iter(_SPREADSHEET_NS + "si")
            ]
        sheet_xml = ET.fromstring(archive.read(target))

    rows: list[list[str]] = []
    data = sheet_xml.find(_SPREADSHEET_NS + "sheetData")
    for row in data if data is not None else []:
        cells: dict[int, str] = {}
        for cell in row:
            cells[_column_index(cell.get("r") or "")] = _cell_text(cell, shared)
        width = max(cells) + 1 if cells else 0
        rows.append([cells.get(i, "") for i in range(width)])

    if not rows:
        return []
    header = [h.strip() for h in rows[0]]
    records = []
    for row in rows[1:]:
        row = row + [""] * (len(header) - len(row))
        records.append(dict(zip(header, row)))
    return records


# ---------------------------------------------------------------------------
# Identity checks
# ---------------------------------------------------------------------------

def names_agree(roster_name: str, candidate: str) -> bool:
    """Do two spellings of a name refer to the same person?

    Requires MIN_SHARED_NAME_TOKENS in common, except where one of the names
    simply has fewer tokens than that -- a mononym can only ever contribute one,
    and failing it would report a mismatch the data cannot support.
    """
    wanted = name_tokens(roster_name)
    got = name_tokens(candidate)
    if not wanted or not got:
        return False
    need = min(MIN_SHARED_NAME_TOKENS, len(wanted), len(got))
    return len(wanted & got) >= need


def pid_name(pid: str) -> str | None:
    """The person's name spelled out inside a named PID, if it is that form.

    "o/KunleOlukotun" -> "Kunle Olukotun". Returns None for a numeric PID, whose
    digits say nothing about who it belongs to.
    """
    match = _NAMED_PID_RE.match(pid)
    if not match:
        return None
    return re.sub(r"(?<!^)(?=[A-Z])", " ", match.group(1))


def pid_url(pid: str) -> str:
    """Canonical DBLP person URL, in the shape dblp_overrides.csv already uses."""
    return f"https://dblp.org/pid/{pid}.html"


def load_overrides(path: str) -> dict[str, str]:
    """{email: dblp link} from the hand-maintained reserve override file.

    The same arrangement dblp_overrides.csv has for the PC: a filled cell is a
    decision someone made on purpose and outranks the workbook. Blank cells are
    the to-do list and are ignored here.
    """
    overrides: dict[str, str] = {}
    if not Path(path).exists():
        return overrides
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            email = (row.get("email") or "").strip().lower()
            link = (row.get("dblp") or "").strip()
            if email and link:
                overrides[email] = link
    return overrides


def classify_row(name: str, vetting: dict[str, str] | None, override: str = "") -> tuple:
    """(pid, problem, detail) for one uploaded reserve reviewer.

    A returned pid means the row is clean so far; shared-PID collisions can only
    be seen across the whole roster, so they are applied afterwards.
    """
    if override:
        # An explicit "none" is a decision too: it says the workbook's link is
        # known to be wrong and there is no replacement yet. Distinguishing that
        # from an unfilled cell matters, because a link left in place goes on
        # competing for its PID and can block whoever really owns it.
        if override.strip().lower() in _EXCLUDED:
            return None, "excluded", "not a reserve reviewer; excluded by hand"
        if override.strip().lower() in _WITHHELD:
            return None, "withheld", "held back by hand; no DBLP page identified yet"
        pid = parse_pid(override)
        if pid is None:
            return None, "unparseable", f"override {override!r} is not a DBLP link"
        return pid, None, ""

    if vetting is None:
        return None, "not_in_vetting", "no row in the vetting workbook"

    raw = (vetting.get("dblp_url") or "").strip()
    if not raw:
        return None, "no_dblp_url", "vetting row has no DBLP link"

    pid = parse_pid(raw)
    if pid is None:
        return None, "unparseable", "not a DBLP person link"

    # parse_pid stops at the first character that cannot be part of a PID, so a
    # hand-written note after the link parses cleanly and would vanish. Anything
    # left over past the PID (and an optional ".html") is a comment on the link.
    trailing = raw.split(pid, 1)[1] if pid in raw else ""
    trailing = re.sub(r"^\.html\b", "", trailing).strip()
    if trailing:
        return pid, "annotated", f"link annotated: {trailing}"

    spelled = pid_name(pid)
    if spelled is not None and not names_agree(name, spelled):
        return pid, "name_mismatch", f"PID names {spelled!r}"

    return pid, None, ""


def apply_shared_pids(resolved: dict[str, dict]) -> None:
    """Hold back every still-clean claimant of a PID that two of them claim.

    Runs last, so a claim already voided for its own reason no longer competes:
    if one of two claimants turns out to name somebody else, the other is the
    sole owner and stays on the roster.

    Where --verify read an institution off the record, it is quoted in the
    detail, since it is usually what tells two same-named people apart. It is
    reported rather than acted on: matching it against the claimants' email
    domains is not safe enough to seat someone on the roster -- "Korea Advanced
    Institute of Science and Technology" would claim korea.ac.kr as readily as
    kaist.ac.kr. Which claimant it is stays the chair's call.
    """
    claimants = defaultdict(list)
    for email, row in resolved.items():
        if row["pid"] and row["problem"] is None:
            claimants[row["pid"]].append(email)
    for pid, emails in claimants.items():
        if len(emails) < 2:
            continue
        affiliations = resolved[emails[0]].get("affiliations") or []
        for email in emails:
            others = sorted(e for e in emails if e != email)
            detail = f"PID {pid} also claimed by {', '.join(others)}"
            if affiliations:
                detail += f"; DBLP lists it at {affiliations[0]!r}"
            resolved[email]["problem"] = "shared_pid"
            resolved[email]["detail"] = detail


def verify_against_dblp(resolved: dict[str, dict], client) -> None:
    """Check each candidate PID's real name, and propose PIDs for rows without one.

    Offline, only a named PID reveals who it belongs to; here every PID's own
    DBLP record is read, so numeric PIDs are checked too -- which is most of
    them. A row left without a usable PID is then searched for by name and the
    hits are written to its detail column, never adopted: a search hit is a
    suggestion for the chair, not an identity.
    """
    for email in sorted(resolved):
        row = resolved[email]
        if row["pid"]:
            profile = client.profile(row["pid"])
            if profile is None:
                row["problem"] = row["problem"] or "unverified"
                row["detail"] = row["detail"] or "DBLP record could not be read"
            else:
                row["affiliations"] = profile["affiliations"]
                if any(names_agree(row["name"], n) for n in profile["names"]):
                    # DBLP confirms the person; only a mismatch guessed from the
                    # PID's own spelling is overturned, never the chair's note.
                    if row["problem"] == "name_mismatch":
                        row["problem"], row["detail"] = None, ""
                elif row["from_override"]:
                    # A hand-entered page outranks this check, which is exactly
                    # the check it was entered to settle: HotCRP says "John
                    # Alsop" and DBLP says "Johnathan Alsop", and a person
                    # decided they are the same man.
                    listed = ", ".join(profile["names"]) or "(unnamed)"
                    print(f"    kept override {row['pid']} for {row['email']} "
                          f"— DBLP spells it {listed!r}", file=sys.stderr)
                else:
                    listed = ", ".join(profile["names"]) or "(unnamed)"
                    row["problem"] = "name_mismatch"
                    row["detail"] = f"DBLP {row['pid']} is {listed}"

        if row["problem"] not in ("no_dblp_url", "name_mismatch"):
            continue
        hits = client.search(row["name"]) or []
        matches = [h for h in hits if names_agree(row["name"], h.get("name", ""))]
        if matches:
            row["detail"] += "; " if row["detail"] else ""
            row["detail"] += "candidates: " + "; ".join(
                f"{h['pid']} ({h['name']})" for h in matches[:5]
            )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_csv(path: str, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    """Write a CSV atomically (tmp + replace), as every cache write here does."""
    target = Path(path)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(target)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--upload", default=DEFAULT_UPLOAD,
        help=f"HotCRP reserve-reviewer upload CSV (default: {DEFAULT_UPLOAD})",
    )
    parser.add_argument(
        "--vetting", default=DEFAULT_VETTING,
        help=f"reserve-reviewer vetting workbook (default: {DEFAULT_VETTING})",
    )
    parser.add_argument(
        "--sheet", default=DEFAULT_SHEET,
        help=f"worksheet holding every candidate (default: {DEFAULT_SHEET})",
    )
    parser.add_argument(
        "--out", default=DEFAULT_OUT,
        help=f"roster of resolved reserve reviewers (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--overrides", default=DEFAULT_OVERRIDES,
        help=f"hand-maintained reserve DBLP links, which win (default: {DEFAULT_OVERRIDES})",
    )
    parser.add_argument(
        "--unresolved", default=DEFAULT_UNRESOLVED,
        help=f"rows needing hand resolution (default: {DEFAULT_UNRESOLVED})",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="check every PID against its DBLP record (slow, rate-limited, cached)",
    )
    parser.add_argument(
        "--profile-cache", default=DEFAULT_PROFILE_CACHE,
        help=f"--verify DBLP person-record cache (default: {DEFAULT_PROFILE_CACHE})",
    )
    parser.add_argument(
        "--search-cache", default=DEFAULT_SEARCH_CACHE,
        help=f"--verify DBLP author-search cache (default: {DEFAULT_SEARCH_CACHE})",
    )
    parser.add_argument(
        "--delay", type=float, default=None,
        help="seconds between live DBLP fetches (default: resolve_trc_members')",
    )
    args = parser.parse_args()

    with open(args.upload, newline="", encoding="utf-8-sig") as f:
        uploaded = list(csv.DictReader(f))
    if not uploaded or "email" not in uploaded[0]:
        raise SystemExit(f"{args.upload}: expected an 'email' column")

    vetting = {}
    for row in read_sheet(args.vetting, args.sheet):
        email = (row.get("email") or "").strip().lower()
        if email:
            vetting[email] = row
    overrides = load_overrides(args.overrides)
    print(
        f"{len(uploaded)} uploaded reserve reviewer(s); "
        f"{len(vetting)} vetted candidate(s) in {args.sheet!r}; "
        f"{len(overrides)} hand-maintained override(s)",
        file=sys.stderr,
    )

    resolved: dict[str, dict] = {}
    for row in uploaded:
        email = (row.get("email") or "").strip().lower()
        if not email:
            continue
        name = (row.get("name") or "").strip()
        entry = vetting.get(email)
        pid, problem, detail = classify_row(name, entry, overrides.get(email, ""))
        resolved[email] = {
            "email": email,
            "name": name,
            "raw": (entry.get("dblp_url") or "").strip() if entry else "",
            "pid": pid,
            "problem": problem,
            "detail": detail,
            "from_override": email in overrides,
            # Filled in by --verify only; the offline run has nothing to put here.
            "affiliations": [],
        }

    if args.verify:
        # Imported lazily: it pulls in requests, and the default run is offline.
        from scripts.resolve_trc_members import DEFAULT_DELAY, Dblp

        client = Dblp(
            args.profile_cache, args.search_cache,
            delay=DEFAULT_DELAY if args.delay is None else args.delay,
        )
        verify_against_dblp(resolved, client)
        print(f"{client.fetches} live DBLP fetch(es)", file=sys.stderr)

    apply_shared_pids(resolved)

    roster, unresolved = [], []
    for email in sorted(resolved):
        row = resolved[email]
        if row["problem"] is None:
            roster.append(
                {"email": email, "name": row["name"], "dblp": pid_url(row["pid"])}
            )
        else:
            unresolved.append({
                "email": email, "name": row["name"], "dblp_url": row["raw"],
                "problem": row["problem"], "detail": row["detail"],
            })

    write_csv(args.out, ROSTER_FIELDS, roster)
    write_csv(args.unresolved, UNRESOLVED_FIELDS, unresolved)

    print(f"\n{len(roster)} resolved -> {args.out}", file=sys.stderr)
    if unresolved:
        # "excluded" is a decision already made, so it is not outstanding work
        # and must not be counted as such -- otherwise the report never reaches
        # zero and stops meaning anything.
        settled = [r for r in unresolved if r["problem"] == "excluded"]
        outstanding = [r for r in unresolved if r["problem"] != "excluded"]
        print(f"{len(unresolved)} off the roster -> {args.unresolved}", file=sys.stderr)
        for problem, count in Counter(r["problem"] for r in unresolved).most_common():
            print(f"    {count:>4}  {problem}", file=sys.stderr)
        if settled:
            one = len(settled) == 1
            print(f"\n{len(settled)} of those {'is' if one else 'are'} excluded on "
                  f"purpose and {'needs' if one else 'need'} nothing.", file=sys.stderr)
        if outstanding:
            print(
                f"\nThe other {len(outstanding)} still need a DBLP link: put one in the "
                f"dblp cell of {args.overrides} (or fix the vetting workbook) and re-run. "
                f"Until then they are absent from the roster and no paper will be "
                f"matched to them.",
                file=sys.stderr,
            )
    if not args.verify:
        checkable = sum(1 for r in resolved.values() if r["pid"] and pid_name(r["pid"]))
        print(
            f"\nWARNING: run without --verify. A PID was only checked against a name "
            f"where the PID itself spells one ({checkable} of {len(resolved)} here); "
            f"the rest are taken on trust, and a link naming the wrong person is "
            f"silently fingerprinted as them. Re-run with --verify to check every "
            f"PID against its own DBLP record -- and note that doing so overwrites "
            f"{args.out}, so a plain run after a verified one loses that result.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
