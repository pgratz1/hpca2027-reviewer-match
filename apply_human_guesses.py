"""Layer human-found DBLP links on top of the automated no-PID resolution.

    python apply_human_guesses.py

Reads no_dblp_lookup_report.csv (the automated pass — see
lookup_no_dblp_reviewers.py) and manual_review_report_my_guesses.csv (a human
reviewer's manually-found DBLP links, in a "Human guess" column, for the rows
the automated pass couldn't confidently resolve). For every reviewer not
already confidently resolved automatically, a single unambiguous DBLP PID in
"Human guess" overrides the automated result: titles are fetched for that PID
and area-validated exactly like the automated dblp_pid path.

Writes final_identity_resolution.csv (one row per no-PID reviewer, with a
`source` column: auto_confident / human_override / low_confidence /
unresolved / non_reviewer) and prints a coverage summary to stderr.
"""

from __future__ import annotations

import csv
import random
import sys
import time

import requests

import dblp
from lookup_no_dblp_reviewers import candidate_summary, dblp_titles_to_works, reviewer_area_tokens, validate_area
from reviewers import load_reviewers

DEFAULT_CSV = "HPCA'27 PC Member Acceptance Form (Responses) - Form Responses 1.csv"
AUTO_REPORT = "no_dblp_lookup_report.csv"
GUESSES_REPORT = "manual_review_report_my_guesses.csv"
OUT_REPORT = "final_identity_resolution.csv"

# An auto-resolution this confident doesn't need a human override.
_CONFIDENT_AUTO = {("dblp_legacy", ""), ("dblp_search", ""), ("orcid", "MATCH"), ("openalex_name", "MATCH")}

OUT_FIELDS = [
    "email", "name", "tier", "affiliation", "declared_areas",
    "source", "dblp_pid", "openalex_author_id",
    "n_recent_works", "area_validation", "note",
]


def load_auto_report(path: str) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return {row["email"]: row for row in csv.DictReader(f)}


def load_guesses(path: str) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return {row["email"]: row for row in csv.DictReader(f)}


def human_dblp_pids(guess_row: dict | None) -> list[str]:
    """PIDs found in the "Human guess" column (may list more than one when
    even the human wasn't sure — see e.g. bjacob@usna.edu's two candidates)."""
    if guess_row is None:
        return []
    raw = (guess_row.get("Human guess") or "").strip()
    if not raw:
        return []
    pids = []
    for chunk in raw.split(";"):
        pid = dblp.parse_pid(chunk.strip())
        if pid and pid not in pids:
            pids.append(pid)
    return pids


def main() -> int:
    reviewers_by_email = {r.email: r for r in load_reviewers(DEFAULT_CSV)}
    auto = load_auto_report(AUTO_REPORT)
    guesses = load_guesses(GUESSES_REPORT)

    dblp_write_cache = dblp.load_cache("dblp_cache.json")
    dblp_readonly_cache = dblp.load_colleague_cache("dblp_pubs_cache.json")
    session = requests.Session()

    rows = []
    counts = {"auto_confident": 0, "human_override": 0, "low_confidence": 0, "unresolved": 0, "non_reviewer": 0}
    last_was_live = False

    for email, auto_row in auto.items():
        r = reviewers_by_email.get(email)
        guess_row = guesses.get(email)
        declared_areas = ", ".join(filter(None, [auto_row["primary"], auto_row["secondary"], auto_row["tertiary"]]))
        key = (auto_row["resolution_path"], auto_row["identity_confidence"])
        was_live_before = last_was_live
        last_was_live = False

        if guess_row is not None and guess_row.get("category") == "likely_test_submission":
            source = "non_reviewer"
            note = guess_row.get("problem", "")
            dblp_pid = openalex_author_id = ""
            n_works = ""
            area_flag = "N/A"

        elif key in _CONFIDENT_AUTO:
            source = "auto_confident"
            note = ""
            dblp_pid = auto_row["dblp_pid"]
            openalex_author_id = auto_row["openalex_author_id"]
            n_works = auto_row["n_recent_works"]
            area_flag = auto_row["area_validation"]

        else:
            pids = human_dblp_pids(guess_row)
            if len(pids) == 1:
                pid = pids[0]
                if was_live_before:
                    time.sleep(max(0.5, 3.0 + random.uniform(-0.5, 0.5) * 3.0))
                try:
                    titles, title_source = dblp.fetch_titles(
                        pid, years=3, session=session,
                        write_cache=dblp_write_cache, readonly_cache=dblp_readonly_cache,
                        fallback_when_thin=True,
                    )
                except requests.RequestException as exc:
                    source = "unresolved"
                    note = f"human-provided DBLP pid {pid} but the fetch failed ({exc}) — rerun to retry"
                    dblp_pid = openalex_author_id = ""
                    n_works = ""
                    area_flag = "N/A"
                else:
                    last_was_live = title_source == "live"
                    if last_was_live:
                        dblp.save_cache(dblp_write_cache, "dblp_cache.json")
                    works = dblp_titles_to_works(titles)
                    tokens = reviewer_area_tokens(r) if r else set()
                    area_flag, _matched = validate_area(tokens, works)
                    source = "human_override"
                    note = f"human-provided DBLP pid {pid}"
                    dblp_pid = pid
                    openalex_author_id = ""
                    n_works = len(works)
            elif len(pids) > 1:
                source = "unresolved"
                note = f"human guess listed {len(pids)} candidate PIDs ({', '.join(pids)}) — needs a final pick"
                dblp_pid = openalex_author_id = ""
                n_works = ""
                area_flag = "N/A"
            elif auto_row["identity_confidence"] == "LOW_CONFIDENCE" and auto_row["openalex_author_id"]:
                source = "low_confidence"
                note = "no human override found; keeping automated low-confidence OpenAlex match"
                dblp_pid = ""
                openalex_author_id = auto_row["openalex_author_id"]
                n_works = auto_row["n_recent_works"]
                area_flag = auto_row["area_validation"]
            else:
                source = "unresolved"
                note = guess_row.get("problem", "") if guess_row is not None else auto_row["note"]
                dblp_pid = openalex_author_id = ""
                n_works = ""
                area_flag = "N/A"

        counts[source] += 1
        rows.append({
            "email": email,
            "name": auto_row["name"],
            "tier": auto_row["tier"],
            "affiliation": auto_row["affiliation"],
            "declared_areas": declared_areas,
            "source": source,
            "dblp_pid": dblp_pid,
            "openalex_author_id": openalex_author_id,
            "n_recent_works": n_works,
            "area_validation": area_flag,
            "note": note,
        })
        print(f"  {email:40s} source={source:16s} area={area_flag}", file=sys.stderr)

    dblp.save_cache(dblp_write_cache, "dblp_cache.json")

    with open(OUT_REPORT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    real_reviewers = len(rows) - counts["non_reviewer"]
    resolved = counts["auto_confident"] + counts["human_override"] + counts["low_confidence"]
    print(f"\nWrote {len(rows)} rows to {OUT_REPORT}", file=sys.stderr)
    print(f"{counts['non_reviewer']} flagged as non-reviewer (test/junk submissions), excluded from the tally below", file=sys.stderr)
    print(
        f"{resolved} of {real_reviewers} real reviewers have a DBLP PID or OpenAlex author "
        f"({counts['auto_confident']} auto-confident, {counts['human_override']} via human override, "
        f"{counts['low_confidence']} low-confidence-but-area-matches)",
        file=sys.stderr,
    )
    print(f"{counts['unresolved']} still unresolved", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
