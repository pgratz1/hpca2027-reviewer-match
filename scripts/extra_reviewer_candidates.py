"""Shortlist PC members to ask for one extra review on papers that are short.

    python -m scripts.extra_reviewer_candidates --pids 178,344,583,707,1723
    python -m scripts.extra_reviewer_candidates --pids 178,344 \\
        --completed-by '2026-09-19 08:00:00 -0400' --shortlist 10

Once reviewers go missing, some papers can only be rescued by asking a
reviewer who has already finished to take on one more. This lists, for each
`--pids` paper, the best-matched full or light PC members (`~~ex-rr`
promotions count as light; reserves, TRC and area chairs never appear) who
had first-submitted every R1 review they hold on a paper still under review
before `--completed-by`. An unsubmitted review on a desk-rejected or withdrawn
paper does not count against them, the same rule the timeliness tags use.

It is a list of people to *ask*, not an assignment: nothing is uploaded, and
the tier load cap is deliberately not applied (everyone here has met theirs;
their current R1 load is shown instead). Everything else the matcher enforces
still binds, all of it imported rather than restated:

  - every COI layer: declared, own-paper, derived co-author and declared
    collaborator (`assign_reviewers.build_pair_scores`);
  - the area gate: in-area candidates are listed first, and the list is
    topped up from the area-released pool, marked `released`;
  - the junior, out-of-area and same-country caps, counted against the
    paper's current live R1 slate, outstanding reviews included, since those
    may still arrive. A candidate who would break one is skipped and counted.

Nobody who holds any review on the paper (R1 or TRC), or was ever unassigned
from it (a decline, a conflict, a swap), is listed for it. Nor is anyone
whose every R1 review was unassigned: HotCRP keeps their PC role, but they
have left.

One suggested first ask per paper is starred: the distinct choice across the
papers, drawn from the shortlists, with the greatest total affinity. Ties go
to the earlier-ranked candidate, so the choice is deterministic.

stdout gives each paper's current slate and its shortlist; `--out` gets the
same rows as CSV. Offline, no GPU: the papers' fingerprints must already be
cached, which `make` sees to. Names people, so the CSV is gitignored with the
other reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

from reviewer_match import affiliation_country
from reviewer_match import assignment_io
from reviewer_match import coauthor_coi
from reviewer_match import collaborator_coi
from reviewer_match import fingerprint as fp
from reviewer_match import hotcrp_log
from reviewer_match import pc_membership
from reviewer_match import review_scores
from reviewer_match.area_chairs import area_chair_emails, drop_area_chairs
from reviewer_match.paper_matching import load_papers, parse_exclude_pids
from reviewer_match.paths import cache_path, report_path
from reviewer_match.reserve_reviewers import DEFAULT_INFO as DEFAULT_RESERVE_INFO
from reviewer_match.reserve_reviewers import load_reserve_reviewers
from reviewer_match.reviewers import DEFAULT_CAP_OVERRIDES, load_reviewers
from reviewer_match.roster import DEFAULT_AREA_CHAIR_CSV

from scripts import assign_reviewers as ar
from scripts.classify_reviewers import DEFAULT_OUT as DEFAULT_SENIORITY, load_seniority
from scripts.fill_open_slots import emptied_reviewers

DEFAULT_LOG = "data/inputs/hpca2027-log.csv"
DEFAULT_COMPLETED_BY = "2026-09-19 08:00:00 -0400"
DEFAULT_SHORTLIST = 8
DEFAULT_OUT = report_path("extra_reviewer_candidates.csv")
PC_TIERS = ("full", "light")
REVIEW_ROUND = review_scores.REVIEW_ROUND

OUT_FIELDS = ["paper", "rank", "suggested", "name", "email", "hotcrp_email", "tier", "seniority",
              "country", "area", "affinity", "r1_reviews", "finished_at", "notes"]


def finished_reviewers(
    live: dict, submitted: set[int], first_submit: dict[int, str], under_review: set[int], completed_by,
) -> dict[str, tuple[int, object]]:
    """{HotCRP address: (R1 reviews held, when the last was first submitted)} for everyone done in time.

    Only live R1 reviews on papers still under review count; anyone holding
    none of those has nothing to have finished and is left out.
    """
    held: dict[str, list[int]] = defaultdict(list)
    for rid, review in live.items():
        if review.round == REVIEW_ROUND and review.pid in under_review:
            held[review.email].append(rid)
    out = {}
    for email, rids in held.items():
        if not all(rid in submitted and rid in first_submit for rid in rids):
            continue
        last = max(hotcrp_log.parse_date(first_submit[rid]) for rid in rids)
        if last < completed_by:
            out[email] = (len(rids), last)
    return out


def ever_on_paper(rows: list[dict[str, str]]) -> dict[int, set[str]]:
    """{pid: HotCRP addresses} of everyone ever assigned a review on it, in any round.

    Covers the live slate and anyone since unassigned, which is what keeps a
    decline or a conflict from being asked back.
    """
    out: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        if hotcrp_log.ASSIGN_RE.match(row["action"]) and row["paper"]:
            out[int(row["paper"])].add(row["affected_email"])
    return out


def suggest(shortlists: dict[int, list[tuple[str, float]]]) -> dict[int, str]:
    """{pid: email}: one distinct candidate per paper maximising total affinity.

    Exhaustive over the shortlists, which are small. A paper may go without
    (its whole list taken elsewhere, or empty), and filling more papers beats
    a better total. Papers are searched in pid order and each list in rank
    order, and only a strictly better result replaces the incumbent, so ties
    go to the earlier-ranked candidate.
    """
    pids = sorted(shortlists)
    best: tuple[tuple[int, float], dict[int, str]] = ((-1, 0.0), {})

    def walk(i: int, used: set[str], chosen: dict[int, str], total: float) -> None:
        nonlocal best
        if i == len(pids):
            key = (len(chosen), round(total, 12))
            if key > best[0]:
                best = (key, dict(chosen))
            return
        pid = pids[i]
        for email, score in shortlists[pid]:
            if email not in used:
                used.add(email)
                chosen[pid] = email
                walk(i + 1, used, chosen, total + score)
                del chosen[pid]
                used.discard(email)
        walk(i + 1, used, chosen, total)

    walk(0, set(), {}, 0.0)
    return best[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pids", required=True, help="comma-separated papers needing one more reviewer")
    parser.add_argument("--completed-by", default=DEFAULT_COMPLETED_BY,
                        help=f"candidates first-submitted their last R1 review before this, with offset (default: {DEFAULT_COMPLETED_BY!r})")
    parser.add_argument("--shortlist", type=int, default=DEFAULT_SHORTLIST, help="candidates listed per paper (default: %(default)s)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="CSV to write (default: %(default)s)")
    parser.add_argument("--log", default=DEFAULT_LOG, help="HotCRP action log CSV export")
    parser.add_argument("--data", default=ar.DEFAULT_DATA, help="HotCRP paper export JSON")
    parser.add_argument("--exclude-pids", default="", help="comma-separated paper IDs to leave out")
    parser.add_argument("--csv", default=ar.DEFAULT_CSV, help="PC acceptance form CSV")
    parser.add_argument("--cap-overrides", default=DEFAULT_CAP_OVERRIDES, help="per-reviewer paper-cap override CSV")
    parser.add_argument("--pcinfo", default=pc_membership.DEFAULT_PCINFO, help="HotCRP user export")
    parser.add_argument("--reserve-info", default=DEFAULT_RESERVE_INFO, help="reserve roster (for ~~ex-rr promotions)")
    parser.add_argument("--fingerprint-cache", default=ar.DEFAULT_FINGERPRINT_CACHE, help="reviewer fingerprint cache")
    parser.add_argument("--reserve-fingerprint-cache", default=cache_path("reserve_fingerprints.json"), help="reserve fingerprint cache")
    parser.add_argument("--paper-cache", default=ar.DEFAULT_PAPER_CACHE, help="paper fingerprint cache")
    parser.add_argument("--seniority", default=DEFAULT_SENIORITY, help="reviewer seniority CSV")
    parser.add_argument("--reserve-seniority", default=report_path("reserve_seniority.csv"), help="reserve seniority CSV")
    parser.add_argument("--area-chair-csv", default=DEFAULT_AREA_CHAIR_CSV, help="area-chair acceptance form")
    parser.add_argument("--max-juniors", type=int, default=2, help="max junior reviewers per paper (default: %(default)s)")
    parser.add_argument("--max-out-of-area", type=int, default=3, help="max out-of-area reviewers per paper (default: %(default)s)")
    parser.add_argument("--same-country-cap", type=int, default=1, metavar="N",
                        help="max reviewers from a paper's majority-author country (default: %(default)s)")
    parser.add_argument("--no-same-country-cap", action="store_true", help="disable the same-country cap")
    parser.add_argument("--region-majority", type=float, default=ar.DEFAULT_REGION_MAJORITY)
    parser.add_argument("--region-min-resolved", type=float, default=ar.DEFAULT_REGION_MIN_RESOLVED)
    parser.add_argument("--affiliation-countries", default=affiliation_country.DEFAULT_COUNTRIES)
    parser.add_argument("--coauthor-years", type=int, default=coauthor_coi.DEFAULT_COAUTHOR_YEARS)
    parser.add_argument("--coauthor-cache", default=coauthor_coi.DEFAULT_COAUTHORS)
    parser.add_argument("--author-names-cache", default=coauthor_coi.DEFAULT_AUTHOR_NAMES)
    args = parser.parse_args()

    try:
        pids = sorted({int(p) for p in args.pids.split(",") if p.strip()})
        completed_by = hotcrp_log.parse_date(args.completed_by)
    except ValueError as exc:
        parser.error(str(exc))
    for path in (args.log, args.data, args.pcinfo):
        if not os.path.exists(path):
            print(f"ERROR: {path} not found", file=sys.stderr)
            return 1

    exclude = parse_exclude_pids(args.exclude_pids)
    under_review, _ = review_scores.eligible_pids(args.data, exclude)
    missing = [pid for pid in pids if pid not in under_review]
    if missing:
        parser.error(f"not under review: {', '.join(map(str, missing))}")
    papers = {p["pid"]: p for p in load_papers(args.data, paper_policy="submitted", exclude_pids=exclude)}
    targets = [papers[pid] for pid in pids]

    # The PC as every other tool loads it: form rows plus ~~ex-rr promotions,
    # true reserves left out, area chairs dropped.
    reviewers_by_email = {r.email: r for r in load_reviewers(args.csv, pcinfo_path=args.pcinfo, cap_overrides_path=args.cap_overrides)}
    reserves = load_reserve_reviewers(args.reserve_info, args.data, pcinfo_path=args.pcinfo, cap_overrides_path=args.cap_overrides)
    promoted, _, _ = ar.split_promoted_reserves(reserves, reviewers_by_email)
    reviewers_by_email.update({r.email: r for r in promoted})
    index = pc_membership.load_pc_accounts(args.pcinfo)
    drop_area_chairs(reviewers_by_email, area_chair_emails(args.area_chair_csv, pcinfo_path=args.pcinfo), index)
    reviewers_by_email = {e: r for e, r in reviewers_by_email.items() if r.tier in PC_TIERS}
    email_map = assignment_io.hotcrp_email_map(reviewers_by_email)
    to_roster = {hotcrp.lower(): roster for roster, hotcrp in email_map.items()}

    rows = hotcrp_log.load_log(args.log)
    live, _ = hotcrp_log.replay_assignments(rows)
    finished = {
        to_roster.get(e, e): v
        for e, v in finished_reviewers(live, hotcrp_log.submitted_review_ids(rows), hotcrp_log.first_submitted_at(rows),
                                       under_review, completed_by).items()
    }
    departed = {to_roster.get(e, e) for e in emptied_reviewers(rows, live)}
    pool = sorted(e for e in reviewers_by_email if e in finished and e not in departed)
    tiers = Counter(reviewers_by_email[e].tier for e in pool)
    print(f"{len(reviewers_by_email)} full/light PC members (area chairs dropped); {len(pool)} finished every R1 "
          f"review before {args.completed_by} ({tiers['full']} full, {tiers['light']} light).", file=sys.stderr)

    reviewer_fp = fp.load_fingerprint_cache(args.fingerprint_cache)
    reviewer_fp.update(fp.load_fingerprint_cache(args.reserve_fingerprint_cache))
    no_fp = [e for e in pool if e not in reviewer_fp]
    if no_fp:
        print(f"Warning: {len(no_fp)} finished reviewer(s) have no fingerprint and cannot be ranked: "
              f"{', '.join(no_fp)}", file=sys.stderr)
    candidates = [e for e in pool if e in reviewer_fp]
    matrix = np.array([reviewer_fp[e]["vector"] for e in candidates], dtype=np.float32)

    paper_cache = fp.load_fingerprint_cache(args.paper_cache)
    uncached = [pid for pid in pids if str(pid) not in paper_cache]
    if uncached:
        print(f"ERROR: no cached fingerprint for paper(s) {', '.join(map(str, uncached))}; run `make` first",
              file=sys.stderr)
        return 1

    with open(args.data, encoding="utf-8") as f:
        all_papers = json.load(f)
    coauthor_index = coauthor_coi.build_index(list(reviewers_by_email.values()),
                                              coauthor_coi.load_coauthors(args.coauthor_cache), years=args.coauthor_years)
    derived = coauthor_coi.derive_conflicts(targets, coauthor_index, coauthor_coi.load_author_names(args.author_names_cache),
                                            all_papers, use_identity=True)
    collab_hard = collaborator_coi.hard_conflicts(collaborator_coi.derive_conflicts(
        targets, collaborator_coi.build_index(list(reviewers_by_email.values()), index), index))
    scores = ar.build_pair_scores(
        targets, paper_cache, candidates, matrix, reviewers_by_email,
        {pid: set(derived.get(pid, ())) | collab_hard.get(pid, set()) for pid in pids},
        area_gate=True, light_cap=0, full_cap=0,
    )

    # Each paper's live R1 slate, submitted or not, in roster addresses where
    # the roster knows them; it is what the caps are counted against.
    slate: dict[int, list[str]] = defaultdict(list)
    submitted = hotcrp_log.submitted_review_ids(rows)
    done_on: dict[int, int] = Counter()
    for rid, review in live.items():
        if review.round == REVIEW_ROUND and review.pid in pids:
            slate[review.pid].append(to_roster.get(review.email, review.email))
            done_on[review.pid] += rid in submitted
    on_paper = {pid: {to_roster.get(e, e) for e in emails} for pid, emails in ever_on_paper(rows).items()}

    seniority = load_seniority(args.seniority)
    seniority.update({e: v for e, v in load_seniority(args.reserve_seniority).items() if e not in seniority})
    klass = {e: (seniority.get(e) or {}).get("class", "") or "unknown" for pid in pids for e in slate[pid]}
    klass.update({e: (seniority.get(e) or {}).get("class", "") or "unknown" for e in candidates})

    layers = affiliation_country.load_layers(args.affiliation_countries, pcinfo_index=index)
    slate_known = sorted({e for pid in pids for e in slate[pid] if e in reviewers_by_email} - set(candidates))
    countries, reviewer_country, _, thin = ar.build_country_caps(
        targets, candidates + slate_known, reviewers_by_email, args.same_country_cap, layers,
        majority=args.region_majority, min_resolved=args.region_min_resolved,
    )
    country_cap = {pid: (c.code, c.members, c.papers[pid]) for c in countries for pid in c.papers}
    if args.no_same_country_cap:
        country_cap = {}

    shortlists: dict[int, list[dict]] = {}
    for pid in pids:
        juniors = sum(1 for e in slate[pid] if klass.get(e) == "junior")
        out_of_area = sum(1 for e in slate[pid] if klass.get(e) == "out-of-area")
        seniors = sum(1 for e in slate[pid] if klass.get(e) == "senior")
        code, members, cap = country_cap.get(pid, ("", frozenset(), None))
        same_country = sum(1 for e in slate[pid] if e in members)
        gated = {e for e, _ in scores.eligible[pid]}
        ranked = sorted(scores.released[pid], key=lambda es: (es[0] not in gated, -es[1], es[0]))
        skipped: Counter = Counter()
        chosen = []
        for email, affinity in ranked:
            if email in on_paper.get(pid, set()):
                skipped["already on or removed from the paper"] += 1
                continue
            cls = klass.get(email, "unknown")
            if cls == "junior" and juniors >= args.max_juniors:
                skipped["junior cap"] += 1
                continue
            if cls == "out-of-area" and out_of_area >= args.max_out_of_area:
                skipped["out-of-area cap"] += 1
                continue
            if email in members and same_country >= cap:
                skipped[f"same-country cap ({code})"] += 1
                continue
            if len(chosen) < args.shortlist:
                notes = "first senior on the paper" if cls == "senior" and seniors == 0 else ""
                chosen.append({"email": email, "affinity": affinity, "class": cls,
                               "area": "in area" if email in gated else "released", "notes": notes})
        shortlists[pid] = chosen
        reasons = "; ".join(f"{n} {why}" for why, n in sorted(skipped.items()))
        print(f"#{pid}: {len(scores.released[pid])} COI-clear finished reviewers, {len(gated)} in area"
              + (f"; skipped {reasons}" if reasons else ""), file=sys.stderr)
    if thin:
        print(f"Note: paper(s) {', '.join(map(str, sorted(thin)))} have too few placed authors for a country cap.",
              file=sys.stderr)

    pick = suggest({pid: [(c["email"], c["affinity"]) for c in shortlists[pid]] for pid in pids})

    out_rows = []
    for pid in pids:
        paper = papers[pid]
        title = paper.get("title") or ""
        mix = Counter(klass.get(e, "unknown") for e in slate[pid])
        print(f"\n#{pid} {title}")
        print(f"  R1 slate: {done_on[pid]} submitted of {len(slate[pid])} assigned; "
              + ", ".join(f"{n} {c}" for c, n in sorted(mix.items()))
              + (f"; country cap {country_cap[pid][0]}={country_cap[pid][2]}" if pid in country_cap else ""))
        if not shortlists[pid]:
            print("  (no eligible candidate)")
        for rank, c in enumerate(shortlists[pid], 1):
            r = reviewers_by_email[c["email"]]
            n, when = finished[c["email"]]
            star = "*" if pick.get(pid) == c["email"] else " "
            print(f"  {star}{rank:2d}. {c['affinity']:.4f}  {r.name} <{r.hotcrp_email}>  {r.tier}, {c['class']}, "
                  f"{c['area']}, {reviewer_country.get(c['email']) or '??'}, {n} R1 done by {when:%a %b %d %H:%M}"
                  + (f"  [{c['notes']}]" if c["notes"] else ""))
            out_rows.append([pid, rank, "yes" if star == "*" else "", r.name, c["email"], r.hotcrp_email, r.tier,
                             c["class"], reviewer_country.get(c["email"]) or "", c["area"], f"{c['affinity']:.6f}",
                             n, when.isoformat(), c["notes"]])
    print("\n* = suggested first ask (one distinct reviewer per paper, greatest total affinity).")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(OUT_FIELDS)
        writer.writerows(out_rows)
    os.replace(tmp, args.out)
    print(f"Wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
