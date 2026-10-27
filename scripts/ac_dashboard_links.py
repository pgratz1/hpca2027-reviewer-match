"""Dropbox links to each track's AC dashboard directory, one row per area chair.

    # After `make ac-dashboard`, once Dropbox reports the files synced:
    python -m scripts.ac_dashboard_links

    # Another dashboard directory, or a Dropbox CLI elsewhere:
    python -m scripts.ac_dashboard_links --out-dir /tmp/x --dropbox ~/bin/dropbox

`make ac-dashboard` writes each track's PDF into its own directory,
`outputs/reports/ac_dashboard/track_NN/`, and the repository lives inside
Dropbox. This asks the local Dropbox daemon (`dropbox sharelink`) for the
shared link to every `track_NN/` directory -- created on first ask, the same
link on every later one -- and writes `outputs/reports/ac_dashboard_links.csv`,
`track,chair,email,folder,pdf,link`, one row per chair of the track, for the
PC chair to send by hand. Nothing is sent. The link is what the account's
default link settings make it, normally anyone-with-the-link, view-only: a
page carries paper titles, IDs and scores, never reviewer names, addresses or
review text.

A directory is linked only once Dropbox reports it `up to date` and only when
it holds exactly one PDF; anything else is an error, never a half-synced link.
`conflict_chair/` is linked for the conflict chair when it exists; `all/` and
`all_<name>/` are the PC chairs' own pages and are never linked. The links
survive every regeneration, since `ac_dashboard.py` only refills a directory;
deleting the output directory breaks them all. Chairs come from the user
export's `~~track_NN` tags (`ac_dashboard.track_chairs`), so a track
reassigned there is reassigned here on the next run. The CSV holds addresses
and live links and is gitignored by name. stdout is one line per row,
`track chair link`; everything else is stderr.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile

from reviewer_match import pc_membership
from reviewer_match.paths import report_path
from scripts import ac_dashboard

DEFAULT_OUT_DIR = ac_dashboard.DEFAULT_OUT_DIR
DEFAULT_PCINFO = pc_membership.DEFAULT_PCINFO
DEFAULT_OUT = report_path("ac_dashboard_links.csv")
DEFAULT_DROPBOX = "dropbox"
TRACK_DIR_RE = re.compile(r"^track_(\d+)$")
CONFLICT_DIR = ac_dashboard.page_dir(ac_dashboard.CONFLICT_PAGE)
LINK_RE = re.compile(r"^https://www\.dropbox\.com/\S+$")
SYNCED = "up to date"
COLUMNS = ["track", "chair", "email", "folder", "pdf", "link"]


def dropbox(binary: str, *args: str) -> str:
    result = subprocess.run([binary, *args], capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise SystemExit(f"ERROR: dropbox {' '.join(args)} failed: {(result.stderr or result.stdout).strip()}")
    return result.stdout.strip()


def share_link(binary: str, path: str) -> str:
    """The shared link for `path`, once Dropbox has synced it."""
    status = dropbox(binary, "filestatus", path)
    if not status.endswith(": " + SYNCED):
        raise SystemExit(f"ERROR: Dropbox has not synced {path} ({status!r}); "
                         "wait for `dropbox status` to read 'Up to date' and rerun")
    link = dropbox(binary, "sharelink", path)
    if not LINK_RE.match(link):
        raise SystemExit(f"ERROR: dropbox sharelink {path} answered {link!r}, not a link")
    return link


def only_pdf(path: str) -> str:
    pdfs = sorted(n for n in os.listdir(path) if n.endswith(".pdf"))
    if len(pdfs) != 1:
        raise SystemExit(f"ERROR: {path} holds {len(pdfs)} PDF(s), not 1; rerun `make ac-dashboard`")
    return pdfs[0]


def build_rows(out_dir: str, pcinfo: str, binary: str) -> list[dict[str, str]]:
    chairs = ac_dashboard.track_chairs(pcinfo)
    readers: dict[str, list[tuple[str, str]]] = {}
    tracks: dict[str, str] = {}
    for name in sorted(os.listdir(out_dir)):
        m = TRACK_DIR_RE.match(name)
        if m and os.path.isdir(os.path.join(out_dir, name)):
            n = int(m.group(1))
            tracks[name] = str(n)
            readers[name] = [(ac_dashboard.clean_name(c.name), c.email) for c in chairs.get(n, [])]
            if not readers[name]:
                print(f"WARNING: track {n} has no chair account tagged ~~track_{n}; its row names nobody",
                      file=sys.stderr)
    if os.path.isdir(os.path.join(out_dir, CONFLICT_DIR)):
        index = pc_membership.load_pc_accounts(pcinfo).by_email
        tracks[CONFLICT_DIR] = "conflict"
        readers[CONFLICT_DIR] = [(ac_dashboard.clean_name(index[e].name), e)
                                 for e in ac_dashboard.conflict_chairs(pcinfo)]
    for n in sorted(set(chairs) - {int(t) for t in tracks.values() if t.isdigit()}):
        print(f"WARNING: track {n} has a chair but no directory under {out_dir}", file=sys.stderr)
    if not tracks:
        raise SystemExit(f"ERROR: no track directories under {out_dir}; run `make ac-dashboard` first")
    rows = []
    for folder in tracks:
        path = os.path.abspath(os.path.join(out_dir, folder))
        pdf = only_pdf(path)
        link = share_link(binary, path)
        for chair, email in readers[folder] or [("", "")]:
            rows.append({"track": tracks[folder], "chair": chair, "email": email,
                         "folder": folder, "pdf": pdf, "link": link})
    return rows


def write_csv(path: str, rows: list[dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help=f"the dashboard directory (default {DEFAULT_OUT_DIR})")
    p.add_argument("--pcinfo", default=DEFAULT_PCINFO, help="HotCRP user export (track chairs)")
    p.add_argument("--out", default=DEFAULT_OUT, help=f"the link table (default {DEFAULT_OUT})")
    p.add_argument("--dropbox", default=DEFAULT_DROPBOX, help=f"Dropbox CLI binary (default {DEFAULT_DROPBOX})")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    binary = shutil.which(args.dropbox)
    if not binary:
        raise SystemExit(f"ERROR: {args.dropbox!r} not found; install the Dropbox client or pass --dropbox")
    rows = build_rows(args.out_dir, args.pcinfo, binary)
    write_csv(args.out, rows)
    for r in rows:
        print(f"{r['track']}\t{r['chair']}\t{r['link']}")
    print(f"Wrote {len(rows)} row(s), {len({r['folder'] for r in rows})} link(s), to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
