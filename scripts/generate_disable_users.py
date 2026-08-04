"""Generate a HotCRP bulk-update CSV that disables everyone except the chairs.

Run from anywhere with:

    python -m scripts.generate_disable_users

By default this reads ``data/inputs/hpca2027-pcinfo.csv`` and writes
``outputs/assignments/hpca2027-disable-users.csv``.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data/inputs/hpca2027-pcinfo.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/assignments/hpca2027-disable-users.csv"


def role_tokens(value: str) -> set[str]:
    return {token.casefold() for token in re.split(r"[\s,;]+", value.strip()) if token}


def generate(input_path: Path, output_path: Path) -> tuple[int, list[str]]:
    with input_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {"email", "roles"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing required column(s): {', '.join(sorted(missing))}")
        rows = list(reader)

    chairs = [
        (row.get("email") or "").strip()
        for row in rows
        if "chair" in role_tokens(row.get("roles") or "")
    ]
    if len(chairs) != 2 or any(not email for email in chairs):
        raise ValueError(
            f"expected exactly two chair accounts with email addresses; found {chairs!r}"
        )

    emails: list[str] = []
    seen: set[str] = set()
    chair_keys = {email.casefold() for email in chairs}
    for row_number, row in enumerate(rows, start=2):
        email = (row.get("email") or "").strip()
        if not email:
            raise ValueError(f"row {row_number} has no email address")
        key = email.casefold()
        if key in chair_keys:
            continue
        if key in seen:
            raise ValueError(f"duplicate email address on row {row_number}: {email}")
        seen.add(key)
        emails.append(email)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination, lineterminator="\n")
        writer.writerow(["email", "disabled"])
        writer.writerows((email, "yes") for email in emails)

    return len(emails), chairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    count, chairs = generate(args.input, args.output)
    print(f"Wrote {count} accounts to {args.output}")
    print(f"Excluded chairs: {', '.join(chairs)}")


if __name__ == "__main__":
    main()
