"""Canonical project paths used by command-line defaults.

All paths are anchored to the repository rather than the caller's working
directory.  Every command still accepts explicit path arguments when a
different dataset or destination is needed.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "inputs"
CURATED_DIR = DATA_DIR / "curated"
CACHE_DIR = DATA_DIR / "cache"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
ASSIGNMENT_DIR = OUTPUT_DIR / "assignments"
REPORT_DIR = OUTPUT_DIR / "reports"
EVALUATION_DIR = OUTPUT_DIR / "evaluations"


def input_path(name: str) -> str:
    return str(INPUT_DIR / name)


def curated_path(name: str) -> str:
    return str(CURATED_DIR / name)


def cache_path(name: str) -> str:
    return str(CACHE_DIR / name)


def assignment_path(name: str) -> str:
    return str(ASSIGNMENT_DIR / name)


def report_path(name: str) -> str:
    return str(REPORT_DIR / name)


def evaluation_path(name: str) -> str:
    return str(EVALUATION_DIR / name)
