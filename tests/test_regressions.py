import argparse
import contextlib
import csv
import io
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
import zipfile
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np
import requests

from reviewer_match import affiliation_country
from reviewer_match import coauthor_coi
from reviewer_match import collaborator_coi
from scripts import assign_reviewers
from scripts import assign_area_chairs
from reviewer_match import area_chairs
from scripts import build_affiliation_countries
from scripts import build_fingerprints
from scripts import build_dblp_snapshot_cache
from scripts import build_reserve_reviewer_info
from reviewer_match import reserve_reviewers
from scripts import resolve_reserve_pids
from scripts import classify_reviewers
from scripts import compare_abstract_rankings
from reviewer_match import dblp
from scripts import enrich_publications
from scripts import estimate_reserve_need
from reviewer_match import paper_matching
from reviewer_match import fingerprint
from reviewer_match import paths
from scripts import resolve_trc_members
from reviewer_match import trc_reviewers
from reviewer_match.trc_reviewers import TrcMember
from scripts import assign_trc_reviews
from scripts import score_abstract_evaluation
from scripts import audit_coauthor_conflicts
from scripts import audit_pc_roster
from scripts import find_duplicate_accounts
from scripts import generate_clear_uploads
from reviewer_match import pc_membership
from reviewer_match import reviewers as reviewers_mod
from reviewer_match import roster as roster_mod
from reviewer_match.reviewers import Reviewer, _parse_override_cap
from reviewer_match import assignment_io
from reviewer_match.assignment_io import Pair
from scripts import diff_assignments
from scripts import fill_open_slots

PCINFO_FIELDS = [
    "given_name", "family_name", "email", "affiliation", "orcid", "country",
    "disabled", "roles", "tags", "collaborators",
    "topic: Memory Systems", "topic: Microarchitecture",
]


class RepositoryPathTests(unittest.TestCase):
    def test_project_root_is_derived_from_the_package(self):
        expected = Path(__file__).resolve().parents[1]
        self.assertEqual(expected, paths.PROJECT_ROOT)

    def test_default_artifacts_are_separated_by_role(self):
        self.assertEqual(paths.INPUT_DIR, Path(assign_reviewers.DEFAULT_DATA).parent)
        self.assertEqual(paths.CURATED_DIR, Path(reviewers_mod.DEFAULT_OVERRIDES).parent)
        self.assertEqual(paths.CACHE_DIR, Path(assign_reviewers.DEFAULT_FINGERPRINT_CACHE).parent)
        self.assertEqual(paths.REPORT_DIR, Path(classify_reviewers.DEFAULT_OUT).parent)


class HotcrpAssignmentCsvTests(unittest.TestCase):
    def read_rows(self, path):
        with Path(path).open(newline="", encoding="utf-8") as f:
            return list(csv.reader(f))

    def test_csv_clears_r1_then_assigns_primary_reviews_deterministically(self):
        slates = {2: ["low@example.com", "high@example.com"], 1: ["one@example.com"]}
        scores = {
            ("low@example.com", 2): 0.5,
            ("high@example.com", 2): 0.9,
            ("one@example.com", 1): 0.7,
        }
        reviewers_by_email = {
            email: reviewer_with_email(email)
            for email in ("low@example.com", "high@example.com", "one@example.com")
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "assignment.csv"
            assign_reviewers.write_hotcrp_csv(str(path), slates, scores, reviewers_by_email)
            rows = self.read_rows(path)
        self.assertEqual(
            [
                ["paper", "action", "email", "round"],
                ["all", "clearreview", "all", "R1"],
                ["1", "primaryreview", "one@example.com", "R1"],
                ["2", "primaryreview", "high@example.com", "R1"],
                ["2", "primaryreview", "low@example.com", "R1"],
            ],
            rows,
        )

    def test_csv_writes_the_hotcrp_email_not_the_roster_email(self):
        # A reviewer pc_membership matched under a second address: the roster
        # key is their form email, but the upload must use the address HotCRP
        # actually has marked `pc`, or it is rejected on upload.
        slates = {1: ["form@example.com"]}
        scores = {("form@example.com", 1): 0.8}
        reviewers_by_email = {
            "form@example.com": reviewer_with_email("form@example.com", hotcrp_email="real@other.edu"),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "assignment.csv"
            assign_reviewers.write_hotcrp_csv(str(path), slates, scores, reviewers_by_email)
            rows = self.read_rows(path)
        self.assertEqual(
            [
                ["paper", "action", "email", "round"],
                ["all", "clearreview", "all", "R1"],
                ["1", "primaryreview", "real@other.edu", "R1"],
            ],
            rows,
        )

    def test_empty_slates_still_clear_existing_r1_assignments(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "assignment.csv"
            assign_reviewers.write_hotcrp_csv(str(path), {1: []}, {}, {})
            rows = self.read_rows(path)
        self.assertEqual(
            [["paper", "action", "email", "round"],
             ["all", "clearreview", "all", "R1"]],
            rows,
        )

    def test_failed_atomic_replace_preserves_the_previous_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "assignment.csv"
            path.write_text("previous\n", encoding="utf-8")
            with mock.patch.object(
                assign_reviewers.os, "replace", side_effect=OSError("replace failed")
            ):
                with self.assertRaises(OSError):
                    assign_reviewers.write_hotcrp_csv(str(path), {1: []}, {}, {})
            self.assertEqual("previous\n", path.read_text(encoding="utf-8"))
            self.assertEqual([], list(Path(tmp).glob(".assignment.csv.*.tmp")))


class AreaChairTrackTagCsvTests(unittest.TestCase):
    def read_rows(self, path):
        with Path(path).open(newline="", encoding="utf-8") as f:
            return list(csv.reader(f))

    def test_paper_and_account_track_tags_are_zero_indexed_and_spelled_alike(self):
        # The paper tag and the account grant are the same `~~track_N`
        # spelling in both HotCRP namespaces -- only the object it tags
        # (paper vs. account) differs.
        self.assertEqual("~~track_0", assign_area_chairs.track_tag(0))
        self.assertEqual("~~track_23", assign_area_chairs.track_tag(23))

    def test_account_tag_csv_is_sorted_by_email_and_uses_safe_add_remove_columns(self):
        track_number = {"b@example.com": 1, "a@example.com": 0}
        chairs_by_email = {
            "a@example.com": chair_with_email("a@example.com"),
            "b@example.com": chair_with_email("b@example.com"),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "area_chair_account_tags.csv"
            assign_area_chairs.write_account_tag_csv(
                str(path), chairs_by_email, track_number, clear_through=3
            )
            rows = self.read_rows(path)
        self.assertEqual(
            [
                ["email", "remove_tags", "add_tags"],
                ["a@example.com", "~~track_0 ~~track_1 ~~track_2", "~~track_0"],
                ["b@example.com", "~~track_0 ~~track_1 ~~track_2", "~~track_1"],
            ],
            rows,
        )

    def test_account_tag_csv_writes_the_hotcrp_email_not_the_roster_email(self):
        # Same reasoning as the reviewer CSV: a chair matched under a second
        # address must be tagged on the account HotCRP actually has marked
        # `pc`, not the one on their acceptance-form row.
        track_number = {"form@example.com": 0}
        chairs_by_email = {
            "form@example.com": chair_with_email("form@example.com", hotcrp_email="real@other.edu"),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "area_chair_account_tags.csv"
            assign_area_chairs.write_account_tag_csv(
                str(path), chairs_by_email, track_number, clear_through=1
            )
            rows = self.read_rows(path)
        self.assertEqual(
            [
                ["email", "remove_tags", "add_tags"],
                ["real@other.edu", "~~track_0", "~~track_0"],
            ],
            rows,
        )

    def test_paper_tag_csv_groups_by_chair_then_sorts_pids(self):
        by_chair = {"b@example.com": [20, 5], "a@example.com": [1]}
        track_number = {"a@example.com": 0, "b@example.com": 1}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "area_chair_paper_tags.csv"
            assign_area_chairs.write_paper_tag_csv(str(path), by_chair, track_number, clear_through=0)
            rows = self.read_rows(path)
        self.assertEqual(
            [
                ["paper", "action", "email", "tag", "round"],
                ["1", "tag", "", "~~track_0", ""],
                ["5", "tag", "", "~~track_1", ""],
                ["20", "tag", "", "~~track_1", ""],
            ],
            rows,
        )

    def test_paper_tag_csv_clears_stale_tracks_before_reapplying(self):
        by_chair = {"a@example.com": [1]}
        track_number = {"a@example.com": 0}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "area_chair_paper_tags.csv"
            assign_area_chairs.write_paper_tag_csv(str(path), by_chair, track_number, clear_through=3)
            rows = self.read_rows(path)
        self.assertEqual(
            [
                ["paper", "action", "email", "tag", "round"],
                ["all", "cleartag", "", "~~track_0", ""],
                ["all", "cleartag", "", "~~track_1", ""],
                ["all", "cleartag", "", "~~track_2", ""],
                ["1", "tag", "", "~~track_0", ""],
            ],
            rows,
        )

    def test_failed_atomic_replace_preserves_the_previous_account_tag_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "area_chair_account_tags.csv"
            path.write_text("previous\n", encoding="utf-8")
            with mock.patch.object(
                assign_area_chairs.os, "replace", side_effect=OSError("replace failed")
            ):
                with self.assertRaises(OSError):
                    assign_area_chairs.write_account_tag_csv(str(path), {}, {})
            self.assertEqual("previous\n", path.read_text(encoding="utf-8"))
            self.assertEqual([], list(Path(tmp).glob(".area_chair_account_tags.csv.*.tmp")))

    def test_failed_atomic_replace_preserves_the_previous_paper_tag_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "area_chair_paper_tags.csv"
            path.write_text("previous\n", encoding="utf-8")
            with mock.patch.object(
                assign_area_chairs.os, "replace", side_effect=OSError("replace failed")
            ):
                with self.assertRaises(OSError):
                    assign_area_chairs.write_paper_tag_csv(str(path), {}, {})
            self.assertEqual("previous\n", path.read_text(encoding="utf-8"))
            self.assertEqual([], list(Path(tmp).glob(".area_chair_paper_tags.csv.*.tmp")))


def write_pcinfo(path, accounts):
    """Write a HotCRP user-export fixture.

    `accounts` is a sequence of dicts with any subset of PCINFO_FIELDS; the
    common case is {"email": ..., "given_name": ..., "family_name": ...,
    "roles": "pc"}. Returns the path, so a caller can inline it.
    """
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PCINFO_FIELDS)
        writer.writeheader()
        for account in accounts:
            writer.writerow({field: account.get(field, "") for field in PCINFO_FIELDS})
    return path


def pc_account(email, given="Test", family="Person", roles="pc", **extra):
    """One PC-marked row of a user-export fixture."""
    return {"email": email, "given_name": given, "family_name": family,
            "roles": roles, **extra}


def reviewer(pid="1/Test"):
    return Reviewer(
        email="person@example.com", first="Test", last="Person", dblp_url="",
        pid=pid, affiliation="Example", primary="Memory", secondary="", tertiary="",
        keywords="caches", tier="full", override_cap=None,
    )


def reviewer_with_email(email, hotcrp_email=""):
    """A minimal Reviewer for CSV-writer tests that only look at the email fields."""
    return Reviewer(
        email=email, hotcrp_email=hotcrp_email, first="Test", last="Person", dblp_url="",
        pid=None, affiliation="Example", primary="Memory", secondary="", tertiary="",
        keywords="", tier="full", override_cap=None,
    )


def chair_with_email(email, hotcrp_email=""):
    """A minimal AreaChair for CSV-writer tests that only look at the email fields."""
    return area_chairs.AreaChair(
        email=email, hotcrp_email=hotcrp_email, first="Test", last="Chair", dblp_url="",
        pid=None, affiliation="Example", primary="Memory", secondary="", tertiary="", keywords="",
    )


class FakeTokenizer:
    sep_token = "[SEP]"


class PublicationEnrichmentTests(unittest.TestCase):
    def test_doi_publisher_and_abstract_cleanup(self):
        self.assertEqual("ieee", enrich_publications.publisher_for_doi("https://doi.org/10.1109/ABC.12"))
        self.assertEqual("acm", enrich_publications.publisher_for_doi("10.1145/123.456"))
        self.assertIsNone(enrich_publications.publisher_for_doi("10.1000/example"))
        self.assertEqual(
            "A cache & memory study.",
            enrich_publications.clean_abstract("<jats:p>A cache &amp; memory study.</jats:p>"),
        )

    def test_http_error_summary_does_not_expose_url_or_key(self):
        response = mock.Mock(status_code=403)
        response.request.method = "GET"
        error = requests.HTTPError(
            "403 for https://example.test/?apikey=top-secret", response=response
        )
        summary = enrich_publications.safe_request_error(error)
        self.assertEqual("HTTP 403 from GET API request", summary)
        self.assertNotIn("top-secret", summary)

    def test_s2_enriches_ieee_and_acm_dois(self):
        cache = {}
        publishers = {"10.1109/a": "ieee", "10.1145/b": "acm"}
        s2 = {
            "10.1109/a": {"status": "found", "abstract": "IEEE abstract", "source": "semantic_scholar"},
            "10.1145/b": {"status": "found", "abstract": "ACM abstract", "source": "semantic_scholar"},
        }
        session = object()
        with mock.patch.object(enrich_publications, "fetch_s2_abstracts", return_value=s2) as fetch_s2:
            found, attempted = enrich_publications.enrich_abstract_cache(
                publishers, cache, s2_key="secret", session=session
            )
        self.assertEqual((2, 2), (found, attempted))
        self.assertEqual("IEEE abstract", cache["10.1109/a"]["abstract"])
        self.assertEqual("semantic_scholar", cache["10.1109/a"]["source"])
        self.assertEqual("semantic_scholar", cache["10.1145/b"]["source"])
        fetch_s2.assert_called_once_with(
            ["10.1109/a", "10.1145/b"], "secret", session
        )

    def test_missing_s2_key_uses_unauthenticated_api(self):
        cache = {"10.1109/a": {"status": "pending_fallback"}}
        publishers = {"10.1109/a": "ieee", "10.1145/b": "acm"}
        s2 = {
            "10.1109/a": {"status": "found", "abstract": "IEEE abstract", "source": "semantic_scholar"},
            "10.1145/b": {"status": "found", "abstract": "ACM abstract", "source": "semantic_scholar"},
        }
        session = object()
        with mock.patch.object(enrich_publications, "fetch_s2_abstracts", return_value=s2) as fetch_s2:
            found, attempted = enrich_publications.enrich_abstract_cache(
                publishers, cache, s2_key="", session=session
            )
        self.assertEqual((2, 2), (found, attempted))
        self.assertEqual("IEEE abstract", cache["10.1109/a"]["abstract"])
        self.assertEqual("ACM abstract", cache["10.1145/b"]["abstract"])
        fetch_s2.assert_called_once_with(
            ["10.1109/a", "10.1145/b"], "", session
        )

    def test_publication_document_uses_native_title_abstract_shape(self):
        self.assertEqual(
            "Title[SEP]Abstract",
            fingerprint.publication_doc_text(FakeTokenizer(), "Title", "Abstract"),
        )

    def test_evaluation_sample_round_robins_topics(self):
        rows = [
            {"paper": {"pid": 1, "topics": ["Memory"]}, "disagreement": 0.9},
            {"paper": {"pid": 2, "topics": ["Memory"]}, "disagreement": 0.8},
            {"paper": {"pid": 3, "topics": ["Security"]}, "disagreement": 0.1},
        ]
        selected = compare_abstract_rankings.choose_stratified(rows, 2)
        self.assertEqual({1, 3}, {row["paper"]["pid"] for row in selected})

    def test_dcg_rewards_higher_early_ratings(self):
        self.assertGreater(
            score_abstract_evaluation.dcg([3, 0, 0]),
            score_abstract_evaluation.dcg([0, 0, 3]),
        )


class AreaChairAssignmentTests(unittest.TestCase):
    def test_area_chair_loader_keeps_latest_explicit_acceptance(self):
        headers = [
            "Timestamp", "Please confirm your HotCRP email address",
            "Area Chair membership", "First Name", "Last Name",
            "Enter your DBLP Link", "institutional affiliation",
            "primary area", "keywords", "secondary area",
        ]
        rows = [
            ["07/01/2026 10:00:00", "chair@example.com", "No, I am unable to accept",
             "Old", "Name", "none", "Example", "Memory", "", "Security"],
            ["07/02/2026 10:00:00", "CHAIR@example.com",
             "Yes, I accept the role of being an Area Chair for HPCA 2027",
             "New", "Name", "https://dblp.org/pid/1/Test", "Example",
             "Microarchitecture", "branches", "Memory"],
            ["07/02/2026 11:00:00", "decline@example.com",
             "No, I would prefer to be a PC full or light member",
             "", "", "none", "", "", "", ""],
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chairs.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(rows)
            chairs = area_chairs.load_area_chairs(
                str(path), str(Path(tmp) / "missing.csv"), pcinfo_path=None
            )
        self.assertEqual(1, len(chairs))
        self.assertEqual("chair@example.com", chairs[0].email)
        self.assertEqual("New Name", chairs[0].name)
        self.assertEqual("1/Test", chairs[0].pid)
        self.assertEqual("", chairs[0].tertiary)

    def test_load_reviewer_assigned_pids(self):
        text = (
            "=== [2] Included\n"
            "    assigned 5 of 5 requested\n"
            "=== [3] Empty\n"
            "    assigned 0 of 5 requested\n"
            # The surplus stage appends to this line; the count must still parse.
            "=== [4] Boosted\n"
            "    assigned 6 of 5 requested  (+1 surplus)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "assignment.txt"
            path.write_text(text, encoding="utf-8")
            self.assertEqual([2, 4], assign_area_chairs.load_reviewer_assigned_pids(str(path)))

    def test_balanced_optimizer_finds_global_maximum(self):
        scores = {
            (1, "a"): 10.0, (1, "b"): 9.0,
            (2, "a"): 8.0, (2, "b"): 0.0,
        }
        result = assign_area_chairs.maximize_balanced_affinity(
            [1, 2], ["a", "b"], scores, 1, 1
        )
        self.assertEqual({1: "b", 2: "a"}, result)

    def test_balanced_optimizer_respects_conflicts_and_loads(self):
        scores = {
            (1, "b"): 5.0,
            (2, "a"): 4.0, (2, "b"): 1.0,
            (3, "a"): 3.0, (3, "b"): 2.0,
            (4, "a"): 1.0, (4, "b"): 4.0,
        }
        result = assign_area_chairs.maximize_balanced_affinity(
            [1, 2, 3, 4], ["a", "b"], scores, 2, 2
        )
        self.assertEqual("b", result[1])
        self.assertEqual({"a": 2, "b": 2}, dict(Counter(result.values())))

    def test_load_bounds_round_inward(self):
        self.assertEqual((30, 35), assign_area_chairs.load_bounds(486, 15, 0.10))

    def test_load_bounds_use_closest_integer_balance_when_tolerance_is_infeasible(self):
        self.assertEqual((3, 4), assign_area_chairs.load_bounds(56, 15, 0.10))

    def test_zero_tolerance_uses_closest_even_split(self):
        self.assertEqual((56, 57), assign_area_chairs.load_bounds(1192, 21, 0.0))
        self.assertEqual((4, 4), assign_area_chairs.load_bounds(60, 15, 0.0))

    def test_zero_load_chair_is_reported_without_crashing(self):
        chairs = [
            area_chairs.AreaChair(
                email=email, first=email[0].upper(), last="Chair", dblp_url="",
                pid=f"1/{email[0].upper()}", affiliation="Example", primary="Memory",
                secondary="", tertiary="", keywords="",
            )
            for email in ("a@example.com", "b@example.com")
        ]
        paper = {
            "pid": 1, "title": "A submitted paper", "abstract": "One. Two.",
            "topics": ["Memory"], "authors": [{"email": "author@example.com"}],
            "status": "submitted", "pc_conflicts": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            assignment = Path(tmp) / "assignment.txt"
            assignment.write_text(
                "=== [1] A submitted paper\n    assigned 1 of 1 requested\n",
                encoding="utf-8",
            )
            data = Path(tmp) / "papers.json"
            data.write_text(json.dumps([paper]), encoding="utf-8")
            chair_cache = Path(tmp) / "chairs.json"
            chair_cache.write_text(json.dumps({
                "a@example.com": {"vector": [1.0, 0.0]},
                "b@example.com": {"vector": [0.0, 1.0]},
            }), encoding="utf-8")
            paper_cache = Path(tmp) / "papers-cache.json"
            paper_cache.write_text(json.dumps({"1": {"vector": [1.0, 0.0]}}), encoding="utf-8")

            argv = [
                "assign_area_chairs.py", "--reviewer-assignment", str(assignment),
                "--data", str(data), "--paper-policy", "submitted",
                "--fingerprint-cache", str(chair_cache), "--paper-cache", str(paper_cache),
                "--no-coauthor-coi",
            ]
            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(assign_area_chairs, "load_roster", return_value=chairs),
                mock.patch.object(assign_area_chairs, "build_paper_fingerprints"),
                mock.patch.object(assign_area_chairs, "chair_conflicts", return_value={1: set()}),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                self.assertEqual(0, assign_area_chairs.main())

        self.assertIn("assigned 0 papers; mean affinity n/a", stdout.getvalue())
        self.assertIn("loads 0..1", stderr.getvalue())


class FingerprintCacheTests(unittest.TestCase):
    def test_publication_exclusions_are_normalized_and_person_specific(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "publication_exclusions.csv"
            path.write_text(
                "email,doi,note\n"
                "PERSON@EXAMPLE.COM,https://doi.org/10.1109/ABC.1,exclude\n"
                "person@example.com,10.1109/abc.1,duplicate\n",
                encoding="utf-8",
            )
            exclusions = build_fingerprints.load_publication_exclusions(str(path))
        self.assertEqual({"10.1109/abc.1"}, exclusions["person@example.com"])
        publications = [
            (2026, "Excluded", "10.1109/abc.1", "Abstract", "semantic_scholar"),
            (2025, "Retained", "10.1145/other", "", ""),
        ]
        filtered, matched = build_fingerprints.apply_publication_exclusions(
            "person@example.com", publications, exclusions
        )
        other_filtered, _ = build_fingerprints.apply_publication_exclusions(
            "other@example.com", publications, exclusions
        )
        self.assertEqual({"10.1109/abc.1"}, matched)
        self.assertEqual(["Retained"], [pub[1] for pub in filtered])
        self.assertEqual(publications, other_filtered)

    def test_malformed_publication_exclusion_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "publication_exclusions.csv"
            path.write_text("email,doi\nperson@example.com,not-a-doi\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid DOI"):
                build_fingerprints.load_publication_exclusions(str(path))

    def test_key_changes_with_inputs_and_policy(self):
        r = reviewer()
        base = build_fingerprints.fingerprint_key(
            r, [(2026, "A title")], years=4, max_titles=None, area_weight=1.0
        )
        changed_title = build_fingerprints.fingerprint_key(
            r, [(2026, "Another title")], years=4, max_titles=None, area_weight=1.0
        )
        changed_policy = build_fingerprints.fingerprint_key(
            r, [(2026, "A title")], years=3, max_titles=None, area_weight=1.0
        )
        self.assertNotEqual(base, changed_title)
        self.assertNotEqual(base, changed_policy)

    def test_key_changes_when_abstract_changes(self):
        r = reviewer()
        title_only = [(2026, "A title", "10.1109/a", "", "")]
        enriched = [(2026, "A title", "10.1109/a", "A useful abstract", "ieee")]
        self.assertNotEqual(
            build_fingerprints.fingerprint_key(
                r, title_only, years=4, max_titles=None, area_weight=1.0
            ),
            build_fingerprints.fingerprint_key(
                r, enriched, years=4, max_titles=None, area_weight=1.0
            ),
        )

    def test_failed_fetch_is_temporary_and_retried(self):
        r = reviewer()
        with tempfile.TemporaryDirectory() as td:
            cache = str(Path(td) / "fingerprints.json")

            def fail_fetch(pids, **kwargs):
                kwargs["on_error"](pids[0], RuntimeError("temporary"))
                return {}

            def succeed_fetch(pids, **kwargs):
                titles = [(2026, "Recovered publication")]
                kwargs["on_result"](pids[0], titles, "cache")
                return {pids[0]: (titles, "cache")}

            def encode(texts, tokenizer, model):
                return np.ones((len(texts), 768), dtype=np.float32)

            common = [
                mock.patch.object(build_fingerprints, "load_roster", return_value=[r]),
                mock.patch.object(build_fingerprints, "load_cache", return_value={}),
                mock.patch.object(build_fingerprints, "load_colleague_cache", return_value={}),
                mock.patch.object(build_fingerprints.specter2_model, "load_model", return_value=(FakeTokenizer(), object())),
                mock.patch.object(build_fingerprints.specter2_model, "encode_texts", side_effect=encode),
            ]
            with contextlib.ExitStack() as stack:
                for patcher in common:
                    stack.enter_context(patcher)
                stack.enter_context(mock.patch.object(build_fingerprints, "fetch_titles_for_pids", side_effect=fail_fetch))
                stack.enter_context(mock.patch.object(sys, "argv", ["build_fingerprints.py", "--fingerprint-cache", cache, "--device", "cpu"]))
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(build_fingerprints.main(), 0)

            first = json.loads(Path(cache).read_text())
            self.assertFalse(first[r.email]["dblp_fetch_complete"])
            self.assertEqual(first[r.email]["n_titles"], 0)

            with contextlib.ExitStack() as stack:
                for patcher in common:
                    stack.enter_context(patcher)
                stack.enter_context(mock.patch.object(build_fingerprints, "fetch_titles_for_pids", side_effect=succeed_fetch))
                stack.enter_context(mock.patch.object(sys, "argv", ["build_fingerprints.py", "--fingerprint-cache", cache, "--device", "cpu"]))
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(build_fingerprints.main(), 0)

            recovered = json.loads(Path(cache).read_text())
            self.assertTrue(recovered[r.email]["dblp_fetch_complete"])
            self.assertEqual(recovered[r.email]["n_titles"], 1)

            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(build_fingerprints, "load_roster", return_value=[r]))
                stack.enter_context(mock.patch.object(build_fingerprints, "load_cache", return_value={}))
                stack.enter_context(mock.patch.object(build_fingerprints, "load_colleague_cache", return_value={}))
                stack.enter_context(mock.patch.object(build_fingerprints, "fetch_titles_for_pids", side_effect=succeed_fetch))
                load_model = stack.enter_context(mock.patch.object(build_fingerprints.specter2_model, "load_model"))
                stack.enter_context(mock.patch.object(sys, "argv", ["build_fingerprints.py", "--fingerprint-cache", cache, "--device", "cpu"]))
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(build_fingerprints.main(), 0)
                load_model.assert_not_called()

    def test_confirmed_no_content_removes_a_stale_fingerprint(self):
        r = reviewer(pid=None)
        r.primary = r.secondary = r.tertiary = r.keywords = ""
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td) / "fingerprints.json"
            cache.write_text(json.dumps({
                r.email: {
                    "vector": [1.0] * 768,
                    "fingerprint_key": "stale",
                    "dblp_fetch_complete": True,
                }
            }), encoding="utf-8")
            with (
                mock.patch.object(build_fingerprints, "load_roster", return_value=[r]),
                mock.patch.object(build_fingerprints, "load_cache", return_value={}),
                mock.patch.object(build_fingerprints, "load_colleague_cache", return_value={}),
                mock.patch.object(
                    build_fingerprints.specter2_model, "load_model",
                    return_value=(FakeTokenizer(), object()),
                ),
                mock.patch.object(
                    build_fingerprints.specter2_model, "encode_texts",
                    return_value=np.empty((0, 768), dtype=np.float32),
                ),
                mock.patch.object(sys, "argv", [
                    "build_fingerprints.py", "--fingerprint-cache", str(cache),
                    "--device", "cpu",
                ]),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(0, build_fingerprints.main())
            self.assertNotIn(r.email, json.loads(cache.read_text(encoding="utf-8")))

    def test_transient_fetch_failure_preserves_last_known_good_fingerprint(self):
        r = reviewer()
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td) / "fingerprints.json"
            original = {
                "vector": [1.0] * 768,
                "fingerprint_key": "old",
                "dblp_fetch_complete": True,
            }
            cache.write_text(json.dumps({r.email: original}), encoding="utf-8")

            def fail_fetch(pids, **kwargs):
                kwargs["on_error"](pids[0], RuntimeError("temporary"))
                return {}

            with (
                mock.patch.object(build_fingerprints, "load_roster", return_value=[r]),
                mock.patch.object(build_fingerprints, "load_cache", return_value={}),
                mock.patch.object(build_fingerprints, "load_colleague_cache", return_value={}),
                mock.patch.object(build_fingerprints, "fetch_titles_for_pids", side_effect=fail_fetch),
                mock.patch.object(sys, "argv", [
                    "build_fingerprints.py", "--fingerprint-cache", str(cache),
                    "--device", "cpu",
                ]),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(0, build_fingerprints.main())
            retained = json.loads(cache.read_text(encoding="utf-8"))[r.email]
            self.assertEqual(original["vector"], retained["vector"])
            self.assertFalse(retained["dblp_fetch_complete"])


class PaperCacheTests(unittest.TestCase):
    def test_area_weight_participates_in_key(self):
        paper = {"pid": 1, "title": "A complete paper", "abstract": "Abstract", "topics": ["Memory"]}
        self.assertNotEqual(
            paper_matching._doc_key(paper, 1.0),
            paper_matching._doc_key(paper, 2.0),
        )


class PaperCompletenessTests(unittest.TestCase):
    COMPLETE = {
        "pid": 1, "title": "A complete paper", "abstract": "Abstract",
        "topics": ["Memory"], "authors": [{"email": "a@example.com"}],
        "status": "submitted",
    }

    def test_each_gap_is_detected(self):
        self.assertEqual([], paper_matching.completeness_gaps(self.COMPLETE))
        cases = {
            "title under 3 words": {"title": "Too short"},
            "no abstract": {"abstract": "  "},
            "no topics": {"topics": []},
            "no authors": {"authors": []},
            "withdrawn": {"withdrawn": True},
        }
        for expected, override in cases.items():
            gaps = paper_matching.completeness_gaps({**self.COMPLETE, **override})
            self.assertEqual([expected], gaps)

    REGISTERED = {
        "pid": 1, "title": "A registered paper", "topics": ["Memory"],
        "abstract": "We build a thing. It goes fast.",
        "authors": [{"email": "a@example.com"}], "status": "draft",
    }

    def test_each_registration_gap_is_detected(self):
        self.assertEqual([], paper_matching.registration_gaps(self.REGISTERED))
        cases = {
            "no title": {"title": "   "},
            "placeholder title": {"title": " Test "},
            "abstract under 2 sentences": {"abstract": "One sentence only."},
            "no authors": {"authors": []},
            "withdrawn": {"withdrawn": True},
        }
        for expected, override in cases.items():
            gaps = paper_matching.registration_gaps({**self.REGISTERED, **override})
            self.assertEqual([expected], gaps)

    def test_registration_keeps_short_titles_and_topicless_drafts(self):
        for override in ({"title": "Saturo"}, {"topics": []}, {"title": "Test TRC"}):
            with self.subTest(override=override):
                self.assertEqual(
                    [], paper_matching.registration_gaps({**self.REGISTERED, **override})
                )

    def test_registration_abstract_sentence_count_ignores_non_prose(self):
        for abstract in ("TBD", "1. 2. 3.", "Abstract goes here", ""):
            with self.subTest(abstract=abstract):
                self.assertEqual(
                    [f"abstract under {paper_matching.MIN_ABSTRACT_SENTENCES} sentences"],
                    paper_matching.registration_gaps({**self.REGISTERED, "abstract": abstract}),
                )

    def test_registered_policy_is_the_default(self):
        papers = [
            self.REGISTERED,
            {**self.REGISTERED, "pid": 2, "status": "withdrawn", "withdrawn": True},
            {**self.REGISTERED, "pid": 3, "title": "test", "abstract": "TBD"},
            {**self.REGISTERED, "pid": 4, "status": "submitted"},
        ]
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "papers.json")
            Path(path).write_text(json.dumps(papers))
            with contextlib.redirect_stderr(io.StringIO()):
                selected, skipped = paper_matching.load_papers(path, with_skipped=True)
        self.assertEqual([1, 4], [p["pid"] for p in selected])
        self.assertEqual(
            [
                (2, ["withdrawn"]),
                (3, ["placeholder title", "abstract under 2 sentences"]),
            ],
            [(s["pid"], s["missing"]) for s in skipped],
        )

    def test_submitted_policy_uses_status_only(self):
        papers = [
            self.COMPLETE,
            {
                "pid": 2, "title": "", "abstract": "", "topics": [], "authors": [],
                "status": "submitted", "withdrawn": True,
            },
            {**self.COMPLETE, "pid": 3, "status": "draft"},
            {**self.COMPLETE, "pid": 4, "status": "withdrawn", "withdrawn": True},
        ]
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "papers.json")
            Path(path).write_text(json.dumps(papers))
            with contextlib.redirect_stderr(io.StringIO()):
                selected, skipped = paper_matching.load_papers(
                    path, paper_policy="submitted", with_skipped=True
                )
        self.assertEqual([1, 2], [p["pid"] for p in selected])
        self.assertEqual(
            [(3, ["status draft"]), (4, ["status withdrawn"])],
            [(s["pid"], s["missing"]) for s in skipped],
        )

    def test_exclude_pids_overrides_every_policy(self):
        # A test paper like "Test TRC" can carry a real submitted status and
        # a title that isn't the bare placeholder "test", so no policy check
        # catches it on content alone -- exclude_pids is the explicit escape
        # hatch, checked before any policy-specific logic runs.
        papers = [self.COMPLETE, {**self.COMPLETE, "pid": 2, "title": "Test TRC"}]
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "papers.json")
            Path(path).write_text(json.dumps(papers))
            with contextlib.redirect_stderr(io.StringIO()):
                selected, skipped = paper_matching.load_papers(
                    path, paper_policy="submitted",
                    exclude_pids=paper_matching.parse_exclude_pids("2"),
                    with_skipped=True,
                )
        self.assertEqual([1], [p["pid"] for p in selected])
        self.assertEqual([(2, ["manually excluded"])],
                          [(s["pid"], s["missing"]) for s in skipped])

    def test_parse_exclude_pids_reads_a_comma_separated_list(self):
        self.assertEqual(frozenset(), paper_matching.parse_exclude_pids(""))
        self.assertEqual(frozenset(), paper_matching.parse_exclude_pids("  "))
        self.assertEqual({1152}, paper_matching.parse_exclude_pids("1152"))
        self.assertEqual({1152, 1200}, paper_matching.parse_exclude_pids("1152,1200"))

    def test_complete_policy_retains_legacy_checks(self):
        papers = [
            self.COMPLETE,
            {
                "pid": 2, "title": "Placeholder", "abstract": "", "topics": [],
                "authors": [], "status": "submitted",
            },
            {**self.COMPLETE, "pid": 3, "status": "draft", "withdrawn": True},
        ]
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "papers.json")
            Path(path).write_text(json.dumps(papers))
            with contextlib.redirect_stderr(io.StringIO()):
                selected, skipped = paper_matching.load_papers(
                    path, paper_policy="complete", with_skipped=True
                )
        self.assertEqual([1], [p["pid"] for p in selected])
        self.assertEqual(
            [
                (2, ["title under 3 words", "no abstract", "no topics", "no authors"]),
                (3, ["withdrawn"]),
            ],
            [(s["pid"], s["missing"]) for s in skipped],
        )


class ReportingAndValidationTests(unittest.TestCase):
    def test_topicless_shortage_is_reported_and_counted(self):
        papers = [{"pid": 1, "title": "A complete paper", "topics": []}]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            missing = assign_reviewers.shortage_report(
                papers, {1: []}, {1: 3}, {}, {}
            )
        self.assertEqual(missing, 3)
        self.assertIn("Unspecified/no matching topic", output.getvalue())
        self.assertIn("missing 3", output.getvalue())

    def test_negative_csv_override_cap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            _parse_override_cap("person@example.com", "-1")

    def test_nonpositive_coauthor_windows_are_rejected_before_io(self):
        commands = (
            assign_reviewers,
            assign_area_chairs,
            audit_coauthor_conflicts,
            build_dblp_snapshot_cache,
        )
        for command in commands:
            for value in ("0", "-1"):
                with self.subTest(command=command.__name__, value=value):
                    with (
                        mock.patch.object(sys, "argv", [
                            f"{command.__name__}.py", "--coauthor-years", value,
                        ]),
                        contextlib.redirect_stderr(io.StringIO()),
                        self.assertRaises(SystemExit) as raised,
                    ):
                        command.main()
                    self.assertEqual(2, raised.exception.code)

    def test_negative_reserve_cap_is_rejected_before_io(self):
        with (
            mock.patch.object(sys, "argv", ["assign_reviewers.py", "--reserve-cap", "-1"]),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            assign_reviewers.main()
        self.assertEqual(2, raised.exception.code)

    def test_zero_reserve_cap_is_valid(self):
        self.assertEqual(0, assign_reviewers.reviewer_paper_cap(
            type("Reserve", (), {"override_cap": None, "tier": "reserve"})(),
            light_cap=7, full_cap=15, reserve_cap=0,
        ))

    def test_fetch_failure_is_not_an_identity_stub_candidate(self):
        # A PCDB-promoted reviewer without a PID still needs identity work.
        rows = [
            {"email": "missing@example.com", "class": "unknown", "pid": ""},
            {"email": "failed@example.com", "class": "unknown", "pid": "1/Known"},
            {"email": "promoted@example.com", "class": "senior", "pid": ""},
        ]
        self.assertEqual(
            ["missing@example.com", "promoted@example.com"],
            [r["email"] for r in classify_reviewers.unresolved_identity_rows(rows)],
        )


class AssignmentPropertyTests(unittest.TestCase):
    def test_random_assignments_obey_caps_and_have_no_blocking_pairs(self):
        rng = random.Random(1)
        for _ in range(250):
            pids = list(range(rng.randint(1, 6)))
            emails = [f"r{i}" for i in range(rng.randint(1, 8))]
            juniors, out_of_area = set(), set()
            for e in emails:
                roll = rng.random()
                if roll < 0.4:
                    juniors.add(e)
                elif roll < 0.6:
                    out_of_area.add(e)
            capped = [(frozenset(juniors), 1), (frozenset(out_of_area), 1)]
            caps = {e: rng.randint(1, 3) for e in emails}
            targets = {pid: rng.randint(1, 4) for pid in pids}
            pairs, prefs, scores = {}, {}, {}
            for pid in pids:
                candidates = []
                for email in emails:
                    if rng.random() < 0.75:
                        score = rng.random()
                        candidates.append((email, score))
                        scores[email, pid] = score
                candidates.sort(key=lambda pair: -pair[1])
                pairs[pid] = candidates
                prefs[pid] = [email for email, _ in candidates]
            held = assign_reviewers.deferred_acceptance(
                pids, prefs, targets, caps, scores, capped
            )
            self.assertTrue(all(len(held[pid]) <= targets[pid] for pid in pids))
            for class_emails, class_cap in capped:
                self.assertTrue(
                    all(sum(e in class_emails for e in held[pid]) <= class_cap for pid in pids)
                )
            self.assertEqual(
                0,
                assign_reviewers.count_blocking_pairs(
                    pairs, held, caps, targets, scores, capped
                ),
            )

    def test_a_reviewer_in_two_classes_counts_against_both(self):
        # Classes used to be keyed one-per-email, last writer winning, so a
        # junior who is also region-affiliated would silently drop out of the
        # junior cap. Both caps have to see them.
        juniors = frozenset({"jr"})
        region = frozenset({"jr", "r2"})
        held = assign_reviewers.deferred_acceptance(
            [1], {1: ["jr", "r2"]}, {1: 2}, {"jr": 1, "r2": 1},
            {("jr", 1): 0.9, ("r2", 1): 0.8}, [(juniors, 1), (region, 1)],
        )
        self.assertEqual(["jr"], held[1])

    def test_a_multi_class_candidate_is_not_stalled_by_an_unrelated_full_class(self):
        # 'jrreg' is deferred because the junior cap is full. Keyed by class
        # rather than by membership signature, it would sit at the head of the
        # region queue blocking 'reg', which the region cap still has room for,
        # and the paper would under-fill for no reason.
        juniors = frozenset({"j1", "jrreg"})
        region = frozenset({"jrreg", "reg"})
        held = assign_reviewers.deferred_acceptance(
            [1], {1: ["j1", "jrreg", "reg"]}, {1: 3},
            {"j1": 1, "jrreg": 1, "reg": 1},
            {("j1", 1): 0.9, ("jrreg", 1): 0.8, ("reg", 1): 0.7},
            [(juniors, 1), (region, 2)],
        )
        self.assertEqual(["j1", "reg"], held[1])

    def test_a_per_paper_cap_binds_only_the_papers_it_names(self):
        # A region cap applies to majority-region papers alone; every other
        # paper must behave as though the class did not exist.
        region = frozenset({"a", "b"})
        held = assign_reviewers.deferred_acceptance(
            [1, 2], {1: ["a", "b"], 2: ["a", "b"]}, {1: 2, 2: 2}, {"a": 2, "b": 2},
            {("a", 1): 0.9, ("b", 1): 0.8, ("a", 2): 0.7, ("b", 2): 0.6},
            [(region, {1: 1})],
        )
        self.assertEqual(["a"], held[1])
        self.assertEqual(["a", "b"], sorted(held[2]))

    def test_random_assignments_with_crossing_classes_obey_every_cap(self):
        # The same sweep as the laminar one, but with a third class that cuts
        # across the first two. Stability is deliberately NOT asserted here --
        # see test_crossing_class_caps_can_leave_a_blocking_pair.
        rng = random.Random(7)
        for _ in range(250):
            pids = list(range(rng.randint(1, 6)))
            emails = [f"r{i}" for i in range(rng.randint(1, 8))]
            juniors, out_of_area, region = set(), set(), set()
            for e in emails:
                roll = rng.random()
                if roll < 0.4:
                    juniors.add(e)
                elif roll < 0.6:
                    out_of_area.add(e)
                if rng.random() < 0.5:
                    region.add(e)
            capped = [
                (frozenset(juniors), 1),
                (frozenset(out_of_area), 1),
                (frozenset(region), {pid: 2 for pid in pids if rng.random() < 0.7}),
            ]
            caps = {e: rng.randint(1, 3) for e in emails}
            targets = {pid: rng.randint(1, 4) for pid in pids}
            prefs, scores = {}, {}
            for pid in pids:
                candidates = []
                for email in emails:
                    if rng.random() < 0.75:
                        score = rng.random()
                        candidates.append((email, score))
                        scores[email, pid] = score
                candidates.sort(key=lambda pair: -pair[1])
                prefs[pid] = [email for email, _ in candidates]
            held = assign_reviewers.deferred_acceptance(
                pids, prefs, targets, caps, scores, capped
            )
            limits = assign_reviewers.resolve_caps(capped, pids)
            for pid in pids:
                self.assertLessEqual(len(held[pid]), targets[pid])
                # proposed at most once, and only to eligible candidates
                self.assertEqual(len(held[pid]), len(set(held[pid])))
                self.assertTrue(set(held[pid]) <= set(prefs[pid]))
                for k, (class_emails, _) in enumerate(capped):
                    self.assertLessEqual(
                        sum(e in class_emails for e in held[pid]), limits[k][pid]
                    )

    def test_laminar_class_families_have_no_blocking_pairs(self):
        # The guarantee that survives: with pairwise-disjoint or nested classes
        # the greedy choice is substitutable, so deferred acceptance is stable.
        rng = random.Random(11)
        for _ in range(250):
            pids = list(range(rng.randint(1, 6)))
            emails = [f"r{i}" for i in range(rng.randint(1, 8))]
            inner, outer = set(), set()
            for e in emails:
                roll = rng.random()
                if roll < 0.3:
                    inner.add(e)
                    outer.add(e)  # nested: inner is a subset of outer
                elif roll < 0.6:
                    outer.add(e)
            capped = [(frozenset(inner), 1), (frozenset(outer), 2)]
            caps = {e: rng.randint(1, 3) for e in emails}
            targets = {pid: rng.randint(1, 4) for pid in pids}
            pairs, prefs, scores = {}, {}, {}
            for pid in pids:
                candidates = []
                for email in emails:
                    if rng.random() < 0.75:
                        score = rng.random()
                        candidates.append((email, score))
                        scores[email, pid] = score
                candidates.sort(key=lambda pair: -pair[1])
                pairs[pid] = candidates
                prefs[pid] = [email for email, _ in candidates]
            held = assign_reviewers.deferred_acceptance(
                pids, prefs, targets, caps, scores, capped
            )
            self.assertEqual(0, assign_reviewers.count_blocking_pairs(
                pairs, held, caps, targets, scores, capped))

    def test_many_classes_cost_no_more_than_the_ones_that_bind(self):
        # One cap per country means ~30 classes and ~60 membership signatures,
        # but any one paper can only ever defer into the cells its own caps
        # block. Allocating a deque per signature per paper made every scan
        # proportional to the number of countries in the conference; the result
        # must be identical to running with only the binding class present.
        emails = [f"r{i}" for i in range(6)]
        scores = {(e, 1): 1.0 - i / 10 for i, e in enumerate(emails)}
        prefs = {1: emails}
        caps = {e: 1 for e in emails}
        binding = frozenset({"r0", "r1", "r2"})
        # 25 further classes that name this paper's reviewers but never bind it.
        idle = [(frozenset({e}), {99: 1}) for e in emails for _ in range(4)]
        lean = assign_reviewers.deferred_acceptance(
            [1], prefs, {1: 3}, caps, scores, [(binding, 1)])
        fat = assign_reviewers.deferred_acceptance(
            [1], prefs, {1: 3}, caps, scores, [(binding, 1), *idle])
        self.assertEqual(lean[1], fat[1])
        self.assertEqual(1, sum(e in binding for e in fat[1]))

    def test_crossing_class_caps_can_leave_a_blocking_pair(self):
        # Not a bug: {juniors, region} is a crossing family, and greedy-by-score
        # choice over one is not substitutable, so no stable outcome is
        # promised. Paper 1 anchors 'a', is bumped off it by paper 2, and by
        # then its region slots hold f and g -- both worse than the deferred e,
        # which it can no longer take without dropping one of them.
        pids = [1, 2]
        prefs = {1: ["a", "e", "f", "g"], 2: ["x1", "x2", "a"]}
        scores = {("a", 1): 1.0, ("e", 1): 0.9, ("f", 1): 0.5, ("g", 1): 0.4,
                  ("x1", 2): 3.0, ("x2", 2): 2.0, ("a", 2): 1.5}
        caps = {e: 1 for e in ["a", "e", "f", "g", "x1", "x2"]}
        capped = [(frozenset({"a", "e"}), 1), (frozenset({"e", "f", "g"}), 2)]
        held = assign_reviewers.deferred_acceptance(
            pids, prefs, {1: 3, 2: 3}, caps, scores, capped)
        self.assertEqual(["f", "g"], held[1])
        pairs = {p: [(e, scores[(e, p)]) for e in prefs[p]] for p in pids}
        self.assertEqual(1, assign_reviewers.count_blocking_pairs(
            pairs, held, caps, {1: 3, 2: 3}, scores, capped))
        # The caps themselves are still hard, which is what the rule needs.
        for class_emails, limit in capped:
            for pid in pids:
                self.assertLessEqual(sum(e in class_emails for e in held[pid]), limit)

    def test_papers_without_a_region_cap_stay_stable(self):
        # The crossing instance again, but the region limit names only paper 2.
        # Paper 1 sees a laminar family and keeps the guarantee.
        pids = [1, 2]
        prefs = {1: ["a", "e", "f", "g"], 2: ["x1", "x2", "a"]}
        scores = {("a", 1): 1.0, ("e", 1): 0.9, ("f", 1): 0.5, ("g", 1): 0.4,
                  ("x1", 2): 3.0, ("x2", 2): 2.0, ("a", 2): 1.5}
        caps = {e: 1 for e in ["a", "e", "f", "g", "x1", "x2"]}
        capped = [(frozenset({"a", "e"}), 1), (frozenset({"e", "f", "g"}), {2: 2})]
        held = assign_reviewers.deferred_acceptance(
            pids, prefs, {1: 3, 2: 3}, caps, scores, capped)
        pairs = {1: [(e, scores[(e, 1)]) for e in prefs[1]]}
        self.assertEqual(0, assign_reviewers.count_blocking_pairs(
            pairs, held, caps, {1: 3, 2: 3}, scores, capped))

    def test_blocking_pair_check_counts_frozen_phase_assignments(self):
        # An anchor phase can seat a region-affiliated senior, so F1 starts with
        # the region class already full even though nobody in this phase's
        # paper_held is in it. Without the seed the check invents a blocking
        # pair the matcher was right to refuse.
        region = frozenset({"r1", "r2"})
        args = ({1: [("r2", 0.9)]}, {1: []}, {"r2": 1}, {1: 1},
                {("r2", 1): 0.9}, [(region, 1)])
        self.assertEqual(1, assign_reviewers.count_blocking_pairs(*args))
        self.assertEqual(0, assign_reviewers.count_blocking_pairs(
            *args, held_counts={1: [1]}))

    def test_held_counts_seed_makes_caps_cumulative(self):
        # j1 (a junior) was frozen onto the paper by an earlier phase; with the
        # cap already consumed, this phase must not add the second junior.
        prefs = {1: ["j2"]}
        scores = {("j2", 1): 0.9}
        capped = [(frozenset({"j1", "j2"}), 1)]
        held = assign_reviewers.deferred_acceptance(
            [1], prefs, {1: 1}, {"j2": 1}, scores, capped, held_counts={1: [1]}
        )
        self.assertEqual([], held[1])
        held = assign_reviewers.deferred_acceptance(
            [1], prefs, {1: 1}, {"j2": 1}, scores, capped
        )
        self.assertEqual(["j2"], held[1])

    def test_under_filled_paper_fills_from_released_pool(self):
        # Paper 1 has one in-area candidate but wants two reviewers; the
        # area-released phase supplies the rest, best fingerprint first.
        gated_prefs = {1: ["in_area"]}
        released_prefs = {1: ["in_area", "far", "near"]}
        scores = {("in_area", 1): 0.95, ("near", 1): 0.93, ("far", 1): 0.90}
        released_prefs[1].sort(key=lambda e: -scores[(e, 1)])
        caps = {"in_area": 1, "near": 1, "far": 1}
        slates = {1: []}
        used = {e: 0 for e in caps}
        assign_reviewers.assignment_phase(
            [1], gated_prefs, {1: 2}, slates, used, caps, scores, set(caps)
        )
        self.assertEqual(["in_area"], slates[1])
        assign_reviewers.assignment_phase(
            [1], released_prefs, {1: 2 - len(slates[1])}, slates, used, caps, scores, set(caps)
        )
        self.assertEqual(["in_area", "near"], slates[1])


class RandomizedBaselineTests(unittest.TestCase):
    """--score-mode random: the matcher ranks on noise, the reports still
    measure the true SPECTER2 affinity of the slate that produces."""

    def reviewer(self, email, primary="Memory Systems"):
        return Reviewer(
            email=email, first="Test", last="Person", dblp_url="", pid=None,
            affiliation="Example", primary=primary, secondary="", tertiary="",
            keywords="", tier="full", override_cap=None,
        )

    def paper(self, pid, topics=("Memory Systems",), **kwargs):
        base = {"pid": pid, "title": f"Paper {pid}.", "topics": list(topics),
                "authors": [], "contacts": [], "pc_conflicts": {},
                "reserve_reviewer": ""}
        base.update(kwargs)
        return base

    def build(self, papers, reviewers, vectors=None, **kwargs):
        """build_pair_scores over a tiny hand-built pool.

        Every vector is the same unit vector unless a test supplies its own, so
        by default every cosine is 1.0 and any spread in the pair lists is the
        randomizer's doing and nothing else's.
        """
        emails = [r.email for r in reviewers]
        matrix = (np.ones((len(emails), 3), dtype=np.float32)
                  if vectors is None else np.array(vectors, dtype=np.float32))
        cache = {str(p["pid"]): {"vector": [1.0, 0.0, 0.0]} for p in papers}
        kwargs.setdefault("light_cap", 7)
        kwargs.setdefault("full_cap", 15)
        return assign_reviewers.build_pair_scores(
            papers, cache, emails, matrix, {r.email: r for r in reviewers}, {}, **kwargs
        )

    def test_the_default_score_mode_ranks_and_measures_on_one_object(self):
        # Not assertEqual: two dicts holding equal floats would drift the first
        # time someone edits one branch and not the other. Identity is the only
        # proof the production path cannot change, and it is what lets every
        # measurement call site switch to affinity_lookup for free.
        scores = self.build([self.paper(1)], [self.reviewer("a@x.edu")])
        self.assertIs(scores.rank, scores.affinity)

    def test_a_random_run_still_measures_the_true_specter2_affinity(self):
        # The whole experiment: rank on noise, measure the cosine. If affinity
        # were overwritten too, every arm would report the same meaningless
        # mean and the comparison would silently be of nothing.
        papers, revs = [self.paper(1)], [self.reviewer("a@x.edu")]
        real = self.build(papers, revs)
        random_run = self.build(papers, revs, score_mode="random", score_seed=1)
        self.assertIsNot(random_run.rank, random_run.affinity)
        self.assertEqual(real.affinity, random_run.affinity)
        self.assertNotEqual(random_run.rank, random_run.affinity)

    def test_the_gated_and_area_released_passes_agree_on_one_pairs_score(self):
        # build_pair_scores scores each paper twice, once gated and once
        # area-released. A stream RNG would hand the same pair two different
        # draws, and the gated preference list would then disagree with the
        # table deferred acceptance bumps on.
        revs = [self.reviewer("in@x.edu"), self.reviewer("out@x.edu", primary="Security")]
        scores = self.build([self.paper(1)], revs, score_mode="random", score_seed=3)
        gated = dict(scores.eligible[1])
        released = dict(scores.released[1])
        self.assertEqual(["in@x.edu"], list(gated))
        self.assertEqual({"in@x.edu", "out@x.edu"}, set(released))
        self.assertEqual(released["in@x.edu"], gated["in@x.edu"])
        self.assertEqual(scores.rank[("in@x.edu", 1)], gated["in@x.edu"])

    def test_the_preference_lists_carry_the_score_the_matcher_bumps_on(self):
        # deferred_acceptance's deferral deques assume the preference list is
        # sorted by the score it bumps on ("the pref list is descending, so
        # appends keep each deque sorted"). Leaving these lists on the cosine
        # while ranking on the draw pops the wrong deferred candidate --
        # silently, with no self-check that catches it.
        revs = [self.reviewer(f"r{i}@x.edu") for i in range(8)]
        # Descending cosines, so an un-rescored list would come out in email
        # order and this test would pass by accident.
        vectors = [[1.0 - i / 100, 0.0, 0.0] for i in range(8)]
        scores = self.build([self.paper(1)], revs, vectors=vectors,
                            score_mode="random", score_seed=5)
        for pairs in (scores.eligible[1], scores.released[1]):
            by_score = [e for e, _ in sorted(pairs, key=lambda es: -es[1])]
            by_rank = sorted((e for e, _ in pairs),
                             key=lambda e: -scores.rank[(e, 1)])
            self.assertEqual(by_rank, by_score)
        # And it really did reorder relative to affinity, or the check is vacuous.
        by_affinity = sorted((e for e, _ in scores.eligible[1]),
                             key=lambda e: -scores.affinity[(e, 1)])
        self.assertNotEqual(by_affinity, [e for e, _ in sorted(
            scores.eligible[1], key=lambda es: -es[1])])

    def test_the_no_area_gate_pair_lists_stay_one_object(self):
        # main() and the surplus stage both pass eligible and released prefs;
        # under --no-area-gate they are the same pool, and rebuilding the list
        # twice would be two independent re-scores to keep in step forever.
        scores = self.build([self.paper(1)], [self.reviewer("a@x.edu")],
                            area_gate=False, score_mode="random")
        self.assertIs(scores.eligible[1], scores.released[1])

    def test_the_random_score_does_not_move_when_python_reseeds_string_hash(self):
        # hash() is salted per process -- the trap that once made the DBLP
        # snapshot files reorder on every run and left cmp unable to tell a
        # real change from none. A rerun of an arm has to be identical.
        code = ("from scripts.assign_reviewers import random_pair_score;"
                "print(random_pair_score(7, 'a@x.edu', 42))")
        root = Path(__file__).resolve().parents[1]
        seen = set()
        for salt in ("0", "1", "random"):
            env = {**os.environ, "PYTHONHASHSEED": salt,
                   "PYTHONPATH": os.pathsep.join((str(root / "src"), str(root)))}
            out = subprocess.run([sys.executable, "-c", code], env=env,
                                 capture_output=True, text=True, check=True)
            seen.add(out.stdout.strip())
        self.assertEqual(1, len(seen), f"salt changed the draw: {seen}")

    def test_two_seeds_disagree_on_the_same_pair(self):
        self.assertNotEqual(assign_reviewers.random_pair_score(1, "a@x.edu", 3),
                            assign_reviewers.random_pair_score(2, "a@x.edu", 3))

    def test_the_random_scores_spread_over_the_unit_interval(self):
        # Catches a digest wired up to return something near-constant, which
        # would leave the arm a tie-break sweep rather than a randomization and
        # would look like a suspiciously small affinity drop.
        draws = [assign_reviewers.random_pair_score(1, f"r{i}@x.edu", 1)
                 for i in range(1000)]
        self.assertTrue(all(0.0 <= d < 1.0 for d in draws))
        self.assertLess(min(draws), 0.05)
        self.assertGreater(max(draws), 0.95)
        self.assertAlmostEqual(0.5, sum(draws) / len(draws), delta=0.05)

    def test_a_random_slate_still_has_no_blocking_pair(self):
        # Stability is a property of the preference ORDER, not of what the
        # numbers mean, so a randomized arm's F1 self-check must still be 0. A
        # failure here means the seam desynchronized the preference lists from
        # the score table, not that the theory stopped applying.
        pids = list(range(6))
        emails = [f"r{i}@x.edu" for i in range(9)]
        scores = {(e, pid): assign_reviewers.random_pair_score(11, e, pid)
                  for e in emails for pid in pids}
        prefs = {pid: sorted(emails, key=lambda e: -scores[(e, pid)]) for pid in pids}
        caps = {e: 3 for e in emails}
        targets = {pid: 4 for pid in pids}
        pairs = {pid: [(e, scores[(e, pid)]) for e in prefs[pid]] for pid in pids}
        held = assign_reviewers.deferred_acceptance(pids, prefs, targets, caps, scores)
        self.assertEqual(0, assign_reviewers.count_blocking_pairs(
            pairs, held, caps, targets, scores))

    def test_a_conflicted_reviewer_is_unreachable_however_the_pair_is_scored(self):
        # COI is applied inside eligible_scores and the randomizer runs strictly
        # after it. The risk is someone "optimising" by drawing scores over
        # candidate_emails directly, which would put every conflict back.
        clean = self.reviewer("clean@x.edu")
        declared = self.reviewer("declared@x.edu")
        author = self.reviewer("author@x.edu")
        derived = self.reviewer("derived@x.edu")
        p = self.paper(1, authors=[{"email": "Author@X.edu"}],
                       pc_conflicts={declared.email: 1})
        scores = assign_reviewers.build_pair_scores(
            [p], {"1": {"vector": [1.0, 0.0, 0.0]}},
            [r.email for r in (clean, declared, author, derived)],
            np.ones((4, 3), dtype=np.float32),
            {r.email: r for r in (clean, declared, author, derived)},
            {1: {derived.email}},
            score_mode="random", score_seed=1, light_cap=7, full_cap=15,
        )
        self.assertEqual([clean.email], [e for e, _ in scores.released[1]])
        self.assertEqual([clean.email], [e for e, _ in scores.eligible[1]])
        for blocked in (declared, author, derived):
            self.assertNotIn((blocked.email, 1), scores.rank)
            self.assertNotIn((blocked.email, 1), scores.affinity)

    def test_a_random_run_refuses_to_write_the_production_hotcrp_csv(self):
        # A baseline slate ignores affinity entirely. Nothing that can be fed
        # to HotCRP should ever come out of one, whatever the filename says.
        with (
            mock.patch.object(sys, "argv", ["assign_reviewers.py", "--score-mode",
                                            "random", "--hotcrp-csv", "out.csv"]),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            assign_reviewers.main()
        self.assertEqual(2, raised.exception.code)

    def test_an_unknown_score_mode_is_rejected_before_io(self):
        with (
            mock.patch.object(sys, "argv", ["assign_reviewers.py", "--score-mode", "uniform"]),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            assign_reviewers.main()
        self.assertEqual(2, raised.exception.code)

    def read_pairs(self, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pairs.csv")
            assign_reviewers.write_pairs_csv(path, **kwargs)
            with open(path, newline="", encoding="utf-8") as f:
                return list(csv.DictReader(f))

    def test_the_pairs_csv_records_the_rank_score_beside_the_true_affinity(self):
        # Two identical columns is the byte-level evidence a file came from the
        # production path; two different ones, that it came from a random arm.
        affinity = {("a@x.edu", 1): 0.9, ("b@x.edu", 1): 0.8}
        same = self.read_pairs(
            slates={1: ["a@x.edu", "b@x.edu"]}, assigned_via={(1, "a@x.edu"): "senior anchor"},
            rank_lookup=affinity, affinity_lookup=affinity,
        )
        self.assertEqual(["a@x.edu", "b@x.edu"], [r["email"] for r in same])
        self.assertEqual(["senior anchor", "fill"], [r["phase"] for r in same])
        for row in same:
            self.assertEqual(row["rank_score"], row["affinity"])
        rank = {("a@x.edu", 1): 0.1, ("b@x.edu", 1): 0.7}
        differ = self.read_pairs(
            slates={1: ["a@x.edu", "b@x.edu"]}, assigned_via={},
            rank_lookup=rank, affinity_lookup=affinity,
        )
        # Still ordered by affinity, not by the draw: the file is a measurement.
        self.assertEqual(["a@x.edu", "b@x.edu"], [r["email"] for r in differ])
        self.assertNotEqual(differ[0]["rank_score"], differ[0]["affinity"])

    def test_the_pairs_csv_keeps_full_precision(self):
        # The reports' three decimals cannot resolve the differences this repo
        # already reasons about, which run to 0.0006.
        scores = {("a@x.edu", 1): 0.96501234567}
        row = self.read_pairs(slates={1: ["a@x.edu"]}, assigned_via={},
                              rank_lookup=scores, affinity_lookup=scores)[0]
        self.assertEqual(0.96501234567, float(row["affinity"]))

    def test_the_goodness_ceiling_is_the_best_k_eligible_ignoring_capacity(self):
        # The bracket exists so a drop can be read as a fraction of the range
        # actually available. A ceiling that quietly counted fewer than k
        # reviewers would inflate it and make every arm look closer to optimal.
        report = self.goodness_report(
            papers=[self.paper(1)],
            goodness={1: 0.5},
            floor={1: 0.25},
            ceiling={1: 0.75},
            ceiling_k=5,
        )
        self.assertIn("pool floor 0.250", report)
        self.assertIn("best-5 ceiling 0.750", report)
        self.assertIn("NOT jointly achievable", report)

    def test_the_bracket_is_absent_when_it_was_not_computed(self):
        # Every other caller of match_goodness_report, and the shape that keeps
        # the production transcript unchanged when the bracket is not wanted.
        self.assertNotIn("Bracket", self.goodness_report(
            papers=[self.paper(1)], goodness={1: 0.5}))

    def goodness_report(self, **kwargs):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assign_reviewers.match_goodness_report(**kwargs)
        return out.getvalue()


class SurplusDistributionTests(unittest.TestCase):
    """The stage that spends leftover capacity on the worst-matched papers."""

    def stub_reviewer(self, name):
        """The two report functions below read only .name and .primary."""
        return argparse.Namespace(name=name, primary="Memory systems")

    def surplus(self, prefs, slates, caps, scores, *, released=None, used=None,
                base=1, surplus=1, capped=()):
        """distribute_surplus over one pool, gated and released prefs the same
        unless a test needs them to differ."""
        used = {e: 0 for e in caps} if used is None else used
        assigned_via = {}
        pids = sorted(slates)
        added, rounds = assign_reviewers.distribute_surplus(
            pids, prefs, released if released is not None else prefs, slates, used,
            caps, scores, assigned_via, capped,
            base_target={pid: base for pid in pids}, surplus_per_paper=surplus,
        )
        return added, rounds, used, assigned_via

    def test_the_worst_matched_paper_gets_the_only_spare_slot(self):
        # Both papers are at target and both want `spare`, but only one slot
        # exists. It has to go to the paper whose slate matches it worst.
        scores = {("held1", 1): 0.90, ("held2", 2): 0.60,
                  ("spare", 1): 0.50, ("spare", 2): 0.50}
        added, _, _, _ = self.surplus(
            {1: ["spare"], 2: ["spare"]}, {1: ["held1"], 2: ["held2"]},
            {"held1": 1, "held2": 1, "spare": 1}, scores,
            used={"held1": 1, "held2": 1, "spare": 0},
        )
        self.assertEqual({2: ["spare"]}, added)

    def test_a_slot_the_worst_paper_cannot_use_flows_to_the_next_worst(self):
        # Paper 2 is worst-matched but its only candidate is already at
        # capacity, so its offer places nothing. The slot must not be lost with
        # it: a later round has to reach paper 1.
        scores = {("held1", 1): 0.90, ("held2", 2): 0.60,
                  ("full", 2): 0.99, ("spare", 1): 0.50}
        added, rounds, _, _ = self.surplus(
            {1: ["spare"], 2: ["full"]}, {1: ["held1"], 2: ["held2"]},
            {"held1": 1, "held2": 1, "full": 1, "spare": 1}, scores,
            used={"held1": 1, "held2": 1, "full": 1, "spare": 0},
        )
        self.assertEqual({1: ["spare"]}, added)
        self.assertEqual(2, rounds)  # round 1 drops paper 2, round 2 serves 1

    def test_surplus_is_purely_additive(self):
        # Nothing an earlier phase froze may move: every base slate stays a
        # prefix of the final one, and capacity only ever gets spent.
        scores = {("a", 1): 0.9, ("b", 1): 0.8, ("c", 1): 0.7,
                  ("a", 2): 0.6, ("b", 2): 0.5, ("c", 2): 0.4}
        slates = {1: ["a"], 2: ["b"]}
        base = {pid: list(v) for pid, v in slates.items()}
        used = {"a": 1, "b": 1, "c": 0}
        self.surplus({1: ["c"], 2: ["c"]}, slates, {"a": 1, "b": 1, "c": 2},
                     scores, used=used)
        for pid, before in base.items():
            self.assertEqual(before, slates[pid][:len(before)])
        self.assertEqual(1, used["a"])
        self.assertEqual(1, used["b"])

    def test_a_paper_stops_at_the_ceiling(self):
        # Capacity is abundant, so only the ceiling can stop the paper growing.
        pool = {f"r{i}": 1 for i in range(6)}
        scores = {(e, 1): 0.5 for e in pool}
        for surplus, expected in ((1, 2), (2, 3)):
            slates = {1: ["seed"]}
            scores[("seed", 1)] = 0.9
            self.surplus({1: sorted(pool)}, slates, {**pool, "seed": 1}, scores,
                         used={**{e: 0 for e in pool}, "seed": 1}, surplus=surplus)
            self.assertEqual(expected, len(slates[1]))

    def test_the_class_caps_still_bind_a_surplus_pick(self):
        # The junior slot is already spent by a frozen assignment, so the
        # better-scoring junior must be passed over for the typical reviewer.
        juniors = frozenset({"j1", "j2"})
        scores = {("j1", 1): 0.9, ("j2", 1): 0.8, ("t1", 1): 0.7}
        slates = {1: ["j1"]}
        added, _, _, _ = self.surplus(
            {1: ["j2", "t1"]}, slates, {"j1": 1, "j2": 1, "t1": 1}, scores,
            used={"j1": 1, "j2": 0, "t1": 0}, capped=[(juniors, 1)],
        )
        self.assertEqual({1: ["t1"]}, added)

    def test_spare_capacity_is_never_exceeded_and_the_loop_terminates(self):
        rng = random.Random(20260801)
        for _ in range(40):
            n_papers = rng.randint(2, 6)
            pool = {f"r{i}": rng.randint(1, 3) for i in range(rng.randint(2, 8))}
            pids = list(range(1, n_papers + 1))
            scores = {(e, pid): rng.random() for e in pool for pid in pids}
            prefs = {pid: sorted(pool, key=lambda e: -scores[(e, pid)]) for pid in pids}
            slates = {pid: [] for pid in pids}
            used = {e: 0 for e in pool}
            # Seed every paper to its base target of 1 so all are eligible.
            for pid in pids:
                pick = next((e for e in prefs[pid] if used[e] < pool[e]), None)
                if pick is not None:
                    slates[pid].append(pick)
                    used[pick] += 1
            spare_before = assign_reviewers.spare_capacity(pool, used)
            added, rounds, used, _ = self.surplus(
                prefs, slates, pool, scores, used=used, surplus=2,
            )
            placed = sum(len(v) for v in added.values())
            self.assertLessEqual(placed, spare_before)
            self.assertLessEqual(rounds, len(pids) + spare_before)
            for e, n in used.items():
                self.assertLessEqual(n, pool[e])
            for pid in pids:
                self.assertLessEqual(len(slates[pid]), 1 + 2)

    def test_a_paper_under_its_target_is_not_offered_a_surplus_slot(self):
        # Paper 1 never reached its target, so it is not a surplus candidate
        # even with a free reviewer in reach — which is what keeps the shortage
        # report, the relaxation report and the exit code honest.
        scores = {("held", 2): 0.9, ("spare", 1): 0.8, ("spare", 2): 0.5}
        slates = {1: [], 2: ["held"]}
        added, _, _, _ = self.surplus(
            {1: ["spare"], 2: ["spare"]}, slates, {"held": 1, "spare": 1}, scores,
            used={"held": 1, "spare": 0},
        )
        self.assertEqual({2: ["spare"]}, added)
        self.assertEqual([], slates[1])

    def test_the_stage_is_off_at_zero(self):
        scores = {("held", 1): 0.9, ("spare", 1): 0.8}
        slates = {1: ["held"]}
        added, rounds, used, _ = self.surplus(
            {1: ["spare"]}, slates, {"held": 1, "spare": 1}, scores,
            used={"held": 1, "spare": 0}, surplus=0,
        )
        self.assertEqual(({}, 0), (added, rounds))
        self.assertEqual(["held"], slates[1])
        self.assertEqual(0, used["spare"])

    def test_no_round_runs_when_every_reviewer_is_at_capacity(self):
        scores = {("held", 1): 0.9, ("full", 1): 0.8}
        added, rounds, _, _ = self.surplus(
            {1: ["full"]}, {1: ["held"]}, {"held": 1, "full": 1}, scores,
            used={"held": 1, "full": 1},
        )
        self.assertEqual(({}, 0), (added, rounds))

    def test_surplus_picks_stay_out_of_the_relaxation_report(self):
        # An area-released surplus pick is an area release, but on a slot the
        # paper was never owed; reporting it would imply a shortfall.
        self.assertTrue(assign_reviewers.SURPLUS_PHASES.isdisjoint(
            assign_reviewers.UNRELAXED_PHASES))
        papers = [{"pid": 1, "title": "T"}]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _, relaxed = assign_reviewers.relaxation_report(
                [], papers, {1: ["a", "b"]}, {1: 2},
                {(1, "a"): "fill", (1, "b"): "surplus (area released)"},
                {1: 0.9}, {("a", 1): 0.9, ("b", 1): 0.9},
                {"a": self.stub_reviewer("A"), "b": self.stub_reviewer("B")}, None,
            )
        self.assertEqual(0, relaxed)

    def test_surplus_report_prints_both_goodness_figures(self):
        # The 6th reviewer scores below the slate that outbid it, so the
        # full-slate mean DROPS. Both numbers have to be on the page or the
        # report reads as if boosting made the paper worse.
        papers = [{"pid": 1, "title": "T"}]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            placed = assign_reviewers.surplus_report(
                papers, {1: ["b"]}, {1: 0.900}, {1: 0.850},
                {("b", 1): 0.800}, {"b": self.stub_reviewer("B")}, None,
                {(1, "b"): "surplus"},
                reviewers_per_paper=5, surplus_per_paper=1,
                spare_before=3, spare_after=2, rounds=1,
            )
        out = buf.getvalue()
        self.assertEqual(1, placed)
        self.assertIn("0.900 -> 0.850", out)
        self.assertIn("3 slot(s) before, 2 after", out)

    def test_surplus_report_names_the_leftover_capacity_when_off(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            placed = assign_reviewers.surplus_report(
                [], {}, {}, {}, {}, {}, None, {},
                reviewers_per_paper=5, surplus_per_paper=0,
                spare_before=17, spare_after=17, rounds=0,
            )
        self.assertEqual(0, placed)
        self.assertIn("17 reviewer-slot(s) left unspent", buf.getvalue())

    def test_reviewer_assignments_report_lists_papers_alphabetically_by_name(self):
        papers = [
            {"pid": 1, "title": "Paper One"},
            {"pid": 2, "title": "Paper Two"},
        ]
        reviewer_papers = {"b@example.com": [1, 2], "a@example.com": [1]}
        reviewers_by_email = {
            "b@example.com": argparse.Namespace(name="Bob", primary="Memory", tier="full"),
            "a@example.com": argparse.Namespace(name="Alice", primary="Memory", tier="light"),
        }
        affinity_lookup = {
            ("b@example.com", 1): 0.5, ("b@example.com", 2): 0.9,
            ("a@example.com", 1): 0.7,
        }
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assign_reviewers.reviewer_assignments_report(
                papers, reviewer_papers, reviewers_by_email, affinity_lookup, None,
            )
        out = buf.getvalue()
        # Alice sorts before Bob alphabetically, regardless of dict insertion order.
        self.assertLess(out.index("Alice"), out.index("Bob"))
        # Bob's own papers rank by score descending: [2] (0.900) before [1] (0.500).
        bob_block = out[out.index("Bob"):]
        self.assertLess(bob_block.index("[2] Paper Two"), bob_block.index("[1] Paper One"))
        self.assertIn("2 paper(s)", out)


class AssignmentIoTests(unittest.TestCase):
    def write_hotcrp_csv_fixture(self, path, rows):
        with Path(path).open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(list(assignment_io.HOTCRP_CSV_HEADER))
            writer.writerow(["all", "clearreview", "all", "R1"])
            for pid, email in rows:
                writer.writerow([pid, "primaryreview", email, "R1"])

    def test_sniff_format_rejects_an_unknown_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            path.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                assignment_io.sniff_format(str(path))

    def test_load_assignment_pairs_skips_the_clearreview_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "assignment.csv"
            self.write_hotcrp_csv_fixture(path, [(1, "a@example.com"), (1, "b@example.com")])
            fmt, pairs = assignment_io.load_assignment_pairs(str(path))
        self.assertEqual("hotcrp", fmt)
        self.assertEqual({"a@example.com", "b@example.com"}, set(pairs[1]))

    def test_load_assignment_pairs_skips_non_primaryreview_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "assignment.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(list(assignment_io.HOTCRP_CSV_HEADER))
                writer.writerow(["all", "clearreview", "all", "R1"])
                writer.writerow([1, "secondaryreview", "c@example.com", "R1"])
                writer.writerow([1, "primaryreview", "a@example.com", "R1"])
            fmt, pairs = assignment_io.load_assignment_pairs(str(path))
        self.assertEqual({"a@example.com"}, set(pairs[1]))

    def test_round_trips_write_hotcrp_csv(self):
        slates = {1: ["a@example.com", "b@example.com"]}
        scores = {("a@example.com", 1): 0.9, ("b@example.com", 1): 0.5}
        reviewers_by_email = {e: reviewer_with_email(e) for e in slates[1]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "assignment.csv"
            assign_reviewers.write_hotcrp_csv(str(path), slates, scores, reviewers_by_email)
            fmt, pairs = assignment_io.load_assignment_pairs(str(path))
        self.assertEqual("hotcrp", fmt)
        self.assertEqual({"a@example.com", "b@example.com"}, set(pairs[1]))

    def test_round_trips_write_pairs_csv(self):
        slates = {1: ["a@example.com"]}
        assigned_via = {(1, "a@example.com"): "fill"}
        scores = {("a@example.com", 1): 0.75}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.csv"
            assign_reviewers.write_pairs_csv(str(path), slates, assigned_via, scores, scores)
            fmt, pairs = assignment_io.load_assignment_pairs(str(path))
        self.assertEqual("pairs", fmt)
        self.assertEqual("fill", pairs[1]["a@example.com"].phase)
        self.assertAlmostEqual(0.75, pairs[1]["a@example.com"].affinity)

    def test_hotcrp_email_map_and_remap_pairs(self):
        reviewers_by_email = {
            "form@example.com": reviewer_with_email("form@example.com", hotcrp_email="real@other.edu"),
            "same@example.com": reviewer_with_email("same@example.com"),
        }
        email_map = assignment_io.hotcrp_email_map(reviewers_by_email)
        self.assertEqual(
            {"form@example.com": "real@other.edu", "same@example.com": "same@example.com"}, email_map
        )
        pairs = {1: {"form@example.com": Pair(), "unknown@example.com": Pair()}}
        remapped = assignment_io.remap_pairs(pairs, email_map)
        self.assertEqual({"real@other.edu", "unknown@example.com"}, set(remapped[1]))


class DiffAssignmentsTests(unittest.TestCase):
    def test_identical_files_have_zero_churn(self):
        old = {1: {"a@example.com": Pair()}, 2: {"b@example.com": Pair()}}
        new = {1: {"a@example.com": Pair()}, 2: {"b@example.com": Pair()}}
        changed, load_delta = diff_assignments.diff_pairs(old, new)
        self.assertEqual({}, changed)
        self.assertEqual(Counter(), load_delta)

    def test_added_and_removed_are_detected(self):
        old = {1: {"a@example.com": Pair()}}
        new = {1: {"b@example.com": Pair()}}
        changed, load_delta = diff_assignments.diff_pairs(old, new)
        self.assertEqual(({"a@example.com"}, {"b@example.com"}), changed[1])
        self.assertEqual(-1, load_delta["a@example.com"])
        self.assertEqual(1, load_delta["b@example.com"])

    def test_reviewer_lineup_changes_finds_a_net_zero_swap_load_delta_misses(self):
        # a@example.com loses pid 1 and gains pid 2 -- net load delta 0, but
        # their actual paper lineup is entirely different.
        old = {1: {"a@example.com": Pair()}, 2: {}}
        new = {1: {}, 2: {"a@example.com": Pair()}}
        changed, load_delta = diff_assignments.diff_pairs(old, new)
        self.assertEqual(0, load_delta["a@example.com"])
        lineups = diff_assignments.reviewer_lineup_changes(changed)
        self.assertEqual(({1}, {2}), lineups["a@example.com"])

    def test_reviewer_lineup_changes_distinguishes_gained_only_from_lost_only(self):
        old = {1: {"lost@example.com": Pair()}, 2: {}}
        new = {1: {}, 2: {"gained@example.com": Pair()}}
        changed, _ = diff_assignments.diff_pairs(old, new)
        lineups = diff_assignments.reviewer_lineup_changes(changed)
        self.assertEqual(({1}, set()), lineups["lost@example.com"])
        self.assertEqual((set(), {2}), lineups["gained@example.com"])
        self.assertNotIn("untouched@example.com", lineups)

    def test_normalization_prevents_false_churn_across_hotcrp_and_roster_email(self):
        # Same reviewer, one file keyed on hotcrp_email (a hotcrp-csv), the
        # other on the roster email (a pairs-csv) -- comparing raw would read
        # as one reviewer leaving and a different one arriving.
        old = {1: {"real@other.edu": Pair()}}
        new = {1: {"form@example.com": Pair()}}
        email_map = {"form@example.com": "real@other.edu"}
        normalized_old = assignment_io.remap_pairs(old, {})
        normalized_new = assignment_io.remap_pairs(new, email_map)
        changed, _ = diff_assignments.diff_pairs(normalized_old, normalized_new)
        self.assertEqual({}, changed)
        changed_raw, _ = diff_assignments.diff_pairs(old, new)
        self.assertEqual(({"real@other.edu"}, {"form@example.com"}), changed_raw[1])

    def test_goodness_delta_uses_paper_goodness_and_skips_unscored_papers(self):
        changed = {
            1: (set(), {"a@example.com"}),
            2: (set(), {"b@example.com"}),
        }
        old = {
            1: {"old-a@example.com": Pair(affinity=0.6)},
            2: {"old-b@example.com": Pair(affinity=0.5)},
        }
        new = {
            1: {"old-a@example.com": Pair(affinity=0.6), "a@example.com": Pair(affinity=1.0)},
            2: {"old-b@example.com": Pair(affinity=0.5), "b@example.com": Pair(affinity=None)},
        }
        scorable, skipped = diff_assignments.goodness_delta(changed, old, new)
        self.assertEqual([2], skipped)
        self.assertAlmostEqual(0.2, scorable[1])  # old mean 0.6 -> new mean (0.6+1.0)/2 = 0.8

    def test_backfill_affinity_matches_the_cached_cosine(self):
        pairs = {1: {"a@example.com": Pair()}}
        paper_cache = {"1": {"vector": [1.0, 0.0]}}
        reviewer_fp = {"a@example.com": {"vector": [1.0, 0.0]}}
        backfilled, misses = diff_assignments.backfill_affinity(pairs, paper_cache, reviewer_fp)
        self.assertEqual(0, misses)
        self.assertAlmostEqual(1.0, backfilled[1]["a@example.com"].affinity, places=5)

    def test_backfill_affinity_counts_a_missing_fingerprint_without_crashing(self):
        pairs = {1: {"a@example.com": Pair()}}
        backfilled, misses = diff_assignments.backfill_affinity(pairs, {}, {})
        self.assertEqual(1, misses)
        self.assertIsNone(backfilled[1]["a@example.com"].affinity)


class FillOpenSlotsTests(unittest.TestCase):
    def test_target_uses_the_baseline_slate_size_not_reviewers_per_paper(self):
        # A 6-reviewer paper (a surplus pick already placed) loses one
        # reviewer; the target must be 1 (6 - 5 survivors), never
        # DEFAULT_REVIEWERS_PER_PAPER (5) - 5 = 0, which would silently skip
        # refilling a paper that still has an open slot.
        baseline_pairs = {
            1: {e: Pair() for e in [
                "r1@example.com", "r2@example.com", "r3@example.com",
                "r4@example.com", "r5@example.com", "removed@example.com",
            ]},
        }
        reviewers_by_email = {
            f"r{i}@example.com": reviewer_with_email(f"r{i}@example.com") for i in range(1, 6)
        }
        slates, target = fill_open_slots.seed_slates_and_targets(
            baseline_pairs, [1], reviewers_by_email, "removed@example.com"
        )
        self.assertEqual(1, target[1])
        self.assertEqual(5, len(slates[1]))
        self.assertNotIn("removed@example.com", slates[1])

    def test_a_survivor_no_longer_on_the_roster_also_opens_a_slot(self):
        baseline_pairs = {1: {e: Pair() for e in ["stays@example.com", "gone@example.com"]}}
        reviewers_by_email = {"stays@example.com": reviewer_with_email("stays@example.com")}
        slates, target = fill_open_slots.seed_slates_and_targets(baseline_pairs, [1], reviewers_by_email, None)
        self.assertEqual(["stays@example.com"], slates[1])
        self.assertEqual(1, target[1])

    def test_derive_orphaned_pids_splits_by_current_paper_selection(self):
        baseline_pairs = {
            1: {"removed@example.com": Pair()},
            2: {"removed@example.com": Pair()},
            3: {"other@example.com": Pair()},
        }
        orphaned, dropped = fill_open_slots.derive_orphaned_pids(
            baseline_pairs, "removed@example.com", current_pids={1, 3}
        )
        self.assertEqual([1], orphaned)
        self.assertEqual([2], dropped)

    def test_seed_used_excludes_only_the_removed_reviewers_own_rows(self):
        baseline_pairs = {
            1: {"a@example.com": Pair(), "removed@example.com": Pair()},
            2: {"a@example.com": Pair()},
        }
        used = fill_open_slots.seed_used(baseline_pairs, "removed@example.com", current_pids={1, 2})
        self.assertEqual(2, used["a@example.com"])
        self.assertNotIn("removed@example.com", used)

    def test_seed_used_excludes_pairs_on_a_paper_no_longer_selected(self):
        # A withdrawn/desk-rejected paper frees its reviewers' capacity --
        # counting their obligation on a paper that no longer exists would
        # hide exactly the free capacity its removal creates.
        baseline_pairs = {
            1: {"a@example.com": Pair()},
            2: {"a@example.com": Pair()},  # pid 2 has since dropped out
        }
        used = fill_open_slots.seed_used(baseline_pairs, None, current_pids={1})
        self.assertEqual(1, used["a@example.com"])

    def test_still_on_roster_requires_force_to_proceed(self):
        reviewers_by_email = {"present@example.com": reviewer_with_email("present@example.com")}
        self.assertTrue(fill_open_slots.still_on_roster("present@example.com", reviewers_by_email, force=False))
        self.assertFalse(fill_open_slots.still_on_roster("present@example.com", reviewers_by_email, force=True))
        self.assertFalse(fill_open_slots.still_on_roster("absent@example.com", reviewers_by_email, force=False))

    def test_check_pid_containment_flags_a_pid_outside_the_orphaned_set(self):
        self.assertEqual([99], fill_open_slots.check_pid_containment([1, 99], orphaned_set={1, 2}))
        self.assertEqual([], fill_open_slots.check_pid_containment([1, 2], orphaned_set={1, 2}))

    def test_check_reviewer_caps_flags_only_reviewers_over_their_own_tier_cap(self):
        reviewers_by_email = {
            "light@example.com": reviewer_with_email("light@example.com"),
            "full@example.com": reviewer_with_email("full@example.com"),
        }
        reviewers_by_email["light@example.com"].tier = "light"
        reviewers_by_email["full@example.com"].tier = "full"
        used = {"light@example.com": 8, "full@example.com": 8}
        over = fill_open_slots.check_reviewer_caps(used, reviewers_by_email, light_cap=7, full_cap=15, reserve_cap=6)
        self.assertEqual(["light@example.com"], over)

    def test_check_hard_class_caps_flags_a_paper_over_its_country_cap(self):
        cn = frozenset({"a@example.com", "b@example.com"})
        capped = [(cn, {1: 1})]
        slates = {1: ["a@example.com", "b@example.com"]}
        violations = fill_open_slots.check_hard_class_caps(slates, [1], capped)
        self.assertEqual({1: [0]}, violations)

    def test_a_candidate_already_at_cap_elsewhere_is_never_picked_despite_scoring_highest(self):
        # seed_used correctly reflects a candidate's baseline load on OTHER
        # papers, so assignment_phase must respect it even though this run
        # only ever scores the one orphaned paper.
        baseline_pairs = {
            2: {"maxed@example.com": Pair()},
            3: {"maxed@example.com": Pair()},
        }
        used = fill_open_slots.seed_used(baseline_pairs, None, current_pids={2, 3})
        reviewer_cap = {"maxed@example.com": 2, "fresh@example.com": 5}
        score_lookup = {("maxed@example.com", 1): 0.99, ("fresh@example.com", 1): 0.10}
        prefs = {1: ["maxed@example.com", "fresh@example.com"]}
        held, _, _ = assign_reviewers.assignment_phase(
            [1], prefs, {1: 1}, {1: []}, used, reviewer_cap, score_lookup, set(reviewer_cap),
        )
        self.assertEqual(["fresh@example.com"], held[1])

    def test_a_paper_whose_survivors_already_hold_the_junior_cap_skips_a_top_scoring_junior(self):
        baseline_pairs = {1: {"survivor-jr@example.com": Pair(), "removed@example.com": Pair()}}
        reviewers_by_email = {"survivor-jr@example.com": reviewer_with_email("survivor-jr@example.com")}
        slates, target = fill_open_slots.seed_slates_and_targets(
            baseline_pairs, [1], reviewers_by_email, "removed@example.com"
        )
        self.assertEqual(1, target[1])
        juniors = frozenset({"survivor-jr@example.com", "new-jr@example.com"})
        reviewer_cap = {"new-jr@example.com": 5, "senior@example.com": 5}
        score_lookup = {("new-jr@example.com", 1): 0.95, ("senior@example.com", 1): 0.10}
        prefs = {1: ["new-jr@example.com", "senior@example.com"]}
        used = fill_open_slots.seed_used({}, None, current_pids=set())
        held, _, _ = assign_reviewers.assignment_phase(
            [1], prefs, {1: target[1]}, slates, used, reviewer_cap, score_lookup, set(reviewer_cap),
            [(juniors, 1)],
        )
        self.assertEqual(["senior@example.com"], held[1])

    def test_write_merged_hotcrp_csv_copies_non_orphaned_rows_verbatim(self):
        baseline_pairs = {
            1: {"x@example.com": Pair()},
            2: {"survivor@example.com": Pair()},
        }
        slates = {2: ["survivor@example.com", "new@example.com"]}
        reviewers_by_email = {
            "survivor@example.com": reviewer_with_email("survivor@example.com"),
            "new@example.com": reviewer_with_email("new@example.com", hotcrp_email="real-new@other.edu"),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "merged.csv"
            fill_open_slots.write_merged_hotcrp_csv(str(path), baseline_pairs, {2}, slates, reviewers_by_email)
            fmt, pairs = assignment_io.load_assignment_pairs(str(path))
        self.assertEqual({"x@example.com"}, set(pairs[1]))
        self.assertEqual({"survivor@example.com", "real-new@other.edu"}, set(pairs[2]))

    def test_removed_email_still_on_roster_is_refused_without_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline = Path(tmp) / "baseline.csv"
            with baseline.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(list(assignment_io.HOTCRP_CSV_HEADER))
                writer.writerow(["all", "clearreview", "all", "R1"])
                writer.writerow([1, "primaryreview", "still-here@example.com", "R1"])
            data = Path(tmp) / "data.json"
            data.write_text(json.dumps([
                {"pid": 1, "title": "T", "abstract": "one two three. four five six.",
                 "authors": [], "status": "submitted", "topics": []},
            ]), encoding="utf-8")
            form = Path(tmp) / "form.csv"
            with form.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp", "Please confirm your HotCRP email address", "PC membership",
                    "First Name", "Last Name", "Enter your DBLP Link",
                    "institutional affiliation", "primary area", "secondary area",
                    "tertiary area", "keywords", "Override paper assignment number",
                ])
                writer.writerow([
                    "07/01/2026 10:00:00", "still-here@example.com",
                    "Yes, I accept as a full PC member", "Ada", "Lovelace", "none",
                    "Example", "Memory", "", "", "", "",
                ])
            with (
                mock.patch.object(sys, "argv", [
                    "fill_open_slots.py", "--baseline", str(baseline), "--data", str(data),
                    "--csv", str(form), "--no-pc-check", "--paper-policy", "submitted",
                    "--removed-email", "still-here@example.com",
                    "--no-seniority", "--no-area-chair-exclusion",
                ]),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as raised,
            ):
                fill_open_slots.main()
            self.assertEqual(2, raised.exception.code)


class ClassificationTests(unittest.TestCase):
    def test_four_class_split(self):
        def label(target_papers, other_papers):
            records = [
                {"title": f"target {i}", "year": 2026, "venue": "ISCA"}
                for i in range(target_papers)
            ] + [
                {"title": f"other {i}", "year": 2026, "venue": "OSDI"}
                for i in range(other_papers)
            ]
            return classify_reviewers.classify(
                records, window=15, current_year=2026, senior_rate=0.8,
                junior_pubs=20, out_of_area_career=7,
            ).label

        self.assertEqual("senior", label(12, 0))       # 12 in-window target papers
        self.assertEqual("junior", label(3, 10))       # 13 pubs overall
        self.assertEqual("out-of-area", label(3, 30))  # 33 pubs, only 3 in target venues
        self.assertEqual("typical", label(8, 20))      # plenty of both, not senior


class PCDBOverrideTests(unittest.TestCase):
    def test_load_pcdb_by_header_names_with_merge_and_skips(self):
        # Columns deliberately reordered vs the real file; the name column
        # keeps its blank header. b@x.org's split rows must merge.
        content = (
            "Email,,#Chair,#PC,#ERC,TopPicks14,TopPicks24\n"
            "a@x.org,Alice,1,2,3,,PC\n"
            "b@x.org,Bob Variant One,0,4,2,,\n"
            "B@X.ORG ,Bob Variant Two,0,1,0,Chair,\n"
            ",No Email,0,9,9,,\n"
            "???,Garbage Email,0,9,9,,\n"
        )
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "pcdb.csv")
            Path(path).write_text(content, encoding="utf-8")
            pcdb = classify_reviewers.load_pcdb(path)
        self.assertEqual({"a@x.org", "b@x.org"}, set(pcdb))
        self.assertEqual((1, 3.5, True), (pcdb["a@x.org"].chair, pcdb["a@x.org"].score, pcdb["a@x.org"].toppicks))
        self.assertEqual((0, 6.0, True), (pcdb["b@x.org"].chair, pcdb["b@x.org"].score, pcdb["b@x.org"].toppicks))

    def test_override_rules_and_precedence(self):
        def apply(label, chair=0, score=0.0, toppicks=False):
            return classify_reviewers.apply_pcdb_override(
                label, classify_reviewers.PCDBRecord(chair, score, toppicks),
                senior_score=6.0, typical_score=2.0,
            )

        self.assertEqual(("senior", "chair"), apply("typical", chair=1))
        self.assertEqual(("senior", "toppicks"), apply("out-of-area", toppicks=True))
        self.assertEqual(("senior", "score 6"), apply("typical", score=6.0))
        self.assertEqual(("typical", "score 2"), apply("junior", score=2.0))
        # Senior rules beat the junior promotion; unknowns can go senior too.
        self.assertEqual(("senior", "chair"), apply("junior", chair=1, score=2.0))
        self.assertEqual(("senior", "chair"), apply("unknown", chair=1))
        # No demotions, no near-miss promotions, no redundant senior marks.
        self.assertEqual(("typical", ""), apply("typical", score=5.5))
        self.assertEqual(("junior", ""), apply("junior", score=1.5))
        self.assertEqual(("senior", ""), apply("senior", chair=1, score=10.0, toppicks=True))


class ReserveNeedTests(unittest.TestCase):
    def test_reserve_count_rounds_up(self):
        self.assertEqual(909, estimate_reserve_need.reserves_needed(3635, 4))
        self.assertEqual(1, estimate_reserve_need.reserves_needed(1, 4))
        self.assertEqual(3635, estimate_reserve_need.reserves_needed(3635, 1))
        self.assertEqual(0, estimate_reserve_need.reserves_needed(0, 4))

    def test_capacity_honors_the_per_reviewer_override(self):
        light = Reviewer(
            email="l@x.org", first="L", last="One", dblp_url="", pid=None, affiliation="",
            primary="Memory", secondary="", tertiary="", keywords="", tier="light",
            override_cap=None,
        )
        capped = Reviewer(**{**light.__dict__, "email": "c@x.org", "override_cap": 2})
        full = Reviewer(**{**light.__dict__, "email": "f@x.org", "tier": "full"})
        caps = [assign_reviewers.reviewer_paper_cap(r, 7, 15) for r in (light, capped, full)]
        self.assertEqual([7, 2, 15], caps)


class DblpNameHelperTests(unittest.TestCase):
    """dblp.py's name and `dblp`-field parsing, which the TRC matching rests on."""

    def test_field_splits_on_either_delimiter_and_keeps_alignment(self):
        split = dblp.split_dblp_field
        self.assertEqual(
            ["https://dblp.org/pid/345/0608.html", "", "https://dblp.org/pid/x/YuanXie.html"],
            split("https://dblp.org/pid/345/0608.html; None; https://dblp.org/pid/x/YuanXie.html"),
        )
        self.assertEqual(
            ["https://dblp.org/pid/345/0608.html", "https://dblp.org/pid/x/YuanXie.html"],
            split("https://dblp.org/pid/345/0608.html,\nhttps://dblp.org/pid/x/YuanXie.html"),
        )
        # A trailing delimiter is punctuation, not a fourth author.
        self.assertEqual(["", "58/292", "", "r/WonWooRo"], split("None; 58/292; N/A; r/WonWooRo;"))
        self.assertEqual([], split(None))

    def test_name_tokens_fold_suffixes_initials_accents_and_order(self):
        tokens = dblp.name_tokens
        self.assertEqual(tokens("Yang Wang 0089"), tokens("Yang Wang"))
        self.assertEqual(tokens("Matthew D. Sinclair"), tokens("Matthew Sinclair"))
        self.assertEqual(tokens("Jos\u00e9 Renau"), tokens("Jose Renau"))
        self.assertEqual(tokens("Won Woo Ro"), tokens("Ro Won Woo"))
        self.assertNotEqual(tokens("Zhe Jiang"), tokens("Hugo Jiang"))


class FakeDblp:
    """Stand-in for resolve_trc_members.Dblp with no network behind it."""

    def __init__(self, profiles=None, searches=None):
        self.profiles = profiles or {}
        self.searches = searches or {}
        self.fetched = []

    def profile(self, pid):
        self.fetched.append(pid)
        return self.profiles.get(pid)

    def search(self, query):
        return self.searches.get(query, [])


def dblp_profile(names, coauthors=(), pubs=1, affiliations=()):
    return {
        "names": list(names), "affiliations": list(affiliations),
        "coauthors": dict(coauthors), "pubs": pubs,
    }


class TrcRosterTests(unittest.TestCase):
    def test_profile_parse_keeps_coauthor_pids_and_skips_homepages(self):
        xml = (
            b'<dblpperson name="Rin Alder" pid="331/0100">'
            b'<person key="homepages/331/0100">'
            b'<author pid="331/0100">Rin Alder</author>'
            b'<note type="affiliation">Northern National University</note></person>'
            b'<r><www><title>Home Page</title><year>2024</year></www></r>'
            b'<r><inproceedings><author pid="331/0100">Rin Alder</author>'
            b'<author pid="10/1/DanaHaywood">Dana Haywood</author>'
            b'<title>A paper</title><year>2026</year></inproceedings></r>'
            b'</dblpperson>'
        )
        profile = resolve_trc_members.parse_profile(xml)
        self.assertEqual(["Rin Alder"], profile["names"])
        self.assertEqual(["Northern National University"], profile["affiliations"])
        self.assertEqual({"331/0100": "Rin Alder", "10/1/DanaHaywood": "Dana Haywood"},
                         profile["coauthors"])
        # The www record is a homepage, not a publication.
        self.assertEqual(1, profile["pubs"])

    def test_search_parse_handles_dblps_collapsed_single_element_lists(self):
        payload = {"result": {"hits": {"hit": {
            "info": {
                "author": "Rea Diaz 0001",
                "url": "https://dblp.org/pid/12/3456-1",
                "aliases": {"alias": "R. Diaz"},
                "notes": {"note": [
                    {"@type": "affiliation", "text": "University of the West Coast"},
                    {"@type": "award", "text": "not an affiliation"},
                ]},
            }}}}}
        self.assertEqual(
            [{"name": "Rea Diaz 0001", "pid": "12/3456-1", "aliases": ["R. Diaz"],
              "affiliations": ["University of the West Coast"]}],
            resolve_trc_members.parse_search(payload),
        )

    def test_search_rejects_the_endpoints_near_misses(self):
        # DBLP's author search is a similarity search: asked for "Cheng Chen"
        # it volunteers people who are not called that at all.
        dblp_stub = FakeDblp(searches={"Cheng Chen": [
            {"name": "Fu-Chen Cheng 0001", "pid": "241/0100-1", "aliases": [], "affiliations": []},
            {"name": "Cheng-Zhong Wu 0001", "pid": "181/0100-1", "aliases": [], "affiliations": []},
            {"name": "Cheng Chen 0012", "pid": "9/9999", "aliases": [], "affiliations": []},
        ]})
        hits = resolve_trc_members.search_candidates(dblp_stub, "Cheng Chen")
        self.assertEqual(["9/9999"], [h["pid"] for h in hits])

    def test_search_falls_back_to_splitting_a_camel_cased_name(self):
        dblp_stub = FakeDblp(searches={
            "AliReza MohammadiFarNejad": [],
            "Ali Reza Mohammadi Far Nejad": [
                {"name": "AliReza MohammadiFarNejad", "pid": "342/0100",
                 "aliases": [], "affiliations": []},
            ],
        })
        hits = resolve_trc_members.search_candidates(
            dblp_stub, "AliReza MohammadiFarNejad"
        )
        self.assertEqual(["342/0100"], [h["pid"] for h in hits])

    def test_self_declared_pids_ignore_a_misaligned_dblp_field(self):
        aligned = {
            "authors": [{"email": "S@x.org"}, {"email": "advisor@x.org"}],
            "dblp": "https://dblp.org/pid/1/Student.html; https://dblp.org/pid/2/Advisor.html",
        }
        misaligned = {
            "authors": [{"email": "s@x.org"}, {"email": "other@x.org"}, {"email": "z@x.org"}],
            "dblp": "1/Wrong; 2/AlsoWrong",
        }
        index = resolve_trc_members.index_self_declared_pids([aligned, misaligned])
        self.assertEqual({"1/Student": 1}, dict(index["s@x.org"]))
        self.assertEqual({"2/Advisor": 1}, dict(index["advisor@x.org"]))
        self.assertNotIn("z@x.org", index)

    def test_advisor_email_prefers_the_pc_form_over_the_author_list(self):
        pc_index = {frozenset({"rea", "diaz"}): {"rdiaz@westcoast.edu": "West Coast U"}}
        author_index = {frozenset({"rea", "diaz"}): {"rea@westcoast.edu": Counter({"UWC": 16})}}
        self.assertEqual(
            ("rdiaz@westcoast.edu", "pc-form"),
            resolve_trc_members.resolve_advisor_email(
                "Rea Diaz", "University of the West Coast", pc_index, author_index
            ),
        )

    def test_two_researchers_sharing_a_name_are_split_by_affiliation(self):
        author_index = {frozenset({"sam", "oyelaran"}): {
            "sam.oyelaran@inst-north.no": Counter({"Northern Institute of Science and Technology": 8}),
            "soyelaran@univ-west.edu": Counter({"University of the West": 6}),
        }}
        self.assertEqual(
            ("sam.oyelaran@inst-north.no", "hotcrp-author+affiliation"),
            resolve_trc_members.resolve_advisor_email(
                "Sam Oyelaran", "Northern Institute of Science and Technology", {}, author_index
            ),
        )
        # With no affiliation to go on, guessing between them is not allowed.
        email, resolution = resolve_trc_members.resolve_advisor_email(
            "Sam Oyelaran", "", {}, author_index
        )
        self.assertIsNone(email)
        self.assertEqual("ambiguous-hotcrp-author(2)", resolution)

    def test_one_person_with_two_addresses_resolves_to_the_one_they_use(self):
        author_index = {frozenset({"nia", "okafor"}): {
            "nok@univ-east.edu.cn": Counter({"UnivEast(GZ)": 21}),
            "niaokafor2022@mailhost.com": Counter({"UnivEast (GZ)": 4}),
        }}
        self.assertEqual(
            ("nok@univ-east.edu.cn", "hotcrp-author+most-used"),
            resolve_trc_members.resolve_advisor_email(
                "Nia Okafor", "UnivEast(GZ)", {}, author_index
            ),
        )

    def test_student_confirmed_when_both_routes_agree(self):
        dblp_stub = FakeDblp(
            profiles={
                "9/Advisor": dblp_profile(["An Advisor"], {"9/Student": "A Student"}),
                "9/Student": dblp_profile(["A Student"], {"9/Advisor": "An Advisor"}, pubs=4),
            },
            searches={"A Student": [
                {"name": "A Student", "pid": "9/Student", "aliases": [], "affiliations": []},
            ]},
        )
        pid, resolution, _ = resolve_trc_members.resolve_student_pid(
            dblp_stub, "A Student", "Example University", "s@x.org", ["9/Advisor"],
            {"s@x.org": Counter({"9/Student": 1})}, 100, 8,
        )
        self.assertEqual("9/Student", pid)
        self.assertEqual("confirmed", resolution)

    def test_dblp_homonym_bucket_is_never_accepted(self):
        # DBLP's bare "Robin Ross" page collects 725 papers by many people; the
        # advisor really has published with it, so co-authorship alone would
        # accept it.
        dblp_stub = FakeDblp(
            profiles={
                "9/Advisor": dblp_profile(["Hai Jin"], {"10/217": "Robin Ross"}),
                "10/217": dblp_profile(["Robin Ross"], {"9/Advisor": "Hai Jin"}, pubs=725),
            },
            searches={"Robin Ross": []},
        )
        pid, resolution, notes = resolve_trc_members.resolve_student_pid(
            dblp_stub, "Robin Ross", "HUST", "c@x.org", ["9/Advisor"], {}, 100, 8,
        )
        self.assertIsNone(pid)
        self.assertEqual("unverified", resolution)
        self.assertIn("725 publications", " ".join(notes))

    def test_a_crowd_of_same_named_people_is_left_unresolved_without_fetching(self):
        hits = [
            {"name": f"Yi Zhang {i:04d}", "pid": f"9/{i}", "aliases": [], "affiliations": []}
            for i in range(1, 21)
        ]
        dblp_stub = FakeDblp(searches={"Yi Zhang": hits})
        pid, resolution, notes = resolve_trc_members.resolve_student_pid(
            dblp_stub, "Yi Zhang", "HUST", "y@x.org", [], {}, 100, 8,
        )
        self.assertIsNone(pid)
        self.assertEqual("ambiguous", resolution)
        self.assertEqual([], dblp_stub.fetched)
        self.assertIn("20 DBLP people share this name", " ".join(notes))

    def test_self_declared_page_carries_a_student_with_no_joint_paper_yet(self):
        dblp_stub = FakeDblp(
            profiles={
                "9/Advisor": dblp_profile(["An Advisor"]),
                "9/New": dblp_profile(["New Student"], pubs=1),
            },
            searches={"New Student": []},
        )
        pid, resolution, notes = resolve_trc_members.resolve_student_pid(
            dblp_stub, "New Student", "Example University", "n@x.org", ["9/Advisor"],
            {"n@x.org": Counter({"9/New": 2})}, 100, 8,
        )
        self.assertEqual("9/New", pid)
        self.assertEqual("self-declared", resolution)
        self.assertIn("no joint publication with the advisor", " ".join(notes))

    def test_a_self_declared_page_belonging_to_someone_else_is_rejected(self):
        # The submission's dblp field was pasted in a different order, so the
        # positional read lands on a co-author's page.
        dblp_stub = FakeDblp(
            profiles={"9/Coauthor": dblp_profile(["Someone Else"], pubs=3)},
            searches={"New Student": []},
        )
        pid, resolution, _ = resolve_trc_members.resolve_student_pid(
            dblp_stub, "New Student", "Example University", "n@x.org", [],
            {"n@x.org": Counter({"9/Coauthor": 1})}, 100, 8,
        )
        self.assertIsNone(pid)
        self.assertEqual("unverified", resolution)

    def test_a_partial_name_match_needs_the_advisor_to_confirm_it(self):
        # "Nam Kim" against DBLP's "Nam Sung Kim": one name is contained in the
        # other, which is suggestive but not an identity.
        profiles = {"9/Student": dblp_profile(["Nam Sung Kim"], pubs=6)}
        alone = FakeDblp(profiles=profiles, searches={"Nam Kim": [
            {"name": "Nam Sung Kim", "pid": "9/Student", "aliases": [], "affiliations": []},
        ]})
        pid, resolution, _ = resolve_trc_members.resolve_student_pid(
            alone, "Nam Kim", "Example University", "n@x.org", [], {}, 100, 8,
        )
        self.assertIsNone(pid)
        self.assertEqual("unverified", resolution)

        confirmed = FakeDblp(
            profiles={
                "9/Advisor": dblp_profile(["An Advisor"], {"9/Student": "Nam Sung Kim"}),
                "9/Student": dblp_profile(["Nam Sung Kim"], {"9/Advisor": "An Advisor"}, pubs=6),
            },
            searches={"Nam Kim": []},
        )
        pid, resolution, notes = resolve_trc_members.resolve_student_pid(
            confirmed, "Nam Kim", "Example University", "n@x.org", ["9/Advisor"], {}, 100, 8,
        )
        self.assertEqual("9/Student", pid)
        self.assertEqual("confirmed-coauthor", resolution)
        self.assertIn("spells the name differently", " ".join(notes))

    def test_a_transliterated_name_survives_when_two_facts_back_it(self):
        # DBLP's spelling shares no word with the roster's, so no name test can
        # match — but the student named this page on their own submission and
        # their advisor has published with it.
        dblp_stub = FakeDblp(
            profiles={
                "9/Advisor": dblp_profile(
                    ["Lieven Eeckhout"], {"342/0100": "AliReza MohammadiFarNejad"}
                ),
                "342/0100": dblp_profile(
                    ["AliReza MohammadiFarNejad"], {"9/Advisor": "Lieven Eeckhout"}, pubs=6
                ),
            },
            searches={"Reza MohammadiFar": [], "Reza Mohammadi Far": []},
        )
        pid, resolution, notes = resolve_trc_members.resolve_student_pid(
            dblp_stub, "Reza MohammadiFar", "Riverside University", "h@x.org", ["9/Advisor"],
            {"h@x.org": Counter({"342/0100": 2})}, 100, 8,
        )
        self.assertEqual("342/0100", pid)
        # Proposed by the student's own submission, confirmed by co-authorship.
        self.assertEqual("confirmed-self-declared", resolution)
        self.assertIn("spells the name differently", " ".join(notes))

        # Self-declared alone, with a name nobody can check, is not enough.
        unbacked = FakeDblp(
            profiles={"342/0100": dblp_profile(["AliReza MohammadiFarNejad"], pubs=6)},
            searches={"Reza MohammadiFar": [], "Reza Mohammadi Far": []},
        )
        pid, resolution, notes = resolve_trc_members.resolve_student_pid(
            unbacked, "Reza MohammadiFar", "Riverside University", "h@x.org", [],
            {"h@x.org": Counter({"342/0100": 2})}, 100, 8,
        )
        self.assertIsNone(pid)
        self.assertEqual("unverified", resolution)
        # The page it looked at is named, so a human can finish in one click.
        self.assertIn("342/0100", " ".join(notes))

    def test_one_name_transliterated_two_ways_still_needs_confirming(self):
        compatible = resolve_trc_members.tokens_compatible
        tokens = dblp.name_tokens
        self.assertEqual("partial", compatible(tokens("Nadya Serrano"), tokens("Nadia Serrano")))
        # One shared token is required, so unrelated short names stay apart.
        self.assertIsNone(compatible(tokens("Jing Li"), tokens("Jung Lee")))
        self.assertIsNone(compatible(tokens("Yi Zhang"), tokens("Yu Huang")))
        # A dropped or added leading consonant is a different given name, not
        # a respelling — "Heng Chen" is not "Cheng Chen", however close the
        # strings score, and both may co-author with the same advisor.
        self.assertIsNone(compatible(tokens("Cheng Chen"), tokens("Heng Chen")))
        self.assertIsNone(compatible(tokens("Jing Zhao"), tokens("Jin Zhao")))
        self.assertIsNone(compatible(tokens("Yong Kim"), tokens("Yang Kim")))

        dblp_stub = FakeDblp(
            profiles={
                "9/Advisor": dblp_profile(["Noel Frame"], {"9/Student": "Nadia Serrano"}),
                "9/Student": dblp_profile(["Nadia Serrano"], {"9/Advisor": "Noel Frame"}, pubs=5),
            },
            searches={"Nadya Serrano": []},
        )
        pid, resolution, _ = resolve_trc_members.resolve_student_pid(
            dblp_stub, "Nadya Serrano", "Eastwood", "m@x.org", ["9/Advisor"], {}, 100, 8,
        )
        self.assertEqual("9/Student", pid)
        self.assertEqual("confirmed-coauthor", resolution)

    def test_namesakes_of_one_advisor_are_split_by_dblps_affiliation(self):
        # DBLP has three "Lior Bem" pages and Rea Diaz has published with
        # more than one of them; only one of the three is at UWC.
        dblp_stub = FakeDblp(profiles={
            "9/Advisor": dblp_profile(["Rea Diaz"], {
                "244/0100": "Lior Bem", "244/0100-2": "Lior Bem 0002",
                "244/0100-5": "Lior Bem 0005",
            }),
            "244/0100": dblp_profile(["Lior Bem"], pubs=7),
            "244/0100-2": dblp_profile(
                ["Lior Bem 0002"], pubs=16,
                affiliations=["Institute of Electronic Science and Technology"],
            ),
            "244/0100-5": dblp_profile(
                ["Lior Bem 0005"], pubs=1,
                affiliations=["University of the West Coast, Bayview, CA, USA"],
            ),
        }, searches={"Lior Bem": []})
        pid, resolution, notes = resolve_trc_members.resolve_student_pid(
            dblp_stub, "Lior Bem", "University of the West Coast", "z@x.org",
            ["9/Advisor"], {}, 100, 8,
        )
        self.assertEqual("244/0100-5", pid)
        self.assertEqual("confirmed-coauthor", resolution)
        self.assertIn("recorded affiliation", " ".join(notes))

        # With no affiliation to go on, all three stay in contention.
        pid, resolution, _ = resolve_trc_members.resolve_student_pid(
            dblp_stub, "Lior Bem", "", "z@x.org", ["9/Advisor"], {}, 100, 8,
        )
        self.assertIsNone(pid)
        self.assertEqual("ambiguous", resolution)

    def test_a_declined_pc_member_is_found_by_their_address_alone(self):
        # Declining the invitation leaves every column blank but the address,
        # which is still the HotCRP account a conflict needs.
        unnamed = [("rosalind@blue.ac.kr", "")]
        self.assertEqual(
            ("rosalind@blue.ac.kr", "pc-form-address"),
            resolve_trc_members.resolve_advisor_email(
                "Rosalind Ng", "Blue University", {}, {}, unnamed
            ),
        )
        # The domain has to be their institution too.
        self.assertEqual(
            (None, "not-found"),
            resolve_trc_members.resolve_advisor_email(
                "Rosalind Ng", "Green University", {}, {}, unnamed
            ),
        )
        # A bare surname names half a department, so it names nobody.
        self.assertEqual(
            (None, "not-found"),
            resolve_trc_members.resolve_advisor_email(
                "Rosalind Ng", "Blue University", {}, {}, [("ng@blue.ac.kr", "")]
            ),
        )

    def test_too_many_matching_pages_are_left_for_a_human(self):
        coauthors = {f"9/{i}": "Lee Park" for i in range(12)}
        dblp_stub = FakeDblp(
            profiles={"9/Advisor": dblp_profile(["An Advisor"], coauthors)},
            searches={"Lee Park": []},
        )
        pid, resolution, notes = resolve_trc_members.resolve_student_pid(
            dblp_stub, "Lee Park", "Example University", "w@x.org", ["9/Advisor"], {}, 100, 8,
        )
        self.assertIsNone(pid)
        self.assertEqual("ambiguous", resolution)
        # Bailing out before fetching twelve pages that cannot be told apart.
        self.assertEqual(["9/Advisor"], dblp_stub.fetched)
        self.assertIn("too many to tell apart", " ".join(notes))

    def test_affiliation_tokens_drop_the_words_every_institution_shares(self):
        tokens = resolve_trc_members.affiliation_tokens
        self.assertEqual(frozenset({"west", "coast"}),
                         tokens("University of the West Coast"))
        # A title prefixed to the cell must not become the institution.
        self.assertEqual(tokens("National University of Singapore"),
                         tokens("Professor, National University of Singapore"))
        self.assertFalse(tokens("University") & tokens("Institute of Technology"))

    def test_co_advised_students_name_both_advisors(self):
        self.assertEqual(["Xiaofei Liao", "Hai Jin"],
                         resolve_trc_members.split_advisor_names("Xiaofei Liao (or Hai Jin)"))
        self.assertEqual(["Solo Advisor"],
                         resolve_trc_members.split_advisor_names("Solo Advisor"))


SNAPSHOT_FIXTURE = """<?xml version="1.0" encoding="ISO-8859-1"?>
<!DOCTYPE dblp SYSTEM "dblp-2023-06-28.dtd">
<dblp>
<www key="homepages/1/Alpha">
<author>Fran&ccedil;ois Alpha</author>
<author>F. Alpha</author>
<title>Home Page</title>
</www>
<www key="homepages/2/Beta">
<author>Mei Lam 0001</author>
<title>Home Page</title>
<note type="affiliation">Blue University, Riverton, Portugal</note>
<note type="award">not an affiliation</note>
</www>
<www key="homepages/9/Other">
<author>Nobody Wanted</author>
<title>Home Page</title>
</www>
<inproceedings key="conf/isca/A1">
<author>Fran&ccedil;ois Alpha</author>
<author>Mei Lam 0001</author>
<author>Zoe Outsider</author>
<title>A <i>conference</i> paper.</title>
<booktitle>ISCA</booktitle>
<year>2025</year>
<ee>https://doi.org/10.1145/1234567.8901234</ee>
</inproceedings>
<article key="journals/tc/A2">
<author>F. Alpha</author>
<title>A journal paper.</title>
<journal>IEEE Trans. Computers</journal>
<year>2023</year>
</article>
<inproceedings key="conf/x/Z9">
<author>Nobody Wanted</author>
<title>Not ours.</title>
<booktitle>NOPE</booktitle>
<year>2024</year>
</inproceedings>
<article key="journals/x/NoYear">
<author>Mei Lam 0001</author>
<title>Undated.</title>
<journal>J</journal>
</article>
</dblp>
"""


TRC_FIELDS = [
    "Timestamp", "Email address", "TRC membership", "First Name", "Last Name",
    "Institutional Affiliation", "DBLP Link", "primary area", "keywords",
    "secondary area", "tertiary area", "Advisor's Full Name", "Advisor's Email Address",
]


def write_trc_csv(path, rows):
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TRC_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({field: row.get(field, "") for field in TRC_FIELDS})
    return path


def trc_row(email, **kwargs):
    row = {
        "Timestamp": "1/1/2026 00:00:00",
        "Email address": email,
        "TRC membership": "Yes, I accept as a TRC member",
        "First Name": "Test",
        "Last Name": "Student",
        "Institutional Affiliation": "Somewhere",
        "DBLP Link": "https://dblp.org/pid/1/Test.html",
        "primary area": "Memory Systems",
        "keywords": "",
        "secondary area": "",
        "tertiary area": "",
        "Advisor's Full Name": "",
        "Advisor's Email Address": "",
    }
    row.update(kwargs)
    return row


class TrcReviewersTests(unittest.TestCase):
    """trc_reviewers.py: the TRC acceptance form, its own membership authority."""

    def load(self, rows, pcinfo=None):
        tmp = Path(tempfile.mkdtemp())
        csv_path = write_trc_csv(tmp / "trc.csv", rows)
        pcinfo_path = None
        if pcinfo is not None:
            pcinfo_path = str(write_pcinfo(tmp / "pcinfo.csv", pcinfo))
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            students = trc_reviewers.load_trc_members(
                str(csv_path), pcinfo_path=pcinfo_path
            )
        return students, stderr.getvalue()

    def test_dedupe_keeps_latest_timestamp_row(self):
        rows = [
            trc_row("a@x.edu", Timestamp="1/1/2026 00:00:00", keywords="old"),
            trc_row("a@x.edu", Timestamp="2/1/2026 00:00:00", keywords="new"),
        ]
        students, _ = self.load(rows)
        self.assertEqual(1, len(students))
        self.assertEqual("new", students[0].keywords)

    def test_decline_is_skipped(self):
        rows = [trc_row("a@x.edu", **{"TRC membership": "Unable to accept"})]
        students, _ = self.load(rows)
        self.assertEqual([], students)

    def test_malformed_email_row_is_skipped_not_guessed(self):
        rows = [trc_row('bad"email.edu'), trc_row("good@x.edu")]
        students, stderr = self.load(rows)
        self.assertEqual(["good@x.edu"], [s.email for s in students])
        self.assertIn('bad"email.edu', stderr)

    def test_dblp_link_resolves_to_a_pid(self):
        rows = [trc_row("a@x.edu", **{"DBLP Link": "https://dblp.org/pid/9/Zed.html"})]
        students, _ = self.load(rows)
        self.assertEqual("9/Zed", students[0].pid)

    def test_none_dblp_link_leaves_pid_unset_and_is_reported(self):
        rows = [trc_row("a@x.edu", **{"DBLP Link": "none"})]
        students, stderr = self.load(rows)
        self.assertIsNone(students[0].pid)
        self.assertIn("a@x.edu", stderr)
        self.assertIn("no DBLP link", stderr)

    def test_override_wins_over_forms_own_dblp_link(self):
        tmp = Path(tempfile.mkdtemp())
        overrides = tmp / "trc_dblp_overrides.csv"
        overrides.write_text("email,dblp,note\na@x.edu,7/Over,\n", encoding="utf-8")
        csv_path = write_trc_csv(
            tmp / "trc.csv",
            [trc_row("a@x.edu", **{"DBLP Link": "https://dblp.org/pid/9/Form.html"})],
        )
        students = trc_reviewers.load_trc_members(
            str(csv_path), overrides_path=str(overrides), pcinfo_path=None
        )
        self.assertEqual("7/Over", students[0].pid)
        self.assertTrue(students[0].pid_from_override)

    def test_advisor_emails_split_on_semicolon_and_lowercased(self):
        rows = [trc_row("a@x.edu", **{
            "Advisor's Full Name": "Ada; Bob",
            "Advisor's Email Address": " Ada@X.edu ; bob@x.edu",
        })]
        students, _ = self.load(rows)
        self.assertEqual(("ada@x.edu", "bob@x.edu"), students[0].advisor_emails)

    def test_no_advisor_declared_is_an_empty_tuple(self):
        students, _ = self.load([trc_row("a@x.edu")])
        self.assertEqual((), students[0].advisor_emails)

    def test_cross_check_warns_both_directions_without_pruning(self):
        # The acceptance form is the membership authority for TRC -- unlike
        # every other roster, a tag disagreement must never drop a row.
        # (A real export always has plenty of `pc`-role accounts alongside
        # the `trc`-tagged, non-`pc` ones; load_pc_accounts refuses an
        # export with none at all, so the fixture needs one too.)
        pcinfo = [
            pc_account("a@x.edu", tags="trc", roles=""),
            pc_account("chair@x.edu", roles="pc"),
        ]
        students, stderr = self.load(
            [trc_row("a@x.edu"), trc_row("untagged@x.edu")], pcinfo=pcinfo,
        )
        self.assertEqual(
            {"a@x.edu", "untagged@x.edu"}, {s.email for s in students}
        )
        self.assertIn("untagged@x.edu", stderr)
        self.assertIn("no `trc`-tagged HotCRP account", stderr)

    def test_missing_tag_emails_returns_both_directions_as_data(self):
        # The function a caller (assign_trc_reviews.py's --missing-tag-csv)
        # uses to act on the disagreement, not just read it off stderr.
        tmp = Path(tempfile.mkdtemp())
        pcinfo_path = str(write_pcinfo(tmp / "pcinfo.csv", [
            pc_account("a@x.edu", tags="trc", roles=""),
            pc_account("tagged-not-accepted@x.edu", tags="trc", roles=""),
            pc_account("chair@x.edu", roles="pc"),
        ]))
        students = [
            TrcMember(
                email="a@x.edu", first="A", last="B", dblp_url="", pid=None,
                affiliation="", primary="Memory", secondary="", tertiary="",
                keywords="", tier="trc", override_cap=None,
            ),
            TrcMember(
                email="untagged@x.edu", first="U", last="V", dblp_url="", pid=None,
                affiliation="", primary="Memory", secondary="", tertiary="",
                keywords="", tier="trc", override_cap=None,
            ),
        ]
        with contextlib.redirect_stderr(io.StringIO()):
            not_yet_tagged, not_yet_accepted = trc_reviewers.missing_tag_emails(
                students, pcinfo_path
            )
        self.assertEqual(["untagged@x.edu"], not_yet_tagged)
        self.assertEqual(["tagged-not-accepted@x.edu"], not_yet_accepted)

    def test_missing_tag_emails_degrades_quietly_without_pcinfo(self):
        students = [TrcMember(
            email="a@x.edu", first="A", last="B", dblp_url="", pid=None,
            affiliation="", primary="Memory", secondary="", tertiary="",
            keywords="", tier="trc", override_cap=None,
        )]
        self.assertEqual(([], []), trc_reviewers.missing_tag_emails(students, None))

    def test_tier_is_trc_and_hotcrp_email_defaults_to_email(self):
        students, _ = self.load([trc_row("a@x.edu")])
        self.assertEqual("trc", students[0].tier)
        self.assertEqual("a@x.edu", students[0].hotcrp_email)


class AssignTrcReviewsTests(unittest.TestCase):
    """assign_trc_reviews.py: nearest-neighbor top-N, not deferred acceptance."""

    def test_person_conflict_pids_checks_all_four_layers_independently(self):
        own_by_pid = {1: {"a@x.com"}, 2: set(), 3: {"b@x.com"}}
        pcc_by_pid = {1: set(), 2: {"c@x.com"}, 3: set()}
        coauthor_derived = {2: {"d@x.com": object()}}
        collab_hard = {3: {"e@x.com"}}
        pids = [1, 2, 3, 4]
        check = lambda email: assign_trc_reviews.person_conflict_pids(
            {email}, pids, own_by_pid, pcc_by_pid, coauthor_derived, collab_hard,
        )
        self.assertEqual({1}, check("a@x.com"))
        self.assertEqual({2}, check("c@x.com"))
        self.assertEqual({2}, check("d@x.com"))
        self.assertEqual({3}, check("e@x.com"))
        self.assertEqual(set(), check("nobody@x.com"))

    def test_person_conflict_pids_unions_every_known_address(self):
        # An advisor's known-address set: authorship under either address
        # must be caught, since accepting under one and authoring under
        # another is ordinary here.
        own_by_pid = {1: {"declared@x.com"}, 2: {"alt@x.com"}}
        pids = [1, 2, 3]
        result = assign_trc_reviews.person_conflict_pids(
            {"declared@x.com", "alt@x.com"}, pids, own_by_pid, {}, {}, {},
        )
        self.assertEqual({1, 2}, result)

    def test_eligible_pids_excludes_own_and_advisor_conflicts(self):
        own_by_pid = {1: {"student@x.com"}, 2: {"advisor@x.com"}, 3: set()}
        topics_by_pid = {1: {"memory"}, 2: {"memory"}, 3: {"memory"}}
        pids, relaxed = assign_trc_reviews.eligible_pids_for_student(
            "student@x.com", {"advisor@x.com"}, [1, 2, 3],
            own_by_pid, {}, {}, {}, topics_by_pid, "Memory", "",
            area_gate=True, advisor_coi=True, target=1,
        )
        self.assertEqual([3], pids)
        self.assertFalse(relaxed)

    def test_no_advisor_coi_flag_restores_the_advisors_papers(self):
        own_by_pid = {1: {"advisor@x.com"}}
        topics_by_pid = {1: {"memory"}}
        pids, _ = assign_trc_reviews.eligible_pids_for_student(
            "student@x.com", {"advisor@x.com"}, [1],
            own_by_pid, {}, {}, {}, topics_by_pid, "Memory", "",
            area_gate=True, advisor_coi=False, target=1,
        )
        self.assertEqual([1], pids)

    def test_eligible_pids_excludes_papers_the_advisor_already_reviews(self):
        # Not a conflict -- a plain overlap the student's own review should
        # stay independent of.
        topics_by_pid = {1: {"memory"}, 2: {"memory"}}
        pids, relaxed = assign_trc_reviews.eligible_pids_for_student(
            "student@x.com", {"advisor@x.com"}, [1, 2],
            {}, {}, {}, {}, topics_by_pid, "Memory", "",
            area_gate=True, advisor_coi=True, target=1,
            advisor_assigned_pids=frozenset({1}),
        )
        self.assertEqual([2], pids)
        self.assertFalse(relaxed)

    def test_advisor_review_exclusion_is_never_restored_by_relaxation(self):
        # Hard exclusion, the same as COI everywhere else -- a thin area-gate
        # pool must not resurrect a paper the advisor is already reviewing.
        topics_by_pid = {1: {"memory"}, 2: {"gpus"}}
        pids, relaxed = assign_trc_reviews.eligible_pids_for_student(
            "student@x.com", set(), [1, 2], {}, {}, {}, {}, topics_by_pid,
            "Memory", "", area_gate=True, target=2,
            advisor_assigned_pids=frozenset({2}),
        )
        self.assertTrue(relaxed)
        self.assertEqual([1], pids)

    def test_load_reviewer_assignments_skips_the_wipe_row(self):
        tmp = Path(tempfile.mkdtemp()) / "assignment.csv"
        tmp.write_text(
            "paper,action,email,round\n"
            "all,clearreview,all,R1\n"
            "12,primaryreview,Advisor@X.edu,R1\n"
            "34,primaryreview,advisor@x.edu,R1\n"
            "56,primaryreview,other@x.edu,R1\n",
            encoding="utf-8",
        )
        assigned = assign_trc_reviews.load_reviewer_assignments(str(tmp))
        self.assertEqual({12, 34}, assigned["advisor@x.edu"])
        self.assertEqual({56}, assigned["other@x.edu"])
        self.assertNotIn("all", assigned)

    def test_tag_csv_rows_clear_before_reapplying(self):
        # A paper that drops out of the TRC track on a later run must
        # actually lose the tag, not just never gain a new `tag` row.
        rows = assign_trc_reviews.build_tag_csv_rows([12, 5], "TRC-track")
        self.assertEqual(
            [
                ["all", "cleartag", "", "TRC-track", ""],
                [12, "tag", "", "TRC-track", ""],
                [5, "tag", "", "TRC-track", ""],
            ],
            rows,
        )

    def test_tag_csv_rows_still_clear_when_nothing_is_assigned(self):
        # Every TRC-track paper reassigned away must still be cleared, even
        # if the current run's slate is empty.
        rows = assign_trc_reviews.build_tag_csv_rows([], "TRC-track")
        self.assertEqual([["all", "cleartag", "", "TRC-track", ""]], rows)

    def test_area_gate_relaxes_when_the_strict_pool_is_too_thin(self):
        # A narrow declared interest must not leave a student under-filled
        # when the COI-eligible pool could still cover them.
        own_by_pid = {}
        topics_by_pid = {1: {"memory"}, 2: {"gpus"}, 3: {"security"}}
        pids, relaxed = assign_trc_reviews.eligible_pids_for_student(
            "student@x.com", set(), [1, 2, 3], own_by_pid, {}, {}, {},
            topics_by_pid, "Memory", "", area_gate=True, advisor_coi=True, target=2,
        )
        self.assertTrue(relaxed)
        self.assertEqual([1, 2, 3], pids)

    def test_area_gate_stays_strict_when_the_pool_is_big_enough(self):
        own_by_pid = {}
        topics_by_pid = {1: {"memory"}, 2: {"gpus"}, 3: {"memory"}}
        pids, relaxed = assign_trc_reviews.eligible_pids_for_student(
            "student@x.com", set(), [1, 2, 3], own_by_pid, {}, {}, {},
            topics_by_pid, "Memory", "", area_gate=True, advisor_coi=True, target=2,
        )
        self.assertFalse(relaxed)
        self.assertEqual({1, 3}, set(pids))

    def test_no_area_gate_never_relaxes_because_it_never_gated(self):
        own_by_pid = {}
        topics_by_pid = {1: {"gpus"}}
        pids, relaxed = assign_trc_reviews.eligible_pids_for_student(
            "student@x.com", set(), [1], own_by_pid, {}, {}, {},
            topics_by_pid, "Memory", "", area_gate=False, advisor_coi=True, target=2,
        )
        self.assertFalse(relaxed)
        self.assertEqual([1], pids)

    def test_greedy_assign_caps_a_paper_and_gives_it_to_the_higher_score(self):
        triples = [
            (0.9, "b@x.com", 1),
            (0.8, "a@x.com", 1),
            (0.7, "a@x.com", 2),
        ]
        picks, load = assign_trc_reviews.greedy_assign(triples, 2, 1)
        self.assertEqual([1], [pid for pid, _ in picks["b@x.com"]])
        self.assertEqual([2], [pid for pid, _ in picks["a@x.com"]])
        self.assertEqual(1, load[1])

    def test_greedy_assign_stops_each_student_at_the_target(self):
        triples = [(0.9, "a@x.com", 1), (0.8, "a@x.com", 2), (0.7, "a@x.com", 3)]
        picks, _ = assign_trc_reviews.greedy_assign(triples, 2, 1)
        self.assertEqual(2, len(picks["a@x.com"]))
        self.assertEqual({1, 2}, {pid for pid, _ in picks["a@x.com"]})

    def test_greedy_assign_zero_cap_means_unlimited(self):
        triples = [(0.9, "a@x.com", 1), (0.8, "b@x.com", 1)]
        picks, load = assign_trc_reviews.greedy_assign(triples, 1, 0)
        self.assertEqual([1], [pid for pid, _ in picks["a@x.com"]])
        self.assertEqual([1], [pid for pid, _ in picks["b@x.com"]])
        self.assertEqual(2, load[1])

    def test_resolve_advisor_finds_pid_via_the_pc_roster_directly(self):
        adv = reviewer(pid="5/Adv")
        pc_reviewers_by_email = {adv.email: adv}
        found, addrs = assign_trc_reviews.resolve_advisor(
            adv.email, pc_reviewers_by_email, None
        )
        self.assertIs(adv, found)
        self.assertIn(adv.email, addrs)

    def test_resolve_advisor_falls_back_to_pc_index_for_a_second_address(self):
        # Accepting under one address while authoring under another is
        # ordinary here -- resolve_advisor must find the pid either way.
        adv = reviewer(pid="5/Adv")
        acct = pc_membership.Account({
            "email": adv.email, "given_name": "Test", "family_name": "Person",
            "roles": "pc",
        })
        index = pc_membership.PcIndex([acct], "fake")
        pc_reviewers_by_email = {adv.email: adv}
        found, addrs = assign_trc_reviews.resolve_advisor(
            "other-address@x.com", pc_reviewers_by_email, index
        )
        # The declared address alone resolves nothing by email, but the
        # index has nobody matching it by name/local-part either in this
        # fixture, so no PC reviewer is found -- only the declared address
        # itself is known.
        self.assertIsNone(found)
        self.assertEqual({"other-address@x.com"}, addrs)

    def test_resolve_advisor_with_no_pc_match_keeps_the_declared_address(self):
        found, addrs = assign_trc_reviews.resolve_advisor(
            "external@nowhere.edu", {}, None
        )
        self.assertIsNone(found)
        self.assertEqual({"external@nowhere.edu"}, addrs)


class DblpSnapshotTests(unittest.TestCase):
    """build_dblp_snapshot_cache.py: serving publications from the local dump."""

    def setUp(self):
        self.path = str(Path(tempfile.mkdtemp()) / "dblp.xml")
        Path(self.path).write_text(SNAPSHOT_FIXTURE, encoding="utf-8")

    def extract(self, wanted):
        names = build_dblp_snapshot_cache.collect_names(self.path, set(wanted))
        by_name = {n: pid for pid, spellings in names.items() for n in spellings}
        return names, build_dblp_snapshot_cache.collect_publications(self.path, by_name)

    def test_person_records_carry_the_affiliation_note(self):
        # The notes sit in the records pass 1 already reads, so the region cap
        # gets an offline country source for free. Only type="affiliation"
        # counts, and collect_names still returns the shape its callers expect.
        people = build_dblp_snapshot_cache.collect_person_records(
            self.path, {"1/Alpha", "2/Beta"}
        )
        self.assertEqual(["Blue University, Riverton, Portugal"],
                         people["2/Beta"]["affiliations"])
        self.assertEqual([], people["1/Alpha"]["affiliations"])
        self.assertEqual(["Mei Lam 0001"], people["2/Beta"]["names"])
        self.assertEqual(
            {pid: rec["names"] for pid, rec in people.items()},
            build_dblp_snapshot_cache.collect_names(self.path, {"1/Alpha", "2/Beta"}),
        )

    def test_person_records_decode_entities_and_list_every_spelling(self):
        # The dump declares a DTD it does not ship and uses named HTML
        # entities; a stock parse dies on the first one.
        names, _ = self.extract({"1/Alpha", "2/Beta"})
        self.assertEqual(["François Alpha", "F. Alpha"], names["1/Alpha"])
        self.assertEqual(["Mei Lam 0001"], names["2/Beta"])
        self.assertNotIn("9/Other", names)

    def test_every_spelling_of_a_name_collects_that_persons_papers(self):
        # DBLP records some papers under an alias. Matching only the canonical
        # spelling would silently lose them.
        _, pubs = self.extract({"1/Alpha"})
        self.assertEqual({"A conference paper.", "A journal paper."},
                         {p["title"] for p in pubs["1/Alpha"]})

    def test_records_match_the_shape_the_live_fetch_returns(self):
        _, pubs = self.extract({"1/Alpha", "2/Beta"})
        conference = next(p for p in pubs["1/Alpha"] if p["venue"] == "ISCA")
        self.assertEqual(
            {"title": "A conference paper.", "year": 2025, "venue": "ISCA",
             "type": "inproceedings", "doi": "10.1145/1234567.8901234"},
            conference,
        )
        # Venue is the booktitle for a conference and the journal for an
        # article, exactly as _fetch_all_records_from_dblp decides it.
        journal = next(p for p in pubs["1/Alpha"] if p["type"] == "article")
        self.assertEqual("IEEE Trans. Computers", journal["venue"])
        self.assertEqual("", journal["doi"])
        # A co-authored paper belongs to both of them.
        self.assertIn("A conference paper.", {p["title"] for p in pubs["2/Beta"]})
        # Year-descending, as the live path sorts.
        years = [p["year"] for p in pubs["1/Alpha"]]
        self.assertEqual(sorted(years, reverse=True), years)

    def test_unwanted_people_undated_records_and_home_pages_are_skipped(self):
        _, pubs = self.extract({"1/Alpha", "2/Beta"})
        self.assertNotIn("9/Other", pubs)
        # A record with no year cannot be windowed, so it is dropped.
        self.assertNotIn("Undated.", {p["title"] for p in pubs["2/Beta"]})
        # A person record lists its owner as an author; it is not a publication.
        self.assertNotIn("Home Page", {p["title"] for p in pubs["2/Beta"]})

    def test_snapshot_only_fills_gaps_it_never_replaces_a_cached_pid(self):
        # The fingerprint key includes the publication list, so re-sourcing an
        # already-cached person would invalidate and re-embed them.
        snapshot = {"a": ["snap"], "b": ["snap"], "c": ["snap"]}
        gaps = dblp.snapshot_gaps(snapshot, {"a": ["colleague"]}, {"b": ["ours"]})
        self.assertEqual({"c": ["snap"]}, gaps)
        merged = {**gaps, **{"a": ["colleague"]}}
        self.assertEqual(["colleague"], merged["a"])

    def coauthors(self, wanted, cutoff=None):
        names = build_dblp_snapshot_cache.collect_names(self.path, set(wanted))
        by_name = {n: pid for pid, spellings in names.items() for n in spellings}
        found = {}
        build_dblp_snapshot_cache.collect_publications(
            self.path, by_name, coauthors=found, coauthor_cutoff=cutoff
        )
        return found

    def test_coauthors_are_collected_without_a_second_pass(self):
        found = self.coauthors({"1/Alpha", "2/Beta"})
        self.assertEqual({"Mei Lam 0001", "Zoe Outsider"}, set(found["1/Alpha"]))
        self.assertEqual([[2025, "A conference paper."]], found["1/Alpha"]["Zoe Outsider"])

    def test_an_alias_of_the_owner_is_not_one_of_their_coauthors(self):
        # "F. Alpha" is 1/Alpha's own second spelling. Matching on the name
        # string alone would make them their own co-author on every paper.
        found = self.coauthors({"1/Alpha", "2/Beta"})
        self.assertNotIn("F. Alpha", found["1/Alpha"])
        self.assertNotIn("François Alpha", found["1/Alpha"])

    def test_a_coauthor_outside_the_cutoff_is_not_kept(self):
        self.assertEqual({}, self.coauthors({"1/Alpha", "2/Beta"}, cutoff=2026))

    def test_collecting_coauthors_does_not_change_the_publication_records(self):
        # The record shape is a contract with the live fetch and with the
        # fingerprint key; the co-author sink must stay strictly beside it.
        names = build_dblp_snapshot_cache.collect_names(self.path, {"1/Alpha"})
        by_name = {n: pid for pid, spellings in names.items() for n in spellings}
        plain = build_dblp_snapshot_cache.collect_publications(self.path, by_name)
        withco = build_dblp_snapshot_cache.collect_publications(
            self.path, by_name, coauthors={}, coauthor_cutoff=2000
        )
        self.assertEqual(plain, withco)


class BackoffTests(unittest.TestCase):
    """dblp.py's retry and pacing behaviour under a hostile server."""

    def test_backoff_restarts_each_call_and_is_capped(self):
        # The doubling is indexed by attempt-within-a-call, so it cannot ratchet
        # upward across a run; and the cap has to survive the jitter, which is
        # applied after it.
        first = [dblp._backoff_delay(15, 0) for _ in range(50)]
        self.assertTrue(all(15 <= d <= 15 * 1.25 for d in first))
        for attempt in range(10):
            delays = [dblp._backoff_delay(15, attempt) for _ in range(50)]
            self.assertTrue(all(15 <= d <= dblp.MAX_BACKOFF for d in delays),
                            f"attempt {attempt} escaped [floor, MAX_BACKOFF]")

    def test_retry_after_is_honoured_but_capped(self):
        # A server may name any delay it likes. An uncapped one is a silent hour
        # that looks exactly like a hang.
        session = mock.Mock()
        limited = mock.Mock(status_code=429, headers={"Retry-After": "86400"})
        ok = mock.Mock(status_code=200, headers={})
        ok.raise_for_status = mock.Mock()
        session.get.side_effect = [limited, ok]
        with mock.patch.object(dblp.time, "sleep") as slept, \
                contextlib.redirect_stderr(io.StringIO()):
            dblp.get_with_retry(session, "https://dblp.org/pid/1/A.xml")
        self.assertEqual(1, slept.call_count)
        self.assertEqual(dblp.MAX_BACKOFF, slept.call_args[0][0])

    def test_server_busy_statuses_back_off_instead_of_raising(self):
        # DBLP answers 503 when shedding load. Raising on it immediately meant a
        # batch burned its whole consecutive-failure budget in seconds without
        # ever pausing -- the opposite of what the server asked for.
        session = mock.Mock()
        busy = mock.Mock(status_code=503, headers={})
        ok = mock.Mock(status_code=200, headers={})
        ok.raise_for_status = mock.Mock()
        session.get.side_effect = [busy, busy, ok]
        with mock.patch.object(dblp.time, "sleep") as slept, \
                contextlib.redirect_stderr(io.StringIO()):
            got = dblp.get_with_retry(session, "https://dblp.org/pid/1/A.xml")
        self.assertIs(ok, got)
        self.assertEqual(2, slept.call_count)
        self.assertTrue(all(c[0][0] >= 15 for c in slept.call_args_list))

    def test_a_failed_fetch_still_paces_the_next_one(self):
        # The bug this guards: clearing the "last request was live" flag on
        # failure sent the very next request out with no delay, at exactly the
        # moment the server had demonstrated it wanted fewer of them.
        calls = []

        def fetch(pid, **kwargs):
            calls.append(pid)
            if pid == "1/A":
                raise requests.ConnectionError("reset by peer")
            return [(2026, "A title")], "live"

        with mock.patch.object(dblp, "fetch_titles", side_effect=fetch), \
                mock.patch.object(dblp.time, "sleep") as slept:
            dblp.fetch_titles_for_pids(["1/A", "2/B"], delay=3.0, on_error=lambda p, e: None)
        self.assertEqual(["1/A", "2/B"], calls)
        self.assertTrue(slept.called, "no pacing delay before the PID after a failure")

    def test_repeated_failures_stop_rather_than_grind(self):
        # DBLP drops connections when it has blocked an IP; retrying cannot
        # clear that, so the loop must give up and keep what it cached.
        def always_fail(pid, **kwargs):
            raise requests.ConnectionError("reset by peer")

        errors = []
        with mock.patch.object(dblp, "fetch_titles", side_effect=always_fail), \
                mock.patch.object(dblp.time, "sleep"):
            with self.assertRaises(RuntimeError) as caught:
                dblp.fetch_titles_for_pids(
                    [f"{i}/X" for i in range(50)], delay=3.0,
                    on_error=lambda p, e: errors.append(p),
                )
        self.assertIn("consecutive", str(caught.exception))
        self.assertEqual(dblp.MAX_CONSECUTIVE_FAILURES, len(errors))

    def test_an_intermittent_failure_does_not_stop_the_run(self):
        # Only *consecutive* failures mean a block; a success resets the count.
        def flaky(pid, **kwargs):
            if pid.startswith(("0", "2", "4")):
                raise requests.ConnectionError("blip")
            return [(2026, "A title")], "live"

        with mock.patch.object(dblp, "fetch_titles", side_effect=flaky), \
                mock.patch.object(dblp.time, "sleep"):
            results = dblp.fetch_titles_for_pids(
                [f"{i}/X" for i in range(10)], delay=3.0, on_error=lambda p, e: None,
            )
        # 0, 2 and 4 fail; the other seven get through.
        self.assertEqual(7, len(results))


class ReserveReviewerLoaderTests(unittest.TestCase):
    """reserve_reviewers.py: a roster with no acceptance form behind it."""

    def load(self, roster, papers, pcinfo=None, cap_overrides=None):
        tmp = Path(tempfile.mkdtemp())
        info, data = tmp / "info.csv", tmp / "papers.json"
        with info.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["email", "name", "dblp"])
            w.writeheader()
            for email, name, link in roster:
                w.writerow({"email": email, "name": name, "dblp": link})
        data.write_text(json.dumps(papers), encoding="utf-8")
        # pcinfo=None means "don't check membership", not "check against the real
        # export" -- a unit test must never depend on today's HotCRP roster.
        path = None
        if pcinfo is not None:
            path = str(write_pcinfo(tmp / "pcinfo.csv", pcinfo))
        cap_overrides_path = tmp / "reviewer_cap_overrides.csv"
        with cap_overrides_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["email", "cap", "note"])
            writer.writeheader()
            for email, cap in (cap_overrides or {}).items():
                writer.writerow({"email": email, "cap": cap, "note": ""})
        return reserve_reviewers.load_reserve_reviewers(
            str(info), str(data), pcinfo_path=path, cap_overrides_path=str(cap_overrides_path)
        )

    def paper(self, topics, authors=(), nominates=""):
        return {"pid": 1, "topics": list(topics), "reserve_reviewer": nominates,
                "authors": [{"email": e, "given_name": "A", "family_name": "B",
                             "affiliation": "Somewhere"} for e in authors],
                "contacts": []}

    def test_reserve_tier_gets_its_own_cap_not_the_full_one(self):
        # reviewer_paper_cap returned full_cap for any tier that wasn't 'light',
        # so a reserve entering the pool would silently be handed a PC load.
        def rv(tier, override=None):
            return Reviewer(
                email="e@x.edu", first="A", last="B", dblp_url="", pid=None,
                affiliation="", primary="", secondary="", tertiary="", keywords="",
                tier=tier, override_cap=override,
            )

        cap = assign_reviewers.reviewer_paper_cap
        self.assertEqual(15, cap(rv("full"), 7, 15))
        self.assertEqual(7, cap(rv("light"), 7, 15))
        self.assertEqual(assign_reviewers.DEFAULT_RESERVE_CAP, cap(rv("reserve"), 7, 15))
        self.assertEqual(2, cap(rv("reserve"), 7, 15, 2))
        # An explicit per-person override still wins over the tier default.
        self.assertEqual(9, cap(rv("reserve", 9), 7, 15))

    def test_areas_come_from_the_papers_they_authored(self):
        # The gate reads primary and secondary, so the two most frequent topics
        # have to land there -- and in frequency order, not paper order.
        reserves = self.load(
            [("a@x.edu", "Ada Lovelace", "https://dblp.org/pid/1/A.html")],
            [self.paper(["Memory Systems", "GPUs"], ["a@x.edu"]),
             self.paper(["Memory Systems"], ["a@x.edu"]),
             self.paper(["Memory Systems", "GPUs"], ["a@x.edu"]),
             self.paper(["Security"], ["someone.else@x.edu"])],
        )
        self.assertEqual(1, len(reserves))
        r = reserves[0]
        self.assertEqual("Memory Systems", r.primary)
        self.assertEqual("GPUs", r.secondary)
        self.assertEqual("reserve", r.tier)
        self.assertEqual("1/A", r.pid)
        self.assertIsNone(r.override_cap)
        # Somebody else's topic must not leak in.
        self.assertNotIn("Security", (r.primary, r.secondary, r.tertiary))

    def test_nomination_topics_are_the_fallback_for_a_non_author(self):
        # A reserve who wrote nothing is still characterised by the papers that
        # nominated them, rather than being left area-less and gated out.
        reserves = self.load(
            [("n@x.edu", "Nom Inee", "https://dblp.org/pid/2/N.html")],
            [self.paper(["Quantum Computing"], [],
                        nominates="Nom Inee / Somewhere / n@x.edu")],
        )
        self.assertEqual("Quantum Computing", reserves[0].primary)

    def test_ties_break_alphabetically_so_areas_are_reproducible(self):
        # Counter.most_common leaves ties in insertion order, which follows the
        # order papers happen to appear in the export -- and every fingerprint
        # built from these areas would shift with it.
        forward = self.load(
            [("a@x.edu", "Ada", "https://dblp.org/pid/1/A.html")],
            [self.paper(["Zebra", "Alpha"], ["a@x.edu"])],
        )
        backward = self.load(
            [("a@x.edu", "Ada", "https://dblp.org/pid/1/A.html")],
            [self.paper(["Alpha", "Zebra"], ["a@x.edu"])],
        )
        self.assertEqual("Alpha", forward[0].primary)
        self.assertEqual(forward[0].primary, backward[0].primary)
        self.assertEqual(forward[0].secondary, backward[0].secondary)

    def test_derived_areas_satisfy_the_real_area_gate(self):
        # The whole point: these areas must pass paper_matching's own gate. The
        # paper they are scored against has to be somebody else's — the paper
        # their areas were *derived* from is one they wrote, and so is
        # conflicted out before the gate is ever consulted.
        own = self.paper(["Memory Systems", "GPUs"], ["a@x.edu"])
        someone_elses = self.paper(["Memory Systems", "GPUs"], ["other@x.edu"])
        r = self.load([("a@x.edu", "Ada", "https://dblp.org/pid/1/A.html")], [own])[0]
        args = ([r.email], np.ones((1, 3), dtype=np.float32),
                np.ones(3, dtype=np.float32), {r.email: r})

        scored = paper_matching.eligible_scores(someone_elses, *args, area_gate=True)
        self.assertEqual([r.email], [e for e, _ in scored])

        # And the conflict outranks the area match on their own paper.
        self.assertEqual([], paper_matching.eligible_scores(own, *args, area_gate=True))

    def test_a_curated_cap_override_reduces_a_reserves_load(self):
        # Reserves never fill in an acceptance form, so reviewer_cap_overrides.csv
        # is the only way to reduce one below the reserve cap at all.
        reserves = self.load(
            [("a@x.edu", "Ada Lovelace", "https://dblp.org/pid/1/A.html")],
            [self.paper(["Memory Systems"], ["a@x.edu"])],
            cap_overrides={"a@x.edu": "2"},
        )
        self.assertEqual(2, reserves[0].override_cap)

    def test_with_no_curated_override_a_reserve_has_none(self):
        reserves = self.load(
            [("a@x.edu", "Ada Lovelace", "https://dblp.org/pid/1/A.html")],
            [self.paper(["Memory Systems"], ["a@x.edu"])],
        )
        self.assertIsNone(reserves[0].override_cap)


class OwnPaperConflictTests(unittest.TestCase):
    """paper_matching.own_paper_conflicts: the COI floor under pc_conflicts."""

    def paper(self, **kwargs):
        base = {"pid": 1, "topics": ["Memory Systems"], "authors": [], "contacts": [],
                "pc_conflicts": {}, "reserve_reviewer": ""}
        base.update(kwargs)
        return base

    def test_authors_and_contacts_conflict_even_with_no_declared_coi(self):
        # Three reserve reviewers currently have no conflict recorded anywhere
        # while authoring 18 submissions; without this the matcher hands them
        # their own papers.
        p = self.paper(
            authors=[{"email": "Author@dept.univ-a.edu"}],
            contacts=[{"email": "contact@x.edu"}],
        )
        self.assertEqual({"author@dept.univ-a.edu", "contact@x.edu"},
                         paper_matching.own_paper_conflicts(p))

    def test_a_nominated_reserve_reviewer_conflicts_with_the_nominating_paper(self):
        # HPCA's nomination field names a senior author of that same paper.
        p = self.paper(reserve_reviewer="Ada Lovelace / Somewhere / ada@x.edu\n"
                                        "Bob Smith / Elsewhere / bob@x.edu")
        self.assertEqual({"ada@x.edu", "bob@x.edu"},
                         paper_matching.own_paper_conflicts(p))

    def test_it_only_adds_to_the_declared_conflicts_never_replaces_them(self):
        r = reviewer()
        p = self.paper(pid=2, authors=[{"email": "someone@else.edu"}],
                       pc_conflicts={r.email: 1})
        args = ([r.email], np.ones((1, 3), dtype=np.float32),
                np.ones(3, dtype=np.float32), {r.email: r})
        # Declared conflict still excludes them, though they wrote nothing.
        self.assertEqual([], paper_matching.eligible_scores(p, *args, area_gate=False))

    def test_pc_conflicts_declared_against_the_hotcrp_email_still_excludes(self):
        # HotCRP's own declared conflicts are keyed by whatever address the
        # account actually has -- hotcrp_email -- not necessarily the roster
        # key candidate_emails uses. Missing this let a real declared
        # conflict through for anyone pc_membership matched under a second
        # address, caught only by HotCRP itself, at upload, as a hard error.
        r = reviewer_with_email("form@a.edu", hotcrp_email="real@b.edu")
        p = self.paper(pid=2, authors=[{"email": "someone@else.edu"}],
                       pc_conflicts={"real@b.edu": 1})
        args = ([r.email], np.ones((1, 3), dtype=np.float32),
                np.ones(3, dtype=np.float32), {r.email: r})
        self.assertEqual([], paper_matching.eligible_scores(p, *args, area_gate=False))

    def test_a_malformed_nomination_line_is_ignored_not_guessed_at(self):
        p = self.paper(reserve_reviewer="Just A Name\nNo Email / Somewhere\n")
        self.assertEqual(set(), paper_matching.own_paper_conflicts(p))

    def test_a_derived_conflict_excludes_just_as_a_declared_one_does(self):
        # The hook the co-author layer enters through. Nothing downstream in
        # assign_reviewers re-checks COI, so if this keyword stops being
        # honoured the layer silently becomes a report and nothing more.
        r = reviewer()
        p = self.paper(pid=3, authors=[{"email": "someone@else.edu"}])
        args = ([r.email], np.ones((1, 3), dtype=np.float32),
                np.ones(3, dtype=np.float32), {r.email: r})
        self.assertEqual([r.email], [
            e for e, _ in paper_matching.eligible_scores(p, *args, area_gate=False)
        ])
        self.assertEqual([], paper_matching.eligible_scores(
            p, *args, area_gate=False, extra_conflicts={r.email.upper()}
        ))


class CoauthorNameMatchTests(unittest.TestCase):
    """coauthor_coi.names_match: who counts as the same person, and who doesn't."""

    def match(self, one, other):
        return coauthor_coi.names_match(
            pc_membership.token_set(one), pc_membership.token_set(other)
        )

    def test_the_same_name_matches_exactly(self):
        self.assertEqual("exact", self.match("Onur Mutlu", "onur mutlu"))

    def test_a_spelled_out_middle_name_still_matches(self):
        # DBLP writes people out in full more often than HotCRP does; requiring
        # equality here would miss the conflict entirely.
        self.assertEqual("partial", self.match("David Albert Wood", "David Wood"))

    def test_an_initial_is_dropped_rather_than_treated_as_a_name(self):
        self.assertEqual("exact", self.match("Matthew D. Sinclair", "Matthew Sinclair"))

    def test_dblps_homonym_suffix_is_not_part_of_the_name(self):
        # Deliberately over-matching: two different people share this token set,
        # and conflating them withholds a reviewer rather than missing a COI.
        self.assertEqual("exact", self.match("Wei Zhang 0001", "Wei Zhang 0025"))

    def test_accents_do_not_split_a_name_in_two(self):
        self.assertEqual("exact", self.match("François Alpha", "Francois Alpha"))

    def test_a_name_written_family_first_still_matches(self):
        self.assertEqual("exact", self.match("Lam Mei", "Mei Lam"))

    def test_one_shared_token_is_a_coincidence_not_a_person(self):
        self.assertIsNone(self.match("Wei Zhang", "Wei Chen"))
        self.assertIsNone(self.match("Onur Mutlu", "Onur Kayiran"))

    def test_two_names_that_merely_overlap_do_not_match(self):
        # Overlap is not containment: neither is a subset of the other, so this
        # stays out even though two tokens are shared.
        self.assertIsNone(self.match("Wei Zhang Chen", "Wei Zhang Liu"))

    def test_a_short_name_inside_a_longer_one_matches_on_purpose(self):
        # The cost of this rule. It is what catches a middle name spelled out,
        # and it also marries a short name to an unrelated longer one that
        # happens to contain it. Erring this way withholds a reviewer; erring
        # the other way hands them a paper they are conflicted on.
        self.assertEqual("partial", self.match("Jing Li", "Jing Li Wang Chen"))

    def test_a_single_token_name_never_matches_anything(self):
        # A lone surname would conflict a reviewer with every namesake alive.
        self.assertIsNone(self.match("Mutlu", "Onur Mutlu"))
        self.assertIsNone(self.match("Mutlu", "Mutlu"))


class CoauthorConflictTests(unittest.TestCase):
    """coauthor_coi: conflicts DBLP implies that nobody declared."""

    def setUp(self):
        self.reviewer = Reviewer(
            email="rev@x.edu", first="Ada", last="Lovelace", dblp_url="",
            pid="1/Ada", affiliation="X", primary="Memory", secondary="",
            tertiary="", keywords="", tier="full", override_cap=None,
        )
        self.coauthors = {
            "1/Ada": {
                "Charles Babbage": [[2025, "An engine."], [2023, "Another engine."]],
                "Old Colleague": [[2004, "Long ago."]],
                "Ada Lovelace": [[2025, "An engine."]],
            }
        }

    def paper(self, authors, **kwargs):
        base = {
            "pid": 7, "title": "A paper.", "topics": [], "contacts": [],
            "pc_conflicts": {}, "reserve_reviewer": "",
            "authors": [
                {"given_name": g, "family_name": f, "email": e, "affiliation": "Y"}
                for g, f, e in authors
            ],
        }
        base.update(kwargs)
        return base

    def derive(self, paper, *, years=5, author_names=None):
        index = coauthor_coi.build_index(
            [self.reviewer], self.coauthors, years=years, current_year=2026
        )
        return index, coauthor_coi.derive_conflicts(
            [paper], index, author_names or {}
        )

    def test_nonpositive_window_is_rejected_by_core_index(self):
        for years in (0, -1):
            with self.subTest(years=years), self.assertRaisesRegex(
                ValueError, "greater than 0"
            ):
                coauthor_coi.build_index(
                    [self.reviewer], self.coauthors, years=years, current_year=2026
                )

    def test_a_recent_coauthor_on_a_paper_conflicts_the_reviewer(self):
        p = self.paper([("Charles", "Babbage", "cb@y.edu")])
        _, derived = self.derive(p)
        coi = derived[7]["rev@x.edu"]
        self.assertEqual(("exact", 2, 2025, ""), (coi.match, coi.shared, coi.latest_year, coi.declared))
        self.assertEqual("cb@y.edu", coi.author_email)

    def test_a_coauthor_outside_the_window_does_not_conflict(self):
        # dblp.filter_by_years' convention: five years in 2026 starts at 2022,
        # so a 2004 paper is out and nothing fires.
        p = self.paper([("Old", "Colleague", "oc@y.edu")])
        self.assertEqual({}, self.derive(p)[1])

    def test_widening_the_window_brings_the_old_coauthor_back(self):
        p = self.paper([("Old", "Colleague", "oc@y.edu")])
        self.assertIn("rev@x.edu", self.derive(p, years=25)[1][7])

    def test_a_reviewer_is_not_their_own_coauthor(self):
        # DBLP lists the owner among the authors of their own papers; without
        # this every reviewer conflicts with every paper a namesake wrote.
        p = self.paper([("Ada", "Lovelace", "someone.else@y.edu")])
        _, derived = self.derive(p)
        self.assertNotIn(7, derived)

    def test_a_declared_conflict_is_marked_as_confirmation_not_news(self):
        p = self.paper([("Charles", "Babbage", "cb@y.edu")],
                       pc_conflicts={"REV@x.edu": True})
        self.assertEqual("pc_conflicts", self.derive(p)[1][7]["rev@x.edu"].declared)

    def test_a_reviewer_on_the_paper_is_marked_own_paper(self):
        p = self.paper([("Charles", "Babbage", "cb@y.edu"),
                        ("Ada", "L", "rev@x.edu")])
        self.assertEqual("own_paper", self.derive(p)[1][7]["rev@x.edu"].declared)

    def test_an_author_is_matched_under_their_dblp_spelling_too(self):
        # The author calls themselves "C. Babbage" in HotCRP; DBLP knows them as
        # "Charles Babbage". Their own declared DBLP link is what bridges the two.
        p = self.paper([("C.", "Babbage", "cb@y.edu")],
                       dblp="https://dblp.org/pid/9/CB.html")
        self.assertEqual({}, self.derive(p)[1])
        _, derived = self.derive(p, author_names={"9/CB": ["Charles Babbage"]})
        self.assertEqual("exact", derived[7]["rev@x.edu"].match)

    def test_a_reviewer_with_no_snapshot_entry_is_reported_not_silently_clean(self):
        # The false negative that matters: no co-author data means every check
        # passes, which is not the same as having no conflicts.
        self.reviewer.pid = "404/Missing"
        index, derived = self.derive(self.paper([("Charles", "Babbage", "cb@y.edu")]))
        self.assertEqual({}, derived)
        self.assertEqual({"rev@x.edu"}, index.uncovered)
        self.assertEqual([self.reviewer], coauthor_coi.coverage_gap(index, [self.reviewer]))

    def test_contacts_count_as_authors_for_this(self):
        p = self.paper([("Someone", "Else", "se@y.edu")],
                       contacts=[{"given_name": "Charles", "family_name": "Babbage",
                                  "email": "cb@y.edu", "affiliation": "Y"}])
        self.assertIn("rev@x.edu", self.derive(p)[1][7])


class CollaboratorLineParsingTests(unittest.TestCase):
    """collaborator_coi.parse_collaborator_lines and institution_tokens."""

    def test_a_named_line_splits_name_from_trailing_institution(self):
        self.assertEqual(
            [("Ada Lovelace", "Analytical Engines Ltd")],
            collaborator_coi.parse_collaborator_lines("Ada Lovelace (Analytical Engines Ltd)"),
        )

    def test_a_line_with_no_institution_has_an_empty_one(self):
        self.assertEqual([("Ada Lovelace", "")],
                         collaborator_coi.parse_collaborator_lines("Ada Lovelace"))

    def test_blank_lines_are_skipped(self):
        self.assertEqual(
            [("Ada Lovelace", "")],
            collaborator_coi.parse_collaborator_lines("\nAda Lovelace\n\n"),
        )

    def test_all_marks_an_institution_wide_line(self):
        self.assertTrue(collaborator_coi.is_institution_wide("All"))
        self.assertTrue(collaborator_coi.is_institution_wide(" all "))
        self.assertFalse(collaborator_coi.is_institution_wide("Alla Smith"))

    def test_institution_tokens_drop_stopwords_and_short_words(self):
        self.assertEqual(
            {"shenzhen"},
            collaborator_coi.institution_tokens("Shenzhen University"),
        )
        self.assertEqual(set(), collaborator_coi.institution_tokens("The Institute of Technology"))


class CollaboratorConflictTests(unittest.TestCase):
    """collaborator_coi: conflicts HotCRP's own declared-collaborator warning catches."""

    def profiles(self, people, accounts):
        tmp = Path(tempfile.mkdtemp())
        index = pc_membership.load_pc_accounts(str(write_pcinfo(tmp / "pc.csv", accounts)))
        return collaborator_coi.build_index(people, index), index

    def paper(self, authors, **kwargs):
        base = {"pid": 9, "title": "A paper.", "topics": [], "contacts": [],
                "pc_conflicts": {}, "reserve_reviewer": "",
                "authors": [
                    {"given_name": g, "family_name": f, "email": e, "affiliation": a}
                    for g, f, e, a in authors
                ]}
        base.update(kwargs)
        return base

    def test_a_reviewers_declared_collaborator_matches_a_paper_author(self):
        r = reviewer_with_email("rev@x.edu")
        profiles, index = self.profiles(
            [r], [pc_account("rev@x.edu", "Ada", "Lovelace",
                              collaborators="Charles Babbage (Difference Engines)")],
        )
        p = self.paper([("Charles", "Babbage", "cb@y.edu", "Difference Engines")])
        conflict = collaborator_coi.derive_conflicts([p], profiles, index)[9]["rev@x.edu"]
        self.assertEqual(("name", "reviewer_declared"), (conflict.kind, conflict.direction))

    def test_an_authors_declared_collaborator_matches_the_reviewer(self):
        # The direction that matters most: the reviewer's own account may say
        # nothing, but the author already named them.
        r = reviewer_with_email("rev@x.edu")  # name "Test Person", see reviewer_with_email
        profiles, index = self.profiles(
            [r],
            [pc_account("rev@x.edu", "Ada", "Lovelace"),
             pc_account("cb@y.edu", "Charles", "Babbage",
                        collaborators="Test Person (Analytical Engines)")],
        )
        p = self.paper([("Charles", "Babbage", "cb@y.edu", "Difference Engines")])
        conflict = collaborator_coi.derive_conflicts([p], profiles, index)[9]["rev@x.edu"]
        self.assertEqual(("name", "author_declared"), (conflict.kind, conflict.direction))

    def test_a_bare_all_line_never_matches_a_person_by_name(self):
        r = reviewer_with_email("rev@x.edu")
        profiles, index = self.profiles(
            [r], [pc_account("rev@x.edu", "Ada", "Lovelace", collaborators="All")],
        )
        p = self.paper([("All", "", "weird@y.edu", "")])
        self.assertEqual({}, collaborator_coi.derive_conflicts([p], profiles, index))

    def test_reviewers_own_affiliation_overlaps_an_authors(self):
        r = reviewer_with_email("rev@x.edu")
        profiles, index = self.profiles(
            [r], [pc_account("rev@x.edu", "Ada", "Lovelace", affiliation="Shenzhen University")],
        )
        p = self.paper([("Ruiheng", "Wang", "rw@y.edu",
                         "School of Engineering, Peking University, Shenzhen")])
        conflict = collaborator_coi.derive_conflicts([p], profiles, index)[9]["rev@x.edu"]
        self.assertEqual(("affiliation", "institution", "shenzhen"),
                         (conflict.kind, conflict.direction, conflict.evidence))

    def test_an_institution_wide_all_line_conflicts_every_author_there(self):
        r = reviewer_with_email("rev@x.edu")
        profiles, index = self.profiles(
            [r], [pc_account("rev@x.edu", "Ada", "Lovelace",
                              collaborators="All (Google Quantum AI)")],
        )
        p = self.paper([("Jingbo", "Wang", "jw@y.edu",
                         "Beijing Academy of Quantum Information Sciences")])
        conflict = collaborator_coi.derive_conflicts([p], profiles, index)[9]["rev@x.edu"]
        self.assertEqual("affiliation", conflict.kind)
        self.assertEqual("quantum", conflict.evidence)

    def test_no_shared_signal_is_not_a_conflict(self):
        r = reviewer_with_email("rev@x.edu")
        profiles, index = self.profiles(
            [r], [pc_account("rev@x.edu", "Ada", "Lovelace", affiliation="MIT")],
        )
        p = self.paper([("Grace", "Hopper", "gh@y.edu", "Yale University")])
        self.assertEqual({}, collaborator_coi.derive_conflicts([p], profiles, index))

    def test_hard_conflicts_keeps_only_the_name_kind(self):
        # The calibration this whole layer landed on: affiliation overlap
        # (measured at 556/688 reviewers, 963/1156 papers on real data,
        # dominated by generic words) is reported but never excludes: only a
        # name match does.
        r = reviewer_with_email("rev@x.edu")
        profiles, index = self.profiles(
            [r], [pc_account("rev@x.edu", "Ada", "Lovelace", affiliation="MIT",
                             collaborators="Charles Babbage (Difference Engines)")],
        )
        p1 = self.paper([("Someone", "AtMIT", "sm@y.edu", "MIT")], pid=1)
        p2 = self.paper([("Charles", "Babbage", "cb@y.edu", "Difference Engines")], pid=2)
        derived = collaborator_coi.derive_conflicts([p1, p2], profiles, index)
        self.assertEqual("affiliation", derived[1]["rev@x.edu"].kind)
        self.assertEqual("name", derived[2]["rev@x.edu"].kind)
        hard = collaborator_coi.hard_conflicts(derived)
        self.assertEqual({"rev@x.edu"}, hard.get(2))
        self.assertEqual(set(), hard.get(1, set()))

    def test_a_name_match_is_reported_over_an_affiliation_match_on_the_same_paper(self):
        # Two authors on one paper: one shares only an affiliation word, the
        # other is a declared collaborator by name. The report should show
        # the higher-precision reason regardless of author order.
        r = reviewer_with_email("rev@x.edu")
        profiles, index = self.profiles(
            [r], [pc_account("rev@x.edu", "Ada", "Lovelace", affiliation="MIT",
                              collaborators="Charles Babbage (Difference Engines)")],
        )
        p = self.paper([
            ("Someone", "AtMIT", "sm@y.edu", "MIT"),
            ("Charles", "Babbage", "cb@y.edu", "Difference Engines"),
        ])
        conflict = collaborator_coi.derive_conflicts([p], profiles, index)[9]["rev@x.edu"]
        self.assertEqual("name", conflict.kind)

    def test_hotcrp_email_not_roster_email_supplies_the_account_signals(self):
        # Same reasoning as the identity fixes elsewhere: the collaborators
        # and affiliation live on the account pc_membership resolved to, not
        # necessarily the roster row's own address.
        r = reviewer_with_email("form@x.edu", hotcrp_email="real@y.edu")
        profiles, index = self.profiles(
            [r], [pc_account("real@y.edu", "Ada", "Lovelace",
                              collaborators="Charles Babbage (Difference Engines)")],
        )
        p = self.paper([("Charles", "Babbage", "cb@y.edu", "Difference Engines")])
        conflict = collaborator_coi.derive_conflicts([p], profiles, index)[9]["form@x.edu"]
        self.assertEqual("name", conflict.kind)


class CoauthorIdentityTests(unittest.TestCase):
    """coauthor_coi: DBLP's homonym numbering, where both sides are known.

    "Wei Zhang" is 24 different researchers among this roster's co-authors.
    Collapsing them conflicts a third of the committee with any paper one of
    them wrote, and the collisions land almost entirely on names that romanise
    into a small space.
    """

    def setUp(self):
        # The reviewer wrote with Wei Zhang 0001, and with nobody else.
        self.reviewer = Reviewer(
            email="rev@x.edu", first="Ada", last="Lovelace", dblp_url="",
            pid="1/Ada", affiliation="X", primary="Memory", secondary="",
            tertiary="", keywords="", tier="full", override_cap=None,
        )
        self.coauthors = {"1/Ada": {"Wei Zhang 0001": [[2025, "A paper."]]}}

    def paper(self, dblp=None):
        p = {
            "pid": 7, "title": "T", "topics": [], "contacts": [], "pc_conflicts": {},
            "reserve_reviewer": "",
            "authors": [{"given_name": "Wei", "family_name": "Zhang",
                         "email": "wz@y.edu", "affiliation": "Y"}],
        }
        if dblp:
            p["dblp"] = dblp
        return p

    def derive(self, paper, author_names, *, use_identity=True):
        index = coauthor_coi.build_index(
            [self.reviewer], self.coauthors, years=5, current_year=2026
        )
        return coauthor_coi.derive_conflicts(
            [paper], index, author_names, use_identity=use_identity
        )

    def test_a_different_homonym_of_the_same_name_is_a_different_person(self):
        p = self.paper(dblp="https://dblp.org/pid/9/WZ.html")
        self.assertEqual({}, self.derive(p, {"9/WZ": ["Wei Zhang 0012"]}))

    def test_the_same_homonym_still_conflicts(self):
        p = self.paper(dblp="https://dblp.org/pid/9/WZ.html")
        derived = self.derive(p, {"9/WZ": ["Wei Zhang 0001"]})
        self.assertEqual("exact", derived[7]["rev@x.edu"].match)

    def test_an_author_who_declared_no_dblp_page_keeps_the_permissive_reading(self):
        # Not knowing which Wei Zhang someone is cannot be evidence that they
        # are not this one. Roughly half the author slots are in this state,
        # which is the real limit on how much identity matching can do.
        self.assertIn("rev@x.edu", self.derive(self.paper(), {})[7])

    def test_the_unnumbered_name_is_itself_an_identity(self):
        # A bare "Wei Zhang" in DBLP is one specific person, not a wildcard.
        p = self.paper(dblp="https://dblp.org/pid/9/WZ.html")
        self.assertEqual({}, self.derive(p, {"9/WZ": ["Wei Zhang"]}))

    def test_one_persons_two_spellings_are_not_read_as_two_people(self):
        # The reason this compares the homonym number and not the raw string:
        # accents and middle initials vary freely within one identity, and
        # string inequality there would silently drop a real conflict.
        self.reviewer.pid = "2/Bob"
        self.coauthors = {"2/Bob": {"José García": [[2025, "A paper."]]}}
        p = {
            "pid": 7, "title": "T", "topics": [], "contacts": [], "pc_conflicts": {},
            "reserve_reviewer": "", "dblp": "https://dblp.org/pid/9/JG.html",
            "authors": [{"given_name": "Jose", "family_name": "Garcia",
                         "email": "jg@y.edu", "affiliation": "Y"}],
        }
        derived = self.derive(p, {"9/JG": ["Jose A. Garcia"]})
        self.assertIn("rev@x.edu", derived[7])

    def test_the_numbering_can_be_ignored_for_a_comparison_run(self):
        p = self.paper(dblp="https://dblp.org/pid/9/WZ.html")
        names = {"9/WZ": ["Wei Zhang 0012"]}
        self.assertEqual({}, self.derive(p, names))
        self.assertIn("rev@x.edu", self.derive(p, names, use_identity=False)[7])

    def test_identity_reads_the_suffix_and_nothing_else(self):
        self.assertEqual("0012", coauthor_coi.identity("Wei Zhang 0012"))
        self.assertEqual("", coauthor_coi.identity("Wei Zhang"))
        self.assertEqual("", coauthor_coi.identity("Matthew D. Sinclair"))


class ReservePidResolverTests(unittest.TestCase):
    """resolve_reserve_pids.py: proposing DBLP pages from the submissions."""

    def test_rerun_keeps_rows_no_longer_listed_as_unresolved(self):
        # Resolving someone drops them off the unresolved list. Rebuilding the
        # override file from that list alone would delete the decision that
        # resolved them, and the next roster rebuild would lose them entirely.
        tmp = Path(tempfile.mkdtemp())
        overrides, unresolved, data = (tmp / "ov.csv", tmp / "un.csv", tmp / "papers.json")
        with overrides.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["email", "dblp", "note"])
            w.writeheader()
            w.writerow({"email": "settled@x.edu", "dblp": "https://dblp.org/pid/1/A.html",
                        "note": "resolved on an earlier run"})
        with unresolved.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["email", "name", "dblp_url", "problem", "detail"])
            w.writeheader()
            w.writerow({"email": "still@x.edu", "name": "Nobody Here",
                        "dblp_url": "", "problem": "no_dblp_url", "detail": ""})
        data.write_text("[]", encoding="utf-8")

        argv = ["resolve_reserve_pids.py", "--unresolved", str(unresolved),
                "--data", str(data), "--out", str(overrides), "--no-network"]
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stderr(io.StringIO()):
            resolve_reserve_pids.main()

        with overrides.open(newline="", encoding="utf-8") as f:
            rows = {r["email"]: r for r in csv.DictReader(f)}
        self.assertIn("settled@x.edu", rows)
        self.assertEqual("https://dblp.org/pid/1/A.html", rows["settled@x.edu"]["dblp"])
        self.assertEqual("resolved on an earlier run", rows["settled@x.edu"]["note"])
        # The still-unresolved person is present too, as a blank to-do row.
        self.assertEqual("", rows["still@x.edu"]["dblp"])


def _cell_ref(column, row):
    letters = ""
    while True:
        column, remainder = divmod(column, 26)
        letters = chr(ord("A") + remainder) + letters
        if column == 0:
            break
        column -= 1
    return f"{letters}{row}"


def write_xlsx(path, rows, *, shared_strings=False, sheet_name="All"):
    """A minimal .xlsx holding one worksheet of text cells.

    Only the parts read_sheet actually opens are written. A None cell is left
    out of the XML entirely, which is how a real writer records a gap.
    """
    interned = []
    sheet = ['<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>']
    for r, row in enumerate(rows, start=1):
        sheet.append(f'<row r="{r}">')
        for c, value in enumerate(row):
            if value is None:
                continue
            ref = _cell_ref(c, r)
            if shared_strings:
                if value not in interned:
                    interned.append(value)
                sheet.append(f'<c r="{ref}" t="s"><v>{interned.index(value)}</v></c>')
            else:
                sheet.append(f'<c r="{ref}" t="inlineStr"><is><t>{value}</t></is></c>')
        sheet.append("</row>")
    sheet.append("</sheetData></worksheet>")

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "xl/workbook.xml",
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
            ' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f'<sheets><sheet name="{sheet_name}" sheetId="1" r:id="rId1"/></sheets></workbook>',
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Target="/xl/worksheets/sheet1.xml" Type="worksheet"/>'
            "</Relationships>",
        )
        archive.writestr("xl/worksheets/sheet1.xml", "".join(sheet))
        if shared_strings:
            items = "".join(f"<si><t>{v}</t></si>" for v in interned)
            archive.writestr(
                "xl/sharedStrings.xml",
                '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                f"{items}</sst>",
            )


class ReserveRosterTests(unittest.TestCase):
    """build_reserve_reviewer_info.py: HotCRP upload + vetting workbook -> roster."""

    def build(self, uploaded, vetted, *, shared_strings=False, overrides=()):
        """Run the script over in-memory inputs; return (roster, unresolved)."""
        tmp = Path(tempfile.mkdtemp())
        upload, vetting = tmp / "upload.csv", tmp / "vetting.xlsx"
        out, unresolved = tmp / "info.csv", tmp / "unresolved.csv"
        override_path = tmp / "overrides.csv"
        with override_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["email", "dblp", "note"])
            writer.writeheader()
            for email, link in overrides:
                writer.writerow({"email": email, "dblp": link, "note": "by hand"})
        with upload.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["email", "name", "roles", "tags"])
            writer.writeheader()
            for email, name in uploaded:
                writer.writerow({"email": email, "name": name,
                                 "roles": "pc", "tags": "reserve-reviewer"})
        write_xlsx(
            vetting, [["name", "email", "dblp_url"], *vetted],
            shared_strings=shared_strings,
        )
        argv = ["build_reserve_reviewer_info.py", "--upload", str(upload),
                "--vetting", str(vetting), "--out", str(out),
                "--unresolved", str(unresolved), "--overrides", str(override_path)]
        with mock.patch.object(sys, "argv", argv), \
                contextlib.redirect_stderr(io.StringIO()):
            build_reserve_reviewer_info.main()
        def read(path):
            with path.open(newline="", encoding="utf-8") as f:
                return list(csv.DictReader(f))

        return read(out), read(unresolved)

    def test_sheet_reader_handles_both_string_encodings_and_gaps(self):
        # Google Sheets inlines its strings; Excel interns them. A re-export in
        # the other format must not read back as a workbook of empty cells.
        rows = [["name", "email", "dblp_url"],
                ["Ada Lovelace", "ada@example.edu", "https://dblp.org/pid/1/Ada"],
                ["Alan Turing", "alan@example.edu", None]]
        tmp = Path(tempfile.mkdtemp())
        parsed = []
        for shared in (False, True):
            path = tmp / f"book-{shared}.xlsx"
            write_xlsx(path, rows, shared_strings=shared)
            parsed.append(build_reserve_reviewer_info.read_sheet(str(path), "All"))
        self.assertEqual(parsed[0], parsed[1])
        self.assertEqual("https://dblp.org/pid/1/Ada", parsed[0][0]["dblp_url"])
        # The omitted trailing cell is a gap, not a missing column.
        self.assertEqual("", parsed[0][1]["dblp_url"])

    def test_shared_pid_holds_back_every_claimant(self):
        # Either one of them is wrong, or one person holds two HotCRP accounts
        # and would draw double the reviews. Both are the chair's call.
        roster, unresolved = self.build(
            [("a@x.edu", "Youhui Zhang"), ("b@x.edu", "Youhui Zhang"),
             ("c@x.edu", "Ada Lovelace")],
            [["Youhui Zhang", "a@x.edu", "https://dblp.org/pid/00/5995"],
             ["Youhui Zhang", "b@x.edu", "https://dblp.org/pid/00/5995.html"],
             ["Ada Lovelace", "c@x.edu", "https://dblp.org/pid/12/3456"]],
        )
        self.assertEqual(["c@x.edu"], [r["email"] for r in roster])
        self.assertEqual({"a@x.edu", "b@x.edu"}, {r["email"] for r in unresolved})
        self.assertEqual({"shared_pid"}, {r["problem"] for r in unresolved})

    def test_a_voided_claim_leaves_the_other_claimant_clean(self):
        # "Tao Zhang" does not own z/HaoZhang2, so that claim never competes and
        # the real Hao Zhang keeps the PID rather than both being held back.
        roster, unresolved = self.build(
            [("tao@x.edu", "Tao Zhang"), ("hao@x.edu", "Hao Zhang")],
            [["Tao Zhang", "tao@x.edu", "https://dblp.org/pid/z/HaoZhang2"],
             ["Hao Zhang", "hao@x.edu", "https://dblp.org/pid/z/HaoZhang2"]],
        )
        self.assertEqual(["hao@x.edu"], [r["email"] for r in roster])
        self.assertEqual([("tao@x.edu", "name_mismatch")],
                         [(r["email"], r["problem"]) for r in unresolved])

    def test_excluded_is_a_finished_decision_not_outstanding_work(self):
        # Both keep someone off the roster, but they mean different things: one
        # is a job still to do, the other is a job done. If they were the same
        # value the report could never reach zero.
        roster, unresolved = self.build(
            [("gone@x.edu", "Not A Reviewer"), ("todo@x.edu", "Needs A Link"),
             ("ok@x.edu", "Ada Lovelace")],
            [["Not A Reviewer", "gone@x.edu", "https://dblp.org/pid/1/A"],
             ["Needs A Link", "todo@x.edu", "https://dblp.org/pid/2/B"],
             ["Ada Lovelace", "ok@x.edu", "https://dblp.org/pid/3/C"]],
            overrides=[("gone@x.edu", "excluded"), ("todo@x.edu", "none")],
        )
        self.assertEqual(["ok@x.edu"], [r["email"] for r in roster])
        self.assertEqual({"gone@x.edu": "excluded", "todo@x.edu": "withheld"},
                         {r["email"]: r["problem"] for r in unresolved})

    def test_withheld_override_stops_a_wrong_link_competing(self):
        # "none" means the workbook's link is known wrong with no replacement.
        # It has to differ from an empty cell: left in place, the wrong link goes
        # on claiming w/LinZhao3 and holds back the Univ-A Lin Zhao who owns it.
        roster, unresolved = self.build(
            [("lzhao@univ-a.edu.cn", "Lin Zhao"), ("zhaolin@lab-b.com.cn", "Lin Zhao")],
            [["Lin Zhao", "lzhao@univ-a.edu.cn", "https://dblp.org/pid/w/LinZhao3"],
             ["Lin Zhao", "zhaolin@lab-b.com.cn", "https://dblp.org/pid/w/LinZhao3"]],
            overrides=[("zhaolin@lab-b.com.cn", "none")],
        )
        self.assertEqual(["lzhao@univ-a.edu.cn"], [r["email"] for r in roster])
        self.assertEqual([("zhaolin@lab-b.com.cn", "withheld")],
                         [(r["email"], r["problem"]) for r in unresolved])

    def test_a_hand_entered_override_beats_the_workbook(self):
        # The workbook's link for the first reviewer named the second; once the
        # right page is entered by hand it wins outright, and none of the
        # workbook's checks (including the shared-PID collision) apply any more.
        roster, unresolved = self.build(
            [("mlam@cse.univ-h.edu.hk", "Mei Lam"), ("other@x.edu", "Wen Zhou")],
            [["Mei Lam", "mlam@cse.univ-h.edu.hk", "https://dblp.org/pid/10/0100-12"],
             ["Wen Zhou", "other@x.edu", "https://dblp.org/pid/10/0100-12"]],
            overrides=[("mlam@cse.univ-h.edu.hk", "https://dblp.org/pid/28/0100-1")],
        )
        self.assertEqual(
            [("mlam@cse.univ-h.edu.hk", "https://dblp.org/pid/28/0100-1.html"),
             ("other@x.edu", "https://dblp.org/pid/10/0100-12.html")],
            [(r["email"], r["dblp"]) for r in roster],
        )
        self.assertEqual([], unresolved)

        # A blank cell is a to-do marker, not a decision: it must not mask the
        # workbook's value, or filling the file in would erase what was there.
        roster, _ = self.build(
            [("a@x.edu", "Ada Lovelace")],
            [["Ada Lovelace", "a@x.edu", "https://dblp.org/pid/12/3456"]],
            overrides=[("a@x.edu", "")],
        )
        self.assertEqual(["https://dblp.org/pid/12/3456.html"],
                         [r["dblp"] for r in roster])

    def test_collision_reports_the_affiliation_without_deciding_on_it(self):
        # Both names fit 43/0100-1 ({wei, tan} overlaps either way), and the
        # domains cannot settle it: "Coastal Advanced Institute of Science and
        # Technology" would claim coastal.ac.kr as readily as caist.ac.kr. So the
        # institution is quoted for the chair, and neither is seated.
        kaist = ["Coastal Advanced Institute of Science and Technology (CAIST), Riverton"]
        rows = {
            "minseo@caist.ac.kr": {"pid": "43/0100-1", "problem": None, "detail": "",
                                     "affiliations": kaist},
            "jiwon@coastal.ac.kr": {"pid": "43/0100-1", "problem": None, "detail": "",
                                      "affiliations": kaist},
        }
        build_reserve_reviewer_info.apply_shared_pids(rows)
        self.assertEqual(["shared_pid", "shared_pid"],
                         [r["problem"] for r in rows.values()])
        for row in rows.values():
            self.assertIn("Coastal Advanced Institute", row["detail"])

    def test_named_pid_naming_someone_else_is_caught(self):
        agree = build_reserve_reviewer_info.names_agree
        self.assertEqual("Ling Wei Tan",
                         build_reserve_reviewer_info.pid_name("c/LingWeiTan"))
        # A numeric PID's digits say nothing about whose page it is.
        self.assertIsNone(build_reserve_reviewer_info.pid_name("26/1737"))
        self.assertIsNone(build_reserve_reviewer_info.pid_name("43/0100-1"))
        self.assertTrue(agree("Ling-Wei Tan", "Ling Wei Tan"))
        self.assertTrue(agree("Dhabaleswar Panda", "Dhabaleswar K Panda"))
        # One shared family name is not an identity: this is the real error.
        self.assertFalse(agree("Tao Zhang", "Hao Zhang"))
        self.assertFalse(agree("Lei Chen", "Chen Li"))
        # A mononym has only one token to give; demanding two would fail it for
        # being short rather than for being someone else.
        self.assertTrue(agree("Bono", "Bono"))
        self.assertFalse(agree("Bono", "Cher"))

    def test_annotation_after_the_link_is_flagged_not_stripped(self):
        # parse_pid stops at the '.', so a hand-written doubt about the link
        # would otherwise disappear and the PID be accepted as verified.
        roster, unresolved = self.build(
            [("z@x.edu", "Zicong Wang"), ("g@x.edu", "Guillem Lopez")],
            [["Zicong Wang", "z@x.edu",
              "https://dblp.org/pid/190/5197.html （disambiguation page）"],
             ["Guillem Lopez", "g@x.edu", ""]],
        )
        self.assertEqual([], roster)
        problems = {r["email"]: r["problem"] for r in unresolved}
        self.assertEqual({"z@x.edu": "annotated", "g@x.edu": "no_dblp_url"}, problems)
        # The unresolved file keeps the cell verbatim, annotation and all.
        kept = next(r["dblp_url"] for r in unresolved if r["email"] == "z@x.edu")
        self.assertIn("（disambiguation page）", kept)

    def test_roster_is_sorted_and_urls_normalised(self):
        roster, unresolved = self.build(
            [("z@x.edu", "Zoe Last"), ("a@x.edu", "Ada First"),
             ("m@x.edu", "Max Middle"), ("n@x.edu", "New Person")],
            [["Zoe Last", "z@x.edu", "https://dblp.org/pid/1/Zoe"],
             ["Ada First", "a@x.edu", "https://dblp.org/pid/26/1737.html"],
             ["Max Middle", "m@x.edu", "https://dblp.org/pid/62/1131"]],
        )
        self.assertEqual(["a@x.edu", "m@x.edu", "z@x.edu"], [r["email"] for r in roster])
        # Every shape the workbook uses lands on one canonical URL.
        self.assertEqual(
            ["https://dblp.org/pid/26/1737.html", "https://dblp.org/pid/62/1131.html",
             "https://dblp.org/pid/1/Zoe.html"],
            [r["dblp"] for r in roster],
        )
        # An uploaded reviewer the workbook never vetted is reported, not dropped.
        self.assertEqual([("n@x.edu", "not_in_vetting")],
                         [(r["email"], r["problem"]) for r in unresolved])


class RegionResolutionTests(unittest.TestCase):
    """affiliation_country.py: placing an institution, and never guessing."""

    def test_hong_kong_is_never_folded_into_china(self):
        # The whole point of the rule: HK, MO, TW and SG are their own codes.
        # DBLP and HotCRP both write "Hong Kong ..., China", so the region has
        # to outrank the sovereign state whenever both are named.
        resolve = affiliation_country.resolve_country
        self.assertEqual("HK", resolve("The Chinese University of Hong Kong")[0])
        self.assertEqual("HK", resolve("Unknown Lab", "someone@cse.univ-h.edu.hk")[0])
        self.assertEqual("HK", resolve(
            "Hong Kong University of Science and Technology, Department of "
            "Electronic and Computer Engineering, China")[0])
        self.assertEqual("MO", resolve("University of Macau, China")[0])
        self.assertEqual("TW", resolve("National Taiwan University, Taipei")[0])
        self.assertEqual("SG", resolve("National University of Singapore")[0])
        # "Chinese" is an adjective, not a location; only the real name counts.
        self.assertEqual(
            "CN",
            resolve("Institute of Computing Technology, Chinese Academy of "
                    "Sciences", "someone@ict.ac.cn")[0],
        )

    def test_waterfall_precedence_and_the_layer_it_reports(self):
        overrides = {affiliation_country.normalize_affiliation("Blue University"): "JP"}
        notes = ["Blue University, Riverton, Portugal"]
        self.assertEqual(
            ("JP", "hand"),
            affiliation_country.resolve_country(
                "Blue University", "x@blue.ac.kr", notes, overrides),
        )
        self.assertEqual(
            ("PT", "dblp"),
            affiliation_country.resolve_country("Blue University", "x@blue.ac.kr", notes),
        )
        self.assertEqual(
            ("KR", "email"),
            affiliation_country.resolve_country("Blue University", "x@blue.ac.kr"),
        )
        self.assertEqual(
            ("PT", "affiliation"),
            affiliation_country.resolve_country("Blue University, Portugal", "x@blue.com"),
        )

    def test_nothing_is_guessed(self):
        resolve = affiliation_country.resolve_country
        # Generic TLDs are sold to anyone anywhere and place nobody.
        for tld in ("com", "edu", "org", "io", "ai", "co", "me"):
            self.assertEqual("", resolve("Some Startup", f"x@acme.{tld}")[0], tld)
        # Country names must match whole tokens, so a lookalike is not a match.
        self.assertEqual("", resolve("Indiana University", "x@iu.edu")[0])
        # Names that are also institutions or US states are left out entirely
        # rather than producing a confident wrong answer.
        self.assertEqual("", resolve("Georgia Institute of Technology")[0])
        self.assertEqual(("", "unresolved"), resolve("Unknown Place", "x@nowhere.com"))

    def test_a_dblp_note_is_chosen_by_matching_the_stated_affiliation(self):
        # DBLP's note order means nothing -- a Tsinghua professor's notes can
        # list UC Santa Barbara first -- so the person's own affiliation picks
        # the note, and no match means this layer declines rather than
        # answering with a former employer's country.
        notes = ["University of California at Santa Barbara, CA, USA",
                 "Blue University, Riverton, Portugal"]
        self.assertEqual("PT", affiliation_country.country_from_dblp(notes, "Blue University"))
        self.assertEqual("US", affiliation_country.country_from_dblp(
            notes, "University of California at Santa Barbara"))
        self.assertEqual("", affiliation_country.country_from_dblp(notes, "Green Institute"))
        self.assertEqual("", affiliation_country.country_from_dblp(notes, ""))

    def test_normalize_folds_only_trivial_spelling_differences(self):
        norm = affiliation_country.normalize_affiliation
        self.assertEqual(norm("The Blue University"), norm("Blue University,"))
        self.assertEqual(norm("Universite de Montreal"), norm("Université de Montréal"))
        # Abbreviations are a judgement call, so they stay distinct.
        self.assertNotEqual(norm("Blue University"), norm("Blue Univ."))

    def test_pcinfo_outranks_every_other_layer_including_hand(self):
        # What the account holder told HotCRP wins even over a human-curated
        # affiliation_countries.csv entry -- the whole point of making it
        # layer 1, not just "another automatic layer".
        overrides = {affiliation_country.normalize_affiliation("Blue University"): "JP"}
        notes = ["Blue University, Riverton, Portugal"]
        self.assertEqual(
            ("DE", "pcinfo"),
            affiliation_country.resolve_country(
                "Blue University", "x@blue.ac.kr", notes, overrides, "DE"),
        )
        # Absent (empty string), the pre-existing waterfall is untouched.
        self.assertEqual(
            ("JP", "hand"),
            affiliation_country.resolve_country(
                "Blue University", "x@blue.ac.kr", notes, overrides, ""),
        )

    def test_country_from_pcinfo_normalizes_messy_hotcrp_text(self):
        # Most accounts already carry a clean ISO code from HotCRP's own
        # picker; a few carry free text instead, observed on the live export.
        cfp = affiliation_country.country_from_pcinfo
        self.assertEqual("US", cfp("US"))
        self.assertEqual("US", cfp("usa"))
        self.assertEqual("US", cfp("USA"))
        self.assertEqual("US", cfp("United States of America"))
        self.assertEqual("HK", cfp("hk"))
        self.assertEqual("", cfp(""))
        self.assertEqual("", cfp("Nowhereland"))

    def test_hong_kong_accounts_that_say_china_resolve_to_china(self):
        # The accepted tradeoff, locked in on purpose: HotCRP's own field
        # outranks the affiliation-text layer even where that layer is
        # correct and pcinfo's "China" is a self-report that elides the SAR.
        # A future change must not silently "fix" this by accident.
        index = pc_membership.load_pc_accounts(str(write_pcinfo(
            Path(tempfile.mkdtemp()) / "pc.csv",
            [pc_account("cuhk@example.edu.hk", "A", "B", country="CN")],
        )))
        layers = affiliation_country.load_layers(
            "/nonexistent", "/nonexistent", "/nonexistent", pcinfo_index=index,
        )
        r = reviewer()
        r.email = "cuhk@example.edu.hk"
        r.hotcrp_email = "cuhk@example.edu.hk"
        r.affiliation = "The Chinese University of Hong Kong"
        self.assertEqual(("CN", "pcinfo"), affiliation_country.reviewer_country(r, layers))
        # The layer below still gets it right when pcinfo has no opinion.
        r2 = reviewer()
        r2.email = "someone-else@example.edu.hk"
        r2.hotcrp_email = "someone-else@example.edu.hk"
        r2.affiliation = "The Chinese University of Hong Kong"
        self.assertEqual(("HK", "affiliation"), affiliation_country.reviewer_country(r2, layers))

    def test_pcinfo_lookup_uses_the_hotcrp_email_not_the_form_email(self):
        # A reviewer who submitted the acceptance form under one address but
        # holds their HotCRP account under another (pc_membership.match's
        # whole reason for existing -- eleven of them on the live roster) must
        # be looked up in pcinfo_by_email under hotcrp_email. Keying on the
        # form email instead would silently miss this layer for every one.
        index = pc_membership.load_pc_accounts(str(write_pcinfo(
            Path(tempfile.mkdtemp()) / "pc.csv",
            [pc_account("real@nd.edu", "A", "B", country="US")],
        )))
        layers = affiliation_country.load_layers(
            "/nonexistent", "/nonexistent", "/nonexistent", pcinfo_index=index,
        )
        r = reviewer()
        r.email = "form-address@cuhk.edu.hk"
        r.hotcrp_email = "real@nd.edu"
        self.assertEqual(("US", "pcinfo"), affiliation_country.reviewer_country(r, layers))

    def test_pcinfo_layer_covers_every_account_not_just_pc_members(self):
        # author_country needs this too: most paper authors are never on the
        # PC, but PcIndex.by_email/.accounts already cover the whole export.
        index = pc_membership.load_pc_accounts(str(write_pcinfo(
            Path(tempfile.mkdtemp()) / "pc.csv",
            [pc_account("nonpc@example.com", "A", "B", roles="", country="FR"),
             pc_account("blank@example.com", "C", "D", country="")],
        )))
        layers = affiliation_country.load_layers(
            "/nonexistent", "/nonexistent", "/nonexistent", pcinfo_index=index,
        )
        self.assertEqual(
            ("FR", "pcinfo"),
            affiliation_country.author_country({"email": "nonpc@example.com"}, layers),
        )
        # A blank country cell contributes nothing -- falls through untouched.
        self.assertEqual(
            ("", "unresolved"),
            affiliation_country.author_country({"email": "blank@example.com"}, layers),
        )

    def test_load_layers_without_a_pcinfo_index_is_unchanged(self):
        # No PcIndex passed -> pcinfo_by_email is empty, the pre-existing
        # four-layer behavior is exactly what callers who don't opt in get.
        layers = affiliation_country.load_layers("/nonexistent", "/nonexistent", "/nonexistent")
        self.assertEqual({}, layers.pcinfo_by_email)


class SameCountryCapTests(unittest.TestCase):
    """assign_reviewers.py: which papers the same-country cap binds, and how."""

    def paper(self, pid, countries):
        # countries: one entry per author, "" for an author we cannot place.
        authors = [
            {"email": f"a{i}@x.edu", "affiliation": c} for i, c in enumerate(countries)
        ]
        return {"pid": pid, "title": f"P{pid}", "authors": authors}

    def build(self, papers, majority=0.5, min_resolved=0.5, cap=2, reviewers=(),
              pcinfo_by_email=None):
        layers = affiliation_country.CountryLayers(pcinfo_by_email=pcinfo_by_email or {})
        by_email = {}
        for email, affiliation in reviewers:
            by_email[email] = reviewer()
            by_email[email].email = email
            by_email[email].hotcrp_email = email
            by_email[email].affiliation = affiliation
            by_email[email].pid = None
        return assign_reviewers.build_country_caps(
            papers, list(by_email), by_email, cap, layers,
            majority=majority, min_resolved=min_resolved,
        )

    def by_code(self, caps):
        return {c.code: c for c in caps}

    def test_every_country_is_capped_on_its_own_papers(self):
        # The point of the rule: nothing names a country. A US paper is capped
        # on US reviewers exactly as a Chinese paper is on Chinese ones, in one
        # run, from one flag.
        papers = [self.paper(1, ["China", "China", "USA"]),
                  self.paper(2, ["USA", "USA", "China"])]
        caps, _, _, _ = self.build(papers)
        by = self.by_code(caps)
        self.assertEqual({"CN", "US"}, set(by))
        self.assertEqual({1: 2}, by["CN"].papers)   # CN caps the CN-majority paper
        self.assertEqual({2: 2}, by["US"].papers)   # and not the US-majority one
        self.assertNotIn(2, by["CN"].papers)
        self.assertNotIn(1, by["US"].papers)

    def test_only_the_majority_country_is_capped_on_a_paper(self):
        # A paper with a Chinese majority and one American author caps Chinese
        # reviewers only; the US class does not exist unless some paper is
        # majority-US.
        caps, _, _, _ = self.build([self.paper(1, ["China", "China", "USA"])])
        self.assertEqual(["CN"], [c.code for c in caps])

    def test_majority_uses_placed_authors_under_a_coverage_floor(self):
        papers = [
            # 3 CN of 5 placed = 60% > 50%, and 5 of 10 placed clears the floor.
            self.paper(1, ["China", "China", "China", "USA", "USA"] + [""] * 5),
            # 1 CN of 1 placed reads as 100%, but 1 of 10 is below the floor:
            # not capped, and reported instead of guessed at.
            self.paper(2, ["China"] + [""] * 9),
            # exactly half is not a majority
            self.paper(3, ["China", "USA"]),
        ]
        caps, _, coverage, thin = self.build(papers)
        cn = self.by_code(caps)["CN"]
        self.assertEqual({1: 2}, cn.papers)
        self.assertEqual([2], thin)
        self.assertEqual((5, 10), coverage[1])
        self.assertEqual((3, 5, 10), cn.shares[1])

    def test_a_paper_with_no_majority_or_no_authors_is_uncapped(self):
        caps, _, _, thin = self.build([self.paper(1, []), self.paper(2, ["China", "USA"])])
        self.assertEqual([], caps)
        self.assertEqual([], thin)   # an authorless paper is not "thin", just absent

    def test_reviewers_are_classed_by_their_own_country(self):
        papers = [self.paper(1, ["China", "China"])]
        caps, by_email, _, _ = self.build(
            papers, reviewers=[("cn@x.edu", "Tsinghua University, China"),
                               ("us@x.edu", "Duke University, USA"),
                               ("no@x.edu", "Somewhere Unknown")])
        cn = self.by_code(caps)["CN"]
        self.assertEqual(frozenset({"cn@x.edu"}), cn.members)
        # An unplaced reviewer is in no class and can never consume a cap.
        self.assertEqual("", by_email["no@x.edu"])

    def test_a_pcinfo_country_overrides_the_reviewers_own_affiliation(self):
        # End-to-end: build_country_caps draws on reviewer_country(), which
        # must consult layers.pcinfo_by_email before the affiliation string.
        # The paper is still majority-CN by its authors' affiliations (this
        # helper's authors are affiliation strings, not pcinfo-backed) -- the
        # thing under test is which *reviewer* class this person lands in.
        papers = [self.paper(1, ["China", "China"])]
        caps, by_email, _, _ = self.build(
            papers,
            reviewers=[("us@x.edu", "Tsinghua University, China")],
            pcinfo_by_email={"us@x.edu": "US"},
        )
        self.assertEqual("US", by_email["us@x.edu"])
        cn = self.by_code(caps)["CN"]
        self.assertNotIn("us@x.edu", cn.members)

    def test_the_thresholds_are_tunable(self):
        papers = [self.paper(1, ["China", "USA"])]
        self.assertEqual([], self.build(papers)[0])
        # A 40% threshold makes an even split a majority — for whichever country
        # Counter.most_common puts first, so just assert one class appeared.
        self.assertEqual(1, len(self.build(papers, majority=0.4)[0]))

    def test_a_zero_cap_is_a_policy_not_an_off_switch(self):
        # 0 means "no reviewer from the paper's own country", which this roster
        # can actually satisfy; switching the policy off is --no-same-country-cap
        # and is handled by main, not here.
        caps, _, _, _ = self.build([self.paper(1, ["China", "China"])], cap=0)
        self.assertEqual({1: 0}, caps[0].papers)


class AffiliationCountryFileTests(unittest.TestCase):
    """build_affiliation_countries.py: the hand layer's to-do file."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.out = str(self.dir / "affiliation_countries.csv")

    def entries(self):
        return {
            affiliation_country.normalize_affiliation("Blue University"): {
                "raw": "Blue University", "emails": {"a@blue.ac.kr"},
                "pids": set(), "roster": True, "people": 1,
            },
        }

    def test_the_generator_never_clobbers_a_hand_entered_country(self):
        layers = affiliation_country.CountryLayers()
        rows = build_affiliation_countries.merge_rows([], self.entries(), layers)
        build_affiliation_countries.write_countries(self.out, rows)
        # The country column is the human's; the generator only ever suggests.
        self.assertEqual("", rows[0]["country"])
        self.assertEqual("KR", rows[0]["suggested"])

        # A hand decision survives a rerun that disagrees with it, and is what
        # gets read back.
        rows[0]["country"] = "JP"
        build_affiliation_countries.write_countries(self.out, rows)
        again = build_affiliation_countries.merge_rows(
            build_affiliation_countries.read_existing(self.out), self.entries(), layers)
        self.assertEqual("JP", again[0]["country"])
        self.assertEqual("KR", again[0]["suggested"])
        self.assertEqual(
            {affiliation_country.normalize_affiliation("Blue University"): "JP"},
            affiliation_country.load_affiliation_countries(self.out),
        )

    def test_decided_by_rides_along_with_the_country_it_explains(self):
        # Provenance has to survive a regenerate for a filled cell to stay
        # auditable. It cannot live in `note`, which merge_rows rewrites.
        layers = affiliation_country.CountryLayers()
        rows = build_affiliation_countries.merge_rows([], self.entries(), layers)
        rows[0]["country"] = "JP"
        rows[0]["decided_by"] = "websearch"
        build_affiliation_countries.write_countries(self.out, rows)
        again = build_affiliation_countries.merge_rows(
            build_affiliation_countries.read_existing(self.out), self.entries(), layers)
        self.assertEqual("JP", again[0]["country"])
        self.assertEqual("websearch", again[0]["decided_by"])
        self.assertEqual("roster", again[0]["note"])   # regenerated, not carried

    def test_rerunning_on_unchanged_input_is_byte_identical(self):
        layers = affiliation_country.CountryLayers()
        first = str(self.dir / "a.csv")
        second = str(self.dir / "b.csv")
        rows = build_affiliation_countries.merge_rows([], self.entries(), layers)
        build_affiliation_countries.write_countries(first, rows)
        build_affiliation_countries.write_countries(
            second,
            build_affiliation_countries.merge_rows(
                build_affiliation_countries.read_existing(first), self.entries(), layers),
        )
        self.assertEqual(Path(first).read_bytes(), Path(second).read_bytes())

    def test_an_affiliation_that_left_the_data_keeps_its_decision(self):
        # The HotCRP export is a moving snapshot; a withdrawn paper must not
        # delete a decision someone already made.
        layers = affiliation_country.CountryLayers()
        rows = build_affiliation_countries.merge_rows([], self.entries(), layers)
        rows[0]["country"] = "KR"
        build_affiliation_countries.write_countries(self.out, rows)
        after = build_affiliation_countries.merge_rows(
            build_affiliation_countries.read_existing(self.out), {}, layers)
        self.assertEqual("KR", after[0]["country"])
        self.assertEqual("0", after[0]["people"])

    def test_an_unknown_country_code_fails_loudly(self):
        Path(self.out).write_text(
            "affiliation,country,suggested,source,people,note\n"
            "Blue University,ZZ,,,,\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            affiliation_country.load_affiliation_countries(self.out)


class PcMembershipTests(unittest.TestCase):
    """pc_membership.py: the HotCRP export is the authority on who is on the PC."""

    def index(self, accounts):
        tmp = Path(tempfile.mkdtemp())
        return pc_membership.load_pc_accounts(str(write_pcinfo(tmp / "pc.csv", accounts)))

    def test_the_roles_column_is_a_token_list_not_a_substring(self):
        # HotCRP writes a chair's roles as "chair pc", so a substring test was
        # needed for them -- but a substring test also lets "spc" or "pcx" pass,
        # which would keep people on the roster who hold no PC role at all.
        self.assertTrue(pc_membership.on_pc("pc"))
        self.assertTrue(pc_membership.on_pc("chair pc"))
        self.assertFalse(pc_membership.on_pc("spc"))
        self.assertFalse(pc_membership.on_pc("pcx"))
        self.assertFalse(pc_membership.on_pc(""))

    def test_a_member_registered_under_a_second_address_is_matched_by_name(self):
        # The failure this exists to prevent: people accept from one address and
        # hold their HotCRP account under another. On the export this was built
        # against that is twelve of the sixteen rows whose own address is not
        # pc, so an email-only rule removes twelve sitting PC members.
        index = self.index([pc_account("real@b.edu", "Ada", "Lovelace")])
        acct, how = index.match("other@a.edu", "Ada", "Lovelace")
        self.assertEqual("real@b.edu", acct.email)
        self.assertEqual("name", how)

    def test_a_member_registered_under_a_second_address_is_matched_by_local_part(self):
        # Same person, same mailbox name, different institution -- and the form
        # row carries no name at all, so the name index cannot save them.
        index = self.index([pc_account("alovelace@b.edu", "Ada", "Lovelace")])
        acct, how = index.match("alovelace@a.edu", "", "")
        self.assertEqual("alovelace@b.edu", acct.email)
        self.assertEqual("local", how)

    def test_an_email_cell_holding_a_name_is_read_as_a_name(self):
        # One acceptance row has a name typed into the email box. That is a
        # form-entry slip, not a resignation; treating the cell as an address
        # only would drop a sitting PC member over a typo.
        index = self.index([pc_account("ada@b.edu", "Ada", "Lovelace")])
        acct, how = index.match("ada lovelace", "", "")
        self.assertEqual("ada@b.edu", acct.email)
        self.assertEqual("name", how)

    def test_only_an_account_that_exists_and_is_not_pc_is_pruned(self):
        index = self.index([
            pc_account("member@a.edu", "Ada", "Lovelace"),
            {"email": "former@a.edu", "given_name": "Alan", "family_name": "Turing",
             "roles": "", "tags": "pc-full"},
        ])
        self.assertIsNotNone(index.match("member@a.edu", "Ada", "Lovelace")[0])
        self.assertIsNone(index.match("former@a.edu", "Alan", "Turing")[0])
        self.assertIsNone(index.match("stranger@a.edu", "Grace", "Hopper")[0])

    def test_a_disabled_pc_account_does_not_keep_its_holder_on_the_roster(self):
        # Disabled means they cannot log in, so they cannot review -- but the
        # account stays in by_email so the audit can say "disabled" rather than
        # "no account".
        index = self.index([
            pc_account("off@a.edu", "Ada", "Lovelace", disabled="yes"),
            pc_account("on@a.edu", "Alan", "Turing"),
        ])
        self.assertIsNone(index.match("off@a.edu", "Ada", "Lovelace")[0])
        self.assertIn("off@a.edu", index.by_email)

    def test_a_missing_export_is_an_error_not_an_empty_index(self):
        # load_dblp_overrides returns {} for a missing file, which is right
        # there and catastrophic here: an empty index prunes the entire roster.
        with self.assertRaises(FileNotFoundError) as caught:
            pc_membership.load_pc_accounts(str(Path(tempfile.mkdtemp()) / "gone.csv"))
        self.assertIn("--no-pc-check", str(caught.exception))

    def test_an_export_with_no_pc_accounts_is_refused(self):
        # A truncated or half-downloaded export parses perfectly and marks
        # nobody pc. Accepting it would drop every reviewer and report every
        # paper unstaffed -- the single most destructive thing this can do.
        with self.assertRaises(ValueError) as caught:
            self.index([{"email": "author@a.edu", "roles": ""}])
        self.assertIn("truncated", str(caught.exception))

    def test_a_name_shared_by_two_pc_accounts_still_matches(self):
        # Common names collide. A collision resolves toward keeping the person,
        # because a false keep leaves the roster as it is today while a false
        # prune silently removes a sitting PC member.
        index = self.index([
            pc_account("b@a.edu", "Ada", "Lovelace"),
            pc_account("a@a.edu", "Ada", "Lovelace"),
        ])
        acct, how = index.match("third@a.edu", "Ada", "Lovelace")
        self.assertEqual("name", how)
        self.assertEqual("a@a.edu", acct.email)  # deterministic, lowest email

    def test_hotcrp_email_for_uses_the_account_only_when_matched_indirectly(self):
        index = self.index([pc_account("real@b.edu", "Ada", "Lovelace")])
        acct, how = index.match("real@b.edu", "Ada", "Lovelace")
        self.assertEqual("real@b.edu", pc_membership.hotcrp_email_for("real@b.edu", acct, how))
        acct, how = index.match("other@a.edu", "Ada", "Lovelace")
        self.assertEqual("real@b.edu", pc_membership.hotcrp_email_for("other@a.edu", acct, how))
        # No match at all: the row would already have been dropped, but the
        # helper still has a defined, harmless answer -- the input unchanged.
        self.assertEqual("nobody@a.edu", pc_membership.hotcrp_email_for("nobody@a.edu", None, ""))

    def test_collapse_by_hotcrp_email_keeps_the_higher_ranked_duplicate(self):
        old = reviewer_with_email("old@a.edu", hotcrp_email="real@b.edu")
        new = reviewer_with_email("new@a.edu", hotcrp_email="real@b.edu")
        other = reviewer_with_email("other@a.edu", hotcrp_email="other@a.edu")
        survivors, collapsed = pc_membership.collapse_by_hotcrp_email(
            [old, new, other], {"old@a.edu": 1, "new@a.edu": 2, "other@a.edu": 0}
        )
        self.assertEqual(["new@a.edu", "other@a.edu"], sorted(r.email for r in survivors))
        self.assertEqual([("new@a.edu", "old@a.edu")], collapsed)

    def test_collapse_by_hotcrp_email_without_ranks_keeps_the_first(self):
        first = reviewer_with_email("first@a.edu", hotcrp_email="real@b.edu")
        second = reviewer_with_email("second@a.edu", hotcrp_email="real@b.edu")
        survivors, collapsed = pc_membership.collapse_by_hotcrp_email([first, second])
        self.assertEqual(["first@a.edu"], [r.email for r in survivors])
        self.assertEqual([("first@a.edu", "second@a.edu")], collapsed)

    def test_collapse_by_hotcrp_email_is_a_no_op_with_no_duplicates(self):
        a = reviewer_with_email("a@a.edu")
        b = reviewer_with_email("b@a.edu")
        survivors, collapsed = pc_membership.collapse_by_hotcrp_email([a, b])
        self.assertEqual([a, b], survivors)
        self.assertEqual([], collapsed)


class ReviewerLoaderTests(unittest.TestCase):
    """reviewers.py: the acceptance form plus the HotCRP membership check."""

    HEADERS = [
        "Timestamp", "Please confirm your HotCRP email address", "PC membership",
        "First Name", "Last Name", "Enter your DBLP Link", "institutional affiliation",
        "primary area", "secondary area", "tertiary area", "keywords",
        "Override paper assignment number",
    ]

    def load(self, rows, accounts, *, check=True):
        tmp = Path(tempfile.mkdtemp())
        form = tmp / "form.csv"
        with form.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.HEADERS)
            writer.writerows(rows)
        pcinfo = str(write_pcinfo(tmp / "pc.csv", accounts)) if check else None
        with contextlib.redirect_stderr(io.StringIO()):
            return reviewers_mod.load_reviewers(
                str(form), str(tmp / "missing-overrides.csv"), pcinfo_path=pcinfo
            )

    def row(self, email, membership="Yes, I accept as a full PC member",
            first="Ada", last="Lovelace", override="", timestamp="07/01/2026 10:00:00"):
        return [timestamp, email, membership, first, last, "none",
                "Example", "Memory", "", "", "", override]

    def test_an_acceptance_whose_account_lost_the_pc_role_is_dropped(self):
        people = self.load(
            [self.row("gone@a.edu"), self.row("here@a.edu", first="Alan", last="Turing")],
            [pc_account("here@a.edu", "Alan", "Turing"),
             {"email": "gone@a.edu", "given_name": "Ada", "family_name": "Lovelace",
              "roles": "", "tags": "pc-full"}],
        )
        self.assertEqual(["here@a.edu"], [r.email for r in people])

    def test_a_reviewer_matched_by_name_gets_the_accounts_own_hotcrp_email(self):
        # accept.csv:form@a.edu and pcinfo:real@b.edu are the same person, found
        # by name -- a bulk upload has to use real@b.edu, the one HotCRP has
        # marked pc, or it is rejected the same way this whole fix exists for.
        people = self.load(
            [self.row("form@a.edu")],
            [pc_account("real@b.edu", "Ada", "Lovelace")],
        )
        self.assertEqual("form@a.edu", people[0].email)
        self.assertEqual("real@b.edu", people[0].hotcrp_email)

    def test_a_direct_email_match_keeps_hotcrp_email_equal_to_email(self):
        people = self.load(
            [self.row("here@a.edu", first="Alan", last="Turing")],
            [pc_account("here@a.edu", "Alan", "Turing")],
        )
        self.assertEqual("here@a.edu", people[0].hotcrp_email)

    def test_no_pc_check_leaves_hotcrp_email_equal_to_email(self):
        people = self.load([self.row("form@a.edu")], [], check=False)
        self.assertEqual("form@a.edu", people[0].hotcrp_email)

    def test_two_submissions_under_different_real_addresses_collapse_to_one(self):
        # The Rohan Basu Roy case: two real, accepted, non-declined form
        # submissions under two different addresses that both resolve by
        # name to the one PC account. Left uncollapsed, the assignment stage
        # would see two distinct candidates and could hand the same real
        # person two reviews on the same paper.
        people = self.load(
            [
                self.row("old@a.edu", timestamp="05/09/2026 00:35:15"),
                self.row("new@a.edu", timestamp="05/28/2026 21:38:58"),
            ],
            [pc_account("new@a.edu", "Ada", "Lovelace")],
        )
        self.assertEqual(["new@a.edu"], [r.email for r in people])

    def test_pcinfo_path_none_keeps_the_pre_check_roster(self):
        # The --no-pc-check contract: the loader must be able to answer "who
        # accepted", which is what audit_pc_roster.py needs to report the drops.
        people = self.load([self.row("gone@a.edu")], [], check=False)
        self.assertEqual(["gone@a.edu"], [r.email for r in people])

    def test_a_decline_is_never_reported_as_a_removal(self):
        # Order of operations: dedupe, then decline, then the PC check. A
        # decliner who also lost their HotCRP role must be counted once, as a
        # decline -- they are not a roster removal anyone needs to look into.
        dropped = []
        with mock.patch.object(pc_membership, "report_pruned",
                               side_effect=lambda d, *a, **k: dropped.extend(d)):
            people = self.load(
                [self.row("no@a.edu", "No, I am unable to accept"),
                 self.row("yes@a.edu", first="Alan", last="Turing")],
                [pc_account("yes@a.edu", "Alan", "Turing")],
            )
        self.assertEqual(["yes@a.edu"], [r.email for r in people])
        self.assertEqual([], dropped)

    def test_a_removed_reviewer_with_a_bad_override_cap_does_not_break_the_load(self):
        # _parse_override_cap raises on a non-numeric cell, by design. Someone
        # who is off the PC must be dropped before that runs, or one stale form
        # cell takes down every script in the pipeline.
        people = self.load(
            [self.row("gone@a.edu", override="two"),
             self.row("here@a.edu", first="Alan", last="Turing")],
            [pc_account("here@a.edu", "Alan", "Turing")],
        )
        self.assertEqual(["here@a.edu"], [r.email for r in people])

    def test_an_override_for_a_removed_reviewer_does_not_warn_about_a_typo(self):
        # The unmatched-override warning says "typo, or they declined?" -- which
        # would send someone hunting for a mistake that isn't there when the
        # real reason is that HotCRP dropped the person.
        tmp = Path(tempfile.mkdtemp())
        form = tmp / "form.csv"
        with form.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.HEADERS)
            writer.writerow(self.row("gone@a.edu"))
        overrides = tmp / "overrides.csv"
        with overrides.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["email", "dblp", "note"])
            writer.writeheader()
            writer.writerow({"email": "gone@a.edu", "dblp": "https://dblp.org/pid/1/A",
                             "note": ""})
        pcinfo = write_pcinfo(tmp / "pc.csv", [pc_account("other@a.edu", "Alan", "Turing")])
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            reviewers_mod.load_reviewers(str(form), str(overrides), pcinfo_path=str(pcinfo))
        # The drop itself must be reported -- asserting only the absence of
        # "typo" would also pass if nothing were written to stderr at all.
        self.assertIn("gone@a.edu", stderr.getvalue())
        self.assertNotIn("typo", stderr.getvalue())

    def test_an_ungated_call_never_flags_a_no_form_row_override_as_a_typo(self):
        # area_chairs.py's _profile_donors calls load_reviewers with
        # pcinfo_path=None deliberately -- it wants a donor pool, not a
        # membership claim -- which means _pc_members_without_a_form_row never
        # runs there. A dblp_overrides.csv row for a real no-form-row PC
        # member (the exact case that mechanism exists for) must not be
        # reported as an unmatched typo just because this particular call
        # never had the HotCRP data to explain it.
        tmp = Path(tempfile.mkdtemp())
        form = tmp / "form.csv"
        with form.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.HEADERS)
        overrides = tmp / "overrides.csv"
        with overrides.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["email", "dblp", "note"])
            writer.writeheader()
            writer.writerow({"email": "new@a.edu", "dblp": "https://dblp.org/pid/1/A",
                             "note": ""})
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            people = reviewers_mod.load_reviewers(str(form), str(overrides), pcinfo_path=None)
        self.assertEqual([], people)
        self.assertNotIn("typo", stderr.getvalue())

    def test_the_forms_own_cell_still_sets_the_cap_with_no_curated_override(self):
        tmp = Path(tempfile.mkdtemp())
        form = tmp / "form.csv"
        with form.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.HEADERS)
            writer.writerow(self.row("here@a.edu", first="Alan", last="Turing", override="4"))
        pcinfo = write_pcinfo(tmp / "pc.csv", [pc_account("here@a.edu", "Alan", "Turing")])
        with contextlib.redirect_stderr(io.StringIO()):
            people = reviewers_mod.load_reviewers(
                str(form), str(tmp / "missing-dblp-overrides.csv"), pcinfo_path=str(pcinfo),
                cap_overrides_path=str(tmp / "missing-cap-overrides.csv"),
            )
        self.assertEqual(4, people[0].override_cap)

    def test_a_curated_cap_override_wins_over_the_forms_own_cell(self):
        # reviewer_cap_overrides.csv is the durable, re-export-proof layer --
        # same relationship dblp_overrides.csv has to the form's DBLP column.
        tmp = Path(tempfile.mkdtemp())
        form = tmp / "form.csv"
        with form.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.HEADERS)
            writer.writerow(self.row("here@a.edu", first="Alan", last="Turing", override="7"))
        pcinfo = write_pcinfo(tmp / "pc.csv", [pc_account("here@a.edu", "Alan", "Turing")])
        cap_overrides = tmp / "cap_overrides.csv"
        with cap_overrides.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["email", "cap", "note"])
            writer.writeheader()
            writer.writerow({"email": "here@a.edu", "cap": "4", "note": ""})
        with contextlib.redirect_stderr(io.StringIO()):
            people = reviewers_mod.load_reviewers(
                str(form), str(tmp / "missing-dblp-overrides.csv"), pcinfo_path=str(pcinfo),
                cap_overrides_path=str(cap_overrides),
            )
        self.assertEqual(4, people[0].override_cap)


class CapOverrideLoaderTests(unittest.TestCase):
    """reviewers.py: reviewer_cap_overrides.csv, the durable per-person load cap
    covering both PC members and reserve reviewers.
    """

    def write(self, tmp, rows):
        path = tmp / "cap_overrides.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["email", "cap", "note"])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return path

    def test_a_missing_file_has_no_overrides(self):
        tmp = Path(tempfile.mkdtemp())
        self.assertEqual({}, reviewers_mod.load_cap_overrides(str(tmp / "absent.csv")))

    def test_a_blank_email_or_cap_is_skipped(self):
        tmp = Path(tempfile.mkdtemp())
        path = self.write(tmp, [
            {"email": "", "cap": "4", "note": ""},
            {"email": "a@x.edu", "cap": "", "note": "not decided yet"},
            {"email": "b@x.edu", "cap": "4", "note": ""},
        ])
        self.assertEqual({"b@x.edu": 4}, reviewers_mod.load_cap_overrides(str(path)))

    def test_a_bad_cap_fails_loudly_with_the_email(self):
        tmp = Path(tempfile.mkdtemp())
        path = self.write(tmp, [{"email": "a@x.edu", "cap": "two", "note": ""}])
        with self.assertRaisesRegex(ValueError, "a@x.edu"):
            reviewers_mod.load_cap_overrides(str(path))

    def test_a_negative_cap_is_rejected(self):
        tmp = Path(tempfile.mkdtemp())
        path = self.write(tmp, [{"email": "a@x.edu", "cap": "-1", "note": ""}])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            reviewers_mod.load_cap_overrides(str(path))


class PcMembersWithoutFormRowTests(unittest.TestCase):
    """reviewers.py: a PC account HotCRP knows about that the acceptance form
    never explains -- someone added straight to HotCRP after the form closed.
    """

    def load(self, accounts, *, form_rows=None, dblp_overrides=None, area_overrides=None,
             reserve_upload=None, cap_overrides=None):
        tmp = Path(tempfile.mkdtemp())
        form = tmp / "form.csv"
        with form.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(ReviewerLoaderTests.HEADERS)
            writer.writerows(form_rows or [])
        pcinfo = write_pcinfo(tmp / "pc.csv", accounts)

        overrides_path = tmp / "dblp_overrides.csv"
        with overrides_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["email", "dblp", "note"])
            writer.writeheader()
            for email, pid in (dblp_overrides or {}).items():
                writer.writerow({"email": email, "dblp": pid, "note": ""})

        area_overrides_path = tmp / "pc_area_overrides.csv"
        with area_overrides_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["email", "primary", "secondary", "tertiary", "note"]
            )
            writer.writeheader()
            for email, areas in (area_overrides or {}).items():
                primary, secondary, tertiary = areas
                writer.writerow({"email": email, "primary": primary,
                                 "secondary": secondary, "tertiary": tertiary, "note": ""})

        reserve_upload_path = tmp / "reserve_reviewer_upload.csv"
        with reserve_upload_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["email", "name", "roles", "tags"])
            writer.writeheader()
            for email in (reserve_upload or []):
                writer.writerow({"email": email, "name": "", "roles": "pc", "tags": ""})

        cap_overrides_path = tmp / "reviewer_cap_overrides.csv"
        with cap_overrides_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["email", "cap", "note"])
            writer.writeheader()
            for email, cap in (cap_overrides or {}).items():
                writer.writerow({"email": email, "cap": cap, "note": ""})

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            people = reviewers_mod.load_reviewers(
                str(form), str(overrides_path), pcinfo_path=str(pcinfo),
                area_overrides_path=str(area_overrides_path),
                reserve_upload_path=str(reserve_upload_path),
                cap_overrides_path=str(cap_overrides_path),
            )
        return people, stderr.getvalue()

    def test_a_tagged_account_gets_a_reviewer_from_hotcrp_tags_and_topics(self):
        people, _ = self.load(
            [pc_account("new@a.edu", "Grace", "Hopper", tags="pc-full",
                        **{"topic: Memory Systems": "2", "topic: Microarchitecture": "1"})],
            dblp_overrides={"new@a.edu": "https://dblp.org/pid/1/Test"},
        )
        self.assertEqual(["new@a.edu"], [p.email for p in people])
        p = people[0]
        self.assertEqual("full", p.tier)
        self.assertEqual("1/Test", p.pid)
        self.assertEqual("Memory Systems", p.primary)
        self.assertEqual("Microarchitecture", p.secondary)

    def test_a_missing_dblp_link_still_yields_a_record(self):
        # Same tolerance the form itself has for a blank DBLP cell:
        # classify_reviewers.py's existing stub-appender is what turns this
        # into a to-do, not an exclusion here.
        people, stderr = self.load(
            [pc_account("new@a.edu", "Grace", "Hopper", tags="pc-light",
                        **{"topic: Memory Systems": "2"})],
        )
        self.assertEqual(["new@a.edu"], [p.email for p in people])
        self.assertIsNone(people[0].pid)
        self.assertIn("new@a.edu", stderr)

    def test_a_manual_area_override_wins_over_topic_interest(self):
        people, _ = self.load(
            [pc_account("new@a.edu", "Grace", "Hopper", tags="pc-full",
                        **{"topic: Memory Systems": "2"})],
            area_overrides={"new@a.edu": ("Security", "GPUs", "")},
        )
        self.assertEqual("Security", people[0].primary)
        self.assertEqual("GPUs", people[0].secondary)

    def test_no_topic_interest_and_no_area_override_is_reported_not_invented(self):
        people, stderr = self.load(
            [pc_account("new@a.edu", "Grace", "Hopper", tags="pc-full")],
        )
        self.assertEqual(["new@a.edu"], [p.email for p in people])
        self.assertEqual("", people[0].primary)
        self.assertIn("new@a.edu", stderr)

    def test_an_unsettled_tier_is_excluded_with_a_warning(self):
        # Neither tag, or both, is not a tier -- the same rule applied to an
        # untiered ~~ex-rr account: never invent a commitment level.
        people, stderr = self.load(
            [pc_account("new@a.edu", "Grace", "Hopper", tags="pc-full pc-light",
                        **{"topic: Memory Systems": "2"})],
        )
        self.assertEqual([], people)
        self.assertIn("new@a.edu", stderr)
        self.assertIn("not settled", stderr)

    def test_a_chair_trc_sysadmin_or_area_chair_is_never_swept_in(self):
        people, _ = self.load(
            [pc_account("chair@a.edu", "Ada", "Lovelace", roles="chair pc", tags="pc-full"),
             pc_account("trc@a.edu", "Alan", "Turing", tags="trc pc-full"),
             pc_account("admin@a.edu", "Grace", "Hopper", roles="sysadmin pc", tags="pc-full"),
             pc_account("ac@a.edu", "Edsger", "Dijkstra", tags="pc-full ~~area-chairs")],
        )
        self.assertEqual([], people)

    def test_an_ex_reserve_is_left_for_the_reserve_loader(self):
        # Promoting a reserve is load_reserve_reviewers's job -- it derives
        # areas from authored papers and keeps the reserve-side fingerprint,
        # neither of which this supplement can reconstruct.
        people, _ = self.load(
            [pc_account("promoted@a.edu", "Grace", "Hopper", tags="pc-full ~~ex-rr")],
        )
        self.assertEqual([], people)

    def test_an_account_with_a_form_row_is_not_duplicated(self):
        accepted_row = ["07/01/2026 10:00:00", "here@a.edu",
                        "Yes, I accept as a full PC member", "Alan", "Turing", "none",
                        "Example", "Memory", "", "", "", ""]
        people, _ = self.load(
            [pc_account("here@a.edu", "Alan", "Turing", tags="pc-full")],
            form_rows=[accepted_row],
        )
        self.assertEqual(["here@a.edu"], [p.email for p in people])

    def test_an_untagged_account_is_silently_not_a_candidate(self):
        # This loader only ever sees the PC acceptance-form roster, so an
        # account with no pc-full/pc-light tag at all is, on the live data,
        # almost always a reserve reviewer this loader cannot recognise as
        # already accounted for elsewhere -- not a real anomaly to warn about.
        # audit_pc_roster.py's no_roster_row category is the tag-agnostic
        # backstop that still catches a genuine no-form, no-tag PC addition.
        people, stderr = self.load(
            [pc_account("reserve-like@a.edu", "Grace", "Hopper")],
        )
        self.assertEqual([], people)
        self.assertNotIn("reserve-like@a.edu", stderr)

    def test_a_cap_override_reaches_a_no_form_row_addition(self):
        # There is no form cell to compare against here, so the curated file
        # is simply the only source -- unlike a form row, where it wins.
        people, _ = self.load(
            [pc_account("new@a.edu", "Grace", "Hopper", tags="pc-full",
                        **{"topic: Memory Systems": "2"})],
            cap_overrides={"new@a.edu": "3"},
        )
        self.assertEqual(3, people[0].override_cap)

    def test_a_raw_reserve_upload_recruit_is_not_mistaken_for_a_pc_addition(self):
        # Observed on the live export: HotCRP had already tagged an account
        # pc-light although it originated as a reserve-upload row whose DBLP
        # identity build_reserve_reviewer_info.py never resolved. Building a
        # Reviewer here would derive their profile from the wrong roster's
        # rules (HotCRP topic interest instead of authored-paper topics,
        # dblp_overrides.csv instead of the vetting workbook).
        people, _ = self.load(
            [pc_account("recruit@a.edu", "Grace", "Hopper", tags="pc-light")],
            reserve_upload=["recruit@a.edu"],
        )
        self.assertEqual([], people)


class AreaChairExclusionTests(unittest.TestCase):
    """An area chair chairs papers and reviews none.

    Membership is the union of HotCRP's `~~area-chairs` tag and the acceptance
    form. On the live export those are 21 and 18 people; the union is 21 and
    each source has been observed to catch someone the other missed, so neither
    alone is sufficient.
    """

    AC_HEADERS = [
        "Timestamp", "Please confirm your HotCRP email address",
        "Area Chair membership", "First Name", "Last Name",
        "Enter your DBLP Link", "institutional affiliation",
        "primary area", "keywords", "secondary area",
    ]
    ACCEPT = "Yes, I accept the role of being an Area Chair for HPCA 2027"

    def ac_row(self, email, membership=None, first="Ada", last="Lovelace"):
        return ["07/01/2026 10:00:00", email, membership or self.ACCEPT, first, last,
                "https://dblp.org/pid/1/Test", "Example", "Memory", "caches", "Security"]

    def chair_emails(self, form_rows, accounts):
        """The exclusion set, from a fake form and a fake user export."""
        tmp = Path(tempfile.mkdtemp())
        form = tmp / "chairs.csv"
        with form.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.AC_HEADERS)
            writer.writerows(form_rows)
        pcinfo = write_pcinfo(tmp / "pcinfo.csv", accounts)
        with contextlib.redirect_stderr(io.StringIO()):
            return area_chairs.area_chair_emails(
                str(form), str(tmp / "missing.csv"), pcinfo_path=str(pcinfo)
            )

    def person(self, email, first="Ada", last="Lovelace", tier="full"):
        return Reviewer(
            email=email, first=first, last=last, dblp_url="", pid="1/Test",
            affiliation="Example", primary="Memory", secondary="", tertiary="",
            keywords="", tier=tier, override_cap=None,
        )

    def test_the_tag_and_the_form_each_catch_someone_the_other_misses(self):
        # The whole requirement in one test. Both directions have occurred on
        # the live data: accounts tagged in HotCRP that never returned the form,
        # and a form signatory the tag did not yet cover.
        found = self.chair_emails(
            [self.ac_row("onform@a.edu"), self.ac_row("both@a.edu")],
            [pc_account("tagged@a.edu", "Grace", "Hopper", tags="pc-full ~~area-chairs"),
             pc_account("both@a.edu", "Ada", "Lovelace", tags="pc-full ~~area-chairs"),
             pc_account("onform@a.edu", "Ada", "Lovelace", tags="pc-full"),
             pc_account("neither@a.edu", "Alan", "Turing", tags="pc-light")],
        )
        self.assertEqual({"onform@a.edu", "both@a.edu", "tagged@a.edu"}, found)

    def test_a_twiddle_tag_is_recognised_and_another_twiddle_tag_is_not(self):
        # HotCRP writes the chair-only tag as `~~area-chairs`, so the test is
        # endswith, never ==, which would match nobody. `~~rr-batch2` on the
        # real export proves other twiddle tags exist.
        self.assertTrue(pc_membership.Account({"tags": "pc-full ~~area-chairs"}).is_area_chair)
        self.assertFalse(pc_membership.Account({"tags": "pc-full ~~rr-batch2"}).is_area_chair)
        self.assertFalse(pc_membership.Account({"tags": "pc-full"}).is_area_chair)
        self.assertFalse(pc_membership.Account({"tags": ""}).is_area_chair)

    def test_the_area_chair_tag_has_one_definition(self):
        # pc_membership owns the test; audit_pc_roster.missing_category must not
        # grow a second copy that can drift from it.
        acct = pc_membership.Account(
            {"email": "c@a.edu", "roles": "pc", "tags": "pc-full ~~area-chairs"}
        )
        self.assertTrue(acct.is_area_chair)
        self.assertEqual("area-chair", audit_pc_roster.missing_category(acct, set(), {}, {})[0])

    def test_an_area_chair_is_dropped_from_the_reviewer_pool(self):
        pool = {"chair@a.edu": self.person("chair@a.edu"),
                "reviewer@a.edu": self.person("reviewer@a.edu")}
        dropped = area_chairs.drop_area_chairs(pool, {"chair@a.edu"})
        self.assertEqual(["chair@a.edu"], dropped)
        self.assertEqual(["reviewer@a.edu"], sorted(pool))

    def test_a_reserve_area_chair_is_dropped_too(self):
        # The ordering guard. Reserves are merged into reviewers_by_email AFTER
        # the PC roster is built, so a filter applied to the PC load alone is
        # undone by that merge. One person on the live data is exactly this
        # case: tagged an area chair while sitting on the reserve roster.
        pool = {"chair@a.edu": self.person("chair@a.edu", tier="reserve"),
                "reserve@a.edu": self.person("reserve@a.edu", tier="reserve")}
        dropped = area_chairs.drop_area_chairs(pool, {"chair@a.edu"})
        self.assertEqual(["chair@a.edu"], dropped)
        self.assertEqual(["reserve@a.edu"], sorted(pool))

    def test_a_chair_reviewing_under_a_second_address_is_dropped(self):
        # Accepting from one address while holding the HotCRP account under
        # another is ordinary here. Comparing raw addresses would let a tagged
        # chair keep reviewing under their other identity.
        tmp = Path(tempfile.mkdtemp())
        pcinfo = write_pcinfo(tmp / "pcinfo.csv", [
            pc_account("ada@b.edu", "Ada", "Lovelace", tags="pc-full ~~area-chairs"),
        ])
        index = pc_membership.load_pc_accounts(str(pcinfo))
        pool = {"ada@a.edu": self.person("ada@a.edu", "Ada", "Lovelace")}
        dropped = area_chairs.drop_area_chairs(pool, {"ada@b.edu"}, index)
        self.assertEqual(["ada@a.edu"], dropped)
        self.assertEqual({}, pool)

    def test_the_form_half_is_read_without_the_membership_gate(self):
        # For a roster, a false prune drops someone no longer on the committee.
        # For an exclusion set the safe direction inverts: a false prune
        # silently readmits an area chair to the reviewer pool.
        found = self.chair_emails(
            [self.ac_row("chair@a.edu")],
            [pc_account("someone@a.edu", "Alan", "Turing")],
        )
        self.assertIn("chair@a.edu", found)

    def test_a_declined_area_chair_can_still_review(self):
        # Reuses load_area_chairs' acceptance semantics rather than scraping
        # addresses: declining to chair is not being a chair.
        found = self.chair_emails(
            [self.ac_row("no@a.edu", "No, I would prefer to be a PC full or light member")],
            [pc_account("no@a.edu", "Ada", "Lovelace", tags="pc-full")],
        )
        self.assertEqual(set(), found)

    def test_a_missing_area_chair_form_is_an_error_not_an_empty_set(self):
        # Same reasoning as the missing user export: an empty exclusion set is
        # indistinguishable from "no area chairs", and silently reinstates them.
        tmp = Path(tempfile.mkdtemp())
        pcinfo = write_pcinfo(tmp / "pcinfo.csv", [pc_account("a@a.edu")])
        with self.assertRaises(FileNotFoundError):
            area_chairs.area_chair_emails(
                str(tmp / "absent.csv"), str(tmp / "missing.csv"), pcinfo_path=str(pcinfo)
            )

    def test_a_tagged_chair_with_no_form_row_borrows_a_profile(self):
        # They are barred from reviewing either way, so without this they would
        # do neither job. The profile comes from whichever roster has them --
        # on the live data the PC form for one, the reserve roster for another.
        tmp = Path(tempfile.mkdtemp())
        form = tmp / "chairs.csv"
        with form.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.AC_HEADERS)
            writer.writerows([self.ac_row("onform@a.edu")])
        pcinfo = write_pcinfo(tmp / "pcinfo.csv", [
            pc_account("onform@a.edu", "Ada", "Lovelace", tags="pc-full ~~area-chairs"),
            pc_account("tagged@a.edu", "Grace", "Hopper", tags="pc-full ~~area-chairs"),
        ])
        donor = self.person("tagged@a.edu", "Grace", "Hopper")
        with contextlib.redirect_stderr(io.StringIO()):
            chairs = area_chairs.load_area_chairs(
                str(form), str(tmp / "missing.csv"),
                pcinfo_path=str(pcinfo), supplement=[donor],
            )
        self.assertEqual(["onform@a.edu", "tagged@a.edu"], sorted(c.email for c in chairs))
        borrowed = next(c for c in chairs if c.email == "tagged@a.edu")
        self.assertEqual("1/Test", borrowed.pid)
        self.assertEqual("Memory", borrowed.primary)

    def test_a_tagged_chair_on_no_roster_is_reported_not_emitted(self):
        # A record with no PID cannot be fingerprinted, so emitting one just
        # moves the failure somewhere further from the cause.
        tmp = Path(tempfile.mkdtemp())
        form = tmp / "chairs.csv"
        with form.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.AC_HEADERS)
        pcinfo = write_pcinfo(tmp / "pcinfo.csv", [
            pc_account("nowhere@a.edu", "Grace", "Hopper", tags="pc-full ~~area-chairs"),
        ])
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            chairs = area_chairs.load_area_chairs(
                str(form), str(tmp / "missing.csv"), pcinfo_path=str(pcinfo), supplement=[],
            )
        self.assertEqual([], chairs)
        self.assertIn("nowhere@a.edu", stderr.getvalue())

    def test_a_pc_chair_is_never_an_area_chair_even_if_they_accept(self):
        # A PC chair already makes the final call on every paper, so chairing
        # one too is a conflict, not a second job -- even though they
        # accepted the area-chair form. structural_role reads this off
        # HotCRP's own roles="chair pc" / tags="chairs", the same signal that
        # already keeps a PC chair from being auto-added as a plain reviewer.
        tmp = Path(tempfile.mkdtemp())
        form = tmp / "chairs.csv"
        with form.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.AC_HEADERS)
            writer.writerows([
                self.ac_row("chair@a.edu", first="Paul", last="Gratz"),
                self.ac_row("onform@a.edu"),
            ])
        pcinfo = write_pcinfo(tmp / "pcinfo.csv", [
            pc_account("chair@a.edu", "Paul", "Gratz", roles="chair pc", tags="chairs"),
            pc_account("onform@a.edu", "Ada", "Lovelace", tags="pc-full"),
        ])
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            chairs = area_chairs.load_area_chairs(
                str(form), str(tmp / "missing.csv"), pcinfo_path=str(pcinfo), supplement=[],
            )
        self.assertEqual(["onform@a.edu"], [c.email for c in chairs])
        self.assertIn("chair@a.edu", stderr.getvalue())

    def test_a_pc_chair_tagged_area_chair_with_no_form_row_is_still_skipped(self):
        # Future-proofing: neither PC chair carries ~~area-chairs today, but
        # if one someday does, the tag-supplement path must not add them back.
        tmp = Path(tempfile.mkdtemp())
        form = tmp / "chairs.csv"
        with form.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.AC_HEADERS)
        pcinfo = write_pcinfo(tmp / "pcinfo.csv", [
            pc_account("chair@a.edu", "Paul", "Gratz", roles="chair pc",
                       tags="chairs ~~area-chairs"),
        ])
        donor = self.person("chair@a.edu", "Paul", "Gratz")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            chairs = area_chairs.load_area_chairs(
                str(form), str(tmp / "missing.csv"), pcinfo_path=str(pcinfo), supplement=[donor],
            )
        self.assertEqual([], chairs)
        self.assertIn("chair@a.edu", stderr.getvalue())

    def test_area_chair_emails_still_excludes_a_pc_chair_as_a_reviewer(self):
        # load_area_chairs no longer treats them as a chair, but the
        # reviewer-pool exclusion set (area_chair_emails) is separate and
        # deliberately untouched: a PC chair still shouldn't review either,
        # and this confirms that wasn't broken by the roster-side fix.
        found = self.chair_emails(
            [self.ac_row("chair@a.edu", first="Paul", last="Gratz")],
            [pc_account("chair@a.edu", "Paul", "Gratz", roles="chair pc", tags="chairs")],
        )
        self.assertIn("chair@a.edu", found)


class RosterDispatchTests(unittest.TestCase):
    """roster.py: every script must agree on who is on the PC."""

    def test_load_roster_prunes_the_same_people_as_the_direct_loader(self):
        # Seven of the eleven roster consumers bypass load_roster and call the
        # loaders directly. If the gate lived in the dispatcher, those seven
        # would keep people the others dropped -- and a reviewer present in the
        # assignment but absent from reviewer_seniority.csv can fill slots while
        # never satisfying the senior requirement.
        tmp = Path(tempfile.mkdtemp())
        form = tmp / "form.csv"
        with form.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(ReviewerLoaderTests.HEADERS)
            writer.writerow(["07/01/2026 10:00:00", "gone@a.edu",
                             "Yes, I accept as a full PC member", "Ada", "Lovelace",
                             "none", "Example", "Memory", "", "", "", ""])
            writer.writerow(["07/01/2026 10:00:00", "here@a.edu",
                             "Yes, I accept as a light PC member", "Alan", "Turing",
                             "none", "Example", "Memory", "", "", "", ""])
        pcinfo = str(write_pcinfo(tmp / "pc.csv", [pc_account("here@a.edu", "Alan", "Turing")]))
        with contextlib.redirect_stderr(io.StringIO()):
            direct = reviewers_mod.load_reviewers(str(form), str(tmp / "none.csv"),
                                                  pcinfo_path=pcinfo)
            dispatched = roster_mod.load_roster("reviewer", str(form), pcinfo_path=pcinfo)
        self.assertEqual(["here@a.edu"], [r.email for r in direct])
        self.assertEqual([r.email for r in direct], [r.email for r in dispatched])

    def test_load_roster_forwards_the_pcinfo_path_to_every_role(self):
        # An unforwarded path silently means "use the default export", which is
        # how one role ends up checked against a different file from the others.
        seen = {}
        for role, module, name in (
            ("reviewer", roster_mod, "load_reviewers"),
            ("area-chair", roster_mod, "load_area_chairs"),
            ("reserve", roster_mod, "load_reserve_reviewers"),
        ):
            with mock.patch.object(
                module, name,
                side_effect=lambda *a, pcinfo_path=None, **k: seen.__setitem__(role, pcinfo_path) or [],
            ):
                roster_mod.load_roster(role, "roster.csv", pcinfo_path="sentinel.csv")
        self.assertEqual({"reviewer": "sentinel.csv", "area-chair": "sentinel.csv",
                          "reserve": "sentinel.csv"}, seen)


class ReserveMembershipTests(unittest.TestCase):
    """reserve_reviewers.py: reserves face the same HotCRP check as the PC."""

    def load(self, roster, accounts):
        with contextlib.redirect_stderr(io.StringIO()):
            return ReserveReviewerLoaderTests.load(
                ReserveReviewerLoaderTests(), roster,
                [{"pid": 1, "topics": ["Memory"], "reserve_reviewer": "",
                  "authors": [], "contacts": []}],
                pcinfo=accounts,
            )

    def test_a_reserve_who_kept_the_tag_but_lost_the_role_is_dropped(self):
        # A stood-down reserve keeps the reserve-reviewer tag on their account,
        # so the roster file alone cannot tell that the pc role is gone.
        people = self.load(
            [("gone@a.edu", "Ada Lovelace", ""), ("here@a.edu", "Alan Turing", "")],
            [pc_account("here@a.edu", "Alan", "Turing"),
             {"email": "gone@a.edu", "given_name": "Ada", "family_name": "Lovelace",
              "roles": "", "tags": "reserve-reviewer"}],
        )
        self.assertEqual(["here@a.edu"], [r.email for r in people])

    def test_a_reserve_on_the_pc_under_another_address_is_kept(self):
        people = self.load(
            [("recruit@a.edu", "Ada Lovelace", "")],
            [pc_account("ada@b.edu", "Ada", "Lovelace")],
        )
        self.assertEqual(["recruit@a.edu"], [r.email for r in people])


class PromotedReserveTests(unittest.TestCase):
    """`~~ex-rr`: a reserve elevated to the PC loads at their PC tier.

    The promotion lives entirely in HotCRP tags -- these people never filled in
    an acceptance form, so there is no other record of it. Getting it wrong is
    quiet in both directions: missing it caps a PC member at the reserve load
    and drops them from a PC-only run, while inventing one hands somebody a
    fifteen-paper assignment they never agreed to.
    """

    def load(self, roster, accounts):
        with contextlib.redirect_stderr(io.StringIO()):
            return ReserveMembershipTests.load(
                ReserveMembershipTests(), roster, accounts
            )

    def tier_of(self, tags):
        people = self.load(
            [("a@a.edu", "Ada Lovelace", "")],
            [pc_account("a@a.edu", "Ada", "Lovelace", tags=tags)],
        )
        self.assertEqual(1, len(people))
        return people[0].tier

    def test_the_marker_and_a_tier_tag_promote_a_reserve(self):
        self.assertEqual("light", self.tier_of("pc-light ~~ex-rr"))
        self.assertEqual("full", self.tier_of("pc-full ~~ex-rr"))

    def test_a_promoted_reserve_stays_on_the_reserve_roster(self):
        # The tier changes; the roster does not. They never filled in a form, so
        # this file is still the only source of their DBLP page and areas, and
        # `make reserves` has to keep building their fingerprint and seniority.
        people = self.load(
            [("a@a.edu", "Ada Lovelace", "https://dblp.org/pid/1/A.html")],
            [pc_account("a@a.edu", "Ada", "Lovelace", tags="pc-light ~~ex-rr")],
        )
        self.assertEqual(["a@a.edu"], [r.email for r in people])
        self.assertEqual("1/A", people[0].pid)

    def test_every_spelling_of_the_tags_resolves(self):
        # HotCRP writes a tag plain, twiddled for a chairs-only tag, and with an
        # order value appended. All three name the same tag.
        self.assertEqual("light", self.tier_of("~~pc-light ~~ex-rr"))
        self.assertEqual("light", self.tier_of("pc-light#3 ~~ex-rr#1"))
        self.assertEqual("full", self.tier_of("~~PC-FULL ~~EX-RR"))

    def test_an_unsettled_tier_leaves_them_a_reserve(self):
        # Neither tag or both is not a tier. Guessing would assign a load the
        # person never agreed to; staying a reserve merely under-uses them until
        # the tag is fixed, which make pc-roster reports.
        self.assertEqual("reserve", self.tier_of("~~ex-rr"))
        self.assertEqual("reserve", self.tier_of("pc-light pc-full ~~ex-rr"))

    def test_a_tier_tag_without_the_marker_promotes_nobody(self):
        # ~474 accounts carry pc-light/pc-full -- effectively the whole PC. The
        # tier tag is only ever read for an ex-reserve; for everyone else the
        # acceptance form is the authority.
        self.assertEqual("reserve", self.tier_of("pc-light"))
        self.assertEqual("reserve", self.tier_of("pc-full reserve-reviewer"))

    def test_a_containing_tag_is_not_the_tag(self):
        # `endswith` would be wrong here the way it is right for ~~area-chairs:
        # pc-full and pc-light are two answers to one question, not a family.
        self.assertEqual("reserve", self.tier_of("~~ex-pc-full ~~ex-rr"))
        self.assertEqual("reserve", self.tier_of("pc-full ~~not-ex-rr"))

    def test_without_the_export_nobody_is_promoted(self):
        # --no-pc-check: no export means no tags, so the promotion is invisible
        # and everyone stays a reserve. assign_reviewers warns about exactly this.
        with contextlib.redirect_stderr(io.StringIO()):
            people = ReserveReviewerLoaderTests.load(
                ReserveReviewerLoaderTests(),
                [("a@a.edu", "Ada Lovelace", "")],
                [{"pid": 1, "topics": ["Memory"], "reserve_reviewer": "",
                  "authors": [], "contacts": []}],
            )
        self.assertEqual(["reserve"], [r.tier for r in people])

    def test_the_tag_tests_have_one_definition(self):
        acct = pc_membership.Account(
            {"email": "a@a.edu", "roles": "pc", "tags": "pc-light ~~ex-rr"}
        )
        self.assertTrue(acct.is_ex_reserve)
        self.assertEqual("light", acct.pc_tier)
        self.assertIsNone(
            pc_membership.Account({"email": "a@a.edu", "tags": "~~ex-rr"}).pc_tier
        )

    def test_a_promoted_reserve_is_capped_as_a_pc_member(self):
        # The point of the whole change: light_cap, not reserve_cap.
        def rv(tier):
            return Reviewer(
                email="e@x.edu", first="A", last="B", dblp_url="", pid=None,
                affiliation="", primary="", secondary="", tertiary="", keywords="",
                tier=tier, override_cap=None,
            )

        cap = assign_reviewers.reviewer_paper_cap
        self.assertEqual(7, cap(rv("light"), 7, 15, 6))
        self.assertEqual(15, cap(rv("full"), 7, 15, 6))
        self.assertEqual(6, cap(rv("reserve"), 7, 15, 6))

    def person(self, email, tier="reserve", hotcrp_email=""):
        return Reviewer(
            email=email, hotcrp_email=hotcrp_email or email, first="A", last="B",
            dblp_url="", pid=None, affiliation="", primary="", secondary="",
            tertiary="", keywords="", tier=tier, override_cap=None,
        )

    def test_the_promoted_are_split_from_the_reserve_bench(self):
        promoted, already_pc, reserves = assign_reviewers.split_promoted_reserves(
            [self.person("up@a.edu", "light"), self.person("bench@a.edu")], {}
        )
        self.assertEqual(["up@a.edu"], [r.email for r in promoted])
        self.assertEqual(["bench@a.edu"], [r.email for r in reserves])
        self.assertEqual([], already_pc)

    def test_a_promoted_reserve_who_also_filled_the_form_is_not_duplicated(self):
        # The form record carries their own declared areas and a PC fingerprint;
        # the reserve record's areas are inferred. The form wins.
        pool = {"ada@a.edu": self.person("ada@a.edu", "light")}
        promoted, already_pc, _ = assign_reviewers.split_promoted_reserves(
            [self.person("recruit@a.edu", "light", hotcrp_email="ada@a.edu")], pool
        )
        self.assertEqual([], promoted)
        self.assertEqual(["recruit@a.edu"], already_pc)

    def test_the_overlap_is_compared_on_the_hotcrp_address(self):
        # Accepting under one address while holding the account under another is
        # ordinary here; comparing roster addresses would file one person as two
        # and assign them twice.
        pool = {"ada@a.edu": self.person("ada@a.edu", "full", hotcrp_email="ada@b.edu")}
        promoted, already_pc, _ = assign_reviewers.split_promoted_reserves(
            [self.person("recruit@c.edu", "light", hotcrp_email="ada@b.edu")], pool
        )
        self.assertEqual([], promoted)
        self.assertEqual(["recruit@c.edu"], already_pc)


class PcRosterAuditTests(unittest.TestCase):
    """audit_pc_roster.py: both directions of the HotCRP cross-check."""

    def run_audit(self, form_rows, accounts, uploaded=(), reserves=()):
        tmp = Path(tempfile.mkdtemp())
        form, ac_form = tmp / "form.csv", tmp / "ac.csv"
        with form.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(ReviewerLoaderTests.HEADERS)
            writer.writerows(form_rows)
        with ac_form.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Please confirm your HotCRP email address",
                             "Area Chair membership", "First Name", "Last Name",
                             "Enter your DBLP Link", "institutional affiliation",
                             "primary area", "keywords", "secondary area"])
        upload = tmp / "upload.csv"
        with upload.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["email", "name", "roles", "tags"])
            writer.writeheader()
            for email, name in uploaded:
                writer.writerow({"email": email, "name": name, "roles": "pc",
                                 "tags": "reserve-reviewer"})
        info = tmp / "info.csv"
        with info.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["email", "name", "dblp"])
            writer.writeheader()
            for email, name in reserves:
                writer.writerow({"email": email, "name": name, "dblp": ""})
        data = tmp / "papers.json"
        data.write_text(json.dumps([]), encoding="utf-8")
        pruned, missing = tmp / "pruned.csv", tmp / "missing.csv"
        argv = ["audit_pc_roster.py",
                "--pcinfo", str(write_pcinfo(tmp / "pc.csv", accounts)),
                "--csv", str(form), "--area-chair-csv", str(ac_form),
                "--reserve-info", str(info), "--upload", str(upload),
                "--data", str(data), "--pruned", str(pruned),
                "--missing", str(missing)]
        with mock.patch.object(sys, "argv", argv), \
                contextlib.redirect_stderr(io.StringIO()):
            audit_pc_roster.main()

        def read(path):
            with path.open(newline="", encoding="utf-8") as f:
                return list(csv.DictReader(f))

        return read(pruned), read(missing)

    def accept(self, email, first="Ada", last="Lovelace",
               membership="Yes, I accept as a full PC member"):
        return ["07/01/2026 10:00:00", email, membership, first, last, "none",
                "Example", "Memory", "", "", "", ""]

    def test_the_pruned_report_separates_a_missing_account_from_a_lost_role(self):
        pruned, _ = self.run_audit(
            [self.accept("nobody@a.edu", "Grace", "Hopper"),
             self.accept("former@a.edu", "Alan", "Turing")],
            [pc_account("keep@a.edu", "Ada", "Lovelace"),
             {"email": "former@a.edu", "given_name": "Alan", "family_name": "Turing",
              "roles": "", "tags": "pc-full"}],
        )
        by_email = {r["email"]: r for r in pruned}
        self.assertEqual("no_account", by_email["nobody@a.edu"]["problem"])
        self.assertEqual("role_removed", by_email["former@a.edu"]["problem"])
        self.assertIn("pc-full", by_email["former@a.edu"]["detail"])

    def test_an_alternate_account_is_reported_as_missing_and_never_pruned(self):
        # The two-file contract: a person reachable under a second PC-marked
        # address is kept on the roster, and the fact that needs acting on is
        # the un-merged HotCRP account, which belongs in the other report.
        pruned, missing = self.run_audit(
            [self.accept("ada@a.edu", "Ada", "Lovelace")],
            [pc_account("ada@b.edu", "Ada", "Lovelace")],
        )
        self.assertEqual([], pruned)
        self.assertEqual(["ada@b.edu"], [r["email"] for r in missing])
        self.assertEqual("alternate_account", missing[0]["category"])
        self.assertIn("ada@a.edu", missing[0]["detail"])

    def test_a_reserve_upload_row_settles_a_pc_account_with_no_acceptance(self):
        # Reserves never fill in the acceptance form, so without the upload
        # every one of them would be reported as an unexplained PC account.
        _, missing = self.run_audit(
            [], [pc_account("recruit@a.edu", "Grace", "Hopper")],
            uploaded=[("recruit@a.edu", "Grace Hopper")],
        )
        self.assertEqual([], missing)

    def test_a_decliner_still_marked_pc_is_outstanding(self):
        # Zero of these today; the category exists so that when it stops being
        # zero it is not invisible.
        _, missing = self.run_audit(
            [self.accept("no@a.edu", "Grace", "Hopper", "No, I am unable to accept")],
            [pc_account("no@a.edu", "Grace", "Hopper")],
        )
        self.assertEqual(["no@a.edu"], [r["email"] for r in missing])
        self.assertEqual("declined", missing[0]["category"])

    def test_a_structural_role_outranks_an_identity_match(self):
        # A chair holding a second account is on the PC because they chair the
        # conference. Filing them under alternate_account would put a settled
        # row on the to-do list.
        _, missing = self.run_audit(
            [self.accept("chair@a.edu", "Ada", "Lovelace")],
            [pc_account("chair@b.edu", "Ada", "Lovelace", tags="chairs")],
        )
        self.assertEqual("chair", missing[0]["category"])

    def test_a_pc_account_with_no_roster_row_at_all_is_outstanding(self):
        _, missing = self.run_audit([], [pc_account("who@a.edu", "Grace", "Hopper")])
        self.assertEqual("no_roster_row", missing[0]["category"])

    def test_both_reports_are_written_even_when_everything_agrees(self):
        # A report that stops being written goes stale silently; one that
        # converges to a bare header says "checked, nothing found".
        pruned, missing = self.run_audit(
            [self.accept("ada@a.edu", "Ada", "Lovelace")],
            [pc_account("ada@a.edu", "Ada", "Lovelace")],
        )
        self.assertEqual([], pruned)
        self.assertEqual([], missing)

    def test_rows_are_sorted_by_email_so_a_rerun_diffs_cleanly(self):
        pruned, _ = self.run_audit(
            [self.accept("c@a.edu", "Grace", "Hopper"),
             self.accept("a@a.edu", "Alan", "Turing"),
             self.accept("b@a.edu", "Edsger", "Dijkstra")],
            [pc_account("keep@a.edu", "Ada", "Lovelace")],
        )
        self.assertEqual(["a@a.edu", "b@a.edu", "c@a.edu"], [r["email"] for r in pruned])


class DuplicateAccountTests(unittest.TestCase):
    """find_duplicate_accounts.py: ranking candidate same-person pairs."""

    def account(self, email, given, family, orcid="", affiliation=""):
        return pc_membership.Account({
            "email": email, "given_name": given, "family_name": family,
            "orcid": orcid, "affiliation": affiliation, "roles": "pc",
        })

    def test_a_shared_orcid_under_one_name_is_the_strongest_evidence(self):
        verdict = find_duplicate_accounts.classify(
            self.account("a@x.edu", "Ada", "Lovelace", orcid="0000-1"),
            self.account("b@y.edu", "Ada", "Lovelace", orcid="0000-1"),
            pc_membership.DEFAULT_TOKEN_RATIO,
        )
        self.assertEqual(("high", "exact_name"), verdict[:2])
        self.assertIn("same_orcid", verdict[2])

    def test_conflicting_orcids_demote_an_exact_name_match(self):
        # Two people sharing a common name is more likely than one person
        # holding two ORCIDs, so the evidence has to cut against the match.
        confidence, reason, _ = find_duplicate_accounts.classify(
            self.account("a@x.edu", "Ada", "Lovelace", orcid="0000-1"),
            self.account("b@y.edu", "Ada", "Lovelace", orcid="0000-2"),
            pc_membership.DEFAULT_TOKEN_RATIO,
        )
        self.assertEqual(("low", "exact_name"), (confidence, reason))

    def test_unrelated_names_sharing_an_orcid_are_flagged_for_review(self):
        confidence, reason, _ = find_duplicate_accounts.classify(
            self.account("a@x.edu", "Ada", "Lovelace", orcid="0000-1"),
            self.account("b@y.edu", "Grace", "Hopper", orcid="0000-1"),
            pc_membership.DEFAULT_TOKEN_RATIO,
        )
        self.assertEqual(("review", "shared_orcid_different_names"), (confidence, reason))

    def test_two_people_with_no_shared_evidence_are_not_a_pair(self):
        self.assertIsNone(find_duplicate_accounts.classify(
            self.account("a@x.edu", "Ada", "Lovelace"),
            self.account("b@y.edu", "Grace", "Hopper"),
            pc_membership.DEFAULT_TOKEN_RATIO,
        ))


class ClearUploadTests(unittest.TestCase):
    """generate_clear_uploads.py: the undo half of the two HotCRP uploads."""

    def run_generator(self, accounts, form_rows=(), papers=(), extra_argv=()):
        tmp = Path(tempfile.mkdtemp())
        form = tmp / "chairs.csv"
        with form.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(AreaChairExclusionTests.AC_HEADERS)
            writer.writerows(form_rows)
        data = tmp / "papers.json"
        data.write_text(json.dumps(list(papers)), encoding="utf-8")
        out = {name: tmp / f"{name}.csv" for name in ("assignment", "paper", "account")}
        argv = ["generate_clear_uploads.py",
                "--pcinfo", str(write_pcinfo(tmp / "pcinfo.csv", accounts)),
                "--csv", str(form), "--overrides", str(tmp / "missing.csv"),
                "--data", str(data),
                "--assignment-out", str(out["assignment"]),
                "--paper-tag-out", str(out["paper"]),
                "--account-tag-out", str(out["account"]), *extra_argv]
        with mock.patch.object(sys, "argv", argv), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            generate_clear_uploads.main()

        def read(path):
            with path.open(newline="", encoding="utf-8") as f:
                return list(csv.reader(f))

        return {name: read(path) for name, path in out.items()}

    def ac_row(self, email, **kwargs):
        return AreaChairExclusionTests.ac_row(AreaChairExclusionTests, email, **kwargs)

    def test_the_review_clear_is_the_line_the_assignment_upload_opens_with(self):
        rows = self.run_generator([pc_account("a@x.edu")])["assignment"]
        self.assertEqual([["paper", "action", "email", "round"],
                          ["all", "clearreview", "all", "R1"]], rows)

    def test_round_all_widens_the_clear_to_every_round(self):
        rows = self.run_generator([pc_account("a@x.edu")],
                                  extra_argv=["--round", "all"])["assignment"]
        self.assertEqual(["all", "clearreview", "all", ""], rows[1])

    def test_the_canonical_track_range_is_cleared_from_every_paper(self):
        rows = self.run_generator([pc_account("a@x.edu")],
                                  extra_argv=["--track-clear-ceiling", "3"])["paper"]
        self.assertEqual([["paper", "action", "email", "tag", "round"],
                          ["all", "cleartag", "", "~~track_0", ""],
                          ["all", "cleartag", "", "~~track_1", ""],
                          ["all", "cleartag", "", "~~track_2", ""]], rows)

    def test_a_hand_made_track_tag_outside_the_range_is_cleared_too(self):
        # `track1` is live on a past export, left behind by setting the
        # track mechanism up by hand; clearing only the canonical `~~track_N`
        # range would strand it.
        rows = self.run_generator(
            [pc_account("a@x.edu")],
            papers=[{"pid": 1, "tags": ["track1#0"]}],
            extra_argv=["--track-clear-ceiling", "2"],
        )["paper"]
        self.assertIn(["all", "cleartag", "", "track1", ""], rows)

    def test_an_out_of_range_paper_track_tag_is_cleared_too(self):
        # `~~paper_track_N` was this repo's own paper-tag spelling before the
        # paper and account namespaces were unified onto `~~track_N`; a
        # paper tagged that way by an older run must still be recognized and
        # swept, the same way an out-of-range number or a hand-made spelling
        # is.
        rows = self.run_generator(
            [pc_account("a@x.edu")],
            papers=[{"pid": 1, "tags": ["~~paper_track_47#0"]}],
            extra_argv=["--track-clear-ceiling", "2"],
        )["paper"]
        self.assertIn(["all", "cleartag", "", "~~paper_track_47", ""], rows)

    def test_a_tag_that_merely_contains_the_word_track_is_left_alone(self):
        rows = self.run_generator(
            [pc_account("a@x.edu")],
            papers=[{"pid": 1, "tags": ["TRC-track#0", "~~needs-reserve#0"]}],
            extra_argv=["--track-clear-ceiling", "2"],
        )["paper"]
        self.assertEqual(3, len(rows))

    def test_an_account_holding_a_stale_track_tag_is_cleared_even_off_the_roster(self):
        # The gap assign_area_chairs.py's clear-then-reinstall cannot reach: a
        # chair who left the roster has no row in either tag CSV, so their tag
        # stays until something reads it off the export instead.
        rows = self.run_generator(
            [pc_account("gone@x.edu", tags="pc-full ~~track_4")],
            extra_argv=["--track-clear-ceiling", "2"],
        )["account"]
        self.assertEqual(["email", "remove_tags", "add_tags"], rows[0])
        self.assertEqual(["gone@x.edu", "~~track_0 ~~track_1 ~~track_4", ""], rows[1])

    def test_a_chair_the_export_has_not_caught_up_with_is_still_cleared(self):
        # An export taken before the last tag upload knows nothing about the
        # tags that upload wrote, so the chair roster has to be a second source.
        rows = self.run_generator(
            [pc_account("chair@x.edu", tags="pc-full")],
            form_rows=[self.ac_row("chair@x.edu")],
            extra_argv=["--track-clear-ceiling", "1"],
        )["account"]
        self.assertEqual([["email", "remove_tags", "add_tags"],
                          ["chair@x.edu", "~~track_0", ""]], rows)

    def test_no_row_is_written_for_an_address_hotcrp_does_not_have(self):
        # The bulk user importer *creates* an account for an unknown email, so
        # a chair whose form address is not their account address must appear
        # under the account address only -- never both, never the form's.
        rows = self.run_generator(
            [pc_account("real@x.edu", given="Ada", family="Lovelace", tags="pc-full")],
            form_rows=[self.ac_row("typo@elsewhere.edu", first="Ada", last="Lovelace")],
            extra_argv=["--track-clear-ceiling", "1"],
        )["account"]
        self.assertEqual([["email", "remove_tags", "add_tags"],
                          ["real@x.edu", "~~track_0", ""]], rows)

    def test_the_account_header_carries_the_two_commas_hotcrp_parses_on(self):
        # `p_profile.php`'s save_bulk reads a first line with fewer than two
        # commas as a plain list of email addresses (unless it also passes the
        # newer `(?:user|email)` test), re-quoting every row into one field and
        # failing it as an email address. Two commas never reach that branch.
        tmp = Path(tempfile.mkdtemp())
        out = tmp / "account.csv"
        generate_clear_uploads.write_clear_account_tag_csv(
            str(out), [["a@x.edu", "~~track_0"]]
        )
        first = out.read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(2, first.count(","))
        self.assertEqual("email,remove_tags,add_tags", first)

    def test_the_area_chair_tag_itself_is_never_touched(self):
        rows = self.run_generator(
            [pc_account("chair@x.edu", tags="pc-full ~~area-chairs ~~track_0")],
            extra_argv=["--track-clear-ceiling", "1"],
        )["account"]
        self.assertNotIn("~~area-chairs", rows[1][1])

    def test_the_tag_spelling_comes_from_the_installing_script(self):
        # One definition, or a clearing file silently misses the tag it is
        # meant to remove.
        self.assertIs(generate_clear_uploads.track_tag, assign_area_chairs.track_tag)


if __name__ == "__main__":
    unittest.main()
