import contextlib
import csv
import io
import json
import random
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np
import requests

import assign_reviewers
import assign_area_chairs
import area_chairs
import build_fingerprints
import classify_reviewers
import compare_abstract_rankings
import dblp
import enrich_publications
import estimate_reserve_need
import paper_matching
import fingerprint
import resolve_trc_members
import score_abstract_evaluation
from reviewers import Reviewer, _parse_override_cap


def reviewer(pid="1/Test"):
    return Reviewer(
        email="person@example.com", first="Test", last="Person", dblp_url="",
        pid=pid, affiliation="Example", primary="Memory", secondary="", tertiary="",
        keywords="caches", tier="full", override_cap=None,
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
            chairs = area_chairs.load_area_chairs(str(path), str(Path(tmp) / "missing.csv"))
        self.assertEqual(1, len(chairs))
        self.assertEqual("chair@example.com", chairs[0].email)
        self.assertEqual("New Name", chairs[0].name)
        self.assertEqual("1/Test", chairs[0].pid)
        self.assertEqual("", chairs[0].tertiary)

    def test_load_reviewer_assigned_pids(self):
        text = (
            "=== [2] Included\n"
            "    assigned 6 of 6 requested\n"
            "=== [3] Empty\n"
            "    assigned 0 of 6 requested\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "assignment.txt"
            path.write_text(text, encoding="utf-8")
            self.assertEqual([2], assign_area_chairs.load_reviewer_assigned_pids(str(path)))

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
                mock.patch.object(build_fingerprints, "load_reviewers", return_value=[r]),
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
                stack.enter_context(mock.patch.object(build_fingerprints, "load_reviewers", return_value=[r]))
                stack.enter_context(mock.patch.object(build_fingerprints, "load_cache", return_value={}))
                stack.enter_context(mock.patch.object(build_fingerprints, "load_colleague_cache", return_value={}))
                stack.enter_context(mock.patch.object(build_fingerprints, "fetch_titles_for_pids", side_effect=succeed_fetch))
                load_model = stack.enter_context(mock.patch.object(build_fingerprints.specter2_model, "load_model"))
                stack.enter_context(mock.patch.object(sys, "argv", ["build_fingerprints.py", "--fingerprint-cache", cache, "--device", "cpu"]))
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(build_fingerprints.main(), 0)
                load_model.assert_not_called()


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
            b'<dblpperson name="Hwayong Nam" pid="331/8145">'
            b'<person key="homepages/331/8145">'
            b'<author pid="331/8145">Hwayong Nam</author>'
            b'<note type="affiliation">Seoul National University</note></person>'
            b'<r><www><title>Home Page</title><year>2024</year></www></r>'
            b'<r><inproceedings><author pid="331/8145">Hwayong Nam</author>'
            b'<author pid="10/1/JungHoAhn">Jung Ho Ahn</author>'
            b'<title>A paper</title><year>2026</year></inproceedings></r>'
            b'</dblpperson>'
        )
        profile = resolve_trc_members.parse_profile(xml)
        self.assertEqual(["Hwayong Nam"], profile["names"])
        self.assertEqual(["Seoul National University"], profile["affiliations"])
        self.assertEqual({"331/8145": "Hwayong Nam", "10/1/JungHoAhn": "Jung Ho Ahn"},
                         profile["coauthors"])
        # The www record is a homepage, not a publication.
        self.assertEqual(1, profile["pubs"])

    def test_search_parse_handles_dblps_collapsed_single_element_lists(self):
        payload = {"result": {"hits": {"hit": {
            "info": {
                "author": "Mingu Kang 0001",
                "url": "https://dblp.org/pid/12/3456-1",
                "aliases": {"alias": "M. Kang"},
                "notes": {"note": [
                    {"@type": "affiliation", "text": "University of California San Diego"},
                    {"@type": "award", "text": "not an affiliation"},
                ]},
            }}}}}
        self.assertEqual(
            [{"name": "Mingu Kang 0001", "pid": "12/3456-1", "aliases": ["M. Kang"],
              "affiliations": ["University of California San Diego"]}],
            resolve_trc_members.parse_search(payload),
        )

    def test_search_rejects_the_endpoints_near_misses(self):
        # DBLP's author search is a similarity search: asked for "Cheng Chen"
        # it volunteers people who are not called that at all.
        dblp_stub = FakeDblp(searches={"Cheng Chen": [
            {"name": "Fu-Chen Cheng 0001", "pid": "241/4667-1", "aliases": [], "affiliations": []},
            {"name": "Cheng-Zhong Xu 0001", "pid": "181/2765-1", "aliases": [], "affiliations": []},
            {"name": "Cheng Chen 0012", "pid": "9/9999", "aliases": [], "affiliations": []},
        ]})
        hits = resolve_trc_members.search_candidates(dblp_stub, "Cheng Chen")
        self.assertEqual(["9/9999"], [h["pid"] for h in hits])

    def test_search_falls_back_to_splitting_a_camel_cased_name(self):
        dblp_stub = FakeDblp(searches={
            "SeyyedHossein SeyyedAghaeiRezaei": [],
            "Seyyed Hossein Seyyed Aghaei Rezaei": [
                {"name": "SeyyedHossein SeyyedAghaeiRezaei", "pid": "342/4454",
                 "aliases": [], "affiliations": []},
            ],
        })
        hits = resolve_trc_members.search_candidates(
            dblp_stub, "SeyyedHossein SeyyedAghaeiRezaei"
        )
        self.assertEqual(["342/4454"], [h["pid"] for h in hits])

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
        pc_index = {frozenset({"mingu", "kang"}): {"m7kang@ucsd.edu": "UC San Diego"}}
        author_index = {frozenset({"mingu", "kang"}): {"mingu@ucsd.edu": Counter({"UCSD": 16})}}
        self.assertEqual(
            ("m7kang@ucsd.edu", "pc-form"),
            resolve_trc_members.resolve_advisor_email(
                "Mingu Kang", "University of California San Diego", pc_index, author_index
            ),
        )

    def test_two_researchers_sharing_a_name_are_split_by_affiliation(self):
        author_index = {frozenset({"rakesh", "kumar"}): {
            "rakesh.kumar@ntnu.no": Counter({"Norwegian University of Science and Technology": 8}),
            "rakeshk@illinois.edu": Counter({"University of Illinois": 6}),
        }}
        self.assertEqual(
            ("rakesh.kumar@ntnu.no", "hotcrp-author+affiliation"),
            resolve_trc_members.resolve_advisor_email(
                "Rakesh Kumar", "Norwegian University of Science and Technology", {}, author_index
            ),
        )
        # With no affiliation to go on, guessing between them is not allowed.
        email, resolution = resolve_trc_members.resolve_advisor_email(
            "Rakesh Kumar", "", {}, author_index
        )
        self.assertIsNone(email)
        self.assertEqual("ambiguous-hotcrp-author(2)", resolution)

    def test_one_person_with_two_addresses_resolves_to_the_one_they_use(self):
        author_index = {frozenset({"jiayi", "huang"}): {
            "hjy@hkust-gz.edu.cn": Counter({"HKUST(GZ)": 21}),
            "jiayihuang2022@163.com": Counter({"HKUST (GZ)": 4}),
        }}
        self.assertEqual(
            ("hjy@hkust-gz.edu.cn", "hotcrp-author+most-used"),
            resolve_trc_members.resolve_advisor_email(
                "Jiayi Huang", "HKUST(GZ)", {}, author_index
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
        # DBLP's bare "Cheng Chen" page collects 725 papers by many people; the
        # advisor really has published with it, so co-authorship alone would
        # accept it.
        dblp_stub = FakeDblp(
            profiles={
                "9/Advisor": dblp_profile(["Hai Jin"], {"10/217": "Cheng Chen"}),
                "10/217": dblp_profile(["Cheng Chen"], {"9/Advisor": "Hai Jin"}, pubs=725),
            },
            searches={"Cheng Chen": []},
        )
        pid, resolution, notes = resolve_trc_members.resolve_student_pid(
            dblp_stub, "Cheng Chen", "HUST", "c@x.org", ["9/Advisor"], {}, 100, 8,
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
                    ["Lieven Eeckhout"], {"342/4454": "SeyyedHossein SeyyedAghaeiRezaei"}
                ),
                "342/4454": dblp_profile(
                    ["SeyyedHossein SeyyedAghaeiRezaei"], {"9/Advisor": "Lieven Eeckhout"}, pubs=6
                ),
            },
            searches={"Hossein SeyyedAghaei": [], "Hossein Seyyed Aghaei": []},
        )
        pid, resolution, notes = resolve_trc_members.resolve_student_pid(
            dblp_stub, "Hossein SeyyedAghaei", "Ghent University", "h@x.org", ["9/Advisor"],
            {"h@x.org": Counter({"342/4454": 2})}, 100, 8,
        )
        self.assertEqual("342/4454", pid)
        # Proposed by the student's own submission, confirmed by co-authorship.
        self.assertEqual("confirmed-self-declared", resolution)
        self.assertIn("spells the name differently", " ".join(notes))

        # Self-declared alone, with a name nobody can check, is not enough.
        unbacked = FakeDblp(
            profiles={"342/4454": dblp_profile(["SeyyedHossein SeyyedAghaeiRezaei"], pubs=6)},
            searches={"Hossein SeyyedAghaei": [], "Hossein Seyyed Aghaei": []},
        )
        pid, resolution, notes = resolve_trc_members.resolve_student_pid(
            unbacked, "Hossein SeyyedAghaei", "Ghent University", "h@x.org", [],
            {"h@x.org": Counter({"342/4454": 2})}, 100, 8,
        )
        self.assertIsNone(pid)
        self.assertEqual("unverified", resolution)
        # The page it looked at is named, so a human can finish in one click.
        self.assertIn("342/4454", " ".join(notes))

    def test_one_name_transliterated_two_ways_still_needs_confirming(self):
        compatible = resolve_trc_members.tokens_compatible
        tokens = dblp.name_tokens
        self.assertEqual("partial", compatible(tokens("Maryam Elgamal"), tokens("Mariam Elgamal")))
        # One shared token is required, so unrelated short names stay apart.
        self.assertIsNone(compatible(tokens("Jing Li"), tokens("Jung Lee")))
        self.assertIsNone(compatible(tokens("Yi Zhang"), tokens("Yu Huang")))
        # A dropped or added leading consonant is a different given name, not
        # a respelling — "Heng Chen" is not "Cheng Chen", however close the
        # strings score, and both really do co-author with the same advisor.
        self.assertIsNone(compatible(tokens("Cheng Chen"), tokens("Heng Chen")))
        self.assertIsNone(compatible(tokens("Jing Zhao"), tokens("Jin Zhao")))
        self.assertIsNone(compatible(tokens("Yong Kim"), tokens("Yang Kim")))

        dblp_stub = FakeDblp(
            profiles={
                "9/Advisor": dblp_profile(["David Brooks"], {"9/Student": "Mariam Elgamal"}),
                "9/Student": dblp_profile(["Mariam Elgamal"], {"9/Advisor": "David Brooks"}, pubs=5),
            },
            searches={"Maryam Elgamal": []},
        )
        pid, resolution, _ = resolve_trc_members.resolve_student_pid(
            dblp_stub, "Maryam Elgamal", "Harvard", "m@x.org", ["9/Advisor"], {}, 100, 8,
        )
        self.assertEqual("9/Student", pid)
        self.assertEqual("confirmed-coauthor", resolution)

    def test_namesakes_of_one_advisor_are_split_by_dblps_affiliation(self):
        # DBLP has three "Zihan Xia" pages and Mingu Kang has published with
        # more than one of them; only one of the three is at UCSD.
        dblp_stub = FakeDblp(profiles={
            "9/Advisor": dblp_profile(["Mingu Kang"], {
                "244/0846": "Zihan Xia", "244/0846-2": "Zihan Xia 0002",
                "244/0846-5": "Zihan Xia 0005",
            }),
            "244/0846": dblp_profile(["Zihan Xia"], pubs=7),
            "244/0846-2": dblp_profile(
                ["Zihan Xia 0002"], pubs=16,
                affiliations=["University of Electronic Science and Technology of China"],
            ),
            "244/0846-5": dblp_profile(
                ["Zihan Xia 0005"], pubs=1,
                affiliations=["University of California, San Diego, CA, USA"],
            ),
        }, searches={"Zihan Xia": []})
        pid, resolution, notes = resolve_trc_members.resolve_student_pid(
            dblp_stub, "Zihan Xia", "University of California San Diego", "z@x.org",
            ["9/Advisor"], {}, 100, 8,
        )
        self.assertEqual("244/0846-5", pid)
        self.assertEqual("confirmed-coauthor", resolution)
        self.assertIn("recorded affiliation", " ".join(notes))

        # With no affiliation to go on, all three stay in contention.
        pid, resolution, _ = resolve_trc_members.resolve_student_pid(
            dblp_stub, "Zihan Xia", "", "z@x.org", ["9/Advisor"], {}, 100, 8,
        )
        self.assertIsNone(pid)
        self.assertEqual("ambiguous", resolution)

    def test_a_declined_pc_member_is_found_by_their_address_alone(self):
        # Declining the invitation leaves every column blank but the address,
        # which is still the HotCRP account a conflict needs.
        unnamed = [("hanjun@yonsei.ac.kr", "")]
        self.assertEqual(
            ("hanjun@yonsei.ac.kr", "pc-form-address"),
            resolve_trc_members.resolve_advisor_email(
                "Hanjun Kim", "Yonsei University", {}, {}, unnamed
            ),
        )
        # The domain has to be their institution too.
        self.assertEqual(
            (None, "not-found"),
            resolve_trc_members.resolve_advisor_email(
                "Hanjun Kim", "Seoul National University", {}, {}, unnamed
            ),
        )
        # A bare surname names half a department, so it names nobody.
        self.assertEqual(
            (None, "not-found"),
            resolve_trc_members.resolve_advisor_email(
                "Hanjun Kim", "Yonsei University", {}, {}, [("kim@yonsei.ac.kr", "")]
            ),
        )

    def test_too_many_matching_pages_are_left_for_a_human(self):
        coauthors = {f"9/{i}": "Wei Wang" for i in range(12)}
        dblp_stub = FakeDblp(
            profiles={"9/Advisor": dblp_profile(["An Advisor"], coauthors)},
            searches={"Wei Wang": []},
        )
        pid, resolution, notes = resolve_trc_members.resolve_student_pid(
            dblp_stub, "Wei Wang", "Example University", "w@x.org", ["9/Advisor"], {}, 100, 8,
        )
        self.assertIsNone(pid)
        self.assertEqual("ambiguous", resolution)
        # Bailing out before fetching twelve pages that cannot be told apart.
        self.assertEqual(["9/Advisor"], dblp_stub.fetched)
        self.assertIn("too many to tell apart", " ".join(notes))

    def test_affiliation_tokens_drop_the_words_every_institution_shares(self):
        tokens = resolve_trc_members.affiliation_tokens
        self.assertEqual(frozenset({"california", "san", "diego"}),
                         tokens("University of California San Diego"))
        # A title prefixed to the cell must not become the institution.
        self.assertEqual(tokens("National University of Singapore"),
                         tokens("Professor, National University of Singapore"))
        self.assertFalse(tokens("University") & tokens("Institute of Technology"))

    def test_co_advised_students_name_both_advisors(self):
        self.assertEqual(["Xiaofei Liao", "Hai Jin"],
                         resolve_trc_members.split_advisor_names("Xiaofei Liao (or Hai Jin)"))
        self.assertEqual(["Solo Advisor"],
                         resolve_trc_members.split_advisor_names("Solo Advisor"))


if __name__ == "__main__":
    unittest.main()
