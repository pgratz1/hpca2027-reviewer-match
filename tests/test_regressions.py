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
import enrich_publications
import paper_matching
import fingerprint
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

    def test_load_papers_reports_skipped(self):
        papers = [
            self.COMPLETE,
            {"pid": 2, "title": "Placeholder", "abstract": "", "topics": [], "authors": []},
            {**self.COMPLETE, "pid": 3, "withdrawn": True},
        ]
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "papers.json")
            Path(path).write_text(json.dumps(papers))
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual([1], [p["pid"] for p in paper_matching.load_papers(path)])
                complete, skipped = paper_matching.load_papers(path, with_skipped=True)
        self.assertEqual([1], [p["pid"] for p in complete])
        self.assertEqual(
            [(2, ["title under 3 words", "no abstract", "no topics", "no authors"]), (3, ["withdrawn"])],
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


if __name__ == "__main__":
    unittest.main()
