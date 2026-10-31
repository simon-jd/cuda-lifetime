"""Independent evaluation oracle tests; never import checker implementation."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from eval import EvaluationError, create_freeze, evaluate, verify_frozen
from eval.metrics import normalized_file


def label(case_id, filename, expect="warning", line=7, rule="CL1", level="error"):
    return {"id": case_id, "file": filename, "ruleId": rule, "line": line,
            "expect": expect, "level": level if expect == "warning" else None,
            "semanticIds": ["S1"]}


def diagnostic(filename, line=7, rule="CL1", level="error"):
    return {"ruleId": rule, "level": level, "message": {"text": "independent test diagnostic"},
            "locations": [{"physicalLocation": {
                "artifactLocation": {"uri": filename}, "region": {"startLine": line}}}]}


def sarif(*results):
    return {"version": "2.1.0", "runs": [{"tool": {"driver": {"name": "fixture"}}, "results": list(results)}]}


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.labels = {"schemaVersion": 1, "split": "seed", "cases": [
            label("positive", "bad.cpp"), label("negative", "good.cpp", "none")]}
        self.coverage = {"tus": {"analyzed": ["bad.cpp", "good.cpp"], "failed": [], "translationUnknown": []}, "budgetExhausted": [], "unmodeledCudaCalls": {}}

    def evaluate(self, artifact, coverage=None, labels=None):
        return evaluate(labels or self.labels, artifact, self.coverage if coverage is None else coverage, root="/tmp/eval-test")

    def test_exact_match_and_covered_negative(self):
        result = self.evaluate(sarif(diagnostic("bad.cpp")))
        self.assertTrue(result["passed"])
        self.assertEqual(result["totals"], {"tp": 1, "fp": 0, "fn": 0, "tn": 1,
            "uncoveredNegative": 0, "precision": 1.0, "recall": 1.0})

    def test_duplicate_results_are_false_positives(self):
        result = self.evaluate(sarif(diagnostic("bad.cpp"), diagnostic("bad.cpp")))
        self.assertEqual(result["totals"]["tp"], 1)
        self.assertEqual(result["totals"]["fp"], 1)
        self.assertEqual(result["totals"]["precision"], 0.5)
        self.assertEqual(result["duplicateDiagnostics"], 1)
        self.assertFalse(result["passed"])

    def test_wrong_line_produces_both_fp_and_fn(self):
        result = self.evaluate(sarif(diagnostic("bad.cpp", line=8)))
        self.assertEqual(result["totals"]["tp"], 0)
        self.assertEqual(result["totals"]["fp"], 1)
        self.assertEqual(result["totals"]["fn"], 1)

    def test_wrong_rule_produces_per_rule_fp_and_fn(self):
        result = self.evaluate(sarif(diagnostic("bad.cpp", rule="CL2")))
        self.assertEqual(result["rules"]["CL1"]["fn"], 1)
        self.assertEqual(result["rules"]["CL2"]["fp"], 1)

    def test_negative_case_catches_any_line_in_file(self):
        result = self.evaluate(sarif(diagnostic("bad.cpp"), diagnostic("good.cpp", line=99)))
        self.assertEqual(result["totals"]["fp"], 1)
        self.assertEqual(result["totals"]["tn"], 0)

    def test_unanalyzed_negative_not_counted_as_tn(self):
        result = self.evaluate(sarif(), coverage={"tus": {"analyzed": []}})
        self.assertEqual(result["totals"]["tn"], 0)
        self.assertEqual(result["totals"]["fn"], 1)
        self.assertEqual(result["totals"]["uncoveredNegative"], 1)
        self.assertEqual(result["coverage"]["uncoveredCases"], ["positive", "negative"])

    def test_parsed_count_does_not_certify_coverage(self):
        result = self.evaluate(sarif(diagnostic("bad.cpp")), coverage={"tus": {"parsed": 2}})
        self.assertEqual(result["totals"]["tp"], 1)
        self.assertEqual(result["totals"]["tn"], 0)
        self.assertFalse(result["passed"])

    def test_missing_telemetry_never_certifies_a_true_negative(self):
        for field in ("budgetExhausted", "unmodeledCudaCalls", "tus.failed", "tus.translationUnknown"):
            with self.subTest(field=field):
                coverage = copy.deepcopy(self.coverage)
                target = coverage["tus"] if field.startswith("tus.") else coverage
                target.pop(field.rsplit(".", 1)[-1])
                coverage["complete"] = True
                result = self.evaluate(sarif(diagnostic("bad.cpp")), coverage)
                self.assertEqual(result["totals"]["tn"], 0)
                self.assertEqual(result["totals"]["uncoveredNegative"], 1)
                self.assertFalse(result["passed"])
                self.assertIn(field, result["coverage"]["missingTelemetry"])

    def test_empty_or_anonymous_sarif_cannot_mean_no_warnings(self):
        for artifact in ({"version": "2.1.0", "runs": []},
                         {"version": "2.1.0", "runs": [{"results": []}]},
                         {"version": "2.1.0", "runs": [{"tool": {"driver": {"name": "fixture"}}}]},
                         {"version": "2.1.0", "runs": [{"tool": {"driver": {"name": ""}}, "results": []}]}):
            with self.subTest(artifact=artifact), self.assertRaises(EvaluationError):
                self.evaluate(artifact)

    def test_malformed_coverage_telemetry_cannot_certify_tn(self):
        for field in ("budgetExhausted", "coverageUnknown", "tus.failed", "tus.translationUnknown", "tus.analyzed"):
            for value in (None, 0, 1, {}):
                with self.subTest(field=field, value=value):
                    coverage = copy.deepcopy(self.coverage)
                    target = coverage["tus"] if field.startswith("tus.") else coverage
                    target[field.rsplit(".", 1)[-1]] = value
                    with self.assertRaises(EvaluationError):
                        self.evaluate(sarif(diagnostic("bad.cpp")), coverage)
        coverage = copy.deepcopy(self.coverage)
        coverage["complete"] = "true"
        with self.assertRaises(EvaluationError):
            self.evaluate(sarif(diagnostic("bad.cpp")), coverage)

    def test_unmodeled_calls_remain_visible_and_prevent_pass_claim(self):
        coverage = copy.deepcopy(self.coverage)
        coverage["unmodeledCudaCalls"] = {"cudaUnmodeled": 2}
        result = self.evaluate(sarif(diagnostic("bad.cpp")), coverage)
        self.assertFalse(result["passed"])
        self.assertEqual(result["coverage"]["unmodeledCudaCalls"], {"cudaUnmodeled": 2})
        self.assertEqual(result["totals"]["tn"], 0)
        self.assertEqual(result["totals"]["uncoveredNegative"], 1)

    def test_global_incompleteness_does_not_credit_silent_negative(self):
        coverage = copy.deepcopy(self.coverage)
        coverage["complete"] = False
        result = self.evaluate(sarif(diagnostic("bad.cpp")), coverage)
        self.assertEqual(result["totals"]["tp"], 1)
        self.assertEqual(result["totals"]["tn"], 0)
        self.assertEqual(result["totals"]["uncoveredNegative"], 1)
        self.assertFalse(result["passed"])

    def test_failure_and_budget_override_analyzed_list(self):
        for location, entry in [("failed", {"file": "good.cpp", "reason": "missing header"}),
                                ("translationUnknown", "good.cpp"), ("skipped", "good.cpp")]:
            with self.subTest(location=location):
                coverage = copy.deepcopy(self.coverage)
                coverage["tus"][location] = [entry]
                result = self.evaluate(sarif(diagnostic("bad.cpp")), coverage=coverage)
                self.assertEqual(result["totals"]["tn"], 0)
        coverage = copy.deepcopy(self.coverage)
        coverage["budgetExhausted"] = [{"file": "good.cpp", "function": "f"}]
        self.assertEqual(self.evaluate(sarif(diagnostic("bad.cpp")), coverage)["totals"]["tn"], 0)
        coverage = copy.deepcopy(self.coverage)
        coverage["coverageUnknown"] = [{"file": "good.cpp", "reason": "missing-budget-telemetry"}]
        self.assertEqual(self.evaluate(sarif(diagnostic("bad.cpp")), coverage)["totals"]["tn"], 0)

    def test_file_uri_percent_encoding_and_relative_segments(self):
        labels = {"schemaVersion": 1, "cases": [label("uri", "sub/my file.cpp")]}
        coverage = {"tus": {"analyzed": ["sub/../sub/my file.cpp"], "failed": [], "translationUnknown": []}, "budgetExhausted": [], "unmodeledCudaCalls": {}}
        result = self.evaluate(sarif(diagnostic("file://localhost/tmp/eval-test/sub/my%20file.cpp")), coverage, labels)
        self.assertEqual(result["totals"]["tp"], 1)
        self.assertTrue(result["passed"])

    def test_artifact_and_rule_indices_with_uri_base(self):
        artifact = sarif(diagnostic("unused"))
        run = artifact["runs"][0]
        run["originalUriBaseIds"] = {"ROOT": {"uri": "file:///tmp/eval-test/"}}
        run["artifacts"] = [{"location": {"uri": "bad.cpp", "uriBaseId": "ROOT"}}]
        run["tool"]["driver"]["rules"] = [{"id": "CL1"}]
        result = run["results"][0]
        del result["ruleId"]
        result["ruleIndex"] = 0
        result["locations"][0]["physicalLocation"]["artifactLocation"] = {"index": 0}
        self.assertTrue(self.evaluate(artifact)["passed"])

    def test_cyclic_uri_bases_fail_loudly(self):
        artifact = sarif(diagnostic("bad.cpp"))
        artifact["runs"][0]["originalUriBaseIds"] = {"A": {"uri": "", "uriBaseId": "A"}}
        artifact["runs"][0]["results"][0]["locations"][0]["physicalLocation"]["artifactLocation"]["uriBaseId"] = "A"
        with self.assertRaises(EvaluationError):
            self.evaluate(artifact)

    def test_severity_mismatch_is_separate_from_matching(self):
        result = self.evaluate(sarif(diagnostic("bad.cpp", level="warning")))
        self.assertEqual(result["totals"]["tp"], 1)
        self.assertEqual(result["severityMismatches"], [{"id": "positive", "expected": "error", "actual": "warning"}])
        self.assertFalse(result["passed"])

    def test_notes_and_foreign_checks_do_not_count_as_defects(self):
        result = self.evaluate(sarif(diagnostic("bad.cpp"), diagnostic("good.cpp", level="note"), diagnostic("good.cpp", rule="unix.Malloc")))
        self.assertTrue(result["passed"])
        self.assertEqual(result["ignoredDiagnostics"], 2)

    def test_outside_corpus_diagnostics_are_not_scored(self):
        result = self.evaluate(sarif(diagnostic("bad.cpp"), diagnostic("elsewhere.cpp")))
        self.assertTrue(result["passed"])
        self.assertEqual(result["outsideCorpusDiagnostics"], 1)

    def test_zero_denominator_is_null(self):
        result = self.evaluate(sarif())
        self.assertIsNone(result["totals"]["precision"])
        self.assertEqual(result["totals"]["recall"], 0)
        self.assertIsNone(result["rules"]["CL4"]["recall"])

    def test_malformed_known_rule_diagnostic_is_not_silently_dropped(self):
        for bad in [{"ruleId": "CL1"}, diagnostic("https://example.org/x.cpp"), diagnostic("bad.cpp", line=0)]:
            with self.subTest(bad=bad), self.assertRaises(EvaluationError):
                self.evaluate(sarif(bad))

    def test_conflicting_and_duplicate_labels_rejected(self):
        labels = copy.deepcopy(self.labels)
        labels["cases"].append(label("duplicate-location", "bad.cpp"))
        with self.assertRaises(EvaluationError):
            self.evaluate(sarif(), labels=labels)
        labels["cases"][-1] = label("duplicate-negative", "good.cpp", "none")
        with self.assertRaisesRegex(EvaluationError, "duplicate negative"):
            self.evaluate(sarif(), labels=labels)
        labels["cases"][-1] = label("conflict", "bad.cpp", "none")
        with self.assertRaises(EvaluationError):
            self.evaluate(sarif(), labels=labels)

    def test_windows_file_uri_normalization(self):
        self.assertEqual(normalized_file("file:///C:/src/my%20file.cpp"), normalized_file("C:\\src\\my file.cpp"))


class FreezeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "case.cpp"
        self.source.write_text("int f() { return 1; }\n")
        self.rules = self.root / "rules.txt"
        self.rules.write_text("frozen semantic rule\n")
        self.labels = self.root / "labels.json"
        self.labels.write_text(json.dumps({"schemaVersion": 1, "split": "holdout", "cases": [label("case", "case.cpp", line=1)]}))
        self.manifest = self.root / "freeze.json"

    def freeze(self):
        return create_freeze(self.labels, self.manifest, self.root, ["rules.txt"])

    def test_freeze_verifies_and_refuses_overwrite(self):
        self.freeze()
        verify_frozen(self.labels, self.manifest, self.root)
        with self.assertRaisesRegex(EvaluationError, "overwrite"):
            self.freeze()

    def test_source_change_invalidates_freeze(self):
        self.freeze()
        self.source.write_text("int f() { return 2; }\n")
        with self.assertRaisesRegex(EvaluationError, "artifact changed"):
            verify_frozen(self.labels, self.manifest, self.root)

    def test_label_change_invalidates_freeze(self):
        self.freeze()
        self.labels.write_text(self.labels.read_text() + "\n")
        with self.assertRaisesRegex(EvaluationError, "artifact changed"):
            verify_frozen(self.labels, self.manifest, self.root)

    def test_rule_change_invalidates_freeze(self):
        self.freeze()
        self.rules.write_text("changed rule\n")
        with self.assertRaisesRegex(EvaluationError, "artifact changed"):
            verify_frozen(self.labels, self.manifest, self.root)

    def test_holdout_freeze_requires_rule_hashes(self):
        with self.assertRaisesRegex(EvaluationError, "rule files"):
            create_freeze(self.labels, self.manifest, self.root)

    def test_nonobject_freeze_is_reported_as_invalid(self):
        self.manifest.write_text("[]")
        with self.assertRaisesRegex(EvaluationError, "invalid freeze"):
            verify_frozen(self.labels, self.manifest, self.root)

    def test_cli_rejects_holdout_without_freeze(self):
        sarif_path, coverage_path = self.root / "out.sarif", self.root / "coverage.json"
        sarif_path.write_text(json.dumps(sarif(diagnostic("case.cpp", line=1))))
        coverage_path.write_text(json.dumps({"tus": {"analyzed": ["case.cpp"], "failed": [], "translationUnknown": []},
                                             "budgetExhausted": [], "unmodeledCudaCalls": {}}))
        command = [sys.executable, "-m", "eval", "--labels", str(self.labels), "--sarif", str(sarif_path),
                   "--coverage", str(coverage_path), "--root", str(self.root)]
        rejected = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("requires --freeze", rejected.stderr)
        self.freeze()
        accepted = subprocess.run([*command, "--freeze", str(self.manifest)], capture_output=True, text=True)
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        self.assertTrue(json.loads(accepted.stdout)["freezeVerified"])


class SeedIntegrityTests(unittest.TestCase):
    def test_twenty_manual_cases_have_exact_annotation_locations(self):
        root = Path(__file__).resolve().parents[1]
        labels = json.loads((root / "corpus/seed/labels.json").read_text())
        self.assertEqual(len(labels["cases"]), 20)
        self.assertEqual(sum(c["expect"] == "warning" for c in labels["cases"]), 10)
        self.assertEqual({c["ruleId"] for c in labels["cases"]}, {"CL1", "CL2", "CL3", "CL4"})
        for case in labels["cases"]:
            with self.subTest(case=case["id"]):
                text = (root / case["file"]).read_text().splitlines()
                line = text[case["line"] - 1]
                expected = "expected-warning{{" + case["ruleId"] + "}}" if case["expect"] == "warning" else "label-site"
                self.assertIn(expected, line)
                partner = case["id"].removesuffix("_bad") + "_good" if case["expect"] == "warning" else case["id"].removesuffix("_good") + "_bad"
                self.assertIn(partner, {c["id"] for c in labels["cases"]})


if __name__ == "__main__":
    unittest.main()
