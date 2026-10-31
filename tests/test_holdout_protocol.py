"""Frozen-run protocol tests use temporary invented files, never the real holdout."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from eval import EvaluationError, create_freeze
from tests.run_holdout import digest, required_rule_files, run_frozen, verify_dependencies


class HoldoutProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "invented.cpp"
        self.source.write_text("void invented() {}\n")
        self.native = self.root / "synthetic-protocol-executable"
        self.native.write_text("not a real analyzer\n")
        self.rule = self.root / "invented-rule.txt"
        self.rule.write_text("predefined rule\n")
        self.library = self.root / "invented-external-library.so"
        self.library.write_text("temporary protocol dependency\n")
        self.dependencies = self.root / "dependencies.json"
        self.dependencies.write_text(json.dumps({"schemaVersion": 1, "files": [
            {"file": str(self.library), "sha256": digest(self.library)}]}))
        self.labels = self.root / "labels.json"
        self.labels.write_text(json.dumps({"schemaVersion": 1, "split": "holdout", "cases": [{
            "id": "invented-negative", "file": "invented.cpp", "ruleId": "CL1", "line": 1,
            "expect": "none", "semanticIds": ["fixture-only"]}]}))
        self.freeze = self.root / "freeze.json"
        self.output = self.root / "first-output"

    def freeze_inputs(self, native=True):
        return create_freeze(self.labels, self.freeze, self.root,
                             [self.rule, self.native, self.dependencies] if native else [self.rule, self.dependencies])

    def run_protocol(self, output=None):
        return run_frozen(self.labels, self.freeze, self.root, self.native, output or self.output,
                          required_files={str(self.rule)}, dependencies_path=self.dependencies)

    def synthetic_analyzer(self, compdb, sarif, coverage_file, **kwargs):
        coverage = {"tus": {"analyzed": [str(self.source)], "failed": [], "translationUnknown": []},
                    "budgetExhausted": [], "coverageUnknown": [], "unmodeledCudaCalls": {}, "complete": True}
        sarif.write_text(json.dumps({"version": "2.1.0", "runs": [{
            "tool": {"driver": {"name": "synthetic-protocol-fixture"}}, "results": []}]}))
        coverage_file.write_text(json.dumps(coverage))
        return coverage

    def test_missing_freeze_never_invokes_analyzer(self):
        with patch("tests.run_holdout.analyze") as analyze:
            with self.assertRaises(OSError):
                self.run_protocol()
            analyze.assert_not_called()

    def test_freeze_boundary_includes_driver_headers_and_environment_lock(self):
        (self.root / "tool").mkdir()
        header = self.root / "tool/SarifConsumer.h"
        header.write_text("// invented protocol header\n")
        required = required_rule_files(self.root, self.native)
        self.assertIn(str(header), required)
        self.assertIn(str(self.root / "environment.lock"), required)

    def test_freeze_must_bind_actual_executable(self):
        self.freeze_inputs(native=False)
        with patch("tests.run_holdout.analyze") as analyze:
            with self.assertRaisesRegex(EvaluationError, "omits required"):
                self.run_protocol()
            analyze.assert_not_called()

    def test_changed_labels_fail_before_attempt(self):
        self.freeze_inputs()
        self.labels.write_text(self.labels.read_text() + "\n")
        with self.assertRaisesRegex(EvaluationError, "frozen artifact changed"):
            self.run_protocol()
        self.assertFalse(self.freeze.with_name("freeze.json.first-run.json").exists())

    def test_first_attempt_is_write_once_across_output_directories(self):
        self.freeze_inputs()
        with patch("tests.run_holdout.analyze", side_effect=self.synthetic_analyzer):
            result = self.run_protocol()
            self.assertTrue(result["passed"])
            saved = (self.output / "first-result.json").read_bytes()
            with self.assertRaisesRegex(EvaluationError, "already has a first-run"):
                self.run_protocol(self.root / "second-output")
            self.assertEqual(saved, (self.output / "first-result.json").read_bytes())

    def test_failed_analysis_is_recorded_and_consumes_first_attempt(self):
        self.freeze_inputs()
        with patch("tests.run_holdout.analyze", side_effect=RuntimeError("invented timeout")):
            result = self.run_protocol()
        self.assertFalse(result["passed"])
        self.assertFalse(result["metricsRecorded"])
        self.assertIn("invented timeout", result["attemptFailures"][0])
        self.assertTrue((self.output / "first-result.json").is_file())
        with self.assertRaisesRegex(EvaluationError, "already has a first-run"):
            self.run_protocol(self.root / "retry")

    def test_rule_change_during_analysis_invalidates_result(self):
        self.freeze_inputs()

        def changed(*args, **kwargs):
            coverage = self.synthetic_analyzer(*args, **kwargs)
            self.rule.write_text("changed after execution started\n")
            return coverage

        with patch("tests.run_holdout.analyze", side_effect=changed):
            result = self.run_protocol()
        self.assertFalse(result["passed"])
        self.assertIn("post-run freeze verification failed", result["attemptFailures"][0])
        self.assertFalse(json.loads((self.output / "metrics.json").read_text())["freezeVerifiedBeforeAndAfter"])

    def test_dependency_manifest_must_be_part_of_freeze(self):
        create_freeze(self.labels, self.freeze, self.root, [self.rule, self.native])
        with self.assertRaisesRegex(EvaluationError, "omits required"):
            self.run_protocol()

    def test_changed_library_fails_before_attempt(self):
        self.freeze_inputs()
        self.library.write_text("modified external bytes\n")
        with self.assertRaisesRegex(EvaluationError, "dependency changed"):
            self.run_protocol()
        self.assertFalse(self.freeze.with_name("freeze.json.first-run.json").exists())

    def test_dependency_change_during_analysis_invalidates_result(self):
        self.freeze_inputs()

        def changed(*args, **kwargs):
            coverage = self.synthetic_analyzer(*args, **kwargs)
            self.library.write_text("modified during execution\n")
            return coverage

        with patch("tests.run_holdout.analyze", side_effect=changed):
            result = self.run_protocol()
        self.assertFalse(result["passed"])
        self.assertIn("post-run dependency verification failed", result["attemptFailures"][0])

    def test_dependency_manifest_rejects_invalid_or_ambiguous_entries(self):
        entry = {"file": str(self.library), "sha256": digest(self.library)}
        for entries in ([], [entry, entry], [{"file": "relative.so", "sha256": entry["sha256"]}],
                        [{"file": str(self.library), "sha256": "bad"}]):
            with self.subTest(entries=entries):
                self.dependencies.write_text(json.dumps({"schemaVersion": 1, "files": entries}))
                with self.assertRaises(EvaluationError):
                    verify_dependencies(self.dependencies)


if __name__ == "__main__":
    unittest.main()
