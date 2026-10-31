"""Replay protocol tests use invented temporary corpora, never the real holdout."""

import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.replay_holdout import (ORIGINAL_ENVIRONMENT, dependency_files, digest,
                                    header_files, run_replay, verify_sources)


class ReplayProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "corpus/holdout/invented.cpp"
        self.source.parent.mkdir(parents=True)
        self.source.write_text("void invented() {}\n")
        self.labels = self.source.parent / "labels.json"
        self.labels.write_text(json.dumps({"schemaVersion": 1, "split": "holdout", "cases": [{
            "id": "invented-negative", "file": str(self.source.relative_to(self.root)), "ruleId": "CL1",
            "line": 1, "expect": "none", "semanticIds": ["synthetic-protocol-only"]}]}))
        self.rule = self.root / "models/base.json"
        self.rule.parent.mkdir()
        self.rule.write_text('{"models": []}\n')
        self.native = self.root / "build/rebuilt"
        self.native.parent.mkdir()
        self.native.write_text("synthetic executable bytes; never executed\n")
        self.headers = self.root / "sdk"
        self.headers.mkdir()
        (self.headers / "stddef.h").write_text("// invented SDK header\n")

        def entry(path):
            return {"file": str(path.relative_to(self.root)), "sha256": digest(path)}

        self.freeze = {"schemaVersion": 1, "split": "holdout", "frozenAt": "fixture timestamp",
                       "labels": entry(self.labels), "sources": [entry(self.source)],
                       "rules": [entry(self.rule)] + [{"file": path, "sha256": "0" * 64}
                                                     for path in sorted(ORIGINAL_ENVIRONMENT)]}
        self.freeze_path = self.root / "freeze.json"
        self.freeze_path.write_text(json.dumps(self.freeze))
        self.freeze_hash = digest(self.freeze_path)
        self.output = self.root / "results/replay"
        self.dependencies = {str(self.native): digest(self.native)}

    def synthetic_analysis(self, compdb, sarif, coverage_file, **kwargs):
        document = {"version": "2.1.0", "runs": [{"tool": {"driver": {
            "name": "synthetic-protocol-not-csa"}}, "results": []}]}
        coverage = {"tus": {"analyzed": [str(self.source)], "failed": [], "translationUnknown": []},
                    "budgetExhausted": [], "coverageUnknown": [], "unmodeledCudaCalls": {}, "complete": True}
        sarif.write_text(json.dumps(document))
        coverage_file.write_text(json.dumps(coverage))
        return coverage

    def replay(self, analysis=None, dependencies=None, expected=None, output=None):
        with patch("scripts.replay_holdout.run_analysis", side_effect=analysis or self.synthetic_analysis) as call:
            with patch("scripts.replay_holdout.dependency_files",
                       side_effect=dependencies or (lambda n, roots: (dict(self.dependencies), "invented ldd"))):
                result = run_replay(self.native, self.freeze_path, output or self.output, [self.headers],
                                    expected or self.freeze_hash, root=self.root)
        return result, call

    def test_success_is_explicitly_replay_and_preserves_first_run(self):
        first = self.root / "results/holdout-first"
        first.mkdir(parents=True)
        sentinel = first / "first-result.json"
        sentinel.write_text("original record\n")
        result, _ = self.replay()
        self.assertTrue(result["replaySucceeded"])
        self.assertTrue(result["notFirstAttempt"])
        self.assertTrue(result["notBitIdenticalReproduction"])
        self.assertEqual(sentinel.read_text(), "original record\n")
        self.assertEqual(digest(self.freeze_path), self.freeze_hash)
        self.assertEqual({entry["file"] for entry in result["originalEnvironmentRecords"]}, ORIGINAL_ENVIRONMENT)

    def test_pinned_freeze_hash_rejects_deleted_rules_and_records_failure(self):
        changed = copy.deepcopy(self.freeze)
        changed["rules"] = changed["rules"][1:]
        self.freeze_path.write_text(json.dumps(changed))
        result, call = self.replay()
        call.assert_not_called()
        self.assertFalse(result["replaySucceeded"])
        self.assertIn("pinned", result["failures"][0]["message"])
        self.assertTrue((self.output / "replay-result.json").is_file())

    def test_portable_source_changes_are_not_environment_exceptions(self):
        self.rule.write_text("modified rule\n")
        result, call = self.replay()
        call.assert_not_called()
        self.assertIn("frozen source changed", result["failures"][0]["message"])

    def test_extra_unfrozen_model_is_rejected(self):
        (self.rule.parent / "injected.json").write_text("{}\n")
        result, call = self.replay()
        call.assert_not_called()
        self.assertIn("inventory changed", result["failures"][0]["message"])

    def test_exact_environment_record_set_is_required(self):
        for changed in (self.freeze["rules"][:-1], self.freeze["rules"] + [
                {"file": "some-other-environment.json", "sha256": "0" * 64}]):
            freeze = copy.deepcopy(self.freeze)
            freeze["rules"] = changed
            with self.subTest(changed=changed), self.assertRaises((ValueError, OSError)):
                verify_sources(freeze, self.root)

    def test_duplicate_and_traversing_entries_are_rejected(self):
        freeze = copy.deepcopy(self.freeze)
        freeze["rules"].append(freeze["sources"][0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            verify_sources(freeze, self.root)
        freeze = copy.deepcopy(self.freeze)
        freeze["sources"][0]["file"] = "../outside.cpp"
        with self.assertRaisesRegex(ValueError, "nonlocal"):
            verify_sources(freeze, self.root)

    def test_analysis_exception_still_records_result_and_post_checks(self):
        result, _ = self.replay(analysis=RuntimeError("invented analyzer failure"))
        self.assertTrue(result["inputsUnchanged"])
        self.assertFalse(result["metricsRecorded"])
        self.assertEqual(result["failures"][0]["stage"], "analysis")
        self.assertTrue((self.output / "replay-result.json").is_file())

    def test_incomplete_coverage_is_not_a_successful_replay(self):
        def incomplete(*args, **kwargs):
            coverage = self.synthetic_analysis(*args, **kwargs)
            coverage["complete"] = False
            return coverage
        result, _ = self.replay(analysis=incomplete)
        self.assertTrue(result["metricsRecorded"])
        self.assertFalse(result["analysisComplete"])
        self.assertFalse(result["replaySucceeded"])

    def test_false_positive_can_be_a_successful_replay_without_passing_labels(self):
        def unexpected(compdb, sarif, coverage_file, **kwargs):
            coverage = self.synthetic_analysis(compdb, sarif, coverage_file, **kwargs)
            document = json.loads(sarif.read_text())
            document["runs"][0]["results"] = [{"ruleId": "CL1", "level": "error",
                "locations": [{"physicalLocation": {"artifactLocation": {"uri": self.source.as_uri()},
                                                     "region": {"startLine": 1}}}]}]
            sarif.write_text(json.dumps(document))
            return coverage
        result, _ = self.replay(analysis=unexpected)
        self.assertTrue(result["replaySucceeded"])
        self.assertFalse(result["passed"])

    def test_source_mutation_during_analysis_is_recorded(self):
        def changed(*args, **kwargs):
            coverage = self.synthetic_analysis(*args, **kwargs)
            self.rule.write_text("changed while analyzing\n")
            return coverage
        result, _ = self.replay(analysis=changed)
        self.assertFalse(result["inputsUnchanged"])
        self.assertFalse(result["passed"])
        self.assertTrue(any(f["stage"] == "post-source-verification" for f in result["failures"]))

    def test_metrics_collision_is_preserved_and_final_failure_is_recorded(self):
        def collision(*args, **kwargs):
            coverage = self.synthetic_analysis(*args, **kwargs)
            (self.output / "metrics.json").write_text("preexisting conflicting artifact\n")
            return coverage
        result, _ = self.replay(analysis=collision)
        self.assertFalse(result["metricsRecorded"])
        self.assertFalse(result["replaySucceeded"])
        self.assertEqual(result["failures"][0]["stage"], "metrics-persistence")
        self.assertEqual((self.output / "metrics.json").read_text(), "preexisting conflicting artifact\n")
        self.assertTrue((self.output / "replay-result.json").is_file())

    def test_dependency_changes_after_analysis_are_recorded(self):
        counter = iter([(self.dependencies, "before"), ({str(self.native): "1" * 64}, "after")])
        result, _ = self.replay(dependencies=lambda n, r: next(counter))
        self.assertFalse(result["inputsUnchanged"])
        self.assertTrue(any(f["stage"] == "post-dependency-verification" for f in result["failures"]))

    def test_post_dependency_discovery_failure_is_recorded(self):
        counter = iter([0, 1])
        def discover(native, roots):
            if next(counter):
                raise OSError("invented missing SDK")
            return self.dependencies, "ldd"
        result, _ = self.replay(dependencies=discover)
        self.assertFalse(result["inputsUnchanged"])
        self.assertIn("missing SDK", result["failures"][0]["message"])

    def test_existing_output_and_first_run_subdirectory_are_refused(self):
        self.output.mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            self.replay()
        with self.assertRaisesRegex(ValueError, "original first-run"):
            self.replay(output=self.root / "results/holdout-first/new-child")
        self.assertFalse((self.root / "results/holdout-first").exists())

    def test_ldd_missing_library_is_never_ignored(self):
        response = subprocess.CompletedProcess(["ldd"], 0, "libLLVM.so => not found\n", "")
        with patch("scripts.replay_holdout.subprocess.run", return_value=response):
            with self.assertRaisesRegex(ValueError, "unresolved shared library"):
                dependency_files(self.native, [self.headers])

    def test_header_symlink_directories_are_hashed_and_empty_roots_rejected(self):
        target = self.root / "other-headers"
        target.mkdir()
        header = target / "extra.h"
        header.write_text("// additional header\n")
        (self.headers / "linked").symlink_to(target, target_is_directory=True)
        (target / "loop").symlink_to(self.headers, target_is_directory=True)
        self.assertIn(header, header_files(self.headers))
        empty = self.root / "empty-headers"
        empty.mkdir()
        with self.assertRaisesRegex(ValueError, "empty dependency root"):
            header_files(empty)


if __name__ == "__main__":
    unittest.main()
