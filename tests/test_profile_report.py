import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.profile_report import build_report, command_hash, metric_summary, original_indices, sha256


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class ProfileReportTests(unittest.TestCase):
    def test_scoped_original_indices_and_duplicate_ambiguity(self):
        a = {"file": "a.cpp", "arguments": ["clang", "-DA"]}
        b = {"file": "a.cpp", "arguments": ["clang", "-DB"]}
        self.assertEqual(original_indices([a, b], [b]), [(1, None)])
        self.assertEqual(original_indices([a, a], [a, a]), [(0, None), (1, None)])
        self.assertEqual(original_indices([a, a, b], [a])[0], (None, "ambiguous-original-index"))
        self.assertEqual(original_indices([a], [b])[0][1], "command-not-in-original-database")

    def test_nonfinite_metrics_do_not_poison_summary(self):
        self.assertEqual(metric_summary([1, 2, float("nan"), float("inf"), -1, None, True])["count"], 2)
        self.assertEqual(metric_summary([])["median"], None)

    def fixture(self, root):
        run, bases = root / "cuda-lifetime", root / "bases"
        native = run / "cuda-lifetime-native"
        native.parent.mkdir(); native.write_text("fixed executable fixture")
        source = root / "same.cpp"; source.write_text("int x;")
        outside = {"file": str(root / "outside.cpp"), "directory": str(root), "arguments": ["clang", "outside.cpp"]}
        entries = [{"file": str(source), "directory": str(root), "arguments": ["clang", define, str(source)]}
                   for define in ("-DA", "-DB")]
        inputs = run / "target/input"
        write(inputs / "original-compile_commands.json", [outside, *entries])
        write(inputs / "compile_commands.json", entries)
        native_hash = sha256(native)
        manifest = {"nativeSha256": native_hash, "jobs": 3, "completedAt": "completed-fixture",
                    "snapshotUnchanged": True, "pythonUnchanged": True,
                    "snapshotSha256": {"cuda-lifetime-native": native_hash},
                    "python": {"executable": str(native), "sha256": native_hash},
                    "targets": [{"target": "target", "inputChanges": [], "translationUnitChanges": [],
                    "inputSha256": {p.name: sha256(p) for p in inputs.iterdir()},
                    "translationUnitSha256": {str(source): sha256(source)}}]}
        write(run / "manifest.json", manifest)
        coverage = {"native": {"sha256": native_hash}, "tus": {"analyzed": [str(source)], "failed": [{"file": str(source), "reason": "analysis-timeout"}]},
                    "summary": {"parsed": 1, "failed": 1}, "coverageUnknown": [{"file": str(source), "reason": "unmodeled"}],
                    "profiles": [{"index": i, "file": str(source), "inputCommandSha256": command_hash(e), "seconds": 2 + i, "peakRssKiB": 100}
                                 for i, e in enumerate(entries)]}
        write(run / "target/coverage.json", coverage)
        key = "0000-" + hashlib.sha256(str(source).encode()).hexdigest()[:10]
        write(run / "target/out.logs" / (key + ".coverage.json"),
              {"tus": {"parsed": 1, "analyzed": [str(source)], "failed": []},
               "budgetExhausted": [{"function": "f"}], "coverageUnknown": [{"reason": "unknown-call"}], "unmodeledCudaCalls": {"cudaX": 1}})
        for mode in ("syntax", "analyze", "tidy"):
            base = bases / "target" / mode
            write(base / "original-compile_commands.json", entries)
            records = [{"file": str(source), "compilationCommandIndex": i, "inputCommandSha256": command_hash(e),
                        "status": "passed", "returncode": 0, "seconds": 1, "peakRssKiB": 50,
                        "sourceSha256": sha256(source), "sourceUnchanged": True, "inputDatabaseUnchanged": True}
                       for i, e in enumerate(entries)]
            write(base / "report.json", {"records": records, "compdbSha256": sha256(base / "original-compile_commands.json"),
                                          "compdbUnchanged": True, "executableUnchanged": True})
        return run, bases, native_hash

    def test_per_command_success_and_coverage_gaps_are_retained(self):
        with tempfile.TemporaryDirectory() as td:
            run, bases, native_hash = self.fixture(Path(td))
            report = build_report(run, bases, native_hash)
            target = report["targets"][0]
            self.assertTrue(report["expectedCandidateMatch"])
            self.assertEqual(target["comparisons"]["syntax"]["matchedSuccessfulCommands"], 1)
            self.assertEqual([r["originalIndex"] for r in target["rows"]], [1, 2])
            self.assertFalse(target["rows"][1]["cudaLifetime"]["parsed"])
            self.assertEqual(target["rows"][0]["cudaLifetime"]["coverage"]["budgetExhausted"], [{"function": "f"}])
            self.assertEqual(target["comparisons"]["syntax"]["ratioOfSummedWallSeconds"], 2)
            self.assertNotIn("sum", target["comparisons"]["syntax"]["peakRssKiB"]["cudaLifetime"])

    def test_failed_baseline_hash_mismatch_and_candidate_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            run, bases, _ = self.fixture(Path(td))
            report_path = bases / "target/syntax/report.json"
            report = json.loads(report_path.read_text())
            report["records"][0].update(status="timeout", returncode=-9)
            report["records"][1]["inputCommandSha256"] = "wrong"
            write(report_path, report)
            result = build_report(run, bases, "different-candidate")
            self.assertFalse(result["expectedCandidateMatch"])
            target = result["targets"][0]
            self.assertEqual(target["comparisons"]["syntax"]["matchedSuccessfulCommands"], 0)
            self.assertIn("baseline-timeout", target["rows"][0]["comparisons"]["syntax"]["reasons"])
            self.assertEqual(target["unmatchedBaselineRecords"]["syntax"][0]["reason"], "record-command-hash-mismatch")

    def test_input_tamper_excludes_all_matches(self):
        with tempfile.TemporaryDirectory() as td:
            run, bases, _ = self.fixture(Path(td))
            path = run / "target/input/compile_commands.json"
            path.write_text(path.read_text() + "\n")
            result = build_report(run, bases)
            self.assertEqual(result["targets"][0]["comparisons"]["syntax"]["matchedSuccessfulCommands"], 0)

    def test_driver_integrity_failure_excludes_all_matches(self):
        with tempfile.TemporaryDirectory() as td:
            run, bases, _ = self.fixture(Path(td))
            path = run / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["snapshotUnchanged"] = False
            write(path, manifest)
            result = build_report(run, bases)
            target = result["targets"][0]
            self.assertFalse(target["runIntegrityVerified"])
            self.assertEqual(target["comparisons"]["syntax"]["matchedSuccessfulCommands"], 0)
            self.assertIn("cuda-lifetime-run-integrity-unverified", target["rows"][0]["comparisons"]["syntax"]["reasons"])


if __name__ == "__main__":
    unittest.main()
