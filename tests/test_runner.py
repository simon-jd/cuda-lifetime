"""Runner protocol tests only: temporary fake executables are not CSA evidence.

Real checker correctness is tested separately against the hand-labelled corpus.
The fixtures below exercise isolation, reporting, metadata conversion and CLI
failure behavior without presenting fabricated output as actual analysis.
"""

import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from tool.cli import main
from tool.runner import analyze, enrich_sarif, validate_native_coverage


def native_fixture_sarif(source="case.cpp", level="error"):
    physical = {"artifactLocation": {"uri": source, "index": 0}, "region": {"startLine": 7}}
    return {"version": "2.1.0", "runs": [{
        "tool": {"driver": {"name": "protocol-fixture-not-csa", "rules": [{"id": "cuda.AsyncLifetime"}]}},
        "artifacts": [{"location": {"uri": source}}],
        "results": [{"ruleId": "cuda.AsyncLifetime", "ruleIndex": 0, "level": "warning",
            "message": {"text": f"CL1 [{level}; memory=pinned; semantic=S1]: pending copy"},
            "locations": [{"physicalLocation": physical}],
            "codeFlows": [{"threadFlows": [{"locations": [{
                "location": {"physicalLocation": physical, "message": {"text": "copy enqueued"}}}]}]}]}]}]}


class RunnerProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "case.cpp"
        self.source.write_text("int f() { return 1; }\n")
        self.compdb = self.root / "compile_commands.json"
        self.compdb.write_text(json.dumps([{"directory": str(self.root), "file": "case.cpp",
                                             "arguments": ["clang++", "-std=c++17", "-c", "case.cpp"]}]))
        self.out = self.root / "out.sarif"
        self.coverage = self.root / "coverage.json"

    def fake_native(self, mode="success", coverage=None, artifact=None):
        coverage = coverage if coverage is not None else {
            "tus": {"parsed": 1, "analyzed": [str(self.source)], "failed": []},
            "budgetExhausted": [], "coverageUnknown": [], "unmodeledCudaCalls": {}}
        artifact = artifact if artifact is not None else native_fixture_sarif(str(self.source))
        executable = self.root / "protocol-fixture-native"
        code = f"#!{sys.executable}\nimport json, pathlib, sys, time\n"
        if mode == "timeout":
            code += "time.sleep(2)\n"
        elif mode == "failure":
            code += "print('fixture parse failure', file=sys.stderr)\nraise SystemExit(9)\n"
        elif mode == "no-output":
            code += "raise SystemExit(0)\n"
        else:
            code += "args = sys.argv[1:]\n"
            code += "assert '--max-nodes' in args and '--model' in args\n"
            code += f"pathlib.Path(args[args.index('--sarif')+1]).write_text({json.dumps(json.dumps(artifact))})\n"
            if mode == "compdb-source":
                code += f"coverage = json.loads({json.dumps(json.dumps(coverage))})\n"
                code += "entry = json.loads((pathlib.Path(args[args.index('-p')+1]) / 'compile_commands.json').read_text())[0]\n"
                code += "coverage['tus']['analyzed'] = [entry['file']]\n"
                code += "pathlib.Path(args[args.index('--coverage')+1]).write_text(json.dumps(coverage))\n"
            else:
                code += f"pathlib.Path(args[args.index('--coverage')+1]).write_text({json.dumps(json.dumps(coverage))})\n"
        executable.write_text(code)
        executable.chmod(0o755)
        return str(executable)

    def run_analyze(self, native=None, **kwargs):
        return analyze(self.compdb, self.out, self.coverage, native or str(self.root / "absent-native"), **kwargs)

    def test_missing_native_reports_file_failure(self):
        cov = self.run_analyze()
        self.assertFalse(cov["complete"])
        self.assertEqual(cov["tus"]["parsed"], 0)
        self.assertEqual(cov["tus"]["failed"][0]["reason"], "native-analyzer-unavailable")
        self.assertEqual(cov["tus"]["failed"][0]["file"], str(self.source))
        self.assertEqual(json.loads(self.out.read_text())["runs"][0]["results"], [])

    def test_corrupted_compdb_still_writes_both_artifacts(self):
        for content in ("not JSON", "{}", '{"commands": []}'):
            with self.subTest(content=content):
                self.compdb.write_text(content)
                cov = self.run_analyze()
                self.assertEqual(cov["tus"]["failed"][0]["reason"], "invalid-compilation-database")
                self.assertTrue(self.out.exists() and self.coverage.exists())

    def test_invalid_compdb_entry_is_not_success(self):
        self.compdb.write_text('[{"file": "case.cpp"}]')
        cov = self.run_analyze()
        self.assertFalse(cov["complete"])
        self.assertEqual(cov["tus"]["failed"][0]["reason"], "invalid-compilation-command")

    def test_timeout_is_failure_and_budget_exhaustion(self):
        cov = self.run_analyze(self.fake_native("timeout"), timeout=0.05)
        self.assertFalse(cov["complete"])
        self.assertEqual(cov["tus"]["parsed"], 0)
        self.assertEqual(cov["tus"]["failed"][0]["reason"], "analysis-timeout")
        self.assertEqual(cov["budgetExhausted"][0]["file"], str(self.source))

    def test_native_nonzero_retains_stderr_log(self):
        cov = self.run_analyze(self.fake_native("failure"))
        failure = cov["tus"]["failed"][0]
        self.assertEqual(failure["returncode"], 9)
        self.assertIn("fixture parse failure", Path(failure["log"]).read_text())

    def test_native_success_without_output_is_failure(self):
        cov = self.run_analyze(self.fake_native("no-output"))
        self.assertFalse(cov["complete"])
        self.assertEqual(cov["tus"]["parsed"], 0)
        self.assertEqual(cov["tus"]["failed"][0]["reason"], "invalid-native-output")

    def test_empty_or_anonymous_native_sarif_is_protocol_failure(self):
        for artifact in ({"version": "2.1.0", "runs": []},
                         {"version": "2.1.0", "runs": [{"results": []}]},
                         {"version": "2.1.0", "runs": [{"tool": {"driver": {"name": "fixture"}}}]}):
            with self.subTest(artifact=artifact):
                cov = self.run_analyze(self.fake_native(artifact=artifact))
                self.assertFalse(cov["complete"])
                self.assertEqual(cov["tus"]["analyzed"], [])
                self.assertEqual(cov["tus"]["failed"][0]["reason"], "invalid-native-output")

    def test_native_coverage_for_another_tu_is_protocol_failure(self):
        coverage = {"tus": {"parsed": 1, "analyzed": [str(self.root / "another.cpp")], "failed": []},
                    "budgetExhausted": [], "unmodeledCudaCalls": {}}
        cov = self.run_analyze(self.fake_native(coverage=coverage))
        self.assertFalse(cov["complete"])
        self.assertEqual(cov["tus"]["analyzed"], [])
        self.assertIn("different translation unit", cov["tus"]["failed"][0]["detail"])

    def test_translation_unknown_remains_visible(self):
        self.compdb.write_text("[]")
        (self.root / "translation-report.json").write_text(json.dumps({"entries": [
            {"file": str(self.source), "status": "translation-unknown", "unknown": ["--mystery"]}]}))
        cov = self.run_analyze()
        self.assertFalse(cov["complete"])
        self.assertEqual(cov["summary"]["translationUnknown"], 1)
        self.assertEqual(cov["tus"]["total"], 1)

    def test_invalid_translation_report_is_coverage_failure(self):
        for report in ([], {"entries": [None]}, {"entries": "bad"}):
            with self.subTest(report=report):
                (self.root / "translation-report.json").write_text(json.dumps(report))
                cov = self.run_analyze(self.fake_native())
                self.assertFalse(cov["complete"])
                self.assertTrue(any(f["reason"] == "invalid-translation-report" for f in cov["tus"]["failed"]))

    def test_invalid_translation_directory_is_coverage_failure(self):
        for directory in (None, [], {}):
            with self.subTest(directory=directory):
                (self.root / "translation-report.json").write_text(json.dumps({"entries": [
                    {"file": "case.cpp", "directory": directory, "status": "translation-unknown"}]}))
                cov = self.run_analyze(self.fake_native())
                self.assertFalse(cov["complete"])
                self.assertTrue(any(f["reason"] == "invalid-translation-report" for f in cov["tus"]["failed"]))

    def test_native_missing_coverage_fields_is_not_complete(self):
        cov = self.run_analyze(self.fake_native(coverage={"tus": {"parsed": 1, "analyzed": [str(self.source)], "failed": []}}))
        self.assertFalse(cov["complete"])
        self.assertTrue(cov["coverageUnknown"])

    def test_native_parsed_zero_cannot_certify_analyzed_file(self):
        cov = self.run_analyze(self.fake_native(coverage={"tus": {"parsed": 0, "analyzed": [], "failed": []},
            "budgetExhausted": [], "unmodeledCudaCalls": {}}))
        self.assertFalse(cov["complete"])
        self.assertNotIn(str(self.source), cov["tus"]["analyzed"])

    def test_unknown_cuda_calls_prevent_complete_claim(self):
        cov = self.run_analyze(self.fake_native(coverage={"tus": {"parsed": 1, "analyzed": [str(self.source)], "failed": []},
            "budgetExhausted": [], "unmodeledCudaCalls": {"cudaMystery": 3}}))
        self.assertFalse(cov["complete"])
        self.assertEqual(cov["summary"]["unmodeledCudaCalls"], 3)

    def test_pending_visit_counters_sum_across_distinct_translation_units(self):
        counters = {"modeledCallVisits": 13, "pointerEscapes": 17, "pendingMemberCallUses": 2,
                    "pendingArgumentUses": 3, "pendingPointerEscapes": 5}
        other = self.root / "other.cpp"
        other.write_text("void other() {}\n")
        entries = json.loads(self.compdb.read_text())
        entries.append({"directory": str(self.root), "file": str(other),
                        "arguments": ["clang++", "-std=c++17", "-c", str(other)]})
        self.compdb.write_text(json.dumps(entries))
        coverage = {"tus": {"parsed": 1, "analyzed": [], "failed": []},
                    "budgetExhausted": [], "coverageUnknown": [], "unmodeledCudaCalls": {}, **counters}
        cov = self.run_analyze(self.fake_native("compdb-source", coverage=coverage))
        self.assertTrue(cov["complete"])
        self.assertEqual(cov["tus"]["parsed"], 2)
        self.assertEqual(cov["visitCounts"], {key: 2 * value for key, value in counters.items()})
        self.assertEqual(json.loads(self.coverage.read_text())["visitCounts"], cov["visitCounts"])
        logs = list((self.root / "out.logs").glob("*.coverage.json"))
        self.assertEqual(len(logs), 2)
        for path in logs:
            native = json.loads(path.read_text())
            self.assertEqual({key: native[key] for key in counters}, counters)

    def test_invalid_pending_counters_are_protocol_errors(self):
        for field in ("pendingMemberCallUses", "pendingArgumentUses", "pendingPointerEscapes"):
            for value in (-1, True, 1.5, "2", None):
                with self.subTest(field=field, value=value):
                    coverage = {"tus": {"parsed": 1, "analyzed": [str(self.source)], "failed": []}, field: value}
                    with self.assertRaisesRegex(ValueError, "invalid native visit counter"):
                        validate_native_coverage(coverage, self.source)
        cov = self.run_analyze(self.fake_native(coverage={"tus": {
            "parsed": 1, "analyzed": [str(self.source)], "failed": []}, "pendingPointerEscapes": True}))
        self.assertFalse(cov["complete"])
        self.assertEqual(cov["visitCounts"]["pendingPointerEscapes"], 0)
        self.assertEqual(cov["tus"]["failed"][0]["reason"], "invalid-native-output")

    def test_optional_counters_do_not_override_coverage_unknown(self):
        cov = self.run_analyze(self.fake_native())
        self.assertTrue(all(value == 0 for value in cov["visitCounts"].values()))
        cov = self.run_analyze(self.fake_native(coverage={"tus": {
            "parsed": 1, "analyzed": [str(self.source)], "failed": []}, "budgetExhausted": [],
            "unmodeledCudaCalls": {}, "pendingPointerEscapes": 7,
            "coverageUnknown": [{"file": str(self.source), "reason": "pending region escapes"}]}))
        self.assertEqual(cov["visitCounts"]["pendingPointerEscapes"], 7)
        self.assertEqual(cov["summary"]["coverageUnknown"], 1)
        self.assertFalse(cov["complete"])

    def test_protocol_success_preserves_path_and_attaches_coverage(self):
        artifact = native_fixture_sarif(str(self.source))
        cov = self.run_analyze(self.fake_native(artifact=artifact))
        self.assertTrue(cov["complete"])
        self.assertEqual(cov["tus"]["analyzed"], [str(self.source)])
        report = json.loads(self.out.read_text())
        result = report["runs"][0]["results"][0]
        self.assertEqual(result["codeFlows"], artifact["runs"][0]["results"][0]["codeFlows"])
        self.assertEqual(result["level"], "error")
        self.assertEqual(result["ruleId"], "CL1")
        self.assertEqual(result["properties"]["semanticId"], "S1")
        self.assertEqual(report["runs"][0]["properties"]["coverage"], cov["summary"])

    def test_native_binary_replaced_during_run_invalidates_coverage(self):
        native = self.fake_native()
        path = Path(native)
        with path.open("a") as stream:
            stream.write("with open(__file__, 'a') as changed: changed.write('\\n# changed during invocation\\n')\n")
        cov = self.run_analyze(native)
        self.assertFalse(cov["complete"])
        self.assertTrue(any(entry["reason"] == "native-binary-changed-during-run"
                            and entry["file"] == str(self.source) for entry in cov["coverageUnknown"]))

    def test_cli_analyze_failure_exit_status_and_eval_delegation(self):
        with contextlib.redirect_stdout(io.StringIO()):
            status = main(["analyze", "--compdb", str(self.compdb), "--sarif", str(self.out),
                           "--native", str(self.root / "absent-native")])
        self.assertEqual(status, 2)
        self.run_analyze(self.fake_native())
        labels = self.root / "labels.json"
        labels.write_text(json.dumps({"schemaVersion": 1, "split": "protocol-test", "cases": [
            {"id": "protocol-positive", "file": str(self.source), "ruleId": "CL1", "line": 7,
             "expect": "warning", "level": "error", "semanticIds": ["S1"]}]}))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = main(["eval", "--labels", str(labels), "--sarif", str(self.out), "--coverage", str(self.coverage)])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())["rules"]["CL1"]["tp"], 1)

    def test_cli_macro_verification_failure_is_not_success(self):
        for entry, expected in (({"status": "project-only"}, 2),
                                ({"status": "not-run"}, 2),
                                ({"status": "compared", "requiredDifferences": {"equal": False}}, 2),
                                ({"status": "compared", "requiredDifferences": {"equal": True}}, 0)):
            with self.subTest(entry=entry), patch("translate.macro_diff_compdb", return_value={"entries": [entry], "summary": {}}):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(["macros", "--compdb", "unused", "--out", "unused"]), expected)


class SarifEnrichmentTests(unittest.TestCase):
    def test_contract_filtering_is_explicit_and_counted(self):
        artifact = native_fixture_sarif(level="warning")
        before = copy.deepcopy(artifact)
        runs, suppressed = enrich_sarif(artifact, "legacy", False)
        self.assertEqual(runs[0]["results"], [])
        self.assertEqual(suppressed, 1)
        self.assertEqual(artifact, before)
        runs, suppressed = enrich_sarif(artifact, "per-thread", True)
        self.assertEqual(suppressed, 0)
        self.assertEqual(runs[0]["results"][0]["properties"]["defaultStream"], "per-thread")

    def test_rule_index_remains_consistent_and_artifact_indices_unchanged(self):
        runs, _ = enrich_sarif(native_fixture_sarif(), "legacy", True)
        result = runs[0]["results"][0]
        self.assertEqual(runs[0]["tool"]["driver"]["rules"][result["ruleIndex"]]["id"], result["ruleId"])
        location = result["locations"][0]["physicalLocation"]["artifactLocation"]
        self.assertEqual(runs[0]["artifacts"][location["index"]]["location"]["uri"], location["uri"])


if __name__ == "__main__":
    unittest.main()
