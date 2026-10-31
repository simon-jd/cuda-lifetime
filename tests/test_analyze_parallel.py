"""Parallel orchestration protocol tests. Fake native output is not CSA evidence.

No real compiler, checker, project corpus, or holdout is executed by this suite.
"""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from scripts.analyze_parallel import (aggregate, analyze_parallel, empty_coverage,
                                     entry_sha, execute_task, finish_coverage, identity, invoke, main)
from tool.runner import analyze


FAKE_NATIVE = r'''import json, pathlib, sys, time
args = sys.argv[1:]
entry = json.loads((pathlib.Path(args[args.index('-p') + 1]) / 'compile_commands.json').read_text())[0]
flags = entry['arguments']
def value(key, default):
    return next((x.split('=', 1)[1] for x in flags if x.startswith('-D' + key + '=')), default)
case = value('CASE', '0')
mode = value('MODE', 'success')
time.sleep(float(value('DELAY', '0')))
if mode == 'timeout':
    time.sleep(2)
if mode == 'failure':
    print('deliberate fixture failure', file=sys.stderr)
    raise SystemExit(9)
if mode == 'crash':
    raise SystemExit(7)
if mode == 'no-output':
    raise SystemExit(0)
if mode == 'invalid-json':
    pathlib.Path(args[args.index('--coverage') + 1]).write_text('{')
    pathlib.Path(args[args.index('--sarif') + 1]).write_text('{}')
    raise SystemExit(0)
source = entry['file']
physical = {'artifactLocation': {'uri': source, 'index': 1}, 'region': {'startLine': int(case) + 1}}
runs = []
for part in range(int(value('RUNS', '1'))):
    runs.append({'tool': {'driver': {'name': 'parallel-protocol-fixture-not-csa', 'rules': [{'id': 'CL1'}]}},
        'artifacts': [{'location': {'uri': source + '.header', 'index': 0}}, {'location': {'uri': source, 'index': 1}}],
        'results': [{'ruleId': 'CL1', 'ruleIndex': 0, 'level': value('LEVEL', 'error'),
            'properties': {'memoryKind': 'pinned', 'semanticId': 'S1'},
            'message': {'text': 'fixture ' + case + ':' + str(part)},
            'locations': [{'physicalLocation': physical}],
            'codeFlows': [{'threadFlows': [{'locations': [{'location': {'physicalLocation': physical,
                'message': {'text': 'fixture enqueue ' + case + ':' + str(part)}}}]}]}]}]})
coverage = {'tus': {'parsed': 1, 'analyzed': [source], 'failed': []},
    'budgetExhausted': [], 'coverageUnknown': [], 'unmodeledCudaCalls': {},
    'modeledCallVisits': int(case) + 1, 'pointerEscapes': 2,
    'pendingMemberCallUses': 3, 'pendingArgumentUses': 4, 'pendingPointerEscapes': 5, 'functionsAnalyzed': 2,
    'peakRssKiB': 1234, 'assumptions': ['fixture success assumption']}
if mode == 'unknown':
    coverage['unmodeledCudaCalls'] = {'cudaFixtureUnknown': 3}
if mode == 'budget':
    coverage['budgetExhausted'] = [{'file': source, 'reason': 'fixture node limit'}]
if mode == 'wrong-source':
    coverage['tus']['analyzed'] = [source + '.wrong']
if mode == 'invalid-sarif':
    runs = []
pathlib.Path(args[args.index('--coverage') + 1]).write_text(json.dumps(coverage))
pathlib.Path(args[args.index('--sarif') + 1]).write_text(json.dumps({'version': '2.1.0', 'runs': runs}))
if mode == 'mutate-native':
    with open(__file__, 'a') as output: output.write('\n# fixture mutation\n')
'''


class ParallelProtocolTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.input = self.root / "input"
        self.input.mkdir()
        self.source = self.input / "same.cpp"
        self.source.write_text("void fixture() {}\n")
        self.compdb = self.input / "compile_commands.json"
        self.native = self.root / "fake-native-not-csa"
        self.native.write_text(f"#!{sys.executable}\n" + FAKE_NATIVE)
        self.native.chmod(0o755)
        self.models = self.root / "models"
        self.models.mkdir()
        (self.models / "fixture.json").write_text('{"protocolFixtureOnly": true}\n')
        self.serial = 0

    def entry(self, index=0, mode="success", delay=0, **flags):
        # Relative directory exercises rebasing from the ORIGINAL database.
        return {"file": "same.cpp", "directory": ".", "arguments": ["clang++", "-std=c++17",
            f"-DCASE={index}", f"-DMODE={mode}", f"-DDELAY={delay}",
            *[f"-D{key}={value}" for key, value in flags.items()], "-c", "same.cpp"]}

    def run_case(self, entries, jobs=2, timeout=3, report=None, native=None, models=None):
        self.compdb.write_text(json.dumps(entries))
        report_path = self.input / "translation-report.json"
        if report is not None:
            report_path.write_text(json.dumps(report))
        elif report_path.exists():
            report_path.unlink()
        out = self.root / f"run-{self.serial}"
        self.serial += 1
        cov = analyze_parallel(self.compdb, out, native or self.native, models or self.models,
                               jobs=jobs, timeout=timeout, include_contract=True)
        return cov, json.loads((out / "out.sarif").read_text()), json.loads((out / "manifest.json").read_text()), out

    def normalized_coverage(self, coverage):
        cov = copy.deepcopy(coverage)
        cov.pop("native", None)
        cov.pop("orchestration", None)
        for profile in cov["profiles"]:
            profile.pop("seconds", None)
        for failure in cov["tus"]["failed"]:
            failure.pop("log", None)
            failure.pop("task", None)
        return cov

    def test_duplicate_paths_remain_distinct_and_completion_order_does_not_reorder_runs(self):
        entries = [self.entry(0, delay=0.35, RUNS=2), self.entry(1), self.entry(1)]
        cov, document, manifest, out = self.run_case(entries, jobs=3)
        self.assertTrue(cov["complete"])
        self.assertEqual(cov["tus"], {"total": 3, "parsed": 3, "analyzed": [str(self.source)] * 3,
                                       "failed": [], "translationUnknown": []})
        messages = [run["results"][0]["message"]["text"] for run in document["runs"]]
        self.assertEqual(messages, ["fixture 0:0", "fixture 0:1", "fixture 1:0", "fixture 1:0"])
        self.assertEqual([p["index"] for p in cov["profiles"]], [0, 1, 2])
        self.assertEqual([p["inputCommandSha256"] for p in cov["profiles"]], [entry_sha(e) for e in entries])
        self.assertEqual(len({task["directory"] for task in manifest["tasks"]}), 3)
        self.assertLess(manifest["tasks"][1]["completedAt"], manifest["tasks"][0]["completedAt"])
        for run in document["runs"]:
            result = run["results"][0]
            self.assertEqual(result["locations"][0]["physicalLocation"]["artifactLocation"]["index"], 1)
            self.assertEqual(run["artifacts"][1]["location"]["uri"], str(self.source))
            self.assertEqual(result["codeFlows"][0]["threadFlows"][0]["locations"][0]["location"]["physicalLocation"]["artifactLocation"]["index"], 1)
        self.assertEqual(sum("coverage" in run.get("properties", {}) for run in document["runs"]), 1)
        self.assertEqual(manifest["before"], manifest["after"])
        self.assertEqual(manifest["snapshotBefore"], manifest["snapshotAfter"])
        self.assertTrue(all(Path(task["log"]).is_file() for task in manifest["tasks"]))

    def test_jobs_one_and_many_match_unchanged_serial_runner(self):
        entries = [self.entry(2, RUNS=2), self.entry(0), self.entry(2)]
        report = {"entries": [{"file": "same.cpp", "directory": ".", "status": "translated", "assumptions": ["first"]},
            {"file": "same.cpp", "directory": ".", "status": "translated", "assumptions": ["second"]},
            {"file": "same.cpp", "directory": ".", "status": "translation-unknown", "assumptions": ["unknown same-file assumption"]}]}
        one, one_doc, _, _ = self.run_case(entries, jobs=1, report=report)
        many, many_doc, _, _ = self.run_case(entries, jobs=3, report=report)
        self.assertEqual(self.normalized_coverage(one), self.normalized_coverage(many))
        self.assertEqual(one_doc, many_doc)
        serial_out = self.root / "serial.sarif"
        serial = analyze(self.compdb, serial_out, native=str(self.native), model=str(self.models), timeout=3, include_contract=True)
        self.assertEqual(self.normalized_coverage(one), self.normalized_coverage(serial))
        self.assertEqual(one_doc, json.loads(serial_out.read_text()))
        self.assertEqual(one["tus"]["total"], 4)
        self.assertEqual(len(one["tus"]["translationUnknown"]), 1)
        assumptions = one_doc["runs"][0]["results"][0]["properties"]["assumptions"]
        self.assertIn("unknown same-file assumption", assumptions)

    def test_failures_timeout_and_invalid_outputs_all_remain_in_denominator(self):
        modes = ["success", "failure", "timeout", "no-output", "invalid-json", "wrong-source", "invalid-sarif", "crash"]
        cov, document, manifest, _ = self.run_case([self.entry(i, mode) for i, mode in enumerate(modes)], jobs=4, timeout=0.3)
        self.assertFalse(cov["complete"])
        self.assertEqual(cov["tus"]["total"], len(modes))
        self.assertEqual(cov["tus"]["parsed"], 1)
        self.assertEqual(len(cov["tus"]["failed"]), len(modes) - 1)
        self.assertEqual(len(cov["budgetExhausted"]), 1)
        self.assertEqual(len(document["runs"]), 1)
        self.assertEqual(len(manifest["tasks"]), len(modes))
        self.assertTrue(all(task["state"] in ("completed", "completed-incomplete") for task in manifest["tasks"]))
        self.assertTrue(all(task["returncode"] in (0, 2) for task in manifest["tasks"]))

    def test_unknown_budget_and_warning_metadata_are_aggregated(self):
        cov, doc, _, _ = self.run_case([self.entry(0, "unknown"), self.entry(1, "budget"), self.entry(2, LEVEL="warning")])
        self.assertFalse(cov["complete"])
        self.assertEqual(cov["tus"]["parsed"], 3)
        self.assertEqual(cov["unmodeledCudaCalls"], {"cudaFixtureUnknown": 3})
        self.assertEqual(len(cov["budgetExhausted"]), 1)
        self.assertEqual(cov["visitCounts"]["modeledCallVisits"], 6)
        self.assertEqual(cov["visitCounts"]["pendingMemberCallUses"], 9)
        self.assertEqual(cov["visitCounts"]["functionsAnalyzed"], 6)
        self.assertEqual([profile["functionsAnalyzed"] for profile in cov["profiles"]], [2, 2, 2])
        self.assertEqual(doc["runs"][-1]["results"][0]["level"], "warning")

    def test_unknown_only_and_empty_scopes_are_explicit(self):
        unknown = {"entries": [{"file": "not-translated.cu", "status": "translation-unknown", "unknown": ["flag"]}]}
        cov, doc, manifest, _ = self.run_case([], report=unknown)
        self.assertEqual(cov["tus"]["total"], 1)
        self.assertFalse(cov["complete"])
        self.assertEqual(manifest["tasks"], [])
        self.assertEqual(doc["runs"][0]["results"], [])
        cov, _, manifest, _ = self.run_case([])
        self.assertTrue(cov["complete"])
        self.assertEqual(cov["tus"]["total"], 0)
        self.assertEqual(manifest["tasks"], [])

    def test_invalid_entry_rebased_to_global_index_and_missing_native_retained(self):
        cov, _, _, _ = self.run_case([self.entry(), {"file": "same.cpp"}, None])
        self.assertEqual(cov["tus"]["total"], 3)
        self.assertEqual(cov["tus"]["parsed"], 1)
        self.assertEqual([f["file"] for f in cov["tus"]["failed"]], ["entry:1", "entry:2"])
        cov, _, _, _ = self.run_case([self.entry(), self.entry()], native=self.root / "missing")
        self.assertFalse(cov["complete"])
        self.assertEqual(cov["tus"]["total"], 2)
        self.assertEqual(len(cov["tus"]["failed"]), 2)

    def test_exception_outside_worker_is_recorded_and_remaining_tasks_run(self):
        def throwing(task, *args):
            if task["index"] == 1:
                raise RuntimeError("fixture executor exception")
            return execute_task(task, *args)
        with patch("scripts.analyze_parallel.execute_task", side_effect=throwing):
            cov, _, manifest, _ = self.run_case([self.entry(i) for i in range(3)])
        self.assertEqual(cov["tus"]["total"], 3)
        self.assertEqual(cov["tus"]["parsed"], 2)
        self.assertEqual(manifest["tasks"][1]["state"], "exception")
        self.assertEqual(cov["tus"]["failed"][0]["reason"], "task-exception")

    def test_outer_worker_timeout_is_visible_and_budgeted(self):
        def timeout_worker(command, *args, **kwargs):
            if "/tasks/" in command[command.index("--compdb") + 1]:
                error = subprocess.TimeoutExpired(command, 63)
                error.returncode = -9
                raise error
            return invoke(command, *args, **kwargs)
        with patch("scripts.analyze_parallel.invoke", side_effect=timeout_worker):
            cov, _, manifest, _ = self.run_case([self.entry()])
        self.assertFalse(cov["complete"])
        self.assertEqual(cov["tus"]["total"], 1)
        self.assertEqual(cov["budgetExhausted"][0]["reason"], "orchestration-wall-clock-timeout")
        self.assertEqual(manifest["tasks"][0]["returncode"], -9)
        self.assertEqual(manifest["tasks"][0]["state"], "exception")

    def test_dynamic_scheduler_never_exceeds_jobs(self):
        lock = threading.Lock()
        active = 0
        maximum = 0
        def count(task, *args):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            try:
                return execute_task(task, *args)
            finally:
                with lock:
                    active -= 1
        with patch("scripts.analyze_parallel.execute_task", side_effect=count):
            cov, _, _, _ = self.run_case([self.entry(i, delay=0.05) for i in range(5)], jobs=2)
        self.assertTrue(cov["complete"])
        self.assertEqual(maximum, 2)
        self.assertEqual(active, 0)

    def test_keyboard_interrupt_retains_all_assigned_occurrences(self):
        def interrupted(task, *args):
            raise KeyboardInterrupt()
        with patch("scripts.analyze_parallel.execute_task", side_effect=interrupted):
            cov, _, manifest, _ = self.run_case([self.entry(), self.entry(), self.entry()], jobs=1)
        self.assertFalse(cov["complete"])
        self.assertEqual(cov["tus"]["total"], 3)
        self.assertEqual(manifest["phase"], "interrupted")
        self.assertEqual([task["state"] for task in manifest["tasks"]], ["not-completed"] * 3)
        self.assertEqual(len(cov["tus"]["failed"]), 3)

    def test_setup_failure_retains_unknown_and_every_uncompleted_task(self):
        report = {"entries": [{"file": "unknown.cu", "status": "translation-unknown"}]}
        cov, _, manifest, _ = self.run_case([self.entry(), self.entry()], report=report, models=self.root / "missing-models")
        self.assertFalse(cov["complete"])
        self.assertEqual(cov["tus"]["total"], 3)
        self.assertEqual(len(cov["tus"]["translationUnknown"]), 1)
        self.assertEqual([task["state"] for task in manifest["tasks"]], ["not-completed"] * 2)
        self.assertEqual(sum(f["reason"] == "task-not-completed" for f in cov["tus"]["failed"]), 2)

    def test_original_input_mutation_is_visible_even_with_valid_worker_outputs(self):
        def mutate(task, *args):
            result = execute_task(task, *args)
            (self.models / "fixture.json").write_text('{"changed": true}\n')
            return result
        with patch("scripts.analyze_parallel.execute_task", side_effect=mutate):
            cov, _, manifest, _ = self.run_case([self.entry()])
        self.assertFalse(cov["complete"])
        self.assertEqual(cov["tus"]["parsed"], 1)
        self.assertTrue(any(x.get("input") == "models" and x.get("category") == "original" for x in cov["coverageUnknown"]))
        self.assertNotEqual(manifest["before"]["models"], manifest["after"]["models"])

    def test_snapshot_native_mutation_is_visible(self):
        cov, _, manifest, _ = self.run_case([self.entry(mode="mutate-native")])
        self.assertFalse(cov["complete"])
        self.assertNotEqual(manifest["snapshotBefore"]["native"], manifest["snapshotAfter"]["native"])
        self.assertEqual(manifest["before"]["native"], manifest["after"]["native"])

    def test_fresh_output_refuses_overwrite_and_invalid_budget_does_not_create_it(self):
        _, _, _, out = self.run_case([self.entry()])
        marker = (out / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            analyze_parallel(self.compdb, out, self.native, self.models)
        self.assertEqual((out / "manifest.json").read_bytes(), marker)
        with self.assertRaises(ValueError):
            analyze_parallel(self.compdb, self.root / "bad-budget", self.native, self.models, timeout=float("nan"))
        self.assertFalse((self.root / "bad-budget").exists())

    def test_invalid_database_and_report_keep_failure_outputs(self):
        self.compdb.write_text('{bad json')
        out = self.root / "invalid-db"
        cov = analyze_parallel(self.compdb, out, self.native, self.models)
        self.assertFalse(cov["complete"])
        self.assertEqual(cov["tus"]["failed"][0]["reason"], "invalid-compilation-database")
        self.assertEqual((out / "inputs/original-compile_commands.json").read_text(), '{bad json')
        cov, _, _, _ = self.run_case([self.entry()], report={"entries": [None]})
        self.assertFalse(cov["complete"])
        self.assertEqual(cov["tus"]["total"], 1)
        self.assertEqual(cov["tus"]["parsed"], 1)
        self.assertTrue(any(f["reason"] == "invalid-translation-report" for f in cov["tus"]["failed"]))

    def test_malformed_report_keeps_preceding_assumptions_like_serial(self):
        report = {"entries": [
            {"file": "same.cpp", "directory": ".", "status": "translated", "assumptions": ["known preceding assumption"]},
            {"file": "broken.cpp", "directory": ".", "status": "translated", "assumptions": 7}]}
        cov, document, _, _ = self.run_case([self.entry()], report=report)
        serial_path = self.root / "malformed-report-serial.sarif"
        serial = analyze(self.compdb, serial_path, native=str(self.native), model=str(self.models), timeout=3, include_contract=True)
        self.assertFalse(cov["complete"])
        self.assertFalse(serial["complete"])
        self.assertNotIn("translationAssumptions", cov)
        self.assertEqual(document, json.loads(serial_path.read_text()))
        self.assertIn("known preceding assumption", document["runs"][0]["results"][0]["properties"]["assumptions"])

    def test_output_inside_copied_inputs_is_rejected_before_any_write(self):
        self.compdb.write_text(json.dumps([self.entry()]))
        driver = self.root / "driver-input"
        (driver / "tool").mkdir(parents=True)
        (driver / "tool/cli.py").write_text("# must never be invoked\n")
        for tree in (self.models, driver / "tool"):
            before = {str(path.relative_to(tree)): path.read_bytes() for path in tree.rglob("*") if path.is_file()}
            out = tree / "new-output"
            with self.assertRaisesRegex(ValueError, "overlaps copied input tree"), patch("scripts.analyze_parallel.shutil.copytree") as copier:
                analyze_parallel(self.compdb, out, self.native, self.models, driver_root=driver)
            copier.assert_not_called()
            self.assertFalse(out.exists())
            self.assertEqual(before, {str(path.relative_to(tree)): path.read_bytes() for path in tree.rglob("*") if path.is_file()})

    def test_output_overlap_through_driver_root_symlink_is_rejected(self):
        self.compdb.write_text(json.dumps([self.entry()]))
        actual = self.root / "actual-tool"
        actual.mkdir()
        (actual / "cli.py").write_text("# must never be invoked\n")
        driver = self.root / "driver-alias"
        driver.mkdir()
        (driver / "tool").symlink_to(actual, target_is_directory=True)
        out = actual / "new-output"
        with self.assertRaisesRegex(ValueError, "overlaps copied input tree"), patch("scripts.analyze_parallel.shutil.copytree") as copier:
            analyze_parallel(self.compdb, out, self.native, self.models, driver_root=driver)
        copier.assert_not_called()
        self.assertFalse(out.exists())

    def test_report_appearing_after_absent_capture_is_not_certified(self):
        self.compdb.write_text(json.dumps([self.entry()]))
        report = self.input / "translation-report.json"
        created = False
        def appearing_report(path, *args, **kwargs):
            nonlocal created
            if Path(path) == self.compdb and not created:
                created = True
                report.write_text(json.dumps({"entries": [{"file": "new.cu", "status": "translation-unknown"}]}))
            return identity(path, *args, **kwargs)
        out = self.root / "report-race"
        with patch("scripts.analyze_parallel.identity", side_effect=appearing_report):
            cov = analyze_parallel(self.compdb, out, self.native, self.models)
        self.assertFalse(cov["complete"])
        self.assertFalse((out / "inputs/original-translation-report.json").exists())
        self.assertTrue(any("translationReport changed" in failure.get("detail", "") for failure in cov["tus"]["failed"]))
        self.assertEqual(cov["tus"]["total"], 1)

    def test_sigterm_cancels_real_fixture_process_and_records_terminal_scope(self):
        pidfile = self.root / "signal-fixture-pid.json"
        marker = "import os\npathlib.Path(" + repr(str(pidfile)) + ").write_text(json.dumps({'pid': os.getpid(), 'pgid': os.getpgrp()}))\ntime.sleep(30)\n"
        self.native.write_text(f"#!{sys.executable}\n" + FAKE_NATIVE.replace("case = value('CASE', '0')", marker + "case = value('CASE', '0')"))
        self.compdb.write_text(json.dumps([self.entry(i) for i in range(3)]))
        out = self.root / "sigterm-run"
        script = Path(__file__).resolve().parents[1] / "scripts/analyze_parallel.py"
        command = [sys.executable, "-B", str(script), "--compdb", str(self.compdb), "--out", str(out),
                   "--native", str(self.native), "--model", str(self.models), "--jobs", "1", "--timeout", "40"]
        ids = None
        process = None
        try:
            with (self.root / "sigterm-parent.log").open("w") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                deadline = time.monotonic() + 8
                while not pidfile.exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.03)
                self.assertTrue(pidfile.exists(), "sleeping fixture child never started")
                ids = json.loads(pidfile.read_text())
                os.kill(process.pid, signal.SIGTERM)
                self.assertEqual(process.wait(timeout=4), 2)
            manifest = json.loads((out / "manifest.json").read_text())
            cov = json.loads((out / "coverage.json").read_text())
            self.assertEqual(manifest["phase"], "interrupted")
            self.assertEqual(manifest["terminationSignal"], "SIGTERM")
            self.assertIn("completedAt", manifest)
            self.assertTrue(all(task["state"] not in ("pending", "running") for task in manifest["tasks"]))
            self.assertEqual(cov["tus"]["total"], 3)
            self.assertFalse(cov["complete"])
            deadline = time.monotonic() + 2
            while True:
                try:
                    os.kill(ids["pid"], 0)
                except ProcessLookupError:
                    break
                self.assertLess(time.monotonic(), deadline, "fixture native remained alive after SIGTERM cleanup")
                time.sleep(0.03)
        finally:
            if ids:
                try:
                    os.killpg(ids["pgid"], signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=4)

    def test_sigterm_during_final_input_hashing_records_terminal_interruption(self):
        self.compdb.write_text(json.dumps([self.entry()]))
        out = self.root / "finalization-signal"
        hits = 0
        def terminated_identity(path, *args, **kwargs):
            nonlocal hits
            if Path(path) == self.compdb:
                hits += 1
                if hits == 2:
                    os.kill(os.getpid(), signal.SIGTERM)
            return identity(path, *args, **kwargs)
        with patch("scripts.analyze_parallel.identity", side_effect=terminated_identity), contextlib.redirect_stdout(io.StringIO()):
            status = main(["--compdb", str(self.compdb), "--out", str(out), "--native", str(self.native),
                           "--model", str(self.models), "--jobs", "1"])
        self.assertEqual(status, 2)
        manifest = json.loads((out / "manifest.json").read_text())
        cov = json.loads((out / "coverage.json").read_text())
        self.assertEqual(manifest["phase"], "interrupted")
        self.assertEqual(manifest["terminationSignal"], "SIGTERM")
        self.assertIn("completedAt", manifest)
        self.assertEqual(cov["tus"]["total"], 1)
        self.assertEqual(cov["tus"]["parsed"], 1)
        self.assertFalse(cov["complete"])

    def test_sigint_during_final_input_hashing_records_terminal_interruption(self):
        self.compdb.write_text(json.dumps([self.entry()]))
        out = self.root / "finalization-signal"
        hits = 0
        def terminated_identity(path, *args, **kwargs):
            nonlocal hits
            if Path(path) == self.compdb:
                hits += 1
                if hits == 2:
                    os.kill(os.getpid(), signal.SIGINT)
            return identity(path, *args, **kwargs)
        with patch("scripts.analyze_parallel.identity", side_effect=terminated_identity), contextlib.redirect_stdout(io.StringIO()):
            status = main(["--compdb", str(self.compdb), "--out", str(out), "--native", str(self.native),
                           "--model", str(self.models), "--jobs", "1"])
        self.assertEqual(status, 2)
        manifest = json.loads((out / "manifest.json").read_text())
        cov = json.loads((out / "coverage.json").read_text())
        self.assertEqual(manifest["phase"], "interrupted")
        self.assertEqual(manifest["terminationSignal"], "SIGINT")
        self.assertIn("completedAt", manifest)
        self.assertEqual(cov["tus"]["total"], 1)
        self.assertEqual(cov["tus"]["parsed"], 1)
        self.assertFalse(cov["complete"])

    def test_completed_output_tampering_is_not_accepted(self):
        def corrupt(task, *args):
            result = execute_task(task, *args)
            (Path(task["directory"]) / "out.sarif").write_text('{}')
            return result
        with patch("scripts.analyze_parallel.execute_task", side_effect=corrupt):
            cov, _, _, _ = self.run_case([self.entry()])
        self.assertFalse(cov["complete"])
        self.assertEqual(cov["tus"]["parsed"], 0)
        self.assertEqual(cov["tus"]["failed"][0]["reason"], "invalid-completed-task-output")

    def test_cli_returns_incomplete_status(self):
        self.compdb.write_text(json.dumps([self.entry(mode="unknown")]))
        with contextlib.redirect_stdout(io.StringIO()):
            status = main(["--compdb", str(self.compdb), "--out", str(self.root / "cli"),
                           "--native", str(self.native), "--model", str(self.models), "--jobs", "2"])
        self.assertEqual(status, 2)


if __name__ == "__main__":
    unittest.main()
