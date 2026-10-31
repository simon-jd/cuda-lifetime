"""Reproduction protocol checks with fake tools; no CSA or target evidence."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class TargetScriptTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def executable(self, path, body):
        path.write_text(f"#!{sys.executable}\n" + body)
        path.chmod(0o755)
        return path

    def test_tidy_runs_each_duplicate_source_configuration_once(self):
        source = self.root / "same.cpp"
        source.write_text("int value;\n")
        entries = [{"directory": str(self.root), "file": str(source),
                    "arguments": ["clang++", "-DVARIANT=" + variant, "-c", str(source)]}
                   for variant in ("A", "B")]
        compdb = self.root / "compile_commands.json"
        compdb.write_text(json.dumps(entries))
        fake = self.executable(self.root / "fake-tidy", '''import json,sys
from pathlib import Path
if '--version' in sys.argv:
    print('protocol fixture, not clang-tidy'); raise SystemExit(0)
database = Path(sys.argv[sys.argv.index('-p') + 1]) / 'compile_commands.json'
entries = json.loads(database.read_text())
for entry in entries:
    variant = next(arg for arg in entry['arguments'] if arg.startswith('-DVARIANT='))
    print(entry['file'] + ':1:1: warning: fixture ' + variant)
artifact = next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('--export-fixes='))
Path(artifact).write_text(json.dumps(entries))
''')
        out = self.root / "baseline"
        command = [sys.executable, str(ROOT / "scripts/baseline.py"), "--compdb", str(compdb),
                   "--out", str(out), "--mode", "tidy", "--tidy", str(fake), "--jobs", "2"]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads((out / "report.json").read_text())
        self.assertEqual(report["warningCount"], 2)
        self.assertEqual(report["summary"]["passed"], 2)
        for index, (entry, row) in enumerate(zip(entries, report["records"])):
            self.assertEqual(row["compilationCommandIndex"], index)
            self.assertEqual(row["inputCommandSha256"], hashlib.sha256(
                json.dumps(entry, sort_keys=True).encode()).hexdigest())
            self.assertEqual(len(row["diagnosticLines"]), 1)
            self.assertEqual(json.loads(Path(row["inputDatabase"]).read_text()), [entry])
            self.assertEqual(json.loads(Path(row["artifact"]).read_text()), [entry])
            self.assertTrue(row["sourceUnchanged"] and row["inputDatabaseUnchanged"])
        before = (out / "report.json").read_bytes()
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
        self.assertEqual((out / "report.json").read_bytes(), before)

    def test_target_driver_is_copied_before_first_target_changes_live_package(self):
        spec = importlib.util.spec_from_file_location("review_analyze_targets", ROOT / "scripts/analyze_targets.py")
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
        for package in ("tool", "translate", "eval"):
            (self.root / package).mkdir()
            (self.root / package / "__init__.py").write_text("")
        (self.root / "models").mkdir()
        (self.root / "models/model.json").write_text("{}\n")
        (self.root / "cuda-lifetime").write_text("# protocol fixture\n")
        live_cli = self.root / "tool/cli.py"
        live_cli.write_text('''import json,os,sys
from pathlib import Path
assert Path.cwd() == Path(__file__).resolve().parents[1]
assert os.environ['PYTHONPATH'] == str(Path.cwd())
assert os.environ['PYTHONNOUSERSITE'] == '1'
Path(''' + repr(str(live_cli)) + ''').write_text('raise SystemExit(77)\\n')
for option in ('--sarif', '--coverage'):
    Path(sys.argv[sys.argv.index(option) + 1]).write_text(json.dumps({'protocolFixture': True}))
print('copied Python driver protocol fixture')
''')
        original = live_cli.read_bytes()
        for name in ("cuda-samples", "rmm"):
            source = self.root / (name + ".cpp")
            source.write_text("int fixture;\n")
            inputs = self.root / "targets" / name / "clang"
            inputs.mkdir(parents=True)
            entry = {"directory": str(self.root), "file": str(source),
                     "arguments": ["clang++", "-c", str(source)]}
            (inputs / "compile_commands.json").write_text(json.dumps([entry]))
            (inputs / "translation-report.json").write_text(json.dumps({
                "entries": [{"file": str(source), "status": "translated"}], "summary": {}}))
        native = self.executable(self.root / "fixture-native", "print('protocol fixture, not CSA')\n")
        out = self.root / "run"
        args = ["analyze_targets.py", "--targets", "cuda-samples", "rmm", "--native", str(native),
                "--out", str(out), "--toolchain-kind", "bootstrap", "--jobs", "1"]
        with patch.object(script, "ROOT", self.root), patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(script.main(), 0)
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertNotEqual(live_cli.read_bytes(), original)
        self.assertEqual((out / "driver/tool/cli.py").read_bytes(), original)
        self.assertTrue(manifest["snapshotUnchanged"] and manifest["pythonUnchanged"])
        self.assertEqual([row["returncode"] for row in manifest["targets"]], [0, 0])
        for row in manifest["targets"]:
            self.assertEqual(row["workingDirectory"], str(out / "driver"))
            self.assertEqual(row["translationUnitChanges"], [])
            self.assertEqual(row["inputChanges"], [])
            self.assertEqual(len(row["inputSha256"]), 4)
        self.assertEqual(manifest["snapshotSha256"]["driver/tool/cli.py"], hashlib.sha256(original).hexdigest())


if __name__ == "__main__":
    unittest.main()
