import json
from pathlib import Path
import sys
import tempfile
import unittest

from scripts.budget_sensitivity import BUDGETS, integrity_errors, reject_holdout, run_manifest, select_seed
from scripts.profile_report import sha256


class BudgetSensitivityTests(unittest.TestCase):
    def test_holdout_rejected_before_reading(self):
        with self.assertRaisesRegex(ValueError, "holdout"):
            reject_holdout("does-not-exist/holdout/labels.json")
        with self.assertRaisesRegex(ValueError, "holdout"):
            select_seed(Path("holdout/compile_commands.json"), Path("holdout/labels.json"))

    def test_seed_strata_follow_declared_label_order(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cases, entries = [], []
            for rule in ("CL1", "CL2", "CL3", "CL4"):
                for expect in ("warning", "none"):
                    for extra in ("first", "later"):
                        source = root / f"{rule}-{expect}-{extra}.cpp"
                        cases.append({"file": str(source), "id": source.stem, "ruleId": rule, "expect": expect})
                        entries.append({"file": str(source), "directory": td, "arguments": ["clang", str(source)]})
            database, labels = root / "compile_commands.json", root / "labels.json"
            database.write_text(json.dumps(entries)); labels.write_text(json.dumps({"split": "seed", "cases": cases}))
            selected = select_seed(database, labels)
            self.assertEqual(len(selected), 8)
            self.assertTrue(all("first" in e["file"] for e in selected))

    def test_failed_and_timeout_runs_retained(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); source = root / "input.cpp"; source.write_text("int x;")
            database = root / "compile_commands.json"; database.write_text("[]")
            native = str(Path(sys.executable).resolve())
            manifest = {"budgets": list(BUDGETS), "timeoutSeconds": .5,
                        "candidate": {"path": native, "sha256": sha256(native)},
                        "snapshotSha256": {}, "limitations": [],
                        "inputs": [{"id": "test", "file": str(source), "sourceSha256": sha256(source),
                                    "isolatedDatabase": str(database), "isolatedDatabaseSha256": sha256(database),
                                    "originalIndex": 7, "inputCommandSha256": "command-hash"}],
                        "runs": [{"id": "test", "maxNodes": budget, "command": [native, "-c", code],
                                  "workingDirectory": td, "outputDirectory": str(root / "runs" / str(budget))}
                                 for budget, code in zip(BUDGETS, ["raise SystemExit(3)", "import time; time.sleep(5)",
                                                                  "raise SystemExit(0)", "raise SystemExit(2)"])]}
            path = root / "manifest.json"; path.write_text(json.dumps(manifest))
            report = run_manifest(path)
            self.assertEqual([r["status"] for r in report["records"]], ["failed", "timeout", "invalid-or-failed-coverage", "failed"])
            self.assertEqual(len(report["records"]), 4)
            self.assertTrue(report["complete"])
            self.assertTrue(report["manifestUnchanged"])
            with self.assertRaisesRegex(ValueError, "overwrite"):
                run_manifest(path)
            source.write_text("changed")
            self.assertIn(str(source), integrity_errors(manifest, root))


if __name__ == "__main__":
    unittest.main()
