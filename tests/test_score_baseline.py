"""Rule-agnostic scoring of compiler baselines against labelled corpora."""
import unittest

from scripts.score_baseline import ROOT, score


class ScoreBaselineTests(unittest.TestCase):
    def test_any_warning_on_a_label_line_counts(self):
        labels = {"cases": [
            {"id": "bad", "file": "corpus/seed/a_bad.cpp", "line": 8, "expect": "warning"},
            {"id": "good", "file": "corpus/seed/a_good.cpp", "line": 8, "expect": "none"},
            {"id": "missed", "file": "corpus/seed/b_bad.cpp", "line": 4, "expect": "warning"},
            {"id": "clean", "file": "corpus/seed/b_good.cpp", "line": 4, "expect": "none"}]}
        report = {"mode": "analyze", "toolVersion": {"stdout": "clang version 23.1.3\n"}, "summary": {"passed": 4},
                  "records": [{"workingDirectory": str(ROOT), "diagnosticLines": [
                      f"{ROOT}/corpus/seed/a_bad.cpp:8:10: warning: Dereference of null pointer [core.NullDereference]",
                      "corpus/seed/a_good.cpp:8:3: warning: Value stored is never read [deadcode.DeadStores]",
                      "corpus/seed/b_bad.cpp:2:3: warning: unused variable",
                      "corpus/seed/b_good.cpp:4:3: error: not a warning"]}]}
        result = score(labels, report)
        self.assertEqual(result["totals"], {"tp": 1, "fp": 1, "fn": 1, "tn": 1})
        self.assertEqual(result["warnings"], 3)
        self.assertEqual(result["offLabelWarnings"], ["corpus/seed/b_bad.cpp:2: unused variable"])


if __name__ == "__main__":
    unittest.main()
