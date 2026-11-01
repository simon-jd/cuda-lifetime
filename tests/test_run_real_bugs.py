"""Matching and path handling of scripts/run_real_bugs.py; no analyzer is run."""
import json
from pathlib import Path
import unittest

from scripts.run_real_bugs import ROOT, expand, on_lines, relative


class RealBugMatchingTests(unittest.TestCase):
    def diagnostic(self, uri, line, flow=()):
        return {"uri": uri, "line": line, "flow": [{"uri": u, "line": n} for u, n in flow]}

    def test_primary_or_flow_location_on_a_site_line_matches(self):
        path, lines = "src/a.cu", [10, 12]
        self.assertTrue(on_lines(self.diagnostic("file:///t/src/a.cu", 12), path, lines))
        self.assertTrue(on_lines(self.diagnostic("file:///t/src/a.cu", 30, [("file:///t/src/a.cu", 10)]), path, lines))
        self.assertFalse(on_lines(self.diagnostic("file:///t/src/a.cu", 11), path, lines))
        # The same line number in another file does not count.
        self.assertFalse(on_lines(self.diagnostic("file:///t/src/b.cu", 10, [("file:///t/include/x.h", 12)]), path, lines))

    def test_tree_paths_become_case_labels(self):
        tree = Path("/work/trees/cudf-0123456789")
        record = {"errors": [f"{tree}/cpp/x.cpp:3:1: error: e"], "uri": f"file://{tree}/cpp/x.cpp",
                  "other": f"{ROOT}/targets/rmm/build/include/y.hpp"}
        self.assertEqual(relative(record, tree, "case@0123456789"),
                         {"errors": ["case@0123456789/cpp/x.cpp:3:1: error: e"], "uri": "case@0123456789/cpp/x.cpp",
                          "other": "targets/rmm/build/include/y.hpp"})

    def test_flags_expand_placeholders(self):
        self.assertEqual(expand(["-I{tree}/include", "--cuda-path={cuda}"], {"tree": "/t", "cuda": "/c"}),
                         ["-I/t/include", "--cuda-path=/c"])

    def test_every_case_has_compile_commands(self):
        cases = json.loads((ROOT / "corpus/real-bugs/cases.json").read_text())["cases"]
        commands = json.loads((ROOT / "corpus/real-bugs/commands.json").read_text())["cases"]
        self.assertEqual(sorted(case["id"] for case in cases), sorted(commands))
        for case in cases:
            self.assertTrue(case["siteLines"] and all(type(n) is int for n in case["siteLines"]), case["id"])
            self.assertEqual(case["inRuleScope"], case["rule"] is not None, case["id"])


if __name__ == "__main__":
    unittest.main()
