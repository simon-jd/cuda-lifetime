"""Matching and path handling of scripts/run_real_bugs.py; no analyzer is run."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_real_bugs import ROOT, compiler_errors, expand, on_lines, overlay, relative, text_results


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


class BaselineOutputTests(unittest.TestCase):
    def test_notes_after_a_warning_are_its_flow(self):
        output = ("/t/a.cu:5:3: warning: Use of memory after it is freed [clang-analyzer-unix.Malloc]\n"
                  "/t/a.cu:3:3: note: Memory is released\n"
                  "/t/b.h:9:1: error: unknown type name 'x'\n"
                  "/t/a.cu:7:1: warning: dead store\n")
        self.assertEqual(text_results(output), [
            {"ruleId": "clang-analyzer-unix.Malloc", "level": "warning", "message": "Use of memory after it is freed",
             "uri": "/t/a.cu", "line": 5, "flow": [{"uri": "/t/a.cu", "line": 3}]},
            {"ruleId": "", "level": "warning", "message": "dead store", "uri": "/t/a.cu", "line": 7, "flow": []}])
        self.assertTrue(on_lines(text_results(output)[0], "a.cu", [3]))


    def test_source_excerpts_are_not_errors(self):
        output = ("/t/a.cu:3:10: fatal error: 'mpi.h' file not found\n"
                  "   88 | // error: expected\n"
                  "   79 |   if (const ::testing::AssertionResult gtest_ar = (expression))\n"
                  "clang++: Sema.cpp:10: void f(): Assertion `x' failed.\n"
                  "error: unable to handle compilation\n"
                  "/t/b.h:9:1: warning: x [bugprone-y]\n")
        self.assertEqual(compiler_errors(output), ["/t/a.cu:3:10: fatal error: 'mpi.h' file not found",
                                                   "clang++: Sema.cpp:10: void f(): Assertion `x' failed.",
                                                   "error: unable to handle compilation"])


class HeaderPatchTests(unittest.TestCase):
    PATCH = {"header": "cpp/include/lib/a.hpp", "includeRoot": "cpp/include",
             "pattern": "^(\\s*)constexpr int f<", "replacement": "\\1HD constexpr int f<"}

    def test_patched_copy_is_searched_first(self):
        with tempfile.TemporaryDirectory() as tree, tempfile.TemporaryDirectory() as temp:
            header = Path(tree) / "cpp/include/lib/a.hpp"
            header.parent.mkdir(parents=True)
            header.write_text("template <> constexpr int f<int>();\n  constexpr int f<char>();\n")
            flags = overlay(temp, Path(tree), [self.PATCH])
            self.assertEqual(flags, [f"-I{temp}/overlay0"])
            self.assertEqual((Path(temp) / "overlay0/lib/a.hpp").read_text(),
                             "template <> constexpr int f<int>();\n  HD constexpr int f<char>();\n")
            self.assertIn("f<char>", header.read_text())  # the project tree is unchanged

    def test_a_template_is_generated_under_its_output_name(self):
        patch = {"header": "src/lib.h.in", "includeRoot": "src", "output": "lib.h",
                 "edits": [{"pattern": "\\$\\{lib:Major\\}", "replacement": "2"},
                           {"pattern": "\\$\\{lib:Minor\\}", "replacement": "28"}]}
        with tempfile.TemporaryDirectory() as tree, tempfile.TemporaryDirectory() as temp:
            (Path(tree) / "src").mkdir()
            (Path(tree) / "src/lib.h.in").write_text("#define MAJOR ${lib:Major}\n#define MINOR ${lib:Minor}\n")
            self.assertEqual(overlay(temp, Path(tree), [patch]), [f"-I{temp}/overlay0"])
            self.assertEqual((Path(temp) / "overlay0/lib.h").read_text(), "#define MAJOR 2\n#define MINOR 28\n")

    def test_a_patch_that_matches_nothing_stops_the_run(self):
        with tempfile.TemporaryDirectory() as tree, tempfile.TemporaryDirectory() as temp:
            header = Path(tree) / "cpp/include/lib/a.hpp"
            header.parent.mkdir(parents=True)
            header.write_text("int g();\n")
            with self.assertRaises(SystemExit):
                overlay(temp, Path(tree), [self.PATCH])

    def test_commands_name_known_patches_and_existing_generated_headers(self):
        for corpus in ("real-bugs", "real-bugs-heldout"):
            path = ROOT / "corpus" / corpus / "commands.json"
            if not path.exists():
                continue
            commands = json.loads(path.read_text())
            for case in commands["cases"].values():
                for name in case.get("headerPatches", []):
                    self.assertIn(name, commands["headerPatches"])
                for name in case.get("generatedHeaders", []):
                    self.assertTrue((path.parent / "generated" / name).is_dir(), name)


if __name__ == "__main__":
    unittest.main()
