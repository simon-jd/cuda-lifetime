"""summarize_scan on a synthetic two-alert scan: no analyzer is run."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.summarize_scan import PROJECTS, function, summarize


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class SummarizeScanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.scan, self.out = self.root / "scan", self.root / "out"
        self.out.mkdir()
        for project, prefix in PROJECTS.items():
            source = self.root / prefix / "src/a.cpp"
            results = []
            if project == "faiss":
                results = [{"ruleId": "CL1", "level": "warning", "message": {"text": "m"},
                            "locations": [{"physicalLocation": {"artifactLocation": {"uri": source.as_uri(), "index": 0},
                                                                "region": {"startLine": 7}}}]}]
            write(self.scan / project / "out.sarif", {"version": "2.1.0", "runs": [{
                "tool": {"driver": {"name": "cuda-lifetime-native", "version": "0.4.0 (clang version 23.1.3)"}},
                "artifacts": [{"location": {"uri": source.as_uri(), "index": 0}}], "results": results}]})
            write(self.scan / project / "coverage.json", {
                "tus": {"total": 2, "parsed": 1, "failed": [{"file": str(self.root / prefix / "src/b.cpp")}]},
                "budget": {"maxNodes": 10, "timeoutSeconds": 1},
                "budgetExhausted": [{"file": str(source), "function": "f"}, {"file": str(source), "reason": "wall-clock-timeout"}],
                "coverageUnknown": [{"file": str(source), "reason": "r"}],
                "unmodeledCudaCalls": {"cudaGraphCreate": 2, "rmm::detail::memcpy_async:unknown-copy-kind": 1},
                "visitCounts": {"functionsAnalyzed": 4},
                "orchestration": {"wallSeconds": 3.25, "jobs": 2},
                "profiles": [{"file": str(source), "seconds": 1.5, "peakRssKiB": 10, "functionsAnalyzed": 4},
                             {"file": str(self.root / prefix / "src/b.cpp"), "seconds": 0.5, "peakRssKiB": None}]})
        self.triage = self.root / "triage.csv"
        self.write_triage(self.triage, [("faiss", "src/a.cpp", 7, "CL1", "warning", "portability")])

    def write_triage(self, path, rows):
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["project", "file", "line", "rule", "level", "message", "category", "reason"])
            writer.writerows([[*row[:5], "m", row[5], "r"] for row in rows])

    def test_summary_sarif_and_timings(self):
        previous = self.root / "previous.csv"
        with previous.open("w", newline="") as stream:  # the 0.2.0 triage's column names
            csv.writer(stream).writerows([["target", "line", "judgment"], ["faiss", 7, "false-positive"],
                                          ["rmm", 1, "insufficient-evidence"]])
        summary = summarize(self.scan, self.triage, self.out, previous, self.root, "0.2.0")
        self.assertEqual(summary["version"], "0.4.0")
        self.assertEqual(summary["alerts"], {"cuda-samples": 0, "faiss": 1, "rmm": 0})
        self.assertEqual(summary["alertsByCategory"], {"portability": 1})
        self.assertEqual(summary["previous"], {"version": "0.2.0", "alerts": {"cuda-samples": 0, "faiss": 1, "rmm": 1},
                                               "alertsByCategory": {"false-positive": 1, "insufficient-evidence": 1}})
        faiss = summary["coverage"]["faiss"]
        self.assertEqual((faiss["functionsAnalyzed"], faiss["functionsOverBudget"], faiss["wallClockTimeouts"]), (4, 1, 1))
        self.assertEqual(faiss["failed"], [".deps/faiss-1.15.1/src/b.cpp"])
        self.assertEqual(faiss["unmodeledCudaFunctions"], ["cudaGraphCreate", "rmm::detail::memcpy_async"])
        self.assertEqual(faiss["tuSeconds"]["sum"], 2.0)
        sarif = json.loads((self.out / "faiss.sarif").read_text())
        self.assertEqual(sarif["runs"][0]["artifacts"][0]["location"]["uri"], ".deps/faiss-1.15.1/src/a.cpp")
        with (self.out / "timings.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]["file"], ".deps/cuda-samples-13.2/src/a.cpp")
        self.assertEqual(rows[0]["functionsOverBudget"], "1")

    def test_triage_must_match_the_scan(self):
        self.write_triage(self.triage, [("faiss", "src/a.cpp", 8, "CL1", "warning", "portability")])
        with self.assertRaises(ValueError):
            summarize(self.scan, self.triage, self.out, None, self.root)

    def test_unmodeled_call_keys(self):
        self.assertEqual(function("cudaMemcpyAsync:unknown-stream"), "cudaMemcpyAsync")
        self.assertEqual(function("ns::wrapper"), "ns::wrapper")
        self.assertEqual(function("kernel-launch:unknown-pointer-operand"), "kernel-launch")


if __name__ == "__main__":
    unittest.main()
