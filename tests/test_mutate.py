"""Mutation provenance/replay tests, independent of analyzer diagnostics."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from eval.mutate import MutationError, digest, generate, unique_replace
from eval.mutation_report import compare


class MutationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "upstream"
        self.source.mkdir()
        self.original = "// original license\nvoid f() { sync(); read(); }\n"
        (self.source / "sample.cu").write_text(self.original)
        self.recipes = {"schemaVersion": 1, "upstream": {"tag": "test"}, "mutations": [{
            "id": "remove_sync", "category": "delete-synchronization", "source": "sample.cu",
            "sourceUrl": "https://example.invalid/fixture/sample.cu", "sourceSha256": digest(self.original),
            "edits": [{"before": "sync(); ", "after": ""}],
            "oracle": {"sites": [{"anchor": "read();", "ruleId": "CL1", "expect": "warning", "level": "error"}]}}]}

    def test_unique_replacement_rejects_missing_and_ambiguous_text(self):
        for source in ("no marker", "sync(); sync();"):
            with self.subTest(source=source), self.assertRaises(MutationError):
                unique_replace(source, "sync();", "")

    def test_generation_retains_upstream_and_license_and_resolves_oracle(self):
        manifest = generate(self.recipes, self.source, self.root / "out")
        self.assertEqual((self.source / "sample.cu").read_text(), self.original)
        self.assertEqual((self.root / "out/originals/sample.cu").read_text(), self.original)
        mutant = (self.root / "out/generated/remove_sync.cu").read_text()
        self.assertEqual(mutant, "// original license\nvoid f() { read(); }\n")
        self.assertEqual(manifest["mutants"][0]["oracle"]["sites"][0]["line"], 2)
        self.assertEqual(manifest["mutants"][0]["edits"][0]["matchCount"], 1)
        self.assertEqual(manifest["mutants"][0]["mutantSha256"], digest(mutant))

    def test_source_hash_mismatch_fails_before_output(self):
        (self.source / "sample.cu").write_text(self.original + "// changed\n")
        with self.assertRaisesRegex(MutationError, "source hash changed"):
            generate(self.recipes, self.source, self.root / "out")
        self.assertFalse((self.root / "out").exists())

    def test_replay_produces_identical_bytes_and_retains_first_generation_date(self):
        first = generate(self.recipes, self.source, self.root / "out")
        again = generate(self.recipes, self.source, self.root / "out")
        self.assertEqual(first, again)
        replay = generate(first["recipes"], self.source, self.root / "replayed", first)
        self.assertEqual(first["mutants"], replay["mutants"])
        self.assertEqual((self.root / "out/generated/remove_sync.cu").read_bytes(),
                         (self.root / "replayed/generated/remove_sync.cu").read_bytes())

    def test_tampered_generated_output_is_never_overwritten(self):
        generate(self.recipes, self.source, self.root / "out")
        path = self.root / "out/generated/remove_sync.cu"
        path.write_text("hand edit\n")
        with self.assertRaisesRegex(MutationError, "overwrite"):
            generate(self.recipes, self.source, self.root / "out")
        self.assertEqual(path.read_text(), "hand edit\n")

    def test_replay_rejects_changed_expected_hash(self):
        first = generate(self.recipes, self.source, self.root / "out")
        first["mutants"][0]["mutantSha256"] = "0" * 64
        with self.assertRaisesRegex(MutationError, "replayed output hashes differ"):
            generate(self.recipes, self.source, self.root / "replayed", first)

    def test_source_tree_is_not_an_allowed_destination(self):
        with self.assertRaisesRegex(MutationError, "outside"):
            generate(self.recipes, self.source, self.source / "mutants")

    def test_recipe_cannot_read_outside_pinned_source_tree(self):
        recipe = copy.deepcopy(self.recipes)
        recipe["mutations"][0]["source"] = "../secret.cu"
        with self.assertRaisesRegex(MutationError, "escapes"):
            generate(recipe, self.source, self.root / "out")


class MutationReportTests(unittest.TestCase):
    """Synthetic protocol tests; these are not claimed as real CSA evidence."""

    setUp = MutationTests.setUp

    def fixture(self, negative=False):
        self.output = self.root / "out"
        self.manifest = generate(self.recipes, self.source, self.output)
        row = self.manifest["mutants"][0]
        if negative:
            row["oracle"]["sites"][0].update(expect="none", level=None)
        self.mutant = str((self.output / row["file"]).resolve())
        self.baseline = str((self.output / row["baseline"]).resolve())
        self.coverage = {"tus": {"analyzed": [self.baseline, self.mutant], "failed": [], "translationUnknown": []},
                         "budgetExhausted": [], "coverageUnknown": [], "unmodeledCudaCalls": {}, "complete": True}
        self.sarif = {"version": "2.1.0", "runs": [{"tool": {"driver": {"name": "synthetic-protocol-fixture"}}, "results": []}]}
        self.logs = self.root / "logs"
        self.logs.mkdir()
        for index, source in enumerate((self.baseline, self.mutant)):
            native = copy.deepcopy(self.coverage)
            native["tus"]["analyzed"] = [source]
            (self.logs / f"{index}.coverage.json").write_text(json.dumps(native))

    def report(self):
        return compare(self.manifest, self.sarif, self.coverage, self.output, self.logs)

    def status(self):
        return self.report()["mutants"][0]["sites"][0]["status"]

    def diagnostic(self, rule="CL1", line=2, level="error"):
        return {"ruleId": rule, "level": level, "locations": [{"physicalLocation": {
            "artifactLocation": {"uri": Path(self.mutant).as_uri()}, "region": {"startLine": line}}}]}

    def update_native(self, mutate):
        path = self.logs / "1.coverage.json"
        native = json.loads(path.read_text())
        mutate(native)
        path.write_text(json.dumps(native))

    def test_relative_paths_resolve_against_paths_root(self):
        self.fixture()
        relative = {absolute: str(Path(absolute).relative_to(self.root.resolve())) for absolute in (self.baseline, self.mutant)}
        self.coverage["tus"]["analyzed"] = list(relative.values())
        for path in self.logs.glob("*.coverage.json"):
            native = json.loads(path.read_text())
            native["tus"]["analyzed"] = [relative[name] for name in native["tus"]["analyzed"]]
            path.write_text(json.dumps(native))
        diagnostic = self.diagnostic()
        diagnostic["locations"][0]["physicalLocation"]["artifactLocation"]["uri"] = relative[self.mutant]
        self.sarif["runs"][0]["results"] = [diagnostic]
        self.assertEqual(self.report()["mutants"][0]["sites"][0]["status"], "unverified-miss")
        report = compare(self.manifest, self.sarif, self.coverage, self.output, self.logs, self.root)
        self.assertEqual(report["mutants"][0]["sites"][0]["status"], "observed-at-required-site")

    def test_negative_requires_complete_telemetry(self):
        self.fixture(negative=True)
        self.assertEqual(self.status(), "verified-no-finding")
        del self.coverage["budgetExhausted"]
        self.assertEqual(self.status(), "unverified-no-finding")

    def test_incomplete_native_control_is_never_credited(self):
        self.fixture(negative=True)
        good = (self.logs / "1.coverage.json").read_text()
        changes = [lambda n: n.pop("unmodeledCudaCalls"),
                   lambda n: n.update(complete=False),
                   lambda n: n.update(unmodeledCudaCalls={"cudaUnknown": 1}),
                   lambda n: n.update(budgetExhausted=[{"file": self.mutant}]),
                   lambda n: n.update(coverageUnknown=[{"file": self.mutant}]),
                   lambda n: n["tus"].update(failed=[{"file": self.mutant}]),
                   lambda n: n["tus"].pop("translationUnknown")]
        for change in changes:
            with self.subTest(change=change):
                (self.logs / "1.coverage.json").write_text(good)
                self.update_native(change)
                self.assertEqual(self.status(), "unverified-no-finding")

    def test_missing_or_different_native_tu_not_credited(self):
        self.fixture(negative=True)
        self.update_native(lambda n: n["tus"].update(analyzed=[str(self.root / "other.cu")]))
        self.assertEqual(self.status(), "unverified-no-finding")
        (self.logs / "1.coverage.json").unlink()
        self.assertEqual(self.status(), "unverified-no-finding")

    def test_aggregate_failures_prevent_negative_credit(self):
        self.fixture(negative=True)
        self.coverage["tus"]["failed"] = [{"file": self.mutant, "reason": "timeout"}]
        self.assertEqual(self.status(), "unverified-no-finding")

    def test_unexplained_aggregate_incompleteness_prevents_negative_credit(self):
        self.fixture(negative=True)
        self.coverage["complete"] = False
        self.assertEqual(self.status(), "unverified-no-finding")

    def test_positive_matching_requires_exact_rule_line_and_level(self):
        self.fixture()
        self.assertEqual(self.status(), "missed-required-site")
        for diagnostic in (self.diagnostic(rule="CL2"), self.diagnostic(line=3)):
            self.sarif["runs"][0]["results"] = [diagnostic]
            self.assertEqual(self.status(), "missed-required-site")
        self.sarif["runs"][0]["results"] = [self.diagnostic(level="warning")]
        self.assertEqual(self.status(), "severity-mismatch")
        self.sarif["runs"][0]["results"] = [self.diagnostic()]
        self.assertEqual(self.status(), "observed-at-required-site")

    def test_missing_positive_kept_when_analysis_failed(self):
        self.fixture()
        self.coverage["tus"]["analyzed"].remove(self.mutant)
        self.assertEqual(self.status(), "unverified-miss")

    def test_unexpected_negative_diagnostic_is_kept_with_incomplete_coverage(self):
        self.fixture(negative=True)
        self.coverage["tus"]["failed"] = [{"file": self.mutant}]
        self.sarif["runs"][0]["results"] = [self.diagnostic(line=3)]
        self.assertEqual(self.status(), "unexpected-finding")

    def test_conflicting_native_logs_rejected(self):
        self.fixture(negative=True)
        other = json.loads((self.logs / "1.coverage.json").read_text())
        other["unmodeledCudaCalls"] = {"cudaUnknown": 1}
        (self.logs / "stale.coverage.json").write_text(json.dumps(other))
        with self.assertRaisesRegex(ValueError, "conflicting"):
            self.report()

    def test_changed_sources_rejected(self):
        self.fixture()
        Path(self.mutant).write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "changed after oracle"):
            self.report()


if __name__ == "__main__":
    unittest.main()
