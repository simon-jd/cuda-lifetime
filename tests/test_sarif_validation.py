"""Malformed-artifact tests, distinct from saved actual-native schema evidence."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

try:
    from tests.validate_sarif import validate
except ModuleNotFoundError as exc:
    if exc.name != "jsonschema":
        raise
    validate = None


@unittest.skipIf(validate is None, "optional jsonschema package is needed for schema tests")
class SarifValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "case.cpp"
        self.source.write_text("void invented() {}\n")
        physical = {"artifactLocation": {"uri": self.source.as_uri(), "index": 0}, "region": {"startLine": 1}}
        self.document = {"version": "2.1.0", "runs": [{
            "tool": {"driver": {"name": "synthetic-invalid-artifact-test", "rules": [{"id": "CL1"}]}},
            "artifacts": [{"location": {"uri": self.source.as_uri()}}], "results": [{
                "ruleId": "CL1", "ruleIndex": 0, "level": "warning", "message": {"text": "fixture"},
                "locations": [{"physicalLocation": physical}],
                "codeFlows": [{"threadFlows": [{"locations": [{"location": {
                    "physicalLocation": copy.deepcopy(physical), "message": {"text": "Operation is queued here"}}}]}]}]}]}]}
        self.schema = json.loads((Path(__file__).parent / "schema/sarif-schema-2.1.0.json").read_text())

    def failures(self):
        return validate(self.document, self.schema, self.root)[0]

    def result(self):
        return self.document["runs"][0]["results"][0]

    def test_valid_protocol_artifact_passes(self):
        failures, counts = validate(self.document, self.schema, self.root)
        self.assertEqual(failures, [])
        self.assertEqual(counts["traceLocations"], 1)

    def test_schema_rejects_zero_end_line(self):
        self.result()["locations"][0]["physicalLocation"]["region"]["endLine"] = 0
        self.assertTrue(any("schema" in failure for failure in self.failures()))

    def test_source_line_must_really_exist(self):
        self.result()["locations"][0]["physicalLocation"]["region"]["startLine"] = 99
        self.assertTrue(any("outside file" in failure for failure in self.failures()))

    def test_rule_index_must_agree_with_rule_id(self):
        self.result()["ruleId"] = "CL2"
        self.assertTrue(any("ruleId/ruleIndex disagree" in failure for failure in self.failures()))

    def test_artifact_index_must_agree_with_uri(self):
        other = self.root / "other.cpp"
        other.write_text("void other() {}\n")
        self.document["runs"][0]["artifacts"][0]["location"]["uri"] = other.as_uri()
        self.assertTrue(any("artifact URI/index disagree" in failure for failure in self.failures()))

    def test_missing_diagnostic_path_fails(self):
        self.result().pop("codeFlows")
        self.assertTrue(any("missing CSA diagnostic path" in failure for failure in self.failures()))


if __name__ == "__main__":
    unittest.main()
