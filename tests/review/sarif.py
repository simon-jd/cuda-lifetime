#!/usr/bin/env python3
"""Check real native review artifacts, including CSA visitor events and enrichment."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tool.runner import enrich_sarif

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--inputs', nargs='+', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
rows = []


def check_location(location, artifacts):
    physical = location['physicalLocation']
    artifact = physical['artifactLocation']
    uri = artifact['uri']
    if 'index' in artifact:
        assert artifacts[artifact['index']]['location']['uri'] == uri, 'artifact index mismatch'
    parsed = urlparse(uri)
    assert parsed.scheme == 'file', 'review fixtures must use real file locations'
    source = Path(unquote(parsed.path))
    lines = source.read_text().splitlines()
    region = physical['region']
    start = region['startLine']
    end = region.get('endLine', start)
    assert 1 <= start <= end <= len(lines), f'invalid line span {region}'
    start_col, end_col = region.get('startColumn', 1), region.get('endColumn', 1)
    assert 1 <= start_col <= len(lines[start - 1]) + 1, f'invalid start column {region}'
    assert 1 <= end_col <= len(lines[end - 1]) + 1, f'invalid end column {region}'
    if start == end:
        assert start_col <= end_col, f'reversed columns {region}'
    return uri


for folder in args.inputs:
    for path in sorted(folder.glob('*.sarif')):
        row = {'file':str(path), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        try:
            document = json.loads(path.read_text())
            assert document['version'] == '2.1.0' and document['runs']
            results = []
            path_count, visitor_count, cross_file_count = 0, 0, 0
            for run in document['runs']:
                driver = run['tool']['driver']
                assert driver['name'] in ('cuda-lifetime-native', 'cuda-lifetime')
                assert driver['version']
                for result in run['results']:
                    match = re.match(r'(CL[1-4]) \[(error|warning|note);', result['message']['text'])
                    assert match, 'missing rule/severity encoding'
                    assert result['ruleId'] == 'cuda.AsyncLifetime'
                    assert driver['rules'][result['ruleIndex']]['id'] == result['ruleId']
                    artifacts = run.get('artifacts', [])
                    for location in result['locations']:
                        check_location(location, artifacts)
                    paths = result['codeFlows'][0]['threadFlows']
                    assert paths and paths[0]['locations'], 'missing native CSA path'
                    for flow in paths:
                        locations = flow['locations']
                        uris = {check_location(item['location'], artifacts) for item in locations}
                        messages = [item['location']['message']['text'] for item in locations]
                        assert messages[-1] == result['message']['text'], 'path lost terminal report'
                        if match[1] in ('CL1', 'CL2', 'CL3'):
                            # This exact message is emitted by PendingVisitor::VisitNode
                            # on an ExplodedNode transition, not by the report's addNote.
                            indices = [i for i, message in enumerate(messages)
                                       if message == 'Asynchronous operation is queued here and remains pending']
                            assert indices and indices[0] < len(messages) - 1, 'missing enqueue-to-use visitor path'
                            visitor_count += 1
                        path_count += 1
                        cross_file_count += len(uris) > 1
                    results.append((match.groups(), copy.deepcopy(result['codeFlows'])))
            enriched, suppressed = enrich_sarif(copy.deepcopy(document), 'legacy', True)
            enriched_results = [result for run in enriched for result in run['results']]
            assert suppressed == 0 and len(enriched_results) == len(results)
            for ((rule, level), flows), enriched_result in zip(results, enriched_results):
                assert (enriched_result['ruleId'], enriched_result['level']) == (rule, level)
                assert enriched_result['codeFlows'] == flows, 'Python enrichment changed CSA paths'
            row.update(passed=True, results=len(results), paths=path_count,
                       pendingVisitorPaths=visitor_count, crossFilePaths=cross_file_count)
        except (AssertionError, OSError, ValueError, KeyError, TypeError, IndexError) as error:
            row.update(passed=False, error=str(error))
        rows.append(row)
summary = {'passed':bool(rows) and all(row['passed'] for row in rows), 'artifacts':rows,
           'provenance':'native artifacts only; visitor message checked against PendingVisitor::VisitNode source'}
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
raise SystemExit(0 if summary['passed'] else 1)
