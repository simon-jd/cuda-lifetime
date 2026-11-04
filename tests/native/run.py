#!/usr/bin/env python3
"""Native CSA integration check: executable, semantic pairs, paths, coverage."""
import json
import pathlib
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urlparse


def check_allocation_provenance(diagnostics):
    for diagnostic in diagnostics:
        if diagnostic['properties']['semanticId'] != 'S5':
            continue
        for flow in diagnostic['codeFlows'][0]['threadFlows']:
            locations = [item['location'] for item in flow['locations']]
            assert locations[-1]['message'] == diagnostic['message'], diagnostic
            origins = [location for location in locations[:-1]
                       if location['message']['text'] == 'Asynchronous operation is queued here and remains pending']
            assert origins, 'S5 missing allocation-state transition visitor'
            for origin in origins:
                physical = origin['physicalLocation']
                source = pathlib.Path(unquote(urlparse(physical['artifactLocation']['uri']).path))
                line = source.read_text().splitlines()[physical['region']['startLine'] - 1]
                assert 'cudaMallocAsync' in line, (source, line)

root = pathlib.Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as temp:
    sarif = pathlib.Path(temp) / 'out.sarif'
    coverage = pathlib.Path(temp) / 'coverage.json'
    run = subprocess.run([sys.argv[1], str(root / 'cases.cpp'), '--sarif', str(sarif), '--coverage', str(coverage), '--', '-std=c++17'], text=True, capture_output=True)
    assert run.returncode == 0, run.stdout + run.stderr
    data = json.loads(sarif.read_text())
    results = [r for run in data['runs'] for r in run['results']]
    actual = {(r['ruleId'], r['locations'][0]['physicalLocation']['region']['startLine']) for r in results}
    expected = {(line.split('// ')[1].strip(), number) for number, line in enumerate((root / 'cases.cpp').read_text().splitlines(), 1) if '// CL' in line}
    assert actual == expected, (actual, expected, run.stderr)
    assert all(r.get('codeFlows') for r in results), 'CSA path diagnostics missing'
    check_allocation_provenance(results)
    cov = json.loads(coverage.read_text())
    assert cov['tus']['parsed'] == 1 and not cov['tus']['failed'], cov
    assert cov['llvmVersion'] == '23.1.3', cov
    # One record per top-level function analyzed, with its nodes.
    assert len(cov['entryPoints']) == cov['functionsAnalyzed'], cov['entryPoints']
    assert cov['nodes'] == sum(e['nodes'] for e in cov['entryPoints']) and all(e['nodes'] > 0 for e in cov['entryPoints']), cov
    assert cov['nodesCounted'] == 'kept after reclamation' and all('profile' not in e for e in cov['entryPoints'])
    profiled = subprocess.run([sys.argv[1], str(root / 'cases.cpp'), '--sarif', str(sarif), '--coverage', str(coverage), '--node-profile', '--', '-std=c++17'], text=True, capture_output=True)
    assert profiled.returncode == 0, profiled.stderr
    profiled_cov = json.loads(coverage.read_text())
    assert profiled_cov['nodesCounted'] == 'all created' and profiled_cov['nodes'] >= cov['nodes'], (profiled_cov['nodes'], cov['nodes'])
    for entry in profiled_cov['entryPoints']:
        assert sum(count for _, count in entry['profile']['byFrame']) <= entry['nodes'], entry
        assert entry['profile']['byLine'] and entry['profile']['bySite'], entry
    small = subprocess.run([sys.argv[1], str(root / 'cases.cpp'), '--sarif', str(sarif), '--coverage', str(coverage), '--max-nodes', '1', '--', '-std=c++17'], text=True, capture_output=True)
    assert small.returncode == 0, small.stderr
    small_cov = json.loads(coverage.read_text())
    assert small_cov['budgetExhausted'], 'Exceeded node budget was not recorded'
    assert any(e['exhausted'] for e in small_cov['entryPoints']), small_cov['entryPoints']
    assert small_cov['functionsAnalyzed'] >= len(small_cov['budgetExhausted']), small_cov
    kernel_source = root / 'kernel.cu'
    kernel_expected = {number for number, line in enumerate(kernel_source.read_text().splitlines(), 1) if '// CL3' in line}
    for mode in ('legacy', 'per-thread'):
        kernel = subprocess.run([sys.argv[1], str(kernel_source), '--sarif', str(sarif), '--coverage', str(coverage), '--', '-x', 'cuda', '--cuda-host-only', '-nocudainc', '-nocudalib', '-std=c++17', '-Xclang', '-fgpu-default-stream=' + mode], text=True, capture_output=True)
        assert kernel.returncode == 0, kernel.stderr
        diagnostics = [r for run in json.loads(sarif.read_text())['runs'] for r in run['results']]
        actual_lines = {r['locations'][0]['physicalLocation']['region']['startLine'] for r in diagnostics}
        assert actual_lines == kernel_expected and len(diagnostics) == len(kernel_expected), (mode, diagnostics)
        assert all(r['ruleId'] == 'CL3' and r['level'] == 'error' for r in diagnostics), diagnostics
        check_allocation_provenance(diagnostics)
        cov = json.loads(coverage.read_text())
        assert cov['kernelLaunches'] == 7 and not cov['unmodeledCudaCalls'], cov
    invalid = subprocess.run([sys.argv[1], str(root / 'cases.cpp'), '--sarif', str(sarif), '--coverage', str(coverage), '--', '-fdeliberately-unknown-cuda-lifetime-option'], text=True, capture_output=True)
    assert invalid.returncode != 0, invalid.stderr
    assert json.loads(coverage.read_text())['tus']['failed'], 'driver diagnostic must fail coverage'
    # Annotated fixtures are lit tests (lit.cfg.py, verify.py); these two also
    # need checks of coverage counters and SARIF artifact indices.
    for fixture in ('pending_uses.cpp', 'artifact_order_z.cpp'):
        run = subprocess.run([sys.argv[1], str(root / fixture), '--sarif', str(sarif), '--coverage', str(coverage), '--include-contract', '--', '-std=c++17'], text=True, capture_output=True)
        assert run.returncode == 0, run.stderr
        diagnostics = [r for run in json.loads(sarif.read_text())['runs'] for r in run['results']]
        cov = json.loads(coverage.read_text())
        if fixture == 'pending_uses.cpp':
            assert cov['pendingArgumentUses'] > 0 and cov['pendingMemberCallUses'] > 0 and cov['pendingPointerEscapes'] > 0, cov
            assert any('opaque ownership' in entry['reason'] for entry in cov['coverageUnknown']), cov
        if fixture == 'artifact_order_z.cpp':
            document = json.loads(sarif.read_text())
            for sarif_run in document['runs']:
                artifacts = sarif_run['artifacts']
                assert len(artifacts) == 2
                # The diagnostic is in z.cpp, while its queue event is in a.h.
                assert artifacts[0]['location']['uri'].endswith('/artifact_order_z.cpp')
                assert artifacts[1]['location']['uri'].endswith('/artifact_order_a.h')
                assert [entry['location']['index'] for entry in artifacts] == [0, 1]
                locations = list(diagnostics[0]['locations'])
                for flow in diagnostics[0]['codeFlows']:
                    for thread in flow['threadFlows']:
                        locations.extend(item['location'] for item in thread['locations'])
                assert any(location.get('message', {}).get('text') == 'Asynchronous operation is queued here and remains pending'
                           and location['physicalLocation']['artifactLocation']['uri'].endswith('/artifact_order_a.h')
                           for location in locations), locations
                for location in locations:
                    reference = location['physicalLocation']['artifactLocation']
                    assert artifacts[reference['index']]['location']['uri'] == reference['uri'], reference
print('Native cases, CUDA kernels, CSA paths, flags, artifact order and budget coverage passed')
