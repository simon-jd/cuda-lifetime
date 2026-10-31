#!/usr/bin/env python3
"""Native CSA integration check: executable, semantic pairs, paths, coverage."""
import json
import pathlib
import re
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urlparse


def check_allocation_provenance(diagnostics):
    for diagnostic in diagnostics:
        if 'semantic=S5' not in diagnostic['message']['text']:
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
    actual = {(r['message']['text'].split()[0], r['locations'][0]['physicalLocation']['region']['startLine']) for r in results}
    expected = {(line.split('// ')[1].strip(), number) for number, line in enumerate((root / 'cases.cpp').read_text().splitlines(), 1) if '// CL' in line}
    assert actual == expected, (actual, expected, run.stderr)
    assert all(r.get('codeFlows') for r in results), 'CSA path diagnostics missing'
    check_allocation_provenance(results)
    cov = json.loads(coverage.read_text())
    assert cov['tus']['parsed'] == 1 and not cov['tus']['failed'], cov
    assert cov['llvmVersion'] == '23.1.3', cov
    small = subprocess.run([sys.argv[1], str(root / 'cases.cpp'), '--sarif', str(sarif), '--coverage', str(coverage), '--max-nodes', '1', '--', '-std=c++17'], text=True, capture_output=True)
    assert small.returncode == 0, small.stderr
    assert json.loads(coverage.read_text())['budgetExhausted'], 'Exceeded node budget was not recorded'
    kernel_source = root / 'kernel.cu'
    kernel_expected = {number for number, line in enumerate(kernel_source.read_text().splitlines(), 1) if '// CL3' in line}
    for mode in ('legacy', 'per-thread'):
        kernel = subprocess.run([sys.argv[1], str(kernel_source), '--sarif', str(sarif), '--coverage', str(coverage), '--', '-x', 'cuda', '--cuda-host-only', '-nocudainc', '-nocudalib', '-std=c++17', '-Xclang', '-fgpu-default-stream=' + mode], text=True, capture_output=True)
        assert kernel.returncode == 0, kernel.stderr
        diagnostics = [r for run in json.loads(sarif.read_text())['runs'] for r in run['results']]
        actual_lines = {r['locations'][0]['physicalLocation']['region']['startLine'] for r in diagnostics}
        assert actual_lines == kernel_expected and len(diagnostics) == len(kernel_expected), (mode, diagnostics)
        assert all('CL3 [error;' in r['message']['text'] for r in diagnostics), diagnostics
        check_allocation_provenance(diagnostics)
        cov = json.loads(coverage.read_text())
        assert cov['kernelLaunches'] == 7 and not cov['unmodeledCudaCalls'], cov
    mixed_source = root / 'mixed_default_streams.cpp'
    mixed_expected = {(match.group(1), match.group(2), number)
                      for number, line in enumerate(mixed_source.read_text().splitlines(), 1)
                      if (match := re.search(r'// (CL[1-4]) (error|warning|note)', line))}
    for mode in ('legacy', 'per-thread'):
        flags = ['-std=c++17']
        if mode == 'per-thread':
            flags.append('-DCUDA_API_PER_THREAD_DEFAULT_STREAM=1')
        mixed = subprocess.run([sys.argv[1], str(mixed_source), '--sarif', str(sarif), '--coverage', str(coverage), '--', *flags], text=True, capture_output=True)
        assert mixed.returncode == 0, mixed.stderr
        diagnostics = [r for run in json.loads(sarif.read_text())['runs'] for r in run['results']]
        mixed_actual = {(match.group(1), match.group(2), r['locations'][0]['physicalLocation']['region']['startLine'])
                        for r in diagnostics
                        if (match := re.match(r'(CL[1-4]) \[(error|warning|note);', r['message']['text']))}
        assert mixed_actual == mixed_expected and len(diagnostics) == len(mixed_expected), (mode, mixed_actual, mixed_expected, mixed.stderr)
        cov = json.loads(coverage.read_text())
        assert cov['defaultStreamMode'] == mode and not cov['unmodeledCudaCalls'] and not cov['budgetExhausted'], cov
    invalid = subprocess.run([sys.argv[1], str(root / 'cases.cpp'), '--sarif', str(sarif), '--coverage', str(coverage), '--', '-fdeliberately-unknown-cuda-lifetime-option'], text=True, capture_output=True)
    assert invalid.returncode != 0, invalid.stderr
    assert json.loads(coverage.read_text())['tus']['failed'], 'driver diagnostic must fail coverage'
    fixtures = [(name, []) for name in (
        'event_elapsed.cpp', 'pending_uses.cpp', 'macro_ranges.cpp', 'artifact_order_z.cpp', 'checked_return_helpers.cpp',
        'opaque_calls.cpp', 'sync_copies.cpp', 'graph_capture.cpp', 'event_history.cpp', 'semantic_repros.cpp',
        'alias_constraints.cpp', 'copy_ranges.cpp', 'registered_range.cpp', 'handle_aliases.cpp', 'event_snapshots.cpp',
        'ordinary_free_note.cpp', 'default_stream_distinct.cpp', 'default_stream_handles.cpp', 'empty_model_alias.cpp')]
    fixtures.append(('default_stream_handles.cpp', ['-DCUDA_API_PER_THREAD_DEFAULT_STREAM=1']))
    for fixture, flags in fixtures:
        source = root / fixture
        expected = {(match.group(1), match.group(2), number)
                    for number, line in enumerate(source.read_text().splitlines(), 1)
                    if (match := re.search(r'// (CL[1-4]) (error|warning|note)', line))}
        run = subprocess.run([sys.argv[1], str(source), '--sarif', str(sarif), '--coverage', str(coverage), '--include-contract', '--', '-std=c++17', *flags], text=True, capture_output=True)
        assert run.returncode == 0, run.stderr
        diagnostics = [r for run in json.loads(sarif.read_text())['runs'] for r in run['results']]
        actual = {(match.group(1), match.group(2), r['locations'][0]['physicalLocation']['region']['startLine'])
                  for r in diagnostics if (match := re.match(r'(CL[1-4]) \[(error|warning|note);', r['message']['text']))}
        assert actual == expected and len(diagnostics) == len(expected), (fixture, flags, actual, expected, run.stderr)
        assert all(r.get('codeFlows') for r in diagnostics), (fixture, 'CSA path diagnostics missing')
        cov = json.loads(coverage.read_text())
        assert cov['tus']['parsed'] == 1 and not cov['tus']['failed'], (fixture, cov)
        assert not cov['unmodeledCudaCalls'] and not cov['budgetExhausted'], cov
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
    names = subprocess.run([sys.argv[1], str(root / 'project_cuda_names.cpp'), '--sarif', str(sarif), '--coverage', str(coverage), '--', '-std=c++17'], text=True, capture_output=True)
    assert names.returncode == 0, names.stderr
    assert not [r for run in json.loads(sarif.read_text())['runs'] for r in run['results']]
    cov = json.loads(coverage.read_text())
    assert set(cov['unmodeledCudaCalls']) == {'cudaGraphCreate'}, cov['unmodeledCudaCalls']
    assert not cov['astScan']['skippedNoCudaCalls'], cov
print('Native CL1–CL4 pairs, CUDA kernels, CSA paths, flags, and budget coverage passed')
