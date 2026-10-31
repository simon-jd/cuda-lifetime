#!/usr/bin/env python3
"""Independent adversarial native review; labels are handwritten from semantics.

These development fixtures were exposed during implementation and are not a
holdout corpus. The runner does not derive expected results from checker output.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--native', default='build/cuda-lifetime-native')
parser.add_argument('--out', default='results/review/final-native')
args = parser.parse_args()
native = (ROOT / args.native).resolve()
out = (ROOT / args.out).resolve()
out.mkdir(parents=True, exist_ok=True)
binary_hash = hashlib.sha256(native.read_bytes()).hexdigest()
# Each tuple: unique case name, source, preprocessor flags, exact findings.
CASES = [
    ('semantic', 'semantic_repros.cpp', [], {(8, 'CL1', 'error'), (16, 'CL1', 'warning')}),
    ('pointer-alias', 'alias_constraints.cpp', [], {(6, 'CL1', 'error'), (16, 'CL3', 'error')}),
    ('ranges', 'copy_ranges.cpp', [], {(6, 'CL1', 'error')}),
    ('registered-range', 'registered_range.cpp', [], {(6, 'CL1', 'error')}),
    ('handle-alias', 'handle_aliases.cpp', [], set()),
    ('event-snapshots', 'event_snapshots.cpp', [], {(18, 'CL1', 'error'), (49, 'CL1', 'error')}),
    ('ordinary-free-note', 'ordinary_free_note.cpp', [], {(6, 'CL3', 'note')}),
    ('default-distinct', 'default_stream_distinct.cpp', [], {(7, 'CL3', 'error')}),
    ('default-legacy', 'default_stream_handles.cpp', [], set()),
    ('default-per-thread', 'default_stream_handles.cpp', ['-DCUDA_API_PER_THREAD_DEFAULT_STREAM=1'], set()),
]
summary = {'binarySha256':binary_hash, 'oracle':'handwritten semantic development fixtures, not holdout', 'cases':[]}
for name, filename, flags, expected in CASES:
    source = ROOT / 'tests/review' / filename
    sarif, cov = out/(name+'.sarif'), out/(name+'-coverage.json')
    command = [str(native), str(source), '--sarif', str(sarif), '--coverage', str(cov), '--', '-std=c++17', *flags]
    run = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
    (out/(name+'.log')).write_text(run.stdout+run.stderr)
    row = {'name':name, 'source':str(source.relative_to(ROOT)), 'sourceSha256':hashlib.sha256(source.read_bytes()).hexdigest(), 'command':command, 'returncode':run.returncode, 'expected':sorted(expected)}
    if run.returncode == 0 and sarif.exists() and cov.exists():
        reports = [result for run in json.loads(sarif.read_text())['runs'] for result in run['results']]
        actual = []
        for report in reports:
            match = re.match(r'(CL[1-4]) \[(error|warning|note);', report['message']['text'])
            actual.append((report['locations'][0]['physicalLocation']['region']['startLine'], *(match.groups() if match else ('unknown','unknown'))))
        coverage = json.loads(cov.read_text())
        row.update(actual=sorted(actual), pathsPresent=all(result.get('codeFlows') for result in reports), coverage=coverage)
        row['passed'] = (len(actual)==len(expected) and set(actual)==expected and row['pathsPresent'] and coverage['tus']['parsed']==1 and not coverage['tus']['failed'] and not coverage['budgetExhausted'] and not coverage['unmodeledCudaCalls'])
    else:
        row['passed'] = False
    summary['cases'].append(row)
summary['binaryUnchanged'] = binary_hash == hashlib.sha256(native.read_bytes()).hexdigest()
summary['passed'] = summary['binaryUnchanged'] and all(row['passed'] for row in summary['cases'])
(out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps({'passed':summary['passed'],'binaryUnchanged':summary['binaryUnchanged'],'cases':[{key:row[key] for key in ('name','passed','expected','actual') if key in row} for row in summary['cases']]},indent=2))
raise SystemExit(0 if summary['passed'] else 1)
