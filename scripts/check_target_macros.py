#!/usr/bin/env python3
"""Collect reproducible macro evidence from real imported target commands."""

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from translate import macro_diff_compdb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets', nargs='+', default=['cuda-samples', 'faiss', 'rmm', 'nvbench'])
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--out', default=str(ROOT / 'results/m0/target-macros'))
    parser.add_argument('--append', action='store_true', help='retain existing summaries for targets not selected in this run')
    args = parser.parse_args()
    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    summary = {'schemaVersion': 1, 'selection': 'first two distinct CUDA source files and first host source file; fill to three distinct files in database order when a language is absent', 'targets': {}}
    if args.append and (destination / 'summary.json').exists():
        existing = json.loads((destination / 'summary.json').read_text())
        summary['targets'].update(existing['targets'])
    for target in args.targets:
        source = ROOT / 'targets' / target / 'clang/translation-report.json'
        translation = json.loads(source.read_text())
        entries = translation['entries']
        selected, seen = [], set()

        def pick(candidates, count):
            for entry in candidates:
                key = (entry.get('directory'), entry.get('file'))
                if key in seen:
                    continue
                selected.append(copy.deepcopy(entry))
                seen.add(key)
                count -= 1
                if count == 0:
                    break

        cuda = [e for e in entries if e.get('language') == 'cuda']
        host = [e for e in entries if e.get('language') != 'cuda']
        pick(cuda, 2)
        pick(host, 1)
        if len(selected) < 3:
            pick(entries, 3 - len(selected))
        target_directory = destination / target
        target_directory.mkdir(parents=True, exist_ok=True)
        sampled = {**translation, 'entries': selected,
                   'sampling': {'sourceReport': str(source), 'availableCuda': len(cuda),
                                'availableHost': len(host), 'selectedOriginalIndices': [e['index'] for e in selected]}}
        (target_directory / 'translation-report.json').write_text(json.dumps(sampled, indent=2) + '\n')
        result = macro_diff_compdb(target_directory, timeout=args.timeout)
        details = []
        for entry in result['entries']:
            differences = entry.get('requiredDifferences')
            details.append({'index': entry['index'], 'file': entry['file'], 'status': entry['status'],
                            'scope': entry.get('scope'), 'reason': entry.get('reason'),
                            'requiredMacrosEqual': differences['equal'] if differences else None,
                            'observedRequiredMacrosEqual': entry.get('observedRequiredDifferences', {}).get('equal'),
                            'requiredDifferences': differences,
                            'report': str((target_directory / 'macro-diff.json').relative_to(ROOT))})
        summary['targets'][target] = {'availableCuda': len(cuda), 'availableHost': len(host),
                                       'selected': len(selected), 'entries': details}
        all_entries = [entry for item in summary['targets'].values() for entry in item['entries']]
        summary['hostVerificationSummary'] = {
            'compared': sum(entry['status'] == 'compared' for entry in all_entries),
            'projectOnly': sum(entry['status'] == 'project-only' for entry in all_entries),
            'cudaHostMacroParityVerified': False,
        }
        (destination / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(target + ': ' + json.dumps([(entry['status'], entry['requiredMacrosEqual']) for entry in details]), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
