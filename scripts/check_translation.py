#!/usr/bin/env python3
"""Translate target databases and record actual Clang host syntax feasibility."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from translate import translate_compdb
from translate.macros import _preprocess_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets', nargs='+', default=['cuda-samples', 'faiss', 'rmm'])
    parser.add_argument('--clang', default=str(ROOT / '.deps/LLVM-23.1.3-Linux-X64/bin/clang++'))
    parser.add_argument('--cuda-path', default='/usr/local/cuda-13.2')
    parser.add_argument('--gcc-toolchain', default=str(ROOT / '.deps/gcc13/usr'))
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--host-samples', type=int, default=10)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--out', default=str(ROOT / 'results/m0/translation-feasibility.json'))
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    logs = out.parent / 'translation-syntax-logs'
    logs.mkdir(exist_ok=True)
    report = {'schemaVersion': 1, 'scope': 'every CUDA TU plus first N host commands in database order',
              'compiler': args.clang, 'cudaPath': args.cuda_path, 'gccToolchain': args.gcc_toolchain,
              'hostSamplesPerTarget': args.host_samples, 'targets': {}}
    for target in args.targets:
        translation = translate_compdb(ROOT / 'targets' / target / 'build',
                                      ROOT / 'targets' / target / 'clang',
                                      cuda_path=args.cuda_path, clang=args.clang,
                                      gcc_toolchain=args.gcc_toolchain)
        entries = json.loads((ROOT / 'targets' / target / 'clang/compile_commands.json').read_text())
        cuda_entries = [e for e in entries if e['file'].endswith(('.cu', '.cuh'))]
        host_entries = [e for e in entries if e not in cuda_entries][:args.host_samples]
        candidates = [('cuda', e) for e in cuda_entries] + [('host', e) for e in host_entries]
        original_cuda = sum(e.get('language') == 'cuda' for e in translation['entries'])
        result = {'translation': translation['summary'], 'cudaTotal': original_cuda,
                  'hostTotal': len(entries) - len(cuda_entries), 'syntax': []}
        report['targets'][target] = result

        def check(number, kind, entry):
            # Only phase/output flags change; semantic options remain intact.
            command = _preprocess_command(entry['arguments'])[:-2]
            command += ['-fsyntax-only']
            started = time.monotonic()
            record = {'index': number, 'kind': kind, 'file': entry['file'], 'command': command,
                      'directory': entry['directory']}
            try:
                process = subprocess.run(command, cwd=entry['directory'], capture_output=True,
                                         text=True, timeout=args.timeout, check=False)
                record.update(returncode=process.returncode, passed=process.returncode == 0)
                log = logs / f'{target}-{number:04}.txt'
                log.write_text(process.stdout + process.stderr)
                record['log'] = str(log.relative_to(ROOT)) if log.is_relative_to(ROOT) else str(log)
                record['errors'] = [line for line in process.stderr.splitlines() if 'error:' in line][:10]
            except (OSError, subprocess.TimeoutExpired) as error:
                record.update(returncode=None, passed=False, errors=[str(error)])
            record['seconds'] = round(time.monotonic() - started, 3)
            return record

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(check, index, kind, entry)
                       for index, (kind, entry) in enumerate(candidates)]
            for future in as_completed(futures):
                record = future.result()
                result['syntax'].append(record)
                if len(result['syntax']) % 10 == 0:
                    print(f'{target}: {len(result["syntax"])}/{len(candidates)} checked', flush=True)
        result['syntax'].sort(key=lambda record: record['index'])
        result['summary'] = {}
        for kind in ('cuda', 'host'):
            tested = [r for r in result['syntax'] if r['kind'] == kind]
            passed = sum(r['passed'] for r in tested)
            denominator = original_cuda if kind == 'cuda' else len(tested)
            result['summary'][kind] = {'tested': len(tested), 'passed': passed,
                                       'translationUnknown': original_cuda - len(cuda_entries) if kind == 'cuda' else 0,
                                       'passRate': passed / denominator if denominator else None}
        out.write_text(json.dumps(report, indent=2) + '\n')
        print(target + ': ' + json.dumps(result['summary']), flush=True)
    return 0 if all(not r['translation']['translationUnknown']
                    and (r['summary']['cuda']['passRate'] is None or r['summary']['cuda']['passRate'] >= 0.9)
                    and r['summary']['host']['passed'] == r['summary']['host']['tested']
                    for r in report['targets'].values()) else 2


if __name__ == '__main__':
    raise SystemExit(main())
