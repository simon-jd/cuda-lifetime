"""Collect real preprocessor evidence with an explicit project-only fallback."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from .compdb import TranslationError, split_nvcc_list


def parse_macros(text):
    """Parse a compiler -dM dump, including empty and function-like definitions."""
    macros = {}
    for line in text.splitlines():
        match = re.match(r'^\s*#\s*define\s+([A-Za-z_]\w*(?:\([^\n]*?\))?)(?:\s+(.*))?$', line)
        if match:
            macros[match[1]] = match[2] or ''
    return macros


def compare_macros(left, right, names=None):
    """Return value-sensitive differences between two macro dictionaries."""
    selected = set(left) | set(right) if names is None else set(names)
    only_left = {name: left[name] for name in sorted(selected & (left.keys() - right.keys()))}
    only_right = {name: right[name] for name in sorted(selected & (right.keys() - left.keys()))}
    changed = {name: {'original': left[name], 'translated': right[name]}
               for name in sorted(selected & left.keys() & right.keys()) if left[name] != right[name]}
    return {'onlyOriginal': only_left, 'onlyTranslated': only_right, 'changed': changed,
            'equal': not (only_left or only_right or changed)}


def _project_macros(args):
    expanded, i = [], 1
    while i < len(args):
        token = args[i]
        key, equal, val = token.partition('=')
        if key in ('-Xcompiler', '--compiler-options'):
            if not equal:
                i += 1
                if i == len(args):
                    raise TranslationError('missing host compiler option')
                val = args[i]
            expanded.extend(split_nvcc_list(val))
        else:
            expanded.append(token)
        i += 1
    macros, i = {}, 0
    while i < len(expanded):
        arg = expanded[i]
        operation, payload = None, None
        if arg in ('-D', '-U', '--define-macro', '--undefine-macro'):
            operation = 'D' if arg in ('-D', '--define-macro') else 'U'
            i += 1
            if i < len(expanded):
                payload = expanded[i]
        elif arg.startswith(('-D', '-U')):
            operation, payload = arg[1], arg[2:]
        elif arg.startswith(('--define-macro=', '--undefine-macro=')):
            operation = 'D' if arg.startswith('--define-macro=') else 'U'
            payload = arg.split('=', 1)[1]
        if operation and payload:
            name, equal, value = payload.partition('=')
            # Retain undefinition intent so a lost -U is also detected.
            macros[name] = (value if equal else '1') if operation == 'D' else None
        i += 1
    return macros


def _preprocess_command(args, nvcc=False):
    result, i = [], 0
    while i < len(args) and Path(args[i]).name in ('ccache', 'sccache'):
        i += 1
    removed_values = {'-o', '--output-file', '-MF', '-MT', '-MQ', '--keep-dir', '-keep-dir'}
    removed_single = {'-c', '--compile', '-dc', '--device-c', '-MD', '-MMD', '-MP',
                      '-fsyntax-only', '--keep', '-keep', '-E', '-dM'}
    while i < len(args):
        arg = args[i]
        key, equal, val = arg.partition('=')
        if key in removed_values:
            i += 1 if equal else 2
            continue
        if arg in removed_single:
            i += 1
            continue
        if arg.startswith(('-MF', '-MT', '-MQ')) and len(arg) > 3:
            i += 1
            continue
        if key in ('-Xcompiler', '--compiler-options'):
            if not equal:
                i += 1
                val = args[i]
            # Recursively remove host dependency/output flags, keeping nvcc's
            # original comma-list quoting convention for macro values.
            clean = _preprocess_command(['host', *split_nvcc_list(val)])[1:-2]
            if clean:
                # A comma that belonged to one host argument must stay protected
                # when rebuilding nvcc's list after output flags are removed.
                encoded = []
                for token in clean:
                    chars, quote = [], None
                    for char in token:
                        if char in ('"', "'"):
                            quote = None if quote == char else char if quote is None else quote
                        if char == ',' and quote is None:
                            chars.append('\\')
                        chars.append(char)
                    encoded.append(''.join(chars))
                result.append('-Xcompiler=' + ','.join(encoded))
        else:
            result.append(arg)
        i += 1
    return result + (['-E', '-Xcompiler=-dM'] if nvcc else ['-E', '-dM'])


def _collect(arguments, directory, timeout, *, raw_evidence=None):
    try:
        process = subprocess.run(arguments, cwd=directory, text=True, encoding='utf-8',
                                 errors='replace', stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        evidence = {'status': 'unavailable', 'command': arguments, 'reason': str(error)}
        if raw_evidence:
            from .compatibility import save_capture
            output = getattr(error, 'stdout', None) or ''
            stderr = getattr(error, 'stderr', None) or ''
            evidence.update(stdout=output.decode('utf-8', 'replace') if isinstance(output, bytes) else output,
                            stderr=stderr.decode('utf-8', 'replace') if isinstance(stderr, bytes) else stderr)
            save_capture(evidence, *raw_evidence)
        return evidence, {}
    evidence = {'status': 'collected', 'command': arguments, 'returncode': process.returncode,
                'stderr': process.stderr[-8000:]}
    if raw_evidence:
        from .compatibility import save_capture
        evidence.update(stdout=process.stdout, stderr=process.stderr)
        save_capture(evidence, *raw_evidence)
    macros = parse_macros(process.stdout)
    if process.returncode != 0 or not macros:
        evidence.update(status='unavailable', reason='preprocessor failed' if process.returncode else 'no macro definitions in preprocessor output')
        return evidence, {}
    evidence['macroCount'] = len(macros)
    return evidence, macros


def macro_diff_compdb(compdb, out=None, timeout=30):
    """Run macro probes from translation-report.json and write macro-diff.json.

    compdb is the translated directory or its translation-report.json. The real
    nvcc preprocessing dump is attempted using -E -Xcompiler=-dM. For CUDA
    inputs this phase defines __CUDA_ARCH__ and does not establish host-pass
    macro equivalence. Its real output is retained, but the verification gate
    falls back to explicit project -D/-U options. Failed/empty compiler output
    also leaves builtin/default-stream verification unverified.
    The translation report is updated with per-entry macro evidence.
    """
    if timeout <= 0:
        raise TranslationError('macro probe timeout must be positive')
    path = Path(compdb)
    if path.is_dir():
        path /= 'translation-report.json'
    try:
        translation = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TranslationError(f'cannot read translation report {path}: {error}') from error
    if not isinstance(translation, dict) or not isinstance(translation.get('entries'), list):
        raise TranslationError('invalid translation report')
    from .compatibility import load_macro_profile, header_provenance, validate_resolved, guard_macro_outputs
    profile = load_macro_profile(translation, path.resolve().parent)
    destination = Path(out) if out else path.parent
    if profile:
        guard_macro_outputs(profile, translation, path, destination)
    destination.mkdir(parents=True, exist_ok=True)
    entries = []
    for entry in translation['entries']:
        record = {'index': entry['index'], 'file': entry.get('file')}
        if entry['status'] != 'translated':
            record.update(status='not-run', reason='translation-unknown')
        else:
            original = entry['referenceArguments'] if profile else entry['expandedArguments']
            translated = entry['translatedArguments']
            compiler = next((arg for arg in original if Path(arg).name not in ('ccache', 'sccache')), '')
            is_nvcc = Path(compiler).name == 'nvcc'
            original_command = _preprocess_command(original, nvcc=is_nvcc)
            translated_command = _preprocess_command(translated)
            raw_prefix = f"{entry['index']:04d}"
            raw_before = {'raw_evidence': (destination, raw_prefix + '-reference-macros')} if profile else {}
            raw_after = {'raw_evidence': (destination, raw_prefix + '-clang-macros')} if profile else {}
            before_evidence, before = _collect(original_command, entry['directory'], timeout, **raw_before)
            after_evidence, after = _collect(translated_command, entry['directory'], timeout, **raw_after)
            project_before, project_after = _project_macros(original), _project_macros(translated)
            record.update(original=before_evidence, translated=after_evidence,
                          projectOptions=compare_macros(project_before, project_after),
                          originalMacros=before, translatedMacros=after)
            nvcc_cuda = is_nvcc and entry.get('language') == 'cuda'
            record['scope'] = 'nvcc-preprocessing-host-unverified' if nvcc_cuda else 'host'
            if before_evidence['status'] == after_evidence['status'] == 'collected':
                required = set(project_before) | set(project_after)
                if entry.get('language') == 'cuda':
                    required.update(('__CUDACC__', 'CUDA_API_PER_THREAD_DEFAULT_STREAM'))
                if nvcc_cuda:
                    record.update(status='project-only', differences=compare_macros(project_before, project_after),
                                  observedPreprocessorDifferences=compare_macros(before, after),
                                  observedRequiredDifferences=compare_macros(before, after, required),
                                  reason='nvcc -E does not isolate the host preprocessing pass; actual dump retained but host macro equivalence is unverified',
                                  unverified=['nvcc host compiler builtins', 'host CUDA_API_PER_THREAD_DEFAULT_STREAM', 'host __CUDACC__', 'host header-defined macros'])
                    record['original']['containsCudaArchMacro'] = '__CUDA_ARCH__' in before
                else:
                    record.update(status='compared', differences=compare_macros(before, after),
                                  requiredDifferences=compare_macros(before, after, required))
            else:
                record.update(status='project-only', differences=compare_macros(project_before, project_after),
                              reason='real preprocessor macros unavailable; only explicit project -D/-U options compared',
                              unverified=['compiler builtins', 'CUDA_API_PER_THREAD_DEFAULT_STREAM', '__CUDACC__', 'header-defined macros'])
            if profile:
                record.update(referenceKind='compatibility-derived',
                              referenceArguments=original, originalArguments=entry['originalArguments'],
                              originalMacrosMeaning='macros of the explicit compatibility-derived reference, not the original dependency baseline')
                record['profileHeaderEvidence'] = header_provenance(profile, original, translated,
                    entry['directory'], timeout, destination, raw_prefix)
                entry['compatibility']['headerIdentity'] = record['profileHeaderEvidence']['status']
        entry['macroDiff'] = record
        entries.append(record)
    if profile:
        validate_resolved(profile)
        translation['compatibility']['headerIdentity'] = ('verified' if entries and all(
            entry.get('profileHeaderEvidence', {}).get('status') in ('verified', 'not-applicable') for entry in entries)
            else 'unverified')
    report = {'schemaVersion': 1, 'source': str(path.resolve()), 'entries': entries,
              'summary': {status: sum(e['status'] == status for e in entries)
                          for status in ('compared', 'project-only', 'not-run')}}
    (destination / 'macro-diff.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    path.write_text(json.dumps(translation, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return report
