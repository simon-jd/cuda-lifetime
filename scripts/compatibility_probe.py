#!/usr/bin/env python3
"""Syntax-only compatibility probe from a hash-pinned candidate manifest.

No checker or holdout execution. A diagnostic subset never satisfies a full
parse gate. Original databases and arguments are retained alongside explicitly
registered compiler/CCCL path changes and explicit host-header discovery.
See docs/compatibility-profile.md.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.baseline import run_one
from scripts.replay_holdout import digest, header_files
from translate.compdb import _argv, expand_response_files, translate_entry
from translate.macros import _preprocess_command, _project_macros, compare_macros, parse_macros


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def check(value, message):
    if not value:
        raise ValueError(message)


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def tree_hash(root):
    root = Path(root).resolve()
    # Use lexical names to detect added/replaced files as well as byte changes.
    mapping = {str(path.relative_to(root)): digest(path) for path in root.rglob('*') if path.is_file()}
    check(mapping, 'empty dependency tree')
    return json_hash(mapping)


def input_path_identities(roots):
    """Retain aliases that a resolved-file hash set alone would lose."""
    identities, visited, pending = {}, set(), []
    for root in roots:
        root = Path(os.path.abspath(root))
        identities[str(root)] = str(root.resolve(strict=True))
        pending.append(root.resolve(strict=True))
    while pending:
        folder = pending.pop()
        if folder in visited:
            continue
        visited.add(folder)
        for entry in folder.iterdir():
            resolved = entry.resolve(strict=True)
            if entry.is_symlink():
                identities[str(entry)] = str(resolved)
            if resolved.is_dir():
                pending.append(resolved)
    return identities


def baseline_path_mismatches(expected, roots):
    mismatches = set()
    for name, target in expected.items():
        try:
            if str(Path(name).resolve(strict=True)) != target:
                mismatches.add(name)
        except OSError:
            mismatches.add(name)
    if roots:
        try:
            current = input_path_identities(roots)
            mismatches.update(name for name in set(current) | set(expected) if current.get(name) != expected.get(name))
        except OSError:
            mismatches.update(map(str, roots))
    return sorted(mismatches)


def rewrite_paths(arguments, changes, prepend=None):
    """Rewrite registered include roots only; never rewrite -D payloads."""
    out, ledger, pending = [], [], False
    for index, token in enumerate(arguments):
        replacement = token
        prefix, value = '', token
        if pending:
            pending = False
        elif token in ('-I', '-isystem', '--include-path', '--system-include'):
            pending = True
            out.append(token)
            continue
        else:
            match = re.match(r'(-isystem=|--include-path=|--system-include=|-I)(.+)', token)
            if not match:
                out.append(token)
                continue
            prefix, value = match.groups()
        for old, new in changes:
            if value == old or value.startswith(old + '/'):
                replacement = prefix + new + value[len(old):]
                break
        out.append(replacement)
        if replacement != token:
            ledger.append({'index': index, 'before': token, 'after': replacement, 'reason': 'registered dependency include-root replacement'})
    check(not pending, 'missing include path operand')
    if prepend:
        includes = [prepend] if isinstance(prepend, str) else prepend
        inserted = [token for folder in includes for token in ('-I', folder)]
        # A compiler wrapper must remain adjacent to its original compiler.
        index = 2 if Path(out[0]).name in ('ccache', 'sccache') else 1
        out[index:index] = inserted
        ledger.append({'index': index, 'before': [], 'after': inserted, 'reason': 'registered dependency override before existing include paths'})
    return out, ledger


def capture(command, directory, output, name, timeout):
    record = {'command': command, 'directory': str(directory)}
    started = datetime.now(timezone.utc)
    try:
        process = subprocess.Popen(command, cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, errors='replace', start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            record.update(returncode=None, stdout=stdout, stderr=stderr, status='timeout')
        else:
            record.update(returncode=process.returncode, stdout=stdout, stderr=stderr,
                      status='completed' if process.returncode >= 0 else 'signal/crash')
    except OSError as exc:
        record.update(returncode=None, stdout='', stderr=str(exc), status='unavailable')
    record['seconds'] = (datetime.now(timezone.utc) - started).total_seconds()
    for field in ('stdout', 'stderr'):
        path = output / (name + '.' + field + '.txt')
        with path.open('x') as stream:
            stream.write(record[field])
        record[field + 'File'] = str(path)
    return record


def dependency_paths(text, directory):
    # Compiler -M output, with continuations and Make's escaped spaces retained.
    text = text.replace('\\\n', ' ')
    match = re.search(r'(?<!\\):', text)
    check(match is not None, 'dependency output has no Make target')
    paths = shlex.split(text[match.end():], comments=False)
    check(paths, 'dependency output is empty')
    return {str((Path(directory) / path).resolve(strict=True)) for path in paths}


def syntax_status(syntax):
    status = {'passed': 'passed', 'failed': 'syntax-error'}.get(syntax['status'], syntax['status'])
    code = syntax.get('returncode')
    if code is not None and code < 0:
        syntax['signal'] = -code
        return 'signal/crash'
    # GNU time returns 128+signal, while a normal program can also exit with
    # that value. Use its explicit retained signal diagnostic, never the number.
    timing = Path(syntax['log']).with_suffix('.rss') if syntax.get('log') else None
    if code and timing and timing.is_file():
        match = re.search(r'^Command terminated by signal ([0-9]+)$', timing.read_text(), re.MULTILINE)
        if match:
            syntax.update(signal=int(match.group(1)), signalEvidence={'file': str(timing), 'sha256': digest(timing), 'message': match.group(0)})
            return 'signal/crash'
    return status


def cccl_parity(reference, translated, cccl, before):
    root = Path(cccl['root']).resolve()
    include_roots = [Path(path).resolve() for path in cccl.get('includeRoots', [str(root)])]
    old = [Path(path).resolve() for path in cccl['oldRoots']]
    def classify(paths):
        selected, foreign = {}, []
        for name in paths:
            path = Path(name)
            pieces = path.parts
            looks_like = any(part in ('cub', 'thrust', '__cccl') for part in pieces) or '/cuda/std/' in name or '/nv/' in name
            if any(path.is_relative_to(folder) for folder in include_roots):
                selected[str(path.relative_to(root))] = digest(path)
            elif path.is_relative_to(root) or any(path.is_relative_to(folder) for folder in old) or looks_like:
                foreign.append(name)
        return selected, foreign
    left, foreign_left = classify(reference)
    right, foreign_right = classify(translated)
    covered = all(before.get(str(root / name)) == value for group in (left, right) for name, value in group.items())
    return {'verified': not foreign_left and not foreign_right and covered, 'applicable': bool(left or right or foreign_left or foreign_right),
            'referenceFiles': left, 'clangFiles': right, 'referenceForeign': foreign_left, 'clangForeign': foreign_right,
            'sameAccessedSetRequired': False, 'allAccessedCandidateBytesMatchBefore': covered}


def prepare_include_after(manifest, tools, roots, baseline, paths, entry_count):
    """Validate the narrow GCC discovery mapping before changing any command."""
    configured = manifest.get('clangIncludeAfter', [])
    check(isinstance(configured, list), 'clangIncludeAfter must be a list')
    check(not configured or manifest['schemaVersion'] == 2, 'clangIncludeAfter requires manifest schemaVersion 2')
    mappings = []
    for index, item in enumerate(configured):
        check(isinstance(item, dict) and set(item) <= {'directory', 'compilerRole', 'discoveryCommand', 'reason',
              'evidenceFiles', 'originalIndices'}, 'unsupported clangIncludeAfter configuration')
        check(isinstance(item.get('reason'), str) and item['reason'].strip(), 'host-header discovery reason required')
        role = item.get('compilerRole')
        check(role in tools and role != 'clang', 'host-header discovery requires a registered host compiler role')
        directory = Path(item['directory'])
        check(directory.is_absolute(), 'host-header discovery directory must be absolute')
        directory = directory.resolve(strict=True)
        check(directory.is_dir() and any(directory.is_relative_to(root) for root in roots),
              'host-header discovery directory must be covered by inputRoots')
        files = header_files(directory)
        check(all(baseline.get(str(path)) == digest(path) for path in files),
              'host-header discovery directory files must match the baseline manifest')
        command = item.get('discoveryCommand')
        check(isinstance(command, list) and len(command) == 2 and command[1] == '-print-file-name=include' and
              isinstance(command[0], str) and Path(command[0]).is_absolute() and
              Path(command[0]).resolve(strict=True) == tools[role],
              'discoveryCommand must query the registered actual host compiler with -print-file-name=include')
        indices = item.get('originalIndices')
        check(indices is None or (isinstance(indices, list) and indices and len(indices) == len(set(indices)) and
              all(type(value) is int and 0 <= value < entry_count for value in indices)),
              'invalid host-header discovery originalIndices')
        evidence = item.get('evidenceFiles', [])
        check(isinstance(evidence, list), 'host-header discovery evidenceFiles must be a list')
        bound_evidence = []
        for artifact in evidence:
            lexical_path = Path(os.path.abspath(artifact['path']))
            path = lexical_path.resolve(strict=True)
            check(digest(path) == artifact['sha256'], 'host-header discovery supporting evidence changed')
            paths.add(path)
            bound_evidence.append({**artifact, 'absolutePath': str(lexical_path), 'resolvedPath': str(path)})
        paths.update(files)
        mappings.append({'mappingIndex': index, 'directory': str(directory), 'compilerRole': role,
                         'compilerResolvedPath': str(tools[role]), 'discoveryCommand': command,
                         'reason': item['reason'], 'originalIndices': indices, 'evidenceFiles': bound_evidence,
                         'baselineFileCount': len(files)})
    return mappings


def probe(manifest_path, expected_sha256, output):
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    check(not any(output.is_relative_to(ROOT / name) for name in
                  ('corpus', 'results/holdout-first', 'releases/frozen-source-v1')),
          'probe output may not enter frozen sources or first-run evidence')
    # Reject output inside immutable input trees before creating any directory.
    try:
        preliminary = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        preliminary = {}
    if isinstance(preliminary, dict):
        protected = preliminary.get('inputRoots', [])
        if isinstance(preliminary.get('cccl'), dict) and preliminary['cccl'].get('root'):
            protected = [*protected, preliminary['cccl']['root']]
        check(not any(output.is_relative_to(Path(folder).resolve()) for folder in protected),
              'output must be outside hashed input roots')
    output.mkdir(parents=True, exist_ok=False)
    record = {'schemaVersion': 2, 'protocolId': 'compatibility-parse-v2', 'kind': 'syntax-only-compatibility-probe', 'analysisExecuted': False,
              'startedAt': datetime.now(timezone.utc).isoformat(), 'manifest': str(manifest_path),
              'manifestSha256': expected_sha256, 'records': [], 'failures': [], 'runComplete': False}
    write(output / 'started.json', record)
    before, roots, tools, driver_paths = None, [], {}, {}
    path_identities_before, identity_roots = None, []
    supporting_paths = {}
    baseline_path_identities, baseline_identity_roots = {}, []
    try:
        check(digest(manifest_path) == expected_sha256, 'manifest hash differs from the preregistered pin')
        manifest = json.loads(manifest_path.read_text())
        check(manifest.get('schemaVersion') in (1, 2), 'unsupported manifest schema')
        check(manifest.get('scope') in ('diagnostic-subset', 'all-configured'), 'scope must be explicit')
        check(manifest.get('target') in ('faiss', 'cuda-samples', 'rmm'), 'target must be explicit')
        record.update(candidateId=manifest['candidateId'], target=manifest['target'], scope=manifest['scope'])
        database = Path(manifest['compdb']['path']).resolve()
        raw = database.read_bytes()
        check(hashlib.sha256(raw).hexdigest() == manifest['compdb']['sha256'], 'original compilation database changed')
        entries = json.loads(raw)
        check(isinstance(entries, list) and entries, 'empty original compilation database')
        (output / 'original-compile_commands.json').write_bytes(raw)
        selected = manifest.get('indices', list(range(len(entries))))
        check(isinstance(selected, list) and selected and all(type(i) is int and 0 <= i < len(entries) for i in selected)
              and len(selected) == len(set(selected)), 'invalid original command indices')
        if manifest['scope'] == 'all-configured':
            check(selected == list(range(len(entries))), 'all-configured scope may not omit or reorder original commands')
        check(set(manifest.get('macroIndices', [])) <= set(selected), 'macro indices must refer to selected original entries')
        record.update(originalEntries=len(entries), selectedIndices=selected, compdbSha256=manifest['compdb']['sha256'])
        # The manifest registers actual executable bytes, not merely version names.
        for role, identity in manifest['tools'].items():
            check(re.fullmatch(r'[a-z][a-z0-9_-]*', role), 'invalid tool role name')
            executable = Path(identity['path']).resolve(strict=True)
            check(digest(executable) == identity['sha256'], 'tool hash differs: ' + role)
            tools[role] = executable
        check('clang' in tools, 'Clang identity missing')
        roots = [Path(path).resolve(strict=True) for path in manifest['inputRoots']]
        identity_roots = manifest['inputRoots']
        check(roots, 'input/header roots must be declared')
        check(not any(output.is_relative_to(folder) for folder in roots), 'output must be outside hashed input roots')
        protocol_sources = [Path(__file__).resolve(), ROOT / 'scripts/baseline.py', ROOT / 'scripts/replay_holdout.py',
                            ROOT / 'translate/compdb.py', ROOT / 'translate/macros.py', ROOT / 'translate/__init__.py']
        record['protocolSourceSha256'] = {str(path): digest(path) for path in protocol_sources}
        record['environment'] = {'pythonExecutable': str(Path(sys.executable).resolve()), 'pythonVersion': sys.version,
                                 'platform': platform.platform(), 'imageId': os.environ.get('CUDA_LIFETIME_IMAGE_ID'),
                                 'compilerVariables': {name: os.environ[name] for name in ('PATH', 'LD_LIBRARY_PATH', 'CUDA_HOME', 'CUDA_PATH', 'CPATH', 'CPLUS_INCLUDE_PATH', 'C_INCLUDE_PATH', 'GCC_EXEC_PREFIX', 'COMPILER_PATH', 'LIBRARY_PATH') if name in os.environ}}
        paths = {manifest_path, database, Path(sys.executable).resolve(), *tools.values(), *protocol_sources}
        paths.update(Path(path).resolve(strict=True) for path in manifest.get('inputFiles', []))
        baseline_pin = manifest.get('baselineInputManifest')
        baseline_inputs = {}
        if baseline_pin:
            baseline_path = Path(baseline_pin['path']).resolve(strict=True)
            check(digest(baseline_path) == baseline_pin['sha256'], 'baseline input manifest changed')
            baseline = json.loads(baseline_path.read_text())
            check(baseline.get('schemaVersion') == 1 and isinstance(baseline.get('files'), dict) and baseline['files'],
                  'baseline manifest must contain a nonempty files mapping')
            baseline_inputs = baseline['files']
            proposed_identities = baseline.get('inputPathIdentities', {})
            proposed_roots = baseline.get('inputRoots', [])
            check(isinstance(proposed_identities, dict) and all(isinstance(name, str) and Path(name).is_absolute() and
                  isinstance(target, str) and Path(target).is_absolute() for name, target in proposed_identities.items()),
                  'invalid baseline input path identities')
            check(isinstance(proposed_roots, list) and all(isinstance(name, str) and Path(name).is_absolute()
                  for name in proposed_roots), 'invalid baseline inputRoots')
            baseline_path_identities, baseline_identity_roots = proposed_identities, proposed_roots
            check(all(Path(name).is_absolute() and re.fullmatch('[0-9a-f]{64}', value)
                      for name, value in baseline_inputs.items()), 'invalid baseline file identity')
            paths.add(baseline_path)
            paths.update(Path(name).resolve(strict=True) for name in baseline_inputs)
        for folder in roots:
            paths.update(header_files(folder))
        include_after = prepare_include_after(manifest, tools, roots, baseline_inputs, paths, len(entries))
        record['clangIncludeAfter'] = include_after
        supporting_paths = {item['absolutePath']: item['resolvedPath'] for mapping in include_after for item in mapping['evidenceFiles']}
        cccl = manifest.get('cccl')
        changes, prepend = [], None
        if cccl:
            check(cccl.get('source') and cccl.get('reason'), 'CCCL provenance and override reason required')
            check(tree_hash(cccl['root']) == cccl['treeSha256'], 'CCCL candidate tree differs from its registered hash')
            source_root = Path(cccl['root']).resolve()
            include_roots = [Path(path).resolve(strict=True) for path in cccl.get('includeRoots', [str(source_root)])]
            check(include_roots and len(set(include_roots)) == len(include_roots) and
                  all(path.is_dir() and path.is_relative_to(source_root) for path in include_roots),
                  'CCCL include roots must be distinct directories within its source tree')
            # A source archive may have several include roots, unlike a flat SDK
            # tree. In that case insertion is explicit; do not invent a mapping.
            replacements = cccl.get('includeReplacements', [] if 'includeRoots' in cccl else
                                    [{'old': old, 'new': str(source_root)} for old in cccl['oldRoots']])
            changes = [(str(Path(item['old']).resolve()), str(Path(item['new']).resolve())) for item in replacements]
            check(all(Path(new) in include_roots and Path(old) in [Path(p).resolve() for p in cccl['oldRoots']]
                      for old, new in changes), 'CCCL replacement must name registered old and new include roots')
            if cccl.get('prependInclude') is True:
                prepend = list(map(str, include_roots))
        prepared = []
        required_baseline = {database, *(Path(path).resolve() for path in manifest.get('inputFiles', []))}
        def registered_driver(token, directory):
            driver = Path(os.path.abspath(directory / token)) if '/' in token else Path(shutil.which(token) or directory / token).absolute()
            actual = driver.resolve(strict=True)
            check(actual.is_file() and actual in tools.values(), 'compiler must have a registered actual identity: ' + str(actual))
            driver_paths[str(driver)] = str(actual)
            return {'path': str(driver), 'resolvedPath': str(actual), 'verified': True}
        for index in selected:
            original = entries[index]
            row = {'originalIndex': index, 'originalEntrySha256': json_hash(original), 'originalEntry': original,
                   'file': original.get('file') if isinstance(original, dict) else None,
                   'isCu': isinstance(original, dict) and isinstance(original.get('file'), str) and Path(original['file']).suffix == '.cu',
                   'translation': {}, 'sourceAvailable': False}
            directory, translated = database.parent, None
            try:
                directory = (database.parent / original['directory']).resolve()
                source = (directory / original['file']).resolve()
                row['file'] = str(source)
                required_baseline.add(source)
                check(source.is_file(), 'source file unavailable: ' + str(source))
                paths.add(source)
                row['sourceAvailable'] = True
                argv = _argv(original)
                row['originalArguments'] = argv
                expanded, responses = expand_response_files(argv, directory)
                row.update(expandedArguments=expanded, responseFiles=responses)
                for response in responses:
                    response_path = Path(response['path']).resolve()
                    paths.add(response_path)
                    required_baseline.add(response_path)
                reference, ledger = rewrite_paths(expanded, changes, prepend)
                row.update(referenceArguments=reference, pathChanges=ledger)
                wrapper = 1 if Path(reference[0]).name in ('ccache', 'sccache') else 0
                token = reference[wrapper]
                row['compilerIdentity'] = registered_driver(token, directory)
                row['hostCompilerIdentity'] = {**row['compilerIdentity'], 'reason': 'original driver is the host compiler'}
                if Path(token).name == 'nvcc':
                    host = None
                    for position, argument in enumerate(reference):
                        key, equal, value = argument.partition('=')
                        if key in ('-ccbin', '--compiler-bindir'):
                            host = value if equal else reference[position + 1]
                    row['hostCompilerIdentity'] = {'verified': False, 'reason': 'implicit or directory host compiler selection has no registered actual-driver evidence'}
                    if host and not (directory / host).is_dir():
                        row['hostCompilerIdentity'] = registered_driver(host, directory)
                # Preserve original spelling, since a resolved symlink can have a
                # target-prefixed name the existing translator does not accept.
                candidate_entry = {'directory': str(directory), 'file': original['file'], 'arguments': reference}
                translated, translation = translate_entry(candidate_entry, clang=str(tools['clang']), cuda_path=manifest.get('cudaPath'),
                                                            gcc_toolchain=manifest.get('gccToolchain'), base_directory=database.parent)
                if translated:
                    translated['arguments'], extra = rewrite_paths(translated['arguments'], changes)
                    ledger.extend({**item, 'phase': 'translated'} for item in extra)
                    row['hostHeaderDiscovery'] = []
                    for mapping in include_after:
                        selected_mapping = mapping['originalIndices'] is None or index in mapping['originalIndices']
                        host_matches = row['hostCompilerIdentity'].get('verified') and row['hostCompilerIdentity'].get('resolvedPath') == mapping['compilerResolvedPath']
                        # For an ordinary host command the original driver is
                        # the actual host compiler; CUDA commands use -ccbin.
                        if Path(token).name != 'nvcc':
                            host_matches = row['compilerIdentity']['resolvedPath'] == mapping['compilerResolvedPath']
                        if mapping['originalIndices'] is not None and selected_mapping:
                            check(host_matches, 'host-header mapping index does not match its original host compiler')
                        applied = bool(selected_mapping and host_matches)
                        row['hostHeaderDiscovery'].append({'mappingIndex': mapping['mappingIndex'], 'applied': applied,
                            'originalHostCompiler': row['hostCompilerIdentity'], 'reason': mapping['reason'] if applied else 'outside registered host-driver/profile scope'})
                        if applied:
                            insertion = ['-idirafter', mapping['directory']]
                            ledger.append({'phase': 'clang-host-header-discovery', 'mappingIndex': mapping['mappingIndex'],
                                           'index': len(translated['arguments']), 'before': [], 'after': insertion,
                                           'reason': mapping['reason']})
                            translated['arguments'].extend(insertion)
                row['translation'] = translation
            except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
                translated = None
                row.update(status='translation-error', error={'type': type(error).__name__, 'message': str(error)})
                row['translation'] = {'status': 'translation-error', 'unknown': [{'reason': str(error)}]}
            prepared.append((row, translated, directory))
        record['toolIdentities'] = {}
        record['referenceDriverPaths'] = driver_paths
        for role, executable in tools.items():
            version = capture([str(executable), '--version'], ROOT, output, 'version-' + role, 30)
            check(version['returncode'] == 0, 'tool version command failed: ' + role)
            ldd = capture(['ldd', str(executable)], ROOT, output, 'ldd-' + role, 30)
            check(ldd['returncode'] == 0 and 'not found' not in ldd['stdout'], 'unresolved tool shared libraries: ' + role)
            paths.update(Path(name).resolve(strict=True) for name in re.findall(r'(?:=>\s+)?(/[^\s()]+)', ldd['stdout']))
            check(digest(executable) == manifest['tools'][role]['sha256'], 'tool changed during identity collection: ' + role)
            record['toolIdentities'][role] = {'path': str(executable), 'sha256': digest(executable), 'version': version, 'ldd': ldd}
        for mapping in include_after:
            command = mapping['discoveryCommand']
            driver_paths[str(Path(command[0]).absolute())] = mapping['compilerResolvedPath']
            discovery = capture(command, ROOT, output, 'host-header-discovery-' + str(mapping['mappingIndex']), 30)
            mapping['discoveryEvidence'] = discovery
            check(discovery['returncode'] == 0 and Path(discovery['stdout'].strip()).is_absolute() and
                  str(Path(discovery['stdout'].strip()).resolve(strict=True)) == mapping['directory'],
                  'actual host compiler discovery does not match the registered directory')
            check(Path(command[0]).resolve(strict=True) == tools[mapping['compilerRole']] and
                  digest(tools[mapping['compilerRole']]) == manifest['tools'][mapping['compilerRole']]['sha256'],
                  'host compiler identity changed during header discovery')
        resource = capture([str(tools['clang']), '-print-resource-dir'], ROOT, output, 'clang-resource-dir', 30)
        check(resource['returncode'] == 0 and Path(resource['stdout'].strip()).is_dir(), 'actual Clang resource directory unavailable')
        record['toolIdentities']['clang']['resourceDirectory'] = resource
        before = {str(path): digest(path) for path in sorted(paths)}
        # Queries run after initial validation. Recheck registered pins so an
        # input changed during discovery cannot become a new accepted baseline.
        registered_pins = {str(manifest_path): expected_sha256, str(database): manifest['compdb']['sha256']}
        registered_pins.update({str(tools[role]): identity['sha256'] for role, identity in manifest['tools'].items()})
        if baseline_pin:
            registered_pins[str(baseline_path)] = baseline_pin['sha256']
        for mapping in include_after:
            registered_pins.update({item['resolvedPath']: item['sha256'] for item in mapping['evidenceFiles']})
        record['registeredInputPinMismatches'] = [name for name, value in registered_pins.items() if before.get(name) != value]
        check(not record['registeredInputPinMismatches'], 'registered input pin changed during preparation/discovery')
        check(all(str(Path(name).resolve()) == value for name, value in supporting_paths.items()),
              'supporting evidence path identity changed during discovery')
        path_identities_before = input_path_identities(identity_roots)
        record['inputPathIdentitiesBefore'] = path_identities_before
        record['baselinePathIdentityMismatches'] = baseline_path_mismatches(baseline_path_identities, baseline_identity_roots)
        check(not record['baselinePathIdentityMismatches'], 'baseline path identity changed before candidate execution')
        candidate_root = Path(cccl['root']).resolve() if cccl else None
        original_roots = [root for root in roots if not (candidate_root and root.is_relative_to(candidate_root))]
        record['baselinePathIdentitiesVerified'] = bool(baseline_path_identities and baseline_identity_roots) and all(
            any(root.is_relative_to(Path(folder).resolve()) for folder in baseline_identity_roots) for root in original_roots)
        record['baselineChangedInputs'] = [name for name, value in baseline_inputs.items()
                                          if before.get(str(Path(name).resolve())) != value]
        check(not record['baselineChangedInputs'], 'baseline input bytes changed before candidate execution')
        record['baselineUnboundInputs'] = sorted(str(path) for path in required_baseline if str(path) not in baseline_inputs)
        record['baselineBindingVerified'] = bool(baseline_inputs) and not record['baselineUnboundInputs'] and record['baselinePathIdentitiesVerified']
        record['inputSha256Before'] = before
        write(output / 'inputs-before.json', before)
        syntax_dir = output / 'syntax'
        syntax_dir.mkdir()
        timeout = manifest.get('syntaxTimeoutSeconds', 180)
        macro_timeout = manifest.get('macroTimeoutSeconds', 120)
        jobs = manifest.get('jobs', 4)
        check(min(timeout, macro_timeout, jobs) > 0, 'budgets must be positive')
        record['budgets'] = {'syntaxTimeoutSeconds': timeout, 'macroTimeoutSeconds': macro_timeout, 'jobs': jobs}
        args = SimpleNamespace(out=syntax_dir, mode='syntax', clang=str(tools['clang']), timeout=timeout)
        def run(item):
            row, translated, directory = item
            if translated is None:
                row.setdefault('status', 'translation-unknown')
                return row
            row['translatedArguments'] = translated['arguments']
            syntax = run_one(row['originalIndex'], translated, args, json_hash(translated))
            row['syntax'] = syntax
            row['status'] = syntax_status(syntax)
            reference = row['referenceArguments']
            is_nvcc = any(Path(token).name == 'nvcc' for token in reference[:2])
            dependency_sets = []
            row['dependencyEvidence'] = []
            for label, argv, nvcc in [('reference', reference, is_nvcc), ('clang', translated['arguments'], False)]:
                command = _preprocess_command(argv, nvcc=nvcc)[:-2] + ['-M']
                evidence = capture(command, directory, output, f"{row['originalIndex']:04d}-{label}-dependencies", macro_timeout)
                try:
                    check(evidence['returncode'] == 0, 'dependency collection failed')
                    dependencies = dependency_paths(evidence['stdout'], directory)
                    evidence['files'] = {name: digest(name) for name in sorted(dependencies)}
                    evidence['allFilesInBeforeManifest'] = all(before.get(name) == value for name, value in evidence['files'].items())
                    evidence['unboundBaselineFiles'] = [name for name, value in evidence['files'].items()
                        if baseline_inputs.get(name) != value and not (cccl and Path(name).is_relative_to(Path(cccl['root']).resolve()))]
                    evidence['allFilesBoundToBaselineOrCandidate'] = not evidence['unboundBaselineFiles']
                    dependency_sets.append(dependencies)
                except (OSError, ValueError) as error:
                    evidence.update(error=str(error), allFilesInBeforeManifest=False, allFilesBoundToBaselineOrCandidate=False)
                    dependency_sets.append(set())
                row['dependencyEvidence'].append(evidence)
            row['inputDependenciesCovered'] = all(evidence['allFilesInBeforeManifest'] for evidence in row['dependencyEvidence'])
            row['inputDependenciesBoundToBaseline'] = all(evidence['allFilesBoundToBaselineOrCandidate'] for evidence in row['dependencyEvidence'])
            if cccl:
                row['dependencyHeaderParity'] = cccl_parity(*dependency_sets, cccl, before)
                row['dependencyHeaderParity']['verified'] &= row['inputDependenciesCovered']
            if row['originalIndex'] in manifest.get('macroIndices', []):
                evidence = [capture(_preprocess_command(argv, nvcc=nvcc), directory, output,
                                    f"{row['originalIndex']:04d}-{label}-macros", macro_timeout)
                            for label, argv, nvcc in [('reference', reference, is_nvcc), ('clang', translated['arguments'], False)]]
                macros = [parse_macros(item['stdout']) for item in evidence]
                available = all(item['returncode'] == 0 for item in evidence) and all(macros)
                comparable = not is_nvcc and available
                required = set(_project_macros(reference)) | set(_project_macros(translated['arguments']))
                if row['translation'].get('language') == 'cuda':
                    required.update(('__CUDACC__', 'CUDA_API_PER_THREAD_DEFAULT_STREAM'))
                equal = compare_macros(*macros, required)['equal']
                macro_status = 'unavailable' if not available else 'project-only' if is_nvcc else 'verified' if equal else 'mismatch'
                row['macroEvidence'] = {'commands': evidence, 'sets': macros, 'status': macro_status,
                                        'hostMacroParityVerified': comparable and compare_macros(*macros, required)['equal'],
                                        'requiredDifferences': compare_macros(*macros, required),
                                        'projectOptions': compare_macros(_project_macros(reference), _project_macros(translated['arguments'])),
                                        'nvccHostPassUnverified': is_nvcc}
            return row
        def guarded_run(item):
            try:
                return run(item)
            except Exception as error:
                row = item[0]
                row.update(status='probe-error', error={'type': type(error).__name__, 'message': str(error)})
                return row
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            record['records'] = list(pool.map(guarded_run, prepared))
        for name, value in before.items():
            check(digest(name) == value, 'input changed during probe: ' + name)
        check(all(header_files(folder) <= {Path(name) for name in before} for folder in roots), 'new file appeared in an input root')
        check(input_path_identities(identity_roots) == path_identities_before, 'input path or symlink identity changed during probe')
        record['inputsUnchanged'] = True
        if cccl:
            check(tree_hash(cccl['root']) == cccl['treeSha256'], 'CCCL tree changed during probe')
        full = manifest['scope'] == 'all-configured'
        all_count = len(entries)
        cu_count = sum(isinstance(entry, dict) and isinstance(entry.get('file'), str) and Path(entry['file']).suffix == '.cu' for entry in entries)
        passed = sum(row['status'] == 'passed' for row in record['records'])
        cu_passed = sum(row['status'] == 'passed' and row['isCu'] for row in record['records'])
        record['counts'] = {'allConfigured': all_count, 'cudaCu': cu_count, 'selected': len(selected),
                            'selectedPassed': passed, 'selectedCuPassed': cu_passed,
                            'translationUnknown': sum(row['status'] == 'translation-unknown' for row in record['records']),
                            'translationError': sum(row['status'] == 'translation-error' for row in record['records'])}
        record['rates'] = {'allConfigured': passed / all_count if full else None,
                           'cudaCu': cu_passed / cu_count if full and cu_count else None}
        record['partitions'] = {}
        for group in ('cudaCu', 'host', 'otherCuda'):
            selected_rows = [row for row in record['records'] if ('cudaCu' if row['isCu'] else
                             'otherCuda' if row['translation'].get('language') == 'cuda' else 'host') == group]
            record['partitions'][group] = {'selected': len(selected_rows), 'passed': sum(row['status'] == 'passed' for row in selected_rows),
                                          'denominator': len(selected_rows) if full else None}
        record['cuThresholdPassed'] = 10 * cu_passed > 9 * cu_count if full and cu_count else None
        record['scopeValid'] = not (full and manifest['target'] in ('faiss', 'cuda-samples') and not cu_count)
        record['toolIdentityComplete'] = all(row.get('compilerIdentity', {}).get('verified', False) and row.get('hostCompilerIdentity', {}).get('verified', False) for row in record['records'])
        record['baselineDependencyBindingComplete'] = all(row.get('inputDependenciesBoundToBaseline', False) for row in record['records'])
        record['inputIntegrityComplete'] = record['baselineBindingVerified'] and record['toolIdentityComplete'] and record['baselineDependencyBindingComplete'] and all(row.get('inputDependenciesCovered', False) and
                                               row.get('syntax', {}).get('sourceUnchanged') is True and
                                               row.get('syntax', {}).get('inputDatabaseUnchanged') is True
                                               for row in record['records'])
        record['dependencyHeaderParityVerified'] = all(row.get('dependencyHeaderParity', {}).get('verified', False) for row in record['records']) if cccl else None
        passed_rows = [row for row in record['records'] if row['status'] == 'passed']
        record['successfulSyntaxInputVerified'] = record['baselineBindingVerified'] and record['toolIdentityComplete'] and bool(passed_rows) and all(row.get('inputDependenciesCovered', False) and row.get('inputDependenciesBoundToBaseline', False) and
            row['syntax'].get('sourceUnchanged') is True and row['syntax'].get('inputDatabaseUnchanged') is True for row in passed_rows)
        record['successfulSyntaxHeaderParityVerified'] = (bool(passed_rows) and all(
            row.get('dependencyHeaderParity', {}).get('verified', False) for row in passed_rows)) if cccl else None
        record['translationComplete'] = record['counts']['translationUnknown'] == 0 and record['counts']['translationError'] == 0
        record['hostMacroParityVerified'] = bool(manifest.get('macroIndices')) and all(
            row.get('macroEvidence', {}).get('hostMacroParityVerified', False) for row in record['records']
            if row['originalIndex'] in manifest['macroIndices'])
        record['diagnosticComplete'] = len(record['records']) == len(selected)
        record['commandInputsUnchanged'] = all((row['status'] in ('translation-unknown', 'translation-error') and row['sourceAvailable']) or
            (row.get('syntax', {}).get('sourceUnchanged') is True and row.get('syntax', {}).get('inputDatabaseUnchanged') is True)
            for row in record['records'])
        record['headerIdentityUnknownIndices'] = [row['originalIndex'] for row in record['records']
                                                  if not row.get('inputDependenciesCovered', False)]
        record['runComplete'] = (full and record['diagnosticComplete'] and record['commandInputsUnchanged'] and
                                 record['baselineBindingVerified'] and record['toolIdentityComplete'] and record['scopeValid'] and
                                 all(row['status'] != 'probe-error' for row in record['records']))
    except Exception as error:
        record['failures'].append({'type': type(error).__name__, 'message': str(error)})
    finally:
        baseline_path_drift = bool(baseline_path_mismatches(baseline_path_identities, baseline_identity_roots))
        if baseline_path_drift:
            record['baselinePathIdentityMismatches'] = baseline_path_mismatches(baseline_path_identities, baseline_identity_roots)
            record['baselinePathIdentitiesVerified'] = False
            record['baselineBindingVerified'] = False
            record['runComplete'] = False
        record['supportingEvidencePathsUnchanged'] = all(str(Path(name).resolve()) == value for name, value in supporting_paths.items())
        evidence_path_drift = not record['supportingEvidencePathsUnchanged']
        if evidence_path_drift:
            record['failures'].append({'type': 'InputIdentityChanged', 'message': 'a supporting evidence path resolved to a different file'})
            record['runComplete'] = False
        path_drift = False
        if path_identities_before is not None:
            try:
                record['inputPathIdentitiesAfter'] = input_path_identities(identity_roots)
                path_drift = record['inputPathIdentitiesAfter'] != path_identities_before
            except OSError as error:
                record['inputPathIdentityError'] = str(error)
                path_drift = True
            record['inputPathIdentitiesUnchanged'] = not path_drift
        record['referenceDriverPathsUnchanged'] = all(str(Path(name).resolve()) == value for name, value in driver_paths.items())
        if not record['referenceDriverPathsUnchanged']:
            record['runComplete'] = False
            record['failures'].append({'type': 'ToolIdentityChanged', 'message': 'a reference compiler path resolved to different executable bytes'})
        if before is not None:
            changed = []
            for name, value in before.items():
                try:
                    if digest(name) != value: changed.append(name)
                except OSError:
                    changed.append(name)
            record['changedInputs'] = changed
            driver_drift = not record['referenceDriverPathsUnchanged']
            pin_drift = bool(record.get('baselineChangedInputs') or record.get('registeredInputPinMismatches') or baseline_path_drift)
            record['inputDriftDetected'] = bool(changed or path_drift or driver_drift or pin_drift or evidence_path_drift)
            if changed or path_drift or driver_drift or pin_drift or evidence_path_drift:
                record['inputsUnchanged'] = False
                record['inputIntegrityComplete'] = False
                record['successfulSyntaxInputVerified'] = False
                record['runComplete'] = False
        record['finishedAt'] = datetime.now(timezone.utc).isoformat()
        record['runComplete'] = record['runComplete'] and not record['failures']
        write(output / 'report.json', record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--expected-manifest-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = probe(args.manifest, args.expected_manifest_sha256, args.out)
    except (OSError, ValueError) as error:
        parser.exit(2, str(error) + '\n')
    print(json.dumps({key: report.get(key) for key in ('candidateId', 'scope', 'runComplete', 'cuThresholdPassed', 'failures')}, indent=2))
    return 0 if not report['failures'] and report.get('diagnosticComplete') and report.get('inputsUnchanged') else 2


if __name__ == '__main__':
    raise SystemExit(main())
