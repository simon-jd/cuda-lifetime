"""Explicit, pinned header compatibility profiles (standard library only).

Templates are deliberately small. Resolution discovers the current inputs and
executes only fixed compiler identity/include-directory queries. No dependency
is downloaded, installed, or modified, and no benchmark code is imported.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from .compdb import TranslationError, _argv, expand_response_files

CCCL_COMMIT = '510cc9de1ea656a453d6072ea3e1f6097410db9a'
CCCL_ARCHIVE_SHA256 = '17bc62b70be51107727a2bc5694693fca1fc73d18c182620522a503c5edfe680'
CCCL_TREE_SHA256 = '62acadd639e08afc90ce4edf83515a27c50e9e9008d5643bbf2e0105210e5baa'
CCCL_LINKS = {'.github/copilot-instructions.md': '../AGENTS.md',
              '.devcontainer/cuda13.0-rapids-conda': '../ci/rapids/cuda13.0-conda'}
CCCL_INCLUDES = ('thrust', 'cub', 'libcudacxx/include')
PROFILES = ('cccl-510cc9d-host-headers', 'host-headers')
_COMPILER = re.compile(r'(nvcc|clang\+\+|clang|g\+\+|gcc|c\+\+|cc)(-[0-9.]+)?')
_GCC = re.compile(r'(g\+\+|gcc|c\+\+|cc)(-[0-9.]+)?')


def check(condition, message):
    if not condition:
        raise TranslationError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_digest(value):
    # Same encoding as the recorded original-command/probe tree identities.
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_identity(path):
    lexical = Path(os.path.abspath(path))
    actual = lexical.resolve(strict=True)
    check(actual.is_file(), f'profile input is not a file: {lexical}')
    return {'path': str(lexical), 'resolvedPath': str(actual), 'sha256': digest(actual)}


def tree_identity(root, *, cccl=False):
    lexical = Path(os.path.abspath(root))
    actual = lexical.resolve(strict=True)
    check(actual.is_dir(), f'profile input is not a directory: {lexical}')
    files, links, resolved_links = {}, {}, {}
    for path in sorted(actual.rglob('*')):
        relative = str(path.relative_to(actual))
        if path.is_symlink():
            resolved = path.resolve(strict=True)
            links[relative] = str(path.readlink())
            resolved_links[relative] = str(resolved)
            if cccl:
                check(resolved.is_relative_to(actual), 'CCCL symlink leaves the pinned source tree')
            else:
                check(not resolved.is_dir(), 'directory symlinks in the discovered host include tree are unsupported')
        if path.is_file():
            files[relative] = digest(path)
    check(files, f'empty profile input directory: {actual}')
    if cccl:
        check(json_digest(files) == CCCL_TREE_SHA256, 'CCCL full source tree does not match the approved commit snapshot')
        check(links == CCCL_LINKS, 'CCCL symlink identities do not match the approved snapshot')
    return {'path': str(lexical), 'resolvedPath': str(actual), 'files': files,
            'treeSha256': json_digest(files), 'symlinks': links, 'resolvedSymlinks': resolved_links}


def capture(command, directory, timeout=30):
    evidence = {'command': list(command), 'directory': str(directory)}
    try:
        process = subprocess.run(command, cwd=directory, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True, encoding='utf-8',
                                 errors='replace', timeout=timeout, check=False)
        evidence.update(status='completed', returncode=process.returncode,
                        stdout=process.stdout, stderr=process.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        evidence.update(status='unavailable', returncode=None, stdout='', stderr=str(error))
    return evidence


def compiler_index(arguments):
    index = 0
    while index < len(arguments) and Path(arguments[index]).name in ('ccache', 'sccache'):
        index += 1
    check(index < len(arguments), 'profile command has no compiler')
    check(_COMPILER.fullmatch(Path(arguments[index]).name), 'unsupported profile compiler or command wrapper')
    return index


def driver_path(token, directory):
    directory = Path(directory).resolve()
    search_path = os.pathsep.join(str(directory / part) for part in os.get_exec_path())
    path = directory / token if '/' in token else Path(shutil.which(token, path=search_path) or directory / token)
    return Path(os.path.abspath(path))


def immutable_entry(entry):
    """Bind all translation semantics, permitting only later evidence updates."""
    result = {key: value for key, value in entry.items() if key != 'macroDiff'}
    if isinstance(result.get('compatibility'), dict):
        result['compatibility'] = {key: value for key, value in result['compatibility'].items()
                                   if key != 'headerIdentity'}
    return result


def reference_arguments(arguments, include_roots):
    """Insert the declared three roots after all cache wrappers and the driver."""
    index = compiler_index(arguments) + 1
    added = [item for root in include_roots for item in ('-I', root)]
    result = list(arguments)
    result[index:index] = added
    changes = [] if not added else [{'phase': 'reference', 'index': index, 'before': [], 'after': added,
                                    'reason': 'explicit intact CCCL dependency override for both original and Clang drivers'}]
    return result, changes


class PreparedProfile:
    def __init__(self, template, source, entries, clang, output):
        template = Path(template).resolve(strict=True)
        try:
            self.template = json.loads(template.read_text(encoding='utf-8'))
        except (OSError, ValueError) as error:
            raise TranslationError(f'cannot read compatibility template: {error}') from error
        check(isinstance(self.template, dict), 'compatibility profile must be an object')
        check(set(self.template) <= {'schemaVersion', 'profile', 'ccclRoot'}, 'unknown compatibility template field')
        check(type(self.template.get('schemaVersion')) is int and self.template['schemaVersion'] == 1,
              'unsupported compatibility template schema')
        name = self.template.get('profile')
        check(name in PROFILES, 'unsupported compatibility profile')
        check(json.loads(source.read_text(encoding='utf-8')) == entries, 'compilation database changed while loading')
        self.inputs = {}
        self.tools = {}
        self.hosts = {}
        self.entries = {}
        self.output = Path(output).resolve()
        self.lock = {'schemaVersion': 1, 'kind': 'resolved-compatibility-profile', 'profile': name,
                     'productVersion': '0.3.0', 'template': self.pin(template), 'inputDatabase': self.pin(source),
                     'originalEntryCount': len(entries), 'cccl': None, 'hostHeaders': {},
                     'discoveryAttempts': [], 'driverResolutions': [],
                     'tools': self.tools, 'inputFiles': self.inputs, 'entryBindings': {},
                     'scope': 'current inputs and derived commands; translation is not syntax or macro equivalence proof'}
        if name == 'cccl-510cc9d-host-headers':
            root = self.template.get('ccclRoot')
            check(isinstance(root, str) and root and '\0' not in root, 'ccclRoot is required')
            tree = tree_identity(template.parent / root, cccl=True)
            check(not self.output.is_relative_to(Path(tree['resolvedPath'])), 'profile output may not enter the CCCL tree')
            self.lock['cccl'] = {'sourceCommit': CCCL_COMMIT, 'formalRelease': False,
                'sourceURL': 'https://github.com/NVIDIA/cccl/commit/' + CCCL_COMMIT,
                'archiveURL': 'https://codeload.github.com/NVIDIA/cccl/tar.gz/' + CCCL_COMMIT,
                'archiveSha256': CCCL_ARCHIVE_SHA256, 'archiveVerification': 'expected official archive pin; complete extracted source verified independently',
                'tree': tree, 'includeRoots': [str(Path(tree['resolvedPath']) / p) for p in CCCL_INCLUDES]}
            check(all(Path(p).is_dir() for p in self.lock['cccl']['includeRoots']), 'missing CCCL include root')
        else:
            check('ccclRoot' not in self.template, 'host-headers profile preserves original CCCL; ccclRoot is not permitted')
        self.clang_token = str(Path(clang).resolve()) if '/' in str(clang) else str(clang)
        self.clang = self.register_driver(self.clang_token, Path.cwd())
        for index, entry in enumerate(entries):
            try:
                self.entries[index] = self.prepare_entry(entry, source.parent)
            except (TranslationError, OSError, ValueError, TypeError, KeyError, IndexError) as error:
                self.entries[index] = {'error': str(error)}
        self.verify()

    def pin(self, path):
        identity = file_identity(path)
        previous = self.inputs.get(identity['path'])
        check(previous is None or previous == identity, 'profile input changed while resolving: ' + identity['path'])
        self.inputs[identity['path']] = identity
        return identity

    def register_driver(self, token, directory):
        identity = self.register_resolution(token, directory)
        actual = identity['resolvedPath']
        if actual not in self.tools:
            version = capture([identity['path'], '--version'], directory)
            self.tools[actual] = {**identity, 'version': version}
        check(self.tools[actual]['version']['returncode'] == 0, 'compiler identity query failed: ' + identity['path'])
        return identity

    def register_resolution(self, token, directory):
        identity = self.pin(driver_path(token, directory))
        record = {'token': token, 'directory': str(directory), 'identity': identity}
        if record not in self.lock['driverResolutions']:
            self.lock['driverResolutions'].append(record)
        return identity

    def prepare_entry(self, entry, base):
        check(isinstance(entry, dict), 'entry must be an object')
        check(isinstance(entry.get('directory'), str) and entry['directory'], 'missing command directory')
        check(isinstance(entry.get('file'), str) and entry['file'], 'missing command source')
        directory = (base / entry['directory']).resolve(strict=True)
        check(self.register_resolution(self.clang_token, directory) == self.clang,
              'Clang driver resolves differently in the original command directory; use an absolute --clang path')
        original = _argv(entry)
        expanded, responses = expand_response_files(original, directory)
        source = self.pin(directory / entry['file'])
        for response in responses:
            response['identity'] = self.pin(response['path'])
        index = compiler_index(expanded)
        for wrapper in expanded[:index]:
            self.register_resolution(wrapper, directory)
        compiler = self.register_driver(expanded[index], directory)
        host_token = expanded[index]
        if Path(host_token).name == 'nvcc':
            host_token = None
            for i, token in enumerate(expanded[index + 1:], index + 1):
                key, equal, value = token.partition('=')
                if key in ('-ccbin', '--compiler-bindir'):
                    host_token = value if equal else expanded[i + 1]
            check(host_token, 'profile requires an explicit actual nvcc host compiler (-ccbin)')
        check(_GCC.fullmatch(Path(host_token).name), 'profile requires an explicit GCC C/C++ host driver')
        host = self.register_driver(host_token, directory)
        actual = host['resolvedPath']
        if actual not in self.hosts:
            query = capture([host['path'], '-print-file-name=include'], directory)
            self.lock['discoveryAttempts'].append(query)
            value = query['stdout'].strip()
            check(query['returncode'] == 0 and value and '\n' not in value and '\r' not in value and Path(value).is_absolute(),
                  'actual host compiler include discovery failed or did not return one absolute directory')
            tree = tree_identity(value)
            check(not self.output.is_relative_to(Path(tree['resolvedPath'])), 'profile output may not enter host headers')
            self.hosts[actual] = {'compiler': host, 'discovery': query, 'tree': tree}
            self.lock['hostHeaders'][actual] = self.hosts[actual]
        cccl = self.lock['cccl']
        reference, changes = reference_arguments(expanded, cccl['includeRoots'] if cccl else [])
        return {'directory': str(directory), 'originalArguments': original, 'expandedArguments': expanded,
                'responseFiles': responses, 'source': source, 'compiler': compiler, 'hostCompiler': host,
                'referenceArguments': reference, 'changes': changes,
                'includeAfter': self.hosts[actual]['tree']['resolvedPath']}

    def bind(self, index, original, report):
        fields = ('originalArguments', 'expandedArguments', 'referenceArguments', 'translatedArguments')
        self.lock['entryBindings'][str(index)] = {
            'originalIndex': index, 'originalEntrySha256': json_digest(original),
            'directory': report.get('directory'), 'file': report.get('file'), 'status': report['status'],
            'metadataSha256': json_digest(immutable_entry(report)),
            'commandSha256': {key: json_digest(report[key]) for key in fields if key in report}}

    def verify(self):
        validate_resolved(self.lock)


def validate_resolved(lock):
    try:
        _validate_resolved(lock)
    except TranslationError:
        raise
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise TranslationError(f'invalid or changed resolved compatibility inputs: {error}') from error


def _validate_resolved(lock):
    check(type(lock.get('schemaVersion')) is int and lock['schemaVersion'] == 1 and
          lock.get('kind') == 'resolved-compatibility-profile', 'invalid resolved compatibility lock')
    check(lock.get('profile') in PROFILES, 'unsupported resolved profile')
    for identity in lock['inputFiles'].values():
        check(file_identity(identity['path']) == identity, 'compatibility input/tool changed: ' + identity['path'])
    for record in lock['driverResolutions']:
        check(file_identity(driver_path(record['token'], Path(record['directory']))) == record['identity'],
              'compatibility command driver resolution changed: ' + record['token'])
    cccl = lock.get('cccl')
    if cccl:
        check(cccl['sourceCommit'] == CCCL_COMMIT and cccl['archiveSha256'] == CCCL_ARCHIVE_SHA256,
              'resolved CCCL source identity mismatch')
        check(tree_identity(cccl['tree']['path'], cccl=True) == cccl['tree'], 'CCCL tree/path identity changed')
        check(cccl['includeRoots'] == [str(Path(cccl['tree']['resolvedPath']) / p) for p in CCCL_INCLUDES], 'resolved CCCL include roots changed')
    else:
        check(lock['profile'] == 'host-headers', 'missing resolved CCCL tree')
    for host in lock['hostHeaders'].values():
        check(tree_identity(host['tree']['path']) == host['tree'], 'host header directory changed')


def load_macro_profile(translation, directory):
    try:
        return _load_macro_profile(translation, directory)
    except TranslationError:
        raise
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise TranslationError(f'invalid or changed compatibility report: {error}') from error


def _load_macro_profile(translation, directory):
    profile = translation.get('compatibility')
    if not profile:
        check(translation.get('schemaVersion') != 2 and
              not (directory / 'compatibility-resolved.json').exists() and
              not any('compatibility' in entry or 'referenceArguments' in entry for entry in translation['entries']),
              'profile translation is missing compatibility metadata')
        return None
    check(translation.get('schemaVersion') == 2 and isinstance(profile, dict), 'invalid profile translation schema')
    check(profile.get('resolvedLock') == 'compatibility-resolved.json', 'invalid resolved lock filename')
    path = directory / profile['resolvedLock']
    check(digest(path) == profile.get('resolvedLockSha256'), 'resolved compatibility lock hash mismatch')
    lock = json.loads(path.read_text())
    validate_resolved(lock)
    check(profile.get('profile') == lock['profile'] and profile.get('profileApplied') is True and
          profile.get('syntaxVerified') is False and
          translation.get('source') == lock['inputDatabase']['path'] and
          translation.get('compiler') == lock['settings']['clang'] and
          translation.get('gccToolchain') == lock['settings']['gccToolchain'],
          'profile translation settings binding mismatch')
    for name, expected in lock['outputDatabases'].items():
        check(Path(name).name == name and digest(directory / name) == expected, 'profile output database changed: ' + name)
    check([entry.get('index') for entry in translation['entries']] == list(range(lock['originalEntryCount'])),
          'profile original entry indices changed')
    for entry in translation['entries']:
        binding = lock['entryBindings'].get(str(entry['index']))
        check(binding is not None and binding['status'] == entry['status'] and
              binding['directory'] == entry.get('directory') and binding['file'] == entry.get('file'),
              'translation entry does not match resolved profile')
        check(binding['originalEntrySha256'] == entry.get('originalEntrySha256'), 'original command identity changed')
        for key, expected in binding['commandSha256'].items():
            check(key in entry and json_digest(entry[key]) == expected, 'profile command binding mismatch: ' + key)
        check(json_digest(immutable_entry(entry)) == binding['metadataSha256'],
              'profile translation metadata binding mismatch')
        if entry['status'] == 'translated':
            check('referenceArguments' in entry, 'profile translation is missing derived reference arguments')
    check(len(translation['entries']) == len(lock['entryBindings']), 'profile entry count changed')
    accepted = sum(entry['status'] == 'translated' for entry in translation['entries'])
    check(translation.get('summary') == {'total': lock['originalEntryCount'], 'translated': accepted,
          'translationUnknown': lock['originalEntryCount'] - accepted}, 'profile summary binding mismatch')
    return lock


def guard_macro_outputs(lock, translation, report_path, destination):
    """Reject collisions before a macro probe can overwrite a bound input."""
    report_path, destination = Path(report_path).resolve(), Path(destination).resolve()
    protected = [Path(value['path']).resolve() for value in lock['inputFiles'].values()]
    protected += [report_path.parent / name for name in lock['outputDatabases']]
    protected.append(report_path.parent / 'compatibility-resolved.json')
    roots = [Path(host['tree']['resolvedPath']) for host in lock['hostHeaders'].values()]
    if lock.get('cccl'):
        roots.append(Path(lock['cccl']['tree']['resolvedPath']))
    outputs = [report_path, destination / 'macro-diff.json']
    for entry in translation['entries']:
        if entry['status'] == 'translated':
            for driver in ('reference', 'clang'):
                for kind in ('macros', 'dependencies'):
                    for stream in ('stdout', 'stderr'):
                        outputs.append(destination / f"{entry['index']:04d}-{driver}-{kind}.{stream}.txt")
    resolved = [path.resolve() for path in outputs]
    check(len(set(resolved)) == len(resolved), 'macro evidence output paths collide')
    protected_paths = {path.resolve() for path in protected}
    def inode(path):
        if not path.exists():
            return None
        info = path.stat()
        return info.st_dev, info.st_ino
    protected_nodes = {node for path in protected_paths if (node := inode(path)) is not None}
    output_nodes = set()
    for path in resolved:
        check(not any(path.is_relative_to(root) for root in roots),
              'macro evidence output may not enter profile input/header trees')
        node = inode(path)
        check(path not in protected_paths and node not in protected_nodes,
              'macro evidence output collides with a bound input: ' + str(path))
        if node is not None:
            check(node not in output_nodes, 'macro evidence output paths collide through a hard link')
            output_nodes.add(node)


def translate_with_profile(source, entries, output, template, *, cuda_path, clang, gcc_toolchain):
    from .compdb import translate_entry
    destination = Path(output).resolve()
    check(not destination.exists() or (destination.is_dir() and not any(destination.iterdir())),
          'compatibility translation requires a new or empty output directory')
    prepared = PreparedProfile(template, source, entries, clang, destination)
    prepared.lock['settings'] = {'clang': str(clang), 'cudaPath': str(Path(cuda_path).resolve()) if cuda_path else None,
                                 'gccToolchain': str(Path(gcc_toolchain).resolve()) if gcc_toolchain else None}
    commands, references, reports = [], [], []
    for index, original in enumerate(entries):
        context = prepared.entries[index]
        candidate = original
        if 'error' not in context:
            candidate = {'directory': context['directory'], 'file': original['file'],
                         'arguments': context['referenceArguments']}
            if isinstance(original.get('output'), str):
                candidate['output'] = original['output']
        translated, report = translate_entry(candidate, cuda_path=cuda_path, clang=clang,
                                            base_directory=source.parent, gcc_toolchain=gcc_toolchain)
        report.update(index=index, originalEntrySha256=json_digest(original),
                      compatibility={'profile': prepared.lock['profile'], 'headerIdentity': 'not-run'})
        if 'error' in context:
            report['unknown'].append({'argument': None, 'reason': 'compatibility profile: ' + context['error']})
            report['status'], translated = 'translation-unknown', None
        else:
            for key in ('originalArguments', 'expandedArguments', 'responseFiles', 'referenceArguments'):
                report[key] = context[key]
            report['referenceEntryIndex'] = len(references)
            references.append(candidate)
            report['compatibility'].update(changes=list(context['changes']), mappingInput='referenceArguments',
                                            originalCompiler=context['compiler'], originalHostCompiler=context['hostCompiler'])
            if translated is not None:
                added = ['-idirafter', context['includeAfter']]
                report['compatibility']['changes'].append({'phase': 'clang-host-header-discovery',
                    'index': len(translated['arguments']), 'before': [], 'after': added,
                    'reason': 'after-system headers discovered from the original actual GCC host compiler'})
                translated['arguments'].extend(added)
                report['translatedArguments'] = translated['arguments']
                report['compatibility']['hostHeaderDiscovery'] = prepared.hosts[context['hostCompiler']['resolvedPath']]['discovery']
        if translated is not None:
            report['translatedEntryIndex'] = len(commands)
            commands.append(translated)
        prepared.bind(index, original, report)
        reports.append(report)
    prepared.verify()
    prepared.lock['inputsUnchangedDuringTranslation'] = True
    original_bytes = source.read_bytes()
    check(hashlib.sha256(original_bytes).hexdigest() == prepared.lock['inputDatabase']['sha256'],
          'original database changed before output')
    destination.mkdir(parents=True, exist_ok=True)
    databases = {'compile_commands.json': commands, 'reference-compile_commands.json': references}
    for name, value in databases.items():
        (destination / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (destination / 'original-compile_commands.json').write_bytes(original_bytes)
    prepared.lock['outputDatabases'] = {name: digest(destination / name)
        for name in (*databases, 'original-compile_commands.json')}
    resolved = destination / 'compatibility-resolved.json'
    resolved.write_text(json.dumps(prepared.lock, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    report = {'schemaVersion': 2, 'source': str(source), 'compiler': str(clang),
              'gccToolchain': prepared.lock['settings']['gccToolchain'],
              'compatibility': {'profile': prepared.lock['profile'], 'resolvedLock': resolved.name,
                                'resolvedLockSha256': digest(resolved), 'profileApplied': True,
                                'headerIdentity': 'not-run', 'syntaxVerified': False},
              'summary': {'total': len(entries), 'translated': len(commands),
                          'translationUnknown': len(entries) - len(commands)}, 'entries': reports}
    (destination / 'translation-report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return report


def dependency_paths(text, directory):
    text = text.replace('\\\n', ' ')
    match = re.search(r'(?<!\\):', text)
    check(match is not None, 'dependency output has no Make target')
    # shlex handles Make backslash-escaped spaces for the supported POSIX inputs.
    import shlex
    tokens = shlex.split(text[match.end():], comments=False)
    check(tokens, 'empty dependency output')
    return sorted({str((Path(directory) / p).resolve(strict=True)) for p in tokens})


def header_provenance(lock, reference, translated, directory, timeout, destination, prefix):
    from .macros import _preprocess_command
    evidence = []
    for label, arguments in [('reference', reference), ('clang', translated)]:
        is_nvcc = label == 'reference' and Path(arguments[compiler_index(arguments)]).name == 'nvcc'
        command = _preprocess_command(arguments, nvcc=is_nvcc)[:-2] + ['-M']
        result = capture(command, directory, timeout)
        save_capture(result, destination, prefix + '-' + label + '-dependencies')
        try:
            check(result['returncode'] == 0, 'dependency preprocessor failed')
            paths = dependency_paths(result['stdout'], directory)
            result['files'] = {p: digest(p) for p in paths}
        except (OSError, ValueError) as error:
            result['error'] = str(error)
        evidence.append(result)
    cccl = lock.get('cccl')
    if not cccl:
        return {'status': 'not-applicable' if all('files' in item for item in evidence) else 'unverified',
                'reason': 'host-headers profile preserves the project CCCL selection; dependency collection remains explicit',
                'dependencies': evidence}
    root, include_roots = Path(cccl['tree']['resolvedPath']), [Path(p) for p in cccl['includeRoots']]
    public_names = set()
    for name in cccl['tree']['files']:
        for prefix in CCCL_INCLUDES:
            if name.startswith(prefix + '/'):
                public = Path(name[len(prefix) + 1:]).parts
                if len(public) > 1:
                    public_names.add(public)
    public_lengths = {len(parts) for parts in public_names}
    valid, accessed = all('files' in item for item in evidence), False
    for item in evidence:
        selected, foreign = {}, []
        for name, sha in item.get('files', {}).items():
            path = Path(name)
            looks_like = (any(part in ('cub', 'thrust', '__cccl') for part in path.parts) or
                          '/cuda/std/' in name or '/nv/' in name or
                          any(path.parts[-length:] in public_names for length in public_lengths))
            if any(path.is_relative_to(p) for p in include_roots):
                relative = str(path.relative_to(root))
                selected[relative] = sha
                valid &= cccl['tree']['files'].get(relative) == sha
            elif path.is_relative_to(root) or looks_like or '/include/cccl/' in name:
                foreign.append(name)
        item.update(candidateFiles=selected, foreignCcclFiles=foreign)
        accessed |= bool(selected or foreign)
        valid &= not foreign
    return {'status': ('verified' if accessed else 'not-applicable') if valid else 'unverified',
            'allAccessedCandidateBytesMatchLock': bool(valid), 'sameAccessedSetRequired': False, 'dependencies': evidence}


def save_capture(evidence, destination, prefix):
    for stream in ('stdout', 'stderr'):
        path = Path(destination) / (prefix + '.' + stream + '.txt')
        path.write_text(evidence.get(stream, ''), encoding='utf-8')
        evidence[stream + 'File'] = str(path.resolve())
        evidence[stream + 'Sha256'] = digest(path)
