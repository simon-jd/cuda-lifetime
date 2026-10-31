"""Translate known compilation flags; reject semantic uncertainty per translation unit.

The default path and translate_entry execute no compiler or shell. An explicit
compatibility profile delegates fixed identity/include discovery queries to the
profile module. Relative response paths use the command working directory.
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path


class TranslationError(ValueError):
    """Invalid compilation database or unsupported command syntax."""


def split_nvcc_list(value: str) -> list[str]:
    """Split comma lists without destroying quotes inside a macro definition."""
    parts, part, quote, escaped = [], [], None, False
    for char in value:
        if escaped:
            # A protected comma belongs to the argument; other escapes retain
            # their spelling for the receiving compiler.
            part.extend([char] if char == ',' else ['\\', char])
            escaped = False
        elif char == '\\':
            escaped = True
        elif char in ('"', "'"):
            quote = None if quote == char else char if quote is None else quote
            part.append(char)
        elif char == ',' and quote is None:
            parts.append(''.join(part))
            part = []
        else:
            part.append(char)
    if escaped:
        part.append('\\')
    if quote:
        raise TranslationError('unclosed quote in nvcc option list')
    parts.append(''.join(part))
    if any(not p for p in parts):
        raise TranslationError('empty element in nvcc option list')
    return parts


def expand_response_files(arguments, directory, *, max_depth=32):
    """Expand @file, -optf and --options-file, detecting cycles and read errors.

    Returns (expanded_arguments, records). Filenames in --options-file may be a
    comma-separated list. Files are POSIX-tokenized; Windows commands are outside
    this translator's supported platform.
    """
    records = []
    root = Path(directory).resolve()

    def expand(args, active):
        if len(active) > max_depth:
            raise TranslationError('response file nesting limit exceeded')
        result, i = [], 0
        while i < len(args):
            token = args[i]
            names = None
            if token.startswith('@'):
                names = [token[1:]]
            elif token in ('--options-file', '-optf'):
                i += 1
                if i == len(args):
                    raise TranslationError(f'missing value for {token}')
                names = split_nvcc_list(args[i])
            elif token.startswith(('--options-file=', '-optf=')):
                names = split_nvcc_list(token.split('=', 1)[1])
            if names is None:
                result.append(token)
            else:
                for name in names:
                    if not name:
                        raise TranslationError('empty response file name')
                    path = (root / name).resolve()
                    if path in active:
                        raise TranslationError('response file cycle: ' + ' -> '.join(map(str, (*active, path))))
                    try:
                        content = path.read_text(encoding='utf-8')
                        nested = shlex.split(content, posix=True)
                    except (OSError, UnicodeError, ValueError) as error:
                        raise TranslationError(f'cannot expand response file {path}: {error}') from error
                    records.append({'path': str(path), 'arguments': nested})
                    result.extend(expand(nested, (*active, path)))
            i += 1
        return result

    return expand(list(arguments), ()), records


def _argv(entry):
    if 'arguments' in entry:
        args = entry['arguments']
        if not isinstance(args, list) or not args or any(not isinstance(a, str) or '\0' in a for a in args):
            raise TranslationError('arguments must be a nonempty list of strings without NUL bytes')
        return args.copy()
    command = entry.get('command')
    if not isinstance(command, str) or not command:
        raise TranslationError('entry needs arguments or command')
    try:
        # Operators are detected as shell syntax even when adjacent to words.
        lexer = shlex.shlex(command, posix=True, punctuation_chars=';&|<>')
        lexer.whitespace_split = True
        lexer.commenters = ''
        args = list(lexer)
    except ValueError as error:
        raise TranslationError(f'invalid shell quoting: {error}') from error
    if any(a in (';', '&&', '||', '|', '&', '>', '>>', '<', '<<') for a in args):
        raise TranslationError('shell operators are not compilation arguments')
    if not args or '\0' in command:
        raise TranslationError('empty or invalid command')
    return args


# Only options whose frontend meaning is understood are accepted. A future flag
# must receive a deliberate mapping instead of being covered by a broad -f glob.
_HOST_EXACT = {
    '-c', '-g', '-g0', '-g1', '-g2', '-g3', '-ggdb', '-ggdb3', '-pipe', '-pthread',
    '-fPIC', '-fpic', '-fPIE', '-fpie', '-fno-PIC', '-fno-pic', '-fno-PIE', '-fno-pie',
    '-fno-exceptions', '-fexceptions', '-fno-rtti', '-frtti', '-fno-omit-frame-pointer',
    '-fomit-frame-pointer', '-fno-strict-aliasing', '-fstrict-aliasing', '-ffast-math',
    '-fno-fast-math', '-fwrapv', '-fno-wrapv', '-fshort-wchar', '-fshort-enums',
    '-fms-extensions', '-fopenmp', '-fno-common', '-fcommon', '-fvisibility-inlines-hidden',
    '-fno-elide-constructors', '-fno-builtin', '-fno-inline', '-m32', '-m64', '-nostdinc',
    '-nostdinc++', '-nostdlibinc', '-pedantic', '-pedantic-errors', '-ansi', '-w',
    '-nobuiltininc', '-nocudainc', '-nocudalib', '--cuda-host-only', '-fgpu-rdc',
    '-fno-gpu-rdc',
    '-fsyntax-only', '-MD', '-MMD', '-MP',
    '-mavx', '-mavx2', '-mavx512f', '-mavx512bw', '-mavx512cd', '-mavx512dq',
    '-mavx512vl', '-mavx512vnni', '-mavx512vpopcntdq', '-mavx512bitalg',
    '-mavx512bf16', '-mavx512fp16', '-mbmi2', '-mf16c', '-mfma', '-mpopcnt',
}
_HOST_VALUES = {'-D', '-U', '-I', '-isystem', '-iquote', '-idirafter', '-include',
                '-imacros', '-isysroot', '--sysroot', '-o', '-MF', '-MT', '-MQ',
                '-target', '--target', '-stdlib', '-std', '-B', '-resource-dir'}
_HOST_JOINED = ('--sysroot=', '--target=', '-stdlib=', '-std=', '-fvisibility=',
                '-fconstexpr-depth=', '-fconstexpr-steps=',
                '-ftemplate-depth=', '-march=', '-mcpu=', '-mtune=', '-mabi=', '-mprefer-vector-width=',
                '--gcc-toolchain=', '-resource-dir=', '--cuda-path=',
                '--offload-arch=', '--cuda-gpu-arch=', '-fgpu-default-stream=')
_DROP_VALUES = {
    '-ccbin': 'nvcc host compiler selection is replaced by the selected Clang driver',
    '--compiler-bindir': 'nvcc host compiler selection is replaced by the selected Clang driver',
    '-Xptxas': 'device assembler options do not affect host-only analysis',
    '--ptxas-options': 'device assembler options do not affect host-only analysis',
    '-Xfatbin': 'fatbinary packaging does not affect host-only analysis',
    '--fatbin-options': 'fatbinary packaging does not affect host-only analysis',
    '-Xnvlink': 'device linker options do not affect host-only analysis',
    '--nvlink-options': 'device linker options do not affect host-only analysis',
    '--diag-suppress': 'nvcc diagnostic suppression does not affect host semantics',
    '-diag-suppress': 'nvcc diagnostic suppression does not affect host semantics',
    '--diag-error': 'nvcc diagnostic severity is not used by Clang',
    '--diag-warn': 'nvcc diagnostic severity is not used by Clang',
    '--keep-dir': 'temporary nvcc output directory is unused',
    '-keep-dir': 'temporary nvcc output directory is unused',
    '--threads': 'nvcc parallelism does not affect host semantics',
    '-t': 'nvcc parallelism does not affect host semantics',
    '--cudart': 'runtime linking mode is unused during static analysis',
    '-cudart': 'runtime linking mode is unused during static analysis',
}
_DROP_EXACT = {
    '--forward-unknown-to-host-compiler': 'nvcc forwarding policy; unknown flags are checked explicitly',
    '-forward-unknown-to-host-compiler': 'nvcc forwarding policy; unknown flags are checked explicitly',
    '--allow-unsupported-compiler': 'nvcc compiler version gate is unused',
    '-allow-unsupported-compiler': 'nvcc compiler version gate is unused',
    '--keep': 'temporary nvcc output retention is unused', '-keep': 'temporary nvcc output retention is unused',
    '-lineinfo': 'device line information is unused', '--generate-line-info': 'device line information is unused',
    '-Wno-deprecated-gpu-targets': 'nvcc GPU target deprecation diagnostic is not understood by Clang',
    '-Wno-subobject-linkage': 'GCC-specific linkage warning suppression has no Clang diagnostic equivalent',
}
_ALIASES = {'--define-macro': '-D', '--undefine-macro': '-U', '--include-path': '-I',
            '--system-include': '-isystem', '--pre-include': '-include', '--output-file': '-o',
            '--std': '-std', '--machine': '-m', '-m': '-m', '--optimize': '-O',
            '-O': '-O', '--language': '-x', '--x': '-x'}


def translate_entry(entry, *, cuda_path=None, clang='clang++', base_directory=None, gcc_toolchain=None):
    """Return (translated_entry_or_None, evidence_record) for one database entry."""
    report = {'file': entry.get('file') if isinstance(entry, dict) else None,
              'status': 'translation-unknown', 'mapped': [], 'dropped': [], 'unknown': [],
              'responseFiles': [], 'defaultStream': 'legacy', 'assumptions': [],
              'macroDiff': {'status': 'not-run', 'reason': 'run the macros command to collect evidence'}}
    if not isinstance(entry, dict):
        report['unknown'].append({'argument': None, 'reason': 'entry must be an object'})
        return None, report

    def record(kind, original, reason, mapped=None):
        item = {'argument': original, 'reason': reason}
        if mapped is not None:
            item['to'] = mapped
        report[kind].append(item)

    try:
        clang = str(Path(clang).resolve()) if '/' in str(clang) else str(clang)
        if not isinstance(entry.get('file'), str) or not entry['file'] or '\0' in entry['file']:
            raise TranslationError('entry file must be a nonempty string without NUL bytes')
        if not isinstance(entry.get('directory'), str) or not entry['directory']:
            raise TranslationError('entry directory must be a nonempty string')
        directory = (Path(base_directory or Path.cwd()) / entry['directory']).resolve()
        report['directory'] = str(directory)
        original = _argv(entry)
        args, report['responseFiles'] = expand_response_files(original, directory)
        report['originalArguments'] = original
        report['expandedArguments'] = args.copy()
        while args and Path(args[0]).name in ('ccache', 'sccache'):
            record('dropped', args.pop(0), 'compiler cache wrapper is not needed for analysis')
        if not args:
            raise TranslationError('command has no compiler')
        compiler = args.pop(0)
        if not re.fullmatch(r'(nvcc|clang\+\+|clang|g\+\+|gcc|c\+\+|cc)(-[0-9.]+)?', Path(compiler).name):
            raise TranslationError(f'unsupported compiler or command wrapper: {compiler}')
        nvcc = Path(compiler).name == 'nvcc'
        record('mapped', compiler, 'use the configured Clang driver', [clang])
        result, i, language, source_seen = [clang], 0, None, False
        relaxed_constexpr = False
        cuda = Path(entry['file']).suffix.lower() in ('.cu', '.cuh')

        # Expand -Xcompiler into its exact host argument positions before mapping.
        forwarded = []
        while i < len(args):
            arg = args[i]
            key, equal, val = arg.partition('=')
            if key in ('-Xcompiler', '--compiler-options'):
                if not equal:
                    i += 1
                    if i == len(args):
                        raise TranslationError(f'missing value for {key}')
                    val = args[i]
                host = split_nvcc_list(val)
                forwarded.extend(host)
                record('mapped', arg if equal else [arg, val], 'unwrap nvcc host compiler options', host)
            else:
                forwarded.append(arg)
            i += 1
        args, i = forwarded, 0
        while i < len(args):
            arg = args[i]
            key, equal, val = arg.partition('=')
            consumed = [arg]

            def value(allow_option=False):
                nonlocal i
                if equal:
                    if not val:
                        raise TranslationError(f'empty value for {key}')
                    return val
                i += 1
                if i == len(args) or not args[i] or (args[i].startswith('-') and not allow_option):
                    raise TranslationError(f'missing value for {key}')
                consumed.append(args[i])
                return args[i]

            mapped = None
            reason = 'preserve host frontend option'
            if key in _DROP_VALUES:
                payload = value(allow_option=key.startswith(('-X', '--ptxas', '--fatbin', '--nvlink')))
                record('dropped', consumed, _DROP_VALUES[key])
                if key in ('-ccbin', '--compiler-bindir'):
                    report['originalHostCompiler'] = payload
                    if gcc_toolchain is None:
                        report['assumptions'].append('nvcc host compiler selection ' + payload + ' was replaced by Clang without an explicit GCC toolchain; host standard-library include paths and compiler builtins remain unverified. Pass gcc_toolchain to preserve the intended GCC installation.')
            elif arg in _DROP_EXACT:
                record('dropped', arg, _DROP_EXACT[arg])
            elif key in ('-Xcudafe', '--cudafe-options'):
                # cudafe accepts semantic options as well as diagnostic options;
                # indiscriminate removal would violate the preservation contract.
                payload = value(allow_option=True)
                if payload in ('--display_error_number', '--no_display_error_number'):
                    record('dropped', consumed, 'cudafe diagnostic numbering only; Clang uses its own diagnostic identifiers')
                elif payload == '--promote_warnings':
                    mapped = ['-Werror']
                    reason = 'preserve warning-as-error policy; diagnostic sets differ between cudafe and Clang'
                elif re.fullmatch(r'--(?:diag_suppress|diag_error|diag_warn)(?:=| )[A-Za-z0-9_,]+', payload):
                    record('dropped', consumed, 'cudafe diagnostic setting does not change host semantics')
                else:
                    record('unknown', consumed, 'cudafe frontend option has no verified semantic mapping')
            elif key in ('--default-stream', '-default-stream'):
                mode = value()
                if mode not in ('legacy', 'per-thread'):
                    record('unknown', consumed, 'default stream must be legacy or per-thread')
                else:
                    report['defaultStream'] = mode
                    mapped = ['-Xclang', '-fgpu-default-stream=' + mode]
                    reason = 'forward default stream to cc1; Clang 23 host-only driver otherwise ignores this flag'
            elif key in ('-rdc', '--relocatable-device-code'):
                mode = value()
                if mode not in ('true', 'false'):
                    record('unknown', consumed, 'relocatable device code value must be true or false')
                else:
                    mapped = ['-fgpu-rdc', '-D__CUDACC_RDC__=1'] if mode == 'true' else ['-fno-gpu-rdc', '-U__CUDACC_RDC__']
            elif arg in ('-dc', '--device-c'):
                mapped = ['-c', '-fgpu-rdc', '-D__CUDACC_RDC__=1']
            elif arg == '--compile':
                mapped = ['-c']
            elif arg in ('--expt-relaxed-constexpr', '-expt-relaxed-constexpr'):
                relaxed_constexpr = True
                mapped = ['-D__CUDACC_RELAXED_CONSTEXPR__=1']
                reason = 'Clang enables host-device constexpr by default; retain the nvcc option macro (M0 positive/negative probes)'
            elif arg in ('--extended-lambda', '--expt-extended-lambda', '-extended-lambda', '-expt-extended-lambda'):
                mapped = ['-D__CUDACC_EXTENDED_LAMBDA__=1']
                reason = 'Clang accepts explicit host/device lambda annotations by default; retain the nvcc option macro (M0 positive/negative probes)'
            elif arg in ('-G', '--device-debug'):
                mapped = ['-D__CUDACC_DEBUG__=1']
                reason = 'omit device debugging output but preserve the nvcc host preprocessing macro'
            elif key in ('--generate-code', '-gencode', '--gpu-architecture', '-arch', '--gpu-code', '-code'):
                spec = value()
                # Host-only compilation deliberately omits device code generation.
                # Validate the syntax before recording this omission; architecture
                # lists can affect CUDA builtins and must be visible to macro diff.
                if key in ('--generate-code', '-gencode'):
                    valid = re.fullmatch(r'arch=compute_\d+[a-z]?,code=(?:sm_\d+[a-z]?|compute_\d+[a-z]?|\[(?:(?:sm|compute)_\d+[a-z]?,?)+\])', spec)
                else:
                    valid = re.fullmatch(r'(?:(?:sm|compute)_\d+[a-z]?)(?:,(?:sm|compute)_\d+[a-z]?)*', spec)
                if not valid:
                    record('unknown', consumed, 'unsupported GPU architecture syntax (native/all are not reproducible)')
                else:
                    record('dropped', consumed, 'device architecture selection omitted for host-only analysis; architecture macro differences require review')
            elif key in ('-x', '--language', '--x'):
                language = value()
                if language not in ('cu', 'cuda', 'c++', 'c', 'c++-header', 'c-header'):
                    record('unknown', consumed, 'unsupported source language')
                else:
                    language = 'cuda' if language == 'cu' else language
                    cuda = language == 'cuda'
                    mapped = ['-x', language]
            elif key in _ALIASES:
                alias, payload = _ALIASES[key], value()
                mapped = ([alias + payload] if alias in ('-m', '-O') else
                          [alias + '=' + payload] if alias == '-std' else [alias, payload])
            elif key in _HOST_VALUES:
                payload = value()
                mapped = [key + '=' + payload] if key in ('-std', '-stdlib', '--sysroot', '--target') else [key, payload]
            elif key == '-Xclang':
                payload = value(allow_option=True)
                if payload in ('-fgpu-default-stream=legacy', '-fgpu-default-stream=per-thread'):
                    report['defaultStream'] = payload.split('=', 1)[1]
                    mapped = ['-Xclang', payload]
                elif payload == '-fno-cuda-host-device-constexpr':
                    mapped = ['-Xclang', payload]
                else:
                    record('unknown', consumed, 'unreviewed cc1 option')
            elif arg in _HOST_EXACT or re.fullmatch(r'-O(?:[0-3sgz]|fast)', arg):
                mapped = [arg]
            elif arg.startswith(('-D', '-U', '-I', '-B')) and len(arg) > 2:
                mapped = [arg]
            elif arg.startswith(_HOST_JOINED):
                mapped = [arg]
                if arg.startswith('-fgpu-default-stream='):
                    report['defaultStream'] = arg.split('=', 1)[1]
                    if report['defaultStream'] not in ('legacy', 'per-thread'):
                        mapped = None
                        record('unknown', arg, 'invalid default stream')
                    else:
                        mapped = ['-Xclang', arg]
            elif arg == '-Werror=all-warnings':
                mapped = ['-Werror']
                reason = 'preserve nvcc all-warnings-as-errors policy using the Clang spelling'
            elif arg.startswith('-W') and not arg.startswith(('-Wl,', '-Wa,', '-Wp,')):
                mapped = [arg]
            elif not arg.startswith('-') and (directory / arg).resolve() == (directory / entry['file']).resolve():
                mapped, source_seen = [arg], True
                reason = 'preserve translation unit input'
            else:
                record('unknown', arg, 'no reviewed mapping for this argument')
            if mapped is not None:
                result.extend(mapped)
                record('mapped', consumed, reason, mapped)
            i += 1
        if not source_seen:
            record('unknown', entry['file'], 'command does not identify its compilation database source file')
        if cuda:
            if language is None:
                result[1:1] = ['-x', 'cuda']
            if '--cuda-host-only' not in result:
                result.append('--cuda-host-only')
            if cuda_path is not None:
                toolkit = Path(cuda_path).resolve()
                result.append('--cuda-path=' + str(toolkit))
                cccl = toolkit / 'include/cccl'
                if nvcc and cccl.is_dir():
                    result.extend(['-isystem', str(cccl)])
                    record('mapped', '<nvcc implicit CUDA include>', 'CUDA 13 nvcc adds toolkit include/cccl as a system search path; preserve it after project include paths', ['-isystem', str(cccl)])
            if nvcc:
                if not relaxed_constexpr:
                    report['assumptions'].append('Clang allows implicit host/device constexpr by default; disabling it breaks Clang CUDA runtime headers. This frontend acceptance difference is not language equivalence evidence.')
                # Clang supplies __CUDACC__ natively in CUDA mode; do not pretend
                # to be nvcc by defining __NVCC__ or its version macros.
                report['macroDiff']['required'] = ['__CUDACC__', 'CUDA_API_PER_THREAD_DEFAULT_STREAM', 'project -D/-U macros']
        if not cuda and language is None and Path(entry['file']).suffix == '.c':
            language = 'c++' if '++' in Path(compiler).name else 'c'
            result[1:1] = ['-x', language]
        if gcc_toolchain is not None:
            toolchain_arg = '--gcc-toolchain=' + str(Path(gcc_toolchain).resolve())
            result.append(toolchain_arg)
            report['gccToolchain'] = str(Path(gcc_toolchain).resolve())
            record('mapped', '<explicit GCC toolchain>', 'preserve the selected host GCC standard-library and support-header search paths', [toolchain_arg])
        report['language'] = 'cuda' if cuda else language or 'c++'
        report['translatedArguments'] = result
        if report['unknown']:
            return None, report
        report['status'] = 'translated'
        translated = {'directory': str(directory), 'file': entry['file'], 'arguments': result}
        if isinstance(entry.get('output'), str):
            translated['output'] = entry['output']
        return translated, report
    except (TranslationError, OSError, ValueError) as error:
        record('unknown', None, str(error))
        return None, report


def load_compdb(compdb):
    path = Path(compdb)
    if path.is_dir():
        path /= 'compile_commands.json'
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TranslationError(f'cannot read compilation database {path}: {error}') from error
    if not isinstance(data, list):
        raise TranslationError('compilation database must contain a JSON array')
    return path.resolve(), data


def translate_compdb(compdb, out, cuda_path=None, clang='clang++', gcc_toolchain=None, *, compatibility_profile=None):
    """Write accepted commands and a complete report, returning the report dict.

    Invalid databases raise TranslationError. Invalid or unknown entries remain
    in the report but never enter the translated compilation database.
    """
    source, entries = load_compdb(compdb)
    if compatibility_profile is not None:
        from .compatibility import translate_with_profile
        return translate_with_profile(source, entries, out, compatibility_profile,
                                      cuda_path=cuda_path, clang=clang, gcc_toolchain=gcc_toolchain)
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    commands, reports = [], []
    for index, entry in enumerate(entries):
        translated, report = translate_entry(entry, cuda_path=cuda_path, clang=clang,
                                            base_directory=source.parent, gcc_toolchain=gcc_toolchain)
        report['index'] = index
        reports.append(report)
        if translated is not None:
            commands.append(translated)
    report = {'schemaVersion': 1, 'source': str(source), 'compiler': str(clang),
              'gccToolchain': str(Path(gcc_toolchain).resolve()) if gcc_toolchain is not None else None,
              'summary': {'total': len(entries), 'translated': len(commands),
                          'translationUnknown': len(entries) - len(commands)},
              'entries': reports}
    for filename, data in [('compile_commands.json', commands), ('translation-report.json', report)]:
        target = destination / filename
        temporary = target.with_suffix(target.suffix + '.tmp')
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        temporary.replace(target)
    return report
