import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from translate import (TranslationError, compare_macros, expand_response_files,
                       macro_diff_compdb, parse_macros, translate_compdb, translate_entry)
from translate.macros import _preprocess_command


class TranslationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def entry(self, *args, file='kernel.cu'):
        return {'directory': str(self.root), 'file': file,
                'arguments': ['nvcc', *args, file]}

    def translated(self, entry):
        command, report = translate_entry(entry)
        self.assertEqual(report['unknown'], [])
        self.assertEqual(report['status'], 'translated')
        return command['arguments'], report

    def test_preserves_semantic_host_options_and_stream(self):
        args, report = self.translated(self.entry(
            '-DNAME=7', '-I', 'include folder', '-include', 'config.h', '-std=c++20',
            '--default-stream', 'per-thread', '-rdc=true', '-x', 'cu', '-c'))
        for item in ['-DNAME=7', 'include folder', 'config.h', '-std=c++20',
                     '-fgpu-default-stream=per-thread', '-fgpu-rdc', 'cuda', '--cuda-host-only']:
            self.assertIn(item, args)
        self.assertEqual(report['defaultStream'], 'per-thread')

    def test_nvbench_cudafe_diagnostic_controls(self):
        args, report = self.translated(self.entry('-Xcudafe=--display_error_number', '-Xcudafe=--promote_warnings'))
        self.assertIn('-Werror', args)
        self.assertNotIn('--display_error_number', args)
        self.assertTrue(any('numbering' in item['reason'] for item in report['dropped']))

    def test_xcompiler_separate_and_joined_preserve_order(self):
        args, _ = self.translated(self.entry(
            '-DCHOICE=1', '-Xcompiler=-D,CHOICE=2,-I,hostinc,-include,forced.h,-std=c++17',
            '-Xcompiler', '-fPIC,-pthread', '-UCHOICE'))
        self.assertLess(args.index('-DCHOICE=1'), args.index('CHOICE=2'))
        self.assertLess(args.index('CHOICE=2'), args.index('-UCHOICE'))
        self.assertIn('hostinc', args)
        self.assertIn('forced.h', args)
        self.assertIn('-std=c++17', args)
        self.assertIn('-pthread', args)

    def test_macro_quotes_and_commas_survive(self):
        args, _ = self.translated(self.entry('-Xcompiler=-DMESSAGE="hello, world",-DPAIR=a\\,b'))
        self.assertIn('-DMESSAGE="hello, world"', args)
        self.assertIn('-DPAIR=a,b', args)

    def test_command_shell_quoting_and_spaces(self):
        entry = {'directory': str(self.root), 'file': 'a b.cu',
                 'command': 'nvcc -I"some folder" -DMSG=\'"hello world"\' -c "a b.cu"'}
        args, _ = self.translated(entry)
        self.assertIn('-Isome folder', args)
        self.assertIn('-DMSG="hello world"', args)
        self.assertIn('a b.cu', args)

    def test_shell_operators_rejected(self):
        for command in ['nvcc kernel.cu && echo bad', 'nvcc kernel.cu;touch bad', 'nvcc kernel.cu > output']:
            result, report = translate_entry({'directory': str(self.root), 'file': 'kernel.cu', 'command': command})
            self.assertIsNone(result)
            self.assertIn('shell operators', report['unknown'][0]['reason'])

    def test_response_file_forms_and_nested_cwd(self):
        (self.root / 'nested').mkdir()
        (self.root / 'root.rsp').write_text('-DROOT -I "a path"')
        (self.root / 'nested' / 'child.rsp').write_text('@root.rsp -DNESTED')
        (self.root / 'other.rsp').write_text('-std=c++17')
        args, report = self.translated(self.entry('--options-file=nested/child.rsp,other.rsp'))
        self.assertIn('-DROOT', args)
        self.assertIn('a path', args)
        self.assertEqual(len(report['responseFiles']), 3)
        expanded, _ = expand_response_files(['-optf', 'root.rsp'], self.root)
        self.assertEqual(expanded, ['-DROOT', '-I', 'a path'])

    def test_response_cycle_rejected(self):
        (self.root / 'a').write_text('@b')
        (self.root / 'b').write_text('--options-file a')
        result, report = translate_entry(self.entry('@a'))
        self.assertIsNone(result)
        self.assertIn('cycle', report['unknown'][0]['reason'])

    def test_response_missing_and_bad_quotes_rejected(self):
        (self.root / 'broken').write_text('-D"unterminated')
        for file in ['missing', 'broken']:
            with self.assertRaises(TranslationError):
                expand_response_files(['@' + file], self.root)

    def test_repeated_response_is_not_cycle(self):
        (self.root / 'a').write_text('-DYES')
        args, _ = expand_response_files(['@a', '@a'], self.root)
        self.assertEqual(args, ['-DYES', '-DYES'])

    def test_response_nesting_limit(self):
        for i in range(4):
            (self.root / str(i)).write_text('@' + str(i + 1))
        with self.assertRaisesRegex(TranslationError, 'nesting limit'):
            expand_response_files(['@0'], self.root, max_depth=2)

    def test_nvcc_architecture_omission_has_evidence(self):
        args, report = self.translated(self.entry('--generate-code=arch=compute_89,code=[sm_89,compute_89]', '-arch=sm_89'))
        self.assertFalse(any('gencode' in a for a in args))
        self.assertEqual(len(report['dropped']), 2)
        self.assertIn('macro differences', report['dropped'][0]['reason'])

    def test_unrecognized_architecture_rejected(self):
        for arch in ['native', 'all', 'potato']:
            command, report = translate_entry(self.entry('-arch=' + arch))
            self.assertIsNone(command)
            self.assertEqual(report['status'], 'translation-unknown')

    def test_drop_device_and_driver_options_with_reasons(self):
        args, report = self.translated(self.entry('-ccbin=/usr/bin/g++', '-Xptxas=-v',
                                                 '-Xfatbin', '-compress-all', '--diag-suppress=20012',
                                                 '-forward-unknown-to-host-compiler'))
        self.assertEqual(len(report['dropped']), 5)
        self.assertTrue(all(r['reason'] for r in report['dropped']))
        self.assertNotIn('/usr/bin/g++', args)

    def test_nvcc_warning_as_error_policy_preserved(self):
        args, report = self.translated(self.entry('-Werror=all-warnings', '-Xcompiler=-Wno-subobject-linkage'))
        self.assertIn('-Werror', args)
        self.assertNotIn('-Werror=all-warnings', args)
        self.assertNotIn('-Wno-subobject-linkage', args)
        self.assertTrue(any('GCC-specific' in item['reason'] for item in report['dropped']))

    def test_unknown_semantic_flags_are_not_guessed(self):
        for flag in ['--future-language-mode', '-funknown-option', '-Xcudafe=--semantic-option']:
            command, report = translate_entry(self.entry(flag))
            self.assertIsNone(command)
            self.assertTrue(report['unknown'])

    def test_cudafe_diagnostic_only_can_be_dropped(self):
        _, report = self.translated(self.entry('-Xcudafe=--diag_suppress=20012'))
        self.assertEqual(len(report['dropped']), 1)

    def test_extensions_keep_their_nvcc_macros(self):
        args, _ = self.translated(self.entry('--extended-lambda', '--expt-relaxed-constexpr', '-G', '-rdc=true'))
        for name in ['__CUDACC_EXTENDED_LAMBDA__', '__CUDACC_RELAXED_CONSTEXPR__', '__CUDACC_DEBUG__', '__CUDACC_RDC__']:
            self.assertIn('-D' + name + '=1', args)
        self.assertNotIn('-fno-cuda-host-device-constexpr', args)

    def test_nvcc_default_records_clang_implicit_relaxed_constexpr_difference(self):
        _, report = self.translated(self.entry())
        self.assertTrue(any('constexpr' in assumption for assumption in report['assumptions']))

    def test_default_stream_reaches_cc1(self):
        args, report = self.translated(self.entry('--default-stream=per-thread', '--default-stream=legacy'))
        index = args.index('-fgpu-default-stream=per-thread')
        self.assertEqual(args[index - 1], '-Xclang')
        self.assertEqual(report['defaultStream'], 'legacy')

    def test_host_cpp_not_forced_into_cuda(self):
        entry = self.entry('-std=c++17', '-DHOST', file='host.cpp')
        entry['arguments'][0] = '/usr/bin/g++-13'
        args, report = self.translated(entry)
        self.assertNotIn('--cuda-host-only', args)
        self.assertNotIn('cuda', args)
        self.assertEqual(report['language'], 'c++')

    def test_explicit_language_overrides_suffix(self):
        args, report = self.translated(self.entry('-x', 'c++'))
        self.assertNotIn('--cuda-host-only', args)
        self.assertEqual(report['language'], 'c++')

    def test_nvcc_cpp_is_host_language(self):
        args, _ = self.translated(self.entry(file='host.cpp'))
        self.assertNotIn('--cuda-host-only', args)

    def test_c_source_preserves_driver_language(self):
        for compiler, language in [('gcc', 'c'), ('nvcc', 'c'), ('g++', 'c++')]:
            entry = self.entry(file='source.c')
            entry['arguments'][0] = compiler
            args, report = self.translated(entry)
            self.assertEqual(args[1:3], ['-x', language])
            self.assertEqual(report['language'], language)

    def test_inferred_cuda_language_precedes_source(self):
        args, _ = self.translated(self.entry(file='source.cuh'))
        self.assertLess(args.index('-x'), args.index('source.cuh'))

    def test_relative_clang_path_is_anchored_to_callers_directory(self):
        command, _ = translate_entry(self.entry(), clang='bin/clang++')
        self.assertEqual(command['arguments'][0], str(Path('bin/clang++').resolve()))

    def test_nvcc_implicit_cccl_system_include_preserved_after_project_include(self):
        toolkit = self.root / 'cuda'
        (toolkit / 'include/cccl').mkdir(parents=True)
        command, report = translate_entry(self.entry('-Iproject'), cuda_path=toolkit)
        args = command['arguments']
        self.assertGreater(args.index(str(toolkit / 'include/cccl')), args.index('-Iproject'))
        self.assertTrue(any('implicit CUDA include' in str(item['argument']) for item in report['mapped']))

    def test_explicit_gcc_toolchain_is_part_of_output_command(self):
        command, report = translate_entry(self.entry('-ccbin=/opt/gcc/bin/g++'), gcc_toolchain='/opt/gcc')
        self.assertIn('--gcc-toolchain=/opt/gcc', command['arguments'])
        self.assertEqual(report['gccToolchain'], '/opt/gcc')
        self.assertEqual(report['originalHostCompiler'], '/opt/gcc/bin/g++')

    def test_ccbin_without_explicit_toolchain_is_not_claimed_equivalent(self):
        _, report = self.translated(self.entry('-ccbin=/custom/g++'))
        self.assertTrue(any('standard-library include' in item for item in report['assumptions']))

    def test_cache_wrapper_removed(self):
        entry = self.entry('-c')
        entry['arguments'].insert(0, 'ccache')
        args, report = self.translated(entry)
        self.assertEqual(args[0], 'clang++')
        self.assertIn('wrapper', report['dropped'][0]['reason'])

    def test_unknown_wrapper_rejected(self):
        entry = self.entry('-c')
        entry['arguments'].insert(0, 'custom-wrapper')
        result, report = translate_entry(entry)
        self.assertIsNone(result)
        self.assertIn('wrapper', report['unknown'][0]['reason'])

    def test_missing_values_are_reported(self):
        for flag in ['-Xcompiler', '-I', '--options-file', '--default-stream', '-rdc']:
            entry = self.entry()
            entry['arguments'].append(flag)
            command, report = translate_entry(entry)
            self.assertIsNone(command)
            self.assertIn('missing value', report['unknown'][-1]['reason'])

    def test_option_cannot_consume_another_option_as_value(self):
        command, report = translate_entry(self.entry('-I', '-c'))
        self.assertIsNone(command)
        self.assertIn('missing value for -I', report['unknown'][-1]['reason'])

    def test_invalid_entries_reported_without_exception(self):
        for entry in [None, 1, {}, {'directory': str(self.root), 'file': 'x', 'arguments': []},
                      {'directory': 4, 'file': 'x', 'command': 'nvcc x'}]:
            result, report = translate_entry(entry)
            self.assertIsNone(result)
            self.assertTrue(report['unknown'])

    def test_unknown_tu_excluded_but_not_hidden(self):
        compdb = self.root / 'compile_commands.json'
        compdb.write_text(json.dumps([self.entry('-DOK'), self.entry('--future-flag')]))
        output = self.root / 'out'
        report = translate_compdb(str(compdb), str(output), '/cuda', '/usr/bin/clang++')
        self.assertEqual(report['summary'], {'total': 2, 'translated': 1, 'translationUnknown': 1})
        commands = json.loads((output / 'compile_commands.json').read_text())
        self.assertEqual(len(commands), 1)
        self.assertIn('--cuda-path=/cuda', commands[0]['arguments'])
        self.assertEqual(json.loads((output / 'translation-report.json').read_text()), report)

    def test_invalid_compdb_is_error(self):
        path = self.root / 'compile_commands.json'
        for contents in ['{broken', '{}']:
            path.write_text(contents)
            with self.assertRaises(TranslationError):
                translate_compdb(path, self.root / 'out')

    def test_source_must_match_compdb(self):
        entry = self.entry()
        entry['arguments'][-1] = 'different.cu'
        result, report = translate_entry(entry)
        self.assertIsNone(result)
        self.assertIn('source file', report['unknown'][-1]['reason'])

    def test_relative_directory_resolved_against_compdb(self):
        entry = self.entry()
        entry['directory'] = 'build'
        source = self.root / 'compile_commands.json'
        source.write_text(json.dumps([entry]))
        translate_compdb(source, self.root / 'out')
        result = json.loads((self.root / 'out/compile_commands.json').read_text())
        self.assertEqual(result[0]['directory'], str(self.root / 'build'))


class MacroTests(unittest.TestCase):
    def test_parse_and_value_sensitive_diff(self):
        before = parse_macros('#define EMPTY\n#define F(x) ((x) + 1)\n#define VALUE 1\n')
        after = parse_macros('#define EMPTY\n#define F(x) ((x) + 2)\n#define ADDED 3\n')
        diff = compare_macros(before, after)
        self.assertEqual(before['EMPTY'], '')
        self.assertEqual(diff['changed']['F(x)']['translated'], '((x) + 2)')
        self.assertEqual(diff['onlyOriginal'], {'VALUE': '1'})
        self.assertEqual(diff['onlyTranslated'], {'ADDED': '3'})
        self.assertTrue(compare_macros(before, after, ['EMPTY'])['equal'])

    def test_probe_strips_output_dependency_and_compile_options(self):
        command = _preprocess_command(['nvcc', '-c', '-o', 'x.o', '-MD', '-MF', 'x.d',
                                       '-Xcompiler=-fPIC,-MD,-MF,host.d,-DOK=1', 'x.cu'], nvcc=True)
        self.assertEqual(command, ['nvcc', '-Xcompiler=-fPIC,-DOK=1', 'x.cu', '-E', '-Xcompiler=-dM'])

    def test_probe_preserves_escaped_macro_commas(self):
        command = _preprocess_command(['nvcc', '-Xcompiler=-DPAIR=a\\,b,-DMSG="a,b",-MD', 'x.cu'], nvcc=True)
        self.assertIn('-Xcompiler=-DPAIR=a\\,b,-DMSG="a,b"', command)

    def make_report(self, root, compiler='nvcc', filename='a.cu'):
        source = root / 'compile_commands.json'
        source.write_text(json.dumps([{'directory': str(root), 'file': filename,
                                       'arguments': [compiler, '-DPROJECT=1', '-UOLD', filename]}]))
        translate_compdb(source, root / 'out')
        return root / 'out'

    def test_real_macro_collection_and_required_differences(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = self.make_report(Path(temporary), compiler='g++', filename='a.cpp')
            results = [subprocess.CompletedProcess([], 0, '#define PROJECT 1\n#define __CUDACC__ 1\n', ''),
                       subprocess.CompletedProcess([], 0, '#define PROJECT 1\n#define __CUDACC__ 1\n#define __clang__ 1\n', '')]
            with patch('translate.macros.subprocess.run', side_effect=results) as run:
                report = macro_diff_compdb(directory)
            self.assertEqual(run.call_count, 2)
            entry = report['entries'][0]
            self.assertEqual(entry['status'], 'compared')
            self.assertTrue(entry['requiredDifferences']['equal'])
            self.assertFalse(entry['differences']['equal'])
            self.assertTrue((directory / 'macro-diff.json').is_file())
            self.assertEqual(json.loads((directory / 'translation-report.json').read_text())['entries'][0]['macroDiff'], entry)

    def test_nvcc_cuda_dump_does_not_prove_host_macro_parity(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = self.make_report(Path(temporary))
            results = [subprocess.CompletedProcess([], 0, '#define PROJECT 1\n#define __CUDA_ARCH__ 890\n#define __CUDACC__ 1\n', ''),
                       subprocess.CompletedProcess([], 0, '#define PROJECT 1\n#define __CUDACC__ 1\n', '')]
            with patch('translate.macros.subprocess.run', side_effect=results):
                report = macro_diff_compdb(directory)
            entry = report['entries'][0]
            self.assertEqual(entry['status'], 'project-only')
            self.assertNotIn('requiredDifferences', entry)
            self.assertTrue(entry['observedRequiredDifferences']['equal'])
            self.assertEqual(entry['originalMacros']['__CUDA_ARCH__'], '890')
            self.assertTrue(entry['original']['containsCudaArchMacro'])

    def test_compiler_unavailable_is_honest_project_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = self.make_report(Path(temporary))
            with patch('translate.macros.subprocess.run', side_effect=FileNotFoundError('no compiler')):
                report = macro_diff_compdb(directory)
            entry = report['entries'][0]
            self.assertEqual(entry['status'], 'project-only')
            self.assertTrue(entry['differences']['equal'])
            self.assertIn('__CUDACC__', entry['unverified'])

    def test_success_without_dump_is_not_macro_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = self.make_report(Path(temporary))
            with patch('translate.macros.subprocess.run', return_value=subprocess.CompletedProcess([], 0, 'int main() {}', '')):
                report = macro_diff_compdb(directory)
            self.assertEqual(report['entries'][0]['status'], 'project-only')

    def test_macro_probe_timeout_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = self.make_report(Path(temporary))
            with patch('translate.macros.subprocess.run', side_effect=subprocess.TimeoutExpired('nvcc', 1)):
                report = macro_diff_compdb(directory, timeout=1)
            self.assertEqual(report['entries'][0]['original']['status'], 'unavailable')


class CompilerIntegrationTests(unittest.TestCase):
    def test_real_host_frontend_preserves_include_macro_and_standard(self):
        compiler = shutil.which('clang++')
        if not compiler:
            self.skipTest('Clang is unavailable')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'config.h').write_text('#define FROM_HEADER 11\n')
            (root / 'main.cpp').write_text('static_assert(PROJECT == 7);\nstatic_assert(FROM_HEADER == 11);\nstatic_assert(__cplusplus >= 202002L);\n')
            entry = {'directory': temporary, 'file': 'main.cpp',
                     'arguments': ['g++', '-Xcompiler=-DPROJECT=7,-include,config.h,-std=c++20', '-c', 'main.cpp']}
            command, report = translate_entry(entry, clang=compiler)
            self.assertEqual(report['status'], 'translated')
            process = subprocess.run(command['arguments'] + ['-fsyntax-only'], cwd=root,
                                     capture_output=True, text=True, timeout=20)
            self.assertEqual(process.returncode, 0, process.stderr)

    def test_locked_cuda_frontend_sets_per_thread_macro(self):
        root = Path(__file__).resolve().parents[1]
        compiler = root / '.deps/LLVM-23.1.3-Linux-X64/bin/clang++'
        toolkit = Path('/usr/local/cuda-13.2')
        if not compiler.is_file() or not toolkit.is_dir():
            self.skipTest('locked LLVM 23.1.3 and CUDA 13.2 installation is unavailable')
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'main.cu'
            source.write_text('#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM\n#error missing per-thread macro\n#endif\n#ifndef __CUDACC__\n#error missing CUDA macro\n#endif\n')
            command, report = translate_entry({'directory': temporary, 'file': str(source),
                                               'arguments': ['nvcc', '--default-stream=per-thread', str(source)]},
                                              clang=str(compiler), cuda_path=toolkit)
            self.assertEqual(report['status'], 'translated')
            process = subprocess.run(command['arguments'] + ['-fsyntax-only'], cwd=temporary,
                                     capture_output=True, text=True, timeout=20)
            self.assertEqual(process.returncode, 0, process.stderr)


if __name__ == '__main__':
    unittest.main()
