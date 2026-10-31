"""Mocked product-profile tests; no checker, GPU, or holdout invocation."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from translate import TranslationError, macro_diff_compdb, translate_compdb, translate_entry
from translate import compatibility as compat
from translate.__main__ import main as standalone_main
from tool.cli import main as product_main


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        for name in ('nvcc', 'g++', 'gcc', 'clang++', 'ccache', 'sccache'):
            (self.bin / name).write_text('mock executable identity: ' + name)
        self.host_include = self.root / 'gcc include'
        self.host_include.mkdir()
        (self.host_include / 'omp.h').write_text('// omp input\n')
        self.cccl = self.root / 'cccl'
        for name in ('thrust/thrust/test.h', 'cub/cub/test.h', 'libcudacxx/include/cuda/std/test.h',
                     'libcudacxx/include/cuda/atomic'):
            p = self.cccl / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('// pristine mock ' + name)
        self.tree_hash = compat.json_digest({str(p.relative_to(self.cccl)): compat.digest(p)
                                            for p in self.cccl.rglob('*') if p.is_file()})
        self.addCleanup(patch.stopall)
        patch.object(compat, 'CCCL_TREE_SHA256', self.tree_hash).start()
        patch.object(compat, 'CCCL_LINKS', {}).start()
        self.calls = []
        self.dump = '#define PROJECT 7\n#define __CUDACC__ 1\n'
        self.dependency_failure = False
        self.foreign_header = None
        self.collect = patch('translate.compatibility.subprocess.run', side_effect=self.run_compiler).start()
        self.source = self.root / 'input.cu'
        self.source.write_text('int input;\n')
        self.database = self.root / 'compile_commands.json'
        self.template = self.root / 'profile.json'
        self.template.write_text(json.dumps({'schemaVersion': 1, 'profile': 'cccl-510cc9d-host-headers',
                                             'ccclRoot': 'cccl'}))

    def run_compiler(self, command, **kwargs):
        self.calls.append(list(command))
        if command[-1] == '--version':
            return subprocess.CompletedProcess(command, 0, 'test compiler 13.3 / Clang 23.1.3\n', '')
        if command[-1] == '-print-file-name=include':
            return subprocess.CompletedProcess(command, 0, str(self.host_include) + '\n', '')
        if command[-1] == '-M':
            if self.dependency_failure:
                return subprocess.CompletedProcess(command, 1, '', 'dependency failure')
            header = self.foreign_header or self.cccl / 'thrust/thrust/test.h'
            escape = lambda value: str(value).replace(' ', '\\ ')
            return subprocess.CompletedProcess(command, 0, 'input.o: ' + escape(self.source) + ' ' + escape(header) + '\n', '')
        return subprocess.CompletedProcess(command, 0, self.dump, 'full diagnostic\n' * 1000)

    def entry(self, host=False, extra=None):
        source = self.source
        if host:
            source = self.root / 'input.cpp'
            source.write_text('int host;\n')
        args = [str(self.bin / ('g++' if host else 'nvcc'))]
        if not host:
            args += ['-ccbin', str(self.bin / 'g++')]
        return {'directory': str(self.root), 'file': str(source),
                'arguments': args + ['-DPROJECT=7', '-I', 'project include', '-std=c++17', *(extra or []), '-c', str(source)]}

    def translate(self, entries=None, output='out'):
        self.database.write_text(json.dumps(entries or [self.entry()]))
        return translate_compdb(self.database, self.root / output, clang=self.bin / 'clang++',
                                gcc_toolchain=self.root / 'gcc', compatibility_profile=self.template)

    def test_three_roots_reference_tail_discovery_and_original_evidence(self):
        original = self.entry(extra=['--default-stream=per-thread', '--extended-lambda'])
        report = self.translate([original])
        row = report['entries'][0]
        roots = [str(self.cccl / p) for p in compat.CCCL_INCLUDES]
        self.assertEqual(row['originalArguments'], original['arguments'])
        self.assertEqual(row['expandedArguments'], original['arguments'])
        self.assertEqual(row['referenceArguments'][1:7], [x for r in roots for x in ('-I', r)])
        self.assertNotIn('-idirafter', row['referenceArguments'])
        self.assertEqual(row['translatedArguments'][-2:], ['-idirafter', str(self.host_include)])
        self.assertEqual(row['defaultStream'], 'per-thread')
        self.assertEqual(report['schemaVersion'], 2)
        self.assertFalse(report['compatibility']['syntaxVerified'])
        out = self.root / 'out'
        self.assertEqual((out / 'original-compile_commands.json').read_bytes(), self.database.read_bytes())
        lock = json.loads((out / 'compatibility-resolved.json').read_text())
        self.assertEqual(lock['entryBindings']['0']['originalEntrySha256'], compat.json_digest(original))
        self.assertEqual(lock['cccl']['tree']['treeSha256'], self.tree_hash)
        self.assertTrue(lock['inputsUnchangedDuringTranslation'])
        self.assertEqual([c[-1] for c in self.calls].count('-print-file-name=include'), 1)

    def test_duplicate_tus_and_unknown_keep_original_indices(self):
        first = self.entry()
        second = self.entry(extra=['--unknown-semantic-option'])
        report = self.translate([first, second, first])
        self.assertEqual(report['summary'], {'total': 3, 'translated': 2, 'translationUnknown': 1})
        self.assertEqual([r['index'] for r in report['entries']], [0, 1, 2])
        self.assertEqual(report['entries'][2]['translatedEntryIndex'], 1)
        self.assertEqual(report['entries'][1]['referenceEntryIndex'], 1)
        self.assertEqual(len(json.loads((self.root / 'out/reference-compile_commands.json').read_text())), 3)

    def test_host_control_keeps_cccl_and_discovers_actual_c_and_cpp_driver(self):
        self.template.write_text(json.dumps({'schemaVersion': 1, 'profile': 'host-headers'}))
        cpp = self.entry(host=True)
        c = {**cpp, 'arguments': [str(self.bin / 'gcc'), *cpp['arguments'][1:]]}
        report = self.translate([cpp, c])
        self.assertEqual(report['entries'][0]['referenceArguments'], cpp['arguments'])
        self.assertNotIn('-x', report['entries'][0]['translatedArguments'])
        lock = json.loads((self.root / 'out/compatibility-resolved.json').read_text())
        self.assertIsNone(lock['cccl'])
        self.assertEqual(len(lock['hostHeaders']), 2)

    def test_response_and_multiple_wrappers_preserved_in_reference(self):
        response = self.root / 'options file'
        response.write_text('-DQUOTED="path/old,hello" -I "include space"')
        entry = self.entry(extra=['@' + str(response)])
        entry['arguments'] = [str(self.bin / 'ccache'), str(self.bin / 'sccache'), *entry['arguments']]
        row = self.translate([entry])['entries'][0]
        self.assertEqual(row['referenceArguments'][:3], entry['arguments'][:3])
        self.assertEqual(row['referenceArguments'][3:5], ['-I', str(self.cccl / 'thrust')])
        self.assertIn('-DQUOTED=path/old,hello', row['referenceArguments'])
        self.assertEqual(row['responseFiles'][0]['identity']['sha256'], compat.digest(response))

    def test_wrong_tree_or_added_file_rejected_without_output(self):
        (self.cccl / 'unexpected.h').write_text('changed')
        with self.assertRaisesRegex(TranslationError, 'full source tree'):
            self.translate()
        self.assertFalse((self.root / 'out').exists())

    def test_source_symlink_replacement_rejected(self):
        p = self.cccl / 'thrust/thrust/test.h'
        original = p.read_text()
        other = self.root / 'outside.h'
        other.write_text(original)
        p.unlink(); p.symlink_to(other)
        with self.assertRaisesRegex(TranslationError, 'symlink'):
            self.translate()

    def test_invalid_template_and_nonempty_output_rejected(self):
        self.template.write_text(json.dumps({'schemaVersion': 1, 'profile': 'host-headers', 'extraFlags': ['-D__NVCC__']}))
        with self.assertRaisesRegex(TranslationError, 'unknown compatibility'):
            self.translate()
        self.template.write_text(json.dumps({'schemaVersion': 1, 'profile': 'host-headers'}))
        self.translate()
        with self.assertRaisesRegex(TranslationError, 'new or empty'):
            self.translate()

    def test_missing_ccbin_and_directory_ccbin_are_reported_unknown(self):
        e = self.entry(); del e['arguments'][1:3]
        report = self.translate([e])
        self.assertEqual(report['summary']['translationUnknown'], 1)
        self.assertIn('explicit actual', report['entries'][0]['unknown'][-1]['reason'])
        e = self.entry(); e['arguments'][2] = str(self.bin)
        self.assertEqual(self.translate([e], 'out2')['summary']['translationUnknown'], 1)

    def test_discovery_failure_and_mutation_are_not_accepted(self):
        def fail(command, **kwargs):
            if command[-1] == '-print-file-name=include':
                return subprocess.CompletedProcess(command, 0, 'relative/include\n', '')
            return self.run_compiler(command, **kwargs)
        self.collect.side_effect = fail
        report = self.translate()
        self.assertEqual(report['summary']['translationUnknown'], 1)
        lock = json.loads((self.root / 'out/compatibility-resolved.json').read_text())
        self.assertEqual(lock['discoveryAttempts'][0]['stdout'], 'relative/include\n')
        self.collect.side_effect = self.run_compiler
        self.translate(output='out2')
        (self.host_include / 'new.h').write_text('// drift')
        with self.assertRaisesRegex(TranslationError, 'host header directory changed'):
            macro_diff_compdb(self.root / 'out2')

    def test_macro_uses_derived_reference_retains_full_outputs_and_cuda_gate(self):
        report = self.translate()
        macros = macro_diff_compdb(self.root / 'out')
        row = macros['entries'][0]
        self.assertEqual(row['status'], 'project-only')
        self.assertEqual(row['referenceKind'], 'compatibility-derived')
        self.assertIn(str(self.cccl / 'thrust'), row['original']['command'])
        self.assertEqual(row['profileHeaderEvidence']['status'], 'verified')
        self.assertGreater(len(Path(row['original']['stderrFile']).read_text()), 8000)
        self.assertEqual(row['originalArguments'], report['entries'][0]['originalArguments'])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(product_main(['macros', '--compdb', str(self.root / 'out'), '--out', str(self.root / 'macros')]), 2)

    def test_profile_missing_reference_or_changed_output_is_rejected(self):
        self.translate()
        path = self.root / 'out/translation-report.json'
        data = json.loads(path.read_text()); del data['entries'][0]['referenceArguments']; path.write_text(json.dumps(data))
        with self.assertRaisesRegex(TranslationError, 'binding mismatch'):
            macro_diff_compdb(path)
        self.translate(output='out2')
        (self.root / 'out2/compile_commands.json').write_text('[]')
        with self.assertRaisesRegex(TranslationError, 'output database changed'):
            macro_diff_compdb(self.root / 'out2')

    def test_mixed_old_header_and_dependency_failure_are_unverified(self):
        self.translate()
        foreign = self.root / 'old/include/cccl/thrust/test.h'
        foreign.parent.mkdir(parents=True); foreign.write_text('// foreign')
        self.foreign_header = foreign
        self.assertEqual(macro_diff_compdb(self.root / 'out')['entries'][0]['profileHeaderEvidence']['status'], 'unverified')
        self.foreign_header = None; self.dependency_failure = True
        self.assertEqual(macro_diff_compdb(self.root / 'out')['entries'][0]['profileHeaderEvidence']['status'], 'unverified')

    def test_both_cli_translate_flags_and_legacy_no_subprocess(self):
        self.database.write_text(json.dumps([self.entry()]))
        for main, out in [(product_main, 'public'), (standalone_main, 'standalone')]:
            with contextlib.redirect_stdout(io.StringIO()):
                status = main(['translate', '--compdb', str(self.database), '--out', str(self.root / out),
                               '--clang', str(self.bin / 'clang++'), '--compatibility-profile', str(self.template)])
            self.assertEqual(status, 0)
        self.collect.reset_mock()
        translate_entry(self.entry(), clang=str(self.bin / 'clang++'))
        legacy = translate_compdb(self.database, self.root / 'legacy', clang=str(self.bin / 'clang++'))
        self.collect.assert_not_called()
        self.assertEqual(legacy['schemaVersion'], 1)
        self.assertNotIn('compatibility', legacy)

    def test_repeated_failed_driver_identity_never_becomes_accepted(self):
        def failed_host(command, **kwargs):
            if command == [str(self.bin / 'g++'), '--version']:
                return subprocess.CompletedProcess(command, 1, '', 'cannot identify host')
            return self.run_compiler(command, **kwargs)
        self.collect.side_effect = failed_host
        self.assertEqual(self.translate([self.entry(), self.entry()])['summary']['translationUnknown'], 2)

    def test_timeout_query_and_boolean_schema_rejected(self):
        def timed_out(command, **kwargs):
            if command[-1] == '-print-file-name=include':
                raise subprocess.TimeoutExpired(command, 30)
            return self.run_compiler(command, **kwargs)
        self.collect.side_effect = timed_out
        self.assertEqual(self.translate()['summary']['translationUnknown'], 1)
        self.template.write_text(json.dumps({'schemaVersion': True, 'profile': 'host-headers'}))
        with self.assertRaisesRegex(TranslationError, 'template schema'):
            self.translate(output='out2')

    def test_output_cannot_modify_header_tree(self):
        with self.assertRaisesRegex(TranslationError, 'output may not enter'):
            self.translate(output='cccl/new-output')
        self.translate()
        with self.assertRaisesRegex(TranslationError, 'output may not enter'):
            macro_diff_compdb(self.root / 'out', self.cccl)
        self.assertFalse((self.cccl / '0000-reference-macros.stdout.txt').exists())

    def test_dependency_sets_may_differ_and_no_cccl_is_not_applicable(self):
        self.translate([self.entry(host=True)])
        def different_sets(command, **kwargs):
            if command[-1] == '-M':
                name = 'thrust/thrust/test.h' if 'clang' in command[0] else 'cub/cub/test.h'
                return subprocess.CompletedProcess(command, 0, f'input.o: {self.cccl / name}\n', '')
            return self.run_compiler(command, **kwargs)
        self.collect.side_effect = different_sets
        row = macro_diff_compdb(self.root / 'out')['entries'][0]
        self.assertEqual(row['status'], 'compared')
        self.assertEqual(row['profileHeaderEvidence']['status'], 'verified')
        def no_candidate(command, **kwargs):
            if command[-1] == '-M':
                return subprocess.CompletedProcess(command, 0, f'input.o: {self.source}\n', '')
            return self.run_compiler(command, **kwargs)
        self.collect.side_effect = no_candidate
        self.assertEqual(macro_diff_compdb(self.root / 'out')['entries'][0]['profileHeaderEvidence']['status'], 'not-applicable')

    def test_source_drift_during_macro_probe_prevents_report_success(self):
        self.translate()
        before = (self.root / 'out/translation-report.json').read_bytes()
        def drift(command, **kwargs):
            result = self.run_compiler(command, **kwargs)
            if command[-1] == '-M':
                self.source.write_text('changed input during evidence collection')
            return result
        self.collect.side_effect = drift
        with self.assertRaisesRegex(TranslationError, 'input/tool changed'):
            macro_diff_compdb(self.root / 'out')
        self.assertEqual((self.root / 'out/translation-report.json').read_bytes(), before)
        self.assertFalse((self.root / 'out/macro-diff.json').exists())

    def test_language_metadata_cannot_upgrade_nvcc_macro_gate(self):
        self.translate()
        path = self.root / 'out/translation-report.json'
        data = json.loads(path.read_text())
        data['entries'][0]['language'] = 'c++'
        path.write_text(json.dumps(data))
        self.calls.clear()
        with self.assertRaisesRegex(TranslationError, 'metadata binding mismatch'):
            macro_diff_compdb(path)
        self.assertFalse(self.calls)

    def test_profile_report_cannot_downgrade_to_legacy(self):
        self.translate([self.entry(host=True)])
        path = self.root / 'out/translation-report.json'
        data = json.loads(path.read_text())
        del data['compatibility']
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(TranslationError, 'missing compatibility metadata'):
            macro_diff_compdb(path)
        data['schemaVersion'] = 1
        del data['entries'][0]['compatibility']
        del data['entries'][0]['referenceArguments']
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(TranslationError, 'missing compatibility metadata'):
            macro_diff_compdb(path)

    def test_macro_output_cannot_overwrite_bound_input_or_symlink(self):
        self.database = self.root / 'macro-diff.json'
        self.translate([self.entry(host=True)])
        before = self.database.read_bytes()
        self.calls.clear()
        with self.assertRaisesRegex(TranslationError, 'collides with a bound input'):
            macro_diff_compdb(self.root / 'out', self.root)
        self.assertEqual(self.database.read_bytes(), before)
        self.assertFalse(self.calls)
        raw = self.root / 'out/0000-reference-macros.stdout.txt'
        raw.symlink_to(self.database)
        with self.assertRaisesRegex(TranslationError, 'collides with a bound input'):
            macro_diff_compdb(self.root / 'out')
        self.assertEqual(self.database.read_bytes(), before)
        raw.unlink()
        raw.hardlink_to(self.database)
        with self.assertRaisesRegex(TranslationError, 'collides with a bound input'):
            macro_diff_compdb(self.root / 'out')
        self.assertEqual(self.database.read_bytes(), before)
        raw.unlink()
        raw.write_text('previous capture')
        (self.root / 'out/0000-reference-macros.stderr.txt').hardlink_to(raw)
        with self.assertRaisesRegex(TranslationError, 'collide through a hard link'):
            macro_diff_compdb(self.root / 'out')
        raw.unlink()
        (self.root / 'out/0000-reference-macros.stderr.txt').unlink()
        report = self.root / 'out/translation-report.json'
        before_report = report.read_bytes()
        (self.root / 'out/macro-diff.json').hardlink_to(report)
        with self.assertRaisesRegex(TranslationError, 'collide through a hard link'):
            macro_diff_compdb(self.root / 'out')
        self.assertEqual(report.read_bytes(), before_report)

    def test_path_driver_resolution_drift_rejected(self):
        for name in ('g++', 'clang++'):
            (self.bin / name).chmod(0o700)
        other = self.root / 'other-bin'
        other.mkdir()
        (other / 'g++').write_text('different compiler')
        (other / 'g++').chmod(0o700)
        entry = self.entry(host=True)
        entry['arguments'][0] = 'g++'
        with patch.dict(os.environ, {'PATH': str(self.bin)}):
            self.translate([entry])
        self.calls.clear()
        with patch.dict(os.environ, {'PATH': str(other) + os.pathsep + str(self.bin)}):
            with self.assertRaisesRegex(TranslationError, 'driver resolution changed'):
                macro_diff_compdb(self.root / 'out')
        self.assertFalse(self.calls)

    def test_foreign_public_cuda_header_is_not_not_applicable(self):
        self.translate([self.entry(host=True)])
        self.foreign_header = self.root / 'other-libcudacxx/include/cuda/atomic'
        self.foreign_header.parent.mkdir(parents=True)
        self.foreign_header.write_text('// wrong dependency')
        row = macro_diff_compdb(self.root / 'out')['entries'][0]
        self.assertEqual(row['profileHeaderEvidence']['status'], 'unverified')
        self.assertEqual(row['profileHeaderEvidence']['dependencies'][0]['foreignCcclFiles'],
                         [str(self.foreign_header)])


if __name__ == '__main__':
    unittest.main()
