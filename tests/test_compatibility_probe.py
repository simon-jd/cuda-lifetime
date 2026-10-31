"""Synthetic syntax-protocol tests; never real CUDA or checker evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import compatibility_probe as probe


def write(path, value):
    path.write_text(json.dumps(value))


class CompatibilityProbeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.sources = self.root / 'sources'
        self.sources.mkdir()
        self.source = self.sources / 'same.cu'
        self.source.write_text('// invented source, no CUDA execution\n')
        self.host = self.sources / 'host.cpp'
        self.host.write_text('// invented host source\n')
        self.new = self.root / 'cccl-new'
        self.old = self.root / 'cccl-old'
        for root in (self.new, self.old):
            (root / 'cuda/std').mkdir(parents=True)
            (root / 'cuda/std/type_traits').write_text('// invented header\n')
        (self.new / 'cuda/std/utility').write_text('// another invented header\n')
        self.clang, self.nvcc = self.root / 'clang++', self.root / 'nvcc'
        self.gcc = self.root / 'g++-13'
        for path in (self.clang, self.nvcc, self.gcc):
            path.write_text('synthetic protocol compiler bytes; never executed\n')
        self.database = self.root / 'compile_commands.json'
        self.entries = [self.entry(self.source, '-DVARIANT=' + str(index)) for index in range(2)]
        write(self.database, self.entries)
        self.manifest = {'schemaVersion': 1, 'candidateId': 'invented-protocol', 'target': 'cuda-samples',
                         'scope': 'all-configured', 'compdb': {'path': str(self.database), 'sha256': probe.digest(self.database)},
                         'tools': {'clang': {'path': str(self.clang), 'sha256': probe.digest(self.clang)},
                                   'gcc': {'path': str(self.gcc), 'sha256': probe.digest(self.gcc)},
                                   'nvcc': {'path': str(self.nvcc), 'sha256': probe.digest(self.nvcc)}},
                         'inputRoots': [str(self.sources), str(self.new)], 'macroIndices': [0], 'jobs': 1,
                         'cccl': {'root': str(self.new), 'oldRoots': [str(self.old)], 'treeSha256': probe.tree_hash(self.new),
                                  'source': {'commit': 'invented-not-an-upstream-release'}, 'reason': 'synthetic protocol fixture', 'prependInclude': True}}
        self.manifest_path = self.root / 'manifest.json'
        baseline_path = self.baseline_path = self.root / 'baseline-inputs.json'
        write(baseline_path, {'schemaVersion': 1, 'inputRoots': [str(self.sources)],
                             'inputPathIdentities': probe.input_path_identities([self.sources]),
                             'files': {str(self.source): probe.digest(self.source),
                                                         str(self.host): probe.digest(self.host),
                                                         str(self.database): probe.digest(self.database)}})
        self.manifest['baselineInputManifest'] = {'path': str(baseline_path), 'sha256': probe.digest(baseline_path)}
        self.out = self.root / 'output'
        self.syntax_calls = []
        self.after_syntax = None
        self.foreign = False
        self.header_missing = False
        self.syntax_integrity = True
        self.no_cccl = False
        self.syntax_failure_indices = set()
        self.missing_dependency_indices = set()
        self.extra_header = None
        self.macro_failure = False
        self.discovery_output = None
        self.discovery_failure = False
        self.after_discovery = None

    def entry(self, source, *extra):
        return {'file': str(source), 'directory': str(self.sources),
                'arguments': [str(self.nvcc), '-ccbin=' + str(self.gcc), '-std=c++17', '-I' + str(self.old), *extra, '-c', str(source)]}

    def bind_database(self):
        self.manifest['compdb']['sha256'] = probe.digest(self.database)
        baseline = json.loads(self.baseline_path.read_text())
        baseline['files'][str(self.database)] = probe.digest(self.database)
        write(self.baseline_path, baseline)
        self.manifest['baselineInputManifest']['sha256'] = probe.digest(self.baseline_path)

    def bind_baseline_paths(self):
        baseline = json.loads(self.baseline_path.read_text())
        baseline['inputPathIdentities'] = probe.input_path_identities(baseline['inputRoots'])
        write(self.baseline_path, baseline)
        self.manifest['baselineInputManifest']['sha256'] = probe.digest(self.baseline_path)

    def configure_header_discovery(self):
        self.gcc_include = self.root / 'gcc-headers'
        self.gcc_include.mkdir()
        (self.gcc_include / 'omp.h').write_text('// synthetic registered omp header\n')
        (self.gcc_include / 'unaccessed.h').write_text('// full directory must be baseline bound\n')
        self.manifest['schemaVersion'] = 2
        self.manifest['inputRoots'].append(str(self.gcc_include))
        baseline = json.loads(self.baseline_path.read_text())
        baseline['inputRoots'].append(str(self.gcc_include))
        baseline['inputPathIdentities'] = probe.input_path_identities(baseline['inputRoots'])
        baseline['files'].update({str(path): probe.digest(path) for path in self.gcc_include.iterdir()})
        write(self.baseline_path, baseline)
        self.manifest['baselineInputManifest']['sha256'] = probe.digest(self.baseline_path)
        self.evidence_file = self.root / 'synthetic-discovery-evidence.json'
        write(self.evidence_file, {'synthetic': True, 'notRealCompilerEvidence': True})
        self.manifest['clangIncludeAfter'] = [{'directory': str(self.gcc_include), 'compilerRole': 'gcc',
            'discoveryCommand': [str(self.gcc), '-print-file-name=include'], 'reason': 'synthetic registered host-header discovery',
            'evidenceFiles': [{'path': str(self.evidence_file), 'sha256': probe.digest(self.evidence_file)}]}]
        self.discovery_output = str(self.gcc_include) + '\n'

    def syntax(self, index, entry, args, command_hash):
        self.syntax_calls.append((index, entry))
        if self.after_syntax:
            self.after_syntax()
        failed = index in self.syntax_failure_indices
        return {'status': 'failed' if failed else 'passed', 'returncode': 1 if failed else 0, 'compilationCommandIndex': index,
                'inputCommandSha256': command_hash, 'command': [*entry['arguments'], '-fsyntax-only'],
                'sourceUnchanged': True, 'inputDatabaseUnchanged': self.syntax_integrity}

    def capture(self, command, directory, output, name, timeout):
        stdout = 'synthetic protocol fixture, not a real compiler\n'
        if command[0] == 'ldd':
            stdout = ''
        elif '-print-resource-dir' in command:
            stdout = str(self.new) + '\n'
        elif '-print-file-name=include' in command:
            stdout = self.discovery_output or ''
            if self.after_discovery:
                self.after_discovery()
        elif '-M' in command:
            source = next(arg for arg in command if arg.endswith(('.cu', '.cpp')))
            folder = self.old if self.foreign and name.endswith('clang-dependencies') else self.new
            headers = [folder / 'cuda/std/type_traits']
            if name.endswith('reference-dependencies'):
                headers.append(self.new / 'cuda/std/utility')  # Different accessed set is legal.
            if self.no_cccl:
                headers = []
            if self.extra_header:
                headers.append(self.extra_header)
            stdout = 'fixture.o: ' + source + ' ' + ' '.join(map(str, headers)) + '\n'
            if self.header_missing:
                stdout = 'fixture.o: /missing-header\n'
        elif '-E' in command:
            stdout = '#define __CUDACC__ 1\n#define VARIANT 0\n#define __CUDA_ARCH__ 890\n'
        value = {'command': command, 'directory': str(directory), 'returncode': 0,
                 'stdout': stdout, 'stderr': '', 'status': 'completed', 'seconds': 0}
        if '-M' in command and int(name.split('-')[0]) in self.missing_dependency_indices:
            value.update(returncode=1, stdout='', stderr='synthetic missing fixture header')
        if '-E' in command and self.macro_failure:
            value.update(returncode=1, stdout='', stderr='synthetic preprocessing failure')
        if '-print-file-name=include' in command and self.discovery_failure:
            value.update(returncode=1, stdout='', stderr='synthetic compiler discovery failure')
        for field in ('stdout', 'stderr'):
            path = output / (name + '.' + field + '.txt')
            path.write_text(value[field])
            value[field + 'File'] = str(path)
        return value

    def execute(self):
        write(self.manifest_path, self.manifest)
        with patch.object(probe, 'capture', side_effect=self.capture), patch.object(probe, 'run_one', side_effect=self.syntax):
            report = probe.probe(self.manifest_path, probe.digest(self.manifest_path), self.out)
        self.assertTrue(report['finishedAt'])
        self.assertEqual(report, json.loads((self.out / 'report.json').read_text()))
        return report

    def test_distinct_configurations_and_different_header_access_sets_are_retained(self):
        report = self.execute()
        self.assertTrue(report['runComplete'], report['failures'])
        self.assertTrue(report['cuThresholdPassed'])
        self.assertTrue(report['dependencyHeaderParityVerified'])
        self.assertEqual([index for index, _ in self.syntax_calls], [0, 1])
        self.assertNotEqual(report['records'][0]['originalEntrySha256'], report['records'][1]['originalEntrySha256'])
        parity = report['records'][0]['dependencyHeaderParity']
        self.assertNotEqual(set(parity['referenceFiles']), set(parity['clangFiles']))
        self.assertTrue(parity['verified'])
        self.assertEqual((self.out / 'original-compile_commands.json').read_bytes(), self.database.read_bytes())

    def test_unknown_remains_in_denominator_and_exact_ninety_percent_fails(self):
        self.entries = [self.entry(self.source, '-DVARIANT=' + str(i)) for i in range(9)]
        self.entries.append(self.entry(self.source, '-fdeliberately-unsupported-option'))
        self.entries.append(self.entry(self.host, '-DHOST=1'))
        write(self.database, self.entries)
        self.bind_database()
        report = self.execute()
        self.assertEqual(report['counts']['cudaCu'], 10)
        self.assertEqual(report['counts']['allConfigured'], 11)
        self.assertEqual(report['counts']['translationUnknown'], 1)
        self.assertEqual(report['rates']['cudaCu'], .9)
        self.assertFalse(report['cuThresholdPassed'])
        self.assertEqual(report['rates']['allConfigured'], 10 / 11)
        self.assertEqual(report['records'][9]['status'], 'translation-unknown')

    def test_diagnostic_subset_never_claims_full_gate(self):
        self.manifest.update(scope='diagnostic-subset', indices=[0])
        report = self.execute()
        self.assertTrue(report['diagnosticComplete'])
        self.assertFalse(report['runComplete'])
        self.assertIsNone(report['cuThresholdPassed'])
        self.assertIsNone(report['rates']['allConfigured'])

    def test_full_scope_cannot_omit_commands(self):
        self.manifest['indices'] = [0]
        report = self.execute()
        self.assertFalse(report['runComplete'])
        self.assertEqual(self.syntax_calls, [])

    def test_actual_tool_hash_mismatch_stops_before_syntax(self):
        self.nvcc.write_text('different compiler bytes')
        report = self.execute()
        self.assertFalse(report['runComplete'])
        self.assertIn('tool hash differs', report['failures'][0]['message'])
        self.assertEqual(self.syntax_calls, [])

    def test_unregistered_original_compiler_is_not_inferred_from_name(self):
        self.manifest['tools'].pop('nvcc')
        report = self.execute()
        self.assertFalse(report['runComplete'])
        self.assertIn('registered actual identity', report['records'][0]['error']['message'])

    def test_changed_original_database_is_rejected(self):
        self.database.write_text(self.database.read_text() + '\n')
        report = self.execute()
        self.assertFalse(report['runComplete'])
        self.assertEqual(self.syntax_calls, [])

    def test_clang_using_old_cccl_is_not_header_parity(self):
        self.foreign = True
        report = self.execute()
        self.assertFalse(report['dependencyHeaderParityVerified'])
        self.assertFalse(report['inputIntegrityComplete'])
        self.assertTrue(report['records'][0]['dependencyHeaderParity']['clangForeign'])

    def test_failed_dependency_identity_does_not_turn_into_verified(self):
        self.header_missing = True
        report = self.execute()
        self.assertFalse(report['dependencyHeaderParityVerified'])
        self.assertFalse(report['inputIntegrityComplete'])
        self.assertTrue(report['runComplete'])
        self.assertEqual(report['headerIdentityUnknownIndices'], [0, 1])

    def test_changed_baseline_source_is_detected_before_syntax(self):
        self.source.write_text('changed before the candidate was started\n')
        report = self.execute()
        self.assertFalse(report['runComplete'])
        self.assertIn('baseline input bytes changed', report['failures'][0]['message'])
        self.assertEqual(self.syntax_calls, [])

    def test_unbound_inputs_can_only_produce_diagnostic_evidence(self):
        self.manifest.pop('baselineInputManifest')
        report = self.execute()
        self.assertTrue(report['diagnosticComplete'])
        self.assertFalse(report['baselineBindingVerified'])
        self.assertFalse(report['runComplete'])

    def test_failed_tu_header_unknown_does_not_hide_successful_scope_or_strict_gate(self):
        self.entries = [self.entry(self.source, '-DVARIANT=' + str(i)) for i in range(11)]
        write(self.database, self.entries)
        self.bind_database()
        self.syntax_failure_indices = {10}
        self.missing_dependency_indices = {10}
        report = self.execute()
        self.assertTrue(report['runComplete'])
        self.assertTrue(report['cuThresholdPassed'])
        self.assertEqual(report['rates']['cudaCu'], 10 / 11)
        self.assertFalse(report['inputIntegrityComplete'])
        self.assertFalse(report['dependencyHeaderParityVerified'])
        self.assertTrue(report['successfulSyntaxInputVerified'])
        self.assertTrue(report['successfulSyntaxHeaderParityVerified'])
        self.assertEqual(report['headerIdentityUnknownIndices'], [10])
        self.assertFalse(report['inputDriftDetected'])

    def test_nvcc_macro_dump_never_claims_host_pass_parity(self):
        report = self.execute()
        macro = report['records'][0]['macroEvidence']
        self.assertEqual(macro['status'], 'project-only')
        self.assertFalse(macro['hostMacroParityVerified'])
        self.assertTrue(macro['nvccHostPassUnverified'])

    def test_failed_macro_collection_is_unavailable_not_project_only(self):
        self.macro_failure = True
        report = self.execute()
        self.assertEqual(report['records'][0]['macroEvidence']['status'], 'unavailable')
        self.assertFalse(report['hostMacroParityVerified'])

    def test_isolated_command_mutation_prevents_complete_input_claim(self):
        self.syntax_integrity = False
        report = self.execute()
        self.assertTrue(report['inputsUnchanged'])
        self.assertFalse(report['inputIntegrityComplete'])
        self.assertFalse(report['runComplete'])

    def test_no_cccl_access_is_not_mislabeled_as_mixed_headers(self):
        self.no_cccl = True
        report = self.execute()
        self.assertTrue(report['dependencyHeaderParityVerified'])
        self.assertFalse(report['records'][0]['dependencyHeaderParity']['applicable'])

    def test_source_archive_multiple_include_roots_keep_order_on_both_compilers(self):
        cub, thrust = self.new / 'cub', self.new / 'thrust'
        cub.mkdir()
        thrust.mkdir()
        roots = [str(thrust), str(cub), str(self.new / 'cuda')]
        self.manifest['cccl']['includeRoots'] = roots
        self.manifest['cccl']['treeSha256'] = probe.tree_hash(self.new)
        report = self.execute()
        self.assertTrue(report['dependencyHeaderParityVerified'])
        row = report['records'][0]
        expected = [token for folder in roots for token in ('-I', folder)]
        self.assertEqual(row['referenceArguments'][1:7], expected)
        self.assertEqual(row['translatedArguments'][3:9], expected)  # -x cuda precedes original options.
        self.assertIn('-I' + str(self.old), row['referenceArguments'])

    def test_header_under_source_tree_but_outside_registered_include_roots_is_rejected(self):
        declared = self.new / 'only-this'
        declared.mkdir()
        self.manifest['cccl']['includeRoots'] = [str(declared)]
        report = self.execute()
        self.assertFalse(report['dependencyHeaderParityVerified'])
        self.assertTrue(report['records'][0]['dependencyHeaderParity']['clangForeign'])

    def test_include_insertion_preserves_compiler_cache_wrapper(self):
        updated, _ = probe.rewrite_paths(['ccache', 'nvcc', '-c', 'a.cu'], [], ['/new/thrust', '/new/cub'])
        self.assertEqual(updated[:6], ['ccache', 'nvcc', '-I', '/new/thrust', '-I', '/new/cub'])

    def test_source_mutation_is_reported_after_probe(self):
        self.after_syntax = lambda: self.source.write_text('changed source\n')
        report = self.execute()
        self.assertFalse(report['inputsUnchanged'])
        self.assertFalse(report['runComplete'])
        self.assertIn(str(self.source), report['changedInputs'])

    def test_include_rewrite_does_not_touch_project_macro_payload(self):
        arguments = ['nvcc', '-I/old', '-isystem', '/old/sub', '-DPATH=/old', '--default-stream', 'per-thread']
        updated, ledger = probe.rewrite_paths(arguments, [('/old', '/new')])
        self.assertEqual(updated, ['nvcc', '-I/new', '-isystem', '/new/sub', '-DPATH=/old', '--default-stream', 'per-thread'])
        self.assertEqual(len(ledger), 2)
        self.assertEqual(arguments[1], '-I/old')

    def test_dependency_make_escaping_and_continuations(self):
        spaced = self.sources / 'with space.h'
        spaced.write_text('fixture')
        value = 'target.o: ' + str(self.source) + ' \\\n ' + str(spaced).replace(' ', '\\ ') + '\n'
        self.assertEqual(probe.dependency_paths(value, self.sources), {str(self.source), str(spaced)})

    def test_frozen_output_is_refused_before_writing(self):
        with self.assertRaisesRegex(ValueError, 'frozen'):
            probe.probe(self.manifest_path, '0' * 64, probe.ROOT / 'corpus/holdout/forbidden-probe-output')
        self.assertFalse((probe.ROOT / 'corpus/holdout/forbidden-probe-output').exists())

    def test_input_tree_output_is_rejected_before_any_files_are_written(self):
        write(self.manifest_path, self.manifest)
        output = self.sources / 'must-not-exist'
        with self.assertRaisesRegex(ValueError, 'outside hashed input roots'):
            probe.probe(self.manifest_path, probe.digest(self.manifest_path), output)
        self.assertFalse(output.exists())

    def test_reference_compiler_symlink_change_is_identity_drift(self):
        original = self.root / 'compiler-a'
        replacement = self.root / 'compiler-b'
        self.nvcc.rename(original)
        replacement.write_bytes(original.read_bytes())
        self.nvcc.symlink_to(original)
        def swap():
            self.nvcc.unlink()
            self.nvcc.symlink_to(replacement)
        self.after_syntax = swap
        report = self.execute()
        self.assertFalse(report['referenceDriverPathsUnchanged'])
        self.assertTrue(report['inputDriftDetected'])
        self.assertFalse(report['inputsUnchanged'])
        self.assertFalse(report['runComplete'])
        self.assertEqual(report['failures'][-1]['type'], 'ToolIdentityChanged')

    def test_one_missing_response_does_not_erase_valid_entries_or_denominator(self):
        self.entries[1]['arguments'].append('@missing.rsp')
        write(self.database, self.entries)
        self.bind_database()
        report = self.execute()
        self.assertEqual([index for index, _ in self.syntax_calls], [0])
        self.assertEqual([row['status'] for row in report['records']], ['passed', 'translation-error'])
        self.assertEqual(report['counts']['cudaCu'], 2)
        self.assertEqual(report['rates']['cudaCu'], .5)
        self.assertEqual(report['counts']['translationError'], 1)
        self.assertFalse(report['translationComplete'])

    def test_unrelated_baseline_file_cannot_certify_unbound_database_and_tu(self):
        write(self.baseline_path, {'schemaVersion': 1, 'files': {str(self.host): probe.digest(self.host)}})
        self.manifest['baselineInputManifest']['sha256'] = probe.digest(self.baseline_path)
        self.source.write_text('changed and never bound to the baseline\n')
        report = self.execute()
        self.assertFalse(report['baselineBindingVerified'])
        self.assertFalse(report['runComplete'])
        self.assertFalse(report['inputIntegrityComplete'])
        self.assertEqual(report['baselineUnboundInputs'], sorted(map(str, [self.database, self.source])))

    def test_current_header_snapshot_cannot_replace_cross_candidate_baseline_binding(self):
        self.extra_header = self.sources / 'unbound-project.h'
        self.extra_header.write_text('bytes are stable during run but were never baseline-bound')
        report = self.execute()
        self.assertTrue(report['inputsUnchanged'])
        self.assertFalse(report['baselineDependencyBindingComplete'])
        self.assertFalse(report['successfulSyntaxInputVerified'])
        self.assertFalse(report['inputIntegrityComplete'])
        self.assertEqual(report['records'][0]['dependencyEvidence'][0]['unboundBaselineFiles'], [str(self.extra_header)])

    def test_explicit_nvcc_host_compiler_must_be_registered(self):
        self.manifest['tools'].pop('gcc')
        report = self.execute()
        self.assertEqual(self.syntax_calls, [])
        self.assertFalse(report['toolIdentityComplete'])
        self.assertFalse(report['runComplete'])
        self.assertIn(str(self.gcc), report['records'][0]['error']['message'])

    def test_explicit_nvcc_host_compiler_byte_change_is_detected(self):
        self.after_syntax = lambda: self.gcc.write_text('changed host compiler')
        report = self.execute()
        self.assertFalse(report['runComplete'])
        self.assertTrue(report['inputDriftDetected'])
        self.assertIn(str(self.gcc), report['changedInputs'])

    def test_implicit_nvcc_host_compiler_is_unverified(self):
        for entry in self.entries:
            entry['arguments'] = [arg for arg in entry['arguments'] if not arg.startswith('-ccbin=')]
        write(self.database, self.entries)
        self.bind_database()
        report = self.execute()
        self.assertTrue(report['diagnosticComplete'])
        self.assertFalse(report['toolIdentityComplete'])
        self.assertFalse(report['runComplete'])

    def test_original_cu_suffix_controls_both_numerator_and_denominator_for_symlinks(self):
        physical = self.sources / 'physical.cpp'
        self.source.rename(physical)
        self.source.symlink_to(physical)
        baseline = json.loads(self.baseline_path.read_text())
        baseline['files'].pop(str(self.source))
        baseline['files'][str(physical)] = probe.digest(physical)
        write(self.baseline_path, baseline)
        self.manifest['baselineInputManifest']['sha256'] = probe.digest(self.baseline_path)
        self.bind_baseline_paths()
        report = self.execute()
        self.assertTrue(report['runComplete'], report['failures'])
        self.assertEqual(report['counts']['cudaCu'], 2)
        self.assertEqual(report['counts']['selectedCuPassed'], 2)
        self.assertEqual(report['rates']['cudaCu'], 1)

    def test_signal_classification_uses_time_diagnostic_not_exit_number(self):
        log = self.root / 'synthetic.log'
        timing = log.with_suffix('.rss')
        syntax = {'status': 'failed', 'returncode': 139, 'log': str(log)}
        self.assertEqual(probe.syntax_status(syntax), 'syntax-error')
        timing.write_text('Command terminated by signal 11\n1024\n')
        self.assertEqual(probe.syntax_status(syntax), 'signal/crash')
        self.assertEqual(syntax['signal'], 11)
        self.assertEqual(syntax['signalEvidence']['sha256'], probe.digest(timing))

    def test_project_header_symlink_retarget_is_not_hidden_by_resolved_file_set(self):
        left, right, alias = [self.sources / name for name in ('a.h', 'b.h', 'active.h')]
        left.write_text('header a')
        right.write_text('header b')
        alias.symlink_to(left)
        self.bind_baseline_paths()
        def swap():
            alias.unlink()
            alias.symlink_to(right)
        self.after_syntax = swap
        report = self.execute()
        self.assertEqual(report['changedInputs'], [])
        self.assertFalse(report['inputsUnchanged'])
        self.assertFalse(report['inputPathIdentitiesUnchanged'])
        self.assertTrue(report['inputDriftDetected'])
        self.assertFalse(report['runComplete'])

    def test_discovered_host_headers_only_affect_all_clang_phases(self):
        self.configure_header_discovery()
        self.extra_header = self.gcc_include / 'omp.h'
        report = self.execute()
        self.assertTrue(report['runComplete'], report['failures'])
        self.assertTrue(report['successfulSyntaxInputVerified'])
        mapping = report['clangIncludeAfter'][0]
        self.assertEqual(mapping['discoveryEvidence']['command'], [str(self.gcc), '-print-file-name=include'])
        self.assertEqual(mapping['baselineFileCount'], 2)
        for row in report['records']:
            self.assertEqual(row['translatedArguments'][-2:], ['-idirafter', str(self.gcc_include)])
            self.assertNotIn('-idirafter', row['referenceArguments'])
            self.assertNotIn('-idirafter', row['originalArguments'])
            self.assertTrue(row['hostHeaderDiscovery'][0]['applied'])
            self.assertEqual(row['hostHeaderDiscovery'][0]['originalHostCompiler']['resolvedPath'], str(self.gcc))
            for command in [row['syntax']['command'], row['dependencyEvidence'][1]['command']]:
                self.assertIn('-idirafter', command)
            self.assertNotIn('-idirafter', row['dependencyEvidence'][0]['command'])
        macro = report['records'][0]['macroEvidence']['commands']
        self.assertNotIn('-idirafter', macro[0]['command'])
        self.assertIn('-idirafter', macro[1]['command'])
        self.assertEqual(json.loads(self.database.read_text()), self.entries)

    def test_control_without_explicit_discovery_keeps_clang_arguments(self):
        report = self.execute()
        self.assertEqual(report['clangIncludeAfter'], [])
        self.assertTrue(all('-idirafter' not in row['translatedArguments'] for row in report['records']))

    def test_discovery_mapping_requires_explicit_v2_manifest(self):
        self.configure_header_discovery()
        self.manifest['schemaVersion'] = 1
        report = self.execute()
        self.assertIn('schemaVersion 2', report['failures'][0]['message'])
        self.assertEqual(self.syntax_calls, [])

    def test_discovery_directory_outside_input_roots_is_rejected(self):
        self.configure_header_discovery()
        self.manifest['inputRoots'].remove(str(self.gcc_include))
        report = self.execute()
        self.assertIn('covered by inputRoots', report['failures'][0]['message'])
        self.assertEqual(self.syntax_calls, [])

    def test_discovery_binds_even_unaccessed_files_in_header_directory(self):
        self.configure_header_discovery()
        baseline = json.loads(self.baseline_path.read_text())
        baseline['files'].pop(str(self.gcc_include / 'unaccessed.h'))
        write(self.baseline_path, baseline)
        self.manifest['baselineInputManifest']['sha256'] = probe.digest(self.baseline_path)
        report = self.execute()
        self.assertIn('files must match the baseline', report['failures'][0]['message'])
        self.assertEqual(self.syntax_calls, [])

    def test_actual_discovery_directory_mismatch_is_rejected(self):
        self.configure_header_discovery()
        self.discovery_output = str(self.sources) + '\n'
        report = self.execute()
        self.assertIn('does not match the registered directory', report['failures'][0]['message'])
        self.assertEqual(self.syntax_calls, [])

    def test_failed_discovery_preserves_diagnostic_and_does_not_execute_syntax(self):
        self.configure_header_discovery()
        self.discovery_failure = True
        report = self.execute()
        self.assertEqual(report['clangIncludeAfter'][0]['discoveryEvidence']['returncode'], 1)
        self.assertFalse(report['runComplete'])
        self.assertEqual(self.syntax_calls, [])

    def test_discovery_query_cannot_inject_other_driver_options(self):
        self.configure_header_discovery()
        self.manifest['clangIncludeAfter'][0]['discoveryCommand'].append('-D__NVCC__=1')
        report = self.execute()
        self.assertIn('discoveryCommand must query', report['failures'][0]['message'])
        self.assertEqual(self.syntax_calls, [])

    def test_discovery_query_compiler_must_match_registered_role(self):
        self.configure_header_discovery()
        self.manifest['clangIncludeAfter'][0]['discoveryCommand'][0] = str(self.clang)
        report = self.execute()
        self.assertIn('registered actual host compiler', report['failures'][0]['message'])
        self.assertEqual(self.syntax_calls, [])

    def test_discovery_evidence_hash_is_enforced(self):
        self.configure_header_discovery()
        self.evidence_file.write_text('different evidence bytes')
        report = self.execute()
        self.assertIn('supporting evidence changed', report['failures'][0]['message'])
        self.assertEqual(self.syntax_calls, [])

    def test_host_compiler_cannot_change_during_discovery_then_be_rebaselined(self):
        self.configure_header_discovery()
        self.after_discovery = lambda: self.gcc.write_text('changed during discovery')
        report = self.execute()
        self.assertIn('identity changed during header discovery', report['failures'][0]['message'])
        self.assertEqual(self.syntax_calls, [])

    def test_supporting_evidence_cannot_be_rebaselined_during_discovery(self):
        self.configure_header_discovery()
        self.after_discovery = lambda: self.evidence_file.write_text('changed during discovery')
        report = self.execute()
        self.assertFalse(report['runComplete'])
        self.assertFalse(report['inputsUnchanged'])
        self.assertTrue(report['inputDriftDetected'])
        self.assertIn(str(self.evidence_file), report['registeredInputPinMismatches'])
        self.assertEqual(self.syntax_calls, [])

    def test_other_registered_tool_cannot_be_rebaselined_during_discovery(self):
        self.configure_header_discovery()
        self.after_discovery = lambda: self.clang.write_text('different Clang during host discovery')
        report = self.execute()
        self.assertFalse(report['runComplete'])
        self.assertTrue(report['inputDriftDetected'])
        self.assertIn(str(self.clang), report['registeredInputPinMismatches'])
        self.assertEqual(self.syntax_calls, [])

    def test_supporting_evidence_alias_outside_input_roots_is_checked_after_run(self):
        self.configure_header_discovery()
        actual = self.root / 'original-evidence.json'
        other = self.root / 'replacement-evidence.json'
        self.evidence_file.rename(actual)
        other.write_text('different supporting evidence')
        self.evidence_file.symlink_to(actual)
        def swap():
            self.evidence_file.unlink()
            self.evidence_file.symlink_to(other)
        self.after_syntax = swap
        report = self.execute()
        self.assertFalse(report['runComplete'])
        self.assertFalse(report['inputsUnchanged'])
        self.assertTrue(report['inputDriftDetected'])
        self.assertFalse(report['supportingEvidencePathsUnchanged'])
        self.assertFalse(report['inputIntegrityComplete'])
        self.assertFalse(report['successfulSyntaxInputVerified'])
        self.assertEqual(report['changedInputs'], [])

    def test_discovery_profile_indices_apply_only_to_registered_rows(self):
        self.configure_header_discovery()
        self.manifest['clangIncludeAfter'][0]['originalIndices'] = [1]
        report = self.execute()
        self.assertTrue(report['runComplete'], report['failures'])
        self.assertNotIn('-idirafter', report['records'][0]['translatedArguments'])
        self.assertIn('-idirafter', report['records'][1]['translatedArguments'])

    def test_host_discovery_mapping_does_not_cross_compiler_identity(self):
        self.configure_header_discovery()
        other = self.root / 'g++-12'
        other.write_text('synthetic different compiler')
        self.manifest['tools']['other'] = {'path': str(other), 'sha256': probe.digest(other)}
        self.entries[1]['arguments'][1] = '-ccbin=' + str(other)
        write(self.database, self.entries)
        self.bind_database()
        report = self.execute()
        self.assertTrue(report['runComplete'], report['failures'])
        self.assertTrue(report['records'][0]['hostHeaderDiscovery'][0]['applied'])
        self.assertFalse(report['records'][1]['hostHeaderDiscovery'][0]['applied'])
        self.assertNotIn('-idirafter', report['records'][1]['translatedArguments'])

    def test_explicit_discovery_profile_with_wrong_host_is_translation_error(self):
        self.configure_header_discovery()
        self.manifest['clangIncludeAfter'][0]['originalIndices'] = [1]
        self.entries[1]['arguments'] = [str(self.clang), '-c', str(self.source)]
        write(self.database, self.entries)
        self.bind_database()
        report = self.execute()
        self.assertEqual([index for index, _ in self.syntax_calls], [0])
        self.assertEqual(report['records'][1]['status'], 'translation-error')
        self.assertEqual(report['counts']['cudaCu'], 2)

    def test_discovery_also_matches_registered_ordinary_cxx_host_driver(self):
        self.configure_header_discovery()
        self.manifest['target'] = 'rmm'
        self.entries = [{'file': str(self.host), 'directory': str(self.sources),
                         'arguments': [str(self.gcc), '-std=c++17', '-c', str(self.host)]}]
        write(self.database, self.entries)
        self.bind_database()
        report = self.execute()
        self.assertTrue(report['runComplete'], report['failures'])
        row = report['records'][0]
        self.assertTrue(row['hostHeaderDiscovery'][0]['applied'])
        self.assertEqual(row['hostHeaderDiscovery'][0]['originalHostCompiler']['resolvedPath'], str(self.gcc))
        self.assertNotIn('-idirafter', row['referenceArguments'])
        self.assertEqual(row['translatedArguments'][-2:], ['-idirafter', str(self.gcc_include)])

    def test_cross_round_header_alias_change_is_detected_against_baseline(self):
        left, right, alias = [self.sources / name for name in ('baseline-a.h', 'baseline-b.h', 'baseline-active.h')]
        left.write_text('header a')
        right.write_text('header b')
        alias.symlink_to(left)
        baseline = json.loads(self.baseline_path.read_text())
        baseline['files'].update({str(path): probe.digest(path) for path in (left, right)})
        write(self.baseline_path, baseline)
        self.bind_baseline_paths()
        self.extra_header = alias
        first = self.execute()
        self.assertTrue(first['runComplete'], first['failures'])
        self.assertTrue(first['successfulSyntaxInputVerified'])
        self.out = self.root / 'candidate-output'
        self.syntax_calls.clear()
        alias.unlink()
        alias.symlink_to(right)
        second = self.execute()
        self.assertFalse(second['runComplete'])
        self.assertFalse(second['baselineBindingVerified'])
        self.assertFalse(second['baselinePathIdentitiesVerified'])
        self.assertTrue(second['inputDriftDetected'])
        self.assertIn(str(alias), second['baselinePathIdentityMismatches'])
        self.assertEqual(self.syntax_calls, [])

    def test_baseline_missing_path_identity_inventory_is_not_full_binding(self):
        baseline = json.loads(self.baseline_path.read_text())
        baseline.pop('inputPathIdentities')
        baseline.pop('inputRoots')
        write(self.baseline_path, baseline)
        self.manifest['baselineInputManifest']['sha256'] = probe.digest(self.baseline_path)
        report = self.execute()
        self.assertTrue(report['diagnosticComplete'])
        self.assertFalse(report['baselinePathIdentitiesVerified'])
        self.assertFalse(report['runComplete'])

    def test_new_header_alias_between_rounds_is_not_ignored(self):
        alias = self.sources / 'new-alias.h'
        alias.symlink_to(self.host)
        report = self.execute()
        self.assertFalse(report['runComplete'])
        self.assertIn(str(alias), report['baselinePathIdentityMismatches'])
        self.assertEqual(self.syntax_calls, [])

    def test_malformed_baseline_path_inventory_still_writes_failure_report(self):
        baseline = json.loads(self.baseline_path.read_text())
        baseline['inputPathIdentities'] = ['invalid']
        write(self.baseline_path, baseline)
        self.manifest['baselineInputManifest']['sha256'] = probe.digest(self.baseline_path)
        report = self.execute()
        self.assertFalse(report['runComplete'])
        self.assertIn('invalid baseline input path identities', report['failures'][0]['message'])


if __name__ == '__main__':
    unittest.main()
