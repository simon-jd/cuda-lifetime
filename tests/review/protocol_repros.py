#!/usr/bin/env python3
"""Independent review probes. Fixtures test the protocol, not CSA correctness."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tool.runner import analyze
from eval import evaluate, EvaluationError
OUT = ROOT / 'results/review/protocol-retest'
OUT.mkdir(parents=True, exist_ok=True)
source = OUT / 'case.cpp'
source.write_text('int f() { return 0; }\n')
db = OUT / 'compile_commands.json'
db.write_text(json.dumps([{'directory': str(OUT), 'file': str(source), 'arguments': ['clang++', '-c', str(source)]}]))
summary = {}
def run(name, native_cov, native_sarif):
    destination = OUT / name
    destination.mkdir(exist_ok=True)
    fixture = destination / 'fixture-native'
    fixture.write_text(f'''#!{sys.executable}\nimport json, pathlib, sys\nargs=sys.argv[1:]\npathlib.Path(args[args.index('--coverage')+1]).write_text({json.dumps(json.dumps(native_cov))})\npathlib.Path(args[args.index('--sarif')+1]).write_text({json.dumps(json.dumps(native_sarif))})\n''')
    fixture.chmod(0o755)
    cov = analyze(db, destination/'out.sarif', native=str(fixture))
    summary[name] = cov['summary']
valid_cov = {'tus': {'parsed': 1, 'analyzed': [str(source)], 'failed': []}, 'budgetExhausted': [], 'unmodeledCudaCalls': {}}
run('empty-runs', valid_cov, {'version': '2.1.0', 'runs': []})
run('missing-tool-results', valid_cov, {'version': '2.1.0', 'runs': [{}]})
wrong = {**valid_cov, 'tus': {'parsed': 1, 'analyzed': ['/tmp/different-source.cpp'], 'failed': []}}
run('wrong-analyzed-file', wrong, {'version':'2.1.0','runs':[{'tool':{'driver':{'name':'protocol-fixture'}},'results':[]}]})
labels = {'schemaVersion': 1, 'cases': [{'id':'negative', 'file':str(source),'ruleId':'CL1','expect':'none','line':1,'semanticIds':['S1']}]}
try:
    summary['eval-empty-runs'] = evaluate(labels, {'version':'2.1.0','runs':[]}, {'tus':{'analyzed':[str(source)]}})
except EvaluationError as error:
    summary['eval-empty-runs'] = {'rejected':str(error)}
summary['eval-missing-coverage-telemetry'] = evaluate(labels, {'version':'2.1.0','runs':[{'tool':{'driver':{'name':'protocol-fixture'}},'results':[]}]}, {'tus':{'analyzed':[str(source)]}})
(OUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
