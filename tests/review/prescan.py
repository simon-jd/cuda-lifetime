#!/usr/bin/env python3
"""Independent reachability pre-scan checks; included unused bodies are not roots."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--native',default='build/cuda-lifetime-native')
p.add_argument('--out',default='results/review/final-prescan')
a=p.parse_args();native=(ROOT/a.native).resolve();out=(ROOT/a.out).resolve();out.mkdir(parents=True,exist_ok=True)
expectations={'unused_header':(True,0),'used_header':(False,1),'local_destructor':(False,0),'virtual_call':(False,0),'macro_definition':(False,1),'global_destructor':(False,0),'template_instantiation':(False,None),'constructor_initializer':(False,0),'member_destructor':(False,0),'base_destructor':(False,0),'global_initializer':(False,None),'static_local_destructor':(False,None),'header_global_destructor':(False,None)}
expectations.update({'host_template_static':(True,0),'cuda_named_constructor':(True,0)})
summary={'binarySha256':hashlib.sha256(native.read_bytes()).hexdigest(),'cases':[]}
for name,(skipped,diagnostics) in expectations.items():
    source=ROOT/'tests/review/prescan'/(name+'.cpp')
    command=[str(native),str(source),'--sarif',str(out/(name+'.sarif')),'--coverage',str(out/(name+'-coverage.json')),'--','-std=c++17']
    run=subprocess.run(command,capture_output=True,text=True,cwd=ROOT);(out/(name+'.log')).write_text(run.stdout+run.stderr)
    read_errors=[]
    def read_output(path):
        try:
            return json.loads(path.read_text())
        except (OSError,ValueError) as error:
            read_errors.append({'file':str(path),'error':str(error)})
            return {}
    cov=read_output(out/(name+'-coverage.json'))
    sarif=read_output(out/(name+'.sarif'))
    count=sum(len(r['results']) for r in sarif.get('runs',[]))
    actual_skip=cov.get('astScan',{}).get('skippedNoCudaCalls')
    row={'name':name,'command':command,'returncode':run.returncode,'expectedSkip':skipped,'actualSkip':actual_skip,'expectedDiagnostics':diagnostics,'actualDiagnostics':count,'coverage':cov,'readErrors':read_errors}
    diagnostics_ok = count == diagnostics if diagnostics is not None else count > 0 or bool(cov.get('coverageUnknown'))
    modeled_ok = cov.get('modeledCallVisits', 0) > 0 if name in ('constructor_initializer','member_destructor','base_destructor','local_destructor') else True
    if name == 'global_destructor': modeled_ok = bool(cov.get('coverageUnknown')) or cov.get('modeledCallVisits', 0) > 0
    clean_skip = not cov.get('coverageUnknown') and not cov.get('unmodeledCudaCalls') if skipped else True
    row['passed']=run.returncode==0 and not read_errors and actual_skip==skipped and diagnostics_ok and modeled_ok and clean_skip and cov.get('tus',{}).get('parsed')==1 and not cov.get('tus',{}).get('failed')
    summary['cases'].append(row)
summary['binaryUnchanged']=summary['binarySha256']==hashlib.sha256(native.read_bytes()).hexdigest()
summary['passed']=summary['binaryUnchanged'] and all(row['passed'] for row in summary['cases'])
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'passed':summary['passed'],'cases':[{k:row[k] for k in ('name','passed','expectedSkip','actualSkip','expectedDiagnostics','actualDiagnostics')} for row in summary['cases']]},indent=2))
raise SystemExit(0 if summary['passed'] else 1)
