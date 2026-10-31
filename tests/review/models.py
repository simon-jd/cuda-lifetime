#!/usr/bin/env python3
"""Independent model-validation matrix, using clearly named synthetic summaries."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--native',default='build/cuda-lifetime-native')
p.add_argument('--out',default='results/review/final-models')
a=p.parse_args()
native=(ROOT/a.native).resolve(); out=(ROOT/a.out).resolve();out.mkdir(parents=True,exist_ok=True)
base={'schemaVersion':1,'models':[{'function':'review_wrapper','effect':'sync_device','args':{},'aliases':[], 'doc':{'url':'https://docs.nvidia.com/cuda/archive/13.2.2/cuda-runtime-api/group__CUDART__DEVICE.html','anchor':'cudaDeviceSynchronize','cudaVersion':'13.2.2'}}]}
variants={}
def variant(name, transform):
    model=copy.deepcopy(base);transform(model);variants[name]=model
variant('bad-schema', lambda m:m.update(schemaVersion='1'))
variant('empty-function', lambda m:m['models'][0].update(function=''))
variant('unsupported-effect', lambda m:m['models'][0].update(effect='silently_guess'))
variant('alias-not-array', lambda m:m['models'][0].update(aliases='bad'))
variant('empty-alias', lambda m:m['models'][0].update(aliases=['']))
variant('bad-assumption', lambda m:m['models'][0].update(userAssumption='true'))
variant('missing-documentation', lambda m:m['models'][0].pop('doc'))
variant('empty-documentation-url', lambda m:m['models'][0]['doc'].update(url=''))
variant('empty-documentation-anchor', lambda m:m['models'][0]['doc'].update(anchor=''))
variant('noninteger-argument', lambda m:m['models'][0].update(args={'stream':'0'}))
variant('missing-required-argument', lambda m:m['models'][0].update(effect='malloc_async',args={'ptr':0}))
variants['valid-disclosed-assumption']=base
summary={'binarySha256':hashlib.sha256(native.read_bytes()).hexdigest(),'cases':[]}
for name,model in variants.items():
    folder=out/name;folder.mkdir(exist_ok=True)
    model_folder=folder/'input';model_folder.mkdir(exist_ok=True)
    (model_folder/'models.json').write_text(json.dumps(model,indent=2)+'\n')
    command=[str(native),str(ROOT/'tests/review/model_repro.cpp'),'--model',str(model_folder),'--sarif',str(folder/'out.sarif'),'--coverage',str(folder/'coverage.json'),'--','-std=c++17']
    run=subprocess.run(command,capture_output=True,text=True,cwd=ROOT)
    (folder/'log').write_text(run.stdout+run.stderr)
    row={'name':name,'returncode':run.returncode,'stderr':run.stderr,'command':command}
    if name=='valid-disclosed-assumption':
        cov=json.loads((folder/'coverage.json').read_text()) if (folder/'coverage.json').exists() else {}
        row['passed']=run.returncode==0 and 'user model: review_wrapper' in cov.get('assumptions',[])
    else: row['passed']=run.returncode!=0
    summary['cases'].append(row)
summary['binaryUnchanged']=summary['binarySha256']==hashlib.sha256(native.read_bytes()).hexdigest()
summary['passed']=summary['binaryUnchanged'] and all(row['passed'] for row in summary['cases'])
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'passed':summary['passed'],'cases':[{k:v for k,v in row.items() if k in ('name','passed','returncode')} for row in summary['cases']]},indent=2))
raise SystemExit(0 if summary['passed'] else 1)
