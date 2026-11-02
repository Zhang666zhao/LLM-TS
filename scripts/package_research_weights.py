"""Package explicitly inventoried historical state_dicts, never datasets.

Read-only CPU inspection on server; binary files stay in ignored artifacts/.
No training, GPU work, checkpoint conversion, or automatic upload.
"""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/weights-release-20261102'
RUNS='/nfs/dataset-ofs-661-142656/qiminhao/Foundation_model/agentic_TS/TSRAG/runs'
PY='/nfs/dataset-ofs-661-142656/qiminhao/Env/tsrag_baselines_20260926/bin/python'

def ssh(code, python='python3', **kw):
    return subprocess.run(['ssh','-o','ClearAllForwardings=yes','-o','BatchMode=yes',
        '-o','ConnectTimeout=10','GPU_Foundation',shlex.quote(python)+' -c '+shlex.quote(code)],check=True,**kw)

def main():
    resume='--resume' in sys.argv
    if OUT.exists() and not resume:raise SystemExit('Output exists; inspect and use --resume explicitly')
    archive=ROOT/'benchmarks/oracle_v1/remote_archive_20261102'
    jobs=json.loads((archive/'manifest.json').read_text())['jobs']
    records=[]
    for j in jobs:
        job=j['source']; c=json.loads((archive/job/'config.json').read_text())
        if j['kind']=='original_backbone':
            method,dataset=Path(job).parent.name.split('_',1)
        else:
            dataset='exchange_rate' if j['kind']=='exchange_oracle' else job.split('/')[1]
        source=job+'/best.pth'
        expected=None
        if 'patchtst_electricity/attempt02' in job:
            source=job.replace('attempt02','attempt01')+'/best.pth'
            expected=c['checkpoint_sha256']
        records.append(dict(result_job=job,checkpoint=source,dataset=dataset,kind=j['kind'],expected_sha256=expected,
                            config='benchmarks/oracle_v1/remote_archive_20261102/'+job+'/config.json'))
    for d in ['ETTh1','ETTh2','ETTm1','ETTm2','weather','exchange_rate','electricity']:
        job='20260922_DLinear_shared_ctx512_pred64/'+d
        records.append(dict(result_job=job,checkpoint=job+'/checkpoints/best.pth',dataset=d,kind='original_dlinear',
                            expected_sha256=None,config='results/benchmark_v1/source/dlinear/'+d+'/config.json'))
    code='''import json,hashlib,torch
from pathlib import Path
torch.set_num_threads(1)
root=Path(RUNS)
rows=json.loads(RECORDS)
for r in rows:
 p=root/r['checkpoint']; b=p.read_bytes(); sha=hashlib.sha256(b).hexdigest()
 if r['expected_sha256'] is not None:assert sha==r['expected_sha256']
 state=torch.load(p,map_location='cpu',weights_only=True)
 assert isinstance(state,dict) and len(state)>0
 assert all(isinstance(k,str) and isinstance(v,torch.Tensor) for k,v in state.items()), str(p)
 assert all(torch.isfinite(v).all().item() for v in state.values()),str(p)
 r.update(sha256=sha,bytes=len(b),tensor_count=len(state),tensor_elements=sum(v.numel() for v in state.values()),format='torch state_dict; tensor values only',weights_only_verified=True)
 del state,b
print(json.dumps(rows))
'''.replace('RUNS',repr(RUNS)).replace('RECORDS',repr(json.dumps(records)))
    if resume:
        rows=json.loads((OUT/'weights_manifest.json').read_text())['weights']
    else:
        r=ssh(code,PY,capture_output=True,text=True)
        rows=json.loads(r.stdout);assert len(rows)==245
        OUT.mkdir(parents=True)
        (OUT/'weights_manifest.json').write_text(json.dumps(dict(result_commit='52ac699d028a086512efeef2abc73c2355722acf',weights=rows),indent=2)+'\n')
    bundles=[]
    for dataset in sorted({r['dataset'] for r in rows}):
        selected=[r for r in rows if r['dataset']==dataset]
        filename='weights-'+dataset+'-20261102.tar'
        code='''import tarfile,sys,json
from pathlib import Path
root=Path(RUNS)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as out:
 for r in json.loads(RECORDS):out.add(root/r['checkpoint'],arcname=r['checkpoint'],recursive=False)
'''.replace('RUNS',repr(RUNS)).replace('RECORDS',repr(json.dumps([{'checkpoint':r['checkpoint']} for r in selected])))
        p=OUT/filename
        if p.exists() and p.stat().st_size==0:
            p.rename(p.with_suffix('.tar.interrupted'))
        if not p.exists():
            with p.open('xb') as f:ssh(code,stdout=f)
        expected={r['checkpoint']:r for r in selected}
        with tarfile.open(p) as t:
            members=t.getmembers();assert len(members)==len(expected)
            for m in members:
                assert m.isfile() and m.name in expected
                h=hashlib.sha256()
                with t.extractfile(m) as f:
                    for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
                assert h.hexdigest()==expected[m.name]['sha256']
        h=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
        assert p.stat().st_size<2*1024**3
        bundles.append(dict(name=filename,sha256=h.hexdigest(),bytes=p.stat().st_size,checkpoints=len(selected)))
        print(json.dumps(bundles[-1]),flush=True)
    (OUT/'bundles.json').write_text(json.dumps(bundles,indent=2)+'\n')
    (OUT/'SHA256SUMS').write_text(''.join(r['sha256']+'  '+r['name']+'\n' for r in bundles))

if __name__=='__main__':main()
