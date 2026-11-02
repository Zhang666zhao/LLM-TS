"""Read-only SSH export of allowlisted historical metrics; never launches jobs.

Copies original bytes to a new local archive. No data, weights, arrays, API
requests/responses, environment dumps, or arbitrary logs are exported.
"""
import base64
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'benchmarks/oracle_v1/remote_archive_20261102'
REMOTE = r'''
import base64,hashlib,json
from pathlib import Path
root=Path('/nfs/dataset-ofs-661-142656/qiminhao/Foundation_model/agentic_TS/TSRAG/runs')
files={}; jobs=[]
allowed={'metrics.json','config.json','status.json','data_version.json','initial_hash.json',
 'best_validation.json','config_alignment.json','provenance.json','training_summary.json',
 'variable_metrics.json','horizon_metrics.json','git_commit.txt','train.jsonl'}
def add(p):
 if not p.is_file():return
 b=p.read_bytes()
 if len(b)>2000000:raise ValueError('Oversized artifact '+str(p))
 files[str(p.relative_to(root))]={'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()}
def job(p,kind):
 p=p.resolve()
 if any(j['source']==str(p.relative_to(root)) for j in jobs):return
 if not p.is_dir():raise ValueError('Missing job '+str(p))
 for required in ['metrics.json','status.json','config.json']:
  if not (p/required).is_file():raise ValueError('Missing '+required+' '+str(p))
 for n in allowed:add(p/n)
 jobs.append({'kind':kind,'source':str(p.relative_to(root))})
five=root/'20261015_five_multivariate_oracle_continue_v1'
reuse=json.loads((five/'reused.json').read_text())
for r in reuse:job(Path(r['source']),'five_oracle')
for dataset in ['ETTh1','ETTh2','ETTm1','ETTm2','weather']:
 for p in sorted((five/dataset/'jobs').iterdir()):
  if p.is_dir():job(p,'five_oracle')
assert len(jobs)==175 and len({j['source'] for j in jobs})==175
exchange=root/'20261014_exchange_multivariate_oracle_v1'
for p in sorted((exchange/'jobs').iterdir()):
 if p.is_dir():job(p,'exchange_oracle')
assert len(jobs)==210
old=root/'20260926_baselines_ctx512_pred64'
for method in ['patchtst','itransformer','frets','fredf']:
 for dataset in ['ETTh1','ETTh2','ETTm1','ETTm2','weather','exchange_rate','electricity']:
  attempt='attempt02' if (method,dataset)==('patchtst','electricity') else 'attempt01'
  job(old/'jobs'/f'{method}_{dataset}'/attempt,'original_backbone')
assert len(jobs)==238
for run in [five,exchange,root/'20261014_six_multivariate_oracle_v1',old]:
 for n in ['comparison.json','paired_checks.json','budget.json','status.json','reused.json',
           'planned_jobs.json','authorization.json','source_versions.json','result_status.json',
           'provenance.json','failures.json','contract_tests.json','continuation_authorization.json']:
  add(run/n)
print(json.dumps({'files':files,'jobs':jobs}))
'''

def main():
    if DEST.exists():raise SystemExit('Archive exists; refusing overwrite')
    data=json.loads(subprocess.check_output(['ssh','-o','ClearAllForwardings=yes','-o','BatchMode=yes',
        '-o','ConnectTimeout=10','GPU_Foundation','python3 -c '+shlex.quote(REMOTE)],text=True))
    decoded={}
    for name,record in data['files'].items():
        p=Path(name)
        assert not p.is_absolute() and '..' not in p.parts
        b=base64.b64decode(record['base64']);assert hashlib.sha256(b).hexdigest()==record['sha256']
        s=b.decode('utf-8')
        if re.search(r'ark-[a-z0-9-]{25,}|[A-Z2-7]{80,}={0,8}|-----BEGIN .*PRIVATE KEY-----',s):
            raise ValueError('Credential-like content; refusing export: '+name)
        decoded[name]=b
    # Compare the server comparison to the already published 175-row summary.
    rows=json.loads(decoded['20261015_five_multivariate_oracle_continue_v1/comparison.json'])
    previous=json.loads((ROOT/'benchmarks/oracle_v1/results/five_datasets.json').read_text())
    index={(r['dataset'],r['method'],r['strategy'],r['label']):r for r in rows}
    assert len(index)==175 and len(previous)==175
    for r in previous:
        current=index[(r['dataset'],r['method'],r['strategy'],r['label'])]
        for k in ('mse','mae','ot_mse','ot_mae'):assert abs(current[k]-r[k])<1e-12
    pairs=json.loads(decoded['20261015_five_multivariate_oracle_continue_v1/paired_checks.json'])
    assert len(pairs)==50 and all(r['initialization_match'] and r['common_epoch_orders_match'] for r in pairs)
    for name,b in decoded.items():
        p=DEST/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
    manifest={'date':'2026-11-02','source_host':'GPU_Foundation','jobs':data['jobs'],
        'verification':'SHA256 byte equality; 175 published summary rows agree; 50 recorded pair checks pass. No prediction-array recomputation.',
        'files':{n:{'sha256':data['files'][n]['sha256'],'bytes':len(b)} for n,b in decoded.items()}}
    (DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'jobs':len(data['jobs']),'files':len(decoded),'bytes':sum(map(len,decoded.values())),'destination':str(DEST)}))

if __name__=='__main__':main()
