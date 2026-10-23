"""Prepare a frozen 30-window validation sample; infer and score in separate phases."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time

from screen import save, metric

PROJECT=Path('/nfs/dataset-ofs-661-142656/qiminhao/Foundation_model/agentic_TS/TSRAG')
ROOT=Path(__file__).resolve().parent


def prepare(run):
    if run.exists():raise ValueError('Run exists; do not relaunch')
    started=time.time()
    sys.path.insert(0,str(PROJECT/'research/multivariate_oracle_six'))
    from dataset import load_data,labels
    import pandas as pd
    values,_,info=load_data(PROJECT,'ETTh1')
    assert info['source_sha256']=='3dbe45a7a25cbf0eb6d82cb84ef4487ea2b73a9341e5ff0d082e340b9d943f1d'
    assert info['split']['train_end']==8640 and info['split']['val_end']==11520
    # Avoid the two previous development target intervals, and avoid target overlap
    # among this round's windows. History windows still overlap.
    starts=[8704+round(i*(11392-8704)/29) for i in range(30)]
    assert len(set(starts))==30 and all(b-a>=64 for a,b in zip(starts,starts[1:]))
    dates=pd.to_datetime(pd.read_csv(PROJECT/'artifacts/baseline_raw_v1/ETTh1.csv',usecols=['date'])['date'])
    assert dates.is_unique and dates.is_monotonic_increasing
    assert (dates.diff().dropna()==pd.Timedelta(hours=1)).all()
    run.mkdir(mode=0o700,parents=True)
    (run/'inputs').mkdir(mode=0o700)
    truth={};windows=[]
    for target in starts:
        x,y=values[target-512:target],values[target:target+64]
        wid='ETTh1-val-'+str(target)
        w=dict(window_id=wid,cutoff=dates.iloc[target-1].isoformat(),
               timestamps=[t.isoformat() for t in dates.iloc[target-512:target]],
               variables=info['variables'],history=x.tolist())
        save(run/'inputs'/(wid+'.json'),w)
        a,b=labels(x,y)
        truth[wid]=[dict(variable=n,L1=int(v1),L2=int(v2)) for n,v1,v2 in zip(info['variables'],a,b)]
        windows.append(wid)
    save(run/'truth.json',truth)
    save(run/'data_version.json',info)
    save(run/'budget.json',dict(start=started,deadline=started+1800,max_requests=30,concurrency=30,
                               max_tokens_per_request=16384,retries=0,gpu_count=0))
    save(run/'manifest.json',dict(dataset='ETTh1',split='validation',windows=windows,target_starts=starts,
          sampling='30 equispaced starts in [8704,11392], before inspecting labels; target intervals disjoint from two pilot windows',
          target_overlap=False,history_overlap=True,context=512,horizon=64,variables=info['variables'],
          host=socket.gethostname(),python=platform.python_version(),prompt='pro-concise frozen from pilot',
          model='aliyun/DeepSeek-v4-pro',code_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*.py')},
          status='screening, not final test',price=None))
    print(json.dumps(dict(prepared=True,windows=len(windows),first_target=starts[0],last_target=starts[-1])),flush=True)


def infer(run):
    budget=json.loads((run/'budget.json').read_text())
    remaining=int(budget['deadline']-time.time())
    if remaining<=0:raise RuntimeError('Original deadline reached')
    key=json.load(sys.stdin)['key']
    env=dict(os.environ,TOKEN_SERVICE_API_KEY=key,SSL_CERT_FILE='/etc/ssl/certs/ca-certificates.crt');del key
    command=[sys.executable,str(ROOT/'concurrent_server.py'),'--inputs',str(run/'inputs'),
             '--config',str(ROOT/'configs/dao_pro_concurrent.json'),'--output',str(run/'inference'),
             '--concurrency','30','--max-requests','30','--wall-seconds',str(remaining),'--send']
    result=subprocess.run(command,env=env)
    env.pop('TOKEN_SERVICE_API_KEY',None)
    save(run/'launcher_status.json',dict(returncode=result.returncode,finished_at=time.time()))
    return result.returncode


def score(run):
    from sklearn.metrics import f1_score,accuracy_score
    manifest=json.loads((run/'manifest.json').read_text());truth=json.loads((run/'truth.json').read_text())
    inference=json.loads((run/'inference/status.json').read_text())
    responses=inference['results'];metrics={};details=[];usage={};latencies=[]
    for r in responses.values():
        if r.get('seconds') is not None:latencies.append(r['seconds'])
        for k,v in (r.get('usage') or {}).items():
            if type(v) is int:usage[k]=usage.get(k,0)+v
    valid=sum(r.get('status')=='valid' for r in responses.values())
    for label in ('L1','L2'):
        actual=[];predicted=[];per={n:([],[]) for n in manifest['variables']};per_window={}
        for wid in manifest['windows']:
            r=responses.get(wid,{})
            found={v['variable']:v[label] for v in r.get('prediction',{}).get('labels',[])} if r.get('status')=='valid' else {}
            wy=[];wz=[]
            for row in truth[wid]:
                n=row['variable'];a=row[label];b=found.get(n)
                actual.append(a);predicted.append(b);wy.append(a);wz.append(b)
                per[n][0].append(a);per[n][1].append(b)
                details.append(dict(window=wid,variable=n,label=label,truth=a,predicted=b,correct=a==b))
            per_window[wid]=metric(wy,wz)
        m=metric(actual,predicted)
        present=[(a,b) for a,b in zip(actual,predicted) if b is not None]
        m.update(always_zero=metric(actual,[0]*len(actual)),per_variable={n:metric(*v) for n,v in per.items()},per_window=per_window,
                 valid_only=metric([a for a,b in present],[b for a,b in present]) if present else None)
        # Independent check on every full-coverage metric.
        if all(v is not None for v in predicted):
            assert abs(f1_score(actual,predicted,labels=[-1,0,1],average='macro',zero_division=0)-m['macro_f1'])<1e-12
            assert abs(accuracy_score(actual,predicted)-m['accuracy'])<1e-12
        metrics[label]=m
    attempts=list((run/'inference').glob('attempt-*.json'))
    times=[json.loads(p.read_text())['at'] for p in attempts]
    end=json.loads((run/'launcher_status.json').read_text())['finished_at']
    result=dict(complete=valid==30,valid_windows=valid,total_windows=30,valid_response_rate=valid/30,
                metrics=metrics,details=details,usage=usage,latency_seconds=latencies,
                wall_seconds=end-json.loads((run/'budget.json').read_text())['start'],
                admission_span_seconds=max(times)-min(times) if times else None,
                warning='Validation screening. Histories overlap. Missing predictions count as incorrect in coverage-inclusive metrics; valid_only also reported.',
                source_model=manifest['model'])
    save(run/'metrics.json',result)
    print(json.dumps({k:result[k] for k in ('complete','valid_windows','wall_seconds','usage','admission_span_seconds')}),flush=True)
    print(json.dumps({k:{f:v[f] for f in ('macro_f1','accuracy','invalid')} for k,v in metrics.items()}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','infer','score']);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();result={'prepare':prepare,'infer':infer,'score':score}[a.phase](a.run.resolve())
    if isinstance(result,int):raise SystemExit(result)
