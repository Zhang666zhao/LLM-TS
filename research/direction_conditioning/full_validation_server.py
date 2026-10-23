"""Fixed ETTh1 validation sweep. Server API inference and offline scoring are separate.
One request per uncached window, 30 concurrent, no automatic retries or tuning.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

import predict
from explore_server import pack, joint_valid
from finalize_server import PROMPT
from screen import save, metric

PROJECT=Path('/nfs/dataset-ofs-661-142656/qiminhao/Foundation_model/agentic_TS/TSRAG')
ROOT=Path(__file__).resolve().parent
CACHE=PROJECT/'runs/20261019_dao_etth1_c30_v1'


def atomic(path,value):
    temp=path.with_suffix('.tmp')
    fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    with os.fdopen(fd,'w') as f:json.dump(value,f,ensure_ascii=False,allow_nan=False)
    os.replace(temp,path)


def messages(payload):
    return [dict(role='system',content=PROMPT),dict(role='user',content=json.dumps(payload,separators=(',',':')))]


def prepare(run):
    if run.exists():raise ValueError('Run exists; refusing overwrite/relaunch')
    start=time.time()
    sys.path.insert(0,str(PROJECT/'research/multivariate_oracle_six'))
    from dataset import load_data,labels
    import pandas as pd
    values,_,info=load_data(PROJECT,'ETTh1')
    assert info['source_sha256']=='3dbe45a7a25cbf0eb6d82cb84ef4487ea2b73a9341e5ff0d082e340b9d943f1d'
    assert info['split']['val_start']==8128 and info['split']['val_end']==11520
    targets=list(range(8640,11457));assert len(targets)==2817
    dates=pd.to_datetime(pd.read_csv(PROJECT/'artifacts/baseline_raw_v1/ETTh1.csv',usecols=['date'])['date'])
    assert dates.is_unique and (dates.diff().dropna()==pd.Timedelta(hours=1)).all()
    cfg=json.loads((ROOT/'configs/dao_pro_concurrent.json').read_text())
    prior=json.loads((CACHE/'inference/manifest.json').read_text())
    assert prior['config']==cfg and prior['prompt']==PROMPT
    cached={j['id']:(i,j) for i,j in enumerate(prior['jobs'])}
    run.mkdir(parents=True,mode=0o700)
    for d in ('inputs','responses','attempts','code'): (run/d).mkdir(mode=0o700)
    (run/'code/configs').mkdir(mode=0o700)
    # Snapshot before launching so later working-copy edits cannot change the run.
    for f in ROOT.glob('*.py'):shutil.copy2(f,run/'code'/f.name)
    for f in (ROOT/'configs').glob('*.json'):shutil.copy2(f,run/'code/configs'/f.name)
    truth={};jobs=[];reused=[]
    for target in targets:
        wid='ETTh1-val-'+str(target);x=values[target-512:target];y=values[target:target+64]
        w=dict(window_id=wid,cutoff=dates.iloc[target-1].isoformat(),timestamps=[t.isoformat() for t in dates.iloc[target-512:target]],variables=info['variables'],history=x.tolist())
        digest=predict.digest(w)
        save(run/'inputs'/(str(target)+'.json'),w)
        l1,l2=labels(x,y)
        truth[wid]=[dict(variable=n,L1=int(a),L2=int(b)) for n,a,b in zip(info['variables'],l1,l2)]
        jobs.append(dict(id=wid,target=target,input_hash=digest))
        if wid in cached:
            i,j=cached[wid];assert digest==j['input_hash']
            request=json.loads((CACHE/'inference'/f'attempt-{i}.json').read_text())
            assert request['messages']==messages(pack(w))
            response=json.loads((CACHE/'inference'/f'response-{i}.json').read_text())
            assert response['status']=='valid' and response['raw_response']['choices'][0]['finish_reason']=='stop'
            prediction=predict.validate_response(json.dumps(response['prediction']),info['variables'])
            assert joint_valid(prediction,pack(w))
            save(run/'responses'/(str(target)+'.json'),dict(id=wid,target=target,status='valid',prediction=prediction,input_hash=digest,
                  reused=True,source=str(CACHE/'inference'/f'response-{i}.json'),source_sha256=hashlib.sha256((CACHE/'inference'/f'response-{i}.json').read_bytes()).hexdigest()))
            reused.append(wid)
    assert len(reused)==30
    save(run/'truth.json',truth);save(run/'data_version.json',info)
    save(run/'config.json',cfg)
    save(run/'budget.json',dict(start=start,deadline=start+21600,wall_seconds=21600,max_requests=2787,concurrency=30,retries=0,gpu_count=0,max_tokens=16384))
    save(run/'manifest.json',dict(dataset='ETTh1',split='full validation',context=512,horizon=64,stride=1,jobs=jobs,
         variables=info['variables'],reused=reused,total_windows=2817,new_requests=2787,prompt=PROMPT,
         host=socket.gethostname(),python=sys.version,code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (run/'code').glob('*.py')},
         source_data_code_sha256=hashlib.sha256((PROJECT/'research/multivariate_oracle_six/dataset.py').read_bytes()).hexdigest(),
         warning='Overlapping validation windows; not an independent test. Price unknown. No automatic retries.'))
    atomic(run/'status.json',dict(status='prepared',total=2817,reused=30,attempted=0,finished=0,valid=30,invalid=0))
    print(json.dumps(dict(prepared=True,total=2817,reused=30,new_requests=2787,deadline=start+21600)),flush=True)


def launch(run):
    budget=json.loads((run/'budget.json').read_text())
    if time.time()>=budget['deadline']:raise ValueError('Deadline passed')
    key=json.load(sys.stdin)['key']
    env=dict(os.environ,TOKEN_SERVICE_API_KEY=key,SSL_CERT_FILE='/etc/ssl/certs/ca-certificates.crt');del key
    # Reserve once before spawning. Never save credentials in launch records.
    save(run/'launch.json',dict(at=time.time(),command='snapshot full_validation_server.py run',deadline=budget['deadline']))
    fd=os.open(run/'launcher.log',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as log:
        child=subprocess.Popen([sys.executable,str(run/'code/full_validation_server.py'),'run','--run',str(run)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    env.pop('TOKEN_SERVICE_API_KEY',None)
    save(run/'pid.json',dict(pid=child.pid))
    print(json.dumps(dict(launched=True,pid=child.pid)),flush=True)


def infer(run):
    budget=json.loads((run/'budget.json').read_text());manifest=json.loads((run/'manifest.json').read_text());cfg=json.loads((run/'config.json').read_text())
    if not os.environ.get(cfg['api_key_env']):raise ValueError('Missing key')
    os.environ['SSL_CERT_FILE']='/etc/ssl/certs/ca-certificates.crt'
    deadline=budget['deadline'];todo=[j for j in manifest['jobs'] if j['id'] not in manifest['reused']]
    assert len(todo)==budget['max_requests']
    attempted=0;finished=0;valid=len(manifest['reused']);invalid=0;stopped=None;pending={};next_index=0
    def progress(state):
        atomic(run/'status.json',dict(status=state,total=2817,reused=30,attempted=attempted,finished=finished,valid=valid,invalid=invalid,
              in_flight=len(pending),pending=len(todo)-next_index,updated_at=time.time(),deadline=deadline,stop_reason=stopped))
    def worker(job):
        target=job['target'];remaining=deadline-time.time()
        w=json.loads((run/'inputs'/(str(target)+'.json')).read_text());assert predict.digest(w)==job['input_hash']
        payload=pack(w);msgs=messages(payload)
        save(run/'attempts'/(str(target)+'.json'),dict(id=job['id'],input_hash=job['input_hash'],at=time.time(),request_hash=predict.digest(dict(config=cfg,messages=msgs))))
        start=time.time();result=dict(id=job['id'],target=target,input_hash=job['input_hash'],reused=False,status='invalid',fatal=False)
        try:
            if remaining<=0:raise RuntimeError('Deadline reached')
            raw=predict.ChatCompletions().complete(dict(cfg,timeout_seconds=min(cfg['timeout_seconds'],remaining)),msgs)
            result.update(raw_response=raw,usage=raw.get('usage'))
            c=raw['choices'][0]
            if c.get('finish_reason')=='stop':
                pred=predict.validate_response(c['message']['content'],manifest['variables'])
                if joint_valid(pred,payload):result.update(status='valid',prediction=pred)
                else:result['error']='joint_label_conflict'
            else:result['error']='incomplete_response'
        except RuntimeError as error:
            result.update(status='failed',fatal=True,error=str(error))
        except (ValueError,KeyError,TypeError,IndexError) as error:
            result.update(status='invalid',error_type=type(error).__name__)
        except Exception as error:result.update(status='failed',fatal=True,error_type=type(error).__name__)
        result['seconds']=time.time()-start
        save(run/'responses'/(str(target)+'.json'),result)
        return result['status'],result['fatal']
    try:
        with ThreadPoolExecutor(max_workers=30) as pool:
            progress('running')
            while pending or (next_index<len(todo) and stopped is None):
                while stopped is None and len(pending)<30 and next_index<len(todo):
                    if time.time()>=deadline:stopped='deadline';break
                    if attempted>=budget['max_requests']:stopped='request_budget';break
                    j=todo[next_index];next_index+=1;attempted+=1
                    pending[pool.submit(worker,j)]=j['id']
                if not pending:break
                done,_=wait(pending,timeout=5,return_when=FIRST_COMPLETED)
                for f in done:
                    pending.pop(f);finished+=1
                    state,fatal=f.result()
                    if state=='valid':valid+=1
                    else:invalid+=1
                    if fatal:stopped='systemic_api_or_io_failure'
                if time.time()>=deadline:stopped='deadline'
                progress('running' if stopped is None else 'draining')
        progress('complete' if valid==2817 else ('finished_with_invalid' if finished==len(todo) else 'stopped'))
    except Exception as error:
        stopped=type(error).__name__;progress('failed');raise
    finally:os.environ.pop('TOKEN_SERVICE_API_KEY',None)
    # True future labels are first read in the separate scoring process.
    subprocess.run([sys.executable,str(Path(__file__).resolve()),'score','--run',str(run)],check=True)


def score(run):
    from sklearn.metrics import f1_score,accuracy_score
    manifest=json.loads((run/'manifest.json').read_text());truth=json.loads((run/'truth.json').read_text());status=json.loads((run/'status.json').read_text())
    records={}
    for p in (run/'responses').glob('*.json'):
        r=json.loads(p.read_text());records[r['id']]=r
    arrays={k:dict(y=[],p=[],per={n:([],[]) for n in manifest['variables']}) for k in ('L1','L2')}
    csv_path=run/'window_variable_results.csv';jsonl_path=run/'window_variable_results.jsonl'
    with csv_path.open('x',newline='') as cf,jsonl_path.open('x') as jf:
        os.chmod(csv_path,0o600);os.chmod(jsonl_path,0o600)
        fields=['window_id','target_start','variable','status','reused','L1_true','L1_pred','L1_correct','L2_true','L2_pred','L2_correct']
        writer=csv.DictWriter(cf,fieldnames=fields);writer.writeheader()
        for job in manifest['jobs']:
            wid=job['id'];r=records.get(wid,{});state=r.get('status','not_sent')
            preds={v['variable']:v for v in r.get('prediction',{}).get('labels',[])} if state=='valid' else {}
            for target in truth[wid]:
                n=target['variable'];row=dict(window_id=wid,target_start=job['target'],variable=n,status=state,reused=r.get('reused',False))
                for lab,a in arrays.items():
                    actual=target[lab];pred=preds.get(n,{}).get(lab)
                    a['y'].append(actual);a['p'].append(pred);a['per'][n][0].append(actual);a['per'][n][1].append(pred)
                    row.update({lab+'_true':actual,lab+'_pred':pred,lab+'_correct':actual==pred})
                writer.writerow(row);jf.write(json.dumps(row)+'\n')
    metrics={}
    for lab,a in arrays.items():
        m=metric(a['y'],a['p']);pairs=[(x,y) for x,y in zip(a['y'],a['p']) if y is not None]
        m.update(per_variable={n:metric(*v) for n,v in a['per'].items()},always_zero=metric(a['y'],[0]*len(a['y'])),
                 valid_only=metric([v[0] for v in pairs],[v[1] for v in pairs]) if pairs else None)
        if pairs:
            assert abs(f1_score([x for x,y in pairs],[y for x,y in pairs],labels=[-1,0,1],average='macro',zero_division=0)-m['valid_only']['macro_f1'])<1e-12
        metrics[lab]=m
    usage={}
    for r in records.values():
        if not r.get('reused'):
            for k,v in (r.get('usage') or {}).items():
                if type(v) is int:usage[k]=usage.get(k,0)+v
    result=dict(complete=status['valid']==2817,total_windows=2817,valid_windows=status['valid'],rows=19719,metrics=metrics,
                new_request_usage=usage,elapsed_seconds=time.time()-json.loads((run/'budget.json').read_text())['start'],
                note='Full validation stride1 with overlapping targets/history. Fixed3-class macroF1; invalid predictions counted incorrect in main metrics, valid_only also given.')
    save(run/'metrics.json',result)
    text=['# ETTh1 complete validation direction labels','',f"Status: {status['status']}; valid {status['valid']}/2817 windows;19719 window-variable rows.",'',
          '|Label|Macro-F1 (coverage-inclusive)|Accuracy|Invalid labels|Always0 F1|','|---|---:|---:|---:|---:|']
    for lab,m in metrics.items():text.append(f"|{lab}|{m['macro_f1']:.6f}|{m['accuracy']:.6f}|{m['invalid']}|{m['always_zero']['macro_f1']:.6f}|")
    text+=['','All judgments: window_variable_results.csv and .jsonl. Per-variable scores: metrics.json.',result['note']]
    (run/'summary.md').write_text('\n'.join(text))
    atomic(run/'completion.json',dict(at=time.time(),complete=result['complete'],status=status['status'],metrics_ready=True))
    print(json.dumps(dict(status=status['status'],valid=status['valid'],metrics={k:dict(macro_f1=v['macro_f1'],accuracy=v['accuracy']) for k,v in metrics.items()})),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','launch','run','score']);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();{'prepare':prepare,'launch':launch,'run':infer,'score':score}[a.phase](a.run.resolve())
