"""Causal 32-window label reuse. Frozen Pro prompt; server-only inference."""
import argparse
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import csv,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path

import predict
from explore_server import pack,joint_valid
from finalize_server import PROMPT
from screen import save,metric
from full_validation_server import PROJECT,ROOT,atomic,messages


def anchor_for(target):
    if not 8640<=target<=11456:raise ValueError('Outside frozen validation')
    return 8640+32*((target-8640)//32)


def prepare(run):
    if run.exists():raise ValueError('Do not overwrite previous runs')
    start=time.time()
    sys.path.insert(0,str(PROJECT/'research/multivariate_oracle_six'))
    from dataset import load_data,labels
    import pandas as pd
    values,_,info=load_data(PROJECT,'ETTh1')
    assert info['source_sha256']=='3dbe45a7a25cbf0eb6d82cb84ef4487ea2b73a9341e5ff0d082e340b9d943f1d'
    assert info['split']['train_end']==8640 and info['split']['val_end']==11520
    dates=pd.to_datetime(pd.read_csv(PROJECT/'artifacts/baseline_raw_v1/ETTh1.csv',usecols=['date'])['date'])
    assert dates.is_unique and dates.is_monotonic_increasing and (dates.diff().dropna()==pd.Timedelta(hours=1)).all()
    cfg=json.loads((ROOT/'configs/dao_pro_concurrent.json').read_text())
    run.mkdir(parents=True,mode=0o700)
    for d in ('inputs','responses','attempts','code'):(run/d).mkdir(mode=0o700)
    (run/'code/configs').mkdir(mode=0o700)
    for f in ROOT.glob('*.py'):shutil.copy2(f,run/'code'/f.name)
    for f in (ROOT/'configs').glob('*.json'):shutil.copy2(f,run/'code/configs'/f.name)
    # Build candidate cache references. Match exact prompt, request payload and model
    # config before reusing any result, rather than assuming matching window IDs suffice.
    cache={}
    old=PROJECT/'runs/20261019_dao_etth1_c30_v1'
    pm=json.loads((old/'inference/manifest.json').read_text())
    assert pm['config']==cfg and pm['prompt']==PROMPT
    for i,j in enumerate(pm['jobs']):cache[j['id']]=(old/'inference'/f'attempt-{i}.json',old/'inference'/f'response-{i}.json')
    pilot=PROJECT/'runs/20261019_dao_compact_f1'
    fm=json.loads((pilot/'concise_v1/manifest.json').read_text())
    # Pilot manifest retained a default model; its actual request overrides it.
    assert dict(fm['config'],model=fm['candidates'][0][1])==cfg and fm['candidates'][0][2]==PROMPT
    for i,target in ((7,8640),(8,11456)):cache['ETTh1-val-'+str(target)]=(pilot/f'attempt-{i}.json',pilot/'concise_v1'/f'response-{i}.json')
    truth={};window_map=[]
    for target in range(8640,11457):
        wid='ETTh1-val-'+str(target);x=values[target-512:target];y=values[target:target+64]
        a,b=labels(x,y)
        truth[wid]=[dict(variable=n,L1=int(v1),L2=int(v2)) for n,v1,v2 in zip(info['variables'],a,b)]
        window_map.append(dict(id=wid,target=target,anchor=anchor_for(target),age=target-anchor_for(target)))
    anchors=[];reused=[]
    for target in range(8640,11457,32):
        wid='ETTh1-val-'+str(target)
        w=dict(window_id=wid,cutoff=dates.iloc[target-1].isoformat(),timestamps=[t.isoformat() for t in dates.iloc[target-512:target]],variables=info['variables'],history=values[target-512:target].tolist())
        predict.evidence(w,cfg);digest=predict.digest(w);save(run/'inputs'/(str(target)+'.json'),w)
        anchors.append(dict(id=wid,target=target,input_hash=digest))
        if wid in cache:
            request_path,response_path=cache[wid]
            request=json.loads(request_path.read_text());r=json.loads(response_path.read_text())
            assert request.get('model',cfg['model'])==cfg['model']
            assert request['messages']==messages(pack(w))
            assert r['status']=='valid' and r['raw_response']['choices'][0]['finish_reason']=='stop'
            assert r.get('model',cfg['model'])==cfg['model']
            pred=predict.validate_response(json.dumps(r['prediction']),info['variables'])
            assert joint_valid(pred,pack(w))
            save(run/'responses'/(str(target)+'.json'),dict(id=wid,target=target,status='valid',prediction=pred,reused=True,
                 source=str(response_path),source_sha256=hashlib.sha256(response_path.read_bytes()).hexdigest(),input_hash=digest))
            reused.append(wid)
    assert len(anchors)==89 and len(window_map)==2817
    save(run/'truth.json',truth);save(run/'data_version.json',info);save(run/'config.json',cfg)
    save(run/'budget.json',dict(start=start,deadline=start+1800,max_requests=89-len(reused),concurrency=30,retries=0,max_tokens=16384,gpu_count=0))
    save(run/'manifest.json',dict(dataset='ETTh1',split='validation',history=512,horizon=64,window_stride=1,label_reuse_stride=32,
           anchors=anchors,window_map=window_map,reused=reused,variables=info['variables'],prompt=PROMPT,
           warning='Each first-window prediction is carried forward causally for up to31 windows. True labels remain window-specific. Overlapping evaluation samples.',
           code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (run/'code').glob('*.py')}))
    atomic(run/'status.json',dict(status='prepared',total_anchors=89,reused=len(reused),attempted=0,finished=0,valid=len(reused),invalid=0))
    print(json.dumps(dict(prepared=True,anchors=89,windows=2817,reused=len(reused),new_requests=89-len(reused))),flush=True)


def launch(run):
    b=json.loads((run/'budget.json').read_text());assert time.time()<b['deadline']
    key=json.load(sys.stdin)['key'];env=dict(os.environ,TOKEN_SERVICE_API_KEY=key,SSL_CERT_FILE='/etc/ssl/certs/ca-certificates.crt');del key
    save(run/'launch.json',dict(at=time.time(),command='snapshot stride32_server.py run'))
    fd=os.open(run/'launcher.log',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as log:
        child=subprocess.Popen([sys.executable,str(run/'code/stride32_server.py'),'run','--run',str(run)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    env.pop('TOKEN_SERVICE_API_KEY',None);save(run/'pid.json',dict(pid=child.pid));print(json.dumps(dict(pid=child.pid)),flush=True)


def resume_prepare(run,source,send_interval=10):
    """One recorded continuation; exact successes reused, original deadline retained."""
    if run.exists():raise ValueError('Continuation exists')
    old_status=json.loads((source/'status.json').read_text())
    if old_status['status'] in ('running','draining','prepared'):raise ValueError('Wait for previous run to drain')
    m=json.loads((source/'manifest.json').read_text());b=json.loads((source/'budget.json').read_text());cfg=json.loads((source/'config.json').read_text())
    assert m['prompt']==PROMPT and time.time()<b['deadline']
    run.mkdir(parents=True,mode=0o700)
    for d in ('responses','attempts','code'):(run/d).mkdir(mode=0o700)
    (run/'code/configs').mkdir(mode=0o700)
    for p in ROOT.glob('*.py'):shutil.copy2(p,run/'code'/p.name)
    for p in (ROOT/'configs').glob('*.json'):shutil.copy2(p,run/'code/configs'/p.name)
    (run/'inputs').symlink_to(source/'inputs',target_is_directory=True)
    for n in ('truth.json','data_version.json'):(run/n).symlink_to(source/n)
    save(run/'config.json',cfg)
    reused=[];prior_usage=dict(m.get('prior_new_usage',{}));failed=[]
    for p in (source/'responses').glob('*.json'):
        r=json.loads(p.read_text())
        if not r.get('reused'):
            for k,v in (r.get('usage') or {}).items():
                if type(v) is int:prior_usage[k]=prior_usage.get(k,0)+v
        if r['status']=='valid':
            w=json.loads((source/'inputs'/p.name).read_text());assert predict.digest(w)==r['input_hash']
            pred=predict.validate_response(json.dumps(r['prediction']),m['variables']);assert joint_valid(pred,pack(w))
            save(run/'responses'/p.name,dict(id=r['id'],target=r['target'],status='valid',prediction=pred,input_hash=r['input_hash'],
                  reused=True,source=str(p),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
            reused.append(r['id'])
        else:failed.append(r['id'])
    old_attempted=m.get('prior_attempted',0)+len(list((source/'attempts').glob('*.json')))
    remaining=89-len(reused)
    # At most89 new API calls across both executions, allowing <=4 repairs beyond
    # the85 uncached anchors. No deadline extension, no repeated successful calls.
    assert old_attempted+remaining<=89
    m.update(reused=reused,continuation_of=str(source),prior_attempted=old_attempted,prior_new_usage=prior_usage,
             repaired_anchor_ids=failed,send_interval_seconds=send_interval,code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (run/'code').glob('*.py')})
    save(run/'manifest.json',m)
    save(run/'budget.json',dict(b,max_requests=remaining,send_interval_seconds=send_interval,total_new_request_cap=89,prior_attempted=old_attempted))
    atomic(run/'status.json',dict(status='prepared',reused=len(reused),valid=len(reused),attempted=0,finished=0,invalid=0,total_anchors=89))
    print(json.dumps(dict(continuation_prepared=True,reused=len(reused),remaining=remaining,prior_attempts=old_attempted,deadline=b['deadline'])),flush=True)


def infer(run):
    m=json.loads((run/'manifest.json').read_text());b=json.loads((run/'budget.json').read_text());cfg=json.loads((run/'config.json').read_text())
    if not os.environ.get(cfg['api_key_env']):raise ValueError('Missing key')
    os.environ['SSL_CERT_FILE']='/etc/ssl/certs/ca-certificates.crt'
    todo=[j for j in m['anchors'] if j['id'] not in m['reused']];assert len(todo)==b['max_requests']
    attempted=0;finished=0;valid=len(m['reused']);invalid=0;pending={};index=0;stop=None;next_send=0
    def progress(state):
        atomic(run/'status.json',dict(status=state,total_anchors=89,total_windows=2817,reused=len(m['reused']),attempted=attempted,
             finished=finished,valid=valid,invalid=invalid,in_flight=len(pending),pending=len(todo)-index,deadline=b['deadline'],updated_at=time.time(),stop_reason=stop))
    def worker(j):
        w=json.loads((run/'inputs'/(str(j['target'])+'.json')).read_text());assert predict.digest(w)==j['input_hash']
        payload=pack(w);msgs=messages(payload)
        save(run/'attempts'/(str(j['target'])+'.json'),dict(id=j['id'],input_hash=j['input_hash'],request_hash=predict.digest(dict(config=cfg,messages=msgs)),at=time.time()))
        start=time.time();r=dict(id=j['id'],target=j['target'],status='invalid',reused=False,input_hash=j['input_hash'],fatal=False)
        try:
            remaining=b['deadline']-time.time()
            if remaining<=0:raise RuntimeError('deadline')
            raw=predict.ChatCompletions().complete(dict(cfg,timeout_seconds=min(remaining,cfg['timeout_seconds'])),msgs)
            r.update(raw_response=raw,usage=raw.get('usage'));c=raw['choices'][0]
            if c.get('finish_reason')=='stop':
                pred=predict.validate_response(c['message']['content'],m['variables'])
                if joint_valid(pred,payload):r.update(status='valid',prediction=pred)
                else:r['error']='anchor_joint_inconsistency'
            else:r['error']='incomplete_response'
        except RuntimeError as e:r.update(status='failed',fatal=True,error=str(e))
        except (ValueError,KeyError,IndexError,TypeError) as e:r['error_type']=type(e).__name__
        except Exception as e:r.update(status='failed',fatal=True,error_type=type(e).__name__)
        r['seconds']=time.time()-start;save(run/'responses'/(str(j['target'])+'.json'),r)
        return r['status'],r['fatal']
    try:
        with ThreadPoolExecutor(max_workers=30) as pool:
            while pending or (index<len(todo) and stop is None):
                while len(pending)<30 and index<len(todo) and stop is None:
                    if time.time()>=b['deadline']:stop='deadline';break
                    if time.time()<next_send:break
                    j=todo[index];index+=1;attempted+=1;pending[pool.submit(worker,j)]=j['id']
                    next_send=time.time()+b.get('send_interval_seconds',0)
                progress('running' if stop is None else 'draining')
                if not pending:
                    if index<len(todo) and stop is None:
                        time.sleep(min(1,max(0,next_send-time.time())));continue
                    break
                timeout=min(5,max(.05,next_send-time.time())) if index<len(todo) and len(pending)<30 and stop is None else 5
                done,_=wait(pending,timeout=timeout,return_when=FIRST_COMPLETED)
                for f in done:
                    pending.pop(f);finished+=1;state,fatal=f.result()
                    if state=='valid':valid+=1
                    else:invalid+=1
                    if fatal:stop='systemic_api_failure'
            progress('complete' if valid==89 else ('finished_with_invalid' if finished==len(todo) else 'stopped'))
    except Exception as e:stop=type(e).__name__;progress('failed');raise
    finally:os.environ.pop('TOKEN_SERVICE_API_KEY',None)
    subprocess.run([sys.executable,str(Path(__file__).resolve()),'score','--run',str(run)],check=True)


def score(run):
    from sklearn.metrics import f1_score,accuracy_score
    m=json.loads((run/'manifest.json').read_text());truth=json.loads((run/'truth.json').read_text());status=json.loads((run/'status.json').read_text())
    records={int(p.stem):json.loads(p.read_text()) for p in (run/'responses').glob('*.json')}
    arrays={lab:dict(y=[],p=[],ay=[],ap=[],anchor_true=[],per={n:([],[]) for n in m['variables']}) for lab in ('L1','L2')}
    csv_path=run/'window_variable_results.csv';jl_path=run/'window_variable_results.jsonl'
    fields=['window_id','target_start','anchor_window_id','anchor_target_start','label_age_steps','variable','status','cached_anchor','L1_true','L1_pred','L1_correct','L2_true','L2_pred','L2_correct']
    with csv_path.open('x',newline='') as cf,jl_path.open('x') as jf:
        os.chmod(csv_path,0o600);os.chmod(jl_path,0o600);writer=csv.DictWriter(cf,fieldnames=fields);writer.writeheader()
        for j in m['window_map']:
            r=records.get(j['anchor'],{});state=r.get('status','not_sent');anchor_id='ETTh1-val-'+str(j['anchor'])
            preds={v['variable']:v for v in r.get('prediction',{}).get('labels',[])} if state=='valid' else {}
            at={v['variable']:v for v in truth[anchor_id]}
            for t in truth[j['id']]:
                n=t['variable'];row=dict(window_id=j['id'],target_start=j['target'],anchor_window_id=anchor_id,anchor_target_start=j['anchor'],label_age_steps=j['age'],variable=n,status=state,cached_anchor=r.get('reused',False))
                for lab,a in arrays.items():
                    y=t[lab];p=preds.get(n,{}).get(lab);a['y'].append(y);a['p'].append(p);a['anchor_true'].append(at[n][lab]);a['per'][n][0].append(y);a['per'][n][1].append(p)
                    if j['age']==0:a['ay'].append(y);a['ap'].append(p)
                    row.update({lab+'_true':y,lab+'_pred':p,lab+'_correct':y==p})
                writer.writerow(row);jf.write(json.dumps(row)+'\n')
    metrics={}
    for lab,a in arrays.items():
        r=metric(a['y'],a['p']);pairs=[(x,y) for x,y in zip(a['y'],a['p']) if y is not None]
        r.update(per_variable={n:metric(*v) for n,v in a['per'].items()},anchor_only=metric(a['ay'],a['ap']),
                 always_zero=metric(a['y'],[0]*len(a['y'])),true_label_carry_agreement=sum(x==y for x,y in zip(a['y'],a['anchor_true']))/len(a['y']),
                 valid_only=metric([x for x,y in pairs],[y for x,y in pairs]) if pairs else None)
        if pairs:
            assert abs(f1_score([x for x,y in pairs],[y for x,y in pairs],labels=[-1,0,1],average='macro',zero_division=0)-r['valid_only']['macro_f1'])<1e-12
            assert abs(accuracy_score([x for x,y in pairs],[y for x,y in pairs])-r['valid_only']['accuracy'])<1e-12
        metrics[lab]=r
    usage={}
    for r in records.values():
        if not r.get('reused'):
            for k,v in (r.get('usage') or {}).items():
                if type(v) is int:usage[k]=usage.get(k,0)+v
    result=dict(complete=status['valid']==89,valid_anchors=status['valid'],total_anchors=89,total_windows=2817,rows=19719,
               metrics=metrics,new_request_usage=usage,combined_new_request_usage={k:usage.get(k,0)+m.get('prior_new_usage',{}).get(k,0) for k in set(usage)|set(m.get('prior_new_usage',{}))},elapsed_seconds=time.time()-json.loads((run/'budget.json').read_text())['start'],
               note='Every32 windows reuse first-window labels causally. Each window scored against own true label. Fixed3class macroF1. Overlapping validation samples.')
    save(run/'metrics.json',result);atomic(run/'completion.json',dict(at=time.time(),complete=result['complete'],metrics_ready=True))
    print(json.dumps(dict(complete=result['complete'],metrics={k:{f:v[f] for f in ('macro_f1','accuracy','invalid')} for k,v in metrics.items()})),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','resume-prepare','launch','run','score']);p.add_argument('--run',type=Path,required=True);p.add_argument('--source',type=Path)
    a=p.parse_args()
    if a.phase=='resume-prepare':resume_prepare(a.run.resolve(),a.source.resolve())
    else:{'prepare':prepare,'launch':launch,'run':infer,'score':score}[a.phase](a.run.resolve())
