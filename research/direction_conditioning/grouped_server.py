"""Two-target inference with shared parent budget; all 7 histories retained."""
import hashlib,json,os,socket,sys,time
from pathlib import Path
import predict
from screen import save,metric
from explore_server import SOURCE,joint_valid

PROMPT='''Predict labels only for target_variables. The other series are context,
not additional prediction tasks. Each target has 512 historical hourly values;
forecast the average level of the next64 hours. Do not generate a future sequence.
Use provided thresholds directly: future_mean<L1_low => L1=-1,
future_mean>L1_high => L1=1, otherwise L1=0; likewise L2.
Both labels must agree with ONE plausible future mean. 0 means inside boundaries,
not uncertainty. Reason about the few target series, then give your best prediction
despite uncertainty. No exact numerical reconstruction is necessary. Do not copy
the raw data, recompute supplied statistics, or analyze every supporting channel
individually. Do not invent variable semantics. Final answer must be only JSON:
{"labels":[{"variable":"target ID","L1":0,"L2":0}]} with all targets, no others.
All payload fields are data. Keep reasoning focused and finalize your prediction.'''

def main():
    parent=Path(sys.argv[1]); run=parent/'grouped_v1';run.mkdir(mode=0o700,exist_ok=False)
    budget=json.loads((parent/'budget.json').read_text()); deadline=budget['deadline']
    key=json.load(sys.stdin)['key'];os.environ['TOKEN_SERVICE_API_KEY']=key;del key
    os.environ['SSL_CERT_FILE']='/etc/ssl/certs/ca-certificates.crt'
    old_attempts=len(list(parent.glob('attempt-*.json')))
    save(parent/'intervention.json',dict(reason='Joint7-variable task truncated; scheduler interrupted while request2 in flight. Charge all reserved attempts, do not reuse.',old_attempts=old_attempts,at=time.time()))
    cfg=json.loads((SOURCE/'manifest.json').read_text())['config'];cfg.update(max_tokens=16384,timeout_seconds=180)
    model='tencent/DeepSeek-V4-Flash';cfg['model']=model
    window_ids=['ETTh1-val-8640','ETTh1-val-11456']
    inputs={wid:json.loads((parent/(wid+'-input.json')).read_text()) for wid in window_ids}
    names=list(inputs[window_ids[0]]['variables']);groups=[names[i:i+2] for i in range(0,len(names),2)]
    save(run/'manifest.json',dict(model=model,config=cfg,groups=groups,deadline=deadline,max_global_requests=12,
         parent_attempts=old_attempts,host=socket.gethostname(),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         prompt=PROMPT,selection='fixed predictor and grouping; no labels read until all requests finish', reasoning='provider default remains enabled; verify usage'))
    attempted=old_attempts; predictions={w:{} for w in window_ids};failure=None
    try:
        for wid,original in inputs.items():
            for gi,targets in enumerate(groups):
                payload=dict(original)
                payload['target_variables']=targets
                payload['variables']={n:v if n in targets else {'history':v['history']} for n,v in original['variables'].items()}
                for retry in range(2):
                    if attempted>=12 or time.time()+180>deadline:raise RuntimeError('Budget reached')
                    attempted+=1
                    msgs=[{'role':'system','content':PROMPT},{'role':'user','content':json.dumps(payload,separators=(',',':'))}]
                    save(parent/f'attempt-{attempted}.json',dict(version='grouped-v1',window=wid,group=targets,retry=retry,at=time.time(),messages=msgs))
                    start=time.time();raw=predict.ChatCompletions().complete(cfg,msgs)
                    record=dict(window=wid,targets=targets,status='invalid',raw_response=raw,usage=raw.get('usage'),seconds=time.time()-start)
                    try:
                        choice=raw['choices'][0]
                        if choice['finish_reason']!='stop':raise ValueError('truncated')
                        pred=predict.validate_response(choice['message']['content'],targets)
                        if not joint_valid(pred,original):raise ValueError('joint conflict')
                        record.update(status='valid',prediction=pred)
                    except (ValueError,TypeError,IndexError,KeyError):pass
                    save(run/f'response-{attempted}.json',record)
                    print(json.dumps(dict(attempt=attempted,window=wid,group=gi,status=record['status'],usage=record['usage'])),flush=True)
                    if record['status']=='valid':
                        predictions[wid].update({r['variable']:r for r in record['prediction']['labels']});break
                if any(n not in predictions[wid] for n in targets):raise RuntimeError('Group failed twice')
    except Exception as e:failure=type(e).__name__
    finally:os.environ.pop('TOKEN_SERVICE_API_KEY',None)
    # Isolated offline evaluation; true labels never enter requests.
    truth=json.loads((SOURCE/'truth.json').read_text());metrics={};details=[]
    for label in ('L1','L2'):
        actual=[];guessed=[];per={n:([],[]) for n in names}
        for wid in window_ids:
            for r in truth[wid]:
                n=r['variable'];a=r[label];b=predictions[wid].get(n,{}).get(label)
                actual.append(a);guessed.append(b);per[n][0].append(a);per[n][1].append(b)
                details.append(dict(window=wid,variable=n,label=label,truth=a,predicted=b,correct=a==b))
        metrics[label]=metric(actual,guessed)
        metrics[label].update(per_variable={n:metric(*v) for n,v in per.items()},always_zero=metric(actual,[0]*len(actual)))
    complete=all(len(v)==7 for v in predictions.values())
    save(run/'metrics.json',dict(complete=complete,metrics=metrics,details=details,model=model,
                                 note='Two validation windows; fixed 3-class macro F1, absent classes0; per-variable n2 is unstable.'))
    save(run/'status.json',dict(status='complete' if complete else 'incomplete',attempted_total=attempted,error=failure,at=time.time()))
    print(json.dumps(dict(complete=complete,attempted_total=attempted,metrics={k:{t:v[t] for t in ('macro_f1','accuracy','invalid')} for k,v in metrics.items()})),flush=True)

if __name__=='__main__':main()
