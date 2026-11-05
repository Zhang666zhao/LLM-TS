"""Separate validation, LLM proposal, freeze, and held-out evaluation stages."""
import argparse,csv,time
from ark_client import request_proposal
from forecast import *

ROOT=Path(__file__).resolve().parent
def dataset(csv_path,end):
    # Read only the requested prefix. Training, proposal and test gates use different end rows.
    with Path(csv_path).open() as f:
        r=csv.DictReader(f)
        if r.fieldnames!=['date']+COLS:raise ValueError('Wrong schema or column order')
        rows=[next(r) for _ in range(end)]
    raw=np.array([[float(row[k]) for k in COLS] for row in rows])
    times=stamp([row['date'] for row in rows])
    if not np.all(np.diff(times)==15) or not np.isfinite(raw).all():raise ValueError('ETTm1 data contract')
    return raw,times
def target(z,origins):
    return z[origins[:,None]+np.arange(H)]
def metric(p,y):
    err=p.astype('float64')-y
    return {'mse':float(np.mean(err*err)),'mae':float(np.mean(np.abs(err)))}
def risk(p,y):
    return ((p.astype('float64')-y)**2).mean(axis=0)
def schema_action(parent,var,source,alpha):
    return dict(kind='mix',parent=parent,variable=var,source=source,alpha=float(alpha))

def select_map(candidates,risks,b,shortlists=None):
    sources=sorted(candidates)
    return dict(kind='select',block=b,map=[[min(shortlists[j] if shortlists else sources,
        key=lambda s:(risks[s][h:h+b,j].mean(),s)) for h in range(0,H,b)] for j in range(D)])

def score_tools(out,csv_path):
    fit=read(out/'fit.json');cfg=fit['contract']
    if digest(out/'tools.npz')!=fit['state_sha256'] or digest(csv_path)!=cfg['sha256']:raise ValueError('Artifact mismatch')
    raw,times=dataset(csv_path,cfg['validation_end'])
    z=((raw-fit['mean'])/fit['std']).astype('float32');orig=np.arange(cfg['train_end'],cfg['validation_end']-H+1)
    with np.load(out/'tools.npz') as s:bank=ToolBank(dict(s),cfg['steps_per_day'])
    store=out/'validation_cache';store.mkdir(exist_ok=False)
    matrices={k:np.lib.format.open_memmap(store/(k+'.npy'),mode='w+',dtype='float32',shape=(len(orig),H,D)) for k in TOOLS}
    with torch.no_grad():
        for group in batches(orig):
            x=tensor(z[group[:,None]+np.arange(-L,0)],'cpu')
            values=bank.forward(x,torch.as_tensor(times[group]))
            for k,v in values.items():matrices[k][group-orig[0]]=v
    y=target(z,orig)
    nodes={};p=dict(matrices);risks={k:risk(v,y) for k,v in p.items()}
    def add(name,node):
        nodes[name]=node;p[name]=compose(p,{name:node})[name]
        risks[name]=risk(p[name],y)
    stand=[k for k in TOOLS if k!='R']
    add('E',select_map(stand,risks,64))
    # Earlier all-program pool C; predeclared mixtures enlarge the eligible library.
    for k,sources in enumerate([['DLinear','ridge_clock'],['day_mean','ridge_full'],['R','ridge_cross']]):
        add('prior'+str(k+1),dict(kind='mean',sources=sources))
    early=list(TOOLS)+['prior1','prior2','prior3']
    add('C',select_map(early,risks,64))
    add('M10',schema_action('C','OT','R',.30))
    pool=early+['C','M10','E']
    # Greedy expansion acts on the sum of per-channel minima.
    per={k:risks[k].mean(0) for k in pool}
    roots=[min(pool,key=lambda k:(per[k].mean(),k))]
    for _ in range(2):
        g=min(pool,key=lambda k:(np.stack([per[j] for j in roots+[k]]).min(0).sum(),k))
        if g not in roots:roots.append(g)
    add('M9',select_map(roots,risks,64))
    # Deduplicate prediction columns before retaining the top five per variable.
    shortlist=[]
    for j in range(D):
        seen=set();selected=[]
        for k in sorted(pool,key=lambda k:(per[k][j],k)):
            fingerprint=hashlib.sha256(np.ascontiguousarray(p[k][:,:,j]).tobytes()).digest()
            if fingerprint in seen:continue
            seen.add(fingerprint);selected.append(k)
            if len(selected)==5:break
        shortlist.append(selected)
    for b,m in [(16,'M3'),(32,'M6'),(64,'M8')]:add(m,select_map(pool,risks,b,shortlist))
    for m,parent,var,alt,w in [('M2','M3','LULL','E',.20),('M1','M2','HUFL','A',.25),
           ('M4','M3','HUFL','E',.5),('M5','M3','HULL','E',.3),('M7','M3','OT','E',.5)]:
        add(m,schema_action(parent,var,alt,w))
    # Use compact sufficient statistics for evaluating one-column edits.
    parents=['M2','M3','M6','M8','M10']; alts=['E','A','R','M9']
    facts={}
    for parent in parents:
        facts[parent]={}
        ep=p[parent].astype(float)-y
        for alt in alts:
            delta=p[alt].astype(float)-p[parent]
            facts[parent][alt]={'cross':(ep*delta).mean((0,1)).tolist(),'squared_delta':(delta**2).mean((0,1)).tolist()}
    scores={k:metric(v,y) for k,v in p.items()}
    plan=dict(dataset='ETTm1',variant=cfg['variant'],context=L,horizon=H,columns=COLS,
              minutes_per_step=15,steps_per_day=96,state='tools.npz',state_sha256=fit['state_sha256'],
              mean=fit['mean'],std=fit['std'],nodes=nodes,outputs={**{f'M{i}':f'M{i}' for i in range(1,11)},'DLinear':'DLinear'})
    write(out/'fixed_draft.json',plan)
    write(out/'validation.json',dict(split='val',origins=[int(orig[0]),int(orig[-1])],count=len(orig),
         scores=scores,per_variable_mse={k:v.mean(0).tolist() for k,v in risks.items()},roots=roots,
         shortlist=shortlist,editable_parents=parents,editable_sources=alts,edit_statistics=facts))
    # Keep only the compact auditable evidence; test time recalculates forecasts from history.
    for mat in matrices.values():mat.flush()
    print('VALIDATION_COMPLETE',len(orig),min((scores[m]['mse'],m) for m in plan['outputs']),flush=True)

def validate_edit(action,ev):
    if set(action)!={'parent','variable','source','alpha'}:raise ValueError('Malformed edit')
    p,s,j,a=(action[k] for k in ('parent','source','variable','alpha'))
    if p not in ev['editable_parents'] or s not in ev['editable_sources'] or j not in COLS or type(a) not in (float,int) or not .05<=a<=.95:
        raise ValueError('Edit out of bounds')
    return schema_action(p,j,s,a)

def edit_score(edit,ev):
    p,s,j,a=edit['parent'],edit['source'],COLS.index(edit['variable']),edit['alpha']
    f=ev['edit_statistics'][p][s]
    return ev['scores'][p]['mse']+(2*a*f['cross'][j]+a*a*f['squared_delta'][j])/D

def planner(out):
    ev=read(out/'validation.json')
    fields='parent, variable, source, alpha'
    ranked=sorted(ev['editable_parents'],key=lambda p:ev['scores'][p]['mse'])
    facts={}
    for parent in ranked:
        facts[parent]=dict(mse=round(ev['scores'][parent]['mse'],8),sources={})
        for src in ev['editable_sources']:
            f=ev['edit_statistics'][parent][src]
            facts[parent]['sources'][src]={'cross':np.round(f['cross'],7).tolist(),'d2':np.round(f['squared_delta'],7).tolist()}
    details='The six load columns are HUFL,HULL,MUFL,MULL,LUFL,LULL; OT is oil temperature. Source meanings: E=per-column selected tool, A=old-history analogue ridge, R=calendar base plus learned training-error tree, M9=greedy root selector. '
    prompt=(details+'ETTm1 is 15-minute data, daily period 96 steps. Context 512, horizon 64. Only validation evidence. '
        +'Propose exactly 3 distinct edits with fields '+fields+'; use alpha between 0.05 and 0.95. '
        +'One edit changes one full variable. Objective minimize macro MSE. Change = (2 alpha cross + alpha^2 d2)/7. '
        +'Return JSON {"reflection":"...","edits":[...]}. Choose numerically complementary alternatives. Columns: '+str(COLS)+' Evidence: '+json.dumps(facts))
    history=[];seen=set()
    for batch in range(2):
        trace=out/f'planner_batch{batch+1}.json'
        if trace.exists():record=read(trace);assert record['prompt']==prompt
        else:
            obj,meta=request_proposal(prompt);record=dict(prompt=prompt,response=obj,meta=meta)
            write(trace,record)
        choices=record['response']['edits']
        if len(choices)!=3:raise ValueError('Expected three edits')
        batch_scores=[]
        for a in choices:
            node=validate_edit(a,ev);fingerprint=json.dumps(node,sort_keys=True)
            item=dict(id='LLM'+str(len(history)+1),node=node,validation_mse=edit_score(node,ev),duplicate=fingerprint in seen)
            seen.add(fingerprint);history.append(item);batch_scores.append(item)
        prompt+='\nVerified feedback from the previous three trials: '+json.dumps(batch_scores)+'. Reflect on successes and failures and propose three distinct additional edits.'
    return history

def freeze(out):
    if (out/'freeze.json').exists():raise FileExistsError('Already frozen')
    ev=read(out/'validation.json');plan=read(out/'fixed_draft.json')
    hist=planner(out)
    # Same six-proposal allowance for a deterministic baseline: fit the optimal
    # convex weight analytically for each possible edit, score its six best proposals.
    opts=[]
    for parent in ev['editable_parents']:
        for src,f in ev['edit_statistics'][parent].items():
            for j,v in enumerate(COLS):
                w=float(np.clip(-f['cross'][j]/max(f['squared_delta'][j],1e-14),.05,.95))
                n=schema_action(parent,v,src,w);opts.append((edit_score(n,ev),parent,src,v,w))
    det=[]
    for i,(score,p,s,v,w) in enumerate(sorted(opts)[:6],1):
        det.append(dict(id='Rule'+str(i),node=schema_action(p,v,s,w),validation_mse=score))
    for item in hist+det:
        plan['nodes'][item['id']]=item['node'];plan['outputs'][item['id']]=item['id']
    fixed=min((ev['scores'][f'M{i}']['mse'],f'M{i}') for i in range(1,11))
    llm=min((x['validation_mse'],x['id']) for x in hist);rule=min((x['validation_mse'],x['id']) for x in det)
    selected=min(fixed,llm,rule)
    plan.update(runtime_sha256=digest(ROOT/'forecast.py'),selected=selected[1],best_fixed=fixed[1],best_llm=llm[1],best_rule=rule[1])
    write(out/'planner_results.json',dict(llm=hist,deterministic=det))
    write(out/'frozen_plan.json',plan)
    write(out/'freeze.json',dict(plan_sha256=digest(out/'frozen_plan.json'),weights_sha256=plan['state_sha256'],
         selection='minimum validation macro MSE among fixed M1–M10 and six proposals per planner',selected=selected,
         frozen_epoch=time.time(),validation_count=ev['count'],train_end=34560,val_end=46080))
    print('FROZEN',selected,fixed,llm,rule,flush=True)

def evaluate(out,csv_path):
    if (out/'test.json').exists():raise FileExistsError('Test already scored')
    f=load_forecaster(out/'frozen_plan.json');cfg=read(out/'fit.json')['contract']
    if digest(csv_path)!=cfg['sha256']:raise ValueError('Data mismatch')
    start,end=cfg['validation_end'],cfg['test_end'];orig=np.arange(start,end-H+1)
    raw,times=dataset(csv_path,int(orig[-1]));delta=np.arange(-L,0)
    dest=out/'test_predictions';dest.mkdir(exist_ok=False)
    comp=load_forecaster(out/'matched_control/frozen_plan.json')
    mats={k:np.lib.format.open_memmap(dest/(k+'.npy'),mode='w+',dtype='float32',shape=(len(orig),H,D)) for k in [*f.plan['outputs'],*comp.plan['outputs']]}
    for group in batches(orig):
        idx=group[:,None]+delta;ts=times[idx].astype('datetime64[m]')
        pred=f.predict_batch(raw[idx],ts,standardized=True)
        pred.update(comp.predict_batch(raw[idx],ts,standardized=True))
        for k,v in pred.items():
            if not np.isfinite(v).all():raise ValueError(k)
            mats[k][group-orig[0]]=v
    for k,m in mats.items():m.flush()
    write(out/'predictions_ready.json',dict(freeze_sha256=digest(out/'freeze.json'),matched_control_freeze=digest(out/'matched_control/freeze.json'),complete_epoch=time.time(),
         origins=[int(orig[0]),int(orig[-1])],files={k:digest(dest/(k+'.npy')) for k in mats}))
    raw,_=dataset(csv_path,end);y=target(((raw-f.plan['mean'])/f.plan['std']).astype('float32'),orig)
    results={k:metric(m,y) for k,m in mats.items()}
    sample=f.predict(raw[start-L:start],times[start-L:start].astype('datetime64[m]'),f.plan['selected'])
    check=np.max(np.abs((sample-f.plan['mean'])/f.plan['std']-mats[f.plan['selected']][0]))
    if check>1e-4:raise ValueError('Single-origin contract failed')
    write(out/'test.json',dict(results=results,n=len(orig),selected=f.plan['selected'],single_window_delta=float(check)))
    print('TEST_COMPLETE',len(orig),f.plan['selected'],results[f.plan['selected']],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['validate','freeze','test'])
    ap.add_argument('--out',type=Path,default=ROOT/'artifacts/ETTm1');ap.add_argument('--csv',type=Path,default=ROOT/'assets/ETTm1.csv')
    args=ap.parse_args();torch.set_num_threads(4)
    if args.stage=='validate':score_tools(args.out,args.csv)
    if args.stage=='freeze':freeze(args.out)
    if args.stage=='test':evaluate(args.out,args.csv)
