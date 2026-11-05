"""A second fixed controller, limited to the same number of scored proposals as the LLM."""
from forecast import *
from run_pipeline import edit_score,metric,schema_action

def prepare(out):
    if (out/'predictions_ready.json').exists() or (out/'test.json').exists():raise RuntimeError('Controls must be frozen before test')
    ev=read(out/'validation.json')
    plan=read(out/'frozen_plan.json')
    assert digest(out/'frozen_plan.json')==read(out/'freeze.json')['plan_sha256']
    ctrl=out/'matched_control';ctrl.mkdir(exist_ok=False)
    # Batch one: use the best editable parent, then rank its columns by the
    # risk gap versus the best eligible full-column alternative. Score three
    # distinct quarter-strength edits. Batch two: respond to the measured sign.
    parent=min(ev['editable_parents'],key=lambda n:ev['scores'][n]['mse'])
    edits=[]
    for j,var in enumerate(COLS):
        src=min(ev['editable_sources'],key=lambda n:ev['per_variable_mse'][n][j])
        gap=ev['per_variable_mse'][parent][j]-ev['per_variable_mse'][src][j]
        edits.append((gap,var,src))
    proposals=sorted(edits,reverse=True)[:3];history=[]
    for batch in [0,1]:
        for j,(_,v,s) in enumerate(proposals):
            w=.25 if batch==0 else (.50 if history[j]['validation_mse']<ev['scores'][parent]['mse'] else .10)
            node=schema_action(parent,v,s,w);name='Control'+str(len(history)+1)
            history.append(dict(id=name,node=node,validation_mse=edit_score(node,ev)))
            plan['nodes'][name]=node
    plan['outputs']={x['id']:x['id'] for x in history}
    plan['selected']=min(history,key=lambda x:x['validation_mse'])['id'];plan['state']='../tools.npz'
    write(ctrl/'frozen_plan.json',plan);write(ctrl/'proposals.json',history)
    write(ctrl/'freeze.json',dict(plan_sha256=digest(ctrl/'frozen_plan.json'),frozen_epoch=__import__('time').time(),
        criterion='Six deterministic edits, three then three, validation only; secondary comparison'))
    return ctrl

def audit(out):
    from run_pipeline import dataset,target
    ev=read(out/'validation.json');cfg=read(out/'fit.json')['contract']
    plan=load_forecaster(out/'frozen_plan.json').plan
    comp=load_forecaster(out/'matched_control/frozen_plan.json').plan
    raw,_=dataset(Path(__file__).parent/'assets/ETTm1.csv',cfg['validation_end'])
    z=((raw-plan['mean'])/plan['std']).astype('float32')
    origins=np.arange(cfg['train_end'],cfg['validation_end']-H+1);y=target(z,origins)
    cache={k:np.load(out/'validation_cache'/(k+'.npy'),mmap_mode='r') for k in TOOLS}
    all_scores={}
    expected={k:v['mse'] for k,v in ev['scores'].items()}
    proposal=read(out/'planner_results.json')
    for x in proposal['llm']+proposal['deterministic']+read(out/'matched_control/proposals.json'):
        expected[x['id']]=x['validation_mse']
    for pp in [plan,comp]:
        p=compose(cache,pp['nodes'])
        for k,n in pp['outputs'].items():
            all_scores[k]=metric(p[n],y)
            if abs(all_scores[k]['mse']-expected[n])>2e-7:raise ValueError(f'Validation reconstruction: {k}')
        del p
    write(out/'validation_reconstruction.json',all_scores)
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path(__file__).parent/'artifacts/ETTm1')
    out=ap.parse_args().out;prepare(out);audit(out)
