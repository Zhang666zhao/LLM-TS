"""Discrete expert selection by variable and fixed lead block, validation only."""
from common import *
from blend_sources import get_source
import sys

def predict(p,cfg):
 assert p.ndim==4 and p.shape[2:]==(64,7)
 indices=np.asarray(cfg['lead_column_indices']);assert indices.shape==(64,7)
 return np.stack([np.stack([p[:,indices[h,j],h,j] for j in range(7)],-1) for h in range(64)],1).astype(np.float32)

def run(tag,pooltag):
 guard(2400);poolpath=RUN/'selection_pools'/pooltag/'pool.json';pool=json.loads(poolpath.read_text());rows=pool['candidates'];short={}
 for j,col in enumerate(COLS):
  short[col]=[];seen=set()
  for r in sorted(rows,key=lambda r:(r['per_variable_mse'][col],r['graph_cost'],r['candidate'])):
   values=get_source('run:'+r['candidate'],'val')[:,:,j];digest=hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
   if digest in seen:continue
   seen.add(digest);short[col].append(r)
   if len(short[col])==5:break
 names=list(dict.fromkeys(r['candidate'] for col in COLS for r in short[col]));truth=np.load(RUN/'cache/val_targets.npy');errors={}
 for n in names:
  guard(1800);pp=get_source('run:'+n,'val');errors[n]=np.mean((pp.astype(float)-truth.astype(float))**2,axis=0)
 for block in (64,32,16):
  name=f'20261105_blockcolumns_{tag}_b{block}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=False);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);chosen=[]
  for start in range(0,64,block):chosen.append([min(short[col],key=lambda r:(float(errors[r['candidate']][start:start+block,j].mean()),r['graph_cost'],r['candidate']))['candidate'] for j,col in enumerate(COLS)])
  ids=list(dict.fromkeys('run:'+n for row in chosen for n in row));idx=np.repeat([[ids.index('run:'+n) for n in row] for row in chosen],block,axis=0);cfg=dict(engine='block_column_selector_v1',source_ids=ids,lead_column_indices=idx.tolist(),block_size=block,shortlist_per_variable={col:[r['candidate'] for r in short[col]] for col in COLS},selection_pool=str(poolpath),selection_pool_sha256=hashlib.sha256(poolpath.read_bytes()).hexdigest(),runtime_llm=False,selection='discrete validation model selection, fixed top5 distinct validation-prediction full-horizon per-variable shortlist; no parameter fitting; all three block sizes predeclared',code_revision=git_head());save_json(out/'config.json',cfg);results={}
  for split in ('dev','confirm','val'):
   p=predict(np.stack([get_source(k,split) for k in ids],1),cfg);np.save(out/f'{split}_predictions.npy',p);results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'))
  expected=sum(errors[ids[idx[h,j]][4:]][h,j] for h in range(64) for j in range(7))/(64*7);assert abs(results['val']['mse']-expected)<1e-10
  save_json(out/'metrics.json',dict(candidate=name,metrics=results,status='evaluated'));save_json(out/'contracts.json',dict(selected_cells_exact=True,recomputed_expected_mse=float(expected),test_access=False));print(name,results['val']['mse'],flush=True)
if __name__=='__main__':run(sys.argv[1],sys.argv[2])
