"""Typed convex horizon profiles proposed offline, never fitted to test."""
from common import *
from blend_sources import get_source
import sys
KNOTS=[7.5,23.5,39.5,55.5]
def canonical(c,registry):
 c=dict(c);c.setdefault('variable_weights',{});assert set(c)=={'source_ids','knot_weights','variable_weights'} and c['source_ids']==registry and len(registry)==4
 def norm(v):
  a=np.asarray(v,float);assert a.shape==(4,4) and np.isfinite(a).all() and (a>=0).all() and (a.sum(1)>0).all();return (a/a.sum(1,keepdims=True)).tolist()
 assert isinstance(c['variable_weights'],dict) and set(c['variable_weights'])<=set(COLS)
 default=norm(c['knot_weights']);overrides={k:norm(v) for k,v in c['variable_weights'].items()};overrides={k:v for k,v in overrides.items() if not np.array_equal(np.asarray(v),np.asarray(default))}
 return dict(source_ids=registry,knot_weights=default,variable_weights=overrides)
def active_sources(c):
 return np.any(np.asarray([c['variable_weights'].get(col,c['knot_weights']) for col in COLS],float)>0,axis=(0,1))
def predict(p,c):
 w=np.empty((64,7,4),float)
 for j,col in enumerate(COLS):
  a=np.asarray(c['variable_weights'].get(col,c['knot_weights']),float);a/=a.sum(1,keepdims=True)
  for s in range(4):w[:,j,s]=np.interp(np.arange(64),KNOTS,a[:,s])
 assert np.allclose(w.sum(-1),1) and (w>=0).all()
 return np.einsum('nshc,hcs->nhc',p.astype(float),w).astype(np.float32)
def run(c,name):
 c=canonical(c,c['source_ids']);invalid=set(json.loads((RUN/'invalid_candidates.json').read_text())['candidates']);assert all(k.startswith('run:') and k[4:] not in invalid for k in c['source_ids'])
 out=ROOT/'runs'/name;out.mkdir(exist_ok=False);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);results={};start=time.time()
 for split in json.loads((RUN/'cache/window_manifest.json').read_text()):
  guard(900);p=predict(np.stack([get_source(k,split) for k in c['source_ids']],1),c);results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'));np.save(out/f'{split}_predictions.npy',p)
 save_json(out/'config.json',dict(c,engine='horizon_policy_v1',knots=KNOTS,runtime_llm=False,code_revision=git_head()));save_json(out/'metrics.json',dict(candidate=name,metrics=results,status='evaluated',elapsed_seconds=time.time()-start));print(name,results['val']['mse'],flush=True)
if __name__=='__main__':run(json.loads(sys.argv[1]),sys.argv[2])
