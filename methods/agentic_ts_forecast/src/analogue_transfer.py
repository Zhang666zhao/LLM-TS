"""Matched simple forecasts from causally old training cases, without foundation model."""
from common import *
from analogue_context import retrieve
import pandas as pd,sys

def predict(z,ts,cfg,dates,trace=None):
 dates=pd.to_datetime(dates);ts=np.asarray(ts);path=pathlib.Path(cfg['training_memory']);assert hashlib.sha256(path.read_bytes()).hexdigest()==cfg['training_memory_sha256'];b=dict(np.load(path));nearest=retrieve(z,ts,dates,cfg,b);ps=[]
 for t,a in zip(ts,nearest):
  p=b['train_z'][a:a+64].astype(float).copy()
  if cfg['transfer_mode']!='raw':
   own=z[t-96:t].astype(float);old=b['train_z'][a-96:a].astype(float);ratio=np.clip(own.std(0)/np.maximum(old.std(0),1e-4),.5,2.) if cfg['transfer_mode']=='location_scale' else np.ones(7);p=(p-old.mean(0))*ratio+own.mean(0)
  ps.append(p.astype(np.float32))
 if trace is not None:np.savez_compressed(trace,query_origins=ts,memory_origins=nearest)
 return np.stack(ps)

def run(mode):
 assert mode in ('raw','location','location_scale');name=f'20261105_analoguetransfer_{mode}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=False);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);guard(1800);path=RUN/'cache/analogue_context_memory_v1.npz';cfg=dict(engine='analogue_transfer_v1',transfer_mode=mode,memory_mode='future_case',training_memory=str(path),training_memory_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),no_fitting=True,location_window=96,scale_ratio_bounds=[.5,2.],scale_floor=1e-4,code_revision=git_head());save_json(out/'config.json',cfg);z=np.load(RUN/'cache/trainval_z.npy');dates=json.loads((RUN/'cache/dates.json').read_text());results={}
 for split,ts in json.loads((RUN/'cache/window_manifest.json').read_text()).items():
  p=predict(z,ts,cfg,dates,out/(split+'_retrieval.npz'));np.save(out/f'{split}_predictions.npy',p);results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'));a=np.load(out/(split+'_retrieval.npz'));old=np.load(ROOT/'runs/20261105_analoguecontext_future_case_mse'/(split+'_retrieval.npz'));assert np.array_equal(a['memory_origins'],old['memory_origins'])
 save_json(out/'contracts.json',dict(neighbors_exactly_match_frozen_foundation_memory=True,memory_before_query_history=True,no_current_future_input=True));save_json(out/'metrics.json',dict(candidate=name,metrics=results,status='evaluated'));print(name,results['val']['mse'],flush=True)
if __name__=='__main__':run(sys.argv[1])
