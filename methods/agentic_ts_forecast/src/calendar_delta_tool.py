"""Explicit clock Fourier delta, train-only GPU ridge; recent24 level is history only."""
from common import *
import torch,pandas as pd,sys

def phi(d,annual):
 d=pd.DatetimeIndex(d);h=np.asarray(d.hour,dtype=float)/24
 a=[fn(2*np.pi*k*h) for k in (1,2,3) for fn in (np.sin,np.cos)]
 year=(np.asarray(d.dayofyear,dtype=float)-1+h)/np.where(d.is_leap_year,366,365)
 a.extend(fn(2*np.pi*k*year) for k in range(1,annual+1) for fn in (np.sin,np.cos))
 return np.stack(a,-1)

def features(dates,ts,annual):
 dates=pd.DatetimeIndex(dates);a=[]
 for t in ts:
  future=pd.date_range(dates[t-1]+pd.Timedelta(hours=1),periods=64,freq='h')
  a.append(phi(future,annual)-phi(dates[t-24:t],annual).mean(0))
 return np.stack(a)

def predict(z,dates,ts,cfg,state):
 a=features(dates,ts,cfg['annual_harmonics'])/np.asarray(cfg['feature_scale'])
 a=np.concatenate([a,np.ones((*a.shape[:-1],1))],-1)
 w=torch.load(state,map_location='cpu',weights_only=True)['weight'].numpy()
 center=np.stack([z[t-24:t].mean(0,dtype=np.float64) for t in ts])
 return (a@w+center[:,None]).astype(np.float32)

def run(annual):
 assert annual in (0,1,2);name=f'20261105_calendardelta_year{annual}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=False);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);guard(2400);torch.set_num_threads(4)
 z=np.load(RUN/'cache/trainval_z.npy');dates=json.loads((RUN/'cache/dates.json').read_text());ts=np.arange(512,8577,4);a=features(dates,ts,annual);scale=np.maximum(a.reshape(-1,a.shape[-1]).std(0),1e-3);a=a/scale;a=np.concatenate([a,np.ones((*a.shape[:-1],1))],-1).reshape(-1,a.shape[-1]+1);b=np.stack([z[t:t+64].astype(float)-z[t-24:t].mean(0,dtype=np.float64) for t in ts]).reshape(-1,7);assert ts[-1]+64<=8640
 aa=torch.tensor(a,device='cuda:0',dtype=torch.float64);bb=torch.tensor(b,device='cuda:0',dtype=torch.float64);reg=torch.eye(a.shape[-1],device='cuda:0',dtype=torch.float64)*.001*len(a);reg[-1,-1]=1e-8;w=torch.linalg.solve(aa.T@aa+reg,aa.T@bb);assert torch.isfinite(w).all();torch.save({'weight':w.cpu()},out/'best.pth');del aa,bb,w;torch.cuda.empty_cache()
 cfg=dict(engine='calendar_delta_v1',annual_harmonics=annual,hour_harmonics=3,alpha=.001,center_hours=24,feature_scale=scale.tolist(),train_origins=[512,8576,4],training_rows=[0,8640],labels='future64 minus own recent24 observed mean; all training targets end<=8640',features='future calendar minus recent24 mean calendar; leap-aware year; no physical causal assertion',fit_device='cuda:0',code_revision=git_head());save_json(out/'config.json',cfg)
 # Query permutation and absence of dependence on rows at/after each origin.
 t=np.array([1024,1536]);p=predict(z,dates,t,cfg,out/'best.pth');assert np.array_equal(p[::-1],predict(z,dates,t[::-1],cfg,out/'best.pth'));zz=z.copy();zz[1024:]+=100;assert np.array_equal(p[:1],predict(zz,dates,t[:1],cfg,out/'best.pth'));save_json(out/'contracts.json',dict(query_permutation_exact=True,future_rows_ignored=True,train_targets_end=8640,finite_gpu_solve=True))
 results={}
 for split,orig in json.loads((RUN/'cache/window_manifest.json').read_text()).items():
  guard(1800);p=predict(z,dates,orig,cfg,out/'best.pth');np.save(out/f'{split}_predictions.npy',p);results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'))
 save_json(out/'metrics.json',dict(candidate=name,metrics=results,status='evaluated'));print(name,{s:m['mse'] for s,m in results.items()},flush=True)
if __name__=='__main__':run(int(sys.argv[1]))
