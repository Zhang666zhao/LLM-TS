"""Hindcast errors from within the current512 observations, using strong tools."""
from common import *
import sys
TOOLS=['20261105_reloadedft_lr1e6_mse','20261105_chronoscalft_lr5e7_mse']
PARENT='20261105_policy_p04b006c1_mse'
def hindcasts(z,dates,ts,out):
 import torch,pandas as pd,gc
 from chronos import Chronos2Pipeline
 from finetune_calendar import cov
 out=pathlib.Path(out);out.mkdir(exist_ok=True,parents=True);dates=pd.to_datetime(dates);ts=np.asarray(ts);torch.set_num_threads(4);result=[]
 for j,name in enumerate(TOOLS):
  path=out/(name+'.npy')
  if path.exists():p=np.load(path)
  else:
   cfg=json.loads((ROOT/'runs'/name/'config.json').read_text());loc=ROOT/'runs'/cfg.get('checkpoint_parent',name)/'trainer/finetuned-ckpt';pipe=Chronos2Pipeline.from_pretrained(str(loc),device_map='cuda:0');pipe.model.eval();ps=[]
   for i in range(0,len(ts),16):
    guard(600);items=[]
    for t in ts[i:i+16]:
     obs=z[t-512:t-64];assert obs.shape==(448,7)
     if j:items.append({'target':obs.T.copy(),'past_covariates':cov(dates[t-512:t-64]),'future_covariates':cov(dates[t-64:t])})
     else:items.append(torch.from_numpy(obs.T.copy()))
    q,_=pipe.predict_quantiles(items,prediction_length=64,batch_size=112,context_length=512,cross_learning=False);ps.extend([a.cpu().numpy().transpose(1,0,2).mean(-1) for a in q])
   p=np.stack(ps).astype(np.float32);np.save(path,p);del pipe;gc.collect();torch.cuda.empty_cache()
  assert p.shape==(len(ts),64,7) and np.isfinite(p).all();result.append(p)
 return np.stack(result,1)
def correction(z,ts,pred):
 x=np.stack([z[t-512:t] for t in ts]);observed=x[:,-64:];residual=observed[:,None]-pred;bias=(.7*residual[:,0]+.3*residual[:,1]).mean(1);bound=.5*np.maximum(x.std(1),.05);return np.clip(bias,-bound,bound)[:,None]*np.linspace(1,.5,64)[None,:,None]
def cache():
 z=np.load(RUN/'cache/trainval_z.npy');dates=json.loads((RUN/'cache/dates.json').read_text());manifest=json.loads((RUN/'cache/window_manifest.json').read_text())
 for split,ts in manifest.items():
  p=hindcasts(z,dates,ts,RUN/'strong_context_cache'/split);np.save(RUN/'cache'/f'{split}_strong_observed_correction.npy',correction(z,ts,p));print('STRONG_HINDCAST_DONE',split,flush=True)
 save_json(RUN/'strong_context_cache/manifest.json',{'tools':TOOLS,'proxy_weights':[.7,.3],'input_scope':'current512 only; first448 predicts last64 observed; no earlier history','temporal_transfer':'past residual used only as proxy for future residual','code_revision':git_head()})
def run(strength,tag):
 name=f'20261105_observedfeedback_{tag}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);results={}
 for split in ('dev','confirm','val'):
  guard(600);p=np.load(ROOT/'runs'/PARENT/f'{split}_predictions.npy');cor=np.load(RUN/'cache'/f'{split}_strong_observed_correction.npy');pr=(p+strength*cor).astype(np.float32);results[split]=metric(pr,np.load(RUN/'cache'/f'{split}_targets.npy'));np.save(out/f'{split}_predictions.npy',pr)
 save_json(out/'config.json',{'engine':'observed_feedback_v1','parent':PARENT,'source_ids':['run:'+PARENT]+['run:'+v for v in TOOLS],'strength':strength,'proxy_weights':[.7,.3],'clip_history_std':.5,'horizon_taper':[1,.5],'history_scope':512,'code_revision':git_head()});save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated'});print(name,{s:m['mse'] for s,m in results.items()},flush=True)
if __name__=='__main__':
 if sys.argv[1]=='cache':cache()
 else:run(float(sys.argv[1]),sys.argv[2])
