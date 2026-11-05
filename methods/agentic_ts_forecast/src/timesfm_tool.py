"""Pinned TimesFM3 research-only weights; no positivity clipping or future covariates."""
from common import *
import torch,sys
from timesfm3 import TimesFM3Evaluator,ModelConfig
sym=bool(int(sys.argv[1])) if len(sys.argv)>1 else False
name=f'20261105_timesfm3_sym{int(sym)}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True)
subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True)
guard(600);torch.set_num_threads(4);torch.manual_seed(2021)
model=RUN/'models/timesfm3_43046b85';pipe=TimesFM3Evaluator(ModelConfig(checkpoint_path=str(model),per_core_batch_size=16,device='cuda:0',local_files_only=True))
z=np.load(RUN/'cache/trainval_z.npy');manifest=json.loads((RUN/'cache/window_manifest.json').read_text());start=time.time();results={}
def forecast(x):return list(pipe.predict_batch([a.T.copy() for a in x],horizon=64,return_quantiles=True,use_symmetric_averaging=sym,make_positive=False,sort_quantiles=True,use_znorm=False,padding_mode='none'))
# Check that another window cannot change this window's forecast through batch mixing.
a=z[512:1024];b=z[1024:1536];pa=forecast(np.stack([a,b]))[0].forecast;pb=forecast(np.stack([a,b+10]))[0].forecast;delta=float(np.max(np.abs(pa-pb)))
assert delta<1e-5,('cross-window influence',delta);save_json(out/'batch_isolation.json',{'perturbed_other_window_max_abs_delta':delta,'tolerance':1e-5})
for split,ts in manifest.items():
 pred=[];qs=[]
 for i in range(0,len(ts),16):
  guard(600);x,_=windows(z,ts[i:i+16]);ans=forecast(x);pred.extend([a.forecast.T for a in ans]);qs.extend([a.quantiles.transpose(1,0,2) for a in ans])
  if i%320==0:print(split,i,len(ts),flush=True)
 p=np.stack(pred).astype(np.float32);q=np.stack(qs).astype(np.float32);results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'))
 np.save(RUN/'cache'/f'{split}_timesfm{int(sym)}.npy',p);np.save(out/f'{split}_quantiles.npy',q);save_json(out/'partial_metrics.json',results)
save_json(out/'config.json',{'model':'google/timesfm-3.0-pytorch','revision':'43046b85ec22d584a13f8098c2ed39c889e129c2','weights_sha256':hashlib.sha256((model/'model.safetensors').read_bytes()).hexdigest(),'license':'timesfm-non-commercial-license-v1.0 research-only','symmetric_averaging':sym,'make_positive':False,'context':512,'horizon':64,'multivariate':True,'future_covariates':False,'code_revision':git_head(),'elapsed_seconds':time.time()-start})
save_json(out/'metrics.json',results);print({s:r['mse'] for s,r in results.items()},flush=True)
