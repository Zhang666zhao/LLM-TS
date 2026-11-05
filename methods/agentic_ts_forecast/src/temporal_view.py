"""Train-only periodic profile / lag-difference views within fixed512 history."""
from common import *
import sys

def make_config(z,dates,mode):
 import pandas as pd
 if mode=='hourprofile':
  hours=pd.to_datetime(dates[:8640]).hour.to_numpy();profile=np.stack([z[:8640][hours==h].mean(0) for h in range(24)]);return {'mode':mode,'profile':profile.tolist(),'lag':0,'internal_context':512}
 lag={'diff24':24,'diff168':168}[mode];return {'mode':mode,'lag':lag,'internal_context':512-lag}
def prepare_train(z,dates,cfg):
 import pandas as pd
 if cfg['lag']:return z[cfg['lag']:8640]-z[:8640-cfg['lag']]
 return (z[:8640]-np.asarray(cfg['profile'])[pd.to_datetime(dates[:8640]).hour]).astype(np.float32)
def latent_inputs(x,hours,cfg):
 lag=cfg['lag']
 if lag:return x[:,lag:]-x[:,:-lag]
 return (x-np.asarray(cfg['profile'])[hours]).astype(np.float32)
def reconstruct(x,point,future_hours,cfg):
 lag=cfg['lag']
 if not lag:return (point+np.asarray(cfg['profile'])[future_hours]).astype(np.float32)
 out=np.empty_like(point)
 for h in range(64):out[:,h]=point[:,h]+(x[:,-lag+h] if h<lag else out[:,h-lag])
 return out

def predict(z,dates,ts,cfg):
 import torch,gc,pandas as pd
 from chronos import Chronos2Pipeline
 torch.set_num_threads(4);dates=pd.to_datetime(dates);loc=ROOT/'runs'/cfg['checkpoint_parent']/'trainer/finetuned-ckpt' if cfg.get('checkpoint_parent') else RUN/'models/chronos2_29ec3766';pipe=Chronos2Pipeline.from_pretrained(str(loc),device_map='cuda:0');pipe.model.eval();ps=[]
 for i in range(0,len(ts),16):
  guard(600);origins=ts[i:i+16];x=np.stack([z[t-512:t] for t in origins]);hours=np.stack([dates[t-512:t].hour for t in origins]);future_hours=np.stack([(dates[t-1].hour+1+np.arange(64))%24 for t in origins]);lat=latent_inputs(x,hours,cfg);assert lat.shape[1]==cfg['internal_context'];q,_=pipe.predict_quantiles([torch.from_numpy(v.T.copy()) for v in lat],prediction_length=64,batch_size=112,context_length=cfg['internal_context'],cross_learning=False);point=np.stack([a.cpu().numpy().transpose(1,0,2).mean(-1) for a in q]);ps.append(reconstruct(x,point,future_hours,cfg))
 del pipe;gc.collect();torch.cuda.empty_cache();return np.concatenate(ps)
def run(mode,finetune):
 name=f'20261105_temporal{"ft" if finetune else ""}_{mode}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);guard(1800);z=np.load(RUN/'cache/trainval_z.npy');dates=json.loads((RUN/'cache/dates.json').read_text());cfg=make_config(z,dates,mode);cfg.update(engine='temporal_view_v1',training_rows=[0,8640],external_context=512,horizon=64,point='average9 latent quantiles then causal inverse, original standardized MSE',runtime_llm=False,code_revision=git_head());save_json(out/'config.json',cfg)
 # Inverse identity checked only on training values, never future test labels.
 import pandas as pd
 ts=np.array([1024,2048]);x=np.stack([z[t-512:t] for t in ts]);truth=np.stack([z[t:t+64] for t in ts]);fh=np.stack([pd.to_datetime(dates[t:t+64]).hour for t in ts]);lag=cfg['lag'];latent_truth=np.stack([z[t:t+64]-z[t-lag:t+64-lag] for t in ts]) if lag else truth-np.asarray(cfg['profile'])[fh];back=reconstruct(x,latent_truth,fh,cfg);delta=float(abs(back-truth).max());assert delta<1e-5;save_json(out/'contract.json',{'training_inverse_roundtrip_max_delta':delta,'no_extra_history':True,'future_measurements_input':False})
 start=time.time()
 if finetune:
  import torch,gc
  from chronos import Chronos2Pipeline
  from transformers import TrainerCallback
  class Deadline(TrainerCallback):
   def on_step_begin(self,args,state,control,**kwargs):guard(1200)
  torch.set_num_threads(4);torch.manual_seed(2021);torch.cuda.manual_seed_all(2021);np.random.seed(2021);train=prepare_train(z,dates,cfg);pipe=Chronos2Pipeline.from_pretrained(str(RUN/'models/chronos2_29ec3766'),device_map='cuda:0');trained=pipe.fit([torch.from_numpy(train.T.copy())],prediction_length=64,validation_inputs=None,finetune_mode='full',context_length=cfg['internal_context'],learning_rate=5e-7,num_steps=500,batch_size=112,min_past=cfg['internal_context'],output_dir=out/'trainer',callbacks=[Deadline()],seed=2021,data_seed=2021,disable_tqdm=True,logging_steps=100,bf16=True,tf32=False);del pipe,trained;gc.collect();torch.cuda.empty_cache();cfg.update(checkpoint_parent=name,finetune_steps=500,finetune_rate=5e-7,seed=2021,postfit_inference='independent checkpoint reload eval');save_json(out/'config.json',cfg)
 results={};mf=json.loads((RUN/'cache/window_manifest.json').read_text())
 for split,ts in mf.items():
  p=predict(z,dates,ts,cfg);results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'));np.save(out/f'{split}_predictions.npy',p);save_json(out/'partial_metrics.json',results)
 save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated','elapsed_seconds':time.time()-start});print(name,{s:m['mse'] for s,m in results.items()},flush=True)
if __name__=='__main__':run(sys.argv[1],bool(int(sys.argv[2])))
