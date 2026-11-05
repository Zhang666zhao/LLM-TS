"""Invertible variable views: empirical residual basis, PCA, random control."""
from common import *
import sys
def transform(train,mode,rotation_seed=2021):
 if mode=='semantic':
  a=np.eye(7);a[2,0]=-1;a[4,0]=-1;a[3,1]=-1;a[5,1]=-1;base=np.diag(np.asarray(CONTRACT['scaler_scale']))@a
 elif mode=='pca':
  values,base=np.linalg.eigh(np.cov(train,rowvar=False,ddof=0));base=base[:,np.argsort(values)[::-1]]
  for j in range(7):
   if base[np.abs(base[:,j]).argmax(),j]<0:base[:,j]*=-1
 elif mode=='random':
  base,r=np.linalg.qr(np.random.default_rng(rotation_seed).normal(size=(7,7)));base*=np.sign(np.diag(r))[None]
 else:raise ValueError(mode)
 v=train@base;mu=v.mean(0);sd=np.maximum(v.std(0),1e-4);w=base/sd;bias=-mu/sd;inverse=np.linalg.inv(w);assert np.linalg.cond(w)<1e5;roundtrip=((train[:1024]@w+bias)-bias)@inverse;delta=float(abs(roundtrip-train[:1024]).max());assert delta<1e-8
 return {'matrix':w.tolist(),'bias':bias.tolist(),'inverse_matrix':inverse.tolist(),'roundtrip_max_delta':delta,'condition_number':float(np.linalg.cond(w))}
def extract_point(q,levels,rule):
 levels=np.asarray(levels,dtype=float)
 if rule=='median':return q[...,int(np.argmin(abs(levels-.5)))]
 if rule=='central_integral':
  use=(levels>=.1)&(levels<=.9);qq=q[...,use].astype(float);return ((qq[...,:-1]+qq[...,1:])*.5*np.diff(levels[use])).sum(-1)/.8
 if rule=='full_integral':
  qq=np.concatenate([q[...,:1],q,q[...,-1:]],axis=-1).astype(float);return ((qq[...,:-1]+qq[...,1:])*.5*np.diff(np.r_[0.,levels,1.])).sum(-1)
 raise ValueError(rule)
def smooth_context(x,k):
 k=int(k);assert k in (1,3,7,25)
 if k==1:return x
 padded=np.concatenate([np.repeat(x[:,:1],k-1,axis=1),x],axis=1).astype(np.float64)
 cs=np.concatenate([np.zeros_like(padded[:,:1]),np.cumsum(padded,axis=1)],axis=1)
 return ((cs[:,k:]-cs[:,:-k])/k).astype(np.float32)
def predict(z,ts,cfg,dates=None):
 import torch,gc
 from chronos import Chronos2Pipeline
 torch.set_num_threads(4);loc=ROOT/'runs'/cfg['checkpoint_parent']/'trainer/finetuned-ckpt' if cfg.get('checkpoint_parent') else RUN/'models/chronos2_29ec3766';pipe=Chronos2Pipeline.from_pretrained(str(loc),device_map='cuda:0');pipe.model.eval();context=int(cfg.get('inference_context',512));assert 64<=context<=512;w=np.asarray(cfg['matrix']);b=np.asarray(cfg['bias']);inv=np.asarray(cfg['inverse_matrix']);ps=[]
 calendar=cfg.get('calendar')
 if calendar:
  import pandas as pd
  assert calendar in ('hour','hourweek','houryear','hourweekyear') and dates is not None;dates=pd.to_datetime(dates)
  def cov(d):
   if calendar!='hour':
    from calendar_features import cov as expanded_cov
    return expanded_cov(d,calendar)
   h=np.asarray(d.hour,dtype=float);return {'hour_sin':np.sin(2*np.pi*h/24).astype(np.float32),'hour_cos':np.cos(2*np.pi*h/24).astype(np.float32)}
 batch=8 if calendar else 16
 for i in range(0,len(ts),batch):
  guard(600);x=np.stack([z[t-context:t] for t in ts[i:i+batch]]);x=smooth_context(x,cfg.get('context_smoothing',1));lat=(x.astype(float)@w+b).astype(np.float32);extra={'quantile_levels':cfg['point_quantile_levels']} if cfg.get('point_rule') else {};items=[torch.from_numpy(v.T.copy()) for v in lat]
  if calendar:items=[{'target':v.T.copy(),'past_covariates':cov(dates[t-context:t]),'future_covariates':cov(pd.date_range(dates[t-1]+pd.Timedelta(hours=1),periods=64,freq='h'))} for t,v in zip(ts[i:i+batch],lat)]
  q,_=pipe.predict_quantiles(items,prediction_length=64,batch_size=112,context_length=context,cross_learning=False,**extra);qq=np.stack([a.cpu().numpy().transpose(1,0,2) for a in q]);point=extract_point(qq,cfg['point_quantile_levels'],cfg['point_rule']) if cfg.get('point_rule') else qq.mean(-1);ps.append(((point-b)@inv).astype(np.float32))
 del pipe;gc.collect();torch.cuda.empty_cache();return np.concatenate(ps)
def run(mode,finetune=False):
 name=f'20261105_transformed{"ft" if finetune else ""}_{mode}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);guard(1800);z=np.load(RUN/'cache/trainval_z.npy');cfg=dict(transform(z[:8640].astype(float),mode),engine='transformed_tool_v1',mode=mode,training_rows=[0,8640],point='average latent quantiles, then inverse linear point transform; NOT calibrated original-variable quantiles',semantic_claim='invertible empirical representation, no physical identity imposed; all7 dimensions retained',code_revision=git_head());save_json(out/'config.json',cfg);manifest=json.loads((RUN/'cache/window_manifest.json').read_text());results={};start=time.time()
 if finetune:
  import torch,gc
  from chronos import Chronos2Pipeline
  from transformers import TrainerCallback
  class Deadline(TrainerCallback):
   def on_step_begin(self,args,state,control,**kwargs):guard(1200)
  torch.set_num_threads(4);torch.manual_seed(2021);torch.cuda.manual_seed_all(2021);np.random.seed(2021);latent=(z[:8640].astype(float)@np.asarray(cfg['matrix'])+np.asarray(cfg['bias'])).astype(np.float32);pipe=Chronos2Pipeline.from_pretrained(str(RUN/'models/chronos2_29ec3766'),device_map='cuda:0');trained=pipe.fit([torch.from_numpy(latent.T.copy())],prediction_length=64,validation_inputs=None,finetune_mode='full',context_length=512,learning_rate=5e-7,num_steps=500,batch_size=112,min_past=512,output_dir=out/'trainer',callbacks=[Deadline()],seed=2021,data_seed=2021,disable_tqdm=True,logging_steps=100,bf16=True,tf32=False);del pipe,trained;gc.collect();torch.cuda.empty_cache();cfg.update(checkpoint_parent=name,finetune_steps=500,finetune_rate=5e-7,seed=2021,postfit_inference='reload saved checkpoint and explicit eval');save_json(out/'config.json',cfg)
 for split,ts in manifest.items():
  p=predict(z,ts,cfg);results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'));np.save(out/f'{split}_predictions.npy',p);save_json(out/'partial_metrics.json',results)
 save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated','elapsed_seconds':time.time()-start});print(name,{s:m['mse'] for s,m in results.items()},flush=True)
if __name__=='__main__':run(sys.argv[1],bool(int(sys.argv[2])) if len(sys.argv)>2 else False)
