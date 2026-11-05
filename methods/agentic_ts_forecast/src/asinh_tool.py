"""Monotone per-variable tail compression; invert quantiles before original-space integration."""
from common import *
from transformed_tool import extract_point
from transformed_calendar import cov
import torch,pandas as pd,gc,sys
from chronos import Chronos2Pipeline
from chronos.chronos2.preprocess import from_list_of_dicts
from transformers import TrainerCallback
class Deadline(TrainerCallback):
 def on_step_begin(self,args,state,control,**kwargs):guard(1800)
def predict(z,ts,cfg,dates):
 torch.set_num_threads(4);dates=pd.to_datetime(dates);parent=cfg.get('checkpoint_parent');loc=ROOT/'runs'/parent/'trainer/finetuned-ckpt' if parent else RUN/'models/chronos2_29ec3766';pipe=Chronos2Pipeline.from_pretrained(str(loc),device_map='cuda:0');pipe.model.eval();ps=[];levels=cfg['point_quantile_levels']
 for i in range(0,len(ts),8):
  guard(600);items=[]
  for t in ts[i:i+8]:
   x=np.arcsinh(z[t-512:t].astype(float)).astype(np.float32);future=pd.date_range(dates[t-1]+pd.Timedelta(hours=1),periods=64,freq='h');items.append({'target':x.T.copy(),'past_covariates':cov(dates[t-512:t]),'future_covariates':cov(future)})
  q,_=pipe.predict_quantiles(items,prediction_length=64,quantile_levels=levels,batch_size=112,context_length=512,cross_learning=False);q=np.stack([v.cpu().numpy().transpose(1,0,2) for v in q]).astype(float)
  p=np.sinh(extract_point(q,levels,'full_integral')) if cfg['inverse_order']=='after_point' else extract_point(np.sinh(q),levels,'full_integral');assert np.isfinite(p).all();ps.append(p.astype(np.float32))
 del pipe;gc.collect();torch.cuda.empty_cache();return np.concatenate(ps)
def run(mode,tag=None):
 assert mode in ('frozen','native','inversepoint');suffix='_'+tag if tag else '';name=f'20261105_asinhtool_{mode}{suffix}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=False);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);guard(3600);torch.set_num_threads(4);torch.manual_seed(2021);torch.cuda.manual_seed_all(2021);np.random.seed(2021);z=np.load(RUN/'cache/trainval_z.npy');dates=pd.to_datetime(json.loads((RUN/'cache/dates.json').read_text()));levels=json.loads((RUN/'models/chronos2_29ec3766/config.json').read_text())['chronos_config']['quantiles'];parent=f'20261105_asinhtool_native{suffix}_mse' if mode!='frozen' else None;cfg=dict(engine='asinh_tool_v1',mode=mode,checkpoint_parent=parent,source_ids=['run:'+parent] if mode=='inversepoint' else [],transform='arcsinh each standardized variable; sinh inverse; no physical unit assertion',inverse_order='after_point' if mode=='inversepoint' else 'before_integral',point_quantile_levels=levels,point_rule='full_integral',training_rows=[0,8640] if mode=='native' else [],calendar='hour',runtime_llm=False,code_revision=git_head());save_json(out/'config.json',cfg);rt=np.sinh(np.arcsinh(z[:8640].astype(float)));assert np.allclose(rt,z[:8640],rtol=1e-12,atol=1e-12);save_json(out/'transform_contract.json',dict(train_roundtrip_max_delta=float(abs(rt-z[:8640]).max()),strictly_monotone=True,all7_retained=True,test_access=False));start=time.time()
 if mode=='native':
  lat=np.arcsinh(z[:8640].astype(float)).astype(np.float32);inputs=from_list_of_dicts([{'target':lat.T.copy(),'past_covariates':cov(dates[:8640])}],prediction_length=64,known_covariates_names=['hour_sin','hour_cos']);pipe=Chronos2Pipeline.from_pretrained(str(RUN/'models/chronos2_29ec3766'),device_map='cuda:0');trained=pipe.fit(inputs,prediction_length=64,validation_inputs=None,finetune_mode='full',context_length=512,learning_rate=5e-7,num_steps=500,batch_size=112,min_past=512,output_dir=out/'trainer',callbacks=[Deadline()],seed=2021,data_seed=2021,disable_tqdm=True,logging_steps=100,bf16=True,tf32=False);del pipe,trained;gc.collect();torch.cuda.empty_cache();cfg.update(finetune_steps=500,finetune_rate=5e-7,seed=2021);save_json(out/'config.json',cfg)
 results={}
 for split,ts in json.loads((RUN/'cache/window_manifest.json').read_text()).items():
  p=predict(z,ts,cfg,dates);np.save(out/f'{split}_predictions.npy',p);results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'))
 save_json(out/'metrics.json',dict(candidate=name,metrics=results,status='evaluated',elapsed_seconds=time.time()-start));print(name,{k:v['mse'] for k,v in results.items()},flush=True)
if __name__=='__main__':run(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else None)
