"""Invertible view plus deterministic hour covariates; train-only native adaptation."""
from common import *
from transformed_tool import transform,predict
import sys,torch,pandas as pd,gc
from chronos import Chronos2Pipeline
from chronos.chronos2.preprocess import from_list_of_dicts
from transformers import TrainerCallback
class Deadline(TrainerCallback):
 def on_step_begin(self,args,state,control,**kwargs):guard(1800)
def cov(d):
 h=np.asarray(d.hour,dtype=float);return {'hour_sin':np.sin(2*np.pi*h/24).astype(np.float32),'hour_cos':np.cos(2*np.pi*h/24).astype(np.float32)}
def run(mode,seed=2021,trainstart=0,tag=None):
 assert 0<=trainstart<=4096 and seed in (2021,2022,2023,2024);assert mode in ('semantic','pca','random');name=f'20261105_transformedcalft_{mode}_mse' if tag is None else f'20261105_calendarvariant_{tag}_mse';out=ROOT/'runs'/name;assert not (out/'config.json').exists();out.mkdir(exist_ok=True);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);guard(3600);torch.set_num_threads(4);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed)
 z=np.load(RUN/'cache/trainval_z.npy');dates=pd.to_datetime(json.loads((RUN/'cache/dates.json').read_text()));levels=json.loads((RUN/'models/chronos2_29ec3766/config.json').read_text())['chronos_config']['quantiles'];cfg=dict(transform(z[:8640].astype(float),mode),engine='transformed_tool_v1',mode=mode,calendar='hour',known_future_calendar=True,timezone='unknown record-clock phase only',training_rows=[trainstart,8640],transform_fit_rows=[0,8640],checkpoint_parent=name,point_rule='full_integral',point_quantile_levels=levels,finetune_steps=500,finetune_rate=5e-7,seed=seed,batch_size_including_variates=112,comparison_caveat='Same native update count and variate batch budget as no-calendar view; adding2 covariates may change effective examples per update. No exact matched sample-order claim.',code_revision=git_head());save_json(out/'config.json',cfg)
 latent=(z[:8640].astype(float)@np.asarray(cfg['matrix'])+np.asarray(cfg['bias'])).astype(np.float32);inputs=from_list_of_dicts([{'target':latent[trainstart:].T.copy(),'past_covariates':cov(dates[trainstart:8640])}],prediction_length=64,known_covariates_names=['hour_sin','hour_cos']);assert inputs[0]['context'].shape[-1]==8640-trainstart;pipe=Chronos2Pipeline.from_pretrained(str(RUN/'models/chronos2_29ec3766'),device_map='cuda:0');start=time.time();trained=pipe.fit(inputs,prediction_length=64,validation_inputs=None,finetune_mode='full',context_length=512,learning_rate=5e-7,num_steps=500,batch_size=112,min_past=512,output_dir=out/'trainer',callbacks=[Deadline()],seed=seed,data_seed=seed,disable_tqdm=True,logging_steps=100,bf16=True,tf32=False);del pipe,trained;gc.collect();torch.cuda.empty_cache();results={}
 for split,ts in json.loads((RUN/'cache/window_manifest.json').read_text()).items():
  p=predict(z,ts,cfg,dates=dates);assert np.isfinite(p).all();np.save(out/f'{split}_predictions.npy',p);results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'));save_json(out/'partial_metrics.json',results)
 save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated','elapsed_seconds':time.time()-start});print(name,{s:m['mse'] for s,m in results.items()},flush=True)
if __name__=='__main__':run(sys.argv[1],int(sys.argv[2]) if len(sys.argv)>2 else 2021,int(sys.argv[3]) if len(sys.argv)>3 else 0,sys.argv[4] if len(sys.argv)>4 else None)
