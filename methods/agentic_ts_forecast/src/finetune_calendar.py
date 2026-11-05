"""Train-only Chronos2 fine-tuning as a numeric tool; full fixed validation afterward."""
from common import *
import torch,sys,pandas as pd
from chronos.chronos2.preprocess import from_list_of_dicts
def cov(d):
 h=np.asarray(d.hour,dtype=float);return {"hour_sin":np.sin(2*np.pi*h/24).astype(np.float32),"hour_cos":np.cos(2*np.pi*h/24).astype(np.float32)}
from chronos import Chronos2Pipeline
from transformers import TrainerCallback
class Deadline(TrainerCallback):
 def on_step_begin(self,args,state,control,**kwargs):guard(900)
def run(rate,tag,steps=500):
 name=f'20261105_chronoscalft_{tag}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True)
 subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True)
 guard(1800);torch.set_num_threads(4);np.random.seed(2021);torch.manual_seed(2021);torch.cuda.manual_seed_all(2021)
 cfg={'learning_rate':rate,'steps':steps,'context_length':512,'prediction_length':64,'min_past':512,'batch_size_including_variates':112,'seed':2021,'mode':'full','calendar':'hour','known_future_calendar':True,'engine':'calendar_finetune_v1','point':'average nine .1-.9 quantiles','training_rows':[0,8640],'checkpoint_selection':'fixed final step; full validation evaluates candidate afterward','code_revision':git_head()};save_json(out/'config.json',cfg)
 z=np.load(RUN/'cache/trainval_z.npy');pipe=Chronos2Pipeline.from_pretrained(str(RUN/'models/chronos2_29ec3766'),device_map='cuda:0');start=time.time()
 dates=pd.to_datetime(json.loads((RUN/'cache/dates.json').read_text()));train_inputs=from_list_of_dicts([{'target':z[:8640].T.copy(),'past_covariates':cov(dates[:8640])}],prediction_length=64,known_covariates_names=['hour_sin','hour_cos']);assert train_inputs[0]['context'].shape[-1]==8640
 trained=pipe.fit(train_inputs,prediction_length=64,validation_inputs=None,finetune_mode='full',context_length=512,learning_rate=rate,num_steps=steps,batch_size=112,min_past=512,output_dir=out/'trainer',callbacks=[Deadline()],seed=2021,data_seed=2021,disable_tqdm=True,logging_steps=50,bf16=True,tf32=False)
 del pipe,trained;torch.cuda.empty_cache();trained=Chronos2Pipeline.from_pretrained(str(out/'trainer/finetuned-ckpt'),device_map='cuda:0');trained.model.eval();assert not trained.model.training;cfg['postfit_inference']='reload saved checkpoint, explicit eval';save_json(out/'config.json',cfg);manifest=json.loads((RUN/'cache/window_manifest.json').read_text());results={}
 for split,ts in manifest.items():
  ps=[];qs=[]
  for i in range(0,len(ts),16):
   guard(600);inputs=[{'target':z[t-512:t].T.copy(),'past_covariates':cov(dates[t-512:t]),'future_covariates':cov(pd.date_range(dates[t-1]+pd.Timedelta(hours=1),periods=64,freq='h'))} for t in ts[i:i+16]];q,p=trained.predict_quantiles(inputs,prediction_length=64,batch_size=112,context_length=512,cross_learning=False);ps.extend([a.cpu().numpy().T for a in p]);qs.extend([a.cpu().numpy().transpose(1,0,2) for a in q])
  median=np.stack(ps).astype(np.float32);q=np.stack(qs).astype(np.float32);p=q.mean(-1);y=np.load(RUN/'cache'/f'{split}_targets.npy');results[split]=metric(p,y);results[split]['median_mse']=metric(median,y)['mse'];np.save(out/f'{split}_predictions.npy',p);np.save(out/f'{split}_quantiles.npy',q);save_json(out/'partial_metrics.json',results)
 save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated','code_revision':cfg['code_revision'],'elapsed_seconds':time.time()-start})
 with (RUN/'experiment_index.jsonl').open('a') as f:f.write(json.dumps({'candidate':name,'val_mse':results['val']['mse'],'status':'evaluated','kind':'finetuned_tsfm_tool'})+'\n')
 print(name,{s:r['mse'] for s,r in results.items()},flush=True)
if __name__=='__main__':run(float(sys.argv[1]),sys.argv[2],int(sys.argv[3]) if len(sys.argv)>3 else 500)
