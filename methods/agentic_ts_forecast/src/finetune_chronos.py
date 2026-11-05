"""Train-only Chronos2 fine-tuning as a numeric tool; full fixed validation afterward."""
from common import *
import torch,sys
from chronos import Chronos2Pipeline
from transformers import TrainerCallback
class Deadline(TrainerCallback):
 def on_step_begin(self,args,state,control,**kwargs):guard(900)
def run(rate,tag):
 name=f'20261105_chronosft_{tag}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True)
 subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True)
 guard(1800);torch.set_num_threads(4);np.random.seed(2021);torch.manual_seed(2021);torch.cuda.manual_seed_all(2021)
 cfg={'learning_rate':rate,'steps':500,'context_length':512,'prediction_length':64,'min_past':512,'batch_size_including_variates':112,'seed':2021,'mode':'full','training_rows':[0,8640],'checkpoint_selection':'fixed final step; full validation evaluates candidate afterward','code_revision':git_head()};save_json(out/'config.json',cfg)
 z=np.load(RUN/'cache/trainval_z.npy');pipe=Chronos2Pipeline.from_pretrained(str(RUN/'models/chronos2_29ec3766'),device_map='cuda:0');start=time.time()
 trained=pipe.fit([torch.from_numpy(z[:8640].T.copy())],prediction_length=64,validation_inputs=None,finetune_mode='full',context_length=512,learning_rate=rate,num_steps=500,batch_size=112,min_past=512,output_dir=out/'trainer',callbacks=[Deadline()],seed=2021,data_seed=2021,disable_tqdm=True,logging_steps=50,bf16=True,tf32=False)
 del pipe,trained;torch.cuda.empty_cache();trained=Chronos2Pipeline.from_pretrained(str(out/'trainer/finetuned-ckpt'),device_map='cuda:0');trained.model.eval();assert not trained.model.training;cfg['postfit_inference']='reload saved checkpoint, explicit eval';save_json(out/'config.json',cfg);manifest=json.loads((RUN/'cache/window_manifest.json').read_text());results={}
 for split,ts in manifest.items():
  ps=[];qs=[]
  for i in range(0,len(ts),16):
   guard(600);x,_=windows(z,ts[i:i+16]);q,p=trained.predict_quantiles([torch.from_numpy(a.T.copy()) for a in x],prediction_length=64,batch_size=112,context_length=512,cross_learning=False);ps.extend([a.cpu().numpy().T for a in p]);qs.extend([a.cpu().numpy().transpose(1,0,2) for a in q])
  p=np.stack(ps).astype(np.float32);q=np.stack(qs).astype(np.float32);y=np.load(RUN/'cache'/f'{split}_targets.npy');results[split]=metric(p,y);results[split]['quantile_average_mse']=metric(q.mean(-1),y)['mse'];np.save(out/f'{split}_predictions.npy',p);np.save(out/f'{split}_quantiles.npy',q);save_json(out/'partial_metrics.json',results)
 save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated','code_revision':cfg['code_revision'],'elapsed_seconds':time.time()-start})
 with (RUN/'experiment_index.jsonl').open('a') as f:f.write(json.dumps({'candidate':name,'val_mse':results['val']['mse'],'status':'evaluated','kind':'finetuned_tsfm_tool'})+'\n')
 print(name,{s:r['mse'] for s,r in results.items()},flush=True)
if __name__=='__main__':run(float(sys.argv[1]),sys.argv[2])
