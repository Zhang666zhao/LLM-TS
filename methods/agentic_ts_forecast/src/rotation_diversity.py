"""Matched full-CDF numeric tools: rotate basis only, keep training seed fixed."""
from common import *
from transformed_tool import transform,predict
import sys,torch,gc
from chronos import Chronos2Pipeline
from transformers import TrainerCallback
class Deadline(TrainerCallback):
 def on_step_begin(self,args,state,control,**kwargs):guard(1800)
def run(seed):
 assert seed in (2022,2023,2024)
 name=f'20261105_rotationseed_{seed}_mse';out=ROOT/'runs'/name;assert not (out/'config.json').exists();out.mkdir(exist_ok=True)
 subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);guard(3600)
 torch.set_num_threads(4);torch.manual_seed(2021);torch.cuda.manual_seed_all(2021);np.random.seed(2021)
 z=np.load(RUN/'cache/trainval_z.npy');levels=json.loads((RUN/'models/chronos2_29ec3766/config.json').read_text())['chronos_config']['quantiles'];cfg=dict(transform(z[:8640].astype(float),'random',rotation_seed=seed),engine='transformed_tool_v1',mode='random',rotation_seed=seed,training_rows=[0,8640],point_rule='full_integral',point_quantile_levels=levels,point='21-quantile CDF trapezoid integral, flat endpoint tails, inverse POINT transform',checkpoint_parent=name,finetune_steps=500,finetune_rate=5e-7,seed=2021,checkpoint_selection='fixed step500, no official validation selection within training',code_revision=git_head());save_json(out/'config.json',cfg)
 latent=(z[:8640].astype(float)@np.asarray(cfg['matrix'])+np.asarray(cfg['bias'])).astype(np.float32);pipe=Chronos2Pipeline.from_pretrained(str(RUN/'models/chronos2_29ec3766'),device_map='cuda:0');start=time.time();trained=pipe.fit([torch.from_numpy(latent.T.copy())],prediction_length=64,validation_inputs=None,finetune_mode='full',context_length=512,learning_rate=5e-7,num_steps=500,batch_size=112,min_past=512,output_dir=out/'trainer',callbacks=[Deadline()],seed=2021,data_seed=2021,disable_tqdm=True,logging_steps=100,bf16=True,tf32=False);del pipe,trained;gc.collect();torch.cuda.empty_cache();cfg['postfit_inference']='reload saved checkpoint and explicit eval';save_json(out/'config.json',cfg)
 results={}
 for split,ts in json.loads((RUN/'cache/window_manifest.json').read_text()).items():
  p=predict(z,ts,cfg);np.save(out/f'{split}_predictions.npy',p);results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'));save_json(out/'partial_metrics.json',results)
 save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated','elapsed_seconds':time.time()-start});print(name,{s:m['mse'] for s,m in results.items()},flush=True)
if __name__=='__main__':run(int(sys.argv[1]))
