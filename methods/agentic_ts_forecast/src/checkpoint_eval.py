"""New explicit eval-mode candidate from existing immutable fitted weights, no retraining."""
from common import *
import torch,sys
from chronos import Chronos2Pipeline
parent=sys.argv[1];tag=sys.argv[2];name=f'20261105_reloadedft_{tag}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True)
guard(900);torch.set_num_threads(4);checkpoint=ROOT/'runs'/parent/'trainer/finetuned-ckpt';pipe=Chronos2Pipeline.from_pretrained(str(checkpoint),device_map='cuda:0');pipe.model.eval();assert not pipe.model.training;z=np.load(RUN/'cache/trainval_z.npy');manifest=json.loads((RUN/'cache/window_manifest.json').read_text());start=time.time();results={}
# Repeated inference must be deterministic before any candidate scoring.
x=torch.from_numpy(z[512:1024].T.copy());q1=pipe.predict_quantiles([x],prediction_length=64,batch_size=112,context_length=512,cross_learning=False)[0][0];q2=pipe.predict_quantiles([x],prediction_length=64,batch_size=112,context_length=512,cross_learning=False)[0][0];repeat_delta=float((q1-q2).abs().max());assert repeat_delta<1e-6
for split,ts in manifest.items():
 qs=[]
 for i in range(0,len(ts),16):
  guard(600);q,_=pipe.predict_quantiles([torch.from_numpy(z[t-512:t].T.copy()) for t in ts[i:i+16]],prediction_length=64,batch_size=112,context_length=512,cross_learning=False);qs.extend([a.cpu().numpy().transpose(1,0,2) for a in q])
 q=np.stack(qs).astype(np.float32);p=q.mean(-1);y=np.load(RUN/'cache'/f'{split}_targets.npy');results[split]=metric(p,y);results[split]['median_mse']=metric(q[:,:,:,4],y)['mse'];np.save(out/f'{split}_predictions.npy',p);np.save(out/f'{split}_quantiles.npy',q);save_json(out/'partial_metrics.json',results)
save_json(out/'config.json',{'engine':'reloaded_finetune_v1','checkpoint_parent':parent,'checkpoint_sha256':hashlib.sha256((checkpoint/'model.safetensors').read_bytes()).hexdigest(),'inference_training_mode':pipe.model.training,'point':'average nine quantiles .1-.9','repeat_max_delta':repeat_delta,'no_retraining':True,'supersedes_parent_evaluation_only':True,'code_revision':git_head()});save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated','elapsed_seconds':time.time()-start});print(name,{s:m['mse'] for s,m in results.items()},flush=True)
