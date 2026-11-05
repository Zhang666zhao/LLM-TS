"""Read-only inference with fixed non-oracle training checkpoints; never read test."""
from common import *
import sys,torch,pandas as pd
PROJECT=ROOT.parent/'TSRAG'
sys.path.insert(0,str(PROJECT/'research/multivariate_oracle_six'))
from conditioning import backbone
method=sys.argv[1]
name='20261105_tool_'+method+'_mse'
out=ROOT/'runs'/name;out.mkdir(exist_ok=True)
subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True)
guard(600);torch.set_num_threads(4)
net,cfg,audit=backbone(PROJECT,method,'ETTh1',7)
ckpt=PROJECT/'runs/20261014_six_multivariate_oracle_v1/ETTh1/jobs'/f'{method}_base_zero'/'best.pth'
state=torch.load(ckpt,map_location='cpu');net.load_state_dict({k.removeprefix('backbone.'):v for k,v in state.items()});net.cuda(0).eval()
z=np.load(RUN/'cache/trainval_z.npy');dt=pd.Series(pd.to_datetime(json.loads((RUN/'cache/dates.json').read_text()))).dt
marks=np.stack([dt.hour.to_numpy()/23-.5,dt.dayofweek.to_numpy()/6-.5,(dt.day.to_numpy()-1)/30-.5,(dt.dayofyear.to_numpy()-1)/365-.5],-1).astype(np.float32)
manifest=json.loads((RUN/'cache/window_manifest.json').read_text());results={}
for split in ('dev','confirm','val'):
 ts=manifest[split];pred=[]
 with torch.no_grad():
  for i in range(0,len(ts),32):
   guard(600);batch=ts[i:i+32];x=torch.tensor(np.stack([z[t-512:t] for t in batch]),device='cuda:0');m=torch.tensor(np.stack([marks[t-512:t] for t in batch]),device='cuda:0')
   p=net(x,m,None,None) if method in ('itransformer','fredf') else net(x)
   pred.append(p.cpu().numpy())
 pred=np.concatenate(pred);y=np.load(RUN/'cache'/f'{split}_targets.npy');results[split]=metric(pred,y);np.save(RUN/'cache'/f'{split}_{method}.npy',pred)
config={'method':method,'checkpoint':str(ckpt),'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),'training_config':cfg,'audit':audit,'code_revision':git_head(),'no_oracle':True,'training_internal_scores_in_sample':True}
save_json(out/'config.json',config);save_json(out/'metrics.json',results);print(json.dumps({s:r['mse'] for s,r in results.items()}),flush=True)
