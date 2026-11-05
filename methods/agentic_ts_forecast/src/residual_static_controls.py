"""Matched static controls for the train-only dynamic residual correction."""
from common import *
from blend_sources import get_source
from view_residual_memory import expand,PRIOR
import sys,torch
PARENT='20261105_auto_p08b001c1_mse';MEMORY='20261105_viewresidual_ridge10_mse'
def run(mode):
 assert mode in ('empirical','intercept','mean_dynamic');name=f'20261105_residualstatic_{mode}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);guard(1800);path=RUN/'cache/view_prefix_memory.npz';d=np.load(path);state=torch.load(ROOT/'runs'/MEMORY/'best.pth',map_location='cpu');base=(d['predictions']*PRIOR[None,:,None,None]).sum(1)
 if mode=='empirical':bins=(d['targets']-base).reshape(len(base),8,8,7).mean((0,2))
 elif mode=='intercept':bins=state['coef'][0].reshape(8,7)
 else:
  a=np.clip((d['features']-state['mean'])/state['scale'],-5,5);a=np.c_[np.ones(len(a)),a];bins=(a@state['coef']).mean(0).reshape(8,7)
 c=expand(bins[None])[0];cfg={'engine':'constant_residual_v1','source_ids':['run:'+PARENT,'run:'+MEMORY],'mode':mode,'correction':c.tolist(),'strength':.25,'training_memory':str(path),'training_memory_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'caveat':'Constant correction computed from training-only memory; full-train dev/confirm in-sample','code_revision':git_head()};save_json(out/'config.json',cfg);results={}
 for split in ('dev','confirm','val'):
  p=(get_source('run:'+PARENT,split).astype(float)+.25*c[None]).astype(np.float32);np.save(out/f'{split}_predictions.npy',p);results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'))
 save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated'});print(name,{s:m['mse'] for s,m in results.items()},flush=True)
if __name__=='__main__':run(sys.argv[1])
