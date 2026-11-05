"""Declared point extraction variants from saved quantiles; no model retraining."""
from common import *
import sys
parent,tag=sys.argv[1:3]
name='20261105_point_'+tag+'_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True)
subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True)
results={}
for split in ('dev','confirm','val'):
 guard(900);q=np.load(ROOT/'runs'/parent/f'{split}_quantiles.npy');p=q.mean(-1).astype(np.float32);np.save(out/f'{split}_predictions.npy',p);results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'))
save_json(out/'config.json',{'engine':'quantile_average_v1','parent':parent,'point':'arithmetic average of nine .1-.9 quantiles; heuristic not exact expectation','code_revision':git_head()})
save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated'})
print(name,{s:m['mse'] for s,m in results.items()},flush=True)
