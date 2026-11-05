from common import *
import sys
name='20261105_chronos2_quantileavg_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True)
subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True)
results={}
for split in ('dev','confirm','val'):
 guard(600);q=np.load(ROOT/'runs/20261105_chronos2_median_mse'/f'{split}_quantiles.npy');p=q.mean(-1);y=np.load(RUN/'cache'/f'{split}_targets.npy');results[split]=metric(p,y);np.save(RUN/'cache'/f'{split}_chronosavg.npy',p)
save_json(out/'config.json',{'point':'arithmetic average of .1-.9 quantiles; heuristic, not exact expectation','code_revision':git_head()});save_json(out/'metrics.json',results);print(results['val']['mse'],flush=True)
