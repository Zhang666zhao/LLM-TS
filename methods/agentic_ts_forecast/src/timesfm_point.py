from common import *
import sys
name='20261105_timesfm3_quantileavg_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True)
subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True)
results={}
for split in ('dev','confirm','val'):
 guard(600);q=np.load(ROOT/'runs/20261105_timesfm3_sym1_mse'/f'{split}_quantiles.npy');p=q.mean(-1);results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'));np.save(RUN/'cache'/f'{split}_timesfmavg.npy',p)
save_json(out/'config.json',{'parent':'TimesFM symmetric averaging','point':'arithmetic average of .1-.9 quantiles, not exact expectation','code_revision':git_head()});save_json(out/'metrics.json',results);print(results['val']['mse'],flush=True)
