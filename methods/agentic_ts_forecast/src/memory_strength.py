from common import *
import sys
strength=float(sys.argv[1]);tag=sys.argv[2];name=f'20261105_memorystrength_{tag}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True)
subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True)
results={}
for split in ('dev','confirm','val'):
 guard(900);prior=sum(np.load(RUN/'cache'/f'{split}_{n}.npy') for n in ('dlinear','chronosavg','timesfmavg'))/3;corrected=np.load(ROOT/'runs/20261105_densememory_xbphase_mse'/f'{split}_predictions.npy');p=(prior+strength*4*(corrected-prior)).astype(np.float32);np.save(out/f'{split}_predictions.npy',p);results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'))
save_json(out/'config.json',{'engine':'dense_memory_strength_v1','strength':strength,'parent':'20261105_densememory_xbphase_mse','code_revision':git_head()});save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated'});print(name,{s:m['mse'] for s,m in results.items()},flush=True)
