"""Predeclared convex tool controls, no LLM and no target fitting."""
from common import *
import sys
cfg=json.loads(sys.argv[1]);name=sys.argv[2];out=ROOT/'runs'/name;out.mkdir(exist_ok=True)
subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True)
results={}
for split in ('dev','confirm','val'):
 guard(600);y=np.load(RUN/'cache'/f'{split}_targets.npy');p=sum(float(w)*np.load(RUN/'cache'/f'{split}_{m}.npy') for m,w in cfg['weights'].items());results[split]=metric(p,y);np.save(out/f'{split}_predictions.npy',p)
assert abs(sum(cfg['weights'].values())-1)<1e-6
save_json(out/'config.json',cfg);save_json(out/'metrics.json',{'candidate':name,'code_revision':git_head(),'metrics':results,'status':'evaluated','llm':False})
with (RUN/'experiment_index.jsonl').open('a') as f:f.write(json.dumps({'candidate':name,'status':'evaluated','val_mse':results['val']['mse'],'kind':'no_llm_ensemble_control'})+'\n')
print(name,results['val']['mse'],flush=True)
