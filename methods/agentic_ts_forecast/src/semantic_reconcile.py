"""Soft empirical raw-scale load relations, never asserted physical identities."""
from common import *
import sys
cfg=json.loads(sys.argv[1]);name=sys.argv[2];out=ROOT/'runs'/name;out.mkdir(exist_ok=True)
subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True)
scale=np.asarray(CONTRACT['scaler_scale']);A=np.zeros((2,7));A[0,[0,2,4]]=scale[[0,2,4]]*np.array([1,-1,-1]);A[1,[1,3,5]]=scale[[1,3,5]]*np.array([1,-1,-1]);P=A.T@np.linalg.inv(A@A.T)
z=np.load(RUN/'cache/trainval_z.npy');manifest=json.loads((RUN/'cache/window_manifest.json').read_text());results={}
for split,ts in manifest.items():
 guard(600);p=np.load(ROOT/'runs'/cfg['parent']/f'{split}_predictions.npy').astype(np.float64)
 target=np.stack([z[t-cfg['history']:t].mean(0)@A.T for t in ts]) if cfg['history'] else np.zeros((len(ts),2))
 error=p@A.T-target[:,None];p=(p-cfg['strength']*(error@P.T)).astype(np.float32)
 results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'));np.save(out/f'{split}_predictions.npy',p)
save_json(out/'config.json',dict(cfg,code_revision=git_head(),relation='soft empirical load balance, not physical law',standardized_relation_matrix=A.tolist(),target='training mean' if not cfg['history'] else 'observed recent residual mean'))
save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated'})
with (RUN/'experiment_index.jsonl').open('a') as f:f.write(json.dumps({'candidate':name,'val_mse':results['val']['mse'],'status':'evaluated','kind':'semantic_soft_relation'})+'\n')
print(name,{s:r['mse'] for s,r in results.items()},flush=True)
