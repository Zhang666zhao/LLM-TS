from common import *
from self_memory import arrays,BANK,CHOICES
from hybrid_router import descriptor
import sys
k=int(sys.argv[1]);name=f'20261105_selfnumeric_k{k}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);b=dict(np.load(BANK));z=np.load(RUN/'cache/trainval_z.npy');manifest=json.loads((RUN/'cache/window_manifest.json').read_text());results={}
for split in ('confirm','val'):
 ts=manifest[split];p,u=arrays(split);alpha=[]
 for i,t in enumerate(ts):
  guard(600);x=z[t-512:t];f=np.concatenate([descriptor(x),u[i].mean(0),u[i].std(0)]);valid=np.where(b['origins']+64<=t-512)[0];d=np.mean(((b['features'][valid]-f)/b['feature_scale'])**2,1);ix=valid[np.argsort(d)[:k]];assert len(ix)==k;alpha.append(CHOICES[b['cost'][ix].mean(0).argmin(0)])
 a=np.stack(alpha);pr=(p+a[:,None]*u).astype(np.float32);results[split]=metric(pr,np.load(RUN/'cache'/f'{split}_targets.npy'));np.save(out/f'{split}_predictions.npy',pr);np.save(out/f'{split}_strength.npy',a)
save_json(out/'config.json',{'engine':'self_error_numeric_v1','k':k,'memory_cases':25,'selection':'argmin average counterfactual TRAINING case MSE among four strengths','same_memory_as_llm':True,'code_revision':git_head()});save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated'});print(name,{s:m['mse'] for s,m in results.items()},flush=True)
