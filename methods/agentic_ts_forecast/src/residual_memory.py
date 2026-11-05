"""Frozen train-only kNN residual adaptation; query labels never enter retrieval."""
from common import *
import sys
from hybrid_router import descriptor

def run(cfg,name):
 out=ROOT/'runs'/name;out.mkdir(exist_ok=True)
 subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True)
 bank=dict(np.load(RUN/'cache/train_error_bank4.npz'));manifest=json.loads((RUN/'cache/window_manifest.json').read_text());z=np.load(RUN/'cache/trainval_z.npy');results={};records={}
 # Use pretrained tool residuals to avoid training-fitted backbone error bias.
 resid=bank['targets']-bank['predictions'][:,3];basefeat=bank['features']/bank['feature_scale'];bs=bank['predictions'][:,3].reshape(-1,8,8,7).mean(2)
 bs=(bs-bs.mean(1,keepdims=True)).reshape(len(bs),-1);bs_scale=np.maximum(bs.std(0),.1);bs/=bs_scale
 for split,ts in manifest.items():
  pred=np.load(RUN/'cache'/f'{split}_chronos2.npy');dl=np.load(RUN/'cache'/f'{split}_dlinear.npy');p=[];inds=[]
  for i,t in enumerate(ts):
   guard(600);x=z[t-512:t];valid=np.where(bank['origins']+64<=t-512)[0];q=descriptor(x)/bank['feature_scale'];d=np.mean((basefeat[valid]-q)**2,axis=1)
   if cfg['retrieval']=='xb':
    qb=pred[i].reshape(8,8,7).mean(1);qb=((qb-qb.mean(0))/bs_scale.reshape(8,7)).reshape(-1);d+=np.mean((bs[valid]-qb)**2,axis=1)
   ix=valid[np.argsort(d)[:cfg['k']]];inds.append(ix.tolist());cor=resid[ix].mean(0) if len(ix) else np.zeros((64,7))
   if cfg['shape']=='level':cor=np.broadcast_to(cor.mean(0),(64,7))
   # Limit correction using only observed query variability.
   bound=.5*np.maximum(x.std(0),.05);cor=np.clip(cor,-bound,bound)
   p.append(.5*dl[i]+.5*(pred[i]+cfg['strength']*cor))
  p=np.stack(p).astype(np.float32);results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'));np.save(out/f'{split}_predictions.npy',p);records[split]=inds
 save_json(out/'config.json',dict(cfg,code_revision=git_head(),memory_sha256=hashlib.sha256((RUN/'cache/train_error_bank4.npz').read_bytes()).hexdigest(),backbone_residual='frozen Chronos median'))
 save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated'});save_json(out/'retrieval_indices.json',records)
 with (RUN/'experiment_index.jsonl').open('a') as f:f.write(json.dumps({'candidate':name,'status':'evaluated','val_mse':results['val']['mse'],'kind':'frozen_residual_memory'})+'\n')
 print(name,{s:r['mse'] for s,r in results.items()},flush=True)
if __name__=='__main__':run(json.loads(sys.argv[1]),sys.argv[2])
