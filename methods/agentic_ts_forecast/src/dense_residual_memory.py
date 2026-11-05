"""Dense episodic and clock-aligned pattern memory from frozen pretrained errors."""
from common import *
from hybrid_router import descriptor
import sys

def run(mode):
 name=f'20261105_densememory_{mode}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True)
 subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True)
 guard(900);rev=git_head();start=time.time();z=np.load(RUN/'cache/trainval_z.npy');ca=np.load(RUN/'cache/dense_chronos_train.npz');ta=np.load(RUN/'cache/dense_timesfm_train.npz');origins=ca['origins'];assert np.array_equal(origins,ta['origins']);tx,ty=windows(z[:8640],origins)
 tp=np.stack([ca['predictions'],ta['predictions']],1);tq=np.stack([ca['quantiles'],ta['quantiles']],1);err=ty[:,None]-tp;fx=np.stack([descriptor(x) for x in tx]);sx=np.maximum(fx.std(0),.1)
 def bfeatures(x,p,q):
  recent=x[:,-96:].mean(1);f=[]
  for m in range(2):f.extend([p[:,m].mean(1)-recent,p[:,m].std(1),(q[:,m,:,:,-1]-q[:,m,:,:,0]).mean(1),(p[:,m]-q[:,m,:,:,4]).mean(1)])
  return np.concatenate(f,axis=1)
 fb=bfeatures(tx,tp,tq);sb=np.maximum(fb.std(0),.1);patterns=np.empty((len(origins),2,24,7),np.float32)
 for i,t in enumerate(origins):
  hours=(t+np.arange(64))%24
  for h in range(24):patterns[i,:,h]=err[i][:,hours==h].mean(1)
 manifest=json.loads((RUN/'cache/window_manifest.json').read_text());results={};neighbors={}
 for split,ts in manifest.items():
  x,y=windows(z,ts);cq=np.load(ROOT/'runs/20261105_chronos2_median_mse'/f'{split}_quantiles.npy');tq=np.load(ROOT/'runs/20261105_timesfm3_sym1_mse'/f'{split}_quantiles.npy');q=np.stack([cq,tq],1);p=q.mean(-1);dl=np.load(RUN/'cache'/f'{split}_dlinear.npy');xb=np.stack([descriptor(a) for a in x]);bb=bfeatures(x,p,q);ps=[];selected=[]
  for i,t in enumerate(ts):
   guard(600);valid=np.where(origins+64<=t-512)[0];dist=np.mean(((fx[valid]-xb[i])/sx)**2,axis=1)
   if mode!='xlevel':dist+=np.mean(((fb[valid]-bb[i])/sb)**2,axis=1)
   chosen=[]
   for j in valid[np.argsort(dist)]:
    if all(abs(int(origins[j])-int(origins[k]))>=64 for k in chosen):chosen.append(j)
    if len(chosen)==8:break
   assert chosen;selected.append(origins[chosen].tolist())
   if mode=='xbphase':cor=patterns[chosen].mean(0)[:,(t+np.arange(64))%24]
   else:cor=np.broadcast_to(err[chosen].mean((0,2))[:,None],(2,64,7))
   bound=.5*np.maximum(x[i].std(0),.05);cor=np.clip(cor,-bound,bound);corrected=p[i]+.25*cor;ps.append((dl[i]+corrected.sum(0))/3)
  pr=np.stack(ps).astype(np.float32);results[split]=metric(pr,y);np.save(out/f'{split}_predictions.npy',pr);neighbors[split]=selected
 save_json(out/'config.json',{'mode':mode,'memory_cases':len(origins),'memory_stride':8,'neighbor_count':8,'neighbor_min_origin_spacing':64,'query_separation':'case end<=query origin-512','strength':.25,'parent':'equal DLinear/Chronosavg/TimesFMavg','target_error':'pretrained tools only','phase_alignment':'relative hourly record phase; no location or timezone inferred','training_screen':'diagnostic, feature scales fit full official training','code_revision':rev})
 save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated','elapsed_seconds':time.time()-start});save_json(out/'neighbor_origins.json',neighbors)
 with (RUN/'experiment_index.jsonl').open('a') as f:f.write(json.dumps({'candidate':name,'val_mse':results['val']['mse'],'status':'evaluated','kind':'dense_error_memory'})+'\n')
 print(name,{k:v['mse'] for k,v in results.items()},flush=True)
if __name__=='__main__':run(sys.argv[1])
