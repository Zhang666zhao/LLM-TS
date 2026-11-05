"""Train-only error retrieval enriched by distribution shape and cross-variable dynamics."""
from common import *
from hybrid_router import descriptor
import sys

def enriched(x,mode):
 x=x[-168:];s=np.maximum(x.std(0),.05);a=np.clip((x-x.mean(0))/s,-10,10);parts=[]
 if mode in ('moments','both'):
  parts.extend([np.mean(a**3,0),np.mean(a**4,0)-3,*np.quantile(a,[.1,.5,.9],axis=0),np.abs(np.diff(x,axis=0)).mean(0)/s])
 if mode in ('coupled','both'):
  for v in (x,np.diff(x,axis=0)):
   v=v-v.mean(0);norm=np.maximum(np.sqrt((v*v).sum(0)),1e-6);corr=(v.T@v)/(norm[:,None]*norm[None]);parts.append(corr[np.triu_indices(7,1)])
  raw=x*np.asarray(CONTRACT['scaler_scale'])+np.asarray(CONTRACT['scaler_mean']);rel=np.stack([raw[:,0]-raw[:,2]-raw[:,4],raw[:,1]-raw[:,3]-raw[:,5]],1);parts.extend([rel.mean(0),rel.std(0),np.array([(np.abs(raw[:,:6])<1e-5).all(1).mean()])])
 return np.concatenate(parts).astype(np.float32)

def predict(z,ts,q,dlinear,mode):
 ca=np.load(RUN/'cache/dense_chronos_train.npz');ta=np.load(RUN/'cache/dense_timesfm_train.npz');orig=ca['origins'];assert np.array_equal(orig,ta['origins']);tx,ty=windows(z[:8640],orig);tp=np.stack([ca['predictions'],ta['predictions']],1);tq=np.stack([ca['quantiles'],ta['quantiles']],1);err=ty[:,None]-tp;fx=np.stack([descriptor(a) for a in tx]);sx=np.maximum(fx.std(0),.1);fe=np.stack([enriched(a,mode) for a in tx]);se=np.maximum(fe.std(0),.1)
 def bf(x,p,q):
  recent=x[:,-96:].mean(1);parts=[]
  for m in range(2):parts.extend([p[:,m].mean(1)-recent,p[:,m].std(1),(q[:,m,:,:,-1]-q[:,m,:,:,0]).mean(1),(p[:,m]-q[:,m,:,:,4]).mean(1)])
  return np.concatenate(parts,1)
 fb=bf(tx,tp,tq);sb=np.maximum(fb.std(0),.1);patterns=np.empty((len(orig),2,24,7),np.float32)
 for i,t in enumerate(orig):
  h=(t+np.arange(64))%24
  for hour in range(24):patterns[i,:,hour]=err[i][:,h==hour].mean(1)
 x=np.stack([z[t-512:t] for t in ts]);p=q.mean(-1);xb=np.stack([descriptor(a) for a in x]);eb=np.stack([enriched(a,mode) for a in x]);bb=bf(x,p,q);pred=[];neighbors=[]
 for i,t in enumerate(ts):
  guard(600);valid=np.where(orig+64<=t-512)[0];dist=np.mean(((fx[valid]-xb[i])/sx)**2,1)+np.mean(((fb[valid]-bb[i])/sb)**2,1)+np.mean(((fe[valid]-eb[i])/se)**2,1);chosen=[]
  for j in valid[np.argsort(dist)]:
   if all(abs(int(orig[j])-int(orig[k]))>=64 for k in chosen):chosen.append(j)
   if len(chosen)==8:break
  neighbors.append(orig[chosen].tolist());cor=patterns[chosen].mean(0)[:,(t+np.arange(64))%24];bound=.5*np.maximum(x[i].std(0),.05);corrected=p[i]+.75*np.clip(cor,-bound,bound);pred.append((dlinear[i]+corrected.sum(0))/3)
 return np.stack(pred).astype(np.float32),neighbors
if __name__=='__main__':
 mode=sys.argv[1];name=f'20261105_enrichedmemory_{mode}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);z=np.load(RUN/'cache/trainval_z.npy');manifest=json.loads((RUN/'cache/window_manifest.json').read_text());results={};neighbors={};start=time.time()
 for split,ts in manifest.items():
  q=np.stack([np.load(ROOT/'runs'/n/f'{split}_quantiles.npy') for n in ('20261105_chronos2_median_mse','20261105_timesfm3_sym1_mse')],1);p,nn=predict(z,ts,q,np.load(RUN/'cache'/f'{split}_dlinear.npy'),mode);results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'));np.save(out/f'{split}_predictions.npy',p);neighbors[split]=nn
 save_json(out/'config.json',{'engine':'enriched_memory_v1','mode':mode,'strength':.75,'neighbor_count':8,'min_neighbor_spacing':64,'query_separation':'case end<=query support start','training_only_scales':True,'code_revision':git_head()});save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated','elapsed_seconds':time.time()-start});save_json(out/'neighbor_origins.json',neighbors);print(name,{s:m['mse'] for s,m in results.items()},flush=True)
