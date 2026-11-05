"""Train-only cross-variable summaries and load-magnitude proxy ridge tools."""
from common import *
import torch,sys

def features(x,mode):
 from ridge_tool import features as own_features
 own,center=own_features(x,dict(center=96,features='full'));x=x.astype(float)
 parts=[]
 for k in (24,96,512):parts.extend([x[:,-k:].mean(1),x[:,-k:].std(1)])
 parts.append(x[:,-24:].mean(1)-x[:,-48:-24].mean(1));extra=np.concatenate(parts,1)
 if mode in ('magnitude','interaction'):
  raw=x*np.asarray(CONTRACT['scaler_scale'])[None,None]+np.asarray(CONTRACT['scaler_mean'])[None,None]
  # Unknown load units/topology: these are empirical proxies, not calibrated power/heat.
  energy=np.concatenate([(raw[:,-k:,:6]**2).mean(1) for k in (6,24,96)],1);extra=np.concatenate([extra,energy],1)
 if mode=='interaction':extra=np.concatenate([extra,(x[:,-96:,:6].mean(1)*x[:,-96:,6].mean(1)[:,None])],1)
 return own,extra,center

def predict(x,cfg,state):
 f,e,c=features(x,cfg['feature_mode']);e=(e-state['extra_mean'])/state['extra_scale'];f=np.concatenate([f,np.broadcast_to(e[:,None],(len(x),7,e.shape[1])),np.ones((len(x),7,1))],2);return (np.einsum('ncf,cfh->nch',f,state['weights'])+c).transpose(0,2,1).astype(np.float32)

def run(mode):
 assert mode in ('summary','magnitude','interaction');name=f'20261105_semanticridge_{mode}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=False);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);guard(2400);torch.set_num_threads(4);z=np.load(RUN/'cache/trainval_z.npy');orig=np.arange(512,8577,4);x,y=windows(z[:8640],orig);f,e,c=features(x,mode);mu=e.mean(0);sd=np.maximum(e.std(0),1e-6);e=(e-mu)/sd;weights=[]
 for j in range(7):
  guard(1800);a=np.concatenate([f[:,j],e,np.ones((len(x),1))],1);b=y[:,:,j].astype(float)-c[:,j];a=torch.from_numpy(a).cuda(0);b=torch.from_numpy(b).cuda(0);reg=torch.eye(a.shape[1],device='cuda:0',dtype=torch.float64)*.1*len(a);reg[-1,-1]=1e-8;w=torch.linalg.solve(a.T@a+reg,a.T@b);assert torch.isfinite(w).all();weights.append(w.cpu().numpy())
 state=dict(weights=np.stack(weights),extra_mean=mu,extra_scale=sd);torch.save(state,out/'best.pth');np.save(out/'training_sample_origins.npy',orig);cfg=dict(engine='semantic_ridge_v1',feature_mode=mode,alpha=.1,center=96,individual=True,train_stride=4,training_rows=[0,8640],training_windows=len(orig),extra_features=e.shape[1],features='512 own lags centered96 plus train-standardized cross-variable past summaries',semantics='First6 columns load signals, OT oil temperature; squared raw load is only empirical magnitude proxy, unknown units/topology; no physical equality imposed',dev_confirm_in_sample=True,code_revision=git_head());save_json(out/'config.json',cfg)
 probe=x[:2].copy();before=predict(probe,cfg,state);probe[1]+=10;assert np.array_equal(before[0],predict(probe,cfg,state)[0]);state=torch.load(out/'best.pth',map_location='cpu',weights_only=False);results={}
 for split,ts in json.loads((RUN/'cache/window_manifest.json').read_text()).items():
  pp=[]
  for i in range(0,len(ts),128):
   guard(1800);bx,_=windows(z,ts[i:i+128]);pp.append(predict(bx,cfg,state))
  pp=np.concatenate(pp);results[split]=metric(pp,np.load(RUN/'cache'/f'{split}_targets.npy'));np.save(out/f'{split}_predictions.npy',pp)
 save_json(out/'contracts.json',dict(query_isolation_exact=True,feature_scaler_training_only=True,targets_end_max=int(orig.max()+64),future_input=False));save_json(out/'metrics.json',dict(candidate=name,metrics=results,status='evaluated'));print(name,results['val']['mse'],results['val']['per_variable_mse'],flush=True)
if __name__=='__main__':run(sys.argv[1])
