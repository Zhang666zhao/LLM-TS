"""Build a frozen TRAIN-only observed-error bank; disclose in-sample learned tools."""
from common import *
import torch,sys
from local_tools import predict as local_predict
from prepare import load_baseline
TOOLS=['dlinear','patchtst','season7']
def descriptor(x):
 feats=[]
 for k in (24,96,512):feats.extend([x[-k:].mean(0),x[-k:].std(0)])
 feats.append(x[-24:].mean(0)-x[-48:-24].mean(0))
 return np.concatenate(feats).astype(np.float32)
if __name__=='__main__':
 guard(600);torch.set_num_threads(4);z=np.load(RUN/'cache/trainval_z.npy')[:8640];origins=np.arange(1024,8577,128);x,y=windows(z,origins);pred=[]
 net,_=load_baseline()
 with torch.no_grad():p=net(torch.tensor(x,device='cuda:0')).cpu().numpy()
 pred.append(p);del net;torch.cuda.empty_cache()
 project=ROOT.parent/'TSRAG';sys.path.insert(0,str(project/'research/multivariate_oracle_six'));from conditioning import backbone
 net,cfg,audit=backbone(project,'patchtst','ETTh1',7);state=torch.load(project/'runs/20261014_six_multivariate_oracle_v1/ETTh1/jobs/patchtst_base_zero/best.pth',map_location='cpu');net.load_state_dict({k.removeprefix('backbone.'):v for k,v in state.items()});net.cuda(0).eval();p=[]
 with torch.no_grad():
  for i in range(0,len(x),16):guard(600);p.append(net(torch.tensor(x[i:i+16],device='cuda:0')).cpu().numpy())
 pred.append(np.concatenate(p));pred.append(local_predict(x,{'kind':'seasonal','period':24,'cycles':7}))
 preds=np.stack(pred,1);d=np.stack([descriptor(a) for a in x]);err=preds-y[:,None]
 np.savez(RUN/'cache/train_error_bank.npz',origins=origins,features=d,feature_scale=np.maximum(d.std(0),.1),mse=np.mean(err.astype(float)**2,axis=2),bias=np.mean(err,axis=2),predictions=preds,targets=y)
 save_json(RUN/'cache/memory_manifest.json',{'tools':TOOLS,'origins':origins.tolist(),'labels':'training only','learned_tools_in_sample':True,'retrieval_rule':'case end <= query origin-512','purpose':'training error feedback, not independent learned-tool performance','code_revision':git_head()})
 print('memory cases',len(origins),flush=True)
