"""Prepare train/validation only. No test rows, targets or predictions are loaded."""
import os,csv,json,hashlib,importlib.util
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch,numpy as np
from types import SimpleNamespace
from common import *
torch.set_num_threads(4)
def load_baseline():
 project=pathlib.Path(CONTRACT['source']).parents[2]
 source=project/'third_party/LTSF-Linear/models/DLinear.py'
 spec=importlib.util.spec_from_file_location('frozen_dlinear',source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 net=mod.Model(SimpleNamespace(seq_len=512,pred_len=64,individual=False,enc_in=7)).cuda().eval()
 state=torch.load(pathlib.Path(CONTRACT['baseline_directory'])/'best.pth',map_location='cpu',weights_only=False)
 net.load_state_dict({k.removeprefix('backbone.'):v for k,v in state.items()});return net,mod
if __name__=='__main__':
 guard();out=RUN/'cache';out.mkdir(exist_ok=True)
 with open(CONTRACT['source']) as f:
  reader=csv.DictReader(f);rows=[next(reader) for _ in range(11520)]
 raw=np.array([[float(r[c]) for c in COLS] for r in rows],dtype=np.float64)
 assert np.isfinite(raw).all()
 assert np.allclose(raw[:8640].mean(0),CONTRACT['scaler_mean'],rtol=0,atol=1e-12)
 # Existing benchmark computes scaler on float64 data then casts to float32.
 z=((raw-np.array(CONTRACT['scaler_mean']))/np.array(CONTRACT['scaler_scale'])).astype(np.float32)
 np.save(out/'trainval_z.npy',z);save_json(out/'dates.json',[r['date'] for r in rows])
 # Disjoint development/confirmation raw support; fixed before any candidate score.
 origins={'dev':np.arange(4608,6145,64),'confirm':np.arange(6784,8577,64),'val':np.arange(8640,11520-64+1)}
 save_json(out/'window_manifest.json',{k:v.tolist() for k,v in origins.items()})
 net,mod=load_baseline()
 with torch.no_grad():
  for split,ts in origins.items():
   x,y=windows(z,ts);ps=[]
   for a in range(0,len(x),128):guard();ps.append(net(torch.from_numpy(x[a:a+128]).cuda()).cpu().numpy())
   pred=np.concatenate(ps);np.save(out/f'{split}_dlinear.npy',pred);np.save(out/f'{split}_targets.npy',y)
   save_json(out/f'{split}_dlinear_metrics.json',metric(pred,y))
   print(split,len(ts),metric(pred,y)['mse'],flush=True)
 np.save(out/'raw_trainval.npy',raw)
 save_json(RUN/'status.json',{'phase':'prepared_train_validation','updated_epoch':time.time(),'test_access':False})
