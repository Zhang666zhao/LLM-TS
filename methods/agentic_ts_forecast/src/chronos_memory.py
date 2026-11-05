from common import *
import torch
from chronos import Chronos2Pipeline
torch.set_num_threads(4);guard(600)
bank=dict(np.load(RUN/'cache/train_error_bank.npz'));z=np.load(RUN/'cache/trainval_z.npy')[:8640];x,y=windows(z,bank['origins']);pipe=Chronos2Pipeline.from_pretrained(str(RUN/'models/chronos2_29ec3766'),device_map='cuda:0');pred=[]
for i in range(0,len(x),16):
 guard(600);_,p=pipe.predict_quantiles([torch.from_numpy(a.T.copy()) for a in x[i:i+16]],prediction_length=64,batch_size=112,context_length=512,cross_learning=False);pred.extend([a.cpu().numpy().T for a in p])
pred=np.stack(pred).astype(np.float32);err=pred-y
bank['predictions']=np.concatenate([bank['predictions'],pred[:,None]],1);bank['mse']=np.concatenate([bank['mse'],np.mean(err.astype(float)**2,axis=1)[:,None]],1);bank['bias']=np.concatenate([bank['bias'],err.mean(1)[:,None]],1)
np.savez(RUN/'cache/train_error_bank4.npz',**bank)
m=json.loads((RUN/'cache/memory_manifest.json').read_text());m['tools'].append('chronos2');m['pretrained_contamination']='Chronos pretraining overlap not audited; pretrained baseline only';save_json(RUN/'cache/memory_manifest4.json',m);print('four tool memory',len(pred),flush=True)
