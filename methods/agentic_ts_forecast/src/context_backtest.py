"""Causal within-512 hindcast evidence; target is already observed history."""
from common import *
import torch
from chronos import Chronos2Pipeline
from prepare import load_baseline
torch.set_num_threads(4);guard(600)
pipe=Chronos2Pipeline.from_pretrained(str(RUN/'models/chronos2_29ec3766'),device_map='cuda:0');dlinear,_=load_baseline()
z=np.load(RUN/'cache/trainval_z.npy');manifest=json.loads((RUN/'cache/window_manifest.json').read_text());manifest['memory']=np.load(RUN/'cache/train_error_bank4.npz')['origins'].tolist()
for split,ts in manifest.items():
 out=RUN/'cache'/f'{split}_context_backtest.npz'
 if out.exists():continue
 errors=[];biases=[];preds=[]
 for i in range(0,len(ts),16):
  guard(600);x=np.stack([z[t-512:t] for t in ts[i:i+16]]);obs=x[:,:448];target=x[:,448:];padded=np.concatenate([np.repeat(obs[:,:1],64,axis=1),obs],axis=1)
  with torch.no_grad():pd=dlinear(torch.from_numpy(padded).cuda(0)).cpu().numpy()
  _,pc=pipe.predict_quantiles([torch.from_numpy(a.T.copy()) for a in obs],prediction_length=64,batch_size=112,context_length=512,cross_learning=False);pc=np.stack([p.cpu().numpy().T for p in pc]);p=np.stack([pd,pc],axis=1);e=p-target[:,None]
  errors.append(np.mean(e.astype(float)**2,axis=2));biases.append(e.mean(2));preds.append(p)
  if i%640==0:print(split,i,len(ts),flush=True)
 np.savez(out,mse=np.concatenate(errors),bias=np.concatenate(biases),predictions=np.concatenate(preds),origins=np.array(ts))
save_json(RUN/'cache/context_backtest_manifest.json',{'tools':['dlinear','chronos2'],'observed_context':512,'hindcast_prefix_real':448,'hindcast_target':'last64 already observed','dlinear_padding':'repeat first observation64 times; no earlier observations','chronos_context':448,'no_query_future':True,'code_revision':git_head()});print('done context backtests',flush=True)
