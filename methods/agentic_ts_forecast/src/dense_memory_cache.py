"""Dense frozen-pretrained training errors for disjoint-development memory learners."""
from common import *
import torch,sys
which=sys.argv[1];torch.set_num_threads(4);guard(1200)
z=np.load(RUN/'cache/trainval_z.npy')[:8640];ts=np.arange(512,8577,8);pred=[];quant=[]
if which=='chronos':
 from chronos import Chronos2Pipeline
 model=Chronos2Pipeline.from_pretrained(str(RUN/'models/chronos2_29ec3766'),device_map='cuda:0')
else:
 from timesfm3 import TimesFM3Evaluator,ModelConfig
 model=TimesFM3Evaluator(ModelConfig(checkpoint_path=str(RUN/'models/timesfm3_43046b85'),per_core_batch_size=16,device='cuda:0',local_files_only=True))
for i in range(0,len(ts),16):
 guard(900);x,_=windows(z,ts[i:i+16])
 if which=='chronos':
  q,p=model.predict_quantiles([torch.from_numpy(a.T.copy()) for a in x],prediction_length=64,batch_size=112,context_length=512,cross_learning=False);qs=np.stack([a.cpu().numpy().transpose(1,0,2) for a in q]);ps=qs.mean(-1)
 else:
  ans=list(model.predict_batch([a.T.copy() for a in x],horizon=64,return_quantiles=True,use_symmetric_averaging=True,make_positive=False,sort_quantiles=True,use_znorm=False,padding_mode='none'));qs=np.stack([a.quantiles.transpose(1,0,2) for a in ans]);ps=qs.mean(-1)
 pred.append(ps.astype(np.float32));quant.append(qs.astype(np.float32))
 if i%160==0:print(which,i,len(ts),flush=True)
np.savez(RUN/'cache'/f'dense_{which}_train.npz',origins=ts,predictions=np.concatenate(pred),quantiles=np.concatenate(quant))
save_json(RUN/'cache'/f'dense_{which}_manifest.json',{'training_only':True,'origins_count':len(ts),'stride':8,'context':512,'horizon':64,'model':'frozen pretrained','point':'decile-average','pretraining_overlap':'not audited','code_revision':git_head()});print('done',which,flush=True)
