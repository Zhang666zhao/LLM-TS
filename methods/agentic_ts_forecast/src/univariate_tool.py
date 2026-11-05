"""Cross-channel attention ablation, all seven forecasts independently from own history."""
from common import *
import torch,sys
which=sys.argv[1];name=f'20261105_univariate_{which}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True)
subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True)
torch.set_num_threads(4);guard(900);start=time.time();z=np.load(RUN/'cache/trainval_z.npy');manifest=json.loads((RUN/'cache/window_manifest.json').read_text());results={}
if which=='chronos':
 from chronos import Chronos2Pipeline
 pipe=Chronos2Pipeline.from_pretrained(str(RUN/'models/chronos2_29ec3766'),device_map='cuda:0')
else:
 from timesfm3 import TimesFM3Evaluator,ModelConfig
 pipe=TimesFM3Evaluator(ModelConfig(checkpoint_path=str(RUN/'models/timesfm3_43046b85'),per_core_batch_size=112,device='cuda:0',local_files_only=True))
for split,ts in manifest.items():
 ps=[];qs=[]
 for i in range(0,len(ts),16):
  guard(600);x,_=windows(z,ts[i:i+16]);flat=[a[:,c].copy() for a in x for c in range(7)]
  if which=='chronos':
   q,_=pipe.predict_quantiles([torch.from_numpy(a) for a in flat],prediction_length=64,batch_size=112,context_length=512,cross_learning=False);q=np.stack([a.cpu().numpy().reshape(64,9) for a in q]).reshape(len(x),7,64,9).transpose(0,2,1,3)
  else:
   ans=list(pipe.predict_batch(flat,horizon=64,return_quantiles=True,use_symmetric_averaging=True,make_positive=False,sort_quantiles=True,use_znorm=False,padding_mode='none'));q=np.stack([a.quantiles for a in ans]).reshape(len(x),7,64,9).transpose(0,2,1,3)
  ps.append(q.mean(-1).astype(np.float32));qs.append(q.astype(np.float32))
  if i%640==0:print(which,split,i,len(ts),flush=True)
 p=np.concatenate(ps);q=np.concatenate(qs);y=np.load(RUN/'cache'/f'{split}_targets.npy');results[split]=metric(p,y);results[split]['median_mse']=metric(q[:,:,:,4],y)['mse'];np.save(RUN/'cache'/f'{split}_{which}uavg.npy',p);np.save(out/f'{split}_quantiles.npy',q);save_json(out/'partial_metrics.json',results)
save_json(out/'config.json',{'model':which,'point':'decile-average','cross_variable_attention':False,'each_output_uses':'own512 history only','context':512,'horizon':64,'code_revision':git_head()});save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated','elapsed_seconds':time.time()-start});print(name,{s:r['mse'] for s,r in results.items()},flush=True)
