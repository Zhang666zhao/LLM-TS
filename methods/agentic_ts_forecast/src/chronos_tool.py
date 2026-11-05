"""Frozen Chronos-2 multivariate inference, exact 512 histories, no test access."""
from common import *
import torch,sys
from chronos import Chronos2Pipeline
name='20261105_chronos2_median_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True)
subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True)
guard(600);torch.set_num_threads(4)
model=RUN/'models/chronos2_29ec3766';start=time.time()
pipe=Chronos2Pipeline.from_pretrained(str(model),device_map='cuda:0')
z=np.load(RUN/'cache/trainval_z.npy');manifest=json.loads((RUN/'cache/window_manifest.json').read_text());results={}
for split in ('dev','confirm','val'):
 ts=manifest[split];pred=[];quant=[]
 for i in range(0,len(ts),16):
  guard(600);x,_=windows(z,ts[i:i+16]);q,p=pipe.predict_quantiles([torch.from_numpy(a.T.copy()) for a in x],prediction_length=64,batch_size=112,context_length=512,cross_learning=False)
  pred.extend([a.cpu().numpy().T for a in p]);quant.extend([a.cpu().numpy().transpose(1,0,2) for a in q])
  if i%320==0:print(split,i,len(ts),flush=True)
 pred=np.stack(pred).astype(np.float32);quant=np.stack(quant).astype(np.float32)
 y=np.load(RUN/'cache'/f'{split}_targets.npy');results[split]=metric(pred,y)
 np.save(RUN/'cache'/f'{split}_chronos2.npy',pred);np.save(out/f'{split}_quantiles.npy',quant)
 save_json(out/'partial_metrics.json',results)
config={'model':'amazon/chronos-2','revision':'29ec3766d36d6f73f0696f85560a422f50e8498c','weights_sha256':hashlib.sha256((model/'model.safetensors').read_bytes()).hexdigest(),'point':'median returned by predict_quantiles mean field','context':512,'horizon':64,'multivariate':True,'cross_learning':False,'code_revision':git_head(),'elapsed_seconds':time.time()-start}
save_json(out/'config.json',config);save_json(out/'metrics.json',results);print({s:r['mse'] for s,r in results.items()},flush=True)
