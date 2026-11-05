"""Calendar covariates derived from observed timestamps, never future measurements."""
from common import *
import torch,sys,pandas as pd
from chronos import Chronos2Pipeline
mode=sys.argv[1];name=f'20261105_calendar_{mode}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True)
subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True)
guard(900);torch.set_num_threads(4);z=np.load(RUN/'cache/trainval_z.npy');dates=pd.to_datetime(json.loads((RUN/'cache/dates.json').read_text()));manifest=json.loads((RUN/'cache/window_manifest.json').read_text());pipe=Chronos2Pipeline.from_pretrained(str(RUN/'models/chronos2_29ec3766'),device_map='cuda:0');start=time.time();results={}
def cov(d):
 h=np.asarray(d.hour,dtype=float);w=np.asarray(d.dayofweek,dtype=float);a={'hour_sin':np.sin(2*np.pi*h/24).astype(np.float32),'hour_cos':np.cos(2*np.pi*h/24).astype(np.float32)}
 if mode in ('hourweek','pastonly'):a.update(week_sin=np.sin(2*np.pi*(w+h/24)/7).astype(np.float32),week_cos=np.cos(2*np.pi*(w+h/24)/7).astype(np.float32))
 return a
for split,ts in manifest.items():
 ps=[];qs=[]
 for i in range(0,len(ts),8):
  guard(600);inputs=[]
  for t in ts[i:i+8]:
   # Future timestamps computed from last known time; no future CSV rows read here.
   fut=pd.date_range(dates[t-1]+pd.Timedelta(hours=1),periods=64,freq='h');item={'target':z[t-512:t].T.copy(),'past_covariates':cov(dates[t-512:t])}
   if mode!='pastonly':item['future_covariates']=cov(fut)
   inputs.append(item)
  q,p=pipe.predict_quantiles(inputs,prediction_length=64,batch_size=112,context_length=512,cross_learning=False);ps.extend([a.cpu().numpy().T for a in p]);qs.extend([a.cpu().numpy().transpose(1,0,2) for a in q])
  if i%640==0:print(mode,split,i,len(ts),flush=True)
 p=np.stack(ps).astype(np.float32);q=np.stack(qs).astype(np.float32);y=np.load(RUN/'cache'/f'{split}_targets.npy');results[split]=metric(p,y);results[split]['quantile_average_mse']=metric(q.mean(-1),y)['mse'];np.save(out/f'{split}_predictions.npy',p);np.save(out/f'{split}_quantiles.npy',q);save_json(out/'partial_metrics.json',results)
save_json(out/'config.json',{'mode':mode,'model':'frozen Chronos2','known_future_calendar':mode!='pastonly','no_future_observations':True,'timezone':'unknown; record-clock phase only','context':512,'horizon':64,'code_revision':git_head()});save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated','elapsed_seconds':time.time()-start});print(name,{s:r['mse'] for s,r in results.items()},flush=True)
