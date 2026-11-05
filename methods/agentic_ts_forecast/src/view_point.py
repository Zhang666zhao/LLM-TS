"""Three declared point-extraction rules for the same transformed probabilistic tool."""
from common import *
from transformed_tool import extract_point
import sys
PARENT='20261105_transformedft_random_mse'

def cache():
 import torch,gc
 from chronos import Chronos2Pipeline
 torch.set_num_threads(4);z=np.load(RUN/'cache/trainval_z.npy');mf=json.loads((RUN/'cache/window_manifest.json').read_text());cfg=json.loads((ROOT/'runs'/PARENT/'config.json').read_text());loc=ROOT/'runs'/cfg['checkpoint_parent']/'trainer/finetuned-ckpt';pipe=Chronos2Pipeline.from_pretrained(str(loc),device_map='cuda:0');pipe.model.eval();levels=pipe.model.chronos_config.quantiles;out=RUN/'view_point_cache';out.mkdir(exist_ok=True);save_json(out/'manifest.json',{'parent':PARENT,'levels':levels,'code_revision':git_head()});w=np.asarray(cfg['matrix']);b=np.asarray(cfg['bias'])
 for split,ts in mf.items():
  ps=[]
  for i in range(0,len(ts),16):
   guard(900);x=np.stack([z[t-512:t] for t in ts[i:i+16]]);lat=(x.astype(float)@w+b).astype(np.float32);q,_=pipe.predict_quantiles([torch.from_numpy(v.T.copy()) for v in lat],prediction_length=64,quantile_levels=levels,batch_size=112,context_length=512,cross_learning=False);ps.extend([a.cpu().numpy().transpose(1,0,2) for a in q])
  q=np.stack(ps);assert q.shape==(len(ts),64,7,len(levels)) and np.isfinite(q).all();np.save(out/(split+'_quantiles.npy'),q);print('VIEW_POINT_CACHE',split,flush=True)
 del pipe;gc.collect();torch.cuda.empty_cache()
def run(rule):
 assert rule in ('median','central_integral','full_integral');name=f'20261105_viewpoint_{rule}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);guard(900);meta=json.loads((RUN/'view_point_cache/manifest.json').read_text());cfg=json.loads((ROOT/'runs'/PARENT/'config.json').read_text());cfg.update(point_rule=rule,point_quantile_levels=meta['levels'],source_ids=['run:'+PARENT],point_caveat='Latent quantile quadrature then inverse POINT transformation; no calibrated original-variable quantile claim. Full integral uses flat extrapolation over0-.01 and.99-1.',code_revision=git_head());save_json(out/'config.json',cfg);b=np.asarray(cfg['bias']);inv=np.asarray(cfg['inverse_matrix']);results={}
 for split in ('dev','confirm','val'):
  q=np.load(RUN/'view_point_cache'/(split+'_quantiles.npy'));point=extract_point(q,meta['levels'],rule);p=((point-b)@inv).astype(np.float32);results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'));np.save(out/f'{split}_predictions.npy',p)
 save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated'});print(name,{s:m['mse'] for s,m in results.items()},flush=True)
if __name__=='__main__':
 if sys.argv[1]=='cache':cache()
 else:run(sys.argv[1])
