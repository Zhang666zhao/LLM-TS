"""Choose invertible-view tools from errors already observed inside the512 context."""
from common import *
import sys
TOOLS=['20261105_transformedft_'+v+'_mse' for v in ('semantic','pca','random')]
def hindcasts(z,ts,out,tools=None):
 import torch,gc
 from chronos import Chronos2Pipeline
 tools=TOOLS if tools is None else tools;out=pathlib.Path(out);out.mkdir(exist_ok=True,parents=True);torch.set_num_threads(4);pred=[];manifest={'origins':list(map(int,ts)),'tools':tools,'context_slice':'t-512:t-64','target_observed_slice':'t-64:t'};mp=out/'manifest.json'
 if mp.exists():assert json.loads(mp.read_text())==manifest
 else:save_json(mp,manifest)
 for name in tools:
  path=out/(name+'.npy')
  if path.exists():p=np.load(path)
  else:
   cfg=json.loads((ROOT/'runs'/name/'config.json').read_text());w=np.asarray(cfg['matrix']);b=np.asarray(cfg['bias']);inv=np.asarray(cfg['inverse_matrix']);pipe=Chronos2Pipeline.from_pretrained(str(ROOT/'runs'/cfg['checkpoint_parent']/'trainer/finetuned-ckpt'),device_map='cuda:0');pipe.model.eval();ps=[]
   for i in range(0,len(ts),16):
    guard(600);x=np.stack([z[t-512:t-64] for t in ts[i:i+16]]);assert x.shape[1:]==(448,7);lat=(x.astype(float)@w+b).astype(np.float32);q,_=pipe.predict_quantiles([torch.from_numpy(v.T.copy()) for v in lat],prediction_length=64,batch_size=112,context_length=512,cross_learning=False);point=np.stack([a.cpu().numpy().transpose(1,0,2).mean(-1) for a in q]);ps.append(((point-b)@inv).astype(np.float32))
   p=np.concatenate(ps);np.save(path,p);del pipe;gc.collect();torch.cuda.empty_cache()
  assert p.shape==(len(ts),64,7) and np.isfinite(p).all();pred.append(p)
 return np.stack(pred,1)
def weights(x,past,strength):
 # Errors are retrospective; temperature is fixed, no future-label fitting.
 err=((past.astype(float)-x[-64:][None])**2).mean(1);relative=err/np.maximum(err.mean(0,keepdims=True),1e-6);logit=-relative;u=np.exp(logit-logit.max(0,keepdims=True));u/=u.sum(0,keepdims=True);return (1-strength)/3+strength*u

def cache():
 z=np.load(RUN/'cache/trainval_z.npy');mf=json.loads((RUN/'cache/window_manifest.json').read_text())
 for split,ts in mf.items():hindcasts(z,ts,RUN/'view_context_cache'/split);print('VIEW_HINDCAST_DONE',split,flush=True)
def run(strength,tag):
 name=f'20261105_viewroute_{tag}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);guard(900);z=np.load(RUN/'cache/trainval_z.npy');mf=json.loads((RUN/'cache/window_manifest.json').read_text());cfg={'engine':'view_routing_v1','strength':strength,'cadence':24,'source_ids':['run:'+n for n in TOOLS],'runtime_llm':False,'code_revision':git_head(),'rule':'softmax negative relative already-observed64-MSE, uniform shrinkage','input_scope':'only current512; first448 hindcast last64; weights24origins; forecasts hourly'};save_json(out/'config.json',cfg);results={}
 for split,ts in mf.items():
  step=24 if split=='val' else 1;past=np.stack([np.load(RUN/'view_context_cache'/split/(n+'.npy')) for n in TOOLS],1);pred=np.stack([np.load(ROOT/'runs'/n/f'{split}_predictions.npy') for n in TOOLS],1);ww=np.empty((len(ts),3,7))
  for i in range(0,len(ts),step):ww[i:i+step]=weights(z[ts[i]-512:ts[i]],past[i],strength)
  p=(pred*ww[:,:,None]).sum(1).astype(np.float32);results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'));np.save(out/f'{split}_predictions.npy',p);np.save(out/f'{split}_weights.npy',ww)
 save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated'});print(name,{s:m['mse'] for s,m in results.items()},flush=True)
if __name__=='__main__':
 if sys.argv[1]=='cache':cache()
 else:run(float(sys.argv[1]),sys.argv[2])
