"""Current-window hindcast routing with full-CDF adapted and calendar view tools."""
from common import *
from transformed_tool import predict as tool_predict
from blend_sources import get_source
import copy,sys
TOOLS=['20261105_transformedcalft_random_mse','20261105_transformedcdfmse_pca_mse','20261105_rotationseed_2024_mse']
PARENT='20261105_p11constant_ranked_mse'
def hindcasts(z,dates,ts,out):
 out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=True);ts=np.asarray(ts);manifest=dict(origins=ts.tolist(),tools=TOOLS,input='first448within512',point='native21quantilefullCDF');mp=out/'manifest.json'
 if mp.exists():assert json.loads(mp.read_text())==manifest
 else:save_json(mp,manifest)
 ps=[]
 for name in TOOLS:
  path=out/(name+'.npy')
  if path.exists():p=np.load(path)
  else:
   cfg=json.loads((ROOT/'runs'/name/'config.json').read_text());cfg['inference_context']=448;p=tool_predict(z,ts-64,cfg,dates=dates);np.save(path,p)
  assert p.shape==(len(ts),64,7) and np.isfinite(p).all();ps.append(p)
 return np.stack(ps,1)
def weights(z,ts,past,mode):
 if mode=='uniform':return np.full((len(ts),3,7),1/3,dtype=float)
 observed=np.stack([z[t-64:t] for t in ts]);err=((past.astype(float)-observed[:,None])**2).mean(2)
 if mode=='uniform':w=np.ones_like(err)/3
 elif mode=='soft':
  v=-err/np.maximum(err.mean(1,keepdims=True),1e-6);w=np.exp(v-v.max(1,keepdims=True));w/=w.sum(1,keepdims=True)
 elif mode=='hard':w=np.eye(3)[err.argmin(1)].transpose(0,2,1)
 else:raise ValueError(mode)
 assert np.isfinite(w).all() and np.allclose(w.sum(1),1);return w
def apply(z,ts,past,pred,parent,cfg):
 w=weights(z,ts,past,cfg['mode']);p=(pred.astype(float)*w[:,:,None]).sum(1);a=cfg['strength'];return ((1-a)*parent.astype(float)+a*p).astype(np.float32)
def cache():
 z=np.load(RUN/'cache/trainval_z.npy');dates=json.loads((RUN/'cache/dates.json').read_text())
 for split,ts in json.loads((RUN/'cache/window_manifest.json').read_text()).items():hindcasts(z,dates,ts,RUN/'strong_view_hindcasts'/split);print('STRONG_VIEW_HINDCAST_DONE',split,flush=True)
def run(mode):
 name=f'20261105_strongviewfeedback_{mode}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=False);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);z=np.load(RUN/'cache/trainval_z.npy');cfg=dict(engine='strong_view_feedback_v1',mode=mode,strength=.2,source_ids=['run:'+PARENT]+['run:'+k for k in TOOLS],tools=TOOLS,runtime_llm=False,code_revision=git_head(),input_scope='512only:448hindcast64observed; future forecast512; hourly routing');results={}
 for split,ts in json.loads((RUN/'cache/window_manifest.json').read_text()).items():
  guard(900);past=np.stack([np.load(RUN/'strong_view_hindcasts'/split/(k+'.npy')) for k in TOOLS],1);pred=np.stack([get_source('run:'+k,split) for k in TOOLS],1);p=apply(z,ts,past,pred,get_source('run:'+PARENT,split),cfg);np.save(out/f'{split}_predictions.npy',p);results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'))
 save_json(out/'config.json',cfg);save_json(out/'metrics.json',dict(candidate=name,metrics=results,status='evaluated'));print(name,{k:v['mse'] for k,v in results.items()},flush=True)
if __name__=='__main__':
 if sys.argv[1]=='cache':cache()
 else:run(sys.argv[1])
