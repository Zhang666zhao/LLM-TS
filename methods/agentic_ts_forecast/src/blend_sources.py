"""Typed, reproducible convex blending and bounded forecast postprocessing."""
from common import *
import sys,re

def get_source(source,split):
 kind,name=source.split(':',1)
 if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*',name):raise ValueError('Invalid source identifier')
 if kind=='cache':p=RUN/'cache'/f'{split}_{name}.npy'
 elif kind=='run':p=ROOT/'runs'/name/f'{split}_predictions.npy'
 else:raise ValueError('Unknown source kind')
 a=np.load(p)
 if a.shape[1:]!=(64,7) or not np.isfinite(a).all():raise ValueError('Invalid source predictions')
 return a

def predict_cached(cfg,split,z,ts):
 sources=cfg['sources'];total=sum(sources.values())
 if total<=0 or any(not np.isfinite(v) or v<0 for v in sources.values()):raise ValueError('Nonnegative weights required')
 p=sum((v/total)*get_source(k,split).astype(np.float64) for k,v in sources.items())
 amplitude=float(cfg.get('amplitude',1));shrink=float(cfg.get('mean_shrink',0));smooth=int(cfg.get('smooth',1));bias=float(cfg.get('proxy_bias_strength',0))
 if not (.5<=amplitude<=1.5 and 0<=shrink<=.5 and smooth in (1,3,5,7) and 0<=bias<=.5):raise ValueError('Postprocessing outside declared bounds')
 center=p.mean(1,keepdims=True);p=center+amplitude*(p-center)
 if shrink:p+=shrink*(np.stack([z[t-96:t].mean(0) for t in ts])[:,None]-center)
 if smooth>1:
  pad=smooth//2;a=np.pad(p,((0,0),(pad,pad),(0,0)),mode='edge');p=sum(a[:,j:j+64] for j in range(smooth))/smooth
 if bias:
  b=np.load(RUN/'cache'/f'{split}_context_backtest.npz')['bias'].mean(1);p-=bias*b[:,None]*np.linspace(1,.5,64)[None,:,None]
 return p.astype(np.float32)

def run(cfg,name):
 out=ROOT/'runs'/name;out.mkdir(exist_ok=True)
 subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True)
 guard(900);rev=git_head();start=time.time();manifest=json.loads((RUN/'cache/window_manifest.json').read_text());z=np.load(RUN/'cache/trainval_z.npy');results={}
 for split,ts in manifest.items():
  p=predict_cached(cfg,split,z,ts);results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'));np.save(out/f'{split}_predictions.npy',p)
 save_json(out/'config.json',dict(cfg,code_revision=rev,engine='blend_sources_v1',runtime_llm=False));save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated','elapsed_seconds':time.time()-start})
 with (RUN/'experiment_index.jsonl').open('a') as f:f.write(json.dumps({'candidate':name,'val_mse':results['val']['mse'],'status':'evaluated','kind':'typed_numeric_pipeline'})+'\n')
 print(name,{k:v['mse'] for k,v in results.items()},flush=True)
if __name__=='__main__':run(json.loads(sys.argv[1]),sys.argv[2])
