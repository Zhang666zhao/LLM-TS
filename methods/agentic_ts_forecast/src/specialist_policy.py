"""Small typed history-dependent policies compiled from LLM proposals; no arbitrary code."""
from common import *
from blend_sources import get_source
import sys
FEATURE_BOUNDS={'acf24':[-1,1],'drift24':[-2,2],'level_shift':[-2,2],'volatility_ratio':[.25,3],'skew168':[-3,3],'kurtosis168':[-2,15],'expert_disagreement':[0,2],}
def features(x,p):
 s=np.maximum(x.std(1),.05);a=x[:,:-24]-x[:,:-24].mean(1,keepdims=True);b=x[:,24:]-x[:,24:].mean(1,keepdims=True);c=(a*b).sum(1)/np.maximum(np.sqrt((a*a).sum(1)*(b*b).sum(1)),1e-6);recent=x[:,-168:];v=np.clip((recent-recent.mean(1,keepdims=True))/np.maximum(recent.std(1,keepdims=True),.05),-10,10)
 return {'acf24':c,'drift24':(x[:,-24:].mean(1)-x[:,-48:-24].mean(1))/s,'level_shift':(x[:,-96:].mean(1)-x.mean(1))/s,'volatility_ratio':x[:,-96:].std(1)/s,'skew168':(v**3).mean(1),'kurtosis168':(v**4).mean(1)-3,'expert_disagreement':p.mean(2).std(1)/s}
def canonical(c,registry):
 assert isinstance(c,dict) and set(c)<=set(['source_ids','base_weights','variable_weights','rules']);assert c['source_ids']==registry and len(registry)==4
 def weights(v):
  a=np.asarray(v,float);assert a.shape==(len(registry),) and np.isfinite(a).all() and (a>=0).all() and a.sum()>0;return (a/a.sum()).round(8).tolist()
 assert isinstance(c.get('variable_weights',{}),dict) and set(c.get('variable_weights',{}))<=set(COLS);out={'source_ids':registry,'base_weights':weights(c['base_weights']),'variable_weights':{k:weights(v) for k,v in c.get('variable_weights',{}).items()},'rules':[]};assert isinstance(c['rules'],list) and len(c['rules'])<=3
 for r in c['rules']:
  assert set(r)==set(['feature','op','threshold','columns','weights']);feature=r['feature'];assert feature in FEATURE_BOUNDS and feature!='ft_mean_gap' and r['op'] in ('<','>');lo,hi=FEATURE_BOUNDS[feature];assert isinstance(r['threshold'],(int,float)) and np.isfinite(r['threshold']) and lo<=r['threshold']<=hi;assert isinstance(r['columns'],list) and r['columns'] and set(r['columns'])<=set(COLS);out['rules'].append(dict(r,weights=weights(r['weights']),columns=sorted(set(r['columns']),key=COLS.index)))
 return out

def predict(x,p,cfg):
 f=features(x,p);base=np.asarray(cfg['base_weights']);base=base/base.sum();w=np.broadcast_to(base,(len(x),7,len(base))).copy();assigned=np.zeros((len(x),7),bool)
 for col,rw in cfg.get('variable_weights',{}).items():
  rw=np.asarray(rw);w[:,COLS.index(col),:]=rw/rw.sum()
 for r in cfg['rules']:
  mask=f[r['feature']]<r['threshold'] if r['op']=='<' else f[r['feature']]>r['threshold'];cols=np.array([v in r['columns'] for v in COLS]);mask&=cols[None]&~assigned;rw=np.asarray(r['weights']);w[mask]=rw/rw.sum();assigned|=mask
 out=np.einsum('nmhc,ncm->nhc',p.astype(np.float64),w).astype(np.float32);return out,w

def run(cfg,name):
 registry=cfg['source_ids'];cfg=canonical(cfg,registry);invalid=set(json.loads((RUN/'invalid_candidates.json').read_text()).get('candidates',[]));assert all(k.startswith('run:') and k.split(':',1)[1] not in invalid for k in registry)
 out=ROOT/'runs'/name;out.mkdir(exist_ok=True);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);z=np.load(RUN/'cache/trainval_z.npy');manifest=json.loads((RUN/'cache/window_manifest.json').read_text());results={};start=time.time()
 for split,ts in manifest.items():
  guard(900);x,_=windows(z,ts);p=np.stack([get_source(k,split) for k in registry],1);pr,w=predict(x,p,cfg);results[split]=metric(pr,np.load(RUN/'cache'/f'{split}_targets.npy'));np.save(out/f'{split}_predictions.npy',pr);np.save(out/f'{split}_weights.npy',w)
 save_json(out/'config.json',dict(cfg,engine='specialist_policy_v1',runtime_llm=False,first_matching_rule_wins=True,feature_bounds=FEATURE_BOUNDS,code_revision=git_head()));save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated','elapsed_seconds':time.time()-start});print(name,{s:m['mse'] for s,m in results.items()},flush=True)
if __name__=='__main__':run(json.loads(sys.argv[1]),sys.argv[2])
