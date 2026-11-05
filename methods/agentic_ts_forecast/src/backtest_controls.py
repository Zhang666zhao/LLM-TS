from common import *
import sys
cfg=json.loads(sys.argv[1]);name=sys.argv[2];out=ROOT/'runs'/name;out.mkdir(exist_ok=True)
subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True)
results={}
for split in ('dev','confirm','val'):
 guard(600);a=np.load(RUN/'cache'/f'{split}_context_backtest.npz');e=a['mse'];bias=a['bias'];d=np.load(RUN/'cache'/f'{split}_dlinear.npy');c=np.load(RUN/'cache'/f'{split}_chronos2.npy')
 w=e[:,1]/np.maximum(e.sum(1),1e-8);w=.5+cfg.get('routing_strength',0)*(w-.5);w=np.clip(w,.1,.9)
 p=w[:,None]*d+(1-w[:,None])*c
 if cfg.get('bias_strength',0):
  b=w*bias[:,0]+(1-w)*bias[:,1];p=p-cfg['bias_strength']*b[:,None]*np.linspace(1,.5,64)[None,:,None]
 p=p.astype(np.float32);results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'));np.save(out/f'{split}_predictions.npy',p);np.save(out/f'{split}_weights.npy',w)
save_json(out/'config.json',dict(cfg,code_revision=git_head(),forecast_tools=['dlinear','chronos2'],signals='last64 observed hindcast errors only'));save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated'})
with (RUN/'experiment_index.jsonl').open('a') as f:f.write(json.dumps({'candidate':name,'val_mse':results['val']['mse'],'status':'evaluated','kind':'causal_backtest_control'})+'\n')
print(name,{s:r['mse'] for s,r in results.items()},flush=True)
