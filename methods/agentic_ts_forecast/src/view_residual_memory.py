"""GPU ridge on chronological out-of-prefix tool errors; no official validation fit."""
from common import *
from view_memory_gate import features
from view_routing import TOOLS
from blend_sources import get_source
import sys
PRIOR=np.array([0.,.4,.6]);STRENGTH=.25

def expand(a):
 centers=np.arange(8)*8+3.5
 return np.stack([np.stack([np.interp(np.arange(64),centers,v[:,j]) for j in range(7)],1) for v in a]).astype(np.float32)

def fit(f,res,regularization):
 import torch
 mean=f.mean(0);scale=np.maximum(f.std(0),.1);a=np.clip((f-mean)/scale,-5,5).astype(np.float64);a=np.c_[np.ones(len(a)),a];y=res.reshape(len(res),8,8,7).mean(2).reshape(len(res),56).astype(np.float64)
 x=torch.from_numpy(a).cuda();target=torch.from_numpy(y).cuda();pen=torch.eye(x.shape[1],device='cuda',dtype=torch.float64)*regularization*len(x);pen[0,0]=1e-6*len(x);coef=torch.linalg.solve(x.T@x+pen,x.T@target).cpu().numpy();assert np.isfinite(coef).all()
 return {'mean':mean,'scale':scale,'coef':coef}

def correction(f,state):
 a=np.clip((f-state['mean'])/state['scale'],-5,5);a=np.c_[np.ones(len(a)),a];return expand((a@state['coef']).reshape(len(a),8,7))

def predict(x,p,state):
 return ((p*PRIOR[None,:,None,None]).sum(1)+STRENGTH*correction(features(x,p),state)).astype(np.float32)

def run(reg,integral=False):
 import torch
 assert reg in (1,10,100);name=f'20261105_viewresidual{"cdf" if integral else ""}_ridge{reg}_mse';out=ROOT/'runs'/name;assert not (out/'config.json').exists();out.mkdir(exist_ok=True);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);guard(2400);torch.set_num_threads(4)
 tools=['20261105_fullpoint_semantic_mse','20261105_fullpoint_pca_mse','20261105_viewpoint_full_integral_mse'] if integral else TOOLS
 path=RUN/'cache'/('view_prefix_integral_memory.npz' if integral else 'view_prefix_memory.npz');d=np.load(path);f=d['features'];p=d['predictions'];y=d['targets'];train=d['fold']<6272;hold=~train;assert d['origins'][train].max()+64<=d['origins'][hold].min()-512;base=(p*PRIOR[None,:,None,None]).sum(1);err=y-base
 state=fit(f[train],err[train],reg);holdpred=(base[hold]+STRENGTH*correction(f[hold],state)).astype(np.float32);diagnostic={'prefix_hold_before':metric(base[hold],y[hold]),'prefix_hold_after':metric(holdpred,y[hold]),'training_cases':int(train.sum()),'hold_cases':int(hold.sum()),'holdout_used_to_fit_coefficients':False}
 state=fit(f,err,reg);torch.save(state,out/'best.pth');cfg={'engine':'view_residual_memory_v1','source_ids':['run:'+n for n in tools],'memory_point':'full_CDF' if integral else 'mean9','regularization':reg,'strength':STRENGTH,'prior':PRIOR.tolist(),'features':203,'residual_target':'eight horizon blocks of8, linearly interpolated to64; flat endpoint extrapolation','training_memory':str(path),'training_memory_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'training_cases':len(f),'fit_device':'cuda:0','scope':'prefix-only model errors; official training only; fulltrain scaler shared; overlapping cases not independent; proxy tools differ from deployed tools','code_revision':git_head()};save_json(out/'config.json',cfg);save_json(out/'prefix_hold_diagnostic.json',diagnostic);z=np.load(RUN/'cache/trainval_z.npy');results={}
 for split,ts in json.loads((RUN/'cache/window_manifest.json').read_text()).items():
  x,_=windows(z,ts);p=np.stack([get_source(k,split) for k in cfg['source_ids']],1);pred=predict(x,p,state);assert np.isfinite(pred).all();np.save(out/f'{split}_predictions.npy',pred);results[split]=metric(pred,np.load(RUN/'cache'/f'{split}_targets.npy'))
 save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated'});print(name,{s:m['mse'] for s,m in results.items()},flush=True)
if __name__=='__main__':run(int(sys.argv[1]),bool(int(sys.argv[2])) if len(sys.argv)>2 else False)
