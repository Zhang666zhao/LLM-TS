"""Train-only covariance-calibrated empirical relation forecasts, with random control."""
from common import *
import torch,sys

def relation_matrix(mode):
 s=np.asarray(CONTRACT['scaler_scale']);a=np.zeros((2,7));a[0,[0,2,4]]=s[[0,2,4]]*np.array([1,-1,-1]);a[1,[1,3,5]]=s[[1,3,5]]*np.array([1,-1,-1])
 if mode=='random_phase':
  r=np.random.default_rng(2021).normal(size=(2,7));a=r/np.linalg.norm(r,axis=1)[:,None]*np.linalg.norm(a,axis=1)[:,None]
 return a

def discrepancy(x,p,a,mode):
 if mode=='semantic_mean':b=np.repeat((x[:,-168:].mean(1)@a.T)[:,None],64,axis=1)
 else:b=(x[:,-168:].reshape(len(x),7,24,7).mean(1)@a.T)[:,np.arange(64)%24]
 return b-p@a.T

def apply(x,p,dl,cfg):
 a=np.asarray(cfg['relation_matrix']);d=discrepancy(x,p,a,cfg['mode']);cor=(d/np.asarray(cfg['feature_scale']))@np.asarray(cfg['weights']);bound=.5*np.maximum(x.std(1),.05)[:,None];return ((dl+2*(p+np.clip(cor,-bound,bound)))/3).astype(np.float32)

def run(mode):
 name=f'20261105_learnedrel_{mode}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);guard(900);torch.set_num_threads(4);z=np.load(RUN/'cache/trainval_z.npy');ca=np.load(RUN/'cache/dense_chronos_train.npz');ta=np.load(RUN/'cache/dense_timesfm_train.npz');assert np.array_equal(ca['origins'],ta['origins']);x,y=windows(z[:8640],ca['origins']);p=(ca['predictions']+ta['predictions'])/2;a=relation_matrix(mode);d=discrepancy(x,p,a,mode);scale=np.maximum(d.reshape(-1,2).std(0),1e-3);xx=torch.tensor((d/scale).reshape(-1,2),dtype=torch.float64,device='cuda:0');ee=torch.tensor((y-p).reshape(-1,7),dtype=torch.float64,device='cuda:0');gram=xx.T@xx/len(xx);cross=xx.T@ee/len(xx);w=torch.linalg.solve(gram+.1*torch.eye(2,dtype=torch.float64,device='cuda:0'),cross).cpu().numpy();assert np.isfinite(w).all()
 cfg={'engine':'learned_relations_v1','mode':mode,'relation_matrix':a.tolist(),'feature_scale':scale.tolist(),'weights':w.tolist(),'ridge':.1,'fit_device':'cuda:0','training_cases':len(x),'supervision':'frozen pretrained tool residuals on official training only','parent':'equal DLinear/Chronosavg/TimesFMavg','correction_clip':'.5 historical512 std, floor.05','no_hard_physical_identity':True,'code_revision':git_head()};save_json(out/'config.json',cfg);manifest=json.loads((RUN/'cache/window_manifest.json').read_text());results={}
 for split,ts in manifest.items():
  guard(600);x,y=windows(z,ts);p=(np.load(RUN/'cache'/f'{split}_chronosavg.npy')+np.load(RUN/'cache'/f'{split}_timesfmavg.npy'))/2;dl=np.load(RUN/'cache'/f'{split}_dlinear.npy');pred=apply(x,p,dl,cfg);results[split]=metric(pred,y);np.save(out/f'{split}_predictions.npy',pred)
 save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated'});print(name,{s:m['mse'] for s,m in results.items()},flush=True)
if __name__=='__main__':run(sys.argv[1])
