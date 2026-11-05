"""GPU-fitted direct ridge numerical tools with train-only labels and fixed horizons."""
from common import *
import torch,sys

def features(x,cfg):
 # Each variable is an independent sample; local centering is reversible.
 center=x[:,-cfg['center']:].mean(1,keepdims=True) if cfg['center'] else np.zeros((len(x),1,7),np.float32)
 v=x-center
 if cfg['features']=='full':f=v
 elif cfg['features']=='multiscale':f=np.concatenate([v.reshape(-1,128,4,7).mean(2),v[:,-96:]],axis=1)
 else:raise ValueError('Unknown feature schema')
 return f.transpose(0,2,1).astype(np.float64),center.transpose(0,2,1).astype(np.float64)

def fit(x,y,cfg):
 f,c=features(x,cfg);t=y.transpose(0,2,1).astype(np.float64)-c
 weights=[]
 groups=range(7) if cfg['individual'] else [None]
 for channel in groups:
  a=f[:,channel] if channel is not None else f.reshape(-1,f.shape[-1]);b=t[:,channel] if channel is not None else t.reshape(-1,64)
  a=np.concatenate([a,np.ones((len(a),1))],1)
  a=torch.from_numpy(a).cuda(0);b=torch.from_numpy(b).cuda(0)
  gram=a.T@a;reg=torch.eye(gram.shape[0],device=a.device,dtype=a.dtype)*cfg['alpha']*len(a);reg[-1,-1]=1e-8
  w=torch.linalg.solve(gram+reg,a.T@b);weights.append(w.cpu().numpy())
 return np.stack(weights)
def predict(x,cfg,w):
 f,c=features(x,cfg);f=np.concatenate([f,np.ones((*f.shape[:2],1))],2)
 y=np.einsum('ncf,cfh->nch',f,w if cfg['individual'] else np.broadcast_to(w,(7,*w.shape[1:])))+c
 return y.transpose(0,2,1).astype(np.float32)

def run(cfg,name):
 out=ROOT/'runs'/name;out.mkdir(exist_ok=True)
 subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True)
 guard(600);torch.set_num_threads(4);z=np.load(RUN/'cache/trainval_z.npy');start=time.time()
 x,y=windows(z,np.arange(512,8640-64+1,4));w=fit(x,y,cfg);np.save(out/'weights.npy',w)
 manifest=json.loads((RUN/'cache/window_manifest.json').read_text());results={}
 for split in ('dev','confirm','val'):
  guard(600);ts=manifest[split];ps=[]
  for i in range(0,len(ts),128):
   bx,_=windows(z,ts[i:i+128]);ps.append(predict(bx,cfg,w))
  p=np.concatenate(ps);target=np.load(RUN/'cache'/f'{split}_targets.npy');results[split]=metric(p,target);np.save(out/(split+'_predictions.npy'),p)
 record={'candidate':name,'config':cfg,'code_revision':git_head(),'elapsed_seconds':time.time()-start,'status':'evaluated','train_stride':4,'dev_confirm_in_sample':True,'metrics':results}
 save_json(out/'config.json',cfg);save_json(out/'metrics.json',record)
 with (RUN/'experiment_index.jsonl').open('a') as f:f.write(json.dumps({'candidate':name,'status':'evaluated','val_mse':results['val']['mse'],'kind':'numeric_tool_control'})+'\n')
 print(name,results['val']['mse'],flush=True)
if __name__=='__main__':
 cfg=json.loads(sys.argv[1]);run(cfg,sys.argv[2])
