"""History-only numeric tools. Internal fits never use any future of query origin."""
from common import *
import torch,sys

def predict(x,cfg):
 kind=cfg['kind'];n=len(x)
 if kind=='seasonal':
  period=cfg.get('period',24);days=cfg.get('cycles',7)
  phase=x[:,-period*days:].reshape(n,days,period,7).mean(1)
  p=phase[:,np.arange(64)%period]
  a=cfg.get('level',0)
  p=p+a*(x[:,-24:].mean(1)-x[:,-period*days:].mean(1))[:,None]
  return p.astype(np.float32)
 if kind=='localridge':
  # Channelwise direct multioutput ridge, solved in batches on GPU0.
  lag=cfg['lag'];stride=cfg.get('stride',4);alpha=cfg['alpha']
  origins=np.arange(lag,512-64+1,stride)
  a=np.stack([x[:,t-lag:t] for t in origins],1).transpose(0,3,1,2)
  b=np.stack([x[:,t:t+64] for t in origins],1).transpose(0,3,1,2)
  center=a.mean(-1,keepdims=True);b=b-center;a=a-center
  q=x[:,-lag:].transpose(0,2,1);qc=q.mean(-1,keepdims=True);q=q-qc
  a=np.concatenate([a,np.ones((*a.shape[:-1],1))],-1);q=np.concatenate([q,np.ones((*q.shape[:-1],1))],-1)
  a=torch.as_tensor(a,dtype=torch.float64,device='cuda:0');b=torch.as_tensor(b,dtype=torch.float64,device='cuda:0');q=torch.as_tensor(q,dtype=torch.float64,device='cuda:0')
  gram=a.transpose(-1,-2)@a;reg=torch.eye(lag+1,device='cuda:0',dtype=torch.float64)*alpha*len(origins);reg[-1,-1]=1e-6
  w=torch.linalg.solve(gram+reg,a.transpose(-1,-2)@b)
  p=(q.unsqueeze(-2)@w).squeeze(-2).cpu().numpy()+qc
  return p.transpose(0,2,1).astype(np.float32)
 raise ValueError(kind)

def run(cfg,name):
 out=ROOT/'runs'/name;out.mkdir(exist_ok=True)
 subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True)
 torch.set_num_threads(4);z=np.load(RUN/'cache/trainval_z.npy');manifest=json.loads((RUN/'cache/window_manifest.json').read_text());start=time.time();results={}
 for split in ('dev','confirm','val'):
  ps=[];ts=manifest[split]
  for i in range(0,len(ts),16):
   guard(600);x,_=windows(z,ts[i:i+16]);ps.append(predict(x,cfg))
  p=np.concatenate(ps);y=np.load(RUN/'cache'/f'{split}_targets.npy');results[split]=metric(p,y);np.save(out/(split+'_predictions.npy'),p)
 record={'candidate':name,'config':cfg,'code_revision':git_head(),'elapsed_seconds':time.time()-start,'metrics':results,'history_only':True,'status':'evaluated'}
 save_json(out/'config.json',cfg);save_json(out/'metrics.json',record)
 with (RUN/'experiment_index.jsonl').open('a') as f:f.write(json.dumps({'candidate':name,'status':'evaluated','val_mse':results['val']['mse'],'kind':'history_only_tool'})+'\n')
 print(name,{s:r['mse'] for s,r in results.items()},flush=True)
if __name__=='__main__':run(json.loads(sys.argv[1]),sys.argv[2])
