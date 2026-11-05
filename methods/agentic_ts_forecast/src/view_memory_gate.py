"""Train numerical tool selectors on chronological prefix-held-out error memories."""
from common import *
from view_routing import TOOLS
import sys

def features(x,p):
 x=x.astype(np.float64);p=p.astype(np.float64);v=[]
 for k in (24,96,512):v.append(x[:,-k:].mean(1))
 for k in (64,512):v.append(x[:,-k:].std(1))
 v.extend([x[:,-24:].mean(1)-x[:,-48:-24].mean(1),x[:,-168:].mean(1)-x[:,-336:-168].mean(1)])
 a=np.clip((x[:,-168:]-x[:,-168:].mean(1,keepdims=True))/np.maximum(x[:,-168:].std(1,keepdims=True),.05),-10,10);v.extend([(a**3).mean(1),(a**4).mean(1)-3])
 for lag in (24,168):
  a=x[:,:-lag]-x[:,:-lag].mean(1,keepdims=True);b=x[:,lag:]-x[:,lag:].mean(1,keepdims=True);v.append((a*b).mean(1)/np.maximum(a.std(1)*b.std(1),1e-5))
 for j in range(3):v.extend([p[:,j].mean(1),p[:,j].std(1),p[:,j,:16].mean(1),p[:,j,-16:].mean(1)])
 for j,k in ((0,1),(0,2),(1,2)):v.append(np.sqrt(((p[:,j]-p[:,k])**2).mean(1)))
 a=(x[:,-168:]-x[:,-168:].mean(1,keepdims=True))/np.maximum(x[:,-168:].std(1,keepdims=True),.05);corr=np.einsum('ntc,ntd->ncd',a,a)/168;v.append(corr[:,np.triu_indices(7,1)[0],np.triu_indices(7,1)[1]]);f=np.concatenate(v,1).astype(np.float32);assert np.isfinite(f).all();return f

def build():
 z=np.load(RUN/'cache/trainval_z.npy')[:8640];orig=[];fold=[];pred=[];truth=[];sources=[]
 for cut in (3072,4096,6272):
  pp=[];ts=None;y=None
  for mode in ('semantic','pca','random'):
   out=ROOT/'runs'/f'20261105_prefixview_{cut}_{mode}_mse';cfg=json.loads((out/'config.json').read_text());t=np.load(out/'origins.npy');yy=np.load(out/'targets.npy');a=np.load(out/'predictions.npy');assert cfg['cutoff']==cut and t.min()-512>=cut and t.max()+64<=cfg['labels_end']<=8640
   if ts is not None:assert np.array_equal(ts,t) and np.array_equal(y,yy)
   ts=t;y=yy;pp.append(a);sources.append({'name':out.name,'config_sha256':hashlib.sha256((out/'config.json').read_bytes()).hexdigest(),'prediction_sha256':hashlib.sha256((out/'predictions.npy').read_bytes()).hexdigest()})
  orig.extend(ts);fold.extend([cut]*len(ts));pred.append(np.stack(pp,1));truth.append(y)
 ts=np.array(orig);pred=np.concatenate(pred);truth=np.concatenate(truth);x,_=windows(z,ts);f=features(x,pred);path=RUN/'cache/view_prefix_memory.npz';assert not path.exists(),'Immutable memory already built';np.savez_compressed(path,origins=ts,fold=np.array(fold),predictions=pred,targets=truth,features=f);assert max(ts[np.array(fold)<6272])+64<=min(ts[np.array(fold)==6272])-512;save_json(path.with_suffix('.json'),{'case_count':len(ts),'sources':sources,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'chronological_development_folds':[3072,4096],'holdout_fold':6272,'caveats':['Overlapping stride8 examples are not independent','Proxy weights and view transforms fit only prefix; official full-training scaler shared','Full-data tools at deployment differ from prefix-fitted tools','External pretraining overlap unverified']});print('VIEW_MEMORY_BUILT',len(ts),f.shape,flush=True)

def make_net(kind,dim):
 import torch
 from torch import nn
 class Gate(nn.Module):
  def __init__(self):
   super().__init__();self.kind=kind
   if kind=='static':self.logits=nn.Parameter(torch.zeros(3,7))
   elif kind=='linear':self.net=nn.Linear(dim,21);nn.init.zeros_(self.net.weight);nn.init.zeros_(self.net.bias)
   elif kind=='mlp':self.net=nn.Sequential(nn.Linear(dim,32),nn.Tanh(),nn.Linear(32,21));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
   else:raise ValueError(kind)
  def forward(self,f):
   a=self.logits[None].expand(len(f),-1,-1) if self.kind=='static' else self.net(f).reshape(-1,3,7);return torch.softmax(a,dim=1)
 return Gate()
def infer(x,p,cfg,state):
 import torch
 f=features(x,p);f=np.clip((f-np.asarray(cfg['feature_mean']))/np.asarray(cfg['feature_scale']),-5,5).astype(np.float32);net=make_net(cfg['kind'],f.shape[1]).cuda();net.load_state_dict(torch.load(state,map_location='cuda:0',weights_only=False));net.eval()
 with torch.no_grad():w=net(torch.from_numpy(f).cuda()).cpu().numpy()
 del net;torch.cuda.empty_cache();return (p*w[:,:,None]).sum(1).astype(np.float32),w

def run(kind):
 import torch
 name=f'20261105_viewmemory_{kind}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);guard(1800);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;data=np.load(RUN/'cache/view_prefix_memory.npz');f=data['features'];p=data['predictions'];y=data['targets'];train=data['fold']<6272;hold=~train;assert data['origins'][train].max()+64<=data['origins'][hold].min()-512;start=time.time()
 def tensors(ix,mu,sd):return [torch.from_numpy(a).cuda() for a in (np.clip((f[ix]-mu)/sd,-5,5).astype(np.float32),p[ix],y[ix])]
 def init():torch.manual_seed(2021);torch.cuda.manual_seed_all(2021);net=make_net(kind,f.shape[1]).cuda();return net,torch.optim.AdamW(net.parameters(),lr=.01,weight_decay=.01)
 def step(net,opt,a):
  net.train();ff,pp,yy=a;opt.zero_grad(set_to_none=True);w=net(ff);pred=(pp*w[:,:,None]).sum(1);loss=(pred-yy).square().mean()+.01*(w-1/3).square().mean()
  if not torch.isfinite(loss):raise RuntimeError('nonfinite memory-gate training')
  loss.backward();g=torch.nn.utils.clip_grad_norm_(net.parameters(),1.);assert torch.isfinite(g);opt.step();return float(loss.detach())
 mu=f[train].mean(0);sd=np.maximum(f[train].std(0),.1);a=tensors(train,mu,sd);b=tensors(hold,mu,sd);net,opt=init();best=float('inf');beststep=0;stale=0;records=[]
 for i in range(300):
  guard(900);loss=step(net,opt,a);net.eval()
  with torch.no_grad():w=net(b[0]);score=float(((b[1]*w[:,:,None]).sum(1)-b[2]).square().mean())
  records.append({'step':i+1,'train_loss_regularized':loss,'heldout_prefix_mse':score})
  if score<best-1e-7:best=score;beststep=i+1;stale=0
  else:stale+=1
  if stale>=30:break
 save_json(out/'development_curve.json',records);del net,opt,a,b;torch.cuda.empty_cache();mu=f.mean(0);sd=np.maximum(f.std(0),.1);allidx=np.ones(len(f),bool);a=tensors(allidx,mu,sd);net,opt=init()
 for i in range(beststep):guard(900);step(net,opt,a)
 torch.save(net.state_dict(),out/'best.pth');del net,opt,a;torch.cuda.empty_cache();cfg={'engine':'view_memory_gate_v1','kind':kind,'feature_mean':mu.tolist(),'feature_scale':sd.tolist(),'feature_clip':5,'selected_steps':beststep,'selection':'last chronological TRAINING prefix fold6272; refit all483 training-memory cases; official validation never fitted','heldout_prefix_mse':best,'source_ids':['run:'+n for n in TOOLS],'training_memory':str(RUN/'cache/view_prefix_memory.npz'),'training_memory_sha256':hashlib.sha256((RUN/'cache/view_prefix_memory.npz').read_bytes()).hexdigest(),'runtime_llm':False,'seed':2021,'code_revision':git_head()};save_json(out/'config.json',cfg);z=np.load(RUN/'cache/trainval_z.npy');mf=json.loads((RUN/'cache/window_manifest.json').read_text());results={}
 for split,ts in mf.items():
  x=np.stack([z[t-512:t] for t in ts]);pp=np.stack([np.load(ROOT/'runs'/n/f'{split}_predictions.npy') for n in TOOLS],1);pr,w=infer(x,pp,cfg,out/'best.pth');results[split]=metric(pr,np.load(RUN/'cache'/f'{split}_targets.npy'));np.save(out/f'{split}_predictions.npy',pr);np.save(out/f'{split}_weights.npy',w)
 save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated','elapsed_seconds':time.time()-start});print(name,'selected_steps',beststep,'prefix_holdout',best,{s:m['mse'] for s,m in results.items()},flush=True)
if __name__=='__main__':
 if sys.argv[1]=='build':build()
 else:run(sys.argv[1])
