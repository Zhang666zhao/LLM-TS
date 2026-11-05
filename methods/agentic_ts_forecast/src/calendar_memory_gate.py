"""Train-only chronological calendar-memory router, static/linear/MLP controls."""
from common import *
from view_memory_gate import features
from blend_sources import get_source
import torch,sys
from torch import nn

def make_net(kind,d):
 class Gate(nn.Module):
  def __init__(self):
   super().__init__();self.kind=kind
   if kind=='static':self.bias=nn.Parameter(torch.zeros(7))
   elif kind=='linear':self.net=nn.Linear(d,7);nn.init.zeros_(self.net.weight);nn.init.zeros_(self.net.bias)
   elif kind=='mlp':self.net=nn.Sequential(nn.Linear(d,32),nn.Tanh(),nn.Linear(32,7));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
   else:raise ValueError(kind)
  def forward(self,x):
   delta=self.bias[None].expand(len(x),-1) if self.kind=='static' else self.net(x)
   return (delta+np.log(.6/.4)).sigmoid()
 return Gate()

def infer(x,p,cfg,state):
 f=np.clip((features(x,p)-np.asarray(cfg['feature_mean']))/np.asarray(cfg['feature_scale']),-5,5).astype(np.float32);net=make_net(cfg['kind'],f.shape[1]).cuda();net.load_state_dict(torch.load(state,map_location='cuda:0',weights_only=True));net.eval()
 with torch.no_grad():w=net(torch.from_numpy(f).cuda()).cpu().numpy()
 del net;torch.cuda.empty_cache();return (p[:,1]*(1-w[:,None])+p[:,2]*w[:,None]).astype(np.float32)

def run(kind,bank='calendar'):
 assert bank in ('calendar','earlycalendar');assert kind in ('static','linear','mlp');name=f'20261105_calendarmemorygate_{bank}_{kind}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=False);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);guard(2400);torch.set_num_threads(4);path=RUN/f'cache/view_prefix_{bank}_memory.npz';d=np.load(path);f=d['features'];p=d['predictions'];y=d['targets'];train=d['fold']<6272;hold=~train;assert d['origins'][train].max()+64<=d['origins'][hold].min()-512
 def tensors(ids,mu,sd):return [torch.from_numpy(a).cuda() for a in (np.clip((f[ids]-mu)/sd,-5,5).astype(np.float32),p[ids],y[ids])]
 def init():torch.manual_seed(2021);torch.cuda.manual_seed_all(2021);return make_net(kind,f.shape[1]).cuda()
 def loss(net,data):
  ff,pp,yy=data;w=net(ff);pred=pp[:,1]*(1-w[:,None])+pp[:,2]*w[:,None];return (pred-yy).square().mean()+.01*(w-.6).square().mean()
 def score(net,data):
  net.eval()
  with torch.no_grad():ff,pp,yy=data;w=net(ff);return float((pp[:,1].double()*(1-w[:,None].double())+pp[:,2].double()*w[:,None].double()-yy.double()).square().mean())
 mu=f[train].mean(0);sd=np.maximum(f[train].std(0),.1);a=tensors(train,mu,sd);b=tensors(hold,mu,sd);net=init();probe=torch.randn(3,f.shape[1],device='cuda:0');w=net(probe);assert torch.allclose(w,torch.full_like(w,.6));changed=probe.clone();changed[2]+=10;assert torch.equal(w[:2],net(changed)[:2]);loss(net,a).backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in net.parameters());net=init();opt=torch.optim.AdamW(net.parameters(),lr=.01,weight_decay=.01);initial=score(net,b);best=initial;beststep=0;stale=0;records=[dict(step=0,prefix_hold_mse=initial)]
 for step in range(1,301):
  guard(1800);net.train();opt.zero_grad(set_to_none=True);ll=loss(net,a);assert torch.isfinite(ll);ll.backward();g=torch.nn.utils.clip_grad_norm_(net.parameters(),1.);assert torch.isfinite(g);opt.step();val=score(net,b);records.append(dict(step=step,train_loss=float(ll.detach()),prefix_hold_mse=val))
  if val<best-1e-9:best=val;beststep=step;stale=0
  else:stale+=1
  if stale>=30:break
 mu=f.mean(0);sd=np.maximum(f.std(0),.1);a=tensors(np.arange(len(f)),mu,sd);net=init();opt=torch.optim.AdamW(net.parameters(),lr=.01,weight_decay=.01)
 for step in range(beststep):
  guard(1800);net.train();opt.zero_grad(set_to_none=True);ll=loss(net,a);assert torch.isfinite(ll);ll.backward();g=torch.nn.utils.clip_grad_norm_(net.parameters(),1.);assert torch.isfinite(g);opt.step()
 torch.save(net.state_dict(),out/'best.pth');save_json(out/'training_selection.json',dict(records=records,selected_steps=beststep,initial_hold_mse=initial,selected_hold_mse=best,prefix_train_cases=int(train.sum()),prefix_hold_cases=int(hold.sum()),refit_cases=len(f),epoch0_eligible=True));save_json(out/'contracts.json',dict(initial_prior=.6,query_isolation=True,finite_backward=True,weights_convex=True));cfg=dict(engine='calendar_memory_gate_v1',kind=kind,feature_mean=mu.tolist(),feature_scale=sd.tolist(),source_ids=['run:20261105_transformedcalft_'+v+'_mse' for v in ('semantic','pca','random')],training_memory=str(path),training_memory_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),outputs='per-variable convex PCA/random weights; semantic predictions in history-only feature evidence, not direct mixture',selection='chronological prefix holdout inside training, zero-update prior eligible, fixed selected steps refit all bank',seed=2021,code_revision=git_head());save_json(out/'config.json',cfg);del net,opt,a,b;torch.cuda.empty_cache();z=np.load(RUN/'cache/trainval_z.npy');results={}
 for split,ts in json.loads((RUN/'cache/window_manifest.json').read_text()).items():
  x,_=windows(z,ts);pp=np.stack([get_source(k,split) for k in cfg['source_ids']],1);pred=infer(x,pp,cfg,out/'best.pth');np.save(out/f'{split}_predictions.npy',pred);results[split]=metric(pred,np.load(RUN/'cache'/f'{split}_targets.npy'))
 save_json(out/'metrics.json',dict(candidate=name,metrics=results,status='evaluated'));print(name,'steps',beststep,'hold',best,{s:m['mse'] for s,m in results.items()},flush=True)
if __name__=='__main__':run(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else 'calendar')
