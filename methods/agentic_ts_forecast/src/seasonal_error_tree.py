"""Hierarchical residual memory, selected on chronological training-prefix holdout."""
from common import *
from view_memory_gate import features
from calendar_features import cov as calendar_cov
from view_residual_memory import PRIOR
from torch import nn
class ErrorTree(nn.Module):
 def __init__(self,d,depth):
  super().__init__();self.depth=depth;self.route_weight=nn.Parameter(torch.empty(7,d,2**depth-1));self.route_bias=nn.Parameter(torch.zeros(7,2**depth-1));self.leaves=nn.Parameter(torch.zeros(7,2**depth,24));nn.init.normal_(self.route_weight,std=.03)
 def forward(self,f,s,phase):
  a=(torch.einsum("ncd,cdl->ncl",f,self.route_weight)+self.route_bias).sigmoid();mass=torch.ones((*f.shape[:2],1),device=f.device);offset=0
  for depth in range(self.depth):
   width=2**depth;v=a[:,:,offset:offset+width];mass=torch.stack([mass*(1-v),mass*v],-1).flatten(-2);offset+=width
  profile=torch.einsum("ncl,clh->nch",mass,self.leaves);corr=.5*profile.tanh()*s[:,:,None];return corr.gather(2,phase[:,None].expand(-1,7,-1)).transpose(1,2),mass
def contracts(out,dim):
 net=ErrorTree(dim,2).cuda();f=torch.randn(3,7,dim,device="cuda");s=torch.ones(3,7,device="cuda");phase=torch.arange(64,device="cuda")[None].repeat(3,1)%24;zero=net(f,s,phase)[0];assert torch.count_nonzero(zero)==0
 with torch.no_grad():net.leaves.normal_(0,.1)
 p=net(f,s,phase)[0];ff=f.clone();ff[2]+=10;assert torch.equal(p[:2],net(ff,s,phase)[0][:2]);ff=f.clone();ff[:,0]+=10;assert torch.equal(p[:,:,1:],net(ff,s,phase)[0][:,:,1:]);loss=p.square().mean();loss.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in net.parameters());save_json(out/"contracts.json",dict(zero_initial_residual=True,query_isolation=True,per_channel_parameter_isolation=True,finite_backward=True));del net;torch.cuda.empty_cache()
from blend_sources import get_source
import torch,pandas as pd,sys
TOOLS=['20261105_fullpoint_semantic_mse','20261105_fullpoint_pca_mse','20261105_viewpoint_full_integral_mse'];STRENGTH=.25

def inputs(z,dates,ts,p,mean,scale,calendar):
 x=np.stack([z[t-512:t] for t in ts]);f=features(x,p);a=np.clip((f-mean)/scale,-5,5);a=np.concatenate([np.repeat(a[:,None,:],7,axis=1),np.broadcast_to(np.eye(7),(len(x),7,7))],-1).astype(np.float32);dt=pd.to_datetime(dates);clock=np.stack(list(calendar_cov(pd.DatetimeIndex([dt[t-1]+pd.Timedelta(hours=1) for t in ts]),calendar).values()),-1);a=np.concatenate([a,np.repeat(clock[:,None,:],7,axis=1)],-1).astype(np.float32);s=np.maximum(x.std(1),.05).astype(np.float32);phase=np.array([((dt[t-1].hour+1)+np.arange(64))%24 for t in ts]);base=(p*PRIOR[None,:,None,None]).sum(1).astype(np.float32)
 return [torch.from_numpy(v).cuda() for v in (a,s,phase,base)]

def infer(z,dates,ts,p,cfg,state):
 net=ErrorTree(cfg['features_dim'],cfg['depth']).cuda();net.load_state_dict(torch.load(state,map_location='cuda:0',weights_only=True));net.eval();a=inputs(z,dates,ts,p,np.asarray(cfg['feature_mean']),np.asarray(cfg['feature_scale']),cfg['routing_calendar'])
 with torch.no_grad():cor,mass=net(a[0],a[1],a[2]);result=(a[3]+cfg['strength']*cor).cpu().numpy()
 del net,a;torch.cuda.empty_cache();return result

def run(calendar,kind="earlycalendar"):
 depth=2;strength=.25;dim=210+{"hour":2,"hourweek":4,"hourweekyear":6}[calendar]
 assert kind in ("cdf","calendar","earlycalendar");tools=TOOLS if kind=="cdf" else [f"20261105_transformedcalft_{v}_mse" for v in ("semantic","pca","random")]
 assert depth in (2,3,4);name=f'20261105_seasonaltree_{kind}_{calendar}_mse';out=ROOT/'runs'/name;assert not (out/'config.json').exists();out.mkdir(exist_ok=True);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);guard(2400);torch.set_num_threads(4);contracts(out,dim);path=RUN/('cache/view_prefix_integral_memory.npz' if kind=='cdf' else f'cache/view_prefix_{kind}_memory.npz');d=np.load(path);ts=d['origins'];p=d['predictions'];y=d['targets'];f=d['features'];train=d['fold']<6272;hold=~train;assert ts[train].max()+64<=ts[hold].min()-512;z=np.load(RUN/'cache/trainval_z.npy');dates=json.loads((RUN/'cache/dates.json').read_text());mean=f[train].mean(0);scale=np.maximum(f[train].std(0),.1);a=inputs(z,dates,ts,p,mean,scale,calendar);target=torch.from_numpy(y).cuda();ix=np.where(train)[0];hh=np.where(hold)[0]
 def init():torch.manual_seed(2021);torch.cuda.manual_seed_all(2021);return ErrorTree(dim,depth).cuda()
 def score(net):
  net.eval()
  with torch.no_grad():return float(((a[3][hh]+strength*net(a[0][hh],a[1][hh],a[2][hh])[0]).double()-target[hh].double()).square().mean())
 def step(net,opt,data,tgt,ids):
  net.train();opt.zero_grad(set_to_none=True);cor,_=net(data[0][ids],data[1][ids],data[2][ids]);loss=(data[3][ids]+strength*cor-tgt[ids]).square().mean()+.001*net.leaves.square().mean();assert torch.isfinite(loss);loss.backward();g=torch.nn.utils.clip_grad_norm_(net.parameters(),1.);assert torch.isfinite(g);opt.step();return float(loss.detach())
 net=init();opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.01);best=score(net);initial=best;best_epoch=0;stale=0;rng=torch.Generator().manual_seed(2021);records=[{'epoch':0,'prefix_hold_mse':best}]
 for epoch in range(1,101):
  guard(1800);loss=0
  for jj in torch.randperm(len(ix),generator=rng).split(64):loss+=step(net,opt,a,target,ix[jj.numpy()])*len(jj)
  val=score(net);records.append({'epoch':epoch,'train_loss':loss/len(ix),'prefix_hold_mse':val})
  if val<best-1e-9:best=val;best_epoch=epoch;stale=0
  else:stale+=1
  if stale>=12:break
 steps=best_epoch*int(np.ceil(len(ix)/64));mean=f.mean(0);scale=np.maximum(f.std(0),.1);a=inputs(z,dates,ts,p,mean,scale,calendar);net=init();opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.01);rng=torch.Generator().manual_seed(2021);order=[]
 for j in range(steps):
  guard(1800)
  if not order:order=list(torch.randperm(len(ts),generator=rng).split(64))
  step(net,opt,a,target,order.pop(0))
 torch.save(net.state_dict(),out/'best.pth');save_json(out/'training_selection.json',{'records':records,'epoch0_eligible':True,'initial_hold_mse':initial,'selected_hold_mse':best,'selected_epoch':best_epoch,'refit_steps':steps,'prefix_train_cases':int(train.sum()),'prefix_hold_cases':int(hold.sum()),'final_fit_cases':len(ts)});del net,opt,a,target;torch.cuda.empty_cache();cfg={'engine':'seasonal_error_tree_v1','depth':depth,'features_dim':dim,'routing_calendar':calendar,'feature_mean':mean.tolist(),'feature_scale':scale.tolist(),'source_ids':['run:'+v for v in tools],'training_memory':str(path),'training_memory_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'strength':strength,'selected_epoch':best_epoch,'refit_steps':steps,'seed':2021,'outputs':'SEPARATE7variable trees and24-hour residual leaf templates, per-variable routed from203 global features plus7-variable identity and known-origin clock phases, bounded0.5 history std before configured strength','selection':'chronological prefix holdout inside training, epoch0 eligible; no official validation fit','code_revision':git_head()};save_json(out/'config.json',cfg);results={}
 for split,orig in json.loads((RUN/'cache/window_manifest.json').read_text()).items():
  pp=np.stack([get_source(k,split) for k in cfg['source_ids']],1);pred=infer(z,dates,orig,pp,cfg,out/'best.pth');assert np.isfinite(pred).all();np.save(out/f'{split}_predictions.npy',pred);results[split]=metric(pred,np.load(RUN/'cache'/f'{split}_targets.npy'))
 save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated'});print(name,'epoch',best_epoch,'hold',best,{s:m['mse'] for s,m in results.items()},flush=True)
if __name__=='__main__':run(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else 'earlycalendar')
