"""Hierarchical residual memory, selected on chronological training-prefix holdout."""
from common import *
from view_memory_gate import features
from view_residual_memory import PRIOR
from error_tree import ErrorTree
from blend_sources import get_source
import torch,pandas as pd,sys
TOOLS=['20261105_fullpoint_semantic_mse','20261105_fullpoint_pca_mse','20261105_viewpoint_full_integral_mse'];STRENGTH=.25

def inputs(z,dates,ts,p,mean,scale):
 x=np.stack([z[t-512:t] for t in ts]);f=features(x,p);a=np.clip((f-mean)/scale,-5,5);a=np.concatenate([np.repeat(a[:,None,:],7,axis=1),np.broadcast_to(np.eye(7),(len(x),7,7))],-1).astype(np.float32);s=np.maximum(x.std(1),.05).astype(np.float32);dt=pd.to_datetime(dates);phase=np.array([((dt[t-1].hour+1)+np.arange(64))%24 for t in ts]);base=(p*PRIOR[None,:,None,None]).sum(1).astype(np.float32)
 return [torch.from_numpy(v).cuda() for v in (a,s,phase,base)]

def infer(z,dates,ts,p,cfg,state):
 net=ErrorTree(210,cfg['depth']).cuda();net.load_state_dict(torch.load(state,map_location='cuda:0',weights_only=True));net.eval();a=inputs(z,dates,ts,p,np.asarray(cfg['feature_mean']),np.asarray(cfg['feature_scale']))
 with torch.no_grad():cor,mass=net(a[0],a[1],a[2]);result=(a[3]+STRENGTH*cor).cpu().numpy()
 del net,a;torch.cuda.empty_cache();return result

def run(depth,kind="cdf",tag=None):
 assert kind in ("cdf","calendar","earlycalendar");tools=TOOLS if kind=="cdf" else [f"20261105_transformedcalft_{v}_mse" for v in ("semantic","pca","random")]
 assert depth in (2,3,4);name=f'20261105_prefixtree_{kind}_d{depth}'+('_'+tag if tag else '')+'_mse';out=ROOT/'runs'/name;assert not (out/'config.json').exists();out.mkdir(exist_ok=True);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);guard(2400);torch.set_num_threads(4);path=RUN/('cache/view_prefix_integral_memory.npz' if kind=='cdf' else f'cache/view_prefix_{kind}_memory.npz');d=np.load(path);ts=d['origins'];p=d['predictions'];y=d['targets'];f=d['features'];train=d['fold']<6272;hold=~train;assert ts[train].max()+64<=ts[hold].min()-512;z=np.load(RUN/'cache/trainval_z.npy');dates=json.loads((RUN/'cache/dates.json').read_text());mean=f[train].mean(0);scale=np.maximum(f[train].std(0),.1);a=inputs(z,dates,ts,p,mean,scale);target=torch.from_numpy(y).cuda();ix=np.where(train)[0];hh=np.where(hold)[0]
 def init():torch.manual_seed(2021);torch.cuda.manual_seed_all(2021);return ErrorTree(210,depth).cuda()
 def score(net):
  net.eval()
  with torch.no_grad():return float(((a[3][hh]+STRENGTH*net(a[0][hh],a[1][hh],a[2][hh])[0]).double()-target[hh].double()).square().mean())
 def step(net,opt,data,tgt,ids):
  net.train();opt.zero_grad(set_to_none=True);cor,_=net(data[0][ids],data[1][ids],data[2][ids]);loss=(data[3][ids]+STRENGTH*cor-tgt[ids]).square().mean()+.001*net.leaves.square().mean();assert torch.isfinite(loss);loss.backward();g=torch.nn.utils.clip_grad_norm_(net.parameters(),1.);assert torch.isfinite(g);opt.step();return float(loss.detach())
 net=init();opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.01);best=score(net);initial=best;best_epoch=0;stale=0;rng=torch.Generator().manual_seed(2021);records=[{'epoch':0,'prefix_hold_mse':best}]
 for epoch in range(1,101):
  guard(1800);loss=0
  for jj in torch.randperm(len(ix),generator=rng).split(64):loss+=step(net,opt,a,target,ix[jj.numpy()])*len(jj)
  val=score(net);records.append({'epoch':epoch,'train_loss':loss/len(ix),'prefix_hold_mse':val})
  if val<best-1e-9:best=val;best_epoch=epoch;stale=0
  else:stale+=1
  if stale>=12:break
 steps=best_epoch*int(np.ceil(len(ix)/64));mean=f.mean(0);scale=np.maximum(f.std(0),.1);a=inputs(z,dates,ts,p,mean,scale);net=init();opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.01);rng=torch.Generator().manual_seed(2021);order=[]
 for j in range(steps):
  guard(1800)
  if not order:order=list(torch.randperm(len(ts),generator=rng).split(64))
  step(net,opt,a,target,order.pop(0))
 torch.save(net.state_dict(),out/'best.pth');save_json(out/'training_selection.json',{'records':records,'epoch0_eligible':True,'initial_hold_mse':initial,'selected_hold_mse':best,'selected_epoch':best_epoch,'refit_steps':steps,'prefix_train_cases':int(train.sum()),'prefix_hold_cases':int(hold.sum()),'final_fit_cases':len(ts)});del net,opt,a,target;torch.cuda.empty_cache();cfg={'engine':'prefix_error_tree_v1','depth':depth,'features_dim':210,'feature_mean':mean.tolist(),'feature_scale':scale.tolist(),'source_ids':['run:'+v for v in tools],'training_memory':str(path),'training_memory_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'strength':STRENGTH,'selected_epoch':best_epoch,'refit_steps':steps,'seed':2021,'outputs':'shared24-hour residual leaf templates, per-variable routed from203 global features plus7-variable identity, bounded0.5 history std beforestrength.25','selection':'chronological prefix holdout inside training, epoch0 eligible; no official validation fit','code_revision':git_head()};save_json(out/'config.json',cfg);results={}
 for split,orig in json.loads((RUN/'cache/window_manifest.json').read_text()).items():
  pp=np.stack([get_source(k,split) for k in cfg['source_ids']],1);pred=infer(z,dates,orig,pp,cfg,out/'best.pth');assert np.isfinite(pred).all();np.save(out/f'{split}_predictions.npy',pred);results[split]=metric(pred,np.load(RUN/'cache'/f'{split}_targets.npy'))
 save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated'});print(name,'epoch',best_epoch,'hold',best,{s:m['mse'] for s,m in results.items()},flush=True)
if __name__=='__main__':run(int(sys.argv[1]),sys.argv[2] if len(sys.argv)>2 else 'cdf',sys.argv[3] if len(sys.argv)>3 else None)
