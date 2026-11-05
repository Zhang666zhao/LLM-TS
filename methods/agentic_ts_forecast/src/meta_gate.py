"""Supervision-disjoint training development of small pretrained-tool error gates."""
from common import *
import torch,sys,pandas as pd
from torch import nn

def features(x,p,q,origins,dates):
 center=x.mean(1);s=np.maximum(x.std(1),.05);f=[center,s]
 for k in (24,96,168):f.extend([(x[:,-k:].mean(1)-center)/s,x[:,-k:].std(1)/s])
 f.append((x[:,-24:].mean(1)-x[:,-48:-24].mean(1))/s)
 a=x[:,:-24]-x[:,:-24].mean(1,keepdims=True);b=x[:,24:]-x[:,24:].mean(1,keepdims=True);f.append((a*b).sum(1)/np.maximum(np.sqrt((a*a).sum(1)*(b*b).sum(1)),1e-6))
 for m in range(2):f.extend([(p[:,m].mean(1)-center)/s,p[:,m].std(1)/s,(q[:,m,:,:,-1]-q[:,m,:,:,0]).mean(1)/s])
 f.append(np.abs(p[:,0]-p[:,1]).mean(1)/s)
 t=dates[np.asarray(origins)-1];hour=np.asarray(t.hour);dow=np.asarray(t.dayofweek)
 for v in (np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),np.sin(2*np.pi*dow/7),np.cos(2*np.pi*dow/7)):f.append(np.repeat(v[:,None],7,axis=1))
 a=np.stack(f,-1);ids=np.broadcast_to(np.eye(7,dtype=np.float32),(len(x),7,7));return np.concatenate([np.clip(a,-10,10),ids],-1).astype(np.float32),s.astype(np.float32)
class Gate(nn.Module):
 def __init__(self,d,kind):
  super().__init__();self.kind=kind;k=3 if kind=='bias' else 2
  self.net=nn.Sequential(nn.Linear(d,k)) if kind=='linear' else nn.Sequential(nn.Linear(d,16),nn.GELU(),nn.Linear(16,k))
  nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
 def forward(self,f,p,s):
  a=self.net(f);w=a[:,:,:2].softmax(-1);y=torch.einsum('nmhc,ncm->nhc',p,w)
  if self.kind=='bias':y=y+.25*a[:,:,2].tanh()[:,None]*s[:,None]
  return y,w

def run(kind):
 name=f'20261105_metagate_{kind}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True)
 subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True)
 guard(1200);torch.set_num_threads(4);rev=git_head();start=time.time();z=np.load(RUN/'cache/trainval_z.npy');dates=pd.to_datetime(json.loads((RUN/'cache/dates.json').read_text()));manifest=json.loads((RUN/'cache/window_manifest.json').read_text())
 ch=np.load(RUN/'cache/dense_chronos_train.npz');tf=np.load(RUN/'cache/dense_timesfm_train.npz');assert np.array_equal(ch['origins'],tf['origins']);ts=ch['origins'];x,y=windows(z,ts);p=np.stack([ch['predictions'],tf['predictions']],1);q=np.stack([ch['quantiles'],tf['quantiles']],1);f,s=features(x,p,q,ts,dates)
 data={}
 for split,origins in manifest.items():
  sx,sy=windows(z,origins);cq=np.load(ROOT/'runs/20261105_chronos2_median_mse'/f'{split}_quantiles.npy');tq=np.load(ROOT/'runs/20261105_timesfm3_sym1_mse'/f'{split}_quantiles.npy');sq=np.stack([cq,tq],1);sp=sq.mean(-1);sf,ss=features(sx,sp,sq,origins,dates);data[split]=[torch.tensor(a,device='cuda:0') for a in (sf,sp,ss,sy)]
 full=[torch.tensor(a,device='cuda:0') for a in (f,p,s,y)];ix=np.where(ts+64<=4096)[0];train=[a[ix] for a in full];assert max(ts[ix]+64)<=min(manifest['dev'])-512
 def init():
  torch.manual_seed(2021);torch.cuda.manual_seed_all(2021);return Gate(f.shape[-1],kind).cuda(0)
 def score(net,items):
  net.eval()
  with torch.no_grad():return float(((net(*items[:3])[0]-items[3]).double()**2).mean())
 net=init();shuffle_rng=torch.Generator().manual_seed(2021);opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.01);best=score(net,data['confirm']);best_epoch=0;stale=0;records=[{'epoch':0,'confirm_mse':best}];torch.save(net.state_dict(),out/'prefix_best.pth')
 for epoch in range(1,81):
  guard(900);net.train();order=torch.randperm(len(ix),generator=shuffle_rng);loss_sum=0.
  for ids in order.split(64):
   opt.zero_grad(set_to_none=True);a=[v[ids] for v in train];pr,w=net(*a[:3]);loss=(pr-a[3]).square().mean()+.01*(w-.5).square().mean()
   if not torch.isfinite(loss):raise RuntimeError('nonfinite meta loss')
   loss.backward();opt.step();loss_sum+=float(loss.detach())*len(ids)
  sc=score(net,data['confirm']);records.append({'epoch':epoch,'train_objective':loss_sum/len(ix),'confirm_mse':sc})
  if sc<best:best=sc;best_epoch=epoch;stale=0;torch.save(net.state_dict(),out/'prefix_best.pth')
  else:stale+=1
  if stale>=10:break
 save_json(out/'development_epochs.json',records);net.load_state_dict(torch.load(out/'prefix_best.pth',map_location='cuda:0'));prefix_scores={k:score(net,v) for k,v in data.items() if k in ('dev','confirm')};steps=best_epoch*int(np.ceil(len(ix)/64))
 net=init();refit_rng=torch.Generator().manual_seed(2021);opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.01);order=[]
 for step in range(steps):
  guard(900)
  if not order:order=list(torch.randperm(len(ts),generator=refit_rng).split(64))
  ids=order.pop(0);a=[v[ids] for v in full];opt.zero_grad(set_to_none=True);pr,w=net(*a[:3]);loss=(pr-a[3]).square().mean()+.01*(w-.5).square().mean();loss.backward();opt.step()
 torch.save(net.state_dict(),out/'best.pth');net.eval();results={}
 with torch.no_grad():
  for split,items in data.items():
   pr,w=net(*items[:3]);pr=pr.cpu().numpy();results[split]=metric(pr,items[3].cpu().numpy());np.save(out/f'{split}_predictions.npy',pr);np.save(out/f'{split}_weights.npy',w.cpu().numpy())
 cfg={'kind':kind,'features_dim':f.shape[-1],'tools':['chronosavg','timesfmavg'],'seed':2021,'prefix_training_label_end':4096,'prefix_raw_support_disjoint_dev':True,'fixed_scaler_scope':'official full training only','prefix_train_windows':len(ix),'full_train_pool_windows':len(ts),'refit_steps':steps,'selected_epoch':best_epoch,'selection':'training-confirm block only; validation not used to fit/select gate checkpoint','initial_noop_allowed':True,'prefix_scores':prefix_scores,'code_revision':rev};save_json(out/'config.json',cfg);save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated','elapsed_seconds':time.time()-start})
 with (RUN/'experiment_index.jsonl').open('a') as h:h.write(json.dumps({'candidate':name,'val_mse':results['val']['mse'],'status':'evaluated','kind':'numeric_meta_gate'})+'\n')
 print(name,'epoch',best_epoch,'steps',steps,{k:v['mse'] for k,v in results.items()},flush=True)
if __name__=='__main__':run(sys.argv[1])
