"""Train-only differentiable hierarchical residual memory for frozen pretrained tools."""
from common import *
from meta_gate import features
import torch,pandas as pd,sys
from torch import nn
class ErrorTree(nn.Module):
 def __init__(self,d,depth):
  super().__init__();self.depth=depth;self.route=nn.Linear(d,2**depth-1);self.leaves=nn.Parameter(torch.zeros(2**depth,24));nn.init.normal_(self.route.weight,std=.03);nn.init.zeros_(self.route.bias)
 def forward(self,f,s,phase):
  a=self.route(f).sigmoid();mass=torch.ones((*f.shape[:2],1),device=f.device);offset=0
  for depth in range(self.depth):
   width=2**depth;v=a[:,:,offset:offset+width];mass=torch.stack([mass*(1-v),mass*v],-1).flatten(-2);offset+=width
  profile=torch.einsum('ncl,lh->nch',mass,self.leaves);corr=.5*profile.tanh()*s[:,:,None];index=phase[:,None].expand(-1,7,-1);return corr.gather(2,index).transpose(1,2),mass

def run(depth):
 name=f'20261105_errortree_d{depth}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True)
 guard(1200);torch.set_num_threads(4);rev=git_head();start=time.time();z=np.load(RUN/'cache/trainval_z.npy');dates=pd.to_datetime(json.loads((RUN/'cache/dates.json').read_text()));manifest=json.loads((RUN/'cache/window_manifest.json').read_text());ch=np.load(RUN/'cache/dense_chronos_train.npz');tf=np.load(RUN/'cache/dense_timesfm_train.npz');assert np.array_equal(ch['origins'],tf['origins']);ts=ch['origins'];x,y=windows(z[:8640],ts);p=np.stack([ch['predictions'],tf['predictions']],1);q=np.stack([ch['quantiles'],tf['quantiles']],1);f,s=features(x,p,q,ts,dates)
 def items(f,s,p,y,orig):return [torch.tensor(a,device='cuda:0') for a in (f,s,p.mean(1),y,(np.asarray(orig)[:,None]+np.arange(64))%24)]
 full=items(f,s,p,y,ts);ix=np.where(ts+64<=4096)[0];train=[a[ix] for a in full];assert max(ts[ix]+64)<=min(manifest['dev'])-512;data={};dlinear={}
 for split,orig in manifest.items():
  xx,yy=windows(z,orig);qq=np.stack([np.load(ROOT/'runs'/n/f'{split}_quantiles.npy') for n in ('20261105_chronos2_median_mse','20261105_timesfm3_sym1_mse')],1);pp=qq.mean(-1);ff,ss=features(xx,pp,qq,orig,dates);data[split]=items(ff,ss,pp,yy,orig);dlinear[split]=np.load(RUN/'cache'/f'{split}_dlinear.npy')
 def init():torch.manual_seed(2021);torch.cuda.manual_seed_all(2021);return ErrorTree(f.shape[-1],depth).cuda()
 def score(net,items):
  net.eval()
  with torch.no_grad():return float(((items[2]+net(items[0],items[1],items[4])[0]-items[3]).double()**2).mean())
 net=init();rng=torch.Generator().manual_seed(2021);opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.01);best=score(net,data['confirm']);best_epoch=0;stale=0;records=[{'epoch':0,'confirm_pretrained_mixture_mse':best}];torch.save(net.state_dict(),out/'prefix_best.pth')
 def trainstep(net,opt,a):
  opt.zero_grad(set_to_none=True);cor,mass=net(a[0],a[1],a[4]);loss=(a[2]+cor-a[3]).square().mean()+.001*net.leaves.square().mean()
  if not torch.isfinite(loss):raise RuntimeError('nonfinite error-memory loss')
  loss.backward();opt.step();return float(loss.detach())
 for epoch in range(1,101):
  guard(900);net.train();loss=0
  for ids in torch.randperm(len(ix),generator=rng).split(64):loss+=trainstep(net,opt,[a[ids] for a in train])*len(ids)
  sc=score(net,data['confirm']);records.append({'epoch':epoch,'train_loss':loss/len(ix),'confirm_pretrained_mixture_mse':sc})
  if sc<best:best=sc;best_epoch=epoch;stale=0;torch.save(net.state_dict(),out/'prefix_best.pth')
  else:stale+=1
  if stale>=12:break
 steps=best_epoch*int(np.ceil(len(ix)/64));save_json(out/'development_epochs.json',records);net=init();rng=torch.Generator().manual_seed(2021);opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.01);order=[]
 for step in range(steps):
  guard(900)
  if not order:order=list(torch.randperm(len(ts),generator=rng).split(64))
  ids=order.pop(0);trainstep(net,opt,[a[ids] for a in full])
 torch.save(net.state_dict(),out/'best.pth');net.eval();results={}
 with torch.no_grad():
  for split,a in data.items():
   cor,mass=net(a[0],a[1],a[4]);p=((dlinear[split]+2*(a[2]+cor).cpu().numpy())/3).astype(np.float32);results[split]=metric(p,a[3].cpu().numpy());np.save(out/f'{split}_predictions.npy',p);np.save(out/f'{split}_leaf_probabilities.npy',mass.cpu().numpy())
 save_json(out/'config.json',{'engine':'error_tree_v1','depth':depth,'leaves':2**depth,'features_dim':f.shape[-1],'leaf_outputs':'24 clock-phase residual templates in units of history std, bounded +/-.5','prefix_training_label_end':4096,'selected_epoch':best_epoch,'refit_steps':steps,'selection':'training-confirm pretrained-mixture MSE only, no fitted DLinear residual supervision','training_cases':len(ts),'seed':2021,'code_revision':rev});save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated','elapsed_seconds':time.time()-start});print(name,'selected_epoch',best_epoch,{s:m['mse'] for s,m in results.items()},flush=True)
if __name__=='__main__':run(int(sys.argv[1]))
