"""Small train-only point calibration from prefix-held-out tool quantiles."""
from common import *
import torch,sys
from torch import nn
PARENT='20261105_reloadedft_lr5e7_mse'
def apply(q,cfg):
 a=np.asarray(cfg['a']);b=np.asarray(cfg['b']);return (q[:,:,:,4]+a*(q.mean(-1)-q[:,:,:,4])+b*(q[:,:,:,8]-q[:,:,:,0])).astype(np.float32)
class Calibrator(nn.Module):
 def __init__(self,kind):
  super().__init__();n=1 if kind=='global' else 7;self.u=nn.Parameter(torch.zeros(n));self.v=nn.Parameter(torch.zeros(n))
 def coeff(self):return 2*self.u.sigmoid(),.1*self.v.tanh()
 def forward(self,q):
  a,b=self.coeff();return q[:,:,:,4]+a*(q.mean(-1)-q[:,:,:,4])+b*(q[:,:,:,8]-q[:,:,:,0])
def run(kind,reg,tag):
 name=f'20261105_quantcal_{tag}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);guard(1200);torch.set_num_threads(4);qs=[];ys=[];origins=[]
 for cut in (3072,4096,6272):
  d=ROOT/'runs'/f'20261105_crossfit_prefix{cut}_mse';q=np.load(d/'ft_quantiles.npy');y=np.load(d/'targets.npy');ts=np.load(d/'query_origins.npy');assert min(ts)-512>=cut;qs.append(q);ys.append(y);origins.extend(ts.tolist())
 q=torch.tensor(np.concatenate(qs),device='cuda:0');y=torch.tensor(np.concatenate(ys),device='cuda:0');assert len(q)==63 and max(np.array(origins[:34])+64)<=min(origins[34:])-512
 def init():return Calibrator(kind).cuda(0)
 def score(net,sl):
  with torch.no_grad():return float((net(q[sl])-y[sl]).double().square().mean())
 def update(net,opt,sl):
  opt.zero_grad(set_to_none=True);a,b=net.coeff();loss=(net(q[sl])-y[sl]).square().mean()+reg*((a-1).square().mean()+100*b.square().mean());assert torch.isfinite(loss);loss.backward();opt.step()
 net=init();opt=torch.optim.Adam(net.parameters(),lr=.03);best=score(net,slice(34,None));best_step=0;stale=0;records=[{'step':0,'training_confirm_mse':best}]
 for step in range(1,501):
  guard(900);update(net,opt,slice(0,34))
  if step%10==0:
   sc=score(net,slice(34,None));records.append({'step':step,'training_confirm_mse':sc})
   if sc<best:best=sc;best_step=step;stale=0
   else:stale+=1
   if stale>=10:break
 save_json(out/'development_steps.json',records);net=init();opt=torch.optim.Adam(net.parameters(),lr=.03)
 for step in range(best_step):guard(600);update(net,opt,slice(None))
 a,b=net.coeff();cfg={'engine':'quantile_calibration_v1','parent':PARENT,'source_ids':['run:'+PARENT],'kind':kind,'reg':reg,'a':a.detach().cpu().tolist(),'b':b.detach().cpu().tolist(),'steps':best_step,'training_confirm_mse':best,'training_cases':63,'selection':'first34fit,last29training-confirm; refit63forselectedsteps','parameter_bounds':{'a':[0,2],'b':[-.1,.1]},'identity':'a1,b0 equals nine-quantile average','proxy_transfer_caveat':True,'code_revision':git_head()};save_json(out/'config.json',cfg);results={}
 # Independently check the JSON-only runtime formula agrees with training forward.
 with torch.no_grad():expected=net(q).cpu().numpy()
 actual=apply(q.cpu().numpy(),cfg);assert np.max(abs(actual-expected))<1e-5
 for split in ('dev','confirm','val'):
  fq=np.load(ROOT/'runs'/PARENT/f'{split}_quantiles.npy');p=apply(fq,cfg);results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'));np.save(out/f'{split}_predictions.npy',p)
 save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated'});print(name,'selected',best_step,'a',cfg['a'],'b',cfg['b'],{s:m['mse'] for s,m in results.items()},flush=True)
if __name__=='__main__':run(sys.argv[1],float(sys.argv[2]),sys.argv[3])
