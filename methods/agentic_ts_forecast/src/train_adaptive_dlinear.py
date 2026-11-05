"""New numerical tool candidates; unchanged official split/scaler/metric."""
from common import *
import torch,sys,importlib.util
from types import SimpleNamespace
from torch import nn
class Adaptive(nn.Module):
 def __init__(self,mode):
  super().__init__();self.mode=mode;source=ROOT.parent/'TSRAG/third_party/LTSF-Linear/models/DLinear.py';spec=importlib.util.spec_from_file_location('new_dlinear',source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);self.net=mod.Model(SimpleNamespace(seq_len=512,pred_len=64,individual=False,enc_in=7))
 def forward(self,x):
  center=x[:,-1:] if self.mode=='last' else x.mean(1,keepdim=True)
  scale=torch.sqrt(x.var(1,keepdim=True,unbiased=False)+1e-5) if self.mode=='meanstd' else 1.
  return self.net((x-center)/scale)*scale+center

def run(mode):
 name=f'20261105_dlinear_{mode}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True)
 subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True)
 guard(600);torch.set_num_threads(4);torch.manual_seed(2021);torch.cuda.manual_seed_all(2021);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 cfg={'mode':mode,'seed_cpu':2021,'seed_cuda':2021,'batch':32,'lr':.005,'max_epochs':10,'patience':3,'scheduler':'lr half after each epoch','training_stride':1,'checkpoint_selection':'full validation macro MSE','code_revision':git_head()};save_json(out/'config.json',cfg)
 z=np.load(RUN/'cache/trainval_z.npy');x,y=windows(z,np.arange(512,8577));x=torch.from_numpy(x).cuda(0);y=torch.from_numpy(y).cuda(0);manifest=json.loads((RUN/'cache/window_manifest.json').read_text());vx,vy=windows(z,manifest['val']);vx=torch.from_numpy(vx).cuda(0);vy=torch.from_numpy(vy).cuda(0)
 net=Adaptive(mode).cuda(0);opt=torch.optim.Adam(net.parameters(),lr=.005);best=float('inf');stale=0;records=[];start=time.time()
 for epoch in range(10):
  guard(600);net.train();order=torch.randperm(len(x),device='cpu');total=0.
  for ids in order.split(32):
   guard(600);ids=ids.cuda(0);opt.zero_grad(set_to_none=True);p=net(x[ids]);loss=(p-y[ids]).square().mean()
   if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
   loss.backward();opt.step();total+=loss.item()*len(ids)
  net.eval();sse=0.;count=0
  with torch.no_grad():
   for i in range(0,len(vx),128):
    p=net(vx[i:i+128]);err=(p-vy[i:i+128]).double();sse+=err.square().sum().item();count+=err.numel()
  score=sse/count;record={'epoch':epoch+1,'train_mse':total/len(x),'validation_mse':score,'lr':opt.param_groups[0]['lr']};records.append(record);save_json(out/'epochs.json',records);print(name,record,flush=True)
  if score<best:best=score;stale=0;torch.save(net.state_dict(),out/'best.pth')
  else:stale+=1
  for g in opt.param_groups:g['lr']=.005*(.5**(epoch+1))
  if stale>=3:break
 net.load_state_dict(torch.load(out/'best.pth',map_location='cuda:0'));net.eval();results={}
 with torch.no_grad():
  for split,ts in manifest.items():
   bx,by=windows(z,ts);ps=[]
   for i in range(0,len(bx),128):guard(600);ps.append(net(torch.from_numpy(bx[i:i+128]).cuda(0)).cpu().numpy())
   p=np.concatenate(ps);results[split]=metric(p,by);np.save(out/f'{split}_predictions.npy',p)
 save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated','code_revision':cfg['code_revision'],'elapsed_seconds':time.time()-start})
 with (RUN/'experiment_index.jsonl').open('a') as f:f.write(json.dumps({'candidate':name,'status':'evaluated','val_mse':results['val']['mse'],'kind':'trained_numeric_tool'})+'\n')
 print(name,'best',results['val']['mse'],flush=True)
if __name__=='__main__':run(sys.argv[1])
