"""No-update forward equivalence and real-data gradient contracts for safe statistics."""
from common import *
import torch,contextlib,inspect
from timesfm3 import TimesFM3Evaluator,ModelConfig
from timesfm3.torch import util
from timesfm_safe_stats import install,restore,update_running_stats
out=RUN/'timesfm_safe_stats_audit';out.mkdir(exist_ok=True);guard(2400);torch.set_num_threads(4);torch.manual_seed(2021);torch.cuda.manual_seed_all(2021);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;z=np.load(RUN/'cache/trainval_z.npy')[:8640];gen=torch.Generator().manual_seed(2021);batches=[]
for j in range(3):
 ts=torch.randint(512,8577,(8,),generator=gen).numpy();x,y=windows(z,ts);batches.append((f'real{j}',torch.from_numpy(x).cuda(),torch.from_numpy(y).cuda(),ts.tolist()))
for value in (0.,1.):batches.append((f'constant{value}',torch.full((2,512,7),value,device='cuda:0'),torch.full((2,64,7),value,device='cuda:0'),[]))
pipe=TimesFM3Evaluator(ModelConfig(checkpoint_path=str(RUN/'models/timesfm3_43046b85'),per_core_batch_size=16,device='cuda:0',local_files_only=True));net=pipe.model;raw=net.decode.__wrapped__;original=util.update_running_stats;results=[]
def direct(x):
 n=len(x);q=raw(net,target=torch.cat([x.transpose(1,2),-x.transpose(1,2)],0),horizon=64);return ((q[:n].mean(-1)-q[n:].mean(-1))*.5).transpose(1,2)
# Fully masked and zero-variance statistics must preserve original forward values.
n=torch.zeros((2,7),device='cuda:0');mu=n.clone();sigma=n.clone();x=torch.ones((2,7,32),device='cuda:0');mask=torch.ones_like(x,dtype=torch.bool);a=original(n,mu,sigma,x,mask);b=update_running_stats(n,mu,sigma,x,mask);assert all(torch.equal(v,w) for v,w in zip(a,b))
for mode in ('fp32','bf16'):
 for label,x,y,origins in batches:
  guard(1800);net.zero_grad(set_to_none=True);net.eval();restore(original)
  with torch.no_grad(),(torch.autocast('cuda',dtype=torch.bfloat16) if mode=='bf16' else contextlib.nullcontext()):reference=direct(x)
  install();net.train()
  with (torch.autocast('cuda',dtype=torch.bfloat16) if mode=='bf16' else contextlib.nullcontext()):p=direct(x);loss=(p.float()-y).square().mean()
  delta=float(abs(p.detach()-reference).max());assert delta<1e-5 and torch.isfinite(loss),('forward contract',mode,label,delta);loss.backward();bad=[k for k,v in net.named_parameters() if v.grad is not None and not torch.isfinite(v.grad).all()];assert not bad,('nonfinite patched gradient',mode,label,bad[:3]);norm=float(torch.sqrt(sum(v.grad.double().square().sum() for v in net.parameters() if v.grad is not None)));assert np.isfinite(norm);row={'precision':mode,'batch':label,'origins':origins,'forward_max_delta':delta,'finite_backward':True,'grad_norm_float64':norm,'loss':float(loss.detach())};results.append(row);save_json(out/'checks.json',results);print(row,flush=True)
restore(original);result={'optimizer_steps':0,'all_passed':True,'cases':len(results),'forward_max_delta':max(r['forward_max_delta'] for r in results),'checks':results,'installed_package_modified':False,'update_source_sha256':hashlib.sha256(pathlib.Path(inspect.getsourcefile(util)).read_bytes()).hexdigest(),'code_revision':git_head()};save_json(RUN/'timesfm_safe_stats_contract.json',result);print('SAFE_STATS_CONTRACT_PASS',result['cases'],result['forward_max_delta'],flush=True)
