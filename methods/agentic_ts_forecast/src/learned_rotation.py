"""Train21 Cayley rotation parameters only; frozen Chronos checkpoint, train-only MSE."""
from common import *
from transformed_tool import extract_point,predict
from transformed_calendar import cov
import torch,pandas as pd,gc,sys
from chronos import Chronos2Pipeline

def run(lr):
 assert lr in (.0003,.001,.003);tag={.0003:'r0003',.001:'r001',.003:'r003'}[lr];name=f'20261105_learnedrotation_{tag}_masked_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=False);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);guard(3600);torch.set_num_threads(4);torch.manual_seed(2021);torch.cuda.manual_seed_all(2021);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 parent='20261105_transformedcalft_random_mse';base=json.loads((ROOT/'runs'/parent/'config.json').read_text());z=np.load(RUN/'cache/trainval_z.npy');dates=pd.to_datetime(json.loads((RUN/'cache/dates.json').read_text()));pipe=Chronos2Pipeline.from_pretrained(str(ROOT/'runs'/parent/'trainer/finetuned-ckpt'),device_map='cuda:0');net=pipe.model;net.eval()
 for p in net.parameters():p.requires_grad_(False)
 def tensor(a):return torch.tensor(np.asarray(a),device='cuda:0',dtype=torch.float32)
 bw=tensor(base['matrix']);bb=tensor(base['bias']);bi=tensor(base['inverse_matrix']);levels=base['point_quantile_levels'];qw=tensor(extract_point(np.eye(len(levels)),levels,'full_integral'));angles=torch.nn.Parameter(torch.zeros(21,device='cuda:0'));ii,jj=torch.triu_indices(7,7,offset=1,device='cuda:0');eye=torch.eye(7,device='cuda:0')
 def matrices():
  upper=torch.zeros((7,7),device='cuda:0').index_put((ii,jj),angles);skew=upper-upper.T;o=torch.linalg.solve(eye+skew,eye-skew);return bw@o,bb@o,o.T@bi,o
 def prepare(ts):
  x=tensor(np.stack([z[t-512:t] for t in ts]));pc=[];fc=[]
  for t in ts:
   pc.append(np.stack(list(cov(dates[t-512:t]).values())));fc.append(np.stack(list(cov(pd.date_range(dates[t-1]+pd.Timedelta(hours=1),periods=64,freq='h')).values())))
  return x,tensor(pc),tensor(fc),torch.arange(len(ts),device='cuda:0').repeat_interleave(9)
 def direct(data):
  x,pc,fc,groups=data;w,b,inv,o=matrices();lat=(x@w+b).transpose(1,2);context=torch.cat([lat,pc],1).reshape(-1,512);future=torch.cat([torch.zeros((len(x),7,64),device='cuda:0'),fc],1).reshape(-1,64);mask=torch.cat([torch.zeros((len(x),7,64),device='cuda:0'),torch.ones_like(fc)],1).reshape(-1,64);q=net(context=context,group_ids=groups,future_covariates=future,future_covariates_mask=mask,num_output_patches=4).quantile_preds[:,:,:64].reshape(len(x),9,len(levels),64)[:,:7];point=(q.float()*qw[None,None,:,None]).sum(2).transpose(1,2);return (point-b)@inv
 data=prepare([1024,1536]);initial=direct(data);public=predict(z,np.array([1024,1536]),base,dates=dates);delta=float(np.max(np.abs(initial.detach().cpu().numpy()-public)));assert delta<1e-5,('initial public match',delta);changed=[v.clone() for v in data];changed[0][1]+=10;assert torch.equal(initial[0],direct(changed)[0]);_,yy=windows(z[:8640],[1024,1536]);loss=(initial-tensor(yy)).square().mean();loss.backward();assert angles.grad is not None and torch.isfinite(angles.grad).all() and angles.grad.abs().sum()>0;assert all(p.grad is None for p in net.parameters());grad=float(angles.grad.norm());angles.grad=None
 save_json(out/'contracts.json',dict(initial_public_max_delta=delta,query_isolation_exact=True,nonzero_finite_adapter_gradient=grad,frozen_backbone_no_grad=True,trainable_parameters=21,future_target_rows_masked=True,unknown_future_placeholder='zero with explicit all-zero target mask; no labels in inputs'));opt=torch.optim.Adam([angles],lr=lr);rng=torch.Generator().manual_seed(2021);records=[];orders=[];start=time.time()
 for step in range(500):
  guard(1800);ts=torch.randint(512,8577,(16,),generator=rng).numpy();orders.append(ts);data=prepare(ts);_,yy=windows(z[:8640],ts);opt.zero_grad(set_to_none=True);pp=direct(data);mse=(pp-tensor(yy)).square().mean();loss=mse+.001*angles.square().mean();assert torch.isfinite(loss);loss.backward();g=torch.nn.utils.clip_grad_norm_([angles],1.);assert torch.isfinite(g);opt.step();opt.param_groups[0]['lr']=lr*(1-(step+1)/500)
  if step%50==0 or step==499:records.append(dict(step=step+1,mse=float(mse.detach()),gradient_norm=float(g),angle_norm=float(angles.detach().norm())));print(name,records[-1],flush=True)
 w,b,inv,o=matrices();orth=float((o.T@o-eye).abs().max());assert orth<1e-5;cfg=dict(base,matrix=w.detach().cpu().numpy().astype(float).tolist(),bias=b.detach().cpu().numpy().astype(float).tolist(),inverse_matrix=inv.detach().cpu().numpy().astype(float).tolist(),source_ids=['run:'+parent],adaptation='21 Cayley angles, backbone eval and all weights frozen',adapter_steps=500,adapter_lr=lr,adapter_penalty=.001,training_rows=[0,8640],orthogonal_max_delta=orth,code_revision=git_head());roundtrip=((z[:1024].astype(float)@np.asarray(cfg['matrix'])+np.asarray(cfg['bias']))-np.asarray(cfg['bias']))@np.asarray(cfg['inverse_matrix']);cfg['roundtrip_max_delta']=float(abs(roundtrip-z[:1024]).max());assert cfg['roundtrip_max_delta']<1e-4
 post=direct(prepare([1024,1536])).detach().cpu().numpy();public_after=predict(z,np.array([1024,1536]),cfg,dates=dates);postdelta=float(abs(post-public_after).max());assert postdelta<1e-5,('learned public match',postdelta);save_json(out/'postfit_contract.json',dict(public_max_delta=postdelta,frozen_backbone_no_grad=all(p.grad is None for p in net.parameters())));save_json(out/'config.json',cfg);torch.save({'angles':angles.detach().cpu()},out/'best.pth');np.save(out/'training_sample_origins.npy',np.stack(orders));assert np.array_equal(np.stack(orders),np.load(ROOT/'runs/20261105_calendarcdfmse_random_mse/training_sample_origins.npy'));save_json(out/'training_log.json',records);del net,pipe,opt,initial,loss,pp,mse,data,w,b,inv,o;gc.collect();torch.cuda.empty_cache();results={}
 for split,ts in json.loads((RUN/'cache/window_manifest.json').read_text()).items():
  pp=predict(z,ts,cfg,dates=dates);np.save(out/f'{split}_predictions.npy',pp);results[split]=metric(pp,np.load(RUN/'cache'/f'{split}_targets.npy'))
 save_json(out/'metrics.json',dict(candidate=name,metrics=results,status='evaluated',elapsed_seconds=time.time()-start));print(name,{s:m['mse'] for s,m in results.items()},flush=True)
if __name__=='__main__':run(float(sys.argv[1]))
