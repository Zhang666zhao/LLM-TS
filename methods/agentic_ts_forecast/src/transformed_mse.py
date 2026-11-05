"""Train invertible views using point MSE in the ORIGINAL standardized metric."""
from common import *
from transformed_tool import transform,predict,extract_point
import torch,sys,gc
from chronos import Chronos2Pipeline

def run(mode,integral=False):
 name=f'20261105_transformed{"cdfmse" if integral else "mse"}_{mode}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);guard(1800)
 torch.set_num_threads(4);torch.manual_seed(2021);torch.cuda.manual_seed_all(2021);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;z=np.load(RUN/'cache/trainval_z.npy');cfg=dict(transform(z[:8640].astype(float),mode),engine='transformed_tool_v1',mode=mode,training_rows=[0,8640],checkpoint_parent=name,finetune_steps=500,finetune_rate=5e-7,seed=2021,loss='all7 original standardized point MSE after inverse transform',point='average9 latent quantiles, inverse point transform',code_revision=git_head());save_json(out/'config.json',cfg)
 pipe=Chronos2Pipeline.from_pretrained(str(RUN/'models/chronos2_29ec3766'),device_map='cuda:0');net=pipe.model;levels=net.chronos_config.quantiles;qi=[min(range(len(levels)),key=lambda i:abs(levels[i]-q)) for q in np.arange(.1,1,.1)];assert all(abs(levels[i]-q)<1e-6 for i,q in zip(qi,np.arange(.1,1,.1)))
 if integral:
  cfg.update(point_rule='full_integral',point_quantile_levels=levels,point='21 native quantile CDF trapezoid integral, flat endpoint tails; inverse point transform');save_json(out/'config.json',cfg)
  integral_weights=extract_point(np.eye(len(levels)),levels,'full_integral');assert abs(integral_weights.sum()-1)<1e-12 and (integral_weights>=0).all();qw=torch.tensor(integral_weights,device='cuda:0',dtype=torch.float32)
 w=torch.tensor(cfg['matrix'],device='cuda:0',dtype=torch.float32);b=torch.tensor(cfg['bias'],device='cuda:0',dtype=torch.float32);inv=torch.tensor(cfg['inverse_matrix'],device='cuda:0',dtype=torch.float32)
 def direct(latent):
  n=len(latent);context=latent.transpose(1,2).reshape(n*7,512);groups=torch.arange(n,device=latent.device).repeat_interleave(7);q=net(context=context,group_ids=groups,num_output_patches=4).quantile_preds[:,:,:64]
  p=((q.float()*qw[None,:,None]).sum(1) if integral else q[:,qi].mean(1)).reshape(n,7,64).transpose(1,2)
  # Explicit FP32 inverse preserves the declared metric even under autocast.
  with torch.autocast('cuda',enabled=False):return (p.float()-b)@inv
 net.eval();raw=np.stack([z[512:1024],z[1024:1536]]);lat=(raw.astype(float)@np.asarray(cfg['matrix'])+np.asarray(cfg['bias'])).astype(np.float32);xx=torch.from_numpy(lat).cuda()
 with torch.no_grad():
  a=direct(xx);pert=xx.clone();pert[1]+=10;bb=direct(pert);extra={'quantile_levels':levels} if integral else {};q,_=pipe.predict_quantiles([v.cpu().T for v in xx],prediction_length=64,batch_size=112,context_length=512,cross_learning=False,**extra);qq=np.stack([v.numpy().transpose(1,0,2) for v in q]);point=extract_point(qq,levels,'full_integral') if integral else qq.mean(-1);public=((point-np.asarray(cfg['bias']))@np.asarray(cfg['inverse_matrix'])).astype(np.float32);agreement=float(abs(a.cpu().numpy()-public).max());isolation=float((a[0]-bb[0]).abs().max())
 assert agreement<1e-5 and isolation<1e-5,('direct inverse inference',agreement,isolation);save_json(out/'contract.json',{'public_inverse_max_delta':agreement,'other_window_perturbation_delta':isolation,'train_transform_roundtrip':cfg['roundtrip_max_delta']})
 params=list(net.parameters());opt=torch.optim.AdamW(params,lr=5e-7,weight_decay=.01);sampler=torch.Generator().manual_seed(2021);records=[];orders=[];start=time.time();net.train()
 for step in range(500):
  guard(1200);ts=torch.randint(512,8577,(16,),generator=sampler).numpy();orders.append(ts);x,y=windows(z[:8640],ts);latent=(x.astype(float)@np.asarray(cfg['matrix'])+np.asarray(cfg['bias'])).astype(np.float32);xx=torch.from_numpy(latent).cuda();yy=torch.from_numpy(y).cuda();opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):p=direct(xx);loss=(p-yy).square().mean()
  if not torch.isfinite(loss):raise RuntimeError('nonfinite original-space point loss')
  loss.backward();grad=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(grad),'nonfinite gradient';opt.step()
  for group in opt.param_groups:group['lr']=5e-7*(1-(step+1)/500)
  if step%50==0 or step==499:records.append({'step':step+1,'loss':float(loss.detach()),'gradient_norm':float(grad)});print(name,records[-1],flush=True)
 pipe.save_pretrained(out/'trainer/finetuned-ckpt');np.save(out/'training_sample_origins.npy',np.stack(orders));save_json(out/'training_log.json',records);del net,pipe,opt,params;gc.collect();torch.cuda.empty_cache();manifest=json.loads((RUN/'cache/window_manifest.json').read_text());results={}
 for split,ts in manifest.items():
  p=predict(z,ts,cfg);results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'));np.save(out/f'{split}_predictions.npy',p);save_json(out/'partial_metrics.json',results)
 cfg.update(batch_windows=16,optimizer='AdamW wd.01, global clip1, linear lr decay500',sampler='independent seed2021 generator uniform512..8576',postfit_inference='save then independently reload eval FP32',comparison_caveat='native quantile fit differs in sampler and optimizer as well as objective; matched raw-point-MSE trainer is the closer control');save_json(out/'config.json',cfg);save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated','elapsed_seconds':time.time()-start});print(name,{s:m['mse'] for s,m in results.items()},flush=True)
if __name__=='__main__':run(sys.argv[1],bool(int(sys.argv[2])) if len(sys.argv)>2 else False)
