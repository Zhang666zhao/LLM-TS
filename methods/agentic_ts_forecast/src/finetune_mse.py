"""Train-only point-MSE adaptation of Chronos2, with explicit task grouping and reload."""
from common import *
import torch,sys,gc
from chronos import Chronos2Pipeline

def run(mode,rate,tag):
 name=f'20261105_mseft_{tag}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True)
 save_json(out/'config.json',{'engine':'point_mse_finetune_v1','mode':mode,'rate':rate,'steps':500,'seed':2021,'context':512,'horizon':64,'code_revision':git_head(),'status':'prepared'})
 guard(1800);torch.set_num_threads(4);torch.manual_seed(2021);torch.cuda.manual_seed_all(2021);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;z=np.load(RUN/'cache/trainval_z.npy');pipe=Chronos2Pipeline.from_pretrained(str(RUN/'models/chronos2_29ec3766'),device_map='cuda:0');net=pipe.model;levels=net.chronos_config.quantiles;qi=[min(range(len(levels)),key=lambda i:abs(levels[i]-q)) for q in np.arange(.1,1,.1)];assert all(abs(levels[i]-q)<1e-6 for i,q in zip(qi,np.arange(.1,1,.1)))
 def direct(x):
  n=len(x);context=x.transpose(1,2).reshape(n*7,512);groups=torch.arange(n,device=x.device).repeat_interleave(7);pred=net(context=context,group_ids=groups,num_output_patches=4).quantile_preds[:,qi,:64].mean(1);return pred.reshape(n,7,64).transpose(1,2)
 # Match public inference before training and verify one task cannot see another task.
 net.eval();xx=torch.tensor(np.stack([z[512:1024],z[1024:1536]]),device='cuda:0')
 with torch.no_grad():
  a=direct(xx);pert=xx.clone();pert[1]+=10;b=direct(pert);q,_=pipe.predict_quantiles([v.cpu().T for v in xx],prediction_length=64,batch_size=112,context_length=512,cross_learning=False);public=torch.stack([v.mean(-1).T.cuda() for v in q]);isolation=float((a[0]-b[0]).abs().max());agreement=float((a-public).abs().max())
 assert isolation<1e-5 and agreement<1e-5,('direct inference contract',isolation,agreement);save_json(out/'contract.json',{'other_window_perturbation_delta':isolation,'direct_public_max_delta':agreement})
 if mode=='head':
  for key,param in net.named_parameters():param.requires_grad_('output_patch_embedding' in key)
 parameters=[p for p in net.parameters() if p.requires_grad];assert parameters;opt=torch.optim.AdamW(parameters,lr=rate,weight_decay=.01);sampler=torch.Generator().manual_seed(2021);steps=500;records=[];orders=[];start=time.time();net.train()
 for step in range(steps):
  guard(900);ts=torch.randint(512,8577,(16,),generator=sampler).numpy();orders.append(ts);x,y=windows(z[:8640],ts);xx=torch.from_numpy(x).cuda();yy=torch.from_numpy(y).cuda();opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):p=direct(xx);loss=(p.float()-yy).square().mean()
  if not torch.isfinite(loss):raise RuntimeError('nonfinite point MSE')
  loss.backward();grad=torch.nn.utils.clip_grad_norm_(parameters,1.);assert torch.isfinite(grad),'Nonfinite gradient';opt.step()
  for g in opt.param_groups:g['lr']=rate*(1-(step+1)/steps)
  if step%50==0 or step==steps-1:records.append({'step':step+1,'loss':float(loss.detach()),'gradient_norm':float(grad)});print(name,records[-1],flush=True)
 checkpoint=out/'trainer/finetuned-ckpt';pipe.save_pretrained(checkpoint);np.save(out/'training_sample_origins.npy',np.stack(orders));save_json(out/'training_log.json',records);del net,pipe,opt,parameters;gc.collect();torch.cuda.empty_cache();pipe=Chronos2Pipeline.from_pretrained(str(checkpoint),device_map='cuda:0');pipe.model.eval();manifest=json.loads((RUN/'cache/window_manifest.json').read_text());results={}
 for split,ts in manifest.items():
  qs=[]
  for i in range(0,len(ts),16):
   guard(600);q,_=pipe.predict_quantiles([torch.from_numpy(z[t-512:t].T.copy()) for t in ts[i:i+16]],prediction_length=64,batch_size=112,context_length=512,cross_learning=False);qs.extend([a.cpu().numpy().transpose(1,0,2) for a in q])
  q=np.stack(qs).astype(np.float32);p=q.mean(-1);results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'));np.save(out/f'{split}_predictions.npy',p);np.save(out/f'{split}_quantiles.npy',q);save_json(out/'partial_metrics.json',results)
 cfg={'engine':'point_mse_finetune_v1','mode':mode,'rate':rate,'steps':steps,'point_loss':'MSE of average nine quantiles against all7 targets, no future covariates','training_rows':[0,8640],'context':512,'horizon':64,'batch_windows':16,'seed':2021,'sampler':'independent CPU generator, origins uniform512..8576','checkpoint_selection':'fixed final500, no validation training','inference':'saved checkpoint reloaded eval FP32','code_revision':git_head()};save_json(out/'config.json',cfg);save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated','elapsed_seconds':time.time()-start});print(name,{s:m['mse'] for s,m in results.items()},flush=True)
if __name__=='__main__':run(sys.argv[1],float(sys.argv[2]),sys.argv[3])
