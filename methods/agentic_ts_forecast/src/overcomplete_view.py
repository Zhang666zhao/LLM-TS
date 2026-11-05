"""Matched-window full-CDF point MSE with known-hour covariates in invertible views."""
from common import *
from transformed_tool import transform,predict,extract_point
import torch,sys,gc,pandas as pd
from chronos import Chronos2Pipeline
from chronos.chronos2.preprocess import from_list_of_dicts

from calendar_features import cov as expanded_cov

def overcomplete(train,mode):
 base=np.eye(7) if mode=='duplicate' else np.asarray(transform(train,mode)['matrix'])
 a=np.concatenate([np.eye(7),base],1);v=train@a;mu=v.mean(0);sd=np.maximum(v.std(0),1e-4);w=a/sd;b=-mu/sd;inv=np.linalg.pinv(w);delta=float(abs(((train[:1024]@w+b)-b)@inv-train[:1024]).max());assert delta<1e-8 and np.linalg.matrix_rank(w)==7
 return dict(matrix=w.tolist(),bias=b.tolist(),inverse_matrix=inv.tolist(),roundtrip_max_delta=delta,condition_number=float(np.linalg.cond(w)),latent_targets=14,semantic_claim='7 original channels plus7 linear views; no physical balance identity imposed; duplicate view matches target count')

def run(mode):
 calendar_mode='hour';num_covariates=2;num_rows=16
 def cov(d):return expanded_cov(d,calendar_mode)
 assert mode in ('semantic','random','duplicate');name=f'20261105_overcomplete_{mode}_mse';out=ROOT/'runs'/name;assert not (out/'config.json').exists();out.mkdir(exist_ok=True);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);guard(3600);torch.set_num_threads(4);torch.manual_seed(2021);torch.cuda.manual_seed_all(2021);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;z=np.load(RUN/'cache/trainval_z.npy');dates=pd.to_datetime(json.loads((RUN/'cache/dates.json').read_text()));pipe=Chronos2Pipeline.from_pretrained(str(RUN/'models/chronos2_29ec3766'),device_map='cuda:0');net=pipe.model;levels=net.chronos_config.quantiles;cfg=dict(overcomplete(z[:8640].astype(float),mode),engine='transformed_tool_v1',mode=mode,calendar=calendar_mode,known_future_calendar=True,training_rows=[0,8640],checkpoint_parent=name,finetune_steps=500,finetune_rate=5e-7,seed=2021,point_rule='full_integral',point_quantile_levels=levels,loss='all7 original standardized full-CDF point MSE after14-to7 pseudoinverse reconstruction; clock rows not target outputs',point='21 native quantile CDF trapezoid integral, flat endpoint tails; inverse POINT transform',batch_windows=16,comparison='Same500x16 trainingorigins as7-target hourCDFMSE;14 latent targets+2clockrows; duplicate/semantic/random share input shape; no future measurements; report increased target compute budget',code_revision=git_head());save_json(out/'config.json',cfg);b=torch.tensor(cfg['bias'],device='cuda:0',dtype=torch.float32);inv=torch.tensor(cfg['inverse_matrix'],device='cuda:0',dtype=torch.float32);qw=torch.tensor(extract_point(np.eye(len(levels)),levels,'full_integral'),device='cuda:0',dtype=torch.float32)
 def prepare(ts):
  x=np.stack([z[t-512:t] for t in ts]);lat=(x.astype(float)@np.asarray(cfg['matrix'])+np.asarray(cfg['bias'])).astype(np.float32);items=[{'target':v.T.copy(),'past_covariates':cov(dates[t-512:t]),'future_covariates':cov(pd.date_range(dates[t-1]+pd.Timedelta(hours=1),periods=64,freq='h'))} for t,v in zip(ts,lat)];prepared=from_list_of_dicts(items,prediction_length=64)
  assert all(a['n_targets']==14 and a['n_covariates']==num_covariates and a['n_future_covariates']==num_covariates and a['context'].shape==(num_rows,512) and torch.isnan(a['future_covariates'][:14]).all() and torch.isfinite(a['future_covariates'][14:]).all() for a in prepared)
  context=torch.cat([a['context'] for a in prepared]).cuda();future=torch.cat([a['future_covariates'] for a in prepared]).cuda();groups=torch.arange(len(ts),device='cuda').repeat_interleave(num_rows);return context,future,groups,items
 def direct(context,future,groups):
  n=len(context)//num_rows;q=net(context=context,group_ids=groups,future_covariates=future,num_output_patches=4).quantile_preds[:,:,:64].reshape(n,num_rows,len(levels),64)[:,:14];p=(q.float()*qw[None,None,:,None]).sum(2).transpose(1,2)
  with torch.autocast('cuda',enabled=False):return (p.float()-b)@inv
 net.eval();xx,ff,gg,items=prepare(np.array([1024,1536]))
 with torch.no_grad():
  a=direct(xx,ff,gg);changed=xx.clone();changed[num_rows:]+=10;bb=direct(changed,ff,gg);q,_=pipe.predict_quantiles(items,prediction_length=64,quantile_levels=levels,batch_size=112,context_length=512,cross_learning=False);qq=np.stack([v.numpy().transpose(1,0,2) for v in q]);public=((extract_point(qq,levels,'full_integral')-np.asarray(cfg['bias']))@np.asarray(cfg['inverse_matrix'])).astype(np.float32);delta=float(abs(a.cpu().numpy()-public).max());isolation=float((a[0]-bb[0]).abs().max())
 assert delta<1e-5 and isolation<1e-5,('Known-covariate direct contract',delta,isolation);save_json(out/'contract.json',{'public_inverse_max_delta':delta,'other_query_perturbation_delta':isolation,'future_target_rows_all_nan':True,'only_future_calendar_rows_known':num_covariates,'num_target_variables':7,'latent_target_rows':14});params=list(net.parameters());opt=torch.optim.AdamW(params,lr=5e-7,weight_decay=.01);sampler=torch.Generator().manual_seed(2021);records=[];orders=[];start=time.time();net.train()
 for step in range(500):
  guard(1800);ts=torch.randint(512,8577,(16,),generator=sampler).numpy();orders.append(ts);xx,ff,gg,_=prepare(ts);_,y=windows(z[:8640],ts);yy=torch.from_numpy(y).cuda();opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):p=direct(xx,ff,gg);loss=(p-yy).square().mean()
  assert torch.isfinite(loss),'nonfinite known-calendar MSE';loss.backward();grad=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(grad),'nonfinite gradient';opt.step()
  for group in opt.param_groups:group['lr']=5e-7*(1-(step+1)/500)
  if step%50==0 or step==499:records.append({'step':step+1,'loss':float(loss.detach()),'gradient_norm':float(grad)});print(name,records[-1],flush=True)
 pipe.save_pretrained(out/'trainer/finetuned-ckpt');orders=np.stack(orders);np.save(out/'training_sample_origins.npy',orders);old=np.load(ROOT/'runs'/'20261105_calendarcdfmse_random_mse/training_sample_origins.npy');assert np.array_equal(orders,old);save_json(out/'sampler_audit.json',{'matched_hour_only_origins_exact':True,'steps':500,'windows_per_step':16});save_json(out/'training_log.json',records);del net,pipe,opt,params;gc.collect();torch.cuda.empty_cache();results={}
 for split,ts in json.loads((RUN/'cache/window_manifest.json').read_text()).items():
  p=predict(z,ts,cfg,dates=dates);assert np.isfinite(p).all();np.save(out/f'{split}_predictions.npy',p);results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'));save_json(out/'partial_metrics.json',results)
 save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated','elapsed_seconds':time.time()-start});print(name,{s:m['mse'] for s,m in results.items()},flush=True)
if __name__=='__main__':run(sys.argv[1])
