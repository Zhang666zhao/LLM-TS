"""Matched-window random-hour tools: point MSE plus native pinball regularization."""
from common import *
from transformed_tool import transform,predict,extract_point
import torch,sys,gc,pandas as pd
from chronos import Chronos2Pipeline
from chronos.chronos2.preprocess import from_list_of_dicts

def cov(d):
 h=np.asarray(d.hour,dtype=float);return {'hour_sin':np.sin(2*np.pi*h/24).astype(np.float32),'hour_cos':np.cos(2*np.pi*h/24).astype(np.float32)}

def run(weight):
 mode="random";assert weight in (.1,1.,10.);tag={.1:"q01",1.:"q1",10.:"q10"}[weight]
 assert mode in ('semantic','pca','random');name=f'20261105_calendarmultiloss_{tag}_mse';out=ROOT/'runs'/name;assert not (out/'config.json').exists();out.mkdir(exist_ok=True);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);guard(3600);torch.set_num_threads(4);torch.manual_seed(2021);torch.cuda.manual_seed_all(2021);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;z=np.load(RUN/'cache/trainval_z.npy');dates=pd.to_datetime(json.loads((RUN/'cache/dates.json').read_text()));pipe=Chronos2Pipeline.from_pretrained(str(RUN/'models/chronos2_29ec3766'),device_map='cuda:0');net=pipe.model;levels=net.chronos_config.quantiles;cfg=dict(transform(z[:8640].astype(float),mode),engine='transformed_tool_v1',mode=mode,calendar='hour',known_future_calendar=True,training_rows=[0,8640],checkpoint_parent=name,finetune_steps=500,finetune_rate=5e-7,seed=2021,point_rule='full_integral',point_quantile_levels=levels,loss='original MSE + weight * mean pinball normalized over21 quantiles and7 target rows',pinball_weight=weight,point='21 native quantile CDF trapezoid integral, flat endpoint tails; inverse POINT transform',batch_windows=16,comparison='Same explicit window RNG/order and500 updates as no-calendar CDF-MSE;9 rows perwindow instead7; dropout realization need not match across shapes',code_revision=git_head());save_json(out/'config.json',cfg);b=torch.tensor(cfg['bias'],device='cuda:0',dtype=torch.float32);inv=torch.tensor(cfg['inverse_matrix'],device='cuda:0',dtype=torch.float32);qw=torch.tensor(extract_point(np.eye(len(levels)),levels,'full_integral'),device='cuda:0',dtype=torch.float32)
 def prepare(ts):
  x=np.stack([z[t-512:t] for t in ts]);lat=(x.astype(float)@np.asarray(cfg['matrix'])+np.asarray(cfg['bias'])).astype(np.float32);items=[{'target':v.T.copy(),'past_covariates':cov(dates[t-512:t]),'future_covariates':cov(pd.date_range(dates[t-1]+pd.Timedelta(hours=1),periods=64,freq='h'))} for t,v in zip(ts,lat)];prepared=from_list_of_dicts(items,prediction_length=64)
  assert all(a['n_targets']==7 and a['n_covariates']==2 and a['n_future_covariates']==2 and a['context'].shape==(9,512) and torch.isnan(a['future_covariates'][:7]).all() and torch.isfinite(a['future_covariates'][7:]).all() for a in prepared)
  context=torch.cat([a['context'] for a in prepared]).cuda();future=torch.cat([a['future_covariates'] for a in prepared]).cuda();groups=torch.arange(len(ts),device='cuda').repeat_interleave(9);return context,future,groups,items
 def direct(context,future,groups,target=None):
  n=len(context)//9;output=net(context=context,group_ids=groups,future_covariates=future,num_output_patches=4,future_target=target);q=output.quantile_preds[:,:,:64].reshape(n,9,len(levels),64)[:,:7];p=(q.float()*qw[None,None,:,None]).sum(2).transpose(1,2)
  with torch.autocast('cuda',enabled=False):point=(p.float()-b)@inv
  return point if target is None else (point,output.loss*9/(7*len(levels)))
 def targets(y):
  latent=(y.astype(float)@np.asarray(cfg['matrix'])+np.asarray(cfg['bias'])).astype(np.float32).transpose(0,2,1);v=np.concatenate([latent,np.full((len(y),2,64),np.nan,dtype=np.float32)],1);return torch.from_numpy(v.reshape(-1,64)).cuda()
 net.eval();xx,ff,gg,items=prepare(np.array([1024,1536]))
 with torch.no_grad():
  a=direct(xx,ff,gg);changed=xx.clone();changed[9:]+=10;bb=direct(changed,ff,gg);q,_=pipe.predict_quantiles(items,prediction_length=64,quantile_levels=levels,batch_size=112,context_length=512,cross_learning=False);qq=np.stack([v.numpy().transpose(1,0,2) for v in q]);public=((extract_point(qq,levels,'full_integral')-np.asarray(cfg['bias']))@np.asarray(cfg['inverse_matrix'])).astype(np.float32);delta=float(abs(a.cpu().numpy()-public).max());isolation=float((a[0]-bb[0]).abs().max())
 _,cy=windows(z[:8640],[1024,1536]);ct=targets(cy)
 with torch.no_grad():
  pp,ll=direct(xx,ff,gg,ct);cp=ct.clone();cp[torch.isfinite(cp)]+=10;pp2,ll2=direct(xx,ff,gg,cp);target_delta=float((pp-pp2).abs().max());assert target_delta==0 and torch.isfinite(ll) and torch.isfinite(ll2) and not torch.allclose(ll,ll2),'Trainingtargets influence onlyloss'
 assert delta<1e-5 and isolation<1e-5,('Known-covariate direct contract',delta,isolation);save_json(out/'contract.json',{'public_inverse_max_delta':delta,'other_query_perturbation_delta':isolation,'future_target_rows_all_nan':True,'only2_future_calendar_rows_known':True,'num_target_variables':7,'training_target_perturbation_prediction_delta':target_delta,'native_loss_normalization':'multiply9/(7*21): exclude2covrows, average21pinballterms'});params=list(net.parameters());opt=torch.optim.AdamW(params,lr=5e-7,weight_decay=.01);sampler=torch.Generator().manual_seed(2021);records=[];orders=[];start=time.time();net.train()
 for step in range(500):
  guard(1800);ts=torch.randint(512,8577,(16,),generator=sampler).numpy();orders.append(ts);xx,ff,gg,_=prepare(ts);_,y=windows(z[:8640],ts);yy=torch.from_numpy(y).cuda();opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):p,pinball=direct(xx,ff,gg,targets(y));pointloss=(p-yy).square().mean();loss=pointloss+weight*pinball
  assert torch.isfinite(loss),'nonfinite known-calendar MSE';loss.backward();grad=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(grad),'nonfinite gradient';opt.step()
  for group in opt.param_groups:group['lr']=5e-7*(1-(step+1)/500)
  if step%50==0 or step==499:records.append({'step':step+1,'loss':float(loss.detach()),'gradient_norm':float(grad),'point_mse':float(pointloss.detach()),'pinball':float(pinball.detach())});print(name,records[-1],flush=True)
 pipe.save_pretrained(out/'trainer/finetuned-ckpt');orders=np.stack(orders);np.save(out/'training_sample_origins.npy',orders);old=np.load(ROOT/'runs'/f'20261105_calendarcdfmse_{mode}_mse/training_sample_origins.npy');assert np.array_equal(orders,old);save_json(out/'sampler_audit.json',{'matched_no_calendar_origins_exact':True,'steps':500,'windows_per_step':16});save_json(out/'training_log.json',records);del net,pipe,opt,params;gc.collect();torch.cuda.empty_cache();results={}
 for split,ts in json.loads((RUN/'cache/window_manifest.json').read_text()).items():
  p=predict(z,ts,cfg,dates=dates);assert np.isfinite(p).all();np.save(out/f'{split}_predictions.npy',p);results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'));save_json(out/'partial_metrics.json',results)
 save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated','elapsed_seconds':time.time()-start});print(name,{s:m['mse'] for s,m in results.items()},flush=True)
if __name__=='__main__':run(float(sys.argv[1]))
