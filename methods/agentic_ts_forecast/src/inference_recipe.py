"""Replay numeric recipes on history-only inputs; no scoring or target construction."""
from common import *
import torch,pandas as pd,gc

class Replay:
 def __init__(self,z,dates,origins,out):
  self.z=z;self.dates=pd.to_datetime(dates);self.ts=np.asarray(origins);self.out=pathlib.Path(out);self.out.mkdir(exist_ok=True,parents=True);self.mem={};torch.set_num_threads(4)
  manifest={'origins':self.ts.tolist(),'context':512,'horizon':64,'rows_available':len(z)};mp=self.out/'input_manifest.json'
  if mp.exists():assert json.loads(mp.read_text())==manifest,'Replay cache input mismatch'
  else:save_json(mp,manifest)
 def inputs(self,start,n):return np.stack([self.z[t-512:t] for t in self.ts[start:start+n]])
 def source(self,key):
  if key in self.mem:return self.mem[key]
  path=self.out/(key.replace(':','__')+'.npy');started=time.time();cached=path.exists()
  if cached:a=np.load(path)
  else:
   kind,name=key.split(':',1)
   if kind=='cache':a=self.cache(name)
   elif kind=='run':a=self.run(name)
   else:raise ValueError(key)
   assert a.shape==(len(self.ts),64,7) and np.isfinite(a).all();np.save(path,a)
  with (self.out/'execution_trace.jsonl').open('a') as f:f.write(json.dumps(dict(source=key,cache_hit=cached,elapsed_inclusive_seconds=time.time()-started,completed_epoch=time.time()))+'\n')
  self.mem[key]=a;return a
 def quantiles(self,kind,model=None,calendar=None,univariate=False,levels=None):
  tag='q_'+kind+'_'+str(model or 'base')+'_'+str(calendar)+'_'+str(univariate)
  if levels is not None:tag+='_'+hashlib.sha256(json.dumps(levels).encode()).hexdigest()[:12]
  path=self.out/(tag+'.npy')
  if path.exists():return np.load(path)
  ps=[]
  if kind=='chronos':
   from chronos import Chronos2Pipeline
   loc=ROOT/'runs'/model/'trainer/finetuned-ckpt' if model else RUN/'models/chronos2_29ec3766';pipe=Chronos2Pipeline.from_pretrained(str(loc),device_map='cuda:0');pipe.model.eval()
   def cov(d):
    h=np.asarray(d.hour,dtype=float);w=np.asarray(d.dayofweek,dtype=float);a={'hour_sin':np.sin(2*np.pi*h/24).astype(np.float32),'hour_cos':np.cos(2*np.pi*h/24).astype(np.float32)}
    if calendar in ('hourweek','pastonly'):a.update(week_sin=np.sin(2*np.pi*(w+h/24)/7).astype(np.float32),week_cos=np.cos(2*np.pi*(w+h/24)/7).astype(np.float32))
    return a
   batch=8 if calendar else 16
   for i in range(0,len(self.ts),batch):
    guard(180);x=self.inputs(i,batch);items=[]
    for j,a in enumerate(x):
     t=self.ts[i+j]
     if calendar:
      item={'target':a.T.copy(),'past_covariates':cov(self.dates[t-512:t])}
      if calendar!='pastonly':item['future_covariates']=cov(pd.date_range(self.dates[t-1]+pd.Timedelta(hours=1),periods=64,freq='h'))
      items.append(item)
     elif univariate:items.extend([torch.from_numpy(a[:,c][None].copy()) for c in range(7)])
     else:items.append(torch.from_numpy(a.T.copy()))
    extra={'quantile_levels':levels} if levels is not None else {}
    q,_=pipe.predict_quantiles(items,prediction_length=64,batch_size=112,context_length=512,cross_learning=False,**extra)
    arr=[v.cpu().numpy().transpose(1,0,2) for v in q]
    if univariate:arr=[np.concatenate(arr[j:j+7],axis=1) for j in range(0,len(arr),7)]
    ps.extend(arr)
  else:
   from timesfm3 import TimesFM3Evaluator,ModelConfig
   loc=ROOT/'runs'/model/'trainer/finetuned-ckpt' if model else RUN/'models/timesfm3_43046b85';pipe=TimesFM3Evaluator(ModelConfig(checkpoint_path=str(loc),per_core_batch_size=16,device='cuda:0',local_files_only=True))
   for i in range(0,len(self.ts),16):
    guard(180);x=self.inputs(i,16);items=[a.T.copy() for a in x] if not univariate else [a[:,c][None].copy() for a in x for c in range(7)]
    ans=list(pipe.predict_batch(items,horizon=64,return_quantiles=True,use_symmetric_averaging=True,make_positive=False,sort_quantiles=True,use_znorm=False,padding_mode='none'));arr=[a.quantiles.transpose(1,0,2) for a in ans]
    if univariate:arr=[np.concatenate(arr[j:j+7],axis=1) for j in range(0,len(arr),7)]
    ps.extend(arr)
  del pipe;gc.collect();torch.cuda.empty_cache();q=np.stack(ps).astype(np.float32);np.save(path,q);return q
 def cache(self,name):
  if name=='dlinear':
   from prepare import load_baseline
   net,_=load_baseline();ps=[]
   with torch.no_grad():
    for i in range(0,len(self.ts),128):guard(180);ps.append(net(torch.from_numpy(self.inputs(i,128)).cuda()).cpu().numpy())
   del net;torch.cuda.empty_cache();return np.concatenate(ps)
  if name in ('chronosavg','chronos2','chronosuavg'):q=self.quantiles('chronos',univariate=name=='chronosuavg');return q[:,:,:,4] if name=='chronos2' else q.mean(-1)
  if name in ('timesfmavg','timesfmuavg'):return self.quantiles('timesfm',univariate=name=='timesfmuavg').mean(-1)
  raise ValueError('Unsupported source '+name)
 def run(self,name):
  cfg=json.loads((ROOT/'runs'/name/'config.json').read_text());engine=cfg.get('engine')
  if engine=='view_routing_v1':
   from view_routing import TOOLS,hindcasts,weights
   base=11520 if int(self.ts.min())>=11520 else 8640;anchor=base+((self.ts-base)//24)*24;unique=np.unique(anchor);past=hindcasts(self.z,unique,self.out/'view_route_anchors');aw=np.stack([weights(self.z[t-512:t],v,cfg['strength']) for t,v in zip(unique,past)]);lookup={int(t):i for i,t in enumerate(unique)};ww=aw[[lookup[int(t)] for t in anchor]];pred=np.stack([self.source('run:'+n) for n in TOOLS],1);return (pred*ww[:,:,None]).sum(1).astype(np.float32)
  if engine=='temporal_view_v1':
   from temporal_view import predict
   return predict(self.z,self.dates,self.ts,cfg)
  if engine=='view_memory_gate_v1':
   from view_memory_gate import infer
   pred=np.stack([self.source(k) for k in cfg['source_ids']],1);return infer(self.inputs(0,len(self.ts)),pred,cfg,ROOT/'runs'/name/'best.pth')[0]
  if engine=='view_agent_v1':
   from view_agent import memory,choose
   from view_routing import hindcasts
   from concurrent.futures import ThreadPoolExecutor,as_completed
   assert cfg['part']=='val' and cfg['cadence']==24
   base=11520 if int(self.ts.min())>=11520 else 8640;is_test=base==11520
   assert np.array_equal(self.ts,np.arange(base,base+2817)),'Runtime view recipe requires full ordered split for exact evidence replay'
   pred=np.stack([self.source(k) for k in cfg['source_ids']],1);past=hindcasts(self.z,self.ts,self.out/'view_agent_observed');bank=memory('val');ww=np.empty((len(self.ts),3,7));records=[]
   def one(i):
    guard(180);w,status,chosen=choose(self.z[self.ts[i]-512:self.ts[i]],pred[i],past[i],self.ts[i],bank,cfg['mode'],cfg['k'],self.out/'view_agent_requests',f'{"test" if is_test else "val"}_{i:04d}',replay_from=None if is_test or not cfg['runtime_llm'] else ROOT/'runs'/name/'requests');return i,w,status
   with ThreadPoolExecutor(max_workers=2) as pool:
    for future in as_completed([pool.submit(one,i) for i in range(0,len(self.ts),24)]):
     i,w,status=future.result();ww[i:i+24]=w;records.append({'index':i,'status':status})
   save_json(self.out/(name+'_runtime_decisions.json'),records);np.save(self.out/(name+'_runtime_weights.npy'),ww);return (pred*ww[:,:,None]).sum(1).astype(np.float32)
  if engine=='timesfm_finetune_v1':return self.quantiles('timesfm',model=name).mean(-1)
  if engine=='transformed_tool_v1':
   from transformed_tool import predict
   return predict(self.z,self.ts,cfg,dates=self.dates)
  if engine=='live_feedback_agent_v1' and not cfg['runtime_llm']:
   from live_feedback_agent import numeric
   from observed_feedback import hindcasts,correction,PARENT
   base=11520 if int(self.ts.min())>=11520 else 8640;anchor=base+((self.ts-base)//24)*24;unique=np.unique(anchor);past=hindcasts(self.z,self.dates,self.ts,self.out/'strong_observed_feedback');ap=hindcasts(self.z,self.dates,unique,self.out/'live_feedback_anchors');aw=np.stack([numeric(self.z[t-512:t],v,cfg['mode']) for t,v in zip(unique,ap)]);lookup={int(t):i for i,t in enumerate(unique)};w=aw[[lookup[int(t)] for t in anchor]];return (self.source('run:'+PARENT)+w[:,None]*correction(self.z,self.ts,past)).astype(np.float32)
  if engine=='quantile_calibration_v1':
   from quantile_calibration import apply
   parent=json.loads((ROOT/'runs'/cfg['parent']/'config.json').read_text());return apply(self.quantiles('chronos',model=parent['checkpoint_parent']),cfg)
  if engine=='observed_feedback_v1':
   from observed_feedback import hindcasts,correction
   p=hindcasts(self.z,self.dates,self.ts,self.out/'strong_observed_feedback');return (self.source('run:'+cfg['parent'])+cfg['strength']*correction(self.z,self.ts,p)).astype(np.float32)
  if engine=='strong_view_feedback_v1':
   from strong_view_feedback import hindcasts,apply
   past=None if cfg['mode']=='uniform' else hindcasts(self.z,self.dates,self.ts,self.out/'strong_view_hindcasts');p=np.stack([self.source(k) for k in cfg['source_ids'][1:]],1);return apply(self.z,self.ts,past,p,self.source(cfg['source_ids'][0]),cfg)
  if engine=='analogue_transfer_v1':
   from analogue_transfer import predict
   return predict(self.z,self.ts,cfg,self.dates,self.out/(name+'_retrieval.npz'))
  if engine=='analogue_context_v1':
   from analogue_context import predict
   return predict(self.z,self.ts,cfg,self.dates,self.out/(name+'_retrieval.npz'))
  if engine=='projected_composition_v1':
   from project_composition import apply
   return apply(np.stack([self.source(k) for k in cfg['source_ids']],1),cfg)
  if engine=='semantic_ridge_v1':
   from semantic_ridge import predict
   state=torch.load(ROOT/'runs'/name/'best.pth',map_location='cpu',weights_only=False);parts=[]
   for i in range(0,len(self.ts),128):
    guard(180);parts.append(predict(self.inputs(i,128),cfg,state))
   return np.concatenate(parts)
  if engine=='individual_dlinear_v1':
   from train_individual_dlinear import Individual
   net=Individual().cuda(0);net.load_state_dict(torch.load(ROOT/'runs'/name/'best.pth',map_location='cuda:0',weights_only=False));net.eval();parts=[]
   with torch.no_grad():
    for i in range(0,len(self.ts),128):
     guard(180);parts.append(net(torch.from_numpy(self.inputs(i,128)).cuda(0)).cpu().numpy())
   del net;torch.cuda.empty_cache();return np.concatenate(parts)
  if engine=='block_column_selector_v1':
   from block_column_selection import predict
   return predict(np.stack([self.source(k) for k in cfg['source_ids']],1),cfg)
  if engine=='column_selector_v1':
   from column_selection import predict
   return predict(np.stack([self.source(k) for k in cfg['source_ids']],1),cfg)
  if name.startswith('20261105_ridge_') and cfg.get('features') in ('full','multiscale'):
   from ridge_tool import predict
   weights=np.load(ROOT/'runs'/name/'weights.npy');parts=[]
   for i in range(0,len(self.ts),128):
    guard(180);parts.append(predict(self.inputs(i,128),cfg,weights))
   return np.concatenate(parts)
  if engine=='seasonal_error_tree_v1':
   from seasonal_error_tree import infer
   return infer(self.z,self.dates,self.ts,np.stack([self.source(k) for k in cfg['source_ids']],1),cfg,ROOT/'runs'/name/'best.pth')
  if engine=='calendar_memory_gate_v1':
   from calendar_memory_gate import infer
   return infer(self.inputs(0,len(self.ts)),np.stack([self.source(k) for k in cfg['source_ids']],1),cfg,ROOT/'runs'/name/'best.pth')
  if engine=='calendar_delta_v1':
   from calendar_delta_tool import predict
   return predict(self.z,self.dates,self.ts,cfg,ROOT/'runs'/name/'best.pth')
  if engine=='asinh_tool_v1':
   from asinh_tool import predict
   return predict(self.z,self.ts,cfg,self.dates)
  if engine=='horizon_policy_v1':
   from horizon_policy import predict,active_sources
   active=active_sources(cfg);zero=np.zeros((len(self.ts),64,7),np.float32)
   return predict(np.stack([self.source(k) if active[i] else zero for i,k in enumerate(cfg['source_ids'])],1),cfg)
  if engine=='specialist_policy_v1':
   from specialist_policy import predict
   p=np.stack([self.source(k) for k in cfg['source_ids']],1);return predict(self.inputs(0,len(self.ts)),p,cfg)[0]
  if engine=='conditional_policy_v1':
   from conditional_policy import predict
   p=np.stack([self.source(k) for k in cfg['source_ids']],1);return predict(self.inputs(0,len(self.ts)),p,cfg)[0]
  if engine in ('point_mse_finetune_v1','lora_finetune_v1'):return self.quantiles('chronos',model=name).mean(-1)
  if engine=='reloaded_finetune_v1':return self.quantiles('chronos',model=cfg['checkpoint_parent']).mean(-1)
  if engine=='error_tree_v1':
   from error_tree import ErrorTree
   from meta_gate import features
   q=np.stack([self.quantiles('chronos'),self.quantiles('timesfm')],1);p=q.mean(-1);x=self.inputs(0,len(self.ts));f,s=features(x,p,q,self.ts,self.dates);phase=(self.ts[:,None]+np.arange(64))%24;net=ErrorTree(cfg['features_dim'],cfg['depth']).cuda();net.load_state_dict(torch.load(ROOT/'runs'/name/'best.pth',map_location='cuda:0',weights_only=False));net.eval()
   with torch.no_grad():cor=net(*[torch.from_numpy(a).cuda() for a in (f,s,phase)])[0].cpu().numpy()
   del net;torch.cuda.empty_cache();return ((self.source('cache:dlinear')+2*(p.mean(1)+cor))/3).astype(np.float32)
  if engine=='learned_relations_v1':
   from learned_relations import apply
   p=(self.source('cache:chronosavg')+self.source('cache:timesfmavg'))/2;return apply(self.inputs(0,len(self.ts)),p,self.source('cache:dlinear'),cfg)
  if engine=='enriched_memory_v1':
   from enriched_memory import predict
   q=np.stack([self.quantiles('chronos'),self.quantiles('timesfm')],1);return predict(self.z,self.ts,q,self.source('cache:dlinear'),cfg['mode'])[0]
  if engine=='blend_sources_v1':return self.blend(cfg)
  if engine=='quantile_average_v1':
   parent=cfg['parent']
   if 'chronoscalft' in parent:return self.quantiles('chronos',model=parent,calendar='hour').mean(-1)
   if 'chronosft' in parent:return self.quantiles('chronos',model=parent).mean(-1)
   if 'calendar' in parent:return self.quantiles('chronos',calendar=json.loads((ROOT/'runs'/parent/'config.json').read_text())['mode']).mean(-1)
   raise ValueError(parent)
  if engine=='fixed_view_prior_v1':
   from view_prior import predict
   return predict(np.stack([self.source(k) for k in cfg['source_ids']],1),cfg['weights'])
  if engine=='masked_residual_strength_v1':
   from masked_residual_strength import apply
   return apply(*[self.source(k) for k in cfg['source_ids']],cfg)
  if engine=='static_memory_v1':
   from static_memory import infer
   return infer(self.z,self.dates,self.ts,np.stack([self.source(k) for k in cfg['source_ids']],1),cfg,ROOT/'runs'/name/'best.pth')
  if engine=='factor_error_tree_v1':
   from factor_error_tree import infer
   return infer(self.z,self.dates,self.ts,np.stack([self.source(k) for k in cfg['source_ids']],1),cfg,ROOT/'runs'/name/'best.pth')
  if engine=='variable_error_tree_v1':
   from variable_error_tree import infer
   return infer(self.z,self.dates,self.ts,np.stack([self.source(k) for k in cfg['source_ids']],1),cfg,ROOT/'runs'/name/'best.pth')
  if engine=='prefix_error_tree_v1':
   from prefix_error_tree import infer
   return infer(self.z,self.dates,self.ts,np.stack([self.source(k) for k in cfg['source_ids']],1),cfg,ROOT/'runs'/name/'best.pth')
  if engine=='legacy_numeric_tool_v1':
   from legacy_tools import predict
   return predict(self.z,self.dates,self.ts,cfg)
  if engine=='constant_residual_v1':return (self.source(cfg['source_ids'][0]).astype(float)+cfg['strength']*np.asarray(cfg['correction'],dtype=np.float32)[None]).astype(np.float32)
  if engine=='residual_transfer_v1':
   from residual_transfer import apply
   return apply(*[self.source(k) for k in cfg['source_ids']],cfg['strength'])
  if engine=='view_residual_memory_v1':
   from view_residual_memory import predict
   state=torch.load(ROOT/'runs'/name/'best.pth',map_location='cpu',weights_only=False);p=np.stack([self.source(k) for k in cfg['source_ids']],1);return predict(self.inputs(0,len(self.ts)),p,state)
  if engine=='calendar_fullpoint_v1':
   from transformed_tool import extract_point
   return extract_point(self.quantiles('chronos',model=cfg['checkpoint_parent'],calendar='hour',levels=cfg['point_quantile_levels']),cfg['point_quantile_levels'],'full_integral').astype(np.float32)
  if engine=='calendar_finetune_v1':return self.quantiles('chronos',model=name,calendar='hour').mean(-1)
  if 'chronosft_' in name:return self.quantiles('chronos',model=name)[:,:,:,4]
  if 'calendar_' in name:return self.quantiles('chronos',calendar=cfg['mode'])[:,:,:,4]
  if 'metagate_' in name:
   from meta_gate import Gate,features
   q=np.stack([self.quantiles('chronos'),self.quantiles('timesfm')],1);p=q.mean(-1);x=self.inputs(0,len(self.ts));f,s=features(x,p,q,self.ts,self.dates);net=Gate(cfg['features_dim'],cfg['kind']).cuda();net.load_state_dict(torch.load(ROOT/'runs'/name/'best.pth',map_location='cuda:0',weights_only=False));net.eval()
   with torch.no_grad():y=net(*[torch.from_numpy(a).cuda() for a in (f,p,s)])[0].cpu().numpy()
   del net;torch.cuda.empty_cache();return y
  if engine=='dense_memory_strength_v1':
   prior=sum(self.source('cache:'+n) for n in ('dlinear','chronosavg','timesfmavg'))/3;corrected=self.source('run:'+cfg['parent']);return (prior+cfg['strength']*4*(corrected-prior)).astype(np.float32)
  if engine=='gated_semantics_v1':
   from gated_semantics import apply
   return apply(self.inputs(0,len(self.ts)),self.source('run:'+cfg['parent']),cfg['k'],cfg['strength'])
  if 'densememory_' in name:return self.dense(cfg['mode'])
  raise ValueError('Unsupported recipe '+name)
 def blend(self,cfg):
  w=cfg['sources'];p=sum(v/sum(w.values())*self.source(k).astype(np.float64) for k,v in w.items());center=p.mean(1,keepdims=True);p=center+cfg.get('amplitude',1)*(p-center)
  if cfg.get('mean_shrink',0):p+=cfg['mean_shrink']*(np.stack([self.z[t-96:t].mean(0) for t in self.ts])[:,None]-center)
  smooth=cfg.get('smooth',1)
  if smooth>1:
   pad=smooth//2;a=np.pad(p,((0,0),(pad,pad),(0,0)),mode='edge');p=sum(a[:,j:j+64] for j in range(smooth))/smooth
  if cfg.get('proxy_bias_strength',0):p-=cfg['proxy_bias_strength']*self.hindcast_bias()[:,None]*np.linspace(1,.5,64)[None,:,None]
  return p.astype(np.float32)
 def hindcast_bias(self):
  path=self.out/'observed_context_bias.npy'
  if path.exists():return np.load(path)
  from chronos import Chronos2Pipeline
  from prepare import load_baseline
  pipe=Chronos2Pipeline.from_pretrained(str(RUN/'models/chronos2_29ec3766'),device_map='cuda:0');pipe.model.eval();net,_=load_baseline();bias=[]
  for i in range(0,len(self.ts),16):
   guard(180);x=self.inputs(i,16);obs=x[:,:448];target=x[:,448:];padded=np.concatenate([np.repeat(obs[:,:1],64,axis=1),obs],axis=1)
   with torch.no_grad():pd=net(torch.from_numpy(padded).cuda()).cpu().numpy()
   _,pc=pipe.predict_quantiles([torch.from_numpy(a.T.copy()) for a in obs],prediction_length=64,batch_size=112,context_length=512,cross_learning=False);pc=np.stack([a.cpu().numpy().T for a in pc]);pred=np.stack([pd,pc],1);bias.append((pred-target[:,None]).mean(2).mean(1))
  del pipe,net;gc.collect();torch.cuda.empty_cache();b=np.concatenate(bias);np.save(path,b);return b
 def dense(self,mode):
  from hybrid_router import descriptor
  ca=np.load(RUN/'cache/dense_chronos_train.npz');ta=np.load(RUN/'cache/dense_timesfm_train.npz');origins=ca['origins'];tx,ty=windows(self.z[:8640],origins);tp=np.stack([ca['predictions'],ta['predictions']],1);tq=np.stack([ca['quantiles'],ta['quantiles']],1);err=ty[:,None]-tp;fx=np.stack([descriptor(x) for x in tx]);sx=np.maximum(fx.std(0),.1)
  def bf(x,p,q):
   recent=x[:,-96:].mean(1);f=[]
   for m in range(2):f.extend([p[:,m].mean(1)-recent,p[:,m].std(1),(q[:,m,:,:,-1]-q[:,m,:,:,0]).mean(1),(p[:,m]-q[:,m,:,:,4]).mean(1)])
   return np.concatenate(f,axis=1)
  fb=bf(tx,tp,tq);sb=np.maximum(fb.std(0),.1);patterns=np.empty((len(origins),2,24,7),np.float32)
  for i,t in enumerate(origins):
   hours=(t+np.arange(64))%24
   for h in range(24):patterns[i,:,h]=err[i][:,hours==h].mean(1)
  x=self.inputs(0,len(self.ts));q=np.stack([self.quantiles('chronos'),self.quantiles('timesfm')],1);p=q.mean(-1);dl=self.source('cache:dlinear');xb=np.stack([descriptor(a) for a in x]);bb=bf(x,p,q);ps=[]
  for i,t in enumerate(self.ts):
   valid=np.where(origins+64<=t-512)[0];dist=np.mean(((fx[valid]-xb[i])/sx)**2,axis=1)
   if mode!='xlevel':dist+=np.mean(((fb[valid]-bb[i])/sb)**2,axis=1)
   chosen=[]
   for j in valid[np.argsort(dist)]:
    if all(abs(int(origins[j])-int(origins[k]))>=64 for k in chosen):chosen.append(j)
    if len(chosen)==8:break
   cor=patterns[chosen].mean(0)[:,(t+np.arange(64))%24] if mode=='xbphase' else np.broadcast_to(err[chosen].mean((0,2))[:,None],(2,64,7));bound=.5*np.maximum(x[i].std(0),.05);corrected=p[i]+.25*np.clip(cor,-bound,bound);ps.append((dl[i]+corrected.sum(0))/3)
  return np.stack(ps).astype(np.float32)
