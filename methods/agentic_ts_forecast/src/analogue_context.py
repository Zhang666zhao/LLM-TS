"""Old training analogues as past/known-future covariates; never current future labels."""
from common import *
from transformed_tool import extract_point
from transformed_calendar import cov
import torch,pandas as pd,sys,gc
from chronos import Chronos2Pipeline
from chronos.chronos2.preprocess import from_list_of_dicts

def descriptor(x):
 x=x.astype(float);c=x[:,-96:].mean(1,keepdims=True);shape=(x-c).reshape(-1,32,16,7).mean(2).reshape(len(x),-1);return np.concatenate([shape,x[:,-24:].mean(1),x[:,-96:].mean(1),x.mean(1),x.std(1),x[:,-24:].mean(1)-x[:,-48:-24].mean(1)],1)

def bank():
 path=RUN/'cache/analogue_context_memory_v1.npz'
 if not path.exists():
  z=np.load(RUN/'cache/trainval_z.npy')[:8640].copy();origins=np.arange(512,8577);parts=[]
  for i in range(0,len(origins),128):parts.append(descriptor(np.stack([z[t-512:t] for t in origins[i:i+128]])))
  f=np.concatenate(parts);mu=f.mean(0);sd=np.maximum(f.std(0),1e-6);dates=pd.to_datetime(json.loads((RUN/'cache/dates.json').read_text()));np.savez_compressed(path,train_z=z,origins=origins,features=(f-mu)/sd,mean=mu,scale=sd,hour=np.asarray(dates[origins-1].hour));save_json(path.with_suffix('.json'),dict(training_rows=[0,8640],origins=len(origins),stride=1,descriptor_dim=f.shape[1],label_end_max=8640,eligibility='caseorigin+64 <= queryorigin-512',feature_scaler='training only',code_revision=git_head()))
 return path

def retrieve(z,ts,dates,cfg,b):
 feats=b['features'];norm=(feats*feats).sum(1);selected=[]
 for i in range(0,len(ts),64):
  orig=ts[i:i+64];x=np.stack([z[t-512:t] for t in orig]);q=(descriptor(x)-b['mean'])/b['scale'];dist=(q*q).sum(1)[:,None]+norm[None]-2*q@feats.T;eligible=b['origins'][None]+64<=orig[:,None]-512
  if cfg['memory_mode']=='phase_future_case':eligible &= b['hour'][None]==np.asarray(dates[orig-1].hour)[:,None]
  assert eligible.any(1).all(),'No causally eligible memory';dist[~eligible]=np.inf;chosen=b['origins'][dist.argmin(1)];assert (chosen+64<=orig-512).all() and (chosen+64<=8640).all();selected.extend(chosen.tolist())
 return np.asarray(selected)

def make_items(z,orig,nearest,cfg,b,dates):
 w=np.asarray(cfg['matrix']);bias=np.asarray(cfg['bias'])
 result=[]
 for t,a in zip(orig,nearest):
  own=(z[t-512:t].astype(float)@w+bias).astype(np.float32);old=(b['train_z'][a-512:a].astype(float)@w+bias).astype(np.float32);oldfuture=(b['train_z'][a:a+64].astype(float)@w+bias).astype(np.float32);past=cov(dates[t-512:t]);future=cov(pd.date_range(dates[t-1]+pd.Timedelta(hours=1),periods=64,freq='h'))
  for j in range(7):
   key='old_training_case_'+str(j);past[key]=old[:,j].copy()
   if cfg['memory_mode']!='history_only':future[key]=oldfuture[:,j].copy()
  result.append(dict(target=own.T.copy(),past_covariates=past,future_covariates=future))
 return result

def predict(z,ts,cfg,dates,trace=None):
 torch.set_num_threads(4);dates=pd.to_datetime(dates);ts=np.asarray(ts);path=pathlib.Path(cfg['training_memory']);assert hashlib.sha256(path.read_bytes()).hexdigest()==cfg['training_memory_sha256'];b=dict(np.load(path));assert len(b['train_z'])==8640;chosen=retrieve(z,ts,dates,cfg,b);w=np.asarray(cfg['matrix']);bias=np.asarray(cfg['bias']);inv=np.asarray(cfg['inverse_matrix']);loc=ROOT/'runs'/cfg['checkpoint_parent']/'trainer/finetuned-ckpt';pipe=Chronos2Pipeline.from_pretrained(str(loc),device_map='cuda:0');pipe.model.eval();levels=cfg['point_quantile_levels'];parts=[]
 items_for=lambda orig,nearest:make_items(z,orig,nearest,cfg,b,dates)
 probe=items_for(ts[:2],chosen[:2]);prepared=from_list_of_dicts(probe,prediction_length=64)
 for a in prepared:
  assert a['n_targets']==7 and a['context'].shape==(16,512) and torch.isnan(a['future_covariates'][:7]).all();known=torch.isfinite(a['future_covariates']).all(1).sum().item();assert known==(2 if cfg['memory_mode']=='history_only' else 9)
 for i in range(0,len(ts),7):
  guard(600);items=items_for(ts[i:i+7],chosen[i:i+7]);q,_=pipe.predict_quantiles(items,prediction_length=64,quantile_levels=levels,batch_size=112,context_length=512,cross_learning=False);qq=np.stack([v.numpy().transpose(1,0,2) for v in q]);parts.append(((extract_point(qq,levels,'full_integral')-bias)@inv).astype(np.float32))
 del pipe;gc.collect();torch.cuda.empty_cache();p=np.concatenate(parts)
 if trace is not None:np.savez_compressed(trace,query_origins=ts,memory_origins=chosen,memory_label_end=chosen+64)
 return p

def run(mode):
 assert mode in ('history_only','future_case','phase_future_case');name=f'20261105_analoguecontext_{mode}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=False);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);guard(2400);path=bank();parent='20261105_learnedrotation_r003_masked_mse';cfg=dict(json.loads((ROOT/'runs'/parent/'config.json').read_text()),engine='analogue_context_v1',memory_mode=mode,training_memory=str(path),training_memory_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_ids=['run:'+parent],adaptation='no new fitting; frozen saved backbone and view',dev_confirm_in_sample=True,code_revision=git_head());save_json(out/'config.json',cfg);z=np.load(RUN/'cache/trainval_z.npy');dates=pd.to_datetime(json.loads((RUN/'cache/dates.json').read_text()));ts=np.array([1536,2048]);a=predict(z,ts,cfg,dates);changed=z.copy();changed[1536:2048]+=10;b=predict(changed,ts,cfg,dates);assert np.array_equal(a[0],b[0]),'Query isolation failed';save_json(out/'contracts.json',dict(query_isolation_exact=True,own_future_target_rows_unknown=True,memory_label_end_max=8640,memory_ends_before_query_history=True,known_future_covariate_rows=2 if mode=='history_only' else 9));results={};start=time.time()
 for split,orig in json.loads((RUN/'cache/window_manifest.json').read_text()).items():
  p=predict(z,orig,cfg,dates,out/(split+'_retrieval.npz'));np.save(out/f'{split}_predictions.npy',p);results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'));save_json(out/'partial_metrics.json',results)
 save_json(out/'metrics.json',dict(candidate=name,metrics=results,status='evaluated',elapsed_seconds=time.time()-start));print(name,results['val']['mse'],flush=True)
if __name__=='__main__':run(sys.argv[1])
