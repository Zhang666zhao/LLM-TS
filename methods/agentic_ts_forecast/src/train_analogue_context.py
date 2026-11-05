"""Matched train-only adaptation of old-analogue covariate interfaces."""
from common import *
from analogue_context import retrieve,make_items,predict
from transformed_tool import extract_point
import torch,pandas as pd,sys,gc
from chronos import Chronos2Pipeline
from chronos.chronos2.preprocess import from_list_of_dicts

def run(mode):
 assert mode in ('history_only','future_case','phase_future_case');parent=f'20261105_analoguecontext_{mode}_mse';base=json.loads((ROOT/'runs'/parent/'config.json').read_text());name=f'20261105_analogueadapt_{mode}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=False);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);guard(2400);torch.set_num_threads(4);torch.manual_seed(2021);torch.cuda.manual_seed_all(2021);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;z=np.load(RUN/'cache/trainval_z.npy');dates=pd.to_datetime(json.loads((RUN/'cache/dates.json').read_text()));b=dict(np.load(base['training_memory']));assert hashlib.sha256(pathlib.Path(base['training_memory']).read_bytes()).hexdigest()==base['training_memory_sha256'];pipe=Chronos2Pipeline.from_pretrained(str(ROOT/'runs'/base['checkpoint_parent']/'trainer/finetuned-ckpt'),device_map='cuda:0');net=pipe.model;net.eval();levels=base['point_quantile_levels'];bias=torch.tensor(base['bias'],device='cuda:0',dtype=torch.float32);inv=torch.tensor(base['inverse_matrix'],device='cuda:0',dtype=torch.float32);qw=torch.tensor(extract_point(np.eye(len(levels)),levels,'full_integral'),device='cuda:0',dtype=torch.float32);cfg=dict(base,source_ids=['run:'+parent],checkpoint_parent=name,adaptation='500 full-backbone point-MSE updates with causally old training analogues',train_query_origin_min=1088,training_rows=[0,8640],finetune_steps=500,finetune_rate=5e-7,batch_windows=16,seed=2021,code_revision=git_head());save_json(out/'config.json',cfg)
 def prepare(ts):
  chosen=retrieve(z,ts,dates,base,b);items=make_items(z,ts,chosen,base,b,dates);a=from_list_of_dicts(items,prediction_length=64)
  assert all(v['context'].shape==(16,512) and torch.isnan(v['future_covariates'][:7]).all() for v in a)
  return torch.cat([v['context'] for v in a]).cuda(),torch.cat([v['future_covariates'] for v in a]).cuda(),torch.arange(len(ts),device='cuda').repeat_interleave(16),items,chosen
 def forward(context,future,groups):
  n=len(context)//16;q=net(context=context,group_ids=groups,future_covariates=future,num_output_patches=4).quantile_preds[:,:,:64].reshape(n,16,len(levels),64)[:,:7];point=(q.float()*qw[None,None,:,None]).sum(2).transpose(1,2)
  with torch.autocast('cuda',enabled=False):return (point.float()-bias)@inv
 xx,ff,gg,items,_=prepare(np.array([1536,2048]))
 with torch.no_grad():
  pp=forward(xx,ff,gg);changed=xx.clone();changed[16:]+=10;assert torch.equal(pp[0],forward(changed,ff,gg)[0]);q,_=pipe.predict_quantiles(items,prediction_length=64,quantile_levels=levels,batch_size=112,context_length=512,cross_learning=False);qq=np.stack([v.numpy().transpose(1,0,2) for v in q]);public=((extract_point(qq,levels,'full_integral')-np.asarray(base['bias']))@np.asarray(base['inverse_matrix'])).astype(np.float32);delta=float(abs(pp.cpu().numpy()-public).max());assert delta<1e-5
 save_json(out/'contracts.json',dict(public_forward_max_delta=delta,query_isolation_exact=True,own_future_unknown=True,memory_labels_strictly_before_query_history=True));opt=torch.optim.AdamW(net.parameters(),lr=5e-7,weight_decay=.01);rng=torch.Generator().manual_seed(2021);orders=[];memory_orders=[];logs=[];net.train();start=time.time()
 for step in range(500):
  guard(1800);ts=torch.randint(1088,8577,(16,),generator=rng).numpy();orders.append(ts);xx,ff,gg,_,chosen=prepare(ts);memory_orders.append(chosen);yy=torch.from_numpy(np.stack([z[t:t+64] for t in ts])).cuda();opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):pp=forward(xx,ff,gg);loss=(pp-yy).square().mean()
  assert torch.isfinite(loss);loss.backward();grad=torch.nn.utils.clip_grad_norm_(net.parameters(),1.);assert torch.isfinite(grad);opt.step()
  for group in opt.param_groups:group['lr']=5e-7*(1-(step+1)/500)
  if step%100==0 or step==499:logs.append(dict(step=step+1,loss=float(loss.detach()),gradient_norm=float(grad)));print(name,logs[-1],flush=True)
 pipe.save_pretrained(out/'trainer/finetuned-ckpt');np.save(out/'training_sample_origins.npy',np.stack(orders));np.save(out/'training_memory_origins.npy',np.stack(memory_orders));assert (np.stack(memory_orders)+64<=np.stack(orders)-512).all();save_json(out/'training_log.json',logs);del pipe,net,opt,pp,loss,xx,ff,gg,yy;gc.collect();torch.cuda.empty_cache();results={}
 for split,ts in json.loads((RUN/'cache/window_manifest.json').read_text()).items():
  p=predict(z,ts,cfg,dates,out/(split+'_retrieval.npz'));np.save(out/f'{split}_predictions.npy',p);results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'));save_json(out/'partial_metrics.json',results)
 save_json(out/'metrics.json',dict(candidate=name,metrics=results,status='evaluated',elapsed_seconds=time.time()-start));print(name,results['val']['mse'],flush=True)
if __name__=='__main__':run(sys.argv[1])
