"""Experimental TimesFM3 point-MSE adaptation using audited differentiable decode."""
from common import *
import torch,sys,gc,inspect
from timesfm3 import TimesFM3Evaluator,ModelConfig

def predict(z,ts,checkpoint):
 pipe=TimesFM3Evaluator(ModelConfig(checkpoint_path=str(checkpoint),per_core_batch_size=16,device='cuda:0',local_files_only=True));pipe.model.eval();ps=[]
 for i in range(0,len(ts),16):
  guard(600);items=[z[t-512:t].T.copy() for t in ts[i:i+16]];ans=list(pipe.predict_batch(items,horizon=64,return_quantiles=True,use_symmetric_averaging=True,make_positive=False,sort_quantiles=True,use_znorm=False,padding_mode='none'));ps.extend([a.quantiles.transpose(1,0,2).mean(-1) for a in ans])
 del pipe;gc.collect();torch.cuda.empty_cache();return np.stack(ps).astype(np.float32)
def run(mode,rate,tag,safe_stats=False):
 contract=json.loads((RUN/'timesfm_gradient_contract.json').read_text());assert contract['finite_backward'] and contract['direct_public_max_delta']<1e-5 and contract['other_window_perturbation_delta']==0;name=f'20261105_timesfmft_{tag}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);guard(3600);torch.set_num_threads(4);torch.manual_seed(2021);torch.cuda.manual_seed_all(2021);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;free,total=torch.cuda.mem_get_info(0);assert free>22*1024**3,'Require22GiB free before TimesFM adaptation';z=np.load(RUN/'cache/trainval_z.npy');pipe=TimesFM3Evaluator(ModelConfig(checkpoint_path=str(RUN/'models/timesfm3_43046b85'),per_core_batch_size=16,device='cuda:0',local_files_only=True));net=pipe.model;external=pathlib.Path(inspect.getsourcefile(type(net)));assert hashlib.sha256(external.read_bytes()).hexdigest()==contract['model_source_sha256'];raw=net.decode.__wrapped__;cfg={'engine':'timesfm_finetune_v1','mode':mode,'rate':rate,'steps':1000,'batch_windows':8,'training_rows':[0,8640],'context':512,'horizon':64,'seed':2021,'point':'symmetric average of all9 quantile outputs, matching public mean-point interface','training_caveat':'Experimental differentiable wrapper around inference-only TimesFM3 decode; no official TimesFM3 training-support claim','external_model_source':str(external),'external_model_source_sha256':contract['model_source_sha256'],'optimizer':'AdamW wd.01, global gradclip1, linear decay1000','code_revision':git_head()};save_json(out/'config.json',cfg)
 if safe_stats:
  from timesfm_safe_stats import install,restore
  audit=json.loads((RUN/'timesfm_safe_stats_contract.json').read_text());assert audit['all_passed'] and audit['cases']==10 and audit['forward_max_delta']<1e-5;positive=json.loads((RUN/'timesfm_safe_positive_gradient.json').read_text());assert positive['forward_exact'] and positive['positive_variance_gradient_max_delta']<1e-12;from timesfm3.torch import util;assert hashlib.sha256(pathlib.Path(inspect.getsourcefile(util)).read_bytes()).hexdigest()==audit['update_source_sha256'];original_stats=install();cfg['gradient_extension']='Forward-preserving masked zero-count denominators and zero sqrt subgradient at variance0';cfg['safe_stats_audit']=audit;save_json(out/'config.json',cfg)
 def direct(x):
  n=len(x);target=torch.cat([x.transpose(1,2),-x.transpose(1,2)],0);q=raw(net,target=target,horizon=64);return ((q[:n].mean(-1)-q[n:].mean(-1))*.5).transpose(1,2)
 if mode=='head':
  for k,v in net.named_parameters():v.requires_grad_(k.startswith('output_head.'))
 def frozen_hash():
  h=hashlib.sha256()
  for k,v in net.named_parameters():
   if not v.requires_grad:h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
  return h.hexdigest()
 before=frozen_hash();params=[v for v in net.parameters() if v.requires_grad];cfg['trainable_parameters']=sum(v.numel() for v in params);assert params;save_json(out/'config.json',cfg);opt=torch.optim.AdamW(params,lr=rate,weight_decay=.01);sampler=torch.Generator().manual_seed(2021);orders=[];records=[];start=time.time();net.train();torch.cuda.reset_peak_memory_stats()
 for step in range(1000):
  guard(1800);ts=torch.randint(512,8577,(8,),generator=sampler).numpy();orders.append(ts);x,y=windows(z[:8640],ts);xx=torch.from_numpy(x).cuda();yy=torch.from_numpy(y).cuda();opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):p=direct(xx);loss=(p.float()-yy).square().mean()
  if not torch.isfinite(loss):raise RuntimeError('Nonfinite TimesFM MSE')
  loss.backward();grad=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
  for group in opt.param_groups:group['lr']=rate*(1-(step+1)/1000)
  if step%100==0 or step==999:records.append({'step':step+1,'loss':float(loss.detach()),'grad_norm':float(grad),'elapsed_seconds':time.time()-start});print(name,records[-1],flush=True);save_json(out/'partial_training_log.json',records)
 if safe_stats:
  net.eval()
  with torch.no_grad():patched=direct(xx[:2]).detach();restore(original_stats);standard=direct(xx[:2]).detach()
  delta=float(abs(patched-standard).max());assert delta<1e-5,'Trained patched/native forward changed';cfg['trained_forward_restore_max_delta']=delta;save_json(out/'forward_restore_audit.json',{'max_delta':delta,'inference_uses_original_installed_statistics':True})
 after=frozen_hash();assert before==after,'Frozen parameter changed';checkpoint=out/'trainer/finetuned-ckpt';net.save_pretrained(checkpoint);np.save(out/'training_sample_origins.npy',np.stack(orders));save_json(out/'training_log.json',records);save_json(out/'training_audit.json',{'frozen_hash_before':before,'frozen_hash_after':after,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'gradient_contract':contract});del net,pipe,opt,params;gc.collect();torch.cuda.empty_cache();mf=json.loads((RUN/'cache/window_manifest.json').read_text());results={}
 for split,ts in mf.items():
  p=predict(z,ts,checkpoint);results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'));np.save(out/f'{split}_predictions.npy',p);save_json(out/'partial_metrics.json',results)
 cfg['postfit_inference']='saved checkpoint freshly loaded in public evaluator eval-mode';save_json(out/'config.json',cfg);save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated','elapsed_seconds':time.time()-start});print(name,{s:m['mse'] for s,m in results.items()},flush=True)
if __name__=='__main__':run(sys.argv[1],float(sys.argv[2]),sys.argv[3],bool(int(sys.argv[4])) if len(sys.argv)>4 else False)
