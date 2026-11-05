"""Diagnose failed FIRST training batch without any optimizer update."""
from common import *
import torch,traceback,contextlib
from timesfm3 import TimesFM3Evaluator,ModelConfig
out=RUN/'timesfm_real_gradient_audit';out.mkdir(exist_ok=True);guard(1800);torch.set_num_threads(4);torch.manual_seed(2021);torch.cuda.manual_seed_all(2021);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;z=np.load(RUN/'cache/trainval_z.npy')[:8640];sampler=torch.Generator().manual_seed(2021);ts=torch.randint(512,8577,(8,),generator=sampler).numpy();x,y=windows(z,ts);xx=torch.from_numpy(x).cuda();yy=torch.from_numpy(y).cuda();pipe=TimesFM3Evaluator(ModelConfig(checkpoint_path=str(RUN/'models/timesfm3_43046b85'),per_core_batch_size=16,device='cuda:0',local_files_only=True));net=pipe.model;raw=net.decode.__wrapped__;net.train();results={}
for mode in ('fp32','bf16'):
 net.zero_grad(set_to_none=True);record={'optimizer_steps':0,'input_origins':ts.tolist(),'historical_std_min_by_variable':x.std(1).min(0).tolist()}
 try:
  with torch.autograd.detect_anomaly(check_nan=True):
   with (torch.autocast('cuda',dtype=torch.bfloat16) if mode=='bf16' else contextlib.nullcontext()):
    q=raw(net,target=torch.cat([xx.transpose(1,2),-xx.transpose(1,2)],0),horizon=64);p=((q[:8].mean(-1)-q[8:].mean(-1))*.5).transpose(1,2);loss=(p.float()-yy).square().mean();record.update(loss=float(loss.detach()),prediction_finite=bool(torch.isfinite(p).all()));loss.backward()
  bad=[k for k,v in net.named_parameters() if v.grad is not None and not torch.isfinite(v.grad).all()];record.update(nonfinite_gradient_names=bad,finite_backward=not bad,grad_norm_float64=float(torch.sqrt(sum(v.grad.double().square().sum() for v in net.parameters() if v.grad is not None))))
 except Exception as e:record.update(exception=type(e).__name__,message=str(e),traceback=traceback.format_exc())
 results[mode]=record;save_json(out/'results.json',results);print(mode,record,flush=True)
save_json(RUN/'timesfm_training_STOPPED.json',{'failed_candidate':'20261105_timesfmft_full1e6_mse','optimizer_updates':0,'remaining_candidates_not_launched':['full5e6','head1e4'],'reason':'Nonfinite gradient norm on first actual training batch; queue stopped','diagnostic_only_no_optimizer':str(out/'results.json'),'epoch':time.time(),'code_revision':git_head()})
