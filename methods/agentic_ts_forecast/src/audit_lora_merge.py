"""Read-only saved adapter versus merged checkpoint precision audit."""
from common import *
import torch,peft,sys,gc
from chronos import Chronos2Pipeline
name=sys.argv[1];out=ROOT/'runs'/name;torch.set_num_threads(4);guard(1200);z=np.load(RUN/'cache/trainval_z.npy');sample=[torch.from_numpy(z[t-512:t].T.copy()) for t in (512,576)];base=Chronos2Pipeline.from_pretrained(str(RUN/'models/chronos2_29ec3766'),device_map='cuda:0');model=peft.PeftModel.from_pretrained(base.model,str(out/'trainer/adapter-ckpt'));model.eval();pipe=Chronos2Pipeline(model=model)
def pred(p):
 q,_=p.predict_quantiles(sample,prediction_length=64,batch_size=112,context_length=512,cross_learning=False);return np.stack([v.float().cpu().numpy() for v in q])
fp=pred(pipe);repeat=pred(pipe)
with torch.autocast('cuda',dtype=torch.bfloat16):bf=pred(pipe)
dtypes=sorted(set(str(p.dtype) for p in model.parameters()));trainable=sum(p.numel() for n,p in model.named_parameters() if 'lora_' in n);del pipe,model,base;gc.collect();torch.cuda.empty_cache();merged=Chronos2Pipeline.from_pretrained(str(out/'trainer/finetuned-ckpt'),device_map='cuda:0');merged.model.eval();mp=pred(merged);result={'candidate':name,'saved_adapter_vs_merged_fp32_max_delta':float(abs(fp-mp).max()),'adapter_repeat_max_delta':float(abs(fp-repeat).max()),'autocast_bf16_adapter_vs_merged_max_delta':float(abs(bf-mp).max()),'parameter_dtypes':dtypes,'lora_parameters':trainable,'sample_origins':[512,576],'test_access':False,'no_retraining':True,'code_revision':git_head()};save_json(out/'saved_adapter_merge_audit.json',result);np.savez(out/'saved_adapter_merge_audit_arrays.npz',adapter_fp32=fp,adapter_bf16=bf,merged_fp32=mp);print(result,flush=True);assert result['saved_adapter_vs_merged_fp32_max_delta']<1e-5 and result['adapter_repeat_max_delta']==0
