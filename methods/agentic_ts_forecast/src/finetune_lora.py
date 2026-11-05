"""Native Chronos LoRA with frozen-base audit and merged checkpoint inference."""
from common import *
import torch,sys,gc,peft
from chronos import Chronos2Pipeline
from transformers import TrainerCallback
class Deadline(TrainerCallback):
 def on_step_begin(self,args,state,control,**kwargs):guard(1200)
def run(rate,tag):
 name=f'20261105_chronoslora_{tag}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);guard(2400);torch.set_num_threads(4);torch.manual_seed(2021);torch.cuda.manual_seed_all(2021);np.random.seed(2021)
 assert peft.__version__=='0.18.1';lora={'r':8,'lora_alpha':16,'target_modules':['self_attention.q','self_attention.v','self_attention.k','self_attention.o','output_patch_embedding.output_layer'],'lora_dropout':0.,'bias':'none'};cfg={'engine':'lora_finetune_v1','mode':'lora','peft_version':peft.__version__,'lora':lora,'learning_rate':rate,'steps':500,'context_length':512,'prediction_length':64,'min_past':512,'batch_size_including_variates':112,'seed':2021,'training_rows':[0,8640],'checkpoint_selection':'fixed final500steps','point':'average nine quantiles','postfit_inference':'merged full checkpoint, reload explicit eval','code_revision':git_head()};save_json(out/'config.json',cfg);start=time.time();z=np.load(RUN/'cache/trainval_z.npy');pipe=Chronos2Pipeline.from_pretrained(str(RUN/'models/chronos2_29ec3766'),device_map='cuda:0');original={n:p.detach().cpu().clone() for n,p in pipe.model.named_parameters()}
 trained=pipe.fit([torch.from_numpy(z[:8640].T.copy())],prediction_length=64,validation_inputs=None,finetune_mode='lora',lora_config=lora,context_length=512,learning_rate=rate,num_steps=500,batch_size=112,min_past=512,output_dir=out/'trainer',callbacks=[Deadline()],seed=2021,data_seed=2021,disable_tqdm=True,logging_steps=100,bf16=True,tf32=False)
 assert isinstance(trained.model,peft.PeftModel),'LoRA silently fell back';checked=[];trainable=0;total=0
 for n,p in trained.model.named_parameters():
  total+=p.numel()
  if 'lora_' in n:assert p.requires_grad;trainable+=p.numel();continue
  assert not p.requires_grad;key=n.removeprefix('base_model.model.').replace('.base_layer.','.');assert key in original,key;actual=p.detach().cpu();assert actual.dtype==original[key].dtype and torch.equal(actual,original[key]),'Frozen base changed: '+key;checked.append(key)
 assert len(checked)==len(original) and 0<trainable<.05*total
 cfg.update(frozen_base_parameters_verified=len(checked),trainable_parameters=trainable,total_parameters=total);save_json(out/'config.json',cfg)
 # Discard Trainer/Accelerate forward wrappers before comparing adapter and merge.
 del original,pipe,trained;gc.collect();torch.cuda.empty_cache();adapter=out/'trainer/finetuned-ckpt';assert (adapter/'adapter_config.json').exists()
 base=Chronos2Pipeline.from_pretrained(str(RUN/'models/chronos2_29ec3766'),device_map='cuda:0');trained=Chronos2Pipeline(model=peft.PeftModel.from_pretrained(base.model,str(adapter)));trained.model.eval();sample=[torch.from_numpy(z[t-512:t].T.copy()) for t in (512,576)];q,_=trained.predict_quantiles(sample,prediction_length=64,batch_size=112,context_length=512,cross_learning=False);before=np.stack([v.cpu().numpy() for v in q])
 adapter.rename(out/'trainer/adapter-ckpt');merged=Chronos2Pipeline(model=trained.model.merge_and_unload(safe_merge=True));merged.model.eval();assert not any('lora_' in n for n,_ in merged.model.named_parameters());merged.save_pretrained(adapter);del trained,merged,base;gc.collect();torch.cuda.empty_cache()
 trained=Chronos2Pipeline.from_pretrained(str(adapter),device_map='cuda:0');trained.model.eval();q,_=trained.predict_quantiles(sample,prediction_length=64,batch_size=112,context_length=512,cross_learning=False);after=np.stack([v.cpu().numpy() for v in q]);delta=float(abs(before-after).max());assert delta<1e-5,('Reloaded adapter merge changed forecast',delta);assert not (adapter/'adapter_config.json').exists();cfg.update(adapter_merge_max_delta=delta,adapter_comparison='fresh saved adapter FP32 eval versus merged FP32 eval, no Trainer wrapper');save_json(out/'config.json',cfg);manifest=json.loads((RUN/'cache/window_manifest.json').read_text());results={}
 for split,ts in manifest.items():
  qs=[]
  for i in range(0,len(ts),16):
   guard(600);x,_=windows(z,ts[i:i+16]);q,_=trained.predict_quantiles([torch.from_numpy(v.T.copy()) for v in x],prediction_length=64,batch_size=112,context_length=512,cross_learning=False);qs.extend([v.cpu().numpy().transpose(1,0,2) for v in q])
  q=np.stack(qs).astype(np.float32);p=q.mean(-1);results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'));np.save(out/f'{split}_quantiles.npy',q);np.save(out/f'{split}_predictions.npy',p);save_json(out/'partial_metrics.json',results)
 save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated','elapsed_seconds':time.time()-start});print(name,'trainable',trainable,'merge_delta',delta,{s:m['mse'] for s,m in results.items()},flush=True)
if __name__=='__main__':run(float(sys.argv[1]),sys.argv[2])
