"""Forward-held-out training examples for tool-performance memory; no official val/test."""
from common import *
from prefix_memory import predict_prefix
from hybrid_router import descriptor
from types import SimpleNamespace
import torch,sys,importlib.util,gc
from torch.utils.data import DataLoader,TensorDataset
from chronos import Chronos2Pipeline
from transformers import TrainerCallback
class Deadline(TrainerCallback):
 def on_step_begin(self,args,state,control,**kwargs):guard(1200)

def run(cut):
 name=f'20261105_crossfit_prefix{cut}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);guard(2400);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;z=np.load(RUN/'cache/trainval_z.npy')[:8640];manifest=json.loads((RUN/'cache/window_manifest.json').read_text());ts=np.arange(3584,4097,64) if cut==3072 else np.array(manifest['dev'] if cut==4096 else manifest['confirm']);assert cut in (3072,4096,6272) and ts.min()-512>=cut and ts.max()+64<=8640;cfg={'cutoff':cut,'query_origins':ts.tolist(),'official_training_only':True,'scaler':'fixed official full-training scaler; proxy labels are held out from weight fitting, not a separately sealed raw-data benchmark','chronos_rate':5e-7,'chronos_steps':500,'dlinear_reference_hyperparameters':{'batch':32,'lr':.005,'max_epochs':10,'patience':3,'weight_decay':0},'internal_dlinear_selection':'last20percent of prefix, stride16; refit full prefix for selected epochs','seed':2021,'code_revision':git_head()};save_json(out/'config.json',cfg);start=time.time()
 source=pathlib.Path(CONTRACT['source']).parents[2]/'third_party/LTSF-Linear/models/DLinear.py';spec=importlib.util.spec_from_file_location('prefix_official_dlinear',source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 def init():torch.manual_seed(2021);torch.cuda.manual_seed_all(2021);return mod.Model(SimpleNamespace(seq_len=512,pred_len=64,individual=False,enc_in=7)).cuda(0)
 def loader(end):
  x,y=windows(z,np.arange(512,end-64+1));return DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(y)),batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(2021),num_workers=0)
 def epoch(net,opt,dl):
  net.train();total=0.;n=0
  for x,y in dl:
   guard(1200);x=x.cuda();y=y.cuda();opt.zero_grad(set_to_none=True);p=net(x);loss=(p-y).square().mean()
   if not torch.isfinite(loss):raise RuntimeError('nonfinite prefix DLinear loss')
   loss.backward();opt.step();total+=float(loss.detach())*len(x);n+=len(x)
  return total/n
 inner=int(.8*cut);dl=loader(inner);vx,vy=windows(z,np.arange(inner,cut-64+1,16));net=init();opt=torch.optim.Adam(net.parameters(),lr=.005);best=float('inf');best_epoch=0;stale=0;records=[]
 for e in range(10):
  loss=epoch(net,opt,dl);net.eval();pred=[]
  with torch.no_grad():
   for i in range(0,len(vx),128):pred.append(net(torch.from_numpy(vx[i:i+128]).cuda()).cpu().numpy())
  score=metric(np.concatenate(pred),vy)['mse'];records.append({'epoch':e+1,'train_mse':loss,'prefix_internal_val_mse':score})
  if score<best:best=score;best_epoch=e+1;stale=0
  else:stale+=1
  for g in opt.param_groups:g['lr']=.005*(.5**e)
  if stale>=3:break
 del net,opt,dl;net=init();opt=torch.optim.Adam(net.parameters(),lr=.005);dl=loader(cut)
 for e in range(best_epoch):
  epoch(net,opt,dl)
  for g in opt.param_groups:g['lr']=.005*(.5**e)
 torch.save(net.state_dict(),out/'dlinear.pth');net.eval();x,y=windows(z,ts)
 with torch.no_grad():dp=net(torch.from_numpy(x).cuda()).cpu().numpy()
 np.save(out/'dlinear_predictions.npy',dp);save_json(out/'dlinear_development_epochs.json',records);del net,opt,dl;gc.collect();torch.cuda.empty_cache();torch.manual_seed(2021);torch.cuda.manual_seed_all(2021)
 pipe=Chronos2Pipeline.from_pretrained(str(RUN/'models/chronos2_29ec3766'),device_map='cuda:0');trained=pipe.fit([torch.from_numpy(z[:cut].T.copy())],prediction_length=64,validation_inputs=None,finetune_mode='full',context_length=512,learning_rate=5e-7,num_steps=500,batch_size=112,min_past=512,output_dir=out/'chronos',callbacks=[Deadline()],seed=2021,data_seed=2021,disable_tqdm=True,logging_steps=100,bf16=True,tf32=False);del pipe,trained;gc.collect();torch.cuda.empty_cache();pipe=Chronos2Pipeline.from_pretrained(str(out/'chronos/finetuned-ckpt'),device_map='cuda:0');pipe.model.eval();qs=[]
 for i in range(0,len(ts),16):
  guard(900);q,_=pipe.predict_quantiles([torch.from_numpy(v.T.copy()) for v in x[i:i+16]],prediction_length=64,batch_size=112,context_length=512,cross_learning=False);qs.extend([v.cpu().numpy().transpose(1,0,2) for v in q])
 fq=np.stack(qs).astype(np.float32);fp=fq.mean(-1);np.save(out/'ft_predictions.npy',fp);np.save(out/'ft_quantiles.npy',fq);del pipe;gc.collect();torch.cuda.empty_cache();ca=np.load(RUN/'cache/dense_chronos_train.npz');ta=np.load(RUN/'cache/dense_timesfm_train.npz');lookup={int(v):i for i,v in enumerate(ca['origins'])};ids=[lookup[int(t)] for t in ts];q=np.stack([ca['quantiles'][ids],ta['quantiles'][ids]],1);mp,nn=predict_prefix(z,ts,q,dp,'both',cut);assert all(max(v)+64<=cut for v in nn);np.save(out/'memory_predictions.npy',mp);np.save(out/'targets.npy',y);np.save(out/'query_origins.npy',ts);np.save(out/'history_features.npy',np.stack([descriptor(v) for v in x]));save_json(out/'neighbor_origins.json',nn);cfg['dlinear_selected_epoch']=best_epoch;save_json(out/'config.json',cfg);metrics={'dlinear':metric(dp,y),'finetuned':metric(fp,y),'memory':metric(mp,y),'fixed70':metric((.7*fp+.3*mp).astype(np.float32),y)};save_json(out/'proxy_metrics.json',{'training_query_only':True,'metrics':metrics,'elapsed_seconds':time.time()-start,'not_official_validation':True});print(name,'DLinear epochs',best_epoch,{k:v['mse'] for k,v in metrics.items()},flush=True)
if __name__=='__main__':run(int(sys.argv[1]))
