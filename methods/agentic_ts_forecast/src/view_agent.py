"""Bounded online choices among strong view tools, with chronological error-case RAG."""
from common import *
from view_routing import TOOLS,hindcasts
from view_memory_gate import features
from hybrid_router import KNOWLEDGE
from concurrent.futures import ThreadPoolExecutor,as_completed
import sys
PRIOR=np.array([0.,.4,.6]);ALPHA=.25

def memory(part):
 data=np.load(RUN/'cache/view_prefix_memory.npz');keep=data['fold']<6272 if part=='screen' else np.ones(len(data['fold']),bool);d={k:data[k][keep] for k in data.files};d['scale']=np.maximum(d['features'].std(0),.1);return d

def packet(x,p,past,t,bank,k):
 f=features(x[None],p[None])[0];allowed=np.where(bank['origins']+64<=t-512)[0];dist=np.mean(((bank['features'][allowed]-f)/bank['scale'])**2,axis=1);chosen=[]
 for idx in allowed[np.argsort(dist,kind='stable')]:
  if all(abs(int(bank['origins'][idx])-int(bank['origins'][j]))>=64 for j in chosen):chosen.append(int(idx))
  if len(chosen)==k:break
 assert len(chosen)==k and all(bank['origins'][j]+64<=t-512 for j in chosen)
 err=bank['targets'][chosen,None]-bank['predictions'][chosen];cost=(err.astype(float)**2).mean(2);mean_cost=cost.mean(0);v=np.exp(-mean_cost/np.maximum(mean_cost.mean(0,keepdims=True),1e-6));teacher=v/v.sum(0,keepdims=True);obs=x[-64:][None]-past
 obj={'columns':COLS,'tool_order':['semantic_view','pca_view','random_view'],'fixed_prior_weights':PRIOR.tolist(),'history_mean24_96_512':[x[-n:].mean(0).round(3).tolist() for n in (24,96,512)],'history_std':x.std(0).round(3).tolist(),'change24':(x[-24:].mean(0)-x[-48:-24].mean(0)).round(3).tolist(),'forecast_8h_block_means':p.reshape(3,8,8,7).mean(2).round(3).tolist(),'observed_hindcast_mse':(obs.astype(float)**2).mean(1).round(3).tolist(),'observed_hindcast_early32_mse':(obs[:,:32].astype(float)**2).mean(1).round(3).tolist(),'observed_hindcast_late32_mse':(obs[:,32:].astype(float)**2).mean(1).round(3).tolist(),'scope':'Hindcasts use first448 of current512 to predict last64 ALREADY OBSERVED. They do not measure errors on the current future64.'}
 cases=[]
 for jj,j in enumerate(chosen):cases.append({'origin':int(bank['origins'][j]),'prefix_weight_fit_cutoff':int(bank['fold'][j]),'feature_distance':round(float(np.mean(((bank['features'][j]-f)/bank['scale'])**2)),4),'future64_mse_by_tool_then_variable':cost[jj].round(3).tolist(),'mean_error_actual_minus_prediction':err[jj].mean(1).round(3).tolist(),'best_tool_index_by_variable':cost[jj].argmin(0).tolist()})
 return obj,cases,teacher,chosen

def choose(x,p,past,t,bank,mode,k,out,prefix,replay_from=None):
 obj,cases,teacher,chosen=packet(x,p,past,t,bank,k)
 if mode=='numeric':return ((1-ALPHA)*PRIOR[:,None]+ALPHA*teacher),'numeric',chosen
 system='Choose a conservative mixture of THREE forecast tools. Output ONLY JSON {"weights":[[3 numbers for HUFL],[3 for HULL],[3 for MUFL],[3 for MULL],[3 for LUFL],[3 for LULL],[3 for OT]]}. Each row nonnegative and sums to1. Tool order semantic_view,pca_view,random_view. Actual final weights=.75*[0,.4,.6]+.25*your row. You may return the prior [0,.4,.6] for any variable. All tools are Chronos2 fine-tuned on training-only data in different INVERTIBLE coordinate systems; all7 dimensions retained. No physical identity is imposed. Their names do not imply predictive superiority. The prior was selected adaptively on official validation; this screen is diagnostic, not an independent test. Forecasts update hourly; your weights persist24 rolling origins on validation/test. Retrospective hindcast ranking need not persist into future64. Do not invent future outcomes or prefer smoothness automatically. The objective is all-variable standardized MSE.'
 if mode in ('semantic','cases'):system+='\n'+KNOWLEDGE
 if mode=='cases':obj['past_error_cases']=cases;obj['case_caveat']='Each case was predicted by models fitted BEFORE the entire query history. Cases are official training examples, with labels ending before current support starts. Prefix-trained tools differ from full-data tools. Similar history is not proof of similar future errors. These are TOOL mistakes, not current future labels.'
 messages=[{'role':'system','content':system},{'role':'user','content':json.dumps(obj,ensure_ascii=False,separators=(',',':'))}];out=pathlib.Path(out);out.mkdir(exist_ok=True,parents=True);request=out/(prefix+'_request.json');response=out/(prefix+'_response.json');status='ok'
 if replay_from is not None:
  original=pathlib.Path(replay_from)/(prefix+'_request.json');assert json.loads(original.read_text())==messages,'Replayed runtime evidence differs from original request'
  ans=json.loads((pathlib.Path(replay_from)/(prefix+'_response.json')).read_text())
 else:
  save_json(request,messages)
  try:ans=api(messages,1300)
  except Exception as e:ans={'status':'client_exception','exception':type(e).__name__}
  save_json(response,ans)
 try:
  assert ans.get('status')=='ok';raw=np.asarray(parse_json(ans['content'])['weights'],float);assert raw.shape==(7,3) and np.isfinite(raw).all() and (raw>=0).all() and (raw<=1).all() and np.all(abs(raw.sum(1)-1)<.001);raw=raw/raw.sum(1,keepdims=True);w=(1-ALPHA)*PRIOR[:,None]+ALPHA*raw.T
 except Exception as e:w=np.repeat(PRIOR[:,None],7,axis=1);status='fallback_'+type(e).__name__
 return w,status,chosen

def run(mode,part,k=3):
 assert mode in ('stats','semantic','cases','numeric') and part in ('screen','val');name=f'20261105_viewagent_{mode}_k{k}_{part}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);guard(2400);z=np.load(RUN/'cache/trainval_z.npy');mf=json.loads((RUN/'cache/window_manifest.json').read_text());split='confirm' if part=='screen' else 'val';ts=mf[split];step=1 if part=='screen' else 24;bank=memory(part);isnum=mode=='numeric'
 if part=='screen':
  full=np.load(RUN/'cache/view_prefix_memory.npz');lookup={int(t):i for i,t in enumerate(full['origins'])};ii=[lookup[int(t)] for t in ts];assert (full['fold'][ii]==6272).all();p=full['predictions'][ii];assert np.array_equal(full['targets'][ii],np.load(RUN/'cache/confirm_targets.npy'));past=hindcasts(z,ts,RUN/'view_context_prefix6272_screen',tools=[f'20261105_prefixview_6272_{v}_mse' for v in ('semantic','pca','random')])
 else:p=np.stack([np.load(ROOT/'runs'/n/'val_predictions.npy') for n in TOOLS],1);past=np.stack([np.load(RUN/'view_context_cache/val'/(n+'.npy')) for n in TOOLS],1)
 cfg={'engine':'view_agent_v1','mode':mode,'part':part,'k':k,'cadence':step,'prior':PRIOR.tolist(),'alpha':ALPHA,'source_ids':['run:'+n for n in TOOLS],'runtime_llm':not isnum,'training_memory':str(RUN/'cache/view_prefix_memory.npz'),'training_memory_sha256':hashlib.sha256((RUN/'cache/view_prefix_memory.npz').read_bytes()).hexdigest(),'code_revision':git_head(),'screen_caveat':'Prefix-fitted query tools and first258 frozen cases, but fixed prior chosen on official validation; no independent test claim'};cfg['query_tool_source_ids']=['run:'+f'20261105_prefixview_6272_{v}_mse' for v in ('semantic','pca','random')] if part=='screen' else cfg['source_ids'];save_json(out/'config.json',cfg);ww=np.empty((len(ts),3,7));records=[];start=time.time()
 def one(i):guard(1800);w,status,chosen=choose(z[ts[i]-512:ts[i]],p[i],past[i],ts[i],bank,mode,k,out/'requests',f'{split}_{i:04d}');return i,w,status,chosen
 if isnum:res=map(one,range(0,len(ts),step))
 else:pool=ThreadPoolExecutor(max_workers=2);res=(f.result() for f in as_completed([pool.submit(one,i) for i in range(0,len(ts),step)]))
 for i,w,status,chosen in res:ww[i:i+step]=w;records.append({'index':i,'origin':ts[i],'status':status,'memory_origins':[int(bank['origins'][j]) for j in chosen]});print(name,i,status,flush=True)
 if not isnum:pool.shutdown()
 pr=(p*ww[:,:,None]).sum(1).astype(np.float32);m=metric(pr,np.load(RUN/'cache'/f'{split}_targets.npy'));m.update(fallback_decisions=sum('fallback' in v['status'] for v in records),decisions=len(records));np.save(out/f'{split}_predictions.npy',pr);np.save(out/f'{split}_weights.npy',ww);save_json(out/f'{split}_decisions.json',records);save_json(out/'metrics.json',{'candidate':name,'metrics':{split:m},'status':'evaluated','elapsed_seconds':time.time()-start});print(name,m['mse'],flush=True)
if __name__=='__main__':run(sys.argv[1],sys.argv[2],int(sys.argv[3]) if len(sys.argv)>3 else 3)
