"""Route tools using prefix-held-out training errors; matched numeric controls."""
from common import *
from hybrid_router import descriptor,KNOWLEDGE
from concurrent.futures import ThreadPoolExecutor,as_completed
import sys
CHOICES=np.array([0.,.25,.5,.75,1.])
BANK=RUN/'cache/prefix_tool_error_bank.npz'
def feature(x,ft,mem):
 return np.concatenate([descriptor(x),ft.mean(0),mem.mean(0),ft.std(0),mem.std(0),ft.mean(0)-mem.mean(0)])
def build():
 z=np.load(RUN/'cache/trainval_z.npy');parts=[]
 for cut in (3072,4096,6272):
  p=ROOT/'runs'/f'20261105_crossfit_prefix{cut}_mse';ts=np.load(p/'query_origins.npy');ft=np.load(p/'ft_predictions.npy');mem=np.load(p/'memory_predictions.npy');y=np.load(p/'targets.npy');assert np.all(ts-512>=cut) and np.all(ts+64<=8640)
  f=np.stack([feature(z[t-512:t],a,b) for t,a,b in zip(ts,ft,mem)]);mix=CHOICES[None,:,None,None]*ft[:,None]+(1-CHOICES[None,:,None,None])*mem[:,None];cost=((mix-y[:,None])**2).mean(2);parts.append({'origins':ts,'cutoffs':np.full(len(ts),cut),'features':f,'cost':cost,'ft':ft,'mem':mem,'targets':y})
 b={k:np.concatenate([p[k] for p in parts]) for k in parts[0]};assert len(b['origins'])==63;b['feature_scale']=np.maximum(b['features'][:34].std(0),.1);np.savez_compressed(BANK,**b);save_json(RUN/'cache/prefix_tool_error_bank.json',{'cases':63,'screen_memory':34,'screen_queries':29,'choices':CHOICES.tolist(),'feature_scale_fit':'first34 only','weights_never_fit_query_labels':True,'scaler':'official full-training scaler shared; no independent raw-data claim','external_pretraining_overlap':'unknown','transfer_limit':'prefix proxies differ from full-training tools; memory is evidence, not exact current risk','sha256':hashlib.sha256(BANK.read_bytes()).hexdigest()});print('PREFIX_BANK_COMPLETE',flush=True)
def retrieve(b,x,ft,mem,t,k):
 valid=np.where(b['origins']+64<=t-512)[0];d=np.mean(((b['features'][valid]-feature(x,ft,mem))/b['feature_scale'])**2,1);ii=np.argsort(d)[:k];ids=valid[ii];dist=d[ii];assert len(ids)==k;w=1/(1+dist);w/=w.sum();cost=np.einsum('n,nac->ac',w,b['cost'][ids]);teacher=CHOICES[cost.argmin(0)];return ids,dist,cost,teacher
def run(mode,part,k=5):
 assert mode in ('numeric','stats','cases','semantic') and part in ('screen','val');name=f'20261105_prefixstrict_{mode}_k{k}_{part}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True);(out/'requests').mkdir(exist_ok=True);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);b=dict(np.load(BANK));z=np.load(RUN/'cache/trainval_z.npy');mf=json.loads((RUN/'cache/window_manifest.json').read_text());split='confirm' if part=='screen' else 'val';ts=np.array(mf[split]);step=1 if part=='screen' else 24
 if part=='screen':
  ix=np.arange(34,63);assert np.array_equal(b['origins'][ix],ts);ft=b['ft'][ix];mem=b['mem'][ix];y=b['targets'][ix]
 else:ft=np.load(ROOT/'runs/20261105_reloadedft_lr5e7_mse/val_predictions.npy');mem=np.load(ROOT/'runs/20261105_enrichedmemory_both_mse/val_predictions.npy');y=np.load(RUN/'cache/val_targets.npy')
 if part=='screen':
  b={key:(val if key=='feature_scale' else val[:34]) for key,val in b.items()};assert max(b['origins']+64)<=min(ts)-512
 cfg={'screen_memory_frozen_cases':34,'engine':'prefix_tool_agent_strict_v2','mode':mode,'k':k,'part':part,'cadence':step,'choices':CHOICES.tolist(),'memory_sha256':hashlib.sha256(BANK.read_bytes()).hexdigest(),'code_revision':git_head(),'runtime_llm':mode!='numeric','screen_tool_weights':'prefix6272 only','validation_tool_weights':'official full training','transfer_caveat':True};save_json(out/'config.json',cfg);weights=np.full((len(ts),7),.5);records=[];start=time.time()
 def one(i):
  guard(1200);t=int(ts[i]);x=z[t-512:t];ids,dist,cost,teacher=retrieve(b,x,ft[i],mem[i],t,k);status='numeric';w=teacher
  if mode!='numeric':
   system='Select tool-mixture weights to minimize future64 squared error. Return ONLY JSON {"ft_weight":[7 values each exactly 0,.25,.5,.75,1]}. Prediction=w*finetuned+(1-w)*memory, columns HUFL,HULL,MUFL,MULL,LUFL,LULL,OT. Finetuned is Chronos2; memory combines DLinear and error-corrected frozen Chronos/TimesFM. Current future labels are never available. For supplied past cases, tool weights were fitted only before each case history began. Those are realized past errors, not errors on the current query. Proxy training sizes differ; nearest cases may fail to transfer. The program supplies exact comparative risks: do not invent new numerical risks. Prefer lower paired-case risk when credible, and moderate mixtures when evidence is ambiguous. Do not equate smoothness or physical plausibility with lower error. Val decisions held24 origins, forecasts updated hourly.'
   obj={'columns':COLS,'history_mean24_96_512':[x[-v:].mean(0).round(3).tolist() for v in (24,96,512)],'history_std':x.std(0).round(3).tolist(),'recent_daily_change':(x[-24:].mean(0)-x[-48:-24].mean(0)).round(3).tolist(),'forecast_8h_means':{'ft':ft[i].reshape(8,8,7).mean(1).round(3).tolist(),'memory':mem[i].reshape(8,8,7).mean(1).round(3).tolist()}}
   if mode in ('cases','semantic'):
    obj.update({'action_weights':CHOICES.tolist(),'weighted_case_mse_action_by_variable':cost.round(5).tolist(),'numeric_teacher_weights':teacher.tolist(),'past_cases':[{'origin':int(b['origins'][j]),'proxy_training_end':int(b['cutoffs'][j]),'distance':round(float(d),3),'mse_action_by_variable':b['cost'][j].round(5).tolist()} for j,d in zip(ids,dist)],'limitations':'No query labels in retrieval. Fixed official train scaler shared. External pretraining overlap unknown. Past proxy error need not equal current full-tool risk.'})
   if mode=='semantic':system+='\n'+KNOWLEDGE
   msgs=[{'role':'system','content':system},{'role':'user','content':json.dumps(obj,separators=(',',':'))}];prefix=f'{split}_{i:04d}';save_json(out/'requests'/(prefix+'.json'),msgs)
   try:
    ans=api(msgs,800);save_json(out/'requests'/(prefix+'_response.json'),ans);w=np.asarray(parse_json(ans['content'])['ft_weight'],float);assert w.shape==(7,) and np.isfinite(w).all() and np.isin(w,CHOICES).all();status='ok'
   except Exception as e:w=teacher if mode!='stats' else np.full(7,.5);status='fallback_'+type(e).__name__
  return i,w,{'index':i,'origin':t,'status':status,'neighbor_origins':b['origins'][ids].tolist(),'teacher_weights':teacher.tolist()}
 if mode=='numeric':res=map(one,range(0,len(ts),step))
 else:pool=ThreadPoolExecutor(max_workers=2);res=(f.result() for f in as_completed([pool.submit(one,i) for i in range(0,len(ts),step)]))
 for i,w,r in res:weights[i:i+step]=w;records.append(r);print(name,split,i,r['status'],flush=True)
 if mode!='numeric':pool.shutdown()
 p=(weights[:,None]*ft+(1-weights[:,None])*mem).astype(np.float32);m=metric(p,y);m['fallback_decisions']=sum(r['status'].startswith('fallback') for r in records);m['decisions']=len(records);np.save(out/f'{split}_predictions.npy',p);np.save(out/f'{split}_weights.npy',weights);save_json(out/f'{split}_decisions.json',records);save_json(out/'metrics.json',{'candidate':name,'metrics':{split:m},'status':'evaluated','elapsed_seconds':time.time()-start,'screen_uses_prefix_tools':part=='screen'});print(name,m['mse'],flush=True)
if __name__=='__main__':
 if sys.argv[1]=='build':build()
 else:run(sys.argv[1],sys.argv[2],int(sys.argv[3]) if len(sys.argv)>3 else 5)
