"""Compact LLM tool router with explicit no-LLM fallback and frozen training memory."""
from common import *
from concurrent.futures import ThreadPoolExecutor,as_completed
import sys
TOOLS=['dlinear','patchtst','season7','chronos2']
KNOWLEDGE='ETTh1 hourly measurements. HUFL/HULL are High Useful/Useless Load, MUFL/MULL Middle, LUFL/LULL Low, OT oil temperature. Exact units, topology, and time zone unknown. Training raw HUFL approximately MUFL+LUFL, HULL approximately MULL+LULL; these are empirical relationships, not hard physical laws and not directly additive in z scores. All inputs/forecasts here use fixed training z scores; negative values are valid even in raw data. All six loads were simultaneously zero for 58 training hours while OT changed; cause unknown. HUFL/MUFL stronger daily patterns than LUFL; OT slow thermal inertia is only a soft prior, environment/cooling unobserved. Contemporaneous correlations are not causal future information. Do not impose monotonicity, positivity, fixed daily peaks, or guessed power units. Use the actual recent evidence.'
def descriptor(x):
 feats=[]
 for k in (24,96,512):feats.extend([x[-k:].mean(0),x[-k:].std(0)])
 feats.append(x[-24:].mean(0)-x[-48:-24].mean(0))
 return np.concatenate(feats).astype(np.float32)
def retrieve(x,t,bank,k=3):
 valid=np.where(bank['origins']+64<=t-512)[0]
 if not len(valid):return []
 d=np.mean(((bank['features'][valid]-descriptor(x))/bank['feature_scale'])**2,axis=1);ix=valid[np.argsort(d)[:k]]
 return [{'training_origin':int(bank['origins'][i]),'distance':round(float(np.mean(((bank['features'][i]-descriptor(x))/bank['feature_scale'])**2)),3),'tool_mse_by_channel':bank['mse'][i].round(3).tolist(),'tool_bias_by_channel':bank['bias'][i].round(3).tolist()} for i in ix]
def payload(x,p,t,level,bank):
 hist={str(k):{'mean':x[-k:].mean(0).round(3).tolist(),'std':x[-k:].std(0).round(3).tolist()} for k in (24,96,512)}
 hist['daily_change']=(x[-24:].mean(0)-x[-48:-24].mean(0)).round(3).tolist()
 hist['acf24']=[round(float(np.corrcoef(x[:-24,c],x[24:,c])[0,1]),3) if x[:,c].std()>1e-8 else 0 for c in range(7)]
 forecasts={m:{'mean':p[j].mean(0).round(3).tolist(),'std':p[j].std(0).round(3).tolist(),'first8_mean':p[j,:8].mean(0).round(3).tolist(),'last8_mean':p[j,-8:].mean(0).round(3).tolist()} for j,m in enumerate(TOOLS)}
 obj={'columns':COLS,'tools':TOOLS,'observed_history_statistics':hist,'future64_tool_forecasts_summary':forecasts}
 if level>=2:obj['past_training_error_cases']=retrieve(x,t,bank);obj['memory_caveat']='DLinear/PatchTST errors in this bank are in-sample training; not unbiased performance estimates. Chronos is frozen pretrained; pretraining overlap unknown. Cases end at least512 hours before query. No query future provided.'
 system='Choose a convex mixture of numeric forecasting tools for each of7 variables to minimize future64 squared error. Return ONLY JSON {"weights":[[4 nonnegative numbers summing to1],...7 rows]}. Row order is columns and tool order is tools. Use modest diversification when evidence is uncertain. Do not fabricate observed future. Do not generate numeric forecasts or offsets. DLinear and PatchTST fit training; Chronos2 frozen pretrained multivariate; season7 repeats mean hourly profile of last7 days. History512 and all tool forecasts available only via summaries.'
 if level>=1:system+='\n'+KNOWLEDGE
 return [{'role':'system','content':system},{'role':'user','content':json.dumps(obj,separators=(',',':'))}]
def parse_weights(response):
 if response.get('status')!='ok':raise ValueError('API status')
 w=np.asarray(parse_json(response['content'])['weights'],dtype=np.float64)
 if w.shape!=(7,4) or not np.isfinite(w).all() or (w<0).any() or (w.sum(1)<=0).any():raise ValueError('invalid weights')
 return (w/w.sum(1,keepdims=True)).astype(np.float32)
def load_tools(split):
 s=np.load(ROOT/'runs/20261105_local_season7_mse'/f'{split}_predictions.npy')
 return np.stack([np.load(RUN/'cache'/f'{split}_{m}.npy') if m!='season7' else s for m in TOOLS],1)
def run(level,split='screen'):
 name=f'20261105_hybrid_level{level}{split}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True);(out/'requests').mkdir(exist_ok=True)
 subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True)
 revision=git_head();cfg={'candidate':name,'level':level,'split':split,'tools':TOOLS,'thinking':'disabled','max_tokens':800,'fallback_weights':[.5,0,0,.5],'code_revision':revision,'decision_cadence':24 if split=='val' else 64,'memory_sha256':hashlib.sha256((RUN/'cache/train_error_bank4.npz').read_bytes()).hexdigest()};save_json(out/'config.json',cfg)
 bank=dict(np.load(RUN/'cache/train_error_bank4.npz'));z=np.load(RUN/'cache/trainval_z.npy');manifest=json.loads((RUN/'cache/window_manifest.json').read_text());results={};start=time.time()
 for part in (('dev','confirm') if split=='screen' else ('val',)):
  ts=manifest[part];allp=load_tools(part);weights=np.tile([.5,0,0,.5],(len(ts),7,1)).astype(np.float32);decisions=list(range(0,len(ts),24)) if part=='val' else list(range(len(ts)));records=[]
  def one(i):
   guard(600);t=ts[i];messages=payload(z[t-512:t],allp[i],t,level,bank);prefix=f'{part}_{i:04d}';save_json(out/'requests'/(prefix+'.json'),messages)
   try:
    ans=api(messages,max_tokens=800);save_json(out/'requests'/(prefix+'_response.json'),ans);w=parse_weights(ans);status='ok'
   except Exception as e:w=np.tile([.5,0,0,.5],(7,1));status='fallback_'+type(e).__name__
   return i,w,status
  with ThreadPoolExecutor(max_workers=4) as pool:
   for f in as_completed([pool.submit(one,i) for i in decisions]):
    i,w,status=f.result();end=min(i+(24 if part=='val' else 1),len(ts));weights[i:end]=w;records.append({'index':i,'origin':ts[i],'status':status,'applied_until_index':end-1});print(name,part,i,status,flush=True)
  p=np.einsum('nmhc,ncm->nhc',allp,weights).astype(np.float32);y=np.load(RUN/'cache'/f'{part}_targets.npy');results[part]=metric(p,y);results[part]['fallback_decisions']=sum(r['status']!='ok' for r in records);results[part]['decisions']=len(records)
  np.save(out/f'{part}_predictions.npy',p);np.save(out/f'{part}_weights.npy',weights);save_json(out/f'{part}_decisions.json',records);save_json(out/'partial_metrics.json',results)
 record={'candidate':name,'metrics':results,'status':'evaluated','elapsed_seconds':time.time()-start,'code_revision':revision};save_json(out/'metrics.json',record)
 with (RUN/'experiment_index.jsonl').open('a') as f:f.write(json.dumps({'candidate':name,'metrics':{s:r['mse'] for s,r in results.items()},'status':'evaluated','kind':'llm_hybrid'})+'\n')
 print(name,{s:r['mse'] for s,r in results.items()},flush=True)
if __name__=='__main__':run(int(sys.argv[1]),sys.argv[2] if len(sys.argv)>2 else 'screen')
