"""Bounded LLM choices around a validated numeric prior, causal history evidence."""
from common import *
from hybrid_router import KNOWLEDGE,retrieve,descriptor
from concurrent.futures import ThreadPoolExecutor,as_completed
import sys

def run(level,split):
 name=f'20261105_reasonlow_level{level}{split}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True);(out/'requests').mkdir(exist_ok=True)
 subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True)
 rev=git_head();cfg={'level':level,'split':split,'tools':['dlinear','chronos2'],'prior_chronos_weight':.5,'weight_bounds':[.35,.65],'bias_bounds':[0,.5],'fallback_bias_strength':.25,'reasoning_effort':'low','thinking':'enabled','max_tokens':7000,'temperature_note':'provider may ignore temperature in thinking mode','code_revision':rev,'decision_every24_for_val':True,'memory_sha256':hashlib.sha256((RUN/'cache/train_error_bank4.npz').read_bytes()).hexdigest()};save_json(out/'config.json',cfg)
 z=np.load(RUN/'cache/trainval_z.npy');manifest=json.loads((RUN/'cache/window_manifest.json').read_text());bank=dict(np.load(RUN/'cache/train_error_bank4.npz'));results={};start=time.time()
 for part in (('dev','confirm') if split=='screen' else ('val',)):
  ts=manifest[part];a=dict(np.load(RUN/'cache'/f'{part}_context_backtest.npz'));d=np.load(RUN/'cache'/f'{part}_dlinear.npy');c=np.load(RUN/'cache'/f'{part}_chronos2.npy');w=np.full((len(ts),7),.5);beta=np.full((len(ts),7),.25);records=[];step=24 if part=='val' else 1
  def one(i):
   guard(600);t=ts[i];x=z[t-512:t];system='Choose conservative forecasting corrections around a strong equal DLinear/Chronos2 mixture. Return ONLY JSON {"chronos_weight":[7 numbers between .35 and .65],"bias_strength":[7 numbers between0 and.5]}. Column order '+','.join(COLS)+'. Forecast is (1-w)*DLinear+w*Chronos, minus bias_strength*((1-w)*recent_DLinear_bias+w*recent_Chronos_bias), with correction decaying1 to.5 over64 hours. Bias means prediction minus actual for already observed historical64 hours. Default weights .5, bias_strength .25. Deviate only when evidence supports it; a single historical64 backtest has high sampling noise. These are numerical policy parameters, not observations. No query future is provided. The prior is fixed before this query. The diagnostic backtest uses448 actual history hours; DLinear repeats the first observation64 times to fill512, Chronos uses448. Do not overinterpret padding-mismatched hindcasts. Current forecasts use the true512.'
   if level>=1:system+='\n'+KNOWLEDGE
   info={'history_means24_96_512':[x[-k:].mean(0).round(3).tolist() for k in (24,96,512)],'history_std':x.std(0).round(3).tolist(),'trend24':(x[-24:].mean(0)-x[-48:-24].mean(0)).round(3).tolist(),'observed_backtest_tool_order':['dlinear','chronos2'],'observed_backtest_mse':a['mse'][i].round(3).tolist(),'observed_backtest_bias':a['bias'][i].round(3).tolist(),'future_forecast_mean_difference_chronos_minus_dlinear':(c[i]-d[i]).mean(0).round(3).tolist()}
   if level>=2:
    cases=retrieve(x,t,bank);info['training_memory_chronos_only']=[{'origin':v['training_origin'],'distance':v['distance'],'chronos_mse':v['tool_mse_by_channel'][3],'chronos_bias':v['tool_bias_by_channel'][3]} for v in cases];info['memory_scope']='Only pretrained Chronos errors shown to avoid in-sample trained-tool ranking bias; training cases end before query support. Training-memory residual may not transfer.'
   messages=[{'role':'system','content':system},{'role':'user','content':json.dumps(info,separators=(',',':'))}];prefix=f'{part}_{i:04d}';save_json(out/'requests'/(prefix+'.json'),messages)
   try:
    ans=api(messages,7000,thinking="enabled",reasoning_effort="low");save_json(out/'requests'/(prefix+'_response.json'),ans)
    if ans['status']!='ok':raise ValueError('API error')
    obj=parse_json(ans['content']);cw=np.asarray(obj['chronos_weight'],float);b=np.asarray(obj['bias_strength'],float)
    if cw.shape!=(7,) or b.shape!=(7,) or not np.isfinite(cw).all() or not np.isfinite(b).all():raise ValueError('shape/finite')
    status='ok' if ((cw>=.35)&(cw<=.65)).all() and ((b>=0)&(b<=.5)).all() else 'clipped_bounds'
    return i,np.clip(cw,.35,.65),np.clip(b,0,.5),status
   except Exception as e:return i,np.full(7,.5),np.full(7,.25),'fallback_'+type(e).__name__
  with ThreadPoolExecutor(max_workers=4) as pool:
   for f in as_completed([pool.submit(one,i) for i in range(0,len(ts),step)]):
    i,cw,b,status=f.result();w[i:i+step]=cw;beta[i:i+step]=b;records.append({'index':i,'origin':ts[i],'status':status});print(name,part,i,status,flush=True)
  p=(1-w[:,None])*d+w[:,None]*c;b=(1-w)*a['bias'][:,0]+w*a['bias'][:,1];p-=beta[:,None]*b[:,None]*np.linspace(1,.5,64)[None,:,None];p=p.astype(np.float32)
  results[part]=metric(p,np.load(RUN/'cache'/f'{part}_targets.npy'));results[part]['fallback_decisions']=sum(r['status'].startswith('fallback') for r in records);results[part]['clipped_decisions']=sum(r['status']=='clipped_bounds' for r in records)
  np.save(out/f'{part}_predictions.npy',p);np.savez(out/f'{part}_policy.npz',chronos_weight=w,bias_strength=beta);save_json(out/f'{part}_decisions.json',records);save_json(out/'partial_metrics.json',results)
 save_json(out/'metrics.json',{'candidate':name,'metrics':results,'code_revision':rev,'status':'evaluated','elapsed_seconds':time.time()-start})
 with (RUN/'experiment_index.jsonl').open('a') as f:f.write(json.dumps({'candidate':name,'metrics':{s:r['mse'] for s,r in results.items()},'status':'evaluated','kind':'constrained_llm_agent'})+'\n')
 print(name,{s:r['mse'] for s,r in results.items()},flush=True)
if __name__=='__main__':run(int(sys.argv[1]),sys.argv[2])
