"""Bounded actor/critic decisions using errors already observable inside512."""
from common import *
from hybrid_router import KNOWLEDGE
from observed_feedback import TOOLS,PARENT,correction
from concurrent.futures import ThreadPoolExecutor,as_completed
import sys
ACTIONS=np.array([0.,.05,.1,.25])
def evidence(x,prior,past):
 e=x[-64:][None]-past;combined=.7*e[0]+.3*e[1]
 return {'columns':COLS,'history_mean24_96_512':[x[-k:].mean(0).round(4).tolist() for k in (24,96,512)],'history_std':x.std(0).round(4).tolist(),'daily_change':(x[-24:].mean(0)-x[-48:-24].mean(0)).round(4).tolist(),'prior_8h_block_means':prior.reshape(8,8,7).mean(1).round(4).tolist(),'observed_hindcast_mse_by_tool':(e.astype(float)**2).mean(1).round(4).tolist(),'observed_hindcast_bias_by_tool_actual_minus_predicted':e.mean(1).round(4).tolist(),'proxy_8h_residual_means':combined.reshape(8,8,7).mean(1).round(4).tolist(),'proxy_early32_bias':combined[:32].mean(0).round(4).tolist(),'proxy_late32_bias':combined[32:].mean(0).round(4).tolist(),'proxy_bias_std':combined.std(0).round(4).tolist(),'warning':'These are retrospective errors on already observed history, not current future errors. Hindcast448 input differs from512 forecast. The proxy tool mixture is not identical to the full prior.'}
def action(x,prior,past,mode,out,prefix):
 obj=evidence(x,prior,past);system='You decide a bounded per-variable correction to an audited forecasting prior. Output ONLY JSON {"strength":[7 values each exactly0,.05,.1,.25]}. Order HUFL,HULL,MUFL,MULL,LUFL,LULL,OT. Forecast=prior+strength*correction. correction is .7*FT1e6+.3*calendarFT5e-7 observed hindcast mean residual (actual-prediction), clipped to +/-.5 historical std and tapered1→.5 over64 future hours. The hindcast uses ONLY first448 of current512 to predict last64 already observed. No additional history or query-future truth. Default strength.1; choose0 when retrospective bias is likely noise or regime-specific. Larger strength needs stable agreement across tools and early/late observed periods; mean bias alone is insufficient. Do not invent current future errors or assume residual persistence. Decisions held24 rolling origins during validation; forecasts update hourly. You may keep.1 for all variables.'
 if mode in ('semantic','critic'):system+='\n'+KNOWLEDGE
 messages=[{'role':'system','content':system},{'role':'user','content':json.dumps(obj,separators=(',',':'))}];save_json(out/(prefix+'_actor_request.json'),messages);status='ok'
 def parse(ans):
  assert ans.get('status')=='ok';a=np.asarray(parse_json(ans['content'])['strength'],float);assert a.shape==(7,) and np.isfinite(a).all() and np.isin(a,ACTIONS).all();return a
 try:
  ans=api(messages,650);save_json(out/(prefix+'_actor_response.json'),ans);a=parse(ans)
 except Exception as e:a=np.full(7,.1);status='actor_fallback_'+type(e).__name__
 if mode=='critic':
  critique=[{'role':'system','content':system+' Review the proposed action below. Correct unsupported confidence, confusion between observed and future errors, contradictory residual signs, or hard physical assumptions. Do not change coefficients merely to be different. Output only the final strengthJSON; never fabricate error improvements.'},{'role':'user','content':json.dumps({'evidence':obj,'actor_strength':a.tolist()},separators=(',',':'))}];save_json(out/(prefix+'_critic_request.json'),critique)
  try:
   ans=api(critique,650);save_json(out/(prefix+'_critic_response.json'),ans);a=parse(ans)
  except Exception as e:status+=';critic_fallback_'+type(e).__name__
 return a,status
def numeric(x,past,mode):
 e=x[-64:][None]-past;v=.7*e[0]+.3*e[1];early=v[:32].mean(0);late=v[32:].mean(0);agree=(e[0].mean(0)*e[1].mean(0)>0)&(early*late>0)
 if mode=='agree10':return .1*agree
 if mode=='agree25':return .25*agree
 blocks=v.reshape(8,8,7).mean(1);b=blocks.mean(0);return .1*np.abs(b)/np.maximum(np.sqrt(b*b+blocks.var(0)),1e-6)
def run(mode,part):
 isnum=mode in ('agree10','agree25','signal10');assert isnum or mode in ('stats','semantic','critic');name=f'20261105_livefeedback_{mode}_{part}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True);req=out/'requests';req.mkdir(exist_ok=True);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);z=np.load(RUN/'cache/trainval_z.npy');mf=json.loads((RUN/'cache/window_manifest.json').read_text());split='confirm' if part=='screen' else 'val';ts=mf[split];step=1 if part=='screen' else 24;prior=np.load(ROOT/'runs'/PARENT/f'{split}_predictions.npy');past=np.stack([np.load(RUN/'strong_context_cache'/split/(n+'.npy')) for n in TOOLS],1);cor=np.load(RUN/'cache'/f'{split}_strong_observed_correction.npy');weights=np.full((len(ts),7),.1);records=[];start=time.time();save_json(out/'config.json',{'engine':'live_feedback_agent_v1','mode':mode,'part':part,'runtime_llm':not isnum,'actions':ACTIONS.tolist(),'cadence':step,'source_ids':['run:'+PARENT]+['run:'+n for n in TOOLS],'code_revision':git_head(),'training_screen_fullfit_caveat':True})
 def one(i):
  guard(1200);x=z[ts[i]-512:ts[i]]
  if isnum:w=numeric(x,past[i],mode);status='numeric'
  else:w,status=action(x,prior[i],past[i],mode,req,f'{split}_{i:04d}')
  return i,w,status
 if isnum:res=map(one,range(0,len(ts),step))
 else:pool=ThreadPoolExecutor(max_workers=2);res=(f.result() for f in as_completed([pool.submit(one,i) for i in range(0,len(ts),step)]))
 for i,w,status in res:weights[i:i+step]=w;records.append({'index':i,'origin':ts[i],'status':status});print(name,split,i,status,flush=True)
 if not isnum:pool.shutdown()
 p=(prior+weights[:,None]*cor).astype(np.float32);m=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'));m['fallback_decisions']=sum('fallback' in r['status'] for r in records);m['decisions']=len(records);np.save(out/f'{split}_predictions.npy',p);np.save(out/f'{split}_strength.npy',weights);save_json(out/f'{split}_decisions.json',records);save_json(out/'metrics.json',{'candidate':name,'metrics':{split:m},'status':'evaluated','elapsed_seconds':time.time()-start});print(name,m['mse'],flush=True)
if __name__=='__main__':run(sys.argv[1],sys.argv[2])
