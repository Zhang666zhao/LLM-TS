"""LLM learns from its own completed training decisions, never current-query future."""
from common import *
from hybrid_router import descriptor,KNOWLEDGE
from concurrent.futures import ThreadPoolExecutor,as_completed
import sys
CHOICES=np.array([0,.25,.5,.75]);BANK=RUN/'cache/self_error_bank.npz';SUMMARY=RUN/'cache/self_error_rules.json'
def arrays(split):
 prior=sum(np.load(RUN/'cache'/f'{split}_{n}.npy') for n in ('dlinear','chronosavg','timesfmavg'))/3;corrected=np.load(ROOT/'runs/20261105_densememory_xbphase_mse'/f'{split}_predictions.npy');return prior,(corrected-prior)*4

def build():
 subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name','20261105_selfmemory_bank_mse'],check=True)
 z=np.load(RUN/'cache/trainval_z.npy');ts=np.array(json.loads((RUN/'cache/window_manifest.json').read_text())['dev']);assert ts.max()+64<=6208;x,y=windows(z[:8640],ts);p,u=arrays('dev');own=np.load(ROOT/'runs/20261105_mempolicy_l0_screen_think0_mse/dev_predictions.npy');weights=np.load(ROOT/'runs/20261105_mempolicy_l0_screen_think0_mse/dev_strength.npy');assert np.max(abs(own-(p+weights[:,None]*u)))<1e-6
 f=np.stack([np.concatenate([descriptor(a),b.mean(0),b.std(0)]) for a,b in zip(x,u)]);cost=np.stack([np.mean((p.astype(float)+v*u-y)**2,axis=1) for v in CHOICES],1);owncost=np.mean((own.astype(float)-y)**2,axis=1);best=CHOICES[cost.argmin(1)];np.savez(BANK,origins=ts,features=f,feature_scale=np.maximum(f.std(0),.1),cost=cost,own_cost=owncost,own_strength=weights,best_strength=best)
 cases=[{'origin':int(t),'history_mean':x[i].mean(0).round(3).tolist(),'history_std':x[i].std(0).round(3).tolist(),'recent_drift':(x[i,-24:].mean(0)-x[i,-48:-24].mean(0)).round(3).tolist(),'unit_correction_mean':u[i].mean(0).round(3).tolist(),'previous_chosen_strength':weights[i].tolist(),'realized_training_mse':owncost[i].round(4).tolist(),'best_strength_in_hindsight':best[i].tolist(),'mse_for_choices_by_variable':cost[i].round(4).tolist()} for i,t in enumerate(ts)]
 messages=[{'role':'system','content':'Summarize lessons from a forecasting agent own completed TRAINING decisions and counterfactual training errors. No validation or test outcomes are supplied. Return JSON {"rules":[{"condition":"...","action":"...","evidence_origins":[],"uncertainty":"..."}],"limitations":[]}. At most6 concise rules. Do not invent a causal law, numerical threshold unsupported by cases, or guaranteed generalization. Own policy was constant.25, numeric alternatives0,.25,.5,.75. Cases use fully training-fitted DLinear, so error comparisons are training feedback, not unbiased generalization evidence. Hindsight best actions are labels for past cases, never current-query truth. Correction may be too strong, too weak, or wrong in sign. Explicitly retain contradictory cases rather than overgeneralizing.'},{'role':'user','content':json.dumps({'columns':COLS,'choices':CHOICES.tolist(),'cases':cases},separators=(',',':'))}];save_json(RUN/'cache/self_error_summary_request.json',messages);ans=api(messages,2200);save_json(SUMMARY,ans)
 if ans.get('status')!='ok':raise RuntimeError('Summary API failure; no silent alternative')
 obj=parse_json(ans['content']);assert isinstance(obj.get('rules'),list);save_json(RUN/'cache/self_error_rules_parsed.json',obj);print('SELF_MEMORY_BUILT',len(ts),flush=True)

def run(mode,part):
 name=f'20261105_selfagent_{mode}_{part}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True);(out/'requests').mkdir(exist_ok=True);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);bank=dict(np.load(BANK));rules=json.loads((RUN/'cache/self_error_rules_audited.json').read_text());z=np.load(RUN/'cache/trainval_z.npy');manifest=json.loads((RUN/'cache/window_manifest.json').read_text());cfg={'engine':'self_error_agent_v1','mode':mode,'part':part,'choices':CHOICES.tolist(),'default':.25,'memory_rows':'25 completed dev queries ending<=6208','query_labels_never_visible':True,'thinking':'disabled','bank_sha256':hashlib.sha256(BANK.read_bytes()).hexdigest(),'rules_sha256':hashlib.sha256((RUN/'cache/self_error_rules_audited.json').read_bytes()).hexdigest(),'code_revision':git_head()};save_json(out/'config.json',cfg);results={};start=time.time()
 for split in (('confirm',) if part=='screen' else ('val',)):
  ts=manifest[split];p,u=arrays(split);alpha=np.full((len(ts),7),.25);step=24 if split=='val' else 1;records=[]
  def one(i):
   guard(900);t=ts[i];x=z[t-512:t];f=np.concatenate([descriptor(x),u[i].mean(0),u[i].std(0)]);valid=np.where(bank['origins']+64<=t-512)[0];assert len(valid)>=3;dist=np.mean(((bank['features'][valid]-f)/bank['feature_scale'])**2,1);ids=valid[np.argsort(dist)[:3]]
   info={'columns':COLS,'history_mean24_96_512':[x[-k:].mean(0).round(3).tolist() for k in (24,96,512)],'history_std':x.std(0).round(3).tolist(),'recent_drift':(x[-24:].mean(0)-x[-48:-24].mean(0)).round(3).tolist(),'current_unit_correction_mean':u[i].mean(0).round(3).tolist(),'current_unit_correction_std':u[i].std(0).round(3).tolist()}
   if mode in ('rules','both'):info['lessons_from_own_training_errors']=rules
   if mode in ('cases','both'):info['own_past_training_error_cases']=[{'origin':int(bank['origins'][j]),'distance':round(float(np.mean(((bank['features'][j]-f)/bank['feature_scale'])**2)),3),'previous_chosen_strength':bank['own_strength'][j].tolist(),'realized_training_mse':bank['own_cost'][j].round(4).tolist(),'counterfactual_mse_for_choices':bank['cost'][j].round(4).tolist(),'best_past_strength_in_hindsight':bank['best_strength'][j].tolist()} for j in ids]
   system='You are correcting your own forecasting policy using remembered TRAINING mistakes. Return ONLY JSON {"strength":[7 numbers each exactly one of0,.25,.5,.75]}. Forecast=equal DLinear/Chronosavg/TimesFMavg prior + strength*fixed dense-memory correction. All forecasts update hourly, your strength is held24 origins on validation. Memory contains previous completed training predictions and their realized errors; all cases end before current history begins. You never see current future or current error. Choose a strength after considering contradictory analogs and state mismatch. Do not copy a past hindsight label as if it were current truth. These cases use full-training-fitted DLinear, so their errors may be optimistic and need not transfer. Keep .25 when no reliable evidence exists. Positive correction raises forecasts. Past best is computed among only the four choices, not an oracle for current time.'
   if mode=='both':system+='\n'+KNOWLEDGE
   msgs=[{'role':'system','content':system},{'role':'user','content':json.dumps(info,ensure_ascii=False,separators=(',',':'))}];prefix=f'{split}_{i:04d}';save_json(out/'requests'/(prefix+'.json'),msgs)
   try:
    ans=api(msgs,500);save_json(out/'requests'/(prefix+'_response.json'),ans);a=np.asarray(parse_json(ans['content'])['strength'],float);assert a.shape==(7,) and np.isin(a,CHOICES).all();status='ok'
   except Exception as e:a=np.full(7,.25);status='fallback_'+type(e).__name__
   return i,a,status
  with ThreadPoolExecutor(max_workers=2) as pool:
   for f in as_completed([pool.submit(one,i) for i in range(0,len(ts),step)]):
    i,a,status=f.result();alpha[i:i+step]=a;records.append({'index':i,'origin':ts[i],'status':status});print(name,split,i,status,flush=True)
  pr=(p+alpha[:,None]*u).astype(np.float32);m=metric(pr,np.load(RUN/'cache'/f'{split}_targets.npy'));m['fallback_decisions']=sum(r['status']!='ok' for r in records);results[split]=m;np.save(out/f'{split}_predictions.npy',pr);np.save(out/f'{split}_strength.npy',alpha);save_json(out/f'{split}_decisions.json',records)
 save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated','elapsed_seconds':time.time()-start});print(name,{s:m['mse'] for s,m in results.items()},flush=True)
if __name__=='__main__':
 if sys.argv[1]=='build':build()
 else:run(sys.argv[1],sys.argv[2])
