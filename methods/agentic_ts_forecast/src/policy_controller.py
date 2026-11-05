"""LLM proposes a bounded, inspectable decision policy; program executes and scores."""
from common import *
from conditional_policy import canonical,features,FEATURE_BOUNDS,predict
from blend_sources import get_source
import sys,copy,random
phase=sys.argv[1];batches=int(sys.argv[2]);out=RUN/'controller'/phase;out.mkdir(parents=True,exist_ok=True)
cal=[]
for tag in ('lr5e7','lr1e6','lr2e7'):
 name='20261105_chronoscalft_'+tag+'_mse';a=json.loads((ROOT/'runs'/name/'metrics.json').read_text());cal.append((a['metrics']['val']['mse'],name))
registry=['run:20261105_auto_p01b001c1_mse','run:20261105_enrichedmemory_both_mse','run:20261105_reloadedft_lr1e6_mse','run:'+min(cal)[1]];default={'source_ids':registry,'base_weights':[.15,.30,.55,0.],'rules':[]};champion=copy.deepcopy(default);history=[];seen=set();rng=random.Random(2021+sum(map(ord,phase)));z=np.load(RUN/'cache/trainval_z.npy');manifest=json.loads((RUN/'cache/window_manifest.json').read_text());x,_=windows(z,manifest['val']);preds=np.stack([get_source(k,'val') for k in registry],1);target=np.load(RUN/'cache/val_targets.npy');base,_=predict(x,preds,default);expected=np.load(ROOT/'runs/20261105_auto_p03b006c2_mse/val_predictions.npy');delta=float(np.max(abs(base-expected)));assert delta<1e-6,('constant-policy contract',delta);reference=metric(base,target);best=reference['mse'];bestid='20261105_auto_p03b006c2_mse';source_scores=[{'id':k,**metric(preds[:,i],target)} for i,k in enumerate(registry)];save_json(out/'constant_policy_contract.json',{'max_delta':delta,'mse':best,'no_test_access':True})
for i,m in enumerate(source_scores):
 if m['mse']<best:
  champion={'source_ids':registry,'base_weights':[float(j==i) for j in range(4)],'rules':[]};best=m['mse'];reference=m;bestid=m['id'].split(':',1)[1]
train_x,_=windows(z,manifest['dev']+manifest['confirm']);train_p=np.stack([np.concatenate([get_source(k,'dev'),get_source(k,'confirm')]) for k in registry],1);f=features(train_x,train_p);feature_quantiles={k:np.quantile(v,[.1,.25,.5,.75,.9],axis=0).round(3).tolist() for k,v in f.items()};del x,preds,train_x,train_p

def key(c):return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
def leaf_diff(a,b,path='policy'):
 if isinstance(a,dict) and isinstance(b,dict):
  return [d for k in sorted(set(a)|set(b)) for d in leaf_diff(a.get(k),b.get(k),path+'.'+k)]
 if isinstance(a,list) and isinstance(b,list) and len(a)==len(b):
  return [d for i,(x,y) in enumerate(zip(a,b)) for d in leaf_diff(x,y,path+'['+str(i)+']')]
 return [] if a==b else [{'path':path,'reference':a,'candidate':b}]

def fallback(j):
 c=copy.deepcopy(champion)
 if j%3==0:
  w=np.asarray(c['base_weights']);w[rng.randrange(4)]+=rng.uniform(.05,.3);c['base_weights']=w.tolist()
 else:
  feature=rng.choice(list(FEATURE_BOUNDS));lo,hi=FEATURE_BOUNDS[feature];threshold=float(np.clip(np.median(f[feature])+rng.uniform(-.2,.2),lo,hi));w=np.asarray(c['base_weights']);w[rng.randrange(4)]+=rng.uniform(.1,.5);rule={'feature':feature,'op':rng.choice(['<','>']),'threshold':threshold,'columns':COLS if j%3==1 else [rng.choice(COLS)],'weights':w.tolist()}
  if len(c['rules'])>=3:c['rules'][rng.randrange(3)]=rule
  else:c['rules'].append(rule)
 return canonical(c,registry)

for batch in range(1,batches+1):
 guard(4200);packet={'objective':'minimize full validation MSE within original deadline; no threshold early stop','source_order':registry,'sources':[{'id':a['id'],'mse':a['mse'],'per_variable_mse':a['per_variable_mse']} for a in source_scores],'incumbent':{'id':bestid,'mse':best,'per_variable_mse':reference['per_variable_mse'],'policy':champion},'recent_results':history[-9:],'feature_bounds':FEATURE_BOUNDS,'training_feature_quantiles':{'probabilities':[.1,.25,.5,.75,.9],'columns':COLS,'values':feature_quantiles},'policy_semantics':'base_weights4 nonnegative numbers normalized. Up to3 rules, first matching rule wins separately pervariable. Rule fields feature,op(< or >),threshold,columns(nonempty subset),weights4 nonnegative numbers. All features use current512 history and tool forecasts only, never current future labels. Policies run numerically every rolling origin; no runtime LLM call. ft_mean_gap compares source2 to mean ofsources0/1. expert_disagreement is std ofsource forecast means /history std. Output is convex mixture, no arbitrary offsets or code.','knowledge':'Hourly record phase is known, timezone/topology/units unconfirmed. Daily pattern strength differs byvariable; negatives valid; H-M-L relation onlyempirical. Higher moments+cross-variable retrieval together helped, correlations alone hurt. Previous in-memory FT45 scores quarantined; only audited eval-mode sources here. No causal conclusions from train vsval gap. These are adaptive same-validation results, not independenttest. Old outer reflections sometimes confused units or attributed multi-change comparisons to onefactor. Use precomputed deltas and recursive leaf differences. A previous reflection falsely inferred OT-HULL coupling because it missed a reverted HULL threshold relative to sequential incumbent. Per-variable disjoint rules cannot interact across variables. Top-level rules field can contain multiple edits. Cite IDs, label mechanism as hypothesis.','budget_remaining_seconds':budget()['deadline_epoch']-time.time()}
 messages=[{'role':'system','content':'Propose exactly3 distinct small conditional forecasting policies. Return JSON {"reflection":{"evidence_ids":[],"findings":[],"hypotheses":[],"next_direction":"..."},"experiments":[{"hypothesis":"...","config":{"base_weights":[4 numbers],"rules":[{"feature":"...","op":">","threshold":0,"columns":["HUFL"],"weights":[4 numbers]}]}}]}. Source IDs are fixed and inserted byprogram; do notreturn them. Prefer isolated edits; one broaderalternative allowed. Include at least one comparison that can reveal whether a rule helps beyond its constant mixture. Do not restate numerical arithmetic: cite supplied IDs and computed directions. Do not claim significance, causal proof, test outcomes or private data. No code, onlyschema.'},{'role':'user','content':json.dumps(packet,separators=(',',':'))}];save_json(out/f'batch{batch:03d}_request.json',messages);save_json(out/'status.json',{'stage':'proposing','batch':batch,'epoch':time.time()});ans=api(messages,3600);save_json(out/f'batch{batch:03d}_response.json',ans)
 try:obj=parse_json(ans['content']);proposals=obj['experiments'];assert len(proposals)==3
 except Exception as e:obj={'reflection':{'error':type(e).__name__}};proposals=[]
 save_json(out/f'batch{batch:03d}_reflection.json',obj.get('reflection',{}));completed=[]
 for j in range(3):
  guard(4200);origin='llm';reason='';original=proposals[j] if j<len(proposals) else None
  try:
   assert set(original['config'])=={'base_weights','rules'};c=canonical(dict(original['config'],source_ids=registry),registry);hyp=original['hypothesis']
  except Exception as e:c=fallback(j);hyp='Deterministic fallback exploration';origin='deterministic_fallback';reason=type(e).__name__
  tries=0
  while key(c) in seen:
   tries+=1;c=fallback(j+tries);origin='deterministic_fallback';reason='duplicate policy';hyp='Deterministic fallback after duplicate'
   if tries>50:raise RuntimeError('Unique policy exhausted')
  seen.add(key(c));name=f'20261105_policy_{phase}b{batch:03d}c{j+1}_mse';save_json(out/f'{name}_proposal.json',{'original_llm_proposal':original,'executed_policy':c,'proposal_origin':origin,'fallback_reason':reason});save_json(out/'status.json',{'stage':'executing','batch':batch,'candidate':name,'epoch':time.time()})
  with (out/f'{name}.log').open('w') as log:r=subprocess.run([sys.executable,str(ROOT/'src/launch_snapshot.py'),'conditional_policy.py',json.dumps(c),name],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
  if r.returncode:save_json(out/'STOPPED.json',{'candidate':name,'exit_code':r.returncode});raise SystemExit(r.returncode)
  m=json.loads((ROOT/'runs'/name/'metrics.json').read_text())['metrics']['val'];changed=[k for k in ('base_weights','rules') if c[k]!=champion[k]];row={'id':name,'hypothesis':hyp,'proposal_origin':origin,'policy':c,'validation_mse':m['mse'],'per_variable_mse':m['per_variable_mse'],'reference':{'id':bestid,'mse':best,'policy':champion,'per_variable_mse':reference['per_variable_mse']},'actual_changed_fields':changed,'recursive_leaf_differences':leaf_diff(champion,c),'attribution_warning':'Reference is sequential incumbent, not necessarily batch proposal source. Multiple entries inside rules can change; use recursive differences. Disjoint variable rules do not cause cross-variable coupling.','absolute_mse_delta':m['mse']-best,'relative_change_percent':100*(m['mse']/best-1),'direction':'improved' if m['mse']<best else 'worsened_or_equal','per_variable_deltas':{k:m['per_variable_mse'][k]-reference['per_variable_mse'][k] for k in COLS},'causal_limit':'multiple fields changed; no isolated mechanism inference' if len(changed)>1 else 'same-validation association only'}
  if m['mse']<best:best=m['mse'];champion=c;reference=m;bestid=name
  history.append(row);completed.append(row);save_json(out/'history.json',history);save_json(out/'champion.json',{'candidate':bestid,'mse':best,'config':champion,'phase':phase,'updated_epoch':time.time()});print(name,m['mse'],'best',best,flush=True)
 save_json(out/f'batch{batch:03d}_evidence.json',{'completed_exactly_three':completed,'best_mse':best,'best_candidate':bestid,'next_requires_reflection':True})
 if batch==batches:
  ans=api([{'role':'system','content':'Reflect on these three experiments. CiteIDs and programcomputed direction/deltas. Distinguish onefield versusmultifield changes. No new arithmetic, inventedcorrelations, causalclaims or testoutcomes. ReturnJSON findings,uncertainties,next hypotheses.'},{'role':'user','content':json.dumps(completed,separators=(',',':'))}],1800);save_json(out/'final_batch_reflection.json',ans)
save_json(out/'status.json',{'stage':'complete','batches':batches,'best_mse':best,'best_candidate':bestid,'epoch':time.time()});print('POLICY_PHASE_COMPLETE',phase,best,flush=True)
