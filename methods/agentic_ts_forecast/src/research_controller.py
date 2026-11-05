"""Bounded schema-only LLM AutoResearch batches: three runs then explicit reflection."""
from common import *
import sys,copy,random
from blend_sources import get_source
PHASE=sys.argv[1];BATCHES=int(sys.argv[2]);OUT=RUN/'controller'/PHASE;OUT.mkdir(parents=True,exist_ok=True)
invalid=set(json.loads((RUN/'invalid_candidates.json').read_text()).get('candidates',[])) if (RUN/'invalid_candidates.json').exists() else set()
REGISTRY=['cache:dlinear','cache:chronosavg','cache:timesfmavg','cache:chronosuavg','cache:timesfmuavg']
valid=[]
for path in (ROOT/'runs').glob('20261105_*/metrics.json'):
 name=path.parent.name
 if name in invalid or any(t in name for t in ('hybrid_','constrained_','reasonlow_','mempolicy_','strongagent_','selfagent_')) or not all((path.parent/(s+'_predictions.npy')).exists() for s in ('dev','confirm','val')):continue
 try:
  a=json.loads(path.read_text());m=a.get('metrics',a)
  cp=path.parent/'config.json'
  if cp.exists() and json.loads(cp.read_text()).get('runtime_llm',False):continue
  if m['val']['windows']==2817:valid.append((m['val']['mse'],name))
 except (KeyError,TypeError):continue
valid.sort();chosen=[];prediction_hashes=set()
for _,name in valid:
 digest=hashlib.sha256((ROOT/'runs'/name/'val_predictions.npy').read_bytes()).hexdigest()
 if digest not in prediction_hashes:chosen.append(name);prediction_hashes.add(digest)
 if len(chosen)>=5:break
REGISTRY+=['run:'+name for name in chosen]
families=[('20261105_timesfmft_',),('20261105_rotationseed_',),('20261105_fullpoint_calendar',),('20261105_fullpoint_pca',),('20261105_fullpoint_semantic',),('20261105_fullpoint_lora',),('20261105_viewpoint_full',),('20261105_chronoslora_','20261105_reloadedft_lora_'),('20261105_transformed_semantic_',),('20261105_transformed_pca_',),('20261105_transformed_random_',),('20261105_transformedft_semantic_',),('20261105_transformedft_pca_',),('20261105_transformedft_random_',),('20261105_transformedmse_semantic_',),('20261105_transformedmse_pca_',),('20261105_transformedmse_random_',),('20261105_contextview_',),('20261105_temporal_',),('20261105_temporalft_',),('20261105_viewmemory_',),('20261105_chronoscalft_',),('20261105_reloadedft_lr',),('20261105_enrichedmemory_',)]
families += [(f'20261105_transformedcdfmse_{v}_',) for v in ('semantic','pca','random')]
families += [(f'20261105_fullpoint_mse_{v}_',) for v in ('semantic','pca','random')]
families += [(f'20261105_rotationseed_{v}_',) for v in (2022,2023,2024)]
families += [(f'20261105_legacy_{v}_',) for v in ('patchtst','itransformer','frets')]
families += [('20261105_timesfmft_head',),('20261105_timesfmft_full5',),('20261105_viewresidualcdf_',),('20261105_viewresidual_',),('20261105_prefixtree_',),('20261105_transformedcalft_',)]
for prefixes in families:
 matches=[name for _,name in valid if name.startswith(prefixes)]
 if matches:REGISTRY.append('run:'+matches[0])
for name in ('20261105_calendar_hour_mse','20261105_metagate_nonlinear_mse'):
 if (ROOT/'runs'/name/'val_predictions.npy').exists() and name not in invalid:REGISTRY.append('run:'+name)
REGISTRY=list(dict.fromkeys(REGISTRY))
DEFAULT={'sources':{REGISTRY[5]:1.},'amplitude':1.,'mean_shrink':0.,'smooth':1,'proxy_bias_strength':0.}
rng=random.Random(2021+sum(map(ord,PHASE)));history=[];seen=set();champion=copy.deepcopy(DEFAULT);best=valid[0][0];bestid=valid[0][1];reference=json.loads((ROOT/'runs'/bestid/'metrics.json').read_text())['metrics']['val']

def canonical(cfg):
 if not isinstance(cfg,dict) or set(cfg)-set(DEFAULT):raise ValueError('Unknown config fields')
 c=copy.deepcopy(DEFAULT);c.update(cfg);w=c['sources']
 if not isinstance(w,dict) or not w or len(w)>6 or any(k not in REGISTRY for k in w):raise ValueError('Unknown numerical tool')
 if any(not isinstance(v,(int,float)) or not np.isfinite(v) or v<0 for v in w.values()) or sum(w.values())<=0:raise ValueError('Invalid weights')
 total=sum(w.values());c['sources']={k:round(v/total,8) for k,v in sorted(w.items()) if v>0}
 for k,lo,hi in [('amplitude',.5,1.5),('mean_shrink',0,.5),('proxy_bias_strength',0,.5)]:
  if not isinstance(c[k],(int,float)) or not np.isfinite(c[k]) or not lo<=c[k]<=hi:raise ValueError('Out of bounds '+k)
 if c['smooth'] not in (1,3,5,7):raise ValueError('Invalid smoothing')
 return c

def key(c):return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
def leaf_diff(a,b,path='config'):
 if isinstance(a,dict) and isinstance(b,dict):return [d for k in sorted(set(a)|set(b)) for d in leaf_diff(a.get(k),b.get(k),path+'.'+k)]
 return [] if a==b else [{'path':path,'reference':a,'candidate':b}]
def fallback(j):
 c=copy.deepcopy(champion)
 if j%3==0:
  k=rng.choice(REGISTRY);c['sources'][k]=c['sources'].get(k,0)+rng.uniform(.05,.4)
  if len(c['sources'])>6:del c['sources'][min(c['sources'],key=c['sources'].get)]
 elif j%3==1:c['amplitude']=round(rng.uniform(.75,1.2),3)
 else:
  k=rng.choice(['mean_shrink','proxy_bias_strength']);c[k]=round(rng.uniform(0,.35),3)
 return canonical(c)

source_scores=[];errors=[];target=np.load(RUN/'cache/val_targets.npy')
for src in REGISTRY:
 p=get_source(src,'val');errors.append((p.astype(np.float64)-target).reshape(-1));m=metric(p,target);source_scores.append({'id':src,'split':'validation','mse':m['mse'],'per_variable_mse':m['per_variable_mse']})
per_variable_ranking={k:[{'id':v['id'],'mse':v['per_variable_mse'][k]} for v in sorted(source_scores,key=lambda v:v['per_variable_mse'][k])] for k in COLS}
residual_correlation=np.corrcoef(np.stack(errors)).round(4).tolist();del errors
for batch in range(1,BATCHES+1):
 guard(4200)
 packet={'objective':'minimize validation MSE; no preset threshold; original deadline fixed','baseline_validation_mse':.5520404025597623,'best_phase_mse':best,'best_phase_config':champion,'source_scores':source_scores,'validation_residual_correlation':{'order':REGISTRY,'matrix':residual_correlation,'meaning':'Pearson correlation of validation residuals; not causal and not independent holdout'},'recent_experiments':history[-12:],'all_tried_config_hashes_count':len(seen),'budget_remaining_seconds':budget()['deadline_epoch']-time.time(),'registry':REGISTRY,'allowed_config':{'sources':'nonnegative scalar weights for <=6 registry sources, normalized automatically','amplitude':[.5,1.5],'mean_shrink':[0,.5],'smooth':[1,3,5,7],'proxy_bias_strength':[0,.5]},'mechanisms':{'amplitude':'scale predicted trajectory around its own64-step mean','mean_shrink':'move predicted64 mean toward observed recent96 mean','smooth':'edge-padded moving average of predictions only','proxy_bias_strength':'subtract decaying mean of DLinear/Chronos hindcast bias from already observed last64; proxy may mismatch selected tools'},'knowledge':'Seven hourly variables; valid negatives, unknown exact units/topology. Daily structure differs by variable; fixed semantic load-balance projection worsened validation. Extra weekly covariates worsened hourly-only calendar. Old post-fit in-memory Chronos evaluation was non-replayable due omitted eval mode;45 affected scores and derived pipelines are quarantined and absent from this packet. Use only supplied current source scores. New reloadedft sources use unchanged saved weights, explicit eval, repeatability checks. Dense pretrained error memory with shape+forecast features and24-hour phase alignment improved its static parent .510843 to .508064 atstrength.25 and .506272 atstrength.75; do not assume correction transfers to a different parent. Training-error ranking of full-train fitted tools is biased; no training score versus validation score comparison. Runtime LLM routers did not beat numeric priors so far. No future measurements; deterministic calendar is permitted.'}
 packet['per_variable_ranking_lower_is_better']=per_variable_ranking;packet['best_candidate_id']=bestid;packet['knowledge']+=' New evidence: saved-LoRA adapter versus merged checkpoint agrees atFP32; Trainer residualautocast caninvalidate mixedprecisioncomparisons. Full2817 replay of observed-feedback numerical incumbent passed below1e-6. Frozen PCA-view tool is weaker standalone but a25percent mixture with that incumbent improvesvalidation to.4936732475. Randomrotation alsohelpsfrozenChronos, so do notclaimphysicalsemanticsalonecausedgain. Strictpastcase onlineprompts failedtobeatconstantblend; own-error rules/cases mostlyreturneddefault. Useactualleafdifferences: a previousreflection falselyinferred OT-HULLcoupling whenHULLthresholdwasreverted againstsequentialincumbent. Some sources alreadycontain observed-error corrections; additional proxycorrection maydoublecount and usesdifferentmodels.'
 packet['knowledge']+=' AUDIT CORRECTIONS: amplitude scaling around own64-step mean leaves that mean EXACTLY unchanged, so amplitude loss cannot prove mean calibration. High error correlation does not identify bias-versus-variance mechanism; no such attribution is established. Read program-sorted per-variable rankings: lower is better. In p05, PCA LUFL.58999 is WORSE than random.58592, while PCA OT.08746 is BETTER than random.08912. The p05 best.48731882095 uses40percent PCA and60percent random; these two standalone FT tools contain NO observed-error correction, so worse proxy correction there is NOT evidence of double-counting. First p05 batch was deterministic fallback after invalid JSON, not LLM-designed. Original-space point MSE models now supplied; compare actual scores, not presumed objective superiority. No causal or significant claims from one adaptive validation split.'
 packet['knowledge']+=' New matched point-extraction evidence: integrating all21 native Chronos quantiles with flat endpoint tails improved several fixed tools. The specialist core+LoRA+calendar integral combination scores .4814624002642113 versus prior p07 .48397973124629473 with identical rules and weights. No interval calibration claim. View-memory numerical gates and nearest-error softmin did not beat fixed40percentPCA/60percentrandom. Online LLM stats routing .48731456018 nearly equals its fixedprior .48731882095, not a substantial demonstrated gain. Different rotation seeds alter ONLY coordinate basis, training RNG fixed2021. TimesFM3 uses a separately audited training-only zero-variance gradient extension with EXACT forward equality; evaluation uses original public inference. No noise-floor or causal attribution without measured evidence.'
 packet['knowledge']+=' Further matched evidence: p09 static outer per-variable specialization retained old train-only residual correction on5 variables, removed it onLULL, used LoRA onOT, scoring .4787296181; extra outer conditional rules did not improve. Replacing PCA/random point-MSE tools with fullCDF-MSE counterparts in this fixed DAG scores .4779537413; training sample orders matched exactly. Point-only changes on oldMSE weights improved all3 views, but retrainedCDF models improved further. CDF-aligned residual memories did NOT beat old memory when inserted into the same strongDAG. Three static average-bias controls worsened the p08 parent while dynamic memory improved it. Three additional rotation bases were weaker standalone than original2021 but may diversify. Non-oracle historical PatchTST/iTransformer/FreTS predictions were independently recomputed with originalruntime and matched priorVALIDATION within1e-5; no oldTEST artifacts were used. They are weaker standalone, so demand measured mixture evidence before using them. All prefix ridge residual holdout diagnostics worsened despite adaptive official validation gains; do not claim cross-period robustness. Identical prediction aliases are not independent successes. No measured noise floor or MSE-resolution uncertainty is supplied.'
 system='Act as a careful research controller. You may propose ONLY the specified numeric config schema, never code or files. These are compiled LLM-designed numeric pipelines, not runtime LLM forecasts. All metrics supplied are same full validation split (2817 windows); not independent test results. Use program-supplied numbers and reference candidate IDs. Absolute MSE delta and relative change percent are DIFFERENT: negative is better. Do not recalculate or confuse units. Per-variable lower MSE is better; a worse individual model can still diversify residuals but this needs a matched mixture. Training-versus-validation differences cannot establish overfitting causally because distributions differ. Never claim significance from point estimates. Never invent significance, causal conclusions, or test outcomes. After reviewing the preceding three experiments, provide a detailed but concise reflection (evidence, failure mechanism hypothesis, what to retain/reject, next isolated change). Propose exactly3 distinct experiments, preferably one controlled change each, to seek smaller MSE; include one broader alternative when local changes are exhausted. Preserve negative outcomes. Return JSON {"reflection":{"evidence_ids":[],"findings":[],"next_direction":"..."},"experiments":[{"hypothesis":"...","config":{"sources":{},"amplitude":1,"mean_shrink":0,"smooth":1,"proxy_bias_strength":0}},...]}. Bounds are strict. Do not assert that semantic plausibility implies predictive gain.'
 save_json(OUT/'status.json',{'stage':'proposing','batch':batch,'updated_epoch':time.time()})
 messages=[{'role':'system','content':system},{'role':'user','content':json.dumps(packet,separators=(',',':'))}];save_json(OUT/f'batch{batch:03d}_request.json',messages)
 ans=api(messages,3600);save_json(OUT/f'batch{batch:03d}_response.json',ans)
 try:
  if ans['status']!='ok':raise ValueError('API proposal failure')
  obj=parse_json(ans['content']);proposals=obj['experiments'];assert len(proposals)==3
 except Exception as e:obj={'reflection':{'status':'proposal_invalid','error':type(e).__name__}};proposals=[]
 save_json(OUT/f'batch{batch:03d}_reflection.json',obj.get('reflection',{}))
 completed=[]
 save_json(OUT/'status.json',{'stage':'executing','batch':batch,'updated_epoch':time.time()})
 for j in range(3):
  guard(4200);reason='deterministic fallback after invalid/duplicate proposal';origin='llm';fallback_reason='';original=proposals[j] if j<len(proposals) else None
  try:c=canonical(original['config']);reason=original['hypothesis']
  except Exception as e:c=fallback(j);origin='deterministic_fallback';fallback_reason=type(e).__name__
  tries=0
  while key(c) in seen:
   tries+=1;c=fallback(j+tries);origin='deterministic_fallback';fallback_reason='duplicate policy';reason='Deterministic fallback after duplicate proposal'
   if tries>50:raise RuntimeError('Cannot generate unique candidate')
  seen.add(key(c));name=f'20261105_auto_{PHASE}b{batch:03d}c{j+1}_mse';cfgfile=OUT/f'{name}_proposal.json';save_json(cfgfile,{'hypothesis':reason,'config':c,'proposal_origin':origin,'fallback_reason':fallback_reason,'original_llm_proposal':original,'proposal_response_sha256':ans.get('request_sha256')})
  log=OUT/f'{name}.log'
  with log.open('w') as f:
   result=subprocess.run([sys.executable,str(ROOT/'src/launch_snapshot.py'),'blend_sources.py',json.dumps(c),name],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  if result.returncode:save_json(OUT/'STOPPED.json',{'candidate':name,'exit_code':result.returncode,'reason':'candidate contract/execution failure; diagnose, do not retry silently'});raise SystemExit(result.returncode)
  a=json.loads((ROOT/'runs'/name/'metrics.json').read_text());m=a['metrics'];row={'id':name,'hypothesis':reason,'config':c,'validation_mse':m['val']['mse'],'training_dev_mse':m['dev']['mse'],'training_confirm_mse':m['confirm']['mse'],'reference_validation_mse_before_candidate':best,'absolute_mse_delta':m['val']['mse']-best,'relative_change_percent':100*(m['val']['mse']/best-1),'per_variable_validation_mse':m['val']['per_variable_mse'],'training_caveat':'learned full-train source scores in-sample; not independent selection evidence'}
  row.update(proposal_origin=origin,reference={'id':bestid,'mse':best,'config':champion,'per_variable_mse':reference['per_variable_mse']},recursive_leaf_differences=leaf_diff(champion,c),per_variable_deltas={k:m['val']['per_variable_mse'][k]-reference['per_variable_mse'][k] for k in COLS},attribution_warning='Sequential incumbent reference can differ from batch proposal source; inspect actual differences, not prose hypothesis.')
  if m['val']['mse']<best:best=m['val']['mse'];champion=c;bestid=name;reference=m['val']
  history.append(row);completed.append(row);save_json(OUT/'history.json',history);save_json(OUT/'champion.json',{'candidate':bestid,'mse':best,'config':champion,'phase':PHASE,'updated_epoch':time.time()});print(name,m['val']['mse'],'best',best,flush=True)
 save_json(OUT/f'batch{batch:03d}_evidence.json',{'completed_exactly_three':completed,'best_mse':best,'best_config':champion,'next_batch_requires_reflection':True})
 # Last batch still requires a reflection, not merely an exit after three runs.
 if batch==BATCHES:
  msg=[{'role':'system','content':'Audit these three same-split validation experiments. Return concise JSON evidence IDs, findings, uncertainties, next hypotheses. No further execution requested. Do not compare in-sample training scores to validation or invent test outcomes. All2817 validation windows are identical across candidates. Predictions are fixed for scoring; no uncertainty/noise-floor estimate has been measured. Do not invent a validation resolution, significance, noise floor, or causal meaning from error correlation. Correlation .99 is very high, not moderate. Amplitude around own64 mean does not change that mean. Inspect provided leaf differences before attributing a delta.'},{'role':'user','content':json.dumps({'completed':completed,'per_variable_ranking_lower_is_better':per_variable_ranking,'source_scores':source_scores,'baseline_validation_mse':.5520404025597623,'validation_windows':2817})}];r=api(msg,1600);save_json(OUT/'final_batch_reflection.json',r)
save_json(OUT/'status.json',{'stage':'complete','batches':BATCHES,'updated_epoch':time.time(),'best_mse':best})
print('PHASE_COMPLETE',PHASE,best,flush=True)
