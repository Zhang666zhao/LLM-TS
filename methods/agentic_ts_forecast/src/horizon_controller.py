"""Three-proposal/reflection cycles for small horizon-dependent convex policies."""
from common import *
from horizon_policy import canonical,predict,KNOTS
from blend_sources import get_source
import sys,copy
phase=sys.argv[1];batches=int(sys.argv[2]);out=RUN/'controller'/phase;out.mkdir(exist_ok=False,parents=True)
registry=['run:20261105_policy_p11b002c2_mse','run:20261105_transformedcalft_random_mse','run:20261105_prefixtree_cdf_d4_mse','run:20261105_rotationseed_2024_mse']
if len(sys.argv)>4:
 registry=json.loads(pathlib.Path(sys.argv[4]).read_text());assert len(registry)==4 and all(k.startswith('run:') for k in registry)
p=np.stack([get_source(k,'val') for k in registry],1);y=np.load(RUN/'cache/val_targets.npy');default={'source_ids':registry,'knot_weights':[[1,0,0,0]]*4,'variable_weights':{}};assert np.array_equal(predict(p,default),p[:,0]);rng=np.random.default_rng(2021);toy=rng.normal(size=(3,4,64,7)).astype('float32');trial=canonical(dict(source_ids=registry,knot_weights=rng.random((4,4)).tolist(),variable_weights={}),registry);a=predict(toy,trial);b=toy.copy();b[2]+=10;assert np.array_equal(a[:2],predict(b,trial)[:2]);assert np.isfinite(a).all();assert (a>=toy.min(1)-1e-6).all() and (a<=toy.max(1)+1e-6).all();save_json(out/'contracts.json',{'constant_exact':True,'query_isolation':True,'convex_envelope':True,'test_access':False})
scores=[]
for i,k in enumerate(registry):
 e=(p[:,i].astype(float)-y)**2;scores.append({'id':k,'mse':float(e.mean()),'variable_by_16step_bin_mse':{col:[float(e[:,b:b+16,j].mean()) for b in range(0,64,16)] for j,col in enumerate(COLS)}})
bin_ranks={col:[[registry[i] for i in sorted(range(4),key=lambda i:scores[i]['variable_by_16step_bin_mse'][col][b])] for b in range(4)] for col in COLS}
evidence_mode=sys.argv[5] if len(sys.argv)>5 else 'stats';assert evidence_mode in ('stats','cov','verified');error_evidence={};identity_evidence={};research_memory=[]
if evidence_mode=='verified':
 memory_path=pathlib.Path(sys.argv[6]);research_memory=json.loads(memory_path.read_text());save_json(out/'research_memory_snapshot.json',dict(source=str(memory_path),sha256=hashlib.sha256(memory_path.read_bytes()).hexdigest(),content=research_memory))
if evidence_mode in ('cov','verified'):
 for j,col in enumerate(COLS):
  e=p[:,:,:,j].astype(float)-y[:,None,:,j].astype(float);blocks=[]
  for b in range(0,64,16):
   eb=e[:,:,b:b+16].transpose(1,0,2).reshape(4,-1);gram=eb@eb.T/eb.shape[1];norm=np.sqrt(np.maximum(np.diag(gram),1e-20));blocks.append(dict(uncentered_error_cosine=np.round(gram/norm[:,None]/norm[None,:],6).tolist(),mean_error=np.round(eb.mean(1),6).tolist()))
  error_evidence[col]=blocks
if evidence_mode=='verified':
 for j,col in enumerate(COLS):
  blocks=[]
  for b in range(0,64,16):
   maxdiff=np.array([[np.max(np.abs(p[:,i,b:b+16,j].astype(float)-p[:,k,b:b+16,j].astype(float))) for k in range(4)] for i in range(4)])
   blocks.append(dict(exact_prediction_equal=(maxdiff==0).tolist(),max_abs_prediction_difference=np.round(maxdiff,6).tolist()))
  identity_evidence[col]=blocks
 save_json(out/'identity_evidence.json',identity_evidence)
save_json(out/'evidence_mode.json',dict(mode=evidence_mode,scope='validation aggregate statistics only for offline architecture selection; never runtime query features or model-fitting targets',error_evidence=error_evidence))
best=float(((p[:,0].astype(float)-y)**2).mean());bestid=registry[0][4:];champion=default;history=[];seen=set();del p,y
if len(sys.argv)>3 and sys.argv[3]!='none':
 prior=RUN/'controller'/sys.argv[3];a=json.loads((prior/'champion.json').read_text());champion=canonical(a['config'],registry);best=a['mse'];bestid=a['candidate'];history=json.loads((prior/'history.json').read_text());seen={json.dumps(r['policy'],sort_keys=True) for r in history};save_json(out/'continuation.json',dict(prior_phase=sys.argv[3],prior_completed=len(history),prior_failure_preserved=(prior/'STOPPED.json').exists(),new_deadline=False))
prediction_seen={hashlib.sha256(np.load(ROOT/'runs'/registry[0][4:]/'val_predictions.npy').tobytes()).hexdigest():registry[0][4:]}
for previous in history:
 ph=hashlib.sha256(np.load(ROOT/'runs'/previous['id']/'val_predictions.npy').tobytes()).hexdigest();prediction_seen.setdefault(ph,previous['id'])
for batch in range(1,batches+1):
 guard(4200);packet={'sources':scores,'verified_research_memory':research_memory,'validation_prediction_identity_by_16step_bin':identity_evidence,'identity_scope':'Equality diagnostics are for observed validation forecasts, not an assertion of identity on all possible inputs. A weight change between exactly equal predictions cannot improve that validation block.','mandatory_constant_control_clarification':'Candidate1 should use identical four rows for default AND EVERY override; a constant default with varying LUFL/OT override is NOT a constant control. Sources themselves may contain horizon-varying logic.','additivity_fact':'For this typed operator, source predictions are fixed and each output variable is blended independently. Per-variable SSE changes add exactly to macroMSE; no cross-variable synergy exists in the outer operator.','evidence_mode':evidence_mode,'validation_aggregate_error_evidence':error_evidence,'error_evidence_interpretation':'When present, each matrix compares4source residuals in one16step bin on fullvalidation. Uncentered cosine1 means aligned errors, NOTproofidenticalpredictions; near1stillmaydiffer. Meanerrors are aggregate diagnostics, NOTpermitted runtime bias corrections. Propose onlytypedconvex weights, no coefficient solver or training onvalidation. Do notinvent correlation evidence if empty.','program_sorted_bin_ranks_best_to_worst':bin_ranks,'source_order':registry,'incumbent':{'id':bestid,'mse':best,'policy':champion},'recent_results':history[-6:],'knots':KNOTS,'columns':COLS,'semantics':'Four knots in horizon index 0..63. Each knot has four nonnegative source weights normalized to1. Linear interpolation between knots, constant endpoint extension. variable_weights optional map from column to4x4 override. Default knot_weights applies all other variables. No input-dependent rules. One of3proposals MUST have identical four rows for ALL profiles (constant across horizon) as a matched control. Prefer isolated small changes and <=2 variable overrides. Existing sources already include nested conditional policies and train-only memories. No future measurements. Only known prediction distance used. Source0 current best; others diverse standalone tools.','execution_reminder':'knot_weights is the DEFAULT FOR EVERY COLUMN, not just the listed overrides. An override identical to default changes nothing and is removed. Inspect actual_changed_variables and duplicate_prediction_of in previous results. Do not invent previous experiments when recent_results is empty. Compare each claimed change to reference_id and reference_mse, not a different implicit baseline.', 'limits':'These are adaptively reused2817validationwindows. No test access. No independence/significance/optimality claims, no physical causality from empirical differences. Repeated deterministic scoring doesnot estimate uncertainty. No learned coefficient optimizer onvalidation. Earlier LLM reflections reversed comparisons (0.6055 called lower than0.6012), confused alternative proposals with incremental additions, and declared thresholdoptimality without evidence. Use program_sorted_bin_ranks and citeIDs; avoid unsupported arithmetic. Strong constant LUFL controls later beat earlier horizonvarying profiles; horizon benefit mustbe compared against adequateconstantweights, not onlyweak0or.25controls. Propose small bounded recipes, avoid copying row-wise minimum as if guaranteed optimal because correlated errors matter.'}
 messages=[{'role':'system','content':'Return JSON reflection and exactly3 experiments. Each experiment {hypothesis,config:{knot_weights:4x4 nonnegative numbers,variable_weights:{optional column:4x4}}}. No source_ids in config. One constant-across-horizon candidate required. Reflect on previous3 using program evidence, then propose3distinct policies. No code.'},{'role':'user','content':json.dumps(packet,separators=(',',':'))}];save_json(out/f'batch{batch:03d}_request.json',messages);ans=api(messages,4500);save_json(out/f'batch{batch:03d}_response.json',ans)
 try:
  obj=parse_json(ans['content']);exps=obj['experiments'];assert len(exps)==3;cs=[canonical(dict(e['config'],source_ids=registry),registry) for e in exps];assert any(all(np.allclose(np.asarray(a),np.asarray(a)[0]) for a in [c['knot_weights'],*c['variable_weights'].values()]) for c in cs)
 except Exception as e:save_json(out/'STOPPED.json',{'batch':batch,'reason':repr(e),'stage':'proposal_schema'});raise
 save_json(out/f'batch{batch:03d}_reflection.json',obj.get('reflection',{}));rows=[]
 for j,(e,c) in enumerate(zip(exps,cs)):
  key=json.dumps(c,sort_keys=True);duplicate=key in seen;seen.add(key);name=f'20261105_horizon_{phase}b{batch:03d}c{j+1}_mse';save_json(out/f'{name}_proposal.json',dict(proposal_origin='llm',proposal=e,executed=c,duplicate_policy=duplicate))
  with (out/f'{name}.log').open('w') as log:r=subprocess.run([sys.executable,str(ROOT/'src/launch_snapshot.py'),'horizon_policy.py',json.dumps(c),name],stdout=log,stderr=subprocess.STDOUT)
  if r.returncode:save_json(out/'STOPPED.json',dict(candidate=name,exit_code=r.returncode));raise SystemExit(r.returncode)
  m=json.loads((ROOT/'runs'/name/'metrics.json').read_text())['metrics']['val'];row=dict(id=name,policy=c,hypothesis=e['hypothesis'],mse=m['mse'],per_variable_mse=m['per_variable_mse'],reference_id=bestid,reference_mse=best,delta=m['mse']-best,duplicate_policy=duplicate)
  prediction=np.load(ROOT/'runs'/name/'val_predictions.npy');reference_prediction=np.load(ROOT/'runs'/bestid/'val_predictions.npy');changes=np.max(np.abs(prediction.astype(float)-reference_prediction),axis=(0,1));ph=hashlib.sha256(prediction.tobytes()).hexdigest();row.update(actual_changed_variables=[col for col,diff in zip(COLS,changes) if diff>0],max_abs_prediction_change_by_variable=dict(zip(COLS,changes.tolist())),duplicate_prediction_of=prediction_seen.get(ph));prediction_seen.setdefault(ph,name)
  if m['mse']<best:best=m['mse'];bestid=name;champion=copy.deepcopy(c)
  history.append(row);rows.append(row);save_json(out/'history.json',history);save_json(out/'champion.json',dict(candidate=bestid,mse=best,config=champion));print(name,m['mse'],'best',best,flush=True)
 save_json(out/f'batch{batch:03d}_evidence.json',rows)
 if batch==batches:
  ans=api([{'role':'system','content':'Reflect on these3observedvalidationexperiments. No test/causal/significance/optimalityclaims. Deterministic duplicates are not independent trials. ReturnJSON findings,limitations,next hypotheses.'},{'role':'user','content':json.dumps(rows)}],2000);save_json(out/'final_batch_reflection.json',ans)
save_json(out/'status.json',dict(stage='complete',batches=batches,best_candidate=bestid,best_mse=best));print('HORIZON_PHASE_COMPLETE',phase,best,flush=True)
