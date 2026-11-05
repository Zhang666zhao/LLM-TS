"""Discrete validation-selected expert columns; no validation coefficient fitting."""
from common import *
from blend_sources import get_source
import sys
STANDALONE={'analogue_transfer_v1','analogue_context_v1','semantic_ridge_v1','individual_dlinear_v1','transformed_tool_v1','reloaded_finetune_v1','point_mse_finetune_v1','lora_finetune_v1','calendar_finetune_v1','calendar_fullpoint_v1','legacy_numeric_tool_v1','timesfm_finetune_v1','variable_error_tree_v1','prefix_error_tree_v1','factor_error_tree_v1','static_memory_v1','seasonal_error_tree_v1','calendar_memory_gate_v1','calendar_delta_v1','asinh_tool_v1','view_residual_memory_v1'}
COMPOSED={'projected_composition_v1','block_column_selector_v1','blend_sources_v1','horizon_policy_v1','specialist_policy_v1','conditional_policy_v1','masked_residual_strength_v1','fixed_view_prior_v1','strong_view_feedback_v1','observed_feedback_v1','residual_transfer_v1','constant_residual_v1','column_selector_v1'}

def predict(p,cfg):
 assert p.ndim==4 and p.shape[2:]==(64,7);return np.stack([p[:,i,:,j] for j,i in enumerate(cfg['column_source_indices'])],-1).astype(np.float32)

def run(tag):
 guard(2400);out=RUN/'selection_pools'/tag;out.mkdir(parents=True,exist_ok=False);invalid=set(json.loads((RUN/'invalid_candidates.json').read_text())['candidates']);rows=[];excluded=[];memo={};runtime_names={}
 def cost(name,seen=None):
  if name in memo:return memo[name]
  seen=set() if seen is None else set(seen)
  if name in seen:raise RuntimeError('Cyclic source graph')
  seen.add(name);cfg=json.loads((ROOT/'runs'/name/'config.json').read_text());deps=list(cfg.get('source_ids',[]))+list(cfg.get('sources',{}));value=1+sum(cost(k[4:],seen) for k in set(deps) if k.startswith('run:'));runtime_names[name]=bool(cfg.get('runtime_llm',False)) or any(runtime_names.get(k[4:],False) for k in deps if k.startswith('run:'));memo[name]=value;return value
 for path in sorted((ROOT/'runs').glob('20261105_*/metrics.json')):
  name=path.parent.name
  if name in invalid:continue
  a=json.loads(path.read_text());m=a.get('metrics',a);v=m.get('val',{})
  if v.get('windows')!=2817 or not np.isfinite(v.get('mse',np.nan)):continue
  cp=path.parent/'config.json'
  if not cp.exists():continue
  cfg=json.loads(cp.read_text());engine=cfg.get('engine');ridge=name.startswith('20261105_ridge_') and cfg.get('features') in ('full','multiscale') and (path.parent/'weights.npy').exists()
  if cfg.get('runtime_llm',False) or not (engine in STANDALONE|COMPOSED or ridge):excluded.append(dict(candidate=name,reason='outside explicitly supported numeric recipe pool'));continue
  graph_cost=cost(name)
  if runtime_names.get(name,False):excluded.append(dict(candidate=name,reason='runtime LLM in dependency graph'));continue
  rows.append(dict(candidate=name,engine=engine or 'ridge_tool_v1',standalone=engine in STANDALONE or ridge,mse=v['mse'],per_variable_mse=v['per_variable_mse'],graph_cost=graph_cost,metrics_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),config_sha256=hashlib.sha256(cp.read_bytes()).hexdigest()))
 assert rows;save_json(out/'pool.json',dict(candidates=rows,excluded=excluded,excluded_quarantined=len(invalid),snapshot_epoch=time.time(),test_access=False,selection='discrete model selection using validation aggregate per-variable MSE; no continuous coefficient optimizer; exact ties prefer smaller expanded dependency count then lexical ID'));poolsha=hashlib.sha256((out/'pool.json').read_bytes()).hexdigest()
 def choose(pool):return [min(pool,key=lambda r:(r['per_variable_mse'][col],r['graph_cost'],r['candidate'])) for col in COLS]
 standalone=choose([r for r in rows if r['standalone']]);allrows=choose(rows);greedy=[min(rows,key=lambda r:(r['mse'],r['graph_cost'],r['candidate']))]
 for step in range(2):
  best=min(rows,key=lambda r:(sum(min([a['per_variable_mse'][col] for a in greedy]+[r['per_variable_mse'][col]]) for col in COLS),r['graph_cost'],r['candidate']))
  if best['candidate'] not in {r['candidate'] for r in greedy}:greedy.append(best)
 candidates={'standalone':standalone,'all':allrows,'greedy3':choose(greedy)}
 for mode,chosen in candidates.items():
  name=f'20261105_columnselect_{tag}_{mode}_mse';d=ROOT/'runs'/name;d.mkdir(exist_ok=False);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);ids=list(dict.fromkeys('run:'+r['candidate'] for r in chosen));indices=[ids.index('run:'+r['candidate']) for r in chosen];cfg=dict(engine='column_selector_v1',source_ids=ids,column_source_indices=indices,selected_by_variable={col:r['candidate'] for col,r in zip(COLS,chosen)},selection_pool=str(out/'pool.json'),selection_pool_sha256=poolsha,selection_mode=mode,pool_size=len(rows),runtime_llm=False,selection='discrete per-variable validation selection; all fitting performed in existing training-only tools; no test access',code_revision=git_head());save_json(d/'config.json',cfg);results={}
  for split in ('dev','confirm','val'):
   guard(1800);p=predict(np.stack([get_source(k,split) for k in ids],1),cfg)
   for j,r in enumerate(chosen):assert np.array_equal(p[:,:,j],get_source('run:'+r['candidate'],split)[:,:,j])
   np.save(d/f'{split}_predictions.npy',p);results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'))
  expected=np.mean([r['per_variable_mse'][col] for col,r in zip(COLS,chosen)]);assert abs(results['val']['mse']-expected)<1e-10;save_json(d/'metrics.json',dict(candidate=name,metrics=results,status='evaluated'));save_json(d/'contracts.json',dict(selected_columns_bitwise_exact=True,expected_validation_mse=float(expected),pool_frozen_before_all_three=True));print(name,results['val']['mse'],cfg['selected_by_variable'],flush=True)
if __name__=='__main__':run(sys.argv[1])
