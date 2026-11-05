"""Compile only required output columns through known linear composition operators."""
from common import *
from blend_sources import get_source
import sys
SUPPORTED={'horizon_policy_v1','column_selector_v1','block_column_selector_v1'}
def apply(p,c):
 op=c['operator'];cfg=c['operator_config']
 if op=='horizon_policy_v1':
  from horizon_policy import predict
 elif op=='column_selector_v1':
  from column_selection import predict
 elif op=='block_column_selector_v1':
  from block_column_selection import predict
 else:raise ValueError(op)
 full=np.zeros((len(p),c['original_slots'],64,7),np.float32)
 for j,slot in enumerate(c['source_slots']):full[:,slot]=p[:,j]
 result=predict(full,cfg);result[:,:,~np.asarray(c['required_columns'],bool)]=0;return result

def run(parent,tag):
 subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',f'20261105_projected_{tag}_compile'],check=True);guard(2400);memo={};audit=[];count=0;invalid=set(json.loads((RUN/'invalid_candidates.json').read_text())['candidates'])
 def compile_node(name,mask):
  nonlocal count
  assert name not in invalid,'Quarantined dependency'
  key=(name,tuple(mask))
  if key in memo:return memo[key]
  orig=ROOT/'runs'/name/'config.json';c=json.loads(orig.read_text());engine=c.get('engine')
  if engine not in SUPPORTED:memo[key]=name;return name
  slots=c['source_ids'];required=np.zeros((len(slots),7),bool)
  for j,col in enumerate(COLS):
   if not mask[j]:continue
   if engine=='horizon_policy_v1':required[:,j]=np.any(np.asarray(c['variable_weights'].get(col,c['knot_weights']))>0,axis=0)
   elif engine=='column_selector_v1':required[c['column_source_indices'][j],j]=True
   else:required[np.unique(np.asarray(c['lead_column_indices'])[:,j]),j]=True
  ids=[];positions=[]
  for slot,k in enumerate(slots):
   if required[slot].any():
    assert k.startswith('run:');ids.append('run:'+compile_node(k[4:],required[slot].tolist()));positions.append(slot)
  count+=1;new=f'20261105_projected_{tag}_node{count}';out=ROOT/'runs'/new;out.mkdir(exist_ok=False);opcfg=dict(c);opcfg.pop('source_ids',None);cfg=dict(engine='projected_composition_v1',operator=engine,operator_config=opcfg,original_slots=len(slots),source_slots=positions,source_ids=ids,required_columns=list(mask),compiled_from_config=str(orig),compiled_from_config_sha256=hashlib.sha256(orig.read_bytes()).hexdigest(),runtime_llm=False,code_revision=git_head())
  for k in ('selection_pool','selection_pool_sha256'):
   if k in c:cfg[k]=c[k]
  save_json(out/'config.json',cfg)
  for split in ('dev','confirm','val'):
   p=apply(np.stack([get_source(k,split) for k in ids],1),cfg);expected=get_source('run:'+name,split).copy();expected[:,:,~np.asarray(mask,bool)]=0;assert np.array_equal(p,expected),(name,split);np.save(out/f'{split}_predictions.npy',p)
  audit.append(dict(original=name,compiled=new,required_columns=list(mask),active_slots=positions,all_three_splits_exact=True));memo[key]=new;return new
 root=compile_node(parent,[True]*7);assert root!=parent;d=ROOT/'runs'/root;subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',root],check=True);results={}
 for split in ('dev','confirm','val'):
  p=get_source('run:'+root,split);assert np.array_equal(p,get_source('run:'+parent,split));results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'))
 save_json(d/'metrics.json',dict(candidate=root,metrics=results,status='evaluated',equivalent_to=parent));save_json(d/'compilation_audit.json',dict(nodes=audit,root=root,original=parent,all_three_splits_exact=True,not_independent_gain=True));print(root,results['val']['mse'],len(audit),flush=True)
if __name__=='__main__':run(sys.argv[1],sys.argv[2])
