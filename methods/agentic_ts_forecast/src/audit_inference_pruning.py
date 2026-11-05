from common import *
from blend_sources import get_source
from horizon_policy import predict,active_sources
from strong_view_feedback import apply,TOOLS,PARENT
seen=set();nodes=[]
def visit(name):
 if name in seen:return
 seen.add(name);c=json.loads((ROOT/'runs'/name/'config.json').read_text())
 if c.get('engine')=='horizon_policy_v1':nodes.append((name,c))
 for k in list(c.get('source_ids',[]))+list(c.get('sources',{})):
  if k.startswith('run:'):visit(k[4:])
visit('20261105_blockcolumns_v0824_b16_mse');records=[]
for name,c in nodes:
 active=active_sources(c)
 for split in ('dev','confirm','val'):
  guard(1800);p=np.stack([get_source(k,split) for k in c['source_ids']],1);old=predict(p,c);p[:,~active]=0;new=predict(p,c);assert np.array_equal(old,new),(name,split)
 records.append(dict(candidate=name,active_source_mask=active.tolist(),all_three_splits_exact=True))
c=json.loads((ROOT/'runs/20261105_strongviewfeedback_uniform_mse/config.json').read_text());z=np.load(RUN/'cache/trainval_z.npy');manifest=json.loads((RUN/'cache/window_manifest.json').read_text())
for split in ('dev','confirm','val'):
 pred=np.stack([get_source('run:'+k,split) for k in TOOLS],1);new=apply(z,manifest[split],None,pred,get_source('run:'+PARENT,split),c);old=get_source('run:20261105_strongviewfeedback_uniform_mse',split);assert np.array_equal(new,old),split
result=dict(horizon_nodes=records,uniform_hindcast_elision_all_splits_exact=True,code_revision=git_head(),test_access=False);save_json(RUN/('inference_pruning_audit_'+git_head()[:7]+'.json'),result);print(json.dumps(result),flush=True)
