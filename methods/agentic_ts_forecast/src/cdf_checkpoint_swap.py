"""Three matched checkpoint swaps in a frozen numeric DAG; intermediates unscored."""
from common import *
from blend_sources import get_source,predict_cached
from specialist_policy import predict as policy_predict
from residual_transfer import apply
import sys,copy
PARENT='20261105_policy_p09b002c2_mse'

def rewrite(parent,tag,mapping):
 assert tag.replace('_','').isalnum() and mapping
 z=np.load(RUN/'cache/trainval_z.npy');mf=json.loads((RUN/'cache/window_manifest.json').read_text());seen={};clones=[]
 def clone(name):
  if name in mapping:return mapping[name]
  if name in seen:return seen[name]
  cfg=json.loads((ROOT/'runs'/name/'config.json').read_text());c=copy.deepcopy(cfg)
  def source(k):return 'run:'+clone(k.split(':',1)[1]) if k.startswith('run:') else k
  if 'sources' in c:c['sources']={source(k):v for k,v in c['sources'].items()}
  if 'source_ids' in c:c['source_ids']=[source(k) for k in c['source_ids']]
  if c==cfg:seen[name]=name;return name
  assert c['engine'] in ('blend_sources_v1','specialist_policy_v1','residual_transfer_v1'),('Unsupported changed node',name,c['engine'])
  new=f'20261105_cdfswap_{tag}_'+('mse' if name==parent else hashlib.sha256(name.encode()).hexdigest()[:12]+'_unscored');out=ROOT/'runs'/new;assert not (out/'config.json').exists();out.mkdir(exist_ok=True);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',new],check=True);guard(1800);c.update(clone_origin=name,artifact_role='scored root' if name==parent else 'unscored DAG intermediate, no target read',checkpoint_replacements=mapping,code_revision=git_head());save_json(out/'config.json',c)
  for split,ts in mf.items():
   if c['engine']=='blend_sources_v1':p=predict_cached(c,split,z,ts)
   elif c['engine']=='specialist_policy_v1':x=np.stack([z[t-512:t] for t in ts]);p=policy_predict(x,np.stack([get_source(k,split) for k in c['source_ids']],1),c)[0]
   else:p=apply(*[get_source(k,split) for k in c['source_ids']],c['strength'])
   assert np.isfinite(p).all();np.save(out/f'{split}_predictions.npy',p)
  seen[name]=new;clones.append({'original':name,'clone':new});return new
 root=clone(parent);assert root!=parent;out=ROOT/'runs'/root;results={s:metric(np.load(out/f'{s}_predictions.npy'),np.load(RUN/'cache'/f'{s}_targets.npy')) for s in mf};save_json(out/'metrics.json',{'candidate':root,'metrics':results,'status':'evaluated'});save_json(out/'ablation_manifest.json',{'parent':parent,'checkpoint_replacements':mapping,'all_other_weights_thresholds_and_rules_unchanged':True,'clones':clones});print(root,{s:m['mse'] for s,m in results.items()},flush=True)
def run(tag):
 assert tag in ('pca','random','both');modes=('pca','random') if tag=='both' else (tag,);mapping={'20261105_transformedmse_'+m+'_mse':'20261105_transformedcdfmse_'+m+'_mse' for m in modes};rewrite(PARENT,tag,mapping)
if __name__=='__main__':run(sys.argv[1])
