"""Three matched point-extraction substitutions with frozen specialist conditions."""
from common import *
from blend_sources import get_source
from specialist_policy import run
import sys,copy
parent='20261105_policy_p07b005c3_mse';original=json.loads((ROOT/'runs'/parent/'config.json').read_text());base={k:original[k] for k in ('source_ids','base_weights','variable_weights','rules')}
# An unscored intermediate DAG node. It does not read any target array.
name='20261105_integral_core_unscored';out=ROOT/'runs'/name;out.mkdir(exist_ok=True);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);cfg={'engine':'blend_sources_v1','sources':{'run:20261105_fullpoint_pca_mse':.18,'run:20261105_viewpoint_full_integral_mse':.27,'run:20261105_transformedmse_pca_mse':.3,'run:20261105_transformedmse_random_mse':.25},'amplitude':1,'mean_shrink':0,'smooth':1,'proxy_bias_strength':0,'artifact_role':'unscored intermediate for matched point-extraction substitution; no labels read','code_revision':git_head()};save_json(out/'config.json',cfg)
for split in ('dev','confirm','val'):
 total=sum(cfg['sources'].values());p=sum(v/total*get_source(k,split).astype(float) for k,v in cfg['sources'].items());center=p.mean(1,keepdims=True);np.save(out/f'{split}_predictions.npy',(center+(p-center)).astype(np.float32))
for tag,index,source in [('lora',2,'run:20261105_fullpoint_lora_mse'),('semantic',1,'run:20261105_fullpoint_semantic_mse'),('core',0,'run:'+name)]:
 c=copy.deepcopy(base);c['source_ids'][index]=source;candidate=f'20261105_specialist_pointswap_{tag}_mse';run(c,candidate);save_json(ROOT/'runs'/candidate/'ablation_manifest.json',{'reference':parent,'source_slot_changed':index,'old_source':base['source_ids'][index],'new_source':source,'weights_and_rules_unchanged':True,'proposal_origin':'predeclared matched point extraction ablation','code_revision':git_head()})
