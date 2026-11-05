"""Matched controls for selected conditional specialist: no validation fitting."""
from common import *
from specialist_policy import run
import copy
PARENT='20261105_policy_p07b005c3_mse'
original=json.loads((ROOT/'runs'/PARENT/'config.json').read_text());base={k:original[k] for k in ('source_ids','base_weights','variable_weights','rules')};trainw=np.concatenate([np.load(ROOT/'runs'/PARENT/(s+'_weights.npy')) for s in ('dev','confirm')]);assert trainw.shape==(54,7,4);configs=[]
a=copy.deepcopy(base);a['rules']=[];configs.append(('norules',a,'Remove conditions, preserve static variable overrides'))
a=copy.deepcopy(base);a['rules']=[];a['variable_weights']={k:trainw[:,j].mean(0).tolist() for j,k in enumerate(COLS)};configs.append(('trainfrequency',a,'Use per-variable mean weights on fixed54 TRAINING queries; no validation-derived weights'))
a=copy.deepcopy(base);a['rules']=[];assigned=set()
for r in base['rules']:
 for col in r['columns']:
  if col not in assigned:a['variable_weights'][col]=r['weights'];assigned.add(col)
configs.append(('alwaysrule',a,'Apply first rule weights for each listed variable unconditionally'))
for tag,cfg,description in configs:
 name=f'20261105_specialist_ablation_{tag}_mse';run(cfg,name);save_json(ROOT/'runs'/name/'ablation_manifest.json',{'reference':PARENT,'description':description,'weights_fit_on_validation':False,'reference_config_sha256':hashlib.sha256((ROOT/'runs'/PARENT/'config.json').read_bytes()).hexdigest()})
