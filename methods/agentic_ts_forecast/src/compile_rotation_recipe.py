"""Compile three active expert branches; require exact equality to expanded recipe."""
from common import *
from blend_sources import get_source
from masked_residual_strength import apply
from horizon_policy import run,predict as policy_predict
import sys
name='20261105_compactrotation_exact_mse';intermediate='20261105_compactmemory_scaled_mse';out=ROOT/'runs'/intermediate;out.mkdir(exist_ok=False);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',intermediate],check=True);memory='run:20261105_variabletree_calendar_d2_mse';cfg=dict(engine='masked_residual_strength_v1',source_ids=[memory,memory,'run:20261105_calendarprior_exact_mse'],columns=['MULL','LUFL'],reference_strength=.25,strength=.5,artifact_role='unscored compiled intermediate, no target read',code_revision=git_head());save_json(out/'config.json',cfg)
for split in ('dev','confirm','val'):
 guard(1800);p=apply(*[get_source(k,split) for k in cfg['source_ids']],cfg);np.save(out/f'{split}_predictions.npy',p)
registry=['run:20261105_p11constant_ranked_mse','run:20261105_learnedrotation_r003_masked_mse','run:'+intermediate,'run:20261105_p11constant_ranked_mse'];c=dict(source_ids=registry,knot_weights=[[1,0,0,0]]*4,variable_weights={**{col:[[0,1,0,0]]*4 for col in ('HUFL','HULL','MUFL')},**{col:[[0,0,1,0]]*4 for col in ('MULL','LUFL')}});reference='20261105_rotationprune_both_mse'
for split in ('dev','confirm','val'):
 p=policy_predict(np.stack([get_source(k,split) for k in registry],1),c);q=np.load(ROOT/'runs'/reference/f'{split}_predictions.npy');assert np.array_equal(p,q),'Pre-score compilation equality failed'
run(c,name);diffs={}
for split in ('dev','confirm','val'):
 p=np.load(ROOT/'runs'/name/f'{split}_predictions.npy');q=np.load(ROOT/'runs'/reference/f'{split}_predictions.npy');diffs[split]=float(np.max(abs(p-q)));assert np.array_equal(p,q),'Compilation must preserve every prediction'
save_json(ROOT/'runs'/name/'compilation_check.json',dict(reference=reference,all_three_splits_bitwise_equal=True,max_deltas=diffs,not_independent_evidence=True,test_access=False));print('COMPILED_EXACT',name,flush=True)
