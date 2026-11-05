"""Matched .5-strength temporal residual templates on only MULL/LUFL."""
from common import *
from masked_residual_strength import apply
from blend_sources import get_source
from horizon_policy import run
import sys
parent='20261105_compactrotation_exact_mse'
for basis in ('lead','sum','interaction'):
 tool='run:20261105_factortree_calendar_'+basis+'_mse';intermediate='20261105_factorblockscaled_'+basis+'_mse';out=ROOT/'runs'/intermediate;out.mkdir(exist_ok=False);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',intermediate],check=True);cfg=dict(engine='masked_residual_strength_v1',source_ids=[tool,tool,'run:20261105_calendarprior_exact_mse'],columns=['MULL','LUFL'],reference_strength=.25,strength=.5,artifact_role='unscored composition intermediate, no targets read',code_revision=git_head());save_json(out/'config.json',cfg)
 for split in ('dev','confirm','val'):
  guard(1800);p=apply(*[get_source(k,split) for k in cfg['source_ids']],cfg);np.save(out/f'{split}_predictions.npy',p)
 name='20261105_factorblockswap_'+basis+'_mse';registry=['run:'+parent,'run:'+intermediate,'run:'+parent,'run:'+parent];c=dict(source_ids=registry,knot_weights=[[1,0,0,0]]*4,variable_weights={col:[[0,1,0,0]]*4 for col in ('MULL','LUFL')});run(c,name)
 for split in ('dev','confirm','val'):
  p=np.load(ROOT/'runs'/name/f'{split}_predictions.npy');q=np.load(ROOT/'runs'/parent/f'{split}_predictions.npy');assert np.array_equal(p[:,:,[0,1,2,5,6]],q[:,:,[0,1,2,5,6]])
 save_json(ROOT/'runs'/name/'isolation_check.json',dict(MULL_LUFL_only=True,other_five_columns_exact=True,strength=.5))
