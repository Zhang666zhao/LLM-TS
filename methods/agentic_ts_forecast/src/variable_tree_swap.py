from common import *
from horizon_policy import run
import copy
parent='20261105_structural_calendarmse_mse';raw=json.loads((ROOT/'runs'/parent/'config.json').read_text());cfg={k:raw[k] for k in ('source_ids','knot_weights','variable_weights')}
for depth in (2,3,4):
 c=copy.deepcopy(cfg);c['source_ids'][2]=f'run:20261105_variabletree_cdf_d{depth}_mse';name=f'20261105_variableswap_d{depth}_mse';run(c,name)
 for split in ('dev','confirm','val'):
  a=np.load(ROOT/'runs'/parent/f'{split}_predictions.npy');b=np.load(ROOT/'runs'/name/f'{split}_predictions.npy');assert np.array_equal(a[:,:,[0,1,2,3,5,6]],b[:,:,[0,1,2,3,5,6]])
 save_json(ROOT/'runs'/name/'isolation_check.json',dict(unchanged_other6_variables=True,test_access=False))
