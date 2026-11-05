"""Replace one expert in a fixed horizon policy; assert unrelated columns unchanged."""
from common import *
from horizon_policy import run
import sys,copy
parent=sys.argv[1];index=int(sys.argv[2]);tag=sys.argv[3];replacements=json.loads(sys.argv[4]);assert 0<=index<4 and len(replacements)==3
raw=json.loads((ROOT/'runs'/parent/'config.json').read_text());assert raw['engine']=='horizon_policy_v1';cfg={k:raw[k] for k in ('source_ids','knot_weights','variable_weights')};unchanged=[j for j,col in enumerate(COLS) if not np.asarray(cfg['variable_weights'].get(col,cfg['knot_weights']))[:,index].any()]
for key,source in replacements.items():
 c=copy.deepcopy(cfg);c['source_ids'][index]='run:'+source;name=f'20261105_{tag}_{key}_mse';run(c,name)
 for split in ('dev','confirm','val'):
  a=np.load(ROOT/'runs'/parent/f'{split}_predictions.npy');b=np.load(ROOT/'runs'/name/f'{split}_predictions.npy');assert np.array_equal(a[:,:,unchanged],b[:,:,unchanged]),'Unrelated columns changed'
 save_json(ROOT/'runs'/name/'isolation_check.json',dict(unchanged_columns=[COLS[j] for j in unchanged],source_index=index,parent=parent,test_access=False))
