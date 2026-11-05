"""Correct mistaken source ranking with three predeclared OT constant controls."""
from common import *
from specialist_policy import run
import copy
parent='20261105_policy_p11b002c2_mse';cfg=json.loads((ROOT/'runs'/parent/'config.json').read_text());base={k:cfg[k] for k in ('source_ids','base_weights','variable_weights','rules')}
for tag,w in [('ranked',[0,1,0,0]),('mix25',[0,.75,0,.25]),('mix50',[0,.5,0,.5])]:
 c=copy.deepcopy(base);c['rules']=[];c['variable_weights']['OT']=w;run(c,f'20261105_p11constant_{tag}_mse')
