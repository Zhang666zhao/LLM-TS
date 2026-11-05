"""Audit LLM default profiles using isolated column changes and full HULL alternative."""
from common import *
from horizon_policy import run
import copy
raw=json.loads((ROOT/'runs/20261105_horizon_p14b001c2_mse/config.json').read_text());reg=raw['source_ids'];base=dict(source_ids=reg,knot_weights=[[1,0,0,0]]*4,variable_weights={});early=raw['knot_weights'];hull=[[0,0,0,1]]*4
for tag,c in [('hull',dict(base,variable_weights={'HULL':hull})),('hullearly',dict(source_ids=reg,knot_weights=early,variable_weights={'HULL':hull})),('earlyonly',dict(base,variable_weights={'HUFL':early,'MUFL':early}))]:
 run(c,f'20261105_p14structural_{tag}_mse')
