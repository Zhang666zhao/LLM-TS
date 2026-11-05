"""Predeclared constant LUFL mixtures to test horizon-profile attribution."""
from common import *
from horizon_policy import run
import copy
c=json.loads((RUN/'controller/p12r/champion.json').read_text())['config']
for a,tag in [(.3,'30'),(.4,'40'),(.5,'50')]:
 cfg=copy.deepcopy(c);cfg['variable_weights']['LUFL']=[[1-a,0,a,0]]*4;run(cfg,f'20261105_horizon_constant{tag}_mse')
