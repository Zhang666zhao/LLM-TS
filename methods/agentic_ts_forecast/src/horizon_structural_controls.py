"""Three simplification/replacement controls after horizon-search reflection."""
from common import *
from horizon_policy import run
import copy
c=json.loads((RUN/'controller/p13/champion.json').read_text())['config'];c['variable_weights']['LUFL']=[[0,0,1,0]]*4;run(copy.deepcopy(c),'20261105_structural_lufltree_mse')
c['source_ids'][0]='run:20261105_p11constant_ranked_mse';run(copy.deepcopy(c),'20261105_structural_nootrule_mse')
c['source_ids'][1]='run:20261105_calendarcdfmse_random_mse';c['variable_weights']['MUFL']=[[0,1,0,0]]*4;run(copy.deepcopy(c),'20261105_structural_calendarmse_mse')
