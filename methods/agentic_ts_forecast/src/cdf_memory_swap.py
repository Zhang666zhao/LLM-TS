"""Replace the residual memory and its matched prior inside the fixed strongest DAG."""
from common import *
from blend_sources import get_source,predict_cached
from cdf_checkpoint_swap import rewrite
import sys
base='20261105_cdf_memory_prior_unscored';out=ROOT/'runs'/base;cfg={'engine':'blend_sources_v1','sources':{'run:20261105_fullpoint_pca_mse':.4,'run:20261105_viewpoint_full_integral_mse':.6},'amplitude':1,'mean_shrink':0,'smooth':1,'proxy_bias_strength':0,'artifact_role':'unscored fixed CDF memory prior, no targets read','code_revision':git_head()}
if not out.exists():
 out.mkdir();subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',base],check=True);save_json(out/'config.json',cfg);z=np.load(RUN/'cache/trainval_z.npy')
 for split,ts in json.loads((RUN/'cache/window_manifest.json').read_text()).items():np.save(out/f'{split}_predictions.npy',predict_cached(cfg,split,z,ts))
else:assert json.loads((out/'config.json').read_text())['sources']==cfg['sources']
arg=sys.argv[1]
if arg.startswith('tree'):
 depth=int(arg[4:]);assert depth in (2,3,4);tag='prefixtree'+str(depth);memory=f'20261105_prefixtree_cdf_d{depth}_mse'
else:
 reg=int(arg);assert reg in (1,10,100);tag='memorycdf'+str(reg);memory=f'20261105_viewresidualcdf_ridge{reg}_mse'
rewrite('20261105_cdfswap_both_mse',tag,{'20261105_viewresidual_ridge10_mse':memory,'20261105_auto_p05b005c1_mse':base})
