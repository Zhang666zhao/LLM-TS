"""New exact-prior controls; original failed bitwise attempt remains preserved."""
from common import *
from cdf_checkpoint_swap import rewrite
from blend_sources import get_source
from view_prior import predict
import sys
mode=sys.argv[1];assert mode in ('tree','ridge','both');mapping={};prior='20261105_tree_prior_exact_unscored';out=ROOT/'runs'/prior;sources=['run:20261105_fullpoint_semantic_mse','run:20261105_fullpoint_pca_mse','run:20261105_viewpoint_full_integral_mse']
if not out.exists():
 out.mkdir();subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',prior],check=True);guard(1800);cfg={'engine':'fixed_view_prior_v1','source_ids':sources,'weights':[0.,.4,.6],'artifact_role':'unscored exact tree baseline, no target arrays read','code_revision':git_head()};save_json(out/'config.json',cfg)
 from prefix_error_tree import inputs
 z=np.load(RUN/'cache/trainval_z.npy');dates=json.loads((RUN/'cache/dates.json').read_text());mf=json.loads((RUN/'cache/window_manifest.json').read_text());checks={}
 for split,ts in mf.items():
  p=np.stack([get_source(k,split) for k in sources],1);base=predict(p,cfg['weights']);actual=inputs(z,dates,ts[:2],p[:2],np.zeros(203),np.ones(203))[3].cpu().numpy();assert np.array_equal(base[:2],actual),'Actual tree input baseline differs';np.save(out/f'{split}_predictions.npy',base);checks[split]={'actual_tree_input_baseline_exact':True,'no_targets_read':True}
 save_json(out/'contract.json',checks)
else:assert json.loads((out/'config.json').read_text())['source_ids']==sources and (out/'contract.json').exists()
if mode in ('tree','both'):mapping['20261105_prefixtree_cdf_d4_mse']=prior
if mode in ('ridge','both'):mapping['20261105_viewresidual_ridge10_mse']='20261105_auto_p05b005c1_mse'
rewrite('20261105_auto_p10b006c2_mse','ablate_exact_'+mode,mapping)
