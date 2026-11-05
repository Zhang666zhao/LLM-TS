"""Same predeclared CDF integral on three already fitted tools, with no re-fitting."""
from common import *
from transformed_tool import predict
import sys
PARENTS={'pca':'20261105_transformedft_pca_mse','semantic':'20261105_transformedft_semantic_mse','lora':'20261105_chronoslora_lr3e-5_p0181_fixed_mse'}
PARENTS.update({'mse_'+v:'20261105_transformedmse_'+v+'_mse' for v in ('semantic','pca','random')})
def run(mode):
 parent=PARENTS[mode];name=f'20261105_fullpoint_{mode}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);guard(1800);cfg=json.loads((ROOT/'runs'/parent/'config.json').read_text());levels=json.loads((RUN/'models/chronos2_29ec3766/config.json').read_text())['chronos_config']['quantiles'];original_point=cfg.get('point');cfg.update(engine='transformed_tool_v1',checkpoint_parent=cfg.get('checkpoint_parent',parent),point_rule='full_integral',point_quantile_levels=levels,point='latent quantile CDF trapezoid integral with flat endpoint tails, then inverse POINT transform',parent_point_metadata=original_point,source_ids=['run:'+parent],code_revision=git_head())
 if 'matrix' not in cfg:cfg.update(matrix=np.eye(7).tolist(),bias=np.zeros(7).tolist(),inverse_matrix=np.eye(7).tolist(),transform='identity; native original-coordinate model')
 save_json(out/'config.json',cfg);z=np.load(RUN/'cache/trainval_z.npy');mf=json.loads((RUN/'cache/window_manifest.json').read_text());results={};start=time.time()
 for split,ts in mf.items():
  p=predict(z,ts,cfg);results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'));np.save(out/f'{split}_predictions.npy',p);save_json(out/'partial_metrics.json',results)
 save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated','elapsed_seconds':time.time()-start});print(name,{s:m['mse'] for s,m in results.items()},flush=True)
if __name__=='__main__':run(sys.argv[1])
