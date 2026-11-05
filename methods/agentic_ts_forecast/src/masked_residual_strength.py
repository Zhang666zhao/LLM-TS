"""Three predeclared strengths on the five variables retaining old residual memory."""
from common import *
from blend_sources import get_source
import sys
SOURCES=['run:20261105_cdfswap_both_mse','run:20261105_viewresidual_ridge10_mse','run:20261105_auto_p05b005c1_mse'];ACTIVE=['HUFL','HULL','MUFL','MULL','LUFL']
def apply(parent,memory,base,cfg):
 mask=np.array([c in cfg['columns'] for c in COLS],dtype=float);return (parent.astype(float)+(cfg['strength']-cfg['reference_strength'])/.25*(memory.astype(float)-base.astype(float))*mask[None,None]).astype(np.float32)
def run(alpha):
 assert alpha in (.1,.5,.75);name=f'20261105_maskedmemory_a{int(alpha*100)}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);guard(1800);cfg={'engine':'masked_residual_strength_v1','source_ids':SOURCES,'columns':ACTIVE,'reference_strength':.25,'strength':alpha,'fixed_other_columns':['LULL','OT'],'code_revision':git_head()};save_json(out/'config.json',cfg);results={}
 for split in ('dev','confirm','val'):
  args=[get_source(k,split) for k in SOURCES];p=apply(*args,cfg);assert np.isfinite(p).all() and np.array_equal(p[:,:,[5,6]],args[0][:,:,[5,6]]);assert np.array_equal(apply(*args,dict(cfg,strength=.25)),args[0]);np.save(out/f'{split}_predictions.npy',p);results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'))
 save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated'});save_json(out/'contract.json',{'reference_strength_exact_identity':True,'LULL_OT_bitwise_unchanged':True});print(name,{s:m['mse'] for s,m in results.items()},flush=True)
if __name__=='__main__':run(float(sys.argv[1]))
