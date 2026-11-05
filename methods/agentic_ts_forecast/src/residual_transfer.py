"""Three bounded strengths for a fixed train-only residual memory on a stronger parent."""
from common import *
from blend_sources import get_source
import sys
PARENT='20261105_auto_p08b001c1_mse';MEMORY='20261105_viewresidual_ridge10_mse';BASE='20261105_auto_p05b005c1_mse'
def apply(parent,memory,base,strength):return (parent.astype(float)+(strength/.25)*(memory.astype(float)-base.astype(float))).astype(np.float32)
def run(strength):
 assert strength in (.1,.25,.5);tag={.1:'10',.25:'25',.5:'50'}[strength];name=f'20261105_residualtransfer_a{tag}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);guard(1800);cfg={'engine':'residual_transfer_v1','source_ids':['run:'+n for n in (PARENT,MEMORY,BASE)],'strength':strength,'memory_parent_strength':.25,'caveat':'Residual fitted to weaker mean9 tool prior, transferred unchanged to stronger policy; fulltrain dev/confirm in-sample, no independent claim','code_revision':git_head()};save_json(out/'config.json',cfg);results={}
 for split in ('dev','confirm','val'):
  p=apply(*[get_source(k,split) for k in cfg['source_ids']],strength);assert np.isfinite(p).all();np.save(out/f'{split}_predictions.npy',p);results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'))
 save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated'});print(name,{s:m['mse'] for s,m in results.items()},flush=True)
if __name__=='__main__':run(float(sys.argv[1]))
