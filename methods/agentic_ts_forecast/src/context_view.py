"""Use a declared subset of fixed512 history with the same frozen view checkpoint."""
from common import *
from transformed_tool import predict
import sys

def run(context):
 assert context in (96,168,336);parent='20261105_transformedft_random_mse';name=f'20261105_contextview_{context}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);guard(1800);cfg=json.loads((ROOT/'runs'/parent/'config.json').read_text());cfg.update(inference_context=context,source_ids=['run:'+parent],external_history_scope=512,code_revision=git_head(),context_caveat='Checkpoint trained at512; only inference tail-context changes; no extra observations');save_json(out/'config.json',cfg);z=np.load(RUN/'cache/trainval_z.npy');mf=json.loads((RUN/'cache/window_manifest.json').read_text());start=time.time();results={}
 for split,ts in mf.items():
  p=predict(z,ts,cfg);results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'));np.save(out/f'{split}_predictions.npy',p);save_json(out/'partial_metrics.json',results)
 save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated','elapsed_seconds':time.time()-start});print(name,{s:m['mse'] for s,m in results.items()},flush=True)
if __name__=='__main__':run(int(sys.argv[1]))
