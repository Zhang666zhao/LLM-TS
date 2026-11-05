"""Confidence-gated empirical relations: correct only excess outside historical band."""
from common import *
import sys

def apply(x,p,k,strength=.25):
 scale=np.asarray(CONTRACT['scaler_scale']);A=np.zeros((2,7));A[0,[0,2,4]]=scale[[0,2,4]]*np.array([1,-1,-1]);A[1,[1,3,5]]=scale[[1,3,5]]*np.array([1,-1,-1]);P=A.T@np.linalg.inv(A@A.T)
 rel=x[:,-168:]@A.T;center=rel.mean(1,keepdims=True);sd=np.maximum(rel.std(1,keepdims=True),.1);e=p@A.T-center;excess=np.sign(e)*np.maximum(np.abs(e)-k*sd,0);return (p-strength*(excess@P.T)).astype(np.float32)
if __name__=='__main__':
 k=float(sys.argv[1]);parent=sys.argv[2];name=sys.argv[3];out=ROOT/'runs'/name;out.mkdir(exist_ok=True)
 subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True)
 z=np.load(RUN/'cache/trainval_z.npy');manifest=json.loads((RUN/'cache/window_manifest.json').read_text());results={}
 for split,ts in manifest.items():
  guard(900);x,y=windows(z,ts);p=np.load(ROOT/'runs'/parent/f'{split}_predictions.npy');pr=apply(x,p,k);results[split]=metric(pr,y);np.save(out/f'{split}_predictions.npy',pr)
 save_json(out/'config.json',{'engine':'gated_semantics_v1','parent':parent,'k':k,'strength':.25,'history':168,'relation':'empirical H-M-L residual; unknown topology, not physical equality','code_revision':git_head()});save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated'});print(name,{s:r['mse'] for s,r in results.items()},flush=True)
