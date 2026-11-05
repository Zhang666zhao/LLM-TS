"""Isolated original-runtime inference on explicitly supplied historical inputs only."""
from common import *
import torch,sys,importlib,pandas
from types import SimpleNamespace
modeldir=pathlib.Path(sys.argv[1]);inputs=pathlib.Path(sys.argv[2]);output=pathlib.Path(sys.argv[3]);cfg=json.loads((modeldir/'config.json').read_text());assert cfg['strategy']=='base' and cfg['label']=='zero' and cfg['oracle'] is False
assert json.loads((modeldir/'runtime.json').read_text())==dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,pandas=pandas.__version__),'Original runtime version changed'
sys.path.insert(0,str(modeldir/'source'));module=importlib.import_module({'patchtst':'models.PatchTST','itransformer':'model.iTransformer','frets':'models.FreTS'}[cfg['method']]);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;net=module.Model(SimpleNamespace(**cfg)).cuda();state=torch.load(modeldir/'best.pth',map_location='cpu',weights_only=True);assert all(k.startswith('backbone.') for k in state);net.load_state_dict({k[len('backbone.'):]:v for k,v in state.items()},strict=True);net.eval();d=np.load(inputs);x=d['history'];marks=d['marks'];assert x.ndim==3 and x.shape[1:]==(512,7) and marks.shape==(len(x),512,4)
def forward(a,m):return net(a,m,None,None) if cfg['method']=='itransformer' else net(a)
ps=[]
with torch.no_grad():
 a=torch.from_numpy(x[:2]).cuda();m=torch.from_numpy(marks[:2]).cuda();p=forward(a,m);a[1]+=10;q=forward(a,m);delta=float((p[0]-q[0]).abs().max());assert delta<1e-5,('query isolation',delta)
 for i in range(0,len(x),cfg['batch']):guard(600);ps.append(forward(torch.from_numpy(x[i:i+cfg['batch']]).cuda(),torch.from_numpy(marks[i:i+cfg['batch']]).cuda()).cpu().numpy())
p=np.concatenate(ps);assert p.shape==(len(x),64,7) and np.isfinite(p).all();np.save(output,p);save_json(output.with_suffix('.json'),{'other_query_perturbation_max_delta':delta,'torch':torch.__version__,'python':sys.version,'inputs_only':'supplied512 historical values and timestamps; no future values, labels, raw CSV or oracle cache'})
