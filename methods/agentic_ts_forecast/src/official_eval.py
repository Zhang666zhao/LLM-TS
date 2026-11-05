"""Protected final evaluator; only run after frozen full predictions are saved."""
import csv,json,pathlib,hashlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parent.parent
RUN=ROOT/'runs/20261105_agentic_continuous_mse'
frozen=json.loads((RUN/'frozen.json').read_text())
p=RUN/'test_predictions.npy'
assert hashlib.sha256(p.read_bytes()).hexdigest()==frozen['prediction_sha256']
output=RUN/'sealed_test_result.json'
if output.exists():raise RuntimeError('Test already scored; do not adapt or repeat selection')
c=json.loads((ROOT/'benchmark/contract.json').read_text());rows=[]
with open(c['source']) as f:
 r=csv.DictReader(f)
 for i in range(14400):rows.append(next(r))
z=((np.array([[float(r[k]) for k in c['variables']] for r in rows])-c['scaler_mean'])/c['scaler_scale']).astype('float32')
y=np.stack([z[t:t+64] for t in range(11520,14400-64+1)])
pred=np.load(p);assert pred.shape==y.shape and np.isfinite(pred).all()
e=pred.astype('float64')-y.astype('float64')
result={'mse':float(np.mean(e*e)),'mae':float(np.mean(abs(e))),'per_variable_mse':dict(zip(c['variables'],np.mean(e*e,axis=(0,1)).tolist())),'windows':len(y),'protocol':'frozen once','prediction_sha256':frozen['prediction_sha256']}
output.write_text(json.dumps(result,indent=2));print(json.dumps(result))
