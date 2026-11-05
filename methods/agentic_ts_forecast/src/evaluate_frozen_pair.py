"""Call the unchanged official evaluator once, then score the fixed baseline on its exact target array."""
import csv,json,pathlib,hashlib,sys,os,time,subprocess,runpy
import numpy as np

def metrics(pred,y,columns):
 assert pred.shape==y.shape and pred.ndim==3 and pred.shape[-1]==len(columns)
 assert np.isfinite(pred).all() and np.isfinite(y).all()
 e=pred.astype('float64')-y.astype('float64')
 return {'mse':float(np.mean(e*e)),'mae':float(np.mean(abs(e))),'per_variable_mse':dict(zip(columns,np.mean(e*e,axis=(0,1)).tolist())),'per_variable_mae':dict(zip(columns,np.mean(abs(e),axis=(0,1)).tolist())),'horizon_mse':np.mean(e*e,axis=(0,2)).tolist(),'windows':len(y)}

def main():
 root=pathlib.Path(os.environ.get('ETTH1_PROJECT_ROOT',pathlib.Path(__file__).resolve().parent.parent));run=root/'runs/20261105_agentic_continuous_mse';output=run/'sealed_test_comparison.json'
 assert not output.exists() and not (run/'sealed_test_result.json').exists(),'Official test already scored; never repeat selection/evaluation'
 frozen=json.loads((run/'frozen.json').read_text());assert time.time()<frozen['deadline_epoch'],'Original budget expired';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 assert sha(run/'selection_frozen.json')==frozen['selection_frozen_sha256'],'Selection manifest changed'
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==frozen['code_revision'],'Code changed after freeze'
 official=root/'src/official_eval.py';assert sha(official)==frozen['files'][str(official)]
 assert sha(pathlib.Path(__file__))==frozen['files'][str(root/'src/evaluate_frozen_pair.py')]
 p=run/'test_predictions.npy';bp=run/'test_baseline_predictions.npy';assert sha(p)==frozen['prediction_sha256'];assert sha(bp)==frozen['baseline_prediction_sha256']
 baseline_pred=np.load(bp);candidate_pred=np.load(p);assert baseline_pred.shape==candidate_pred.shape==(2817,64,7) and np.isfinite(baseline_pred).all() and np.isfinite(candidate_pred).all()
 cp=root/'benchmark/contract.json';assert sha(cp)==frozen['files'][str(cp)];c=json.loads(cp.read_text());source=pathlib.Path(c['source']);assert sha(source)==frozen['files'][str(source)]
 # The benchmark's byte-identical original script constructs targets and seals the selected score.
 ns=runpy.run_path(str(official),run_name='__main__');selected=ns['result'];y=ns['y'];baseline=metrics(baseline_pred,y,c['variables']);assert abs(metrics(candidate_pred,y,c['variables'])['mse']-selected['mse'])<1e-12 and baseline['mse']>0
 result=dict(candidate=frozen['candidate'],selected=selected,baseline=baseline,mse_ratio=selected['mse']/baseline['mse'],relative_mse_improvement_percent=100*(1-selected['mse']/baseline['mse']),protocol='unchanged official evaluator invoked once; baseline scored on the SAME returned targets',prediction_sha256=frozen['prediction_sha256'],baseline_prediction_sha256=frozen['baseline_prediction_sha256'],official_result_sha256=sha(run/'sealed_test_result.json'),test_used_for_selection=False,evaluated_epoch=time.time(),deadline_epoch=frozen['deadline_epoch']);assert result['evaluated_epoch']<=result['deadline_epoch']
 with output.open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps({k:result[k] for k in ('candidate','mse_ratio','relative_mse_improvement_percent','evaluated_epoch')}),flush=True)
if __name__=='__main__':main()
