"""Import non-oracle frozen tools, and reproduce validation in original runtime."""
from common import *
import sys,shutil,tempfile,pandas as pd
import os
BASEPY=os.environ.get('AGENTIC_TS_LEGACY_PYTHON', sys.executable)

def prepare(method):
 project=ROOT.parent/'TSRAG';src=project/'runs/20261014_six_multivariate_oracle_v1/ETTh1/jobs'/f'{method}_base_zero';out=RUN/'models'/f'legacy_{method}';assert not out.exists();cfg=json.loads((src/'config.json').read_text());assert cfg['strategy']=='base' and cfg['label']=='zero' and cfg['oracle'] is False and cfg['dataset']=='ETTh1' and cfg['seq_len']==512 and cfg['pred_len']==64 and cfg['enc_in']==7
 # Only benchmark identity fields retained; no labels, class counts or test metrics used.
 meta=json.loads((src/'data_version.json').read_text());safe={k:meta[k] for k in ('source_sha256','variables','scaler_mean','scaler_scale','context','horizon','stride')};assert safe['variables']==COLS and np.array_equal(safe['scaler_mean'],CONTRACT['scaler_mean']) and np.array_equal(safe['scaler_scale'],CONTRACT['scaler_scale']);assert safe['source_sha256']=='3dbe45a7a25cbf0eb6d82cb84ef4487ea2b73a9341e5ff0d082e340b9d943f1d'
 repo=project/'research/baselines_512_64/sources'/{'patchtst':'PatchTST/PatchTST_supervised','itransformer':'iTransformer','frets':'FreTS'}[method];out.mkdir();env=os.environ.copy();env.pop('PYTHONPATH',None);versions=json.loads(subprocess.check_output([BASEPY,'-c','import json,sys,torch,numpy,pandas; print(json.dumps(dict(python=sys.version,torch=torch.__version__,numpy=numpy.__version__,pandas=pandas.__version__)))'],text=True,env=env));save_json(out/'runtime.json',versions);save_json(out/'config.json',cfg);save_json(out/'data_contract.json',safe);shutil.copy2(src/'best.pth',out/'best.pth');shutil.copy2(src/'best_validation.json',out/'best_validation.json')
 for p in repo.rglob('*.py'):
  if '.git' in p.parts:continue
  dest=out/'source'/p.relative_to(repo);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
 save_json(out/'provenance.json',{'checkpoint_sha256':hashlib.sha256((out/'best.pth').read_bytes()).hexdigest(),'source_job':str(src),'source_tree':str(repo),'copied_files':[str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()],'no_test_artifact_copied':True,'oracle':False})
 return out

def predict(z,dates,ts,cfg):
 dt=pd.to_datetime(dates);marks=np.stack([np.asarray(dt.hour)/23-.5,np.asarray(dt.dayofweek)/6-.5,(np.asarray(dt.day)-1)/30-.5,(np.asarray(dt.dayofyear)-1)/365-.5],-1).astype(np.float32);x=np.stack([z[t-512:t] for t in ts]);m=np.stack([marks[t-512:t] for t in ts]);env=os.environ.copy();env.pop('PYTHONPATH',None)
 with tempfile.TemporaryDirectory(prefix='legacy_history_',dir=RUN) as tmp:
  tmp=pathlib.Path(tmp);np.savez(tmp/'input.npz',history=x,marks=m);subprocess.run([BASEPY,str(pathlib.Path(__file__).with_name('legacy_worker.py')),cfg['legacy_model_dir'],str(tmp/'input.npz'),str(tmp/'output.npy')],check=True,env=env);p=np.load(tmp/'output.npy');audit=json.loads((tmp/'output.json').read_text());assert audit['other_query_perturbation_max_delta']<1e-5
 return p

def run(method):
 assert method in ('patchtst','itransformer','frets');name=f'20261105_legacy_{method}_mse';out=ROOT/'runs'/name;out.mkdir(exist_ok=True);subprocess.run([sys.executable,str(ROOT/'src/preflight.py'),'--root',str(ROOT),'--run-name',name],check=True);guard(2400);model=prepare(method);cfg={'engine':'legacy_numeric_tool_v1','method':method,'legacy_model_dir':str(model),'oracle':False,'training_source':'historical identical ETTh1 train/val benchmark, frozen base_zero checkpoint; no new training','runtime_python':BASEPY,'code_revision':git_head()};save_json(out/'config.json',cfg);z=np.load(RUN/'cache/trainval_z.npy');dates=json.loads((RUN/'cache/dates.json').read_text());results={}
 for split,ts in json.loads((RUN/'cache/window_manifest.json').read_text()).items():
  p=predict(z,dates,ts,cfg);np.save(out/f'{split}_predictions.npy',p);results[split]=metric(p,np.load(RUN/'cache'/f'{split}_targets.npy'));save_json(out/'partial_metrics.json',results)
 old=json.loads((model/'best_validation.json').read_text())['validation']['mse'];delta=abs(results['val']['mse']-old);save_json(out/'validation_reproduction.json',{'historical_validation_mse':old,'recomputed_validation_mse':results['val']['mse'],'absolute_difference':delta,'tolerance':1e-5,'test_results_read':False});assert delta<1e-5,('historical validation mismatch',delta)
 save_json(out/'metrics.json',{'candidate':name,'metrics':results,'status':'evaluated'});print(name,{s:m['mse'] for s,m in results.items()},flush=True)
if __name__=='__main__':run(sys.argv[1])
