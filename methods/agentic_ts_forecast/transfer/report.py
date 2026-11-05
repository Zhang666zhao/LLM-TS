"""Summarize the already frozen experiment; package and smoke-check a deployment."""
import csv,sys,subprocess,tempfile,zipfile,shutil,json,platform
from pathlib import Path
import numpy as np
import torch
from forecast import digest,read,write,COLS

def main():
    root=Path(__file__).resolve().parent;p=root/'artifacts/ETTm1'
    f=read(p/'frozen_plan.json');t=read(p/'test.json');v=read(p/'validation_reconstruction.json')
    base=t['results']['DLinear']['mse']
    fixed={'M'+str(i) for i in range(1,11)};rows=[]
    for k,r in t['results'].items():
        family=('baseline' if k=='DLinear' else 'fixed' if k in fixed else 'LLM' if k.startswith('LLM') else 'six-rule' if k.startswith('Control') else 'analytic')
        rows.append(dict(method=k,family=family,val_mse=v[k]['mse'],val_mae=v[k]['mae'],
                         test_mse=r['mse'],test_mae=r['mae'],gain_vs_dlinear_pct=100*(1-r['mse']/base)))
    with (p/'results.csv').open('w',newline='') as file:
        w=csv.DictWriter(file,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    bundle=root/'ETTm1_deployment.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as zipf:
        for n in ['forecast.py','predict.py','DEPLOY.md']:zipf.write(root/n,n)
        for n in ['tools.npz','frozen_plan.json','freeze.json']:zipf.write(p/n,n)
    with tempfile.TemporaryDirectory() as tmp:
        wd=Path(tmp)
        with zipfile.ZipFile(bundle) as z:z.extractall(wd)
        # The latest 512 rows that precede the first test target constitute
        # a legitimate rolling input. Only these rows are exposed to the CLI.
        with (root/'assets/ETTm1.csv').open() as file:
            rdr=csv.DictReader(file);past_rows=[next(rdr) for _ in range(46080)][-512:]
        with (wd/'history.csv').open('w',newline='') as file:
            w=csv.DictWriter(file,fieldnames=['date']+COLS);w.writeheader();w.writerows(past_rows)
        subprocess.run([sys.executable,'-I', '-c',
          "import sys,runpy;sys.path.insert(0,'.');runpy.run_path('predict.py',run_name='__main__')",
          '--plan','frozen_plan.json','--history','history.csv','--output','forecast.csv'],check=True,cwd=wd)
        with (wd/'forecast.csv').open() as file:
            out=list(csv.DictReader(file))
        a=np.array([[float(r[k]) for k in COLS] for r in out])
        ref=np.load(p/'test_predictions'/(f['selected']+'.npy'),mmap_mode='r')[0]
        delta=float(np.max(abs((a-f['mean'])/f['std']-ref)))
        assert a.shape==(64,7) and np.isfinite(a).all() and delta<1e-4
        shutil.copyfile(wd/'forecast.csv',p/'example_forecast.csv')
        shutil.copyfile(wd/'history.csv',p/'example_history.csv')
    groups={}
    for group in ['fixed','LLM','six-rule','analytic']:
        row=min([r for r in rows if r['family']==group],key=lambda x:x['val_mse'])
        groups[group]=row
    aliases=[]
    ready=read(p/'predictions_ready.json')
    for names in [['M2','M3'],['M8','M9'],['M1','LLM5']]:
        if len({ready['files'][k] for k in names})==1:aliases.append(names)
    write(p/'summary.json',dict(dataset='ETTm1',variant=f['variant'],selected=f['selected'],best_by_validation=groups,
          full_test_windows=t['n'],context_steps=512,forecast_steps=64,history_hours=128,forecast_hours=16,
          deployment_clean_directory_delta=delta,deploy_zip_sha256=digest(bundle),equal_test_outputs=aliases,
          historical_dlinear_test_mse=.29252926863834333,matched_dlinear_delta=base-.29252926863834333,
          platform=platform.platform(),numpy=np.__version__,torch=torch.__version__,source_hashes={
              n:digest(root/n) for n in ['fit.py','forecast.py','controls.py','run_pipeline.py','predict.py','contract.json']}))
    print('DEPLOYMENT_VERIFIED',a.shape,delta,groups,aliases)

if __name__=='__main__':main()
