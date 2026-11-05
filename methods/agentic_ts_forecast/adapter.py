"""LLM-TS adapter: compute test metrics using the preselected program for each dataset."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

HERE=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('dataset','output','python','run-id','data-manifest-sha256','benchmark-id','artifact-root','data-root'):
        ap.add_argument('--'+key,required=True)
    ap.add_argument('--seed',type=int,default=2021);ap.add_argument('--gpu',type=int)
    ap.add_argument('--batch-size',type=int);ap.add_argument('--dry-run',action='store_true')
    a=ap.parse_args()
    if a.seed!=2021:ap.error('Frozen experiment uses seed 2021')
    if a.dataset=='ETTh1':
        entry=HERE/'run_local.py';method='M1'
    elif a.dataset=='ETTm1':
        entry=HERE/'transfer/reproduce.py'
        method=json.loads((HERE/'transfer/artifacts/ETTm1/frozen_plan.json').read_text())['selected']
    else:ap.error('Choose ETTh1 or ETTm1')
    cmd=[a.python,str(entry),'--split','test']
    if a.dataset=='ETTh1':cmd+=['--method','M1']
    if a.dry_run:print(cmd);return
    output=Path(a.output)
    if (output/'metrics.json').exists():ap.error('Choose a new run ID')
    log=subprocess.run(cmd,cwd=HERE,text=True,stdout=subprocess.PIPE,check=True)
    report=json.loads(Path(log.stdout.strip().splitlines()[-1].removeprefix('Report: ')).read_text())
    row=next(r for r in report['metrics'] if r['split']=='test' and r['method']==method)
    if not row['passed']:raise ValueError('Frozen metric comparison failed')
    metrics=dict(schema_version='1.0',benchmark_id=a.benchmark_id,run_id=a.run_id,
        method='agentic_ts_forecast',configuration=method,dataset=a.dataset,seed=a.seed,
        context_length=512,prediction_length=64,metrics={'mse':row['mse'],'mae':row['mae']},
        sample_count=row['windows']*7,value_count=row['windows']*7*64,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
        data_manifest_sha256=a.data_manifest_sha256,status='complete')
    output.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',dir=output,delete=False) as tmp:
        json.dump(metrics,tmp,indent=2)
    Path(tmp.name).replace(output/'metrics.json')
    print(output/'metrics.json')

if __name__=='__main__':main()
