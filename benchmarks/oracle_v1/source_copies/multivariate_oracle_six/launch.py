"""User-authorized six-dataset/210-job suite, two GPUs, shared10h clock."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
from dataset import dump,sha,DATASETS
from alignment import historical_run,load_historical_config

METHODS=('dlinear','patchtst','itransformer','frets','fredf')
VARIANTS=[('base','zero')]+[(s,l) for s in ('A','B') for l in ('zero','l1','l2')]
STOP=False
def stop(*_):
    global STOP
    STOP=True
signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)

def summary(out):
    rows=[]; pairs=[]
    for d in DATASETS:
        for p in sorted((out/d/'jobs').glob('*/metrics.json')):
            r=json.loads(p.read_text()); r['job']=p.parent.name; rows.append(r)
        for m in METHODS:
            for s in ('A','B'):
                paths=[out/d/'jobs'/f'{m}_{s}_{l}' for l in ('zero','l1','l2')]
                if not all((p/'metrics.json').exists() for p in paths):continue
                initial=[json.loads((p/'initial_hash.json').read_text()) for p in paths]
                orders=[[json.loads(x)['sample_order_sha256'] for x in (p/'train.jsonl').read_text().splitlines()] for p in paths]
                n=min(map(len,orders))
                assert initial[0]==initial[1]==initial[2], 'Paired initial hash mismatch'
                assert orders[0][:n]==orders[1][:n]==orders[2][:n], 'Paired sampler mismatch'
                pairs.append(dict(dataset=d,method=m,strategy=s,initialization_match=True,common_epoch_orders_match=True,common_epochs=n))
    for r in rows:
        b=next((x for x in rows if x['dataset']==r['dataset'] and x['method']==r['method'] and x['strategy']=='base'),None)
        z=next((x for x in rows if x['dataset']==r['dataset'] and x['method']==r['method'] and x['strategy']==r['strategy'] and x['label']=='zero'),None)
        if b:r['delta_mse_vs_base']=r['mse']-b['mse']
        if z:r['delta_mse_vs_matched_zero']=r['mse']-z['mse']
    dump(out/'comparison.json',rows); dump(out/'paired_checks.json',pairs)

def audit_baselines(project,out,d):
    rows=[]
    for m in METHODS:
        old=historical_run(project,m,d)
        p=np.load(old/'predictions_f32.npy',mmap_mode='r'); y=np.load(old/'targets_f32.npy',mmap_mode='r')
        sq=ab=n=0
        for i in range(0,len(p),128):
            e=p[i:i+128].astype(np.float64)-y[i:i+128]; sq+=(e*e).sum(); ab+=np.abs(e).sum(); n+=e.size
        new=json.loads((out/d/'jobs'/f'{m}_base_zero'/'metrics.json').read_text())
        rows.append(dict(method=m,old_mse=float(sq/n),old_mae=float(ab/n),new_mse=new['mse'],new_mae=new['mae'],
                         relative_mse_difference=abs(new['mse']-sq/n)/(sq/n)))
    dump(out/d/'baseline_comparison.json',dict(rows=rows,policy='Record differences; user accepted nonblocking reproduction deviations'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parent
    a.output.mkdir(parents=True,exist_ok=False)
    start=time.time();deadline=start+36000
    dump(a.output/'budget.json',dict(start_epoch=start,deadline_epoch=deadline,max_seconds=36000,gpus=[0,1],datasets=list(DATASETS),
         max_jobs=210,monitor_seconds=600,automatic_retries=False,exclude_exchange=True))
    queue=[(d,m,s,l) for s,l in VARIANTS for m in METHODS for d in DATASETS]
    active={}; finished=[]; failure=None; audited=set();last_resource=0
    def bounded(command,log,gpu=None,limit=600):
        if STOP or time.time()>=deadline-60:raise TimeoutError('Budget exhausted during preparation')
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
        with log.open('w') as f:
            subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=min(limit,max(1,deadline-time.time()-60)))
    try:
        dump(a.output/'status.json',dict(status='preparing',phase='data_and_contracts',finished=[],pending_count=210))
        sources=[a.project/'third_party/LTSF-Linear',a.project/'research/baselines_512_64']
        dump(a.output/'source_versions.json',dict(commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
             dependencies={str(p):sha(p) for source in sources for p in source.rglob('*.py')}))
        for d in DATASETS:
            (a.output/d/'jobs').mkdir(parents=True)
            for m in METHODS:load_historical_config(a.project,m,d)
            bounded([sys.executable,str(root/'dataset.py'),'--project',str(a.project),'--dataset',d,'--cache',str(a.output/d/'labels')],
                    a.output/d/'cache.log',limit=1800)
            for m in METHODS:
                bounded([sys.executable,str(root/'check_contract.py'),'--project',str(a.project),'--dataset',d,'--method',m],
                        a.output/d/(m+'_contract.log'),gpu=1,limit=300)
            dump(a.output/'status.json',dict(status='preparing',phase='data_and_contracts',last_prepared=d,finished=[],pending_count=210))
        while queue or active:
            if STOP or time.time()>=deadline-60:raise TimeoutError('Shared10h deadline/stop reached')
            for gpu,j in list(active.items()):
                code=j['process'].poll()
                if code is not None:
                    j['log'].close();finished.append(dict(dataset=j['dataset'],job=j['name'],gpu=gpu,returncode=code));del active[gpu]
                    if code:raise RuntimeError('Failed '+j['dataset']+'/'+j['name'])
                    summary(a.output)
            for d in DATASETS:
                if d not in audited and all((a.output/d/'jobs'/f'{m}_base_zero'/'metrics.json').exists() for m in METHODS):
                    audit_baselines(a.project,a.output,d);audited.add(d)
            mem=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
            free={int(x.split(',')[0]):int(x.split(',')[1]) for x in mem.strip().splitlines()}
            for gpu in (0,1):
                if gpu in active:continue
                eligible=[]
                for i,(d,m,s,l) in enumerate(queue):
                    # Largest known historical footprint16.6GiB; leave generous headroom.
                    required=26000 if d=='electricity' and m=='frets' else 13000
                    if free[gpu]<required:continue
                    if s!='base' and d not in audited:continue
                    eligible.append(i)
                if not eligible:continue
                d,m,s,l=queue.pop(eligible[0]);name=f'{m}_{s}_{l}';dest=a.output/d
                subprocess.run([sys.executable,str(root/'preflight.py'),'--root',str(root),'--run-name',f'20261014_{d}_{name}_mse',
                                '--data-version','six-datasets-multivariate-oracle-v1; frozen manifest'],check=True)
                cmd=[sys.executable,str(root/'train.py'),'--project',str(a.project),'--dataset',d,'--cache',str(dest/'labels'),
                     '--output',str(dest/'jobs'/name),'--method',m,'--strategy',s,'--label',l,'--deadline',str(deadline-60)]
                if s=='B':cmd+=['--baseline',str(dest/'jobs'/f'{m}_base_zero')]
                log=(dest/(name+'.log')).open('w')
                proc=subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'),
                                      stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[gpu]=dict(process=proc,log=log,dataset=d,name=name)
                dump(dest/(name+'_launch.json'),dict(command=cmd,pid=proc.pid,gpu=gpu,started=time.time()))
            if time.time()-last_resource>=600:
                with (a.output/'resources.log').open('a') as f:f.write(str(time.time())+'\n'+subprocess.check_output(['nvidia-smi'],text=True))
                last_resource=time.time()
            dump(a.output/'status.json',dict(status='running',finished=finished,pending=queue,
                 active={str(g):dict(dataset=j['dataset'],job=j['name'],pid=j['process'].pid) for g,j in active.items()}))
            if queue or active:time.sleep(5)
        summary(a.output)
        assert len(finished)==210
    except BaseException as e:failure=repr(e)
    finally:
        for j in active.values():
            if j['process'].poll() is None:os.killpg(j['process'].pid,signal.SIGTERM)
        until=min(time.time()+30,deadline-5)
        while any(j['process'].poll() is None for j in active.values()) and time.time()<until:time.sleep(1)
        for j in active.values():
            if j['process'].poll() is None:os.killpg(j['process'].pid,signal.SIGKILL)
            j['process'].wait();j['log'].close()
        dump(a.output/'status.json',dict(status='failed' if failure else 'complete',error=failure,finished=finished,pending=queue,
             interrupted=[dict(dataset=j['dataset'],job=j['name']) for j in active.values()],elapsed_seconds=time.time()-start))
    if failure:raise RuntimeError(failure)

if __name__=='__main__':main()
