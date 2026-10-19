"""Fixed 35 jobs, two GPUs, one 10h deadline; baseline reproduction gate."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
from dataset import dump,prepare,sha
from alignment import historical_run

METHODS=('dlinear','patchtst','itransformer','frets','fredf')
STOP=False
def stop(*_):
    global STOP
    STOP=True
signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)

def baseline_gate(project,output):
    rows=[]
    for m in METHODS:
        old=historical_run(project,m,'exchange_rate')
        new=output/'jobs'/(m+'_base_zero')
        y=np.load(old/'targets_f32.npy',mmap_mode='r'); p=np.load(old/'predictions_f32.npy',mmap_mode='r')
        assert np.array_equal(y,np.load(new/'targets_f32.npy',mmap_mode='r'))
        e=p.astype(np.float64)-y
        oldm=dict(mse=float((e*e).mean()),mae=float(np.abs(e).mean()))
        now=json.loads((new/'metrics.json').read_text())
        rel={k:abs(now[k]-oldm[k])/max(abs(oldm[k]),1e-12) for k in oldm}
        rows.append(dict(method=m,old=oldm,new={k:now[k] for k in oldm},relative_difference=rel,
                         passed=max(rel.values())<=.01))
    dump(output/'baseline_gate.json',dict(tolerance_relative=.01,rows=rows,passed=all(r['passed'] for r in rows)))
    if not all(r['passed'] for r in rows): raise RuntimeError('Baseline reproduction differs by >1%; stop for diagnosis, no oracle jobs launched')

def summarize(output):
    rows=[]
    for p in sorted((output/'jobs').glob('*/metrics.json')):
        r=json.loads(p.read_text()); r['job']=p.parent.name; rows.append(r)
    for r in rows:
        base=next(x for x in rows if x['method']==r['method'] and x['strategy']=='base')
        r['delta_mse_vs_base']=r['mse']-base['mse']
        if r['strategy']!='base':
            control=next(x for x in rows if x['method']==r['method'] and x['strategy']==r['strategy'] and x['label']=='zero')
            r['delta_mse_vs_matched_zero']=r['mse']-control['mse']
    pairing=[]
    for m in METHODS:
        for s in ('A','B'):
            paths=[output/'jobs'/('_'.join((m,s,l))) for l in ('zero','l1','l2')]
            hashes=[json.loads((p/'initial_hash.json').read_text()) for p in paths]
            assert hashes[0]==hashes[1]==hashes[2], 'Paired initialization differs'
            orders=[[json.loads(line)['sample_order_sha256'] for line in (p/'train.jsonl').read_text().splitlines()] for p in paths]
            common=min(map(len,orders))
            assert orders[0][:common]==orders[1][:common]==orders[2][:common], 'Paired sample order differs'
            pairing.append(dict(method=m,strategy=s,initialization_match=True,common_epoch_orders_match=True,common_epochs=common))
    dump(output/'paired_checks.json',pairing)
    dump(output/'comparison.json',rows)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--project',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--continue-after-gate',action='store_true',help='Explicit user-approved continuation of untouched30 jobs; retain original deadline')
    a=p.parse_args(); root=Path(__file__).resolve().parent
    active={}; finished=[]; failure=None; last_resource=0; phase='baselines'
    queue=[(m,'base','zero') for m in METHODS]
    if a.continue_after_gate:
        previous=json.loads((a.output/'status.json').read_text())
        assert previous['status']=='failed' and previous['phase']=='baselines'
        assert 'Baseline reproduction differs' in previous['error']
        assert set(p.name for p in (a.output/'jobs').iterdir())=={m+'_base_zero' for m in METHODS}
        for m in METHODS:
            assert json.loads((a.output/'jobs'/(m+'_base_zero')/'status.json').read_text())['status']=='complete'
        budget=json.loads((a.output/'budget.json').read_text())
        started=budget['start_epoch']; deadline=budget['deadline_epoch']
        assert time.time()<deadline-60, 'Original budget expired'
        # Exclusive marker prevents accidental repeat continuation or baseline overwrite.
        with (a.output/'continuation_authorization.json').open('x') as f:
            json.dump(dict(time=time.time(),reason='User explicitly accepted >1% baseline deviation and requested remaining20 oracle+10 controls',
                original_deadline=deadline,baseline_reruns=0,model_code_unchanged=True,
                actual_rng='CPU/backbone seed2021; wrapper resets CUDA RNG2022, retained for paired consistency',
                commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()),f,indent=2)
        dump(a.output/'status_before_authorized_continuation.json',previous)
        finished=previous['finished']; phase='oracle_and_controls'
        queue=[(m,s,l) for s in ('A','B') for l in ('zero','l1','l2') for m in METHODS]
    else:
        a.output.mkdir(parents=True,exist_ok=False); (a.output/'jobs').mkdir()
        started=time.time(); deadline=started+36000
        dump(a.output/'budget.json',dict(start_epoch=started,deadline_epoch=deadline,max_seconds=36000,gpus=[0,1],
             max_jobs=35,dataset='exchange_rate',scope='multivariate oracle stage2 fixed matrix',monitor_seconds=600,
             stop_on_failure=True,automatic_retries=False))
    try:
        if not a.continue_after_gate:
            sources=[a.project/'third_party/LTSF-Linear',a.project/'research/baselines_512_64']
            dump(a.output/'source_versions.json',dict(code_commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
                 dependencies={str(p):sha(p) for source in sources for p in source.rglob('*.py')}))
            prepare(a.project,a.output/'labels')
            # Cache created once, before any training. GPU contracts are separate from optimizer runs.
            for m in METHODS:
                log=a.output/(m+'_contract.log')
                with log.open('w') as f:
                    subprocess.run([sys.executable,str(root/'check_contract.py'),'--project',str(a.project),'--method',m],
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',PYTHONDONTWRITEBYTECODE='1'),stdout=f,stderr=subprocess.STDOUT,
                        check=True,timeout=min(180,max(1,deadline-time.time()-60)))
        while queue or active or phase=='baselines':
            if STOP or time.time()>=deadline-60: raise TimeoutError('Shared 10h deadline or stop request')
            for gpu,j in list(active.items()):
                code=j['process'].poll()
                if code is not None:
                    j['log'].close(); finished.append(dict(job=j['name'],gpu=gpu,returncode=code)); del active[gpu]
                    if code: raise RuntimeError('Failed job '+j['name'])
            if phase=='baselines' and not queue and not active:
                baseline_gate(a.project,a.output); phase='oracle_and_controls'
                queue=[(m,s,l) for s in ('A','B') for l in ('zero','l1','l2') for m in METHODS]
            text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
            free={int(x.split(',')[0]):int(x.split(',')[1]) for x in text.strip().splitlines()}
            for gpu in (0,1):
                if not queue or gpu in active or free[gpu]<12288: continue
                m,s,l=queue.pop(0); name='_'.join((m,s,l))
                subprocess.run([sys.executable,str(root/'preflight.py'),'--root',str(root),'--run-name','20261014_'+name+'_mse',
                                '--data-version','exchange-multivariate-oracle-v1; frozen manifest and label hashes checked'],check=True)
                command=[sys.executable,str(root/'train.py'),'--project',str(a.project),'--cache',str(a.output/'labels'),
                         '--output',str(a.output/'jobs'/name),'--method',m,'--strategy',s,'--label',l,'--deadline',str(deadline-60)]
                if s=='B': command+=['--baseline',str(a.output/'jobs'/(m+'_base_zero'))]
                log=(a.output/(name+'.log')).open('w')
                proc=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                       PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[gpu]=dict(process=proc,log=log,name=name)
                dump(a.output/(name+'_launch.json'),dict(command=command,pid=proc.pid,gpu=gpu,started=time.time()))
            if time.time()-last_resource>600:
                with (a.output/'resources.log').open('a') as f: f.write(str(time.time())+'\n'+subprocess.check_output(['nvidia-smi'],text=True))
                last_resource=time.time()
            dump(a.output/'status.json',dict(status='running',phase=phase,finished=finished,pending=queue,
                 active={str(g):dict(job=j['name'],pid=j['process'].pid) for g,j in active.items()}))
            if queue or active: time.sleep(5)
        summarize(a.output)
    except BaseException as e: failure=repr(e)
    finally:
        for j in active.values():
            if j['process'].poll() is None: os.killpg(j['process'].pid,signal.SIGTERM)
        until=min(time.time()+30,deadline-5)
        while any(j['process'].poll() is None for j in active.values()) and time.time()<until: time.sleep(1)
        for j in active.values():
            if j['process'].poll() is None: os.killpg(j['process'].pid,signal.SIGKILL)
            j['process'].wait(); j['log'].close()
        dump(a.output/'status.json',dict(status='failed' if failure else 'complete',error=failure,phase=phase,finished=finished,elapsed_seconds=time.time()-started))
    if failure: raise RuntimeError(failure)

if __name__=='__main__': main()
