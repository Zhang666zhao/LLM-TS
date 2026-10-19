"""Authorized five-dataset continuation; original training implementation unchanged."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

STOP = False
def stop(*args):
    global STOP
    STOP = True
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)

def main():
    p = argparse.ArgumentParser()
    for name in ('project', 'previous', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    source = a.project/'research/multivariate_oracle_six'
    sys.path.insert(0, str(source))
    import launch as original
    from dataset import dump, sha
    datasets = ('ETTh1','ETTh2','ETTm1','ETTm2','weather')
    assert a.previous != a.output
    assert json.loads((a.previous/'status.json').read_text())['status'] == 'failed'
    assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip() == 'ad69667428a419212e8e38fd79eaf4893562b8a0'
    assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()
    a.output.mkdir(parents=True, exist_ok=False)
    start = time.time(); deadline = start+36000
    dump(a.output/'budget.json',dict(start_epoch=start,deadline_epoch=deadline,max_seconds=36000,gpus=[0,1],datasets=datasets,max_new_jobs=66,prior_completed=109,total_target=175,automatic_retries=False,monitor_seconds=600))
    dump(a.output/'authorization.json',dict(scope='User authorized excluding Electricity, two A6000 GPUs, NEW 10h, unchanged configs',previous=str(a.previous),interrupted_restart='weather/patchtst_B_zero from seed; no full optimizer/RNG checkpoint',source_commit='ad69667428a419212e8e38fd79eaf4893562b8a0',launcher_sha256=sha(Path(__file__)),command=sys.argv))
    queue=[]; reused=[]; finished=[]; active={}; failure=None
    try:
        saved=json.loads((a.previous/'source_versions.json').read_text())
        for path,digest in saved['dependencies'].items():
            assert sha(Path(path)) == digest, 'Dependency changed: '+path
        for d in datasets:
            dest=a.output/d; (dest/'jobs').mkdir(parents=True)
            (dest/'labels').symlink_to(a.previous/d/'labels',target_is_directory=True)
            for m in original.METHODS:
                assert (a.previous/d/(m+'_contract.log')).exists()
            for s,l in original.VARIANTS:
                for m in original.METHODS:
                    name=f'{m}_{s}_{l}'; old=a.previous/d/'jobs'/name
                    if (old/'metrics.json').exists():
                        assert json.loads((old/'status.json').read_text())['status']=='complete'
                        assert json.loads((old/'metrics.json').read_text())['exact_targets_verified']
                        (dest/'jobs'/name).symlink_to(old,target_is_directory=True)
                        reused.append(dict(dataset=d,job=name,source=str(old)))
        for s,l in original.VARIANTS:
            for m in original.METHODS:
                for d in datasets:
                    if not (a.output/d/'jobs'/f'{m}_{s}_{l}'/'metrics.json').exists(): queue.append((d,m,s,l))
        assert len(reused)==109 and len(queue)==66
        dump(a.output/'reused.json',reused);dump(a.output/'planned_jobs.json',queue)
        original.summary(a.output)
        last_resource=0
        while queue or active:
            if STOP or time.time()>=deadline-60: raise TimeoutError('New continuation10h deadline/stop reached')
            for gpu,j in list(active.items()):
                code=j['process'].poll()
                if code is not None:
                    j['log'].close();finished.append(dict(dataset=j['dataset'],job=j['name'],gpu=gpu,returncode=code));del active[gpu]
                    if code: raise RuntimeError('Failed '+j['dataset']+'/'+j['name'])
                    original.summary(a.output)
            mem=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
            free={int(x.split(',')[0]):int(x.split(',')[1]) for x in mem.strip().splitlines()}
            for gpu in (0,1):
                if gpu in active or not queue or free[gpu]<13000: continue
                d,m,s,l=queue.pop(0); name=f'{m}_{s}_{l}';dest=a.output/d
                subprocess.run([sys.executable,str(source/'preflight.py'),'--root',str(source),'--run-name',f'20261015_{d}_{name}_mse','--data-version','unchanged frozen six-dataset-v1; five-dataset continuation'],check=True)
                cmd=[sys.executable,str(source/'train.py'),'--project',str(a.project),'--dataset',d,'--cache',str(dest/'labels'),'--output',str(dest/'jobs'/name),'--method',m,'--strategy',s,'--label',l,'--deadline',str(deadline-60)]
                if s=='B':cmd+=['--baseline',str(dest/'jobs'/f'{m}_base_zero')]
                log=(dest/(name+'.log')).open('x')
                proc=subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[gpu]=dict(process=proc,log=log,dataset=d,name=name)
                dump(dest/(name+'_launch.json'),dict(command=cmd,pid=proc.pid,gpu=gpu,started=time.time()))
            if time.time()-last_resource>=600:
                with (a.output/'resources.log').open('a') as f:f.write(str(time.time())+'\n'+subprocess.check_output(['nvidia-smi'],text=True))
                last_resource=time.time()
            dump(a.output/'status.json',dict(status='running',reused_count=109,finished=finished,pending=queue,active={str(g):dict(dataset=j['dataset'],job=j['name'],pid=j['process'].pid) for g,j in active.items()}))
            if queue or active:time.sleep(5)
        original.summary(a.output)
        assert len(finished)==66
        assert len(json.loads((a.output/'comparison.json').read_text()))==175
        assert len(json.loads((a.output/'paired_checks.json').read_text()))==50
    except BaseException as e:failure=repr(e)
    finally:
        for j in active.values():
            if j['process'].poll() is None:os.killpg(j['process'].pid,signal.SIGTERM)
        until=min(time.time()+30,deadline-5)
        while any(j['process'].poll() is None for j in active.values()) and time.time()<until:time.sleep(1)
        for j in active.values():
            if j['process'].poll() is None:os.killpg(j['process'].pid,signal.SIGKILL)
            j['process'].wait();j['log'].close()
        dump(a.output/'status.json',dict(status='failed' if failure else 'complete',error=failure,reused_count=len(reused),finished=finished,pending=queue,interrupted=[dict(dataset=j['dataset'],job=j['name']) for j in active.values()],elapsed_seconds=time.time()-start))
    if failure:raise RuntimeError(failure)

if __name__=='__main__':main()
