"""One-shot 40-job suite. No retries, API requests, tuning, or deadline reset."""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from worker import bind, ROOT, PINNED

STOP = False
def stop(*_):
    global STOP
    STOP=True
signal.signal(signal.SIGTERM,stop)
signal.signal(signal.SIGINT,stop)

def prepare(a,t):
    import numpy as np
    values,marks,info=t.load_data(a.project,'ETTh1')
    source=a.project/'runs/20261019_dao_etth1_allsplits_v1/labels'
    manifest=json.loads((source/'manifest.json').read_text())
    assert manifest['complete'] and manifest['source_data_sha256']==info['source_sha256']
    assert manifest['variables']==info['variables']
    cache=a.run/'llm_cache'; cache.mkdir()
    hashes={}
    for split,n in [('train',8065),('val',2817),('test',2817)]:
        path=source/(split+'.npz')
        assert t.sha(path)==manifest['splits'][split]['sha256']
        with np.load(path) as z:
            expected=np.arange(info['split'][split+'_start'],info['split'][split+'_end']-575)
            assert len(expected)==n and np.array_equal(z['starts'],expected)
            assert np.array_equal(z['target_starts'],expected+512)
            assert np.array_equal(z['anchor_target_starts'],expected[0]+512+32*(np.arange(n)//32))
            assert np.array_equal(z['label_age_steps'],np.arange(n)%32)
            for label in ('l1','l2'):
                assert z[label].shape==(n,7) and z[label].dtype.kind=='i' and np.isin(z[label],[-1,0,1]).all()
        shutil.copy2(path,cache/path.name); hashes[split]=t.sha(cache/path.name)
    # Historical trainer requires original numerical-data metadata. Keep the full
    # upstream LLM manifest separately; numerical scaling is NOT the API scaler.
    t.dump(cache/'manifest.json',dict(info,cache_sha256=hashes,label_source='LLM predicted',oracle=False))
    shutil.copy2(source/'manifest.json',cache/'llm_source_manifest.json')
    refs={}
    for method in t.METHODS:
        for suffix in ['base_zero']+[f'{s}_{l}' for s in ('A','B') for l in ('l1','l2','zero')]:
            job=a.oldjobs/f'{method}_{suffix}'
            assert json.loads((job/'status.json').read_text())['status']=='complete'
            assert (job/'best.pth').is_file()
            refs[job.name]=dict(path=str(job),checkpoint_sha256=t.sha(job/'best.pth'),
                               config_sha256=t.sha(job/'config.json'),metrics_sha256=t.sha(job/'metrics.json'))
    t.dump(a.run/'historical_references.json',refs)
    t.dump(a.run/'data_version.json',info)
    (a.run/'data_version.txt').write_text(info['source_sha256']+'\n')
    t.dump(a.run/'label_contract.json',dict(passed=True,windows=13699,variables=7,hashes=hashes,
           numeric_scaler='original full training scaler',LLM_evidence_scaler='causal prefix during train'))
    t.dump(a.run/'source_snapshot.json',dict(historical=PINNED,
        downstream={p.name:t.sha(p) for p in ROOT.glob('*.py')}))
    snapshot=a.run/'code_snapshot'; snapshot.mkdir()
    for path in ROOT.glob('*'):
        if path.is_file(): shutil.copy2(path,snapshot/path.name)
    (a.run/'env_snapshot.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))

def free_memory():
    out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    return {int(row.split(',')[0]):int(row.split(',')[1]) for row in out.splitlines()}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',type=Path,required=True); p.add_argument('--run',type=Path,required=True)
    a=p.parse_args(); t=bind(a.project)
    a.oldjobs=a.project/'runs/20261015_five_multivariate_oracle_continue_v1/ETTh1/jobs'
    a.run.mkdir(parents=True,exist_ok=False)
    start=time.time(); deadline=start+36000
    t.dump(a.run/'budget.json',dict(start=start,deadline=deadline,max_seconds=36000,max_training_jobs=20,
           max_inference_jobs=20,gpus=[0,1],note='new conservative safety ceiling; fixed suite, no search'))
    t.dump(a.run/'pid.json',dict(pid=os.getpid()))
    running={}; completed=[]; pending=[]; handles=[]
    def status(state,**extra):
        t.dump(a.run/'status.json',dict(status=state,completed=completed,pending=pending,
               running={str(g):dict(job=x[0],pid=x[1].pid) for g,x in running.items()},
               total=40,updated_at=time.time(),**extra))
    try:
        status('preparing'); prepare(a,t)
        # Synthetic contract checks do not train or update an optimizer.
        status('contracts')
        for method in t.METHODS:
            while free_memory().get(1,0)<13312:
                if STOP or time.time()>=deadline: raise TimeoutError('GPU contract wait deadline')
                time.sleep(5)
            command=[sys.executable,str(a.project/'research/multivariate_oracle_six/check_contract.py'),
                     '--project',str(a.project),'--method',method,'--dataset','ETTh1']
            result=subprocess.run(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES='1'),capture_output=True,text=True,
                                  timeout=max(1,min(300,deadline-time.time())))
            (a.run/(method+'_contract.log')).write_text(result.stdout+result.stderr)
            assert result.returncode==0, method+' contract failed'
        jobs=[f'{g}__{m}__{s}__{l}' for g in ('G1','G2') for m in t.METHODS for s in ('A','B') for l in ('l1','l2')]
        t.dump(a.run/'task_manifest.json',dict(jobs=jobs,total=40,reused_oracle=20,reused_base=5,reused_zero=10))
        pending=jobs.copy(); (a.run/'jobs').mkdir(); (a.run/'logs').mkdir()
        while pending or running:
            if STOP or time.time()>=deadline: raise TimeoutError('Stopped or shared deadline reached')
            for gpu,(key,proc) in list(running.items()):
                code=proc.poll()
                if code is not None:
                    del running[gpu]
                    if code: raise RuntimeError(f'{key} failed exit={code}')
                    completed.append(key)
            free=free_memory()
            for gpu in (0,1):
                if pending and gpu not in running and free.get(gpu,0)>=13312:
                    key=pending.pop(0); group,method,strategy,label=key.split('__')
                    command=[sys.executable,str(ROOT/'worker.py'),'--project',str(a.project),'--cache',str(a.run/'llm_cache'),
                             '--oldjobs',str(a.oldjobs),'--output',str(a.run/'jobs'/key),'--group',group,'--method',method,
                             '--strategy',strategy,'--label',label,'--deadline',str(deadline)]
                    f=(a.run/'logs'/(key+'.log')).open('w'); handles.append(f)
                    proc=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,
                                          env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)))
                    running[gpu]=(key,proc)
                    print(json.dumps(dict(started=key,gpu=gpu,pid=proc.pid)),flush=True)
            status('running'); time.sleep(3)
        status('verifying')
        result=subprocess.run([sys.executable,str(ROOT/'verify.py'),'--project',str(a.project),'--run',str(a.run)],
                              timeout=max(1,deadline-time.time()))
        assert result.returncode==0,'Final verification failed'
        status('complete'); t.dump(a.run/'completion.json',dict(complete=True,seconds=time.time()-start,finished_at=time.time()))
    except BaseException as e:
        for key,proc in running.values():
            if proc.poll() is None: os.killpg(proc.pid,signal.SIGTERM)
        for key,proc in running.values():
            try: proc.wait(timeout=20)
            except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL); proc.wait()
        status('timeout' if isinstance(e,TimeoutError) else 'failed',error=repr(e))
        raise
    finally:
        for f in handles: f.close()

if __name__=='__main__': main()
