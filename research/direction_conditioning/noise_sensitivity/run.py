"""Fixed ETTh1 inference-only noise sweep: 80 evaluations, 20 reused oracle results."""
import argparse
import hashlib
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from noise import make_noise,LEVELS,SEED

ROOT=Path(__file__).resolve().parent
METHODS=('dlinear','patchtst','itransformer','frets','fredf')
STOP=False
def stop(*_):
    global STOP
    STOP=True
signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj):
    p=Path(p); tmp=p.with_suffix('.tmp'); tmp.write_text(json.dumps(obj,indent=2,allow_nan=False,ensure_ascii=False)); tmp.replace(p)
def oldjobs(project): return project/'runs/20261015_five_multivariate_oracle_continue_v1/ETTh1/jobs'
def bind(project):
    dep=project/'research/direction_downstream_v1'
    previous=read(project/'runs/20261019_etth1_llm_downstream_v1/source_snapshot.json')
    assert sha(dep/'worker.py')==previous['downstream']['worker.py']
    sys.path.insert(0,str(dep))
    from worker import bind as original_bind
    t=original_bind(project)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    return t

def prepare(a):
    import numpy as np
    t=bind(a.project); values,marks,info=t.load_data(a.project,'ETTh1')
    starts=np.arange(11008,13825,dtype=np.int64)
    computed=np.array([t.labels(values[s:s+512],values[s+512:s+576]) for s in starts])
    truth={l:computed[:,i,:] for i,l in enumerate(('l1','l2'))}
    refs={}
    for m in METHODS:
        for s in ('A','B'):
            for l in ('l1','l2'):
                key=f'{m}_{s}_{l}'; p=oldjobs(a.project)/key
                assert read(p/'status.json')['status']=='complete'
                assert np.array_equal(np.load(p/'labels.npy'),truth[l])
                assert np.array_equal(np.load(p/'window_start.npy'),starts)
                refs[key]=dict(path=str(p),checkpoint_sha256=sha(p/'best.pth'),metrics_sha256=sha(p/'metrics.json'))
    cache=a.run/'cache'; cache.mkdir()
    arrays={}; noise={}; audit={}
    for label in ('l1','l2'):
        noise[label],ranks,offsets=make_noise(truth[label])
        arrays[label+'_truth']=truth[label]
    # Same seed yields identical locations and offsets for L1 and L2, A/B, all models.
    arrays.update(starts=starts,target_starts=starts+512,ranks=ranks,offsets=offsets)
    np.savez_compressed(cache/'corruption_audit.npz',**arrays)
    for level in LEVELS:
        p=cache/str(level); p.mkdir()
        np.savez(p/'test.npz',starts=starts,**{l:noise[l][level][0] for l in ('l1','l2')})
        dump(p/'manifest.json',dict(info,cache_sha256={'test':sha(p/'test.npz')},label_source='synthetic noisy oracle'))
        audit[str(level)]={l:dict(actual_accuracy=float((noise[l][level][0]==truth[l]).mean()),
            changed_per_variable=noise[l][level][1].sum(0).tolist(),changed_total=int(noise[l][level][1].sum())) for l in truth}
    dump(a.run/'noise_manifest.json',dict(seed=SEED,levels=LEVELS,rows=2817,variables=info['variables'],
         sampling='per-variable exact rounded counts, nested masks, uniform alternative class',
         shared_masks='all methods, A/B, L1/L2',unit='window x variable; no stride32 reuse',stats=audit,
         cache_sha256={str(k):sha(cache/str(k)/'test.npz') for k in LEVELS},audit_sha256=sha(cache/'corruption_audit.npz')))
    dump(a.run/'references.json',refs);dump(a.run/'data_version.json',info)
    snapshot=a.run/'code_snapshot';snapshot.mkdir()
    for p in ROOT.glob('*'):
        if p.is_file():shutil.copy2(p,snapshot/p.name)
    dump(a.run/'provenance.json',dict(code_revision=os.environ['NOISE_CODE_REVISION'],
         source_sha256={p.name:sha(p) for p in ROOT.glob('*.py')},original_source=read(a.project/'runs/20261019_etth1_llm_downstream_v1/source_snapshot.json'),
         model_protocol='CPU2021/CUDA2022, frozen old oracle weights',new_training_jobs=0,new_API_calls=0))
    (a.run/'env_snapshot.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))

def worker(a):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    t=bind(a.project); torch.set_num_threads(4)
    random.seed(2021);np.random.seed(2021);torch.manual_seed(2021);torch.cuda.manual_seed_all(2021)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    values,marks,info=t.load_data(a.project,'ETTh1')
    for label in ('l1','l2'):
        previous=oldjobs(a.project)/f'{a.method}_{a.strategy}_{label}'
        net,cfg,_=t.backbone(a.project,a.method,'ETTh1',7)
        model=t.Forecast(net,a.method,a.strategy,7).cuda()
        model.load_state_dict(torch.load(previous/'best.pth',map_location='cuda',weights_only=True))
        model.eval();before=t.state_hash(model)
        for level in LEVELS[:-1]:
            if STOP or time.time()>=a.deadline: raise TimeoutError('Shared deadline/stop')
            out=a.run/'jobs'/f'{a.method}_{a.strategy}_{label}_acc{level}'
            out.mkdir(exist_ok=False);dump(out/'status.json',dict(status='evaluating',pid=os.getpid()))
            ds=t.Windows(values,marks,a.run/'cache'/str(level),'test',label)
            loader=DataLoader(ds,batch_size=cfg['batch'],shuffle=False,num_workers=cfg['workers'],pin_memory=True,
                              generator=torch.Generator().manual_seed(2021))
            metrics,arrays=t.evaluate(model,loader,a.deadline,True)
            assert np.array_equal(arrays[1],np.load(previous/'targets_f32.npy'))
            assert np.array_equal(arrays[2],np.load(previous/'window_start.npy'))
            for name,arr in zip(('predictions_f32','targets_f32','window_start','labels'),arrays):np.save(out/(name+'.npy'),arr)
            metrics.update(ot_mse=metrics['variable_mse'][-1],ot_mae=metrics['variable_mae'][-1],
                 method=a.method,strategy=a.strategy,label=label,nominal_accuracy=level,oracle_checkpoint=str(previous/'best.pth'),
                 checkpoint_sha256=sha(previous/'best.pth'),noise_seed=SEED,variables=info['variables'])
            dump(out/'metrics.json',metrics);dump(out/'config.json',dict(cfg,test_label_accuracy=level,training=False))
            assert t.state_hash(model)==before,'Weights/buffers changed during evaluation'
            dump(out/'status.json',dict(status='complete',model_state_unchanged=True))
            print(json.dumps(dict(completed=out.name,mse=metrics['mse'])),flush=True)
        del model,net;torch.cuda.empty_cache()

def verify(a):
    import numpy as np
    manifest=read(a.run/'noise_manifest.json'); refs=read(a.run/'references.json')
    z=np.load(a.run/'cache/corruption_audit.npz');starts=z['starts']
    for label in ('l1','l2'):
        made,ranks,offsets=make_noise(z[label+'_truth'])
        assert np.array_equal(ranks,z['ranks']) and np.array_equal(offsets,z['offsets'])
        for level in LEVELS:
            p=a.run/'cache'/str(level)/'test.npz';assert sha(p)==manifest['cache_sha256'][str(level)]
            assert np.array_equal(np.load(p)[label],made[level][0])
    reference=np.load(Path(next(iter(refs.values()))['path'])/'targets_f32.npy')
    rows=[]
    for method in METHODS:
        for strategy in ('A','B'):
            for label in ('l1','l2'):
                ref=refs[f'{method}_{strategy}_{label}']; old=Path(ref['path'])
                assert sha(old/'best.pth')==ref['checkpoint_sha256'] and sha(old/'metrics.json')==ref['metrics_sha256']
                for level in LEVELS:
                    if STOP or time.time()>=read(a.run/'budget.json')['deadline']:raise TimeoutError('Verification deadline/stop')
                    p=old if level==100 else a.run/'jobs'/f'{method}_{strategy}_{label}_acc{level}'
                    assert read(p/'status.json')['status']=='complete'
                    pred=np.load(p/'predictions_f32.npy');truth=np.load(p/'targets_f32.npy')
                    assert pred.shape==truth.shape==(2817,64,7) and np.isfinite(pred).all()
                    assert np.array_equal(truth,reference) and np.array_equal(np.load(p/'window_start.npy'),starts)
                    assert np.array_equal(np.load(p/'labels.npy'),np.load(a.run/'cache'/str(level)/'test.npz')[label])
                    e=pred.astype(np.float64)-truth;mse=np.mean(e*e,(0,1));mae=np.mean(abs(e),(0,1))
                    m=read(p/'metrics.json')
                    np.testing.assert_allclose(mse,m['variable_mse'],rtol=1e-10,atol=1e-12)
                    np.testing.assert_allclose(mae,m['variable_mae'],rtol=1e-10,atol=1e-12)
                    if level!=100:assert m['checkpoint_sha256']==ref['checkpoint_sha256'] and read(p/'status.json')['model_state_unchanged']
                    rows.append(dict(method=method,strategy=strategy,label=label,accuracy=level,
                        mse=float(mse.mean()),mae=float(mae.mean()),ot_mse=float(mse[-1]),ot_mae=float(mae[-1]),
                        variable_mse=mse.tolist(),variable_mae=mae.tolist(),source=str(p),reused=level==100))
    assert len(rows)==100 and sum(not r['reused'] for r in rows)==80
    means=[]
    for s in ('A','B'):
        for l in ('l1','l2'):
            for level in LEVELS:
                rs=[r for r in rows if (r['strategy'],r['label'],r['accuracy'])==(s,l,level)]
                assert len(rs)==5
                means.append(dict(strategy=s,label=l,accuracy=level,**{m:float(np.mean([r[m] for r in rs])) for m in ('mse','mae','ot_mse','ot_mae')}))
    dump(a.run/'comparison.json',dict(rows=rows,means=means,noise_manifest=manifest))
    dump(a.run/'verification.json',dict(passed=True,recomputed_results=100,new_evaluations=80,reused_oracle=20,
         exact_targets=True,exact_windows=True,labels_verified=True,weights_unchanged=True,nested_masks_verified=True))
    lines=['# ETTh1 Oracle标签噪声敏感性','','五种模型等权平均全变量MSE，越低越好。正确率而非错误率；100%复用旧Oracle。',
           '只替换测试标签，不训练；逐窗口×变量随机错误，不采用LLM stride32。单噪声种子20261019。','',
           '|方案|60%|70%|80%|90%|100% Oracle|','|---|---:|---:|---:|---:|---:|']
    for s in ('A','B'):
        for l in ('l1','l2'):
            cells=[f'{s}-{l.upper()}']+[f"{next(r['mse'] for r in means if (r['strategy'],r['label'],r['accuracy'])==(s,l,k)):.6f}" for k in LEVELS]
            lines.append('| '+' | '.join(cells)+' |')
    lines+=['','不是可部署结果：大部分标签仍来自真实未来。随机对称噪声不代表LLM系统性错误；不能仅按正确率解释LLM差距。',
            '每变量2817标签，改错数量四舍五入，实际比例见noise_manifest.json；误标在另外两类均匀抽取且不会碰巧保持正确。',
            '全部逐模型、逐变量MSE/MAE及OT见comparison.json。只执行固定80次测试，不根据结果调整噪声或选择模型。']
    (a.run/'summary.md').write_text('\n'.join(lines)+'\n')

def launch(a):
    a.run.mkdir(parents=True,exist_ok=False);start=time.time();deadline=start+7200
    dump(a.run/'budget.json',dict(start=start,deadline=deadline,max_seconds=7200,max_evaluations=80,new_training=0,gpus=[0,1]))
    dump(a.run/'pid.json',dict(pid=os.getpid()));running={};completed=[];pending=[];handles=[]
    def status(state,**kw):dump(a.run/'status.json',dict(status=state,completed_batches=completed,pending_batches=pending,
        running={str(g):dict(batch=k,pid=p.pid) for g,(k,p) in running.items()},total_batches=10,evaluations_per_batch=8,updated_at=time.time(),**kw))
    try:
        status('preparing');prepare(a);(a.run/'jobs').mkdir();(a.run/'logs').mkdir()
        pending=[f'{m}__{s}' for m in METHODS for s in ('A','B')]
        dump(a.run/'task_manifest.json',dict(batches=pending,levels=list(LEVELS[:-1]),labels=['l1','l2'],new_evaluations=80,reused_oracle=20))
        while pending or running:
            if STOP or time.time()>=deadline:raise TimeoutError('Shared deadline/stop')
            for gpu,(key,p) in list(running.items()):
                code=p.poll()
                if code is not None:
                    del running[gpu]
                    if code:raise RuntimeError(f'{key} failed exit={code}')
                    completed.append(key)
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
            free={int(x.split(',')[0]):int(x.split(',')[1]) for x in raw.splitlines()}
            for gpu in (0,1):
                if pending and gpu not in running and free.get(gpu,0)>=13312:
                    key=pending.pop(0);method,strategy=key.split('__');f=(a.run/'logs'/(key+'.log')).open('w');handles.append(f)
                    cmd=[sys.executable,str(ROOT/'run.py'),'worker','--project',str(a.project),'--run',str(a.run),
                         '--method',method,'--strategy',strategy,'--deadline',str(deadline)]
                    p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)))
                    running[gpu]=(key,p)
            status('running');time.sleep(3)
        status('verifying');verify(a);status('complete')
        dump(a.run/'completion.json',dict(complete=True,finished_at=time.time(),seconds=time.time()-start))
    except BaseException as e:
        for key,p in running.values():
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for key,p in running.values():
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        status('timeout' if isinstance(e,TimeoutError) else 'failed',error=repr(e));raise
    finally:
        for f in handles:f.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['launch','worker','verify'])
    p.add_argument('--project',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--method',choices=METHODS);p.add_argument('--strategy',choices=['A','B']);p.add_argument('--deadline',type=float)
    a=p.parse_args();globals()[a.mode](a)
