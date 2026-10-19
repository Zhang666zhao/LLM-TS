"""One fixed multivariate experiment. GPU only, no automatic retries."""
import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import load_data, Windows, dump, sha, VERSION
from conditioning import backbone, Forecast, objective, state_hash, METHODS
from alignment import historical_run, parameter_hash

STOP=False
def stop(*_):
    global STOP
    STOP=True
signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
def check(deadline):
    if STOP or time.time()>=deadline: raise TimeoutError('Stopped/shared deadline reached')

def evaluate(model,loader,deadline,save=False):
    model.eval(); sq=np.zeros(8); ab=np.zeros(8); count=0; ps=[]; ys=[]; ids=[]; zs=[]
    with torch.no_grad():
        for x,y,marks,z,idx in loader:
            check(deadline)
            pred=model(x.cuda(),marks.cuda(),z.cuda())
            err=pred.double()-y.cuda().double()
            assert torch.isfinite(err).all() and pred.shape==y.shape
            sq+=err.square().sum((0,1)).cpu().numpy(); ab+=err.abs().sum((0,1)).cpu().numpy(); count+=len(x)*64
            if save:
                ps.append(pred.cpu().numpy()); ys.append(y.numpy()); ids.append(idx.numpy()); zs.append(z.numpy())
    metrics=dict(mse=float(sq.mean()/count),mae=float(ab.mean()/count),
                 variable_mse=(sq/count).tolist(),variable_mae=(ab/count).tolist(),count_per_variable=count)
    return metrics, [np.concatenate(a) for a in (ps,ys,ids,zs)] if save else None

def run(a):
    a.output.mkdir(parents=True,exist_ok=False)
    dump(a.output/'status.json',dict(status='initializing',pid=os.getpid()))
    try: execute(a)
    except BaseException as e:
        dump(a.output/'status.json',dict(status='failed',error=repr(e))); raise

def execute(a):
    assert a.strategy!='base' or a.label=='zero'
    check(a.deadline)
    torch.set_num_threads(4); random.seed(2021); np.random.seed(2021); torch.manual_seed(2021); torch.cuda.manual_seed_all(2021)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    values,marks,info=load_data(a.project)
    cache_info=json.loads((a.cache/'manifest.json').read_text())
    for k,v in info.items(): assert cache_info[k]==v, 'Cache data identity mismatch '+k
    net,cfg,audit=backbone(a.project,a.method)
    init=parameter_hash(net)
    old=historical_run(a.project,a.method,'exchange_rate')
    if (old/'initial_hash.json').exists(): assert init==json.loads((old/'initial_hash.json').read_text())['all'], 'Historical initialization mismatch'
    audit.update(initial_parameter_sha256=init, historical_initial_verified=(old/'initial_hash.json').exists())
    if a.strategy=='B':
        assert a.baseline and json.loads((a.baseline/'status.json').read_text())['status']=='complete'
        state=torch.load(a.baseline/'best.pth',map_location='cpu',weights_only=True)
        net.load_state_dict({k[len('backbone.'):]:v for k,v in state.items() if k.startswith('backbone.')},strict=True)
        audit['base_checkpoint_sha256']=sha(a.baseline/'best.pth')
    model=Forecast(net,a.method,a.strategy).cuda()
    assert parameter_hash(net)==init or a.strategy=='B'
    frozen=state_hash(net) if a.strategy=='B' else None
    cfg.update(strategy=a.strategy,label=a.label,benchmark_version=VERSION,
               oracle=a.label!='zero',checkpoint_metric='all-variable-validation-MSE',
               module_seed=2022,embedding_dim=8,hidden_width=32,zero_initialized_output=True,
               parameters=sum(p.numel() for p in model.parameters()),
               trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    dump(a.output/'config.json',cfg); dump(a.output/'config_alignment.json',audit)
    dump(a.output/'data_version.json',cache_info)
    dump(a.output/'initial_hash.json',dict(backbone=init,condition=parameter_hash(model.condition) if a.strategy!='base' else None))
    root=Path(__file__).resolve().parent
    dump(a.output/'provenance.json',dict(command=sys.argv,commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
         source_sha256={p.name:sha(p) for p in root.glob('*.py')},torch=torch.__version__,python=sys.version))
    (a.output/'env_snapshot.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    ds={s:Windows(values,marks,a.cache,s,a.label) for s in ('train','val','test')}
    loaders={s:DataLoader(d,batch_size=cfg['batch'],shuffle=s=='train',drop_last=s=='train',num_workers=cfg['workers'],
                         pin_memory=True,generator=torch.Generator().manual_seed(2021)) for s,d in ds.items()}
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.Adam(params,lr=cfg['lr'],weight_decay=cfg.get('weight_decay',0.))
    best=float('inf'); stale=0; step=0; start=time.time()
    for epoch in range(1,cfg['epochs']+1):
        model.train(); total=seen=0; order=[]; epoch_start=time.time()
        for x,y,t,z,idx in loaders['train']:
            check(a.deadline); order.extend(idx.tolist()); opt.zero_grad(set_to_none=True)
            p=model(x.cuda(),t.cuda(),z.cuda()); loss=objective(p,y.cuda(),a.method,cfg)
            assert torch.isfinite(loss), 'Nonfinite loss'
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(params,cfg.get('clip') or float('inf'))
            assert torch.isfinite(norm), 'Nonfinite gradient'
            if a.strategy=='B': assert all(q.grad is None for q in net.parameters())
            if a.method=='frets' and a.strategy!='B' and step==0: assert net.r1.grad is not None
            log=step%100==0
            if log:
                probe=next(q for q in params if q.grad is not None); before=probe.detach().clone()
                zero=sum(int((q.grad==0).sum()) for q in params if q.grad is not None)
                n=sum(q.numel() for q in params if q.grad is not None)
            opt.step(); step+=1; total+=loss.item()*len(x); seen+=len(x)
            if log:
                row=dict(epoch=epoch,step=step,loss=loss.item(),grad_norm=float(norm),grad_zero_fraction=zero/n,
                         parameter_update_ratio=float((probe-before).norm()/(before.norm()+1e-12)),
                         weight_mean=float(probe.mean()),weight_std=float(probe.std(unbiased=False)),
                         lr=opt.param_groups[0]['lr'],peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                         elapsed_seconds=time.time()-start)
                with (a.output/'diagnostics.jsonl').open('a') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
        val,_=evaluate(model,loaders['val'],a.deadline)
        row=dict(epoch=epoch,train_loss=total/seen,validation=val,lr=opt.param_groups[0]['lr'],
                 sample_order_sha256=__import__('hashlib').sha256(np.array(order,dtype=np.int64).tobytes()).hexdigest(),
                 epoch_seconds=time.time()-epoch_start)
        with (a.output/'train.jsonl').open('a') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row),flush=True); dump(a.output/'status.json',dict(status='training',**row))
        if val['mse']<best:
            best=val['mse']; stale=0; torch.save(model.state_dict(),a.output/'best.pth'); dump(a.output/'best_validation.json',row)
        else: stale+=1
        if stale>=cfg['patience']: break
        factor=.5**(epoch-1) if cfg['scheduler']=='type1' else (1. if epoch<3 else .9**(epoch-3))
        for group in opt.param_groups: group['lr']=cfg['lr']*factor
    model.load_state_dict(torch.load(a.output/'best.pth',map_location='cuda',weights_only=True))
    if frozen: assert state_hash(net)==frozen, 'Frozen backbone/buffers changed'
    metrics,arrays=evaluate(model,loaders['test'],a.deadline,True)
    pred,true,idx,z=arrays
    ref=np.load(old/'targets_f32.npy',mmap_mode='r')
    assert np.array_equal(ref,true), 'Historical target mismatch'
    assert np.array_equal(idx,ds['test'].starts)
    for name,arr in zip(('predictions_f32','targets_f32','window_start','labels'),arrays): np.save(a.output/(name+'.npy'),arr)
    metrics.update(method=a.method,strategy=a.strategy,label=a.label,oracle=a.label!='zero',variables=info['variables'],
                   ot_mse=metrics['variable_mse'][info['variables'].index('OT')],ot_mae=metrics['variable_mae'][info['variables'].index('OT')],
                   exact_targets_verified=True,seconds=time.time()-start)
    per_class=[]
    for c,v in enumerate(info['variables']):
        for k in (-1,0,1):
            mask=z[:,c]==k; e=pred[mask,:,c].astype(np.float64)-true[mask,:,c]
            per_class.append(dict(variable=v,label=k,windows=int(mask.sum()),mse=float((e*e).mean()) if e.size else None,
                                  mae=float(np.abs(e).mean()) if e.size else None))
    dump(a.output/'class_metrics.json',per_class); dump(a.output/'metrics.json',metrics)
    dump(a.output/'status.json',dict(status='complete',best_validation_mse=best,seconds=time.time()-start))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--project',type=Path,required=True); p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--strategy',choices=['base','A','B'],required=True); p.add_argument('--label',choices=['zero','l1','l2'],required=True)
    p.add_argument('--baseline',type=Path); p.add_argument('--deadline',type=float,required=True)
    run(p.parse_args())
