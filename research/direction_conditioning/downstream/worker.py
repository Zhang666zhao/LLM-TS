"""Fixed ETTh1 label substitution; delegates G2 to checksum-pinned historical trainer."""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PINNED = {
    'alignment.py': 'e5fd77c9973d40e7d9ecf4d1a47421cc22fd625ec38cb69e89cc9737ec1f4d31',
    'conditioning.py': 'c6699bfcc01b92bcdad08c85734e7e5325449c08f88bc74a19a4cba9a5240f8c',
    'dataset.py': 'fa89534f8e9439b8cb145628288028b6b117ba6f796c9df384fee77db4f4f826',
    'train.py': 'a75ea99c11592a7fd70ba7fe4135b4de2a55b7e32a2629eb9a6fdfbe24acf95c',
    'check_contract.py': '32c0c84a2e09b7e749fd3d7189945e3f2d228612ddd799ab8e5bee538a2c5af3',
}

def bind(project):
    import hashlib
    source = project/'research/multivariate_oracle_six'
    for name, expected in PINNED.items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest() == expected, name
    sys.path.insert(0, str(source))
    import train
    return train

def main():
    p = argparse.ArgumentParser()
    for key in ('project', 'cache', 'oldjobs', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--group', choices=['G1','G2'], required=True)
    p.add_argument('--method', required=True)
    p.add_argument('--strategy', choices=['A','B'], required=True)
    p.add_argument('--label', choices=['l1','l2'], required=True)
    p.add_argument('--deadline', type=float, required=True)
    a = p.parse_args()
    t = bind(a.project)
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    a.dataset = 'ETTh1'
    a.baseline = a.oldjobs/(a.method+'_base_zero')
    old = a.oldjobs/f'{a.method}_{a.strategy}_{a.label}'
    original_dump = t.dump
    def annotated_dump(path, obj):
        obj = dict(obj) if isinstance(obj, dict) else obj
        if Path(path).name in ('config.json','metrics.json'):
            obj.update(oracle=False, group=a.group, conditioning_source='frozen LLM stride32',
                       train_conditions='oracle' if a.group=='G1' else 'LLM', test_conditions='LLM')
        if Path(path).name == 'initial_hash.json':
            assert obj == json.loads((old/'initial_hash.json').read_text()), 'Paired initialization mismatch'
        if Path(path).name == 'provenance.json':
            obj.update(downstream_source_sha256={f.name:t.sha(f) for f in ROOT.glob('*.py')},
                       historical_source_pins=PINNED, old_oracle=str(old))
        if Path(path).name == 'config.json':
            oldcfg = json.loads((old/'config.json').read_text())
            ignored = {'benchmark_version','oracle','group','conditioning_source','train_conditions','test_conditions'}
            assert {k:v for k,v in obj.items() if k not in ignored} == {k:v for k,v in oldcfg.items() if k not in ignored}, 'Configuration drift'
        original_dump(path,obj)
    t.dump = annotated_dump
    if a.group == 'G2':
        # No changes to optimizer, RNG, sampling, loss, early stopping or evaluation.
        # The only new model-facing data are the two predicted-label NPZ arrays.
        t.run(a)
        rows = [json.loads(x) for x in (a.output/'train.jsonl').read_text().splitlines()]
        previous = [json.loads(x) for x in (old/'train.jsonl').read_text().splitlines()]
        common = min(len(rows),len(previous))
        assert common > 0
        assert all(rows[i]['sample_order_sha256']==previous[i]['sample_order_sha256'] for i in range(common))
        original_dump(a.output/'paired_check.json',dict(initialization=True,common_epoch_order=True,
                      common_epochs=common,oracle_job=str(old),frozen_backbone_verified=a.strategy=='B'))
    else:
        a.output.mkdir(parents=True,exist_ok=False)
        original_dump(a.output/'status.json',dict(status='evaluating',pid=os.getpid()))
        torch.set_num_threads(4); random.seed(2021); np.random.seed(2021)
        torch.manual_seed(2021); torch.cuda.manual_seed_all(2021)
        torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
        values,marks,info=t.load_data(a.project,'ETTh1')
        net,cfg,_=t.backbone(a.project,a.method,'ETTh1',7)
        model=t.Forecast(net,a.method,a.strategy,7).cuda()
        checkpoint=old/'best.pth'; checkpoint_sha=t.sha(checkpoint)
        original_dump(a.output/'config.json',dict(json.loads((old/'config.json').read_text()),
                      group='G1',oracle=False,train_conditions='oracle',test_conditions='LLM'))
        original_dump(a.output/'provenance.json',dict(old_oracle=str(old),historical_source_pins=PINNED,
                      downstream_source_sha256={f.name:t.sha(f) for f in ROOT.glob('*.py')},command=sys.argv))
        model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True),strict=True)
        ds=t.Windows(values,marks,a.cache,'test',a.label)
        loader=DataLoader(ds,batch_size=cfg['batch'],shuffle=False,num_workers=cfg['workers'],
                          pin_memory=True,generator=torch.Generator().manual_seed(2021))
        metrics,arrays=t.evaluate(model,loader,a.deadline,True)
        assert np.array_equal(arrays[1],np.load(old/'targets_f32.npy'))
        assert np.array_equal(arrays[2],np.load(old/'window_start.npy'))
        for name,arr in zip(('predictions_f32','targets_f32','window_start','labels'),arrays):
            np.save(a.output/(name+'.npy'),arr)
        metrics.update(variables=info['variables'],ot_mse=metrics['variable_mse'][-1],ot_mae=metrics['variable_mae'][-1])
        annotated_dump(a.output/'metrics.json',metrics)
        original_dump(a.output/'checkpoint_reference.json',dict(path=str(checkpoint),sha256=checkpoint_sha,
                      same_as_G3=True,weights_unchanged=t.sha(checkpoint)==checkpoint_sha))
        original_dump(a.output/'status.json',dict(status='complete',finished_at=time.time()))

if __name__=='__main__':
    main()
