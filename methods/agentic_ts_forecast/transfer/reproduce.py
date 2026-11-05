"""Offline replay of every frozen ETTm1 method from raw rolling histories."""
import argparse
import csv
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from forecast import COLS, D, H, L, batches, digest, load_forecaster, read, write
from run_pipeline import dataset

HERE = Path(__file__).resolve().parent
METRIC_ATOL = 1e-7

def replay_split(split, forecasters, reference, scratch):
    contract = read(HERE / 'contract.json')
    start = contract['train_end'] if split == 'val' else contract['validation_end']
    end = contract['validation_end'] if split == 'val' else contract['test_end']
    origins = np.arange(start, end - H + 1)
    path = HERE / 'assets/ETTm1.csv'
    raw, times = dataset(path, int(origins[-1]))
    names = [name for model in forecasters for name in model.plan['outputs']]
    if set(names) != set(reference) or len(set(names)) != len(names):
        raise ValueError('Frozen outputs and reference table disagree')
    n = len(origins)
    preds = {name: np.lib.format.open_memmap(scratch / f'{name}.npy', mode='w+',
             dtype='float32', shape=(n,H,D)) for name in names}
    for group in batches(origins):
        history = group[:,None] + np.arange(-L,0)
        stamps = times[history].astype('datetime64[m]')
        for model in forecasters:
            for name, value in model.predict_batch(raw[history], stamps, standardized=True).items():
                if value.shape != (len(group),H,D) or not np.isfinite(value).all():
                    raise ValueError(f'Invalid predictions from {name}')
                preds[name][group-start] = value
    # Score only after all methods have generated the split.
    raw, _ = dataset(path, end)
    plan = forecasters[0].plan
    z = ((raw-plan['mean'])/plan['std']).astype('float32')
    rows = []
    for name, pred in preds.items():
        squared = absolute = 0.0
        for group in batches(origins):
            diff = pred[group-start].astype('float64') - z[group[:,None] + np.arange(H)]
            squared += np.square(diff).sum()
            absolute += np.abs(diff).sum()
        mse, mae = squared/(n*H*D), absolute/(n*H*D)
        ref = reference[name]
        dm = mse-float(ref[f'{split}_mse'])
        da = mae-float(ref[f'{split}_mae'])
        rows.append(dict(method=name,split=split,windows=n,mse=float(mse),mae=float(mae),
                         delta_mse=float(dm),delta_mae=float(da),passed=bool(max(abs(dm),abs(da))<=METRIC_ATOL)))
    return rows

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=['both','val','test'],default='both')
    args=parser.parse_args()
    torch.set_num_threads(4)
    artifacts=HERE/'artifacts/ETTm1'
    contract=read(HERE/'contract.json')
    if digest(HERE/'assets/ETTm1.csv')!=contract['sha256']:
        raise ValueError('ETTm1 dataset digest mismatch')
    models=[load_forecaster(artifacts/'frozen_plan.json'),
            load_forecaster(artifacts/'matched_control/frozen_plan.json')]
    with (artifacts/'results.csv').open() as f:
        reference={r['method']:r for r in csv.DictReader(f)}
    out=HERE.parent/'outputs'/'ettm1'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    rows=[]
    for split in (['val','test'] if args.split=='both' else [args.split]):
        with TemporaryDirectory(dir=out) as directory:
            part=replay_split(split,models,reference,Path(directory))
        rows.extend(part)
        print(f"{split}: {sum(r['passed'] for r in part)}/{len(part)} passed",flush=True)
    write(out/'report.json',dict(metric_atol=METRIC_ATOL,metrics=rows,
         selected=models[0].plan['selected'],torch=torch.__version__,numpy=np.__version__))
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys())
        writer.writeheader();writer.writerows(rows)
    print(out/'report.json')
    if not all(r['passed'] for r in rows):raise SystemExit('Reference mismatch')

if __name__=='__main__':main()
