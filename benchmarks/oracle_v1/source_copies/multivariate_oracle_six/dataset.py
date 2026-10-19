"""Frozen Exchange windows and a shared, discrete-only oracle cache."""
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

VERSION = 'six-datasets-multivariate-oracle-v1'
DATASETS = ('ETTh1','ETTh2','ETTm1','ETTm2','weather','electricity')

def dump(path, obj):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def load_data(project, dataset):
    assert dataset in DATASETS
    sys.path.insert(0, str(project/'research/baselines_512_64'))
    from frozen_benchmark import split_spec
    source = project/'artifacts/baseline_raw_v1'/(dataset+'.csv')
    manifest = json.loads((source.parent/'manifest.json').read_text())['datasets'][dataset]
    assert sha(source) == manifest['output_sha256'], 'Frozen checksum mismatch'
    frame = pd.read_csv(source)
    variables = list(frame.columns[1:])
    assert len(variables) == {'weather':21,'electricity':321}.get(dataset,7) and 'OT' in variables
    raw = frame[variables].to_numpy(dtype=np.float64)
    assert np.isfinite(raw).all()
    splits = split_spec(dataset, len(raw), 512)
    scaler = StandardScaler().fit(raw[:splits.train_end])
    values = scaler.transform(raw).astype(np.float32)
    dt = pd.to_datetime(frame['date']).dt
    mark_list = [dt.hour.to_numpy()/23-.5,dt.dayofweek.to_numpy()/6-.5, (dt.day.to_numpy()-1)/30-.5,
                 (dt.dayofyear.to_numpy()-1)/365-.5]
    if dataset in ('ETTm1','ETTm2','weather'): mark_list.insert(0,dt.minute.to_numpy()/59-.5)
    marks = np.stack(mark_list,-1).astype(np.float32)
    info = dict(version=VERSION, dataset=dataset, source_sha256=sha(source), variables=variables,
                split=asdict(splits), scaler_mean=scaler.mean_.tolist(),
                scaler_scale=scaler.scale_.tolist(), tau=.5, ddof=0,
                epsilon_standardized=1e-3,
                epsilon_definition='0.001 times training scaler scale in raw units',
                context=512, horizon=64, stride=1, seed=2021)
    return values, marks, info

def labels(x, y):
    x, y = x.astype(np.float64), y.astype(np.float64)
    scale = np.maximum(x.std(axis=0, ddof=0), .001)
    future = y.mean(axis=0)
    rs = [(future-x[-64:].mean(0))/scale, (future-x.mean(0))/scale]
    assert all(np.isfinite(r).all() for r in rs)
    return [np.where(r>.5, 1, np.where(r<-.5, -1, 0)).astype(np.int8) for r in rs]

def prepare(project, cache, dataset):
    values, marks, info = load_data(project,dataset)
    cache.mkdir(parents=True, exist_ok=False)
    distribution = {}; files = {}
    for split in ('train','val','test'):
        begin, end = info['split'][split+'_start'], info['split'][split+'_end']
        starts = np.arange(begin, end-575, dtype=np.int64)
        assert len(starts)>0
        z = [labels(values[s:s+512], values[s+512:s+576]) for s in starts]
        l1 = np.stack([a for a,b in z]); l2 = np.stack([b for a,b in z])
        np.savez(cache/(split+'.npz'), starts=starts, l1=l1, l2=l2)
        files[split] = sha(cache/(split+'.npz'))
        distribution[split] = {name:[{str(k):int((arr[:,c]==k).sum()) for k in (-1,0,1)}
                                      for c in range(values.shape[1])] for name,arr in [('l1',l1),('l2',l2)]}
        if split == 'train':
            assert all(np.unique(arr[:,c]).size>1 for arr in (l1,l2) for c in range(values.shape[1])), 'Degenerate training labels'
    info.update(cache_sha256=files, class_counts=distribution)
    dump(cache/'manifest.json', info)
    return info

class Windows(Dataset):
    def __init__(self, values, marks, cache, split, label):
        self.values, self.marks = values, marks
        manifest = json.loads((cache/'manifest.json').read_text())
        assert sha(cache/(split+'.npz')) == manifest['cache_sha256'][split]
        with np.load(cache/(split+'.npz')) as a:
            self.starts = a['starts'].copy()
            self.z = np.zeros_like(a['l1']) if label=='zero' else a[label].copy()
        assert self.z.shape == (len(self.starts),values.shape[1]) and np.isin(self.z,[-1,0,1]).all()
    def __len__(self): return len(self.starts)
    def __getitem__(self, index):
        s = self.starts[index]
        return self.values[s:s+512], self.values[s+512:s+576], self.marks[s:s+512], self.z[index].astype(np.int64), s

if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--project',type=Path,required=True); p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--dataset',choices=DATASETS,required=True)
    a=p.parse_args(); print(json.dumps(prepare(a.project,a.cache,a.dataset),indent=2))
