"""Offline CPU reproduction of the fixed ETTh1 validation and test benchmark."""
import argparse
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
os.environ['ETTH1_PROJECT_ROOT'] = str(ROOT)

import block_column_selection
import column_selection
import horizon_policy
import masked_residual_strength
import ridge_tool

SHAPE = (2817, 64, 7)
RIDGE = '20261105_ridge_individual1_mse'
# Absolute tolerances on train-standardized values and macro metrics.
PRED_TOL = 1e-5
METRIC_TOL = 1e-7

def read(path):
    return json.loads(path.read_text())

def hash_file(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

class Project:
    def __init__(self):
        self.contract = read(ROOT/'benchmark/contract.json')
        self.assets = read(ROOT/'assets/manifest.json')
        self.config_hashes = read(ROOT/'config_hashes.json')
        self.methods = read(ROOT/'methods.json')
        self.names = {m['method']:m['candidate'] for m in self.methods}
        self.verified = set()

    def asset(self, rel):
        p = ROOT/rel
        if rel not in self.verified:
            if hash_file(p) != self.assets[rel]['sha256']:
                raise ValueError(f'Local asset hash mismatch: {rel}')
            self.verified.add(rel)
        return p

    def config(self, name):
        rel = f'runs/{name}/config.json'
        p = ROOT/rel
        if hash_file(p) != self.config_hashes[rel]:
            raise ValueError(f'Frozen config hash mismatch: {name}')
        return read(p)

    def observations(self, end):
        with self.asset('assets/ETTh1.csv').open() as f:
            data = csv.DictReader(f)
            rows = [next(data) for _ in range(end)]
        raw = np.array([[float(row[v]) for v in self.contract['variables']] for row in rows])
        return ((raw - self.contract['scaler_mean']) / self.contract['scaler_scale']).astype(np.float32)

def windows(z, origins):
    for offset in range(0,len(origins),128):
        yield np.stack([z[t-512:t] for t in origins[offset:offset+128]])

def baseline(project, z, origins):
    source = next((ROOT/'external').glob('**/LTSF-Linear/models/DLinear.py'))
    spec = importlib.util.spec_from_file_location('etth1_dlinear', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    net = module.Model(SimpleNamespace(seq_len=512,pred_len=64,individual=False,enc_in=7)).cpu().eval()
    state = torch.load(project.asset('assets/weights/dlinear.pth'),map_location='cpu',weights_only=True)
    net.load_state_dict({k.removeprefix('backbone.'):v for k,v in state.items()})
    with torch.no_grad():
        return np.concatenate([net(torch.from_numpy(x)).numpy() for x in windows(z,origins)])

class LocalGraph:
    def __init__(self, project, split, z, origins):
        self.project, self.split, self.z, self.origins = project, split, z, origins
        self.memo, self.kinds = {}, {}
        self.zero = np.broadcast_to(np.float32(0),SHAPE)

    def source(self, key):
        kind, name = key.split(':',1)
        if kind != 'run':raise ValueError(f'Unknown source: {key}')
        if name in self.memo:return self.memo[name]
        asset = f'assets/{self.split}/{name}.npy'
        if name == RIDGE:
            cfg = self.project.config(name)
            w = np.load(self.project.asset('assets/weights/ridge.npy'))
            value = np.concatenate([ridge_tool.predict(x,cfg,w) for x in windows(self.z,self.origins)])
            self.kinds[name] = 'history + frozen ridge weights'
        elif asset in self.project.assets:
            value = np.load(self.project.asset(asset),mmap_mode='r')
            self.kinds[name] = 'verified numerical source'
        else:
            cfg = self.project.config(name)
            engine = cfg['engine']
            self.kinds[name] = engine
            ids = cfg['source_ids']
            if engine == 'horizon_policy_v1':
                ps = [self.source(k) if active else self.zero
                      for k,active in zip(ids,horizon_policy.active_sources(cfg))]
                value = horizon_policy.predict(np.stack(ps,axis=1),cfg)
            elif engine in ('column_selector_v1','block_column_selector_v1'):
                op = column_selection if engine=='column_selector_v1' else block_column_selection
                value = op.predict(np.stack([self.source(k) for k in ids],axis=1),cfg)
            elif engine == 'masked_residual_strength_v1':
                value = masked_residual_strength.apply(*[self.source(k) for k in ids],cfg)
            else:
                raise ValueError(f'Unsupported frozen engine for {name}: {engine}')
        if value.shape != SHAPE or not np.isfinite(value).all():
            raise ValueError(f'Invalid forecast: {name}')
        self.memo[name] = value
        return value

def errors(p,y):
    e = p.astype(np.float64)-y
    return float((e*e).mean()),float(np.abs(e).mean())

def run_split(project, split, selected, output):
    history_starts = np.load(ROOT/'benchmark'/f'{split}_starts.npy')
    origins = history_starts + project.contract['context']
    lo = 8640 if split=='val' else 11520
    np.testing.assert_array_equal(origins,np.arange(lo,lo+2817))
    z = project.observations(int(origins[-1]))
    graph = LocalGraph(project,split,z,origins)
    forecasts = {m:graph.source('run:'+project.names[m]) for m in selected}
    forecasts['DLinear'] = baseline(project,z,origins)
    folder = output/split
    folder.mkdir()
    for method,forecast in forecasts.items():
        np.save(folder/f'{method}.npy',forecast)
    # Only now read future labels and the independent reference outputs.
    z = project.observations(int(origins[-1])+64)
    target = np.stack([z[t:t+64] for t in origins])
    published = read(ROOT/'evaluation/results.json')
    refs = {x['candidate']:x for x in published['results']}
    rows=[]
    for method,p in forecasts.items():
        name = project.names.get(method,'dlinear')
        ref = np.load(project.asset(f'reference/{split}/{name}.npy'),mmap_mode='r')
        mse,mae = errors(p,target)
        if method=='DLinear':
            ref_mse,ref_mae = ((published['baseline']['mse'],published['baseline']['mae']) if split=='test' else errors(ref,target))
        elif split=='test':
            ref_mse,ref_mae = (refs[name]['test'][k] for k in ['mse','mae'])
        else:
            ref_mse,ref_mae = (refs[name][k] for k in ['validation_mse','validation_mae'])
        delta = float(np.max(np.abs(p.astype(np.float64)-ref)))
        ok = delta<=PRED_TOL and abs(mse-ref_mse)<=METRIC_TOL and abs(mae-ref_mae)<=METRIC_TOL
        rows.append(dict(method=method,split=split,windows=len(origins),mse=mse,mae=mae,
                         reference_mse=ref_mse,reference_mae=ref_mae,delta_mse=mse-ref_mse,
                         delta_mae=mae-ref_mae,max_prediction_delta=delta,
                         differing_values=int(np.count_nonzero(p!=ref)),passed=ok))
        print(f'{split:4} {method:7} MSE {mse:.9f}  MAE {mae:.9f}  max prediction delta {delta:.2g}',flush=True)
    (folder/'source_trace.json').write_text(json.dumps(graph.kinds,indent=2))
    return rows

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=('both','val','test'),default='both')
    parser.add_argument('--method',choices=['all']+[f'M{i}' for i in range(1,11)],default='all')
    args = parser.parse_args()
    torch.set_num_threads(4)
    project = Project()
    output = ROOT/'local_results'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    rows=[]
    for split in (['val','test'] if args.split=='both' else [args.split]):
        rows.extend(run_split(project,split, project.names if args.method=='all' else [args.method],output))
    with (output/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    passed = all(r['passed'] for r in rows)
    summary = dict(passed=passed,prediction_atol=PRED_TOL,metric_atol=METRIC_TOL,platform=platform.platform(),
                   python=sys.version,numpy=np.__version__,torch=torch.__version__,runtime='local CPU',
                   checked_assets=len(project.verified),metrics=rows)
    (output/'report.json').write_text(json.dumps(summary,indent=2))
    print('Report:',output/'report.json')
    if not passed:raise SystemExit('Reference comparison outside tolerance')

if __name__=='__main__':
    main()
