"""Run read-only on GPU_Foundation using its existing baseline Python environment.
Stdout is a private dataset package, not a prompt. Only history goes to API.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT = Path('/nfs/dataset-ofs-661-142656/qiminhao/Foundation_model/agentic_TS/TSRAG')
sys.path.insert(0, str(PROJECT / 'research/multivariate_oracle_six'))
from dataset import load_data, labels

values, _, info = load_data(PROJECT, 'ETTh1')
expected = '3dbe45a7a25cbf0eb6d82cb84ef4487ea2b73a9341e5ff0d082e340b9d943f1d'
assert info['source_sha256'] == expected
assert info['split']['train_end'] == 8640 and info['split']['val_end'] == 11520
source = PROJECT / 'artifacts/baseline_raw_v1/ETTh1.csv'
dates = pd.to_datetime(pd.read_csv(source, usecols=['date'])['date'])
assert dates.is_monotonic_increasing and dates.is_unique
assert (dates.diff().dropna() == pd.Timedelta(hours=1)).all()
# First and last validation target window, fixed BEFORE inspecting labels.
windows, truth = [], {}
for target_start in (8640, 11456):
    start = target_start - 512
    x, y = values[start:target_start], values[target_start:target_start+64]
    assert x.shape == (512, 7) and y.shape == (64, 7)
    assert target_start >= 8640 and target_start + 64 <= 11520
    window_id = 'ETTh1-val-' + str(target_start)
    windows.append(dict(window_id=window_id, cutoff=dates.iloc[target_start-1].isoformat(),
                        timestamps=[v.isoformat() for v in dates.iloc[start:target_start]],
                        variables=info['variables'], history=x.tolist()))
    l1, l2 = labels(x, y)
    truth[window_id] = [dict(variable=n, L1=int(a), L2=int(b))
                        for n, a, b in zip(info['variables'], l1, l2)]
print(json.dumps(dict(windows=windows, truth=truth, provenance=info,
                     source_code_sha256=hashlib.sha256((PROJECT/'research/multivariate_oracle_six/dataset.py').read_bytes()).hexdigest()), allow_nan=False))
