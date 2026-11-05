"""Synthetic, offline operator smoke test. No physical series or benchmark scores."""
import json,sys
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'src'))
from block_column_selection import predict as block
from column_selection import predict as column
from horizon_policy import predict as blend

ops={'column_selector_v1':column,'block_column_selector_v1':block,'horizon_policy_v1':blend}
rng=np.random.default_rng(2021)
for m in json.loads((root/'methods.json').read_text()):
    cfg=json.loads((root/m['config']).read_text())
    x=rng.standard_normal((1,len(cfg['source_ids']),64,7)).astype(np.float32)
    y=ops[cfg['engine']](x,cfg)
    assert y.shape==(1,64,7) and np.isfinite(y).all()
    print(f"{m['method']}: {cfg['engine']} -> synthetic output {y.shape}")
