"""Read-only integration check for the actual pinned server label source."""
import sys
from pathlib import Path
import numpy as np
from run import bind,oldjobs,METHODS

project=Path(sys.argv[1]);t=bind(project)
from dataset import labels as oracle_labels
values,marks,info=t.load_data(project,'ETTh1')
starts=np.arange(11008,13825)
computed=np.array([oracle_labels(values[s:s+512],values[s+512:s+576]) for s in starts])
assert computed.shape==(2817,2,7)
for method in METHODS:
    for strategy in ('A','B'):
        for i,label in enumerate(('l1','l2')):
            old=oldjobs(project)/f'{method}_{strategy}_{label}'
            assert np.array_equal(computed[:,i,:],np.load(old/'labels.npy'))
            assert np.array_equal(starts,np.load(old/'window_start.npy'))
print('PASS: pinned dataset.labels integration; all20 oracle labels and2817 windows match; no GPU inference')
