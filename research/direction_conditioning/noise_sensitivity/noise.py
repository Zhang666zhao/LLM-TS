"""Paired, nested, symmetric ternary label corruption; no model or API access."""
import numpy as np

LEVELS=(60,70,80,90,100)
SEED=20261019

def make_noise(truth,seed=SEED):
    assert truth.ndim==2 and truth.dtype.kind=='i' and np.isin(truth,[-1,0,1]).all()
    n,c=truth.shape
    rng=np.random.default_rng(seed)
    ranks=np.stack([rng.permutation(n) for _ in range(c)],axis=1)
    offsets=rng.integers(1,3,size=(n,c),dtype=np.int8)
    wrong=((truth+1+offsets)%3-1).astype(np.int8)
    assert np.all(wrong!=truth)
    result={}
    for accuracy in LEVELS:
        count=int(np.floor(n*(100-accuracy)/100+.5))
        mask=ranks<count
        z=np.where(mask,wrong,truth).astype(np.int8)
        assert np.array_equal(z!=truth,mask) and (mask.sum(0)==count).all()
        result[accuracy]=(z,mask)
    return result,ranks,offsets
