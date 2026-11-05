"""Exact weighted-sum prior used by hierarchical memory inputs, without recentering."""
import numpy as np
def predict(p,weights):return (p*np.asarray(weights)[None,:,None,None]).sum(1).astype(np.float32)
