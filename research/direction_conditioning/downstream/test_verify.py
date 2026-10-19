"""Small numerical verification tests; no training, GPU, or API."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from verify import measure

class SavedMetricsTests(unittest.TestCase):
    def check_case(self, bad_metrics=False, nonfinite=False):
        # Broadcast views avoid allocating real benchmark tensors for this unit test.
        y=np.broadcast_to(np.arange(7,dtype=np.float32),(2817,64,7))
        p=y+np.arange(7,dtype=np.float32)/8
        e=p.astype(np.float64)-y
        mse=(e*e).mean((0,1)); mae=np.abs(e).mean((0,1))
        obj=dict(mse=float(mse.mean()),mae=float(mae.mean()),variable_mse=mse.tolist(),variable_mae=mae.tolist())
        if bad_metrics: obj['mse']+=1
        if nonfinite: p[0,0,0]=float('nan')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp); (path/'metrics.json').write_text(json.dumps(obj))
            with patch('verify.np.load',side_effect=[p,y]): return measure(path)
    def test_exact(self): self.assertEqual(self.check_case()['ot_mse'],.5625)
    def test_wrong_aggregate(self):
        with self.assertRaises(AssertionError): self.check_case(bad_metrics=True)
    def test_nonfinite(self):
        with self.assertRaises(AssertionError): self.check_case(nonfinite=True)

if __name__=='__main__': unittest.main()
