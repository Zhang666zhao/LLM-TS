import unittest
import numpy as np
from noise import make_noise,LEVELS

class NoiseTests(unittest.TestCase):
    def setUp(self):
        self.truth=np.random.default_rng(42).integers(-1,2,(2817,7),dtype=np.int8)
        self.made,self.ranks,self.offsets=make_noise(self.truth)
    def test_counts_and_wrong_classes(self):
        for level in LEVELS:
            z,mask=self.made[level]
            self.assertTrue(np.array_equal(z!=self.truth,mask))
            self.assertTrue(np.isin(z,[-1,0,1]).all())
            self.assertTrue((mask.sum(0)==int(np.floor(2817*(100-level)/100+.5))).all())
    def test_nested_and_consistent_replacements(self):
        for low,high in zip(LEVELS,LEVELS[1:]):
            zl,ml=self.made[low];zh,mh=self.made[high]
            self.assertTrue((~mh|ml).all())
            self.assertTrue(np.array_equal(zl[mh],zh[mh]))
    def test_reproducible_shared_locations(self):
        other=np.zeros_like(self.truth)
        m,r,o=make_noise(other)
        self.assertTrue(np.array_equal(r,self.ranks) and np.array_equal(o,self.offsets))
        self.assertTrue(np.array_equal(m[80][1],self.made[80][1]))
    def test_100_exact(self):self.assertTrue(np.array_equal(self.made[100][0],self.truth))
    def test_reject_invalid(self):
        with self.assertRaises(AssertionError):make_noise(np.ones((8,7),dtype=np.int8)*2)

if __name__=='__main__':unittest.main()
