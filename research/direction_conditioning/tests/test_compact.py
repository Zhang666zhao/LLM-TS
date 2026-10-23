import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from explore_server import pack,joint_valid

class Compact(unittest.TestCase):
    def test_preserve_all_steps(self):
        w={'window_id':'s','timestamps':['unspecified']*512,'cutoff':'unspecified',
           'variables':['x'],'history':[[i/1000] for i in range(512)]}
        p=pack(w)
        self.assertEqual(len(p['variables']['x']['history']),512)
        self.assertAlmostEqual(p['variables']['x']['recent64_mean'],.4795)

    def test_joint(self):
        p={'variables':{'x':{'L1_low':0,'L1_high':1,'L2_low':2,'L2_high':3}}}
        self.assertFalse(joint_valid({'labels':[{'variable':'x','L1':-1,'L2':1}]},p))
        self.assertTrue(joint_valid({'labels':[{'variable':'x','L1':1,'L2':-1}]},p))

if __name__=='__main__':unittest.main()
