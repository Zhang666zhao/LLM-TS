import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stride32_server import anchor_for

class Stride32(unittest.TestCase):
    def test_all_windows_covered_causally(self):
        groups={}
        for target in range(8640,11457):
            a=anchor_for(target)
            self.assertLessEqual(a,target)
            self.assertLess(target-a,32)
            groups.setdefault(a,[]).append(target)
        self.assertEqual(len(groups),89)
        self.assertEqual([len(v) for v in groups.values()],[32]*88+[1])
        self.assertEqual(anchor_for(8671),8640)
        self.assertEqual(anchor_for(8672),8672)
        self.assertEqual(anchor_for(11456),11456)

if __name__=='__main__':unittest.main()
