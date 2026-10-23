from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from screen import metric


class Scoring(unittest.TestCase):
    def test_perfect_three_class(self):
        self.assertEqual(metric([-1, 0, 1], [-1, 0, 1])['macro_f1'], 1)

    def test_missing_classes_are_zero(self):
        self.assertAlmostEqual(metric([0, 0], [0, 0])['macro_f1'], 1/3)

    def test_invalid_is_not_zero(self):
        result = metric([0, 1, -1], [None, None, None])
        self.assertEqual(result['invalid'], 3)
        self.assertEqual(result['macro_f1'], 0)

    def test_fixed_confusion(self):
        result = metric([-1, 0, 1], [0, 0, 1])
        self.assertAlmostEqual(result['macro_f1'], (0+2/3+1)/3)
        self.assertEqual(result['confusion']['-1']['0'], 1)


if __name__ == '__main__':
    unittest.main()
