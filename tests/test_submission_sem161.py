import unittest
import numpy as np
from src.submission_sem161 import fuse


class FrozenFusionTests(unittest.TestCase):
    def test_small_grid_and_protection(self):
        base = np.array([[0, 1, 2, 3, 4, 4]], dtype=np.uint8)
        five = np.zeros((5, 1, 6), dtype=np.float32)
        five[0] = 1
        five[0, 0, 5] = 0.8
        five[4, 0, 5] = 0.2
        cable = np.array([[0, 1, 1, 1, 1, 1]], dtype=np.float32)
        deck = np.ones((1, 6), dtype=np.float32)
        stages = fuse(base, five, cable, deck)
        np.testing.assert_array_equal(stages["SEM-161"], [[1, 1, 2, 3, 0, 4]])
        np.testing.assert_array_equal(stages["SEM-001"], base)
        np.testing.assert_array_equal(stages["SEM-151"], [[0, 1, 2, 3, 0, 4]])


if __name__ == "__main__":
    unittest.main()
