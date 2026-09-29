import unittest
import numpy as np
from src.quality_context_background_gate import background_gate


class BackgroundGateTest(unittest.TestCase):
    def test_inclusive_boundary_and_protected_classes(self):
        raw = np.array([[2, 2, 0, 1, 3, 4]], dtype=np.uint8)
        p = np.zeros((5, 1, 6), np.float32)
        p[0] = [.9, .899, .99, .99, .99, .99]
        p[2] = 1 - p[0]
        out = background_gate(raw, p, .9)
        np.testing.assert_array_equal(out, [[0, 2, 0, 1, 3, 4]])
        np.testing.assert_array_equal(raw, [[2, 2, 0, 1, 3, 4]])

    def test_off_and_background_argmax_required(self):
        raw = np.array([[2]], dtype=np.uint8)
        p = np.array([.4, .0, .6, .0, .0], np.float32).reshape(5, 1, 1)
        np.testing.assert_array_equal(background_gate(raw, p, .3), raw)
        np.testing.assert_array_equal(background_gate(raw, p, None), raw)
