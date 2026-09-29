import unittest
import numpy as np
from scripts.diagnose_quality_selector_fit_gap import cube, summary


class FitGapTests(unittest.TestCase):
    def test_transition_semantics_and_sample_partition(self):
        # A correction, a harm, a wrong-to-wrong change and an unchanged pixel.
        gt = np.array([4, 0, 2, 1], dtype=np.uint8)
        raw = np.array([0, 0, 3, 1], dtype=np.uint8)
        out = np.array([4, 4, 4, 1], dtype=np.uint8)
        full = cube(gt, raw, out)
        np.testing.assert_array_equal(full, cube(gt, raw, out, reference=True))
        sample = cube(gt[:2], raw[:2], out[:2])
        np.testing.assert_array_equal(full-sample, cube(gt[2:], raw[2:], out[2:]))
        result = summary([full])
        self.assertEqual(result['corrected_by_class'], [0, 0, 0, 0, 1])
        self.assertEqual(result['harmed_by_class'], [1, 0, 0, 0, 0])
        self.assertEqual(result['net_corrected'], 0)
        self.assertEqual(result['selector']['per_class']['foundation']['fp'], 2)

    def test_repeated_visit_is_additive_not_an_independent_view(self):
        gt = np.array([0, 1, 2, 3, 4], dtype=np.uint8)
        raw = gt.copy()
        out = np.array([2, 1, 2, 3, 4], dtype=np.uint8)
        one = cube(gt, raw, out)
        a, b = summary([one]), summary([one, one])
        self.assertEqual(a['delta_miou_pp'], b['delta_miou_pp'])
        self.assertEqual(b['harmed_by_class'][0], 2*a['harmed_by_class'][0])
        self.assertEqual(sum(map(sum, b['raw']['confusion'])), 10)
