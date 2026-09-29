import unittest
import numpy as np
try:
    from scripts.diagnose_quality_expert_strata import boundary_bins, crop_coverage, stratified_counts
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, 'Optional diagnostic dependencies unavailable')
class ExpertStrataTests(unittest.TestCase):
    def test_rectangles_and_flip_do_not_change_exposure(self):
        samples = [dict(top=0, left=0, flip=True), dict(top=2, left=3, flip=False)]
        actual = crop_coverage(samples, (5, 7), 3)
        expected = np.zeros((5, 7), int)
        expected[:3, :3] += 1
        expected[2:5, 3:6] += 1
        np.testing.assert_array_equal(actual, expected)

    def test_no_boundary_and_two_sided_transition(self):
        gt = np.zeros((20, 80), np.uint8)
        self.assertTrue(np.all(boundary_bins(gt) == 3))
        gt[:, 40:] = 2
        bins = boundary_bins(gt)
        self.assertTrue(np.all(bins[:, 37:43] == 0))
        self.assertEqual(int(bins[10, 0]), 3)

    def test_common_and_confident_errors_keep_gt_prediction_axes(self):
        gt = np.array([[0, 2, 2, 1]], np.uint8)
        pred = np.array([[2, 0, 2, 1]], np.uint8)
        bins = np.array([[0, 1, 2, 3]])
        common = np.array([[False, True, False, False]])
        high = np.array([[True, False, True, True]])
        out = stratified_counts(gt, pred, bins, common, high)
        self.assertEqual(out[..., 0].sum(), 4)
        self.assertEqual(out[..., 1].sum(), 2)
        self.assertEqual(out[2, 0, 1, 2], 1)
        self.assertEqual(out[0, 2, 0, 3], 1)
