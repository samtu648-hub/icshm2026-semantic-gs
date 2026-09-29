import unittest
try:
    import numpy as np
    from src.quality_complementarity import BRANCHES, positive_proposal, diagnose, independent_diagnose
except ImportError:
    np = None


@unittest.skipIf(np is None, 'Optional NumPy runtime unavailable')
class ComplementarityTests(unittest.TestCase):
    def test_binary_negative_is_abstention(self):
        np.testing.assert_array_equal(positive_proposal(np.array([[0, 1]], np.uint8), 2), [[5, 2]])
        with self.assertRaises(ValueError):
            positive_proposal(np.array([[2]], np.uint8), 2)

    def test_oracle_cannot_invent_truth_and_protects_masks(self):
        raw = np.array([[2, 0, 3, 4, 0]], np.uint8)
        gt = np.array([[3, 4, 4, 4, 1]], np.uint8)
        maps = {k: raw.copy() for k in BRANCHES}
        maps['static'] = np.array([[3, 4, 4, 0, 2]], np.uint8)
        five = np.zeros((5, 1, 5), np.float32)
        five[3, 0, 2], five[4, 0, 2] = .51, .49
        out = diagnose(raw, raw, maps, five, gt)
        self.assertEqual(out, independent_diagnose(raw, raw, maps, five, gt))
        repair = np.array(out['scopes']['all']['recovered_confusion'])
        self.assertEqual(int(repair.sum()), 3)
        self.assertEqual(int(repair[1].sum()), 0)
        self.assertEqual(np.array(out['scopes']['lock_deck_stay_cable']['recovered_confusion']).sum(), 2)
        self.assertEqual(np.array(out['scopes']['background_foundation']['recovered_confusion']).sum(), 1)
        self.assertEqual(out['scopes']['all']['correct_control_pixels_exposed'][4], 1)

    def test_selected_baseline_no_cascade(self):
        raw = np.array([[2, 3]], np.uint8)
        selected = np.array([[2, 4]], np.uint8)
        gt = np.array([[4, 3]], np.uint8)
        maps = {k: np.array([[3, 4]], np.uint8) for k in BRANCHES}
        five = np.zeros((5, 1, 2), np.float32)
        five[3] = .51
        five[4] = .49
        out = diagnose(selected, raw, maps, five, gt)
        self.assertEqual(out, independent_diagnose(selected, raw, maps, five, gt))
        self.assertEqual(np.array(out['scopes']['approved_foundation05']['recovered_confusion']).sum(), 0)
        self.assertEqual(np.array(out['scopes']['tower_proposal']['recovered_confusion']).sum(), 0)


if __name__ == '__main__':
    unittest.main()
