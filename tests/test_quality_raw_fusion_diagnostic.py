import unittest
import numpy as np
from src.quality_raw_fusion_diagnostic import transition_cube, describe, provenance_regions


class RawFusionDiagnosticTests(unittest.TestCase):
    def test_known_recall_tradeoff_and_wrong_class_conversion(self):
        raw = np.array([[0, 4, 1, 2, 3, 0]], np.uint8)
        fusion = np.array([[4, 0, 3, 2, 3, 1]], np.uint8)
        truth = np.array([[4, 4, 4, 2, 0, 0]], np.uint8)
        r = describe(transition_cube(raw, fusion, truth))
        self.assertEqual((r['corrected'], r['harmed'], r['wrong_to_wrong']), (1, 2, 1))
        self.assertEqual((r['both_correct'], r['both_wrong_same_class']), (1, 1))
        self.assertEqual(r['per_class']['foundation']['net_tp'], 0)
        self.assertEqual(r['per_class']['foundation']['gained_tp'], 1)
        self.assertEqual(r['per_class']['foundation']['lost_tp'], 1)

    def test_identical_and_empty_region(self):
        a = np.arange(5, dtype=np.uint8)[None]
        self.assertEqual(describe(transition_cube(a, a, a))['changed'], 0)
        self.assertIsNone(describe(np.zeros((5,5,5), np.int64))['delta_miou_pp'])

    def test_invalid_classes_and_provenance_rejected(self):
        with self.assertRaises(ValueError):
            transition_cube(np.array([[5]], np.uint8), np.array([[0]], np.uint8), np.array([[0]], np.uint8))
        with self.assertRaises(ValueError):
            provenance_regions(np.array([[0]], np.uint8), np.array([[4]], np.uint8))

    def test_all_directions_preserve_confusion_and_recall(self):
        raw, fusion, truth = (a.reshape(5,25).astype(np.uint8) for a in np.indices((5,5,5)))
        r = describe(transition_cube(raw, fusion, truth))
        self.assertEqual(r['raw']['confusion'], [[5]*5]*5)
        self.assertEqual(r['calibrated_fusion']['confusion'], [[5]*5]*5)
        self.assertEqual((r['corrected'], r['harmed'], r['wrong_to_wrong']), (20,20,60))


if __name__ == '__main__':
    unittest.main()
