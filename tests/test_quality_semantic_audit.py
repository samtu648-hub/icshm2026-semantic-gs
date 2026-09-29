import unittest
import numpy as np
from src.quality_semantic_audit import check_replay, independent_confusion, metrics


class QualitySemanticAuditTests(unittest.TestCase):
    def test_confusion_two_independent_methods_and_class_metrics(self):
        gt = np.array([[0, 1, 1, 2, 3, 4]], np.uint8)
        pred = np.array([[0, 1, 0, 2, 4, 4]], np.uint8)
        matrix = independent_confusion(gt, pred)
        result = metrics(matrix)
        self.assertEqual(matrix.sum(), 6)
        self.assertEqual(result["per_class"]["deck"]["IoU"], .5)
        self.assertEqual(result["per_class"]["foundation"]["precision"], .5)
        self.assertEqual(result["mIoU_with_background"], .5)

    def test_absent_class_is_none_not_perfect_or_nan(self):
        result = metrics(np.zeros((5, 5), np.int64))
        self.assertIsNone(result["mIoU_without_background"])
        self.assertIsNone(result["per_class"]["deck"]["IoU"])

    def test_exact_probability_and_png_replay_required(self):
        p = np.full((5, 2, 3), .2, np.float32)
        png = np.zeros((2, 3), np.uint8)
        check_replay(p, p.copy(), png)
        bad = p.copy()
        bad[0, 0, 0] = np.nextafter(bad[0, 0, 0], np.float32(1))
        with self.assertRaises(ValueError):
            check_replay(p, bad, png)
        with self.assertRaises(ValueError):
            check_replay(p, p, png + 1)

    def test_no_ignore_ids_or_silent_grid_conversion(self):
        with self.assertRaises(ValueError):
            independent_confusion(np.array([[255]], np.uint8), np.zeros((1, 1), np.uint8))
        with self.assertRaises(ValueError):
            independent_confusion(np.zeros((1, 1), np.uint8), np.zeros((1, 2), np.uint8))
