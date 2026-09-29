import unittest

import numpy as np

from src.rgb_semantic_coupling import (
    average_ranks,
    confusion_matrix,
    pearson_correlation,
    reference_edge_strength,
    semantic_metrics,
    spearman_correlation,
)


class RgbSemanticCouplingTests(unittest.TestCase):
    def test_correlations_and_tied_ranks(self) -> None:
        self.assertTrue(np.array_equal(average_ranks([3, 1, 1]), [3.0, 1.5, 1.5]))
        self.assertAlmostEqual(pearson_correlation([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertAlmostEqual(spearman_correlation([1, 2, 3], [9, 8, 7]), -1.0)
        self.assertIsNone(pearson_correlation([1, 1], [2, 3]))

    def test_confusion_and_metrics(self) -> None:
        target = np.array([[0, 1, 2, 3, 4]])
        prediction = np.array([[0, 1, 0, 3, 4]])
        confusion = confusion_matrix(target, prediction)
        self.assertEqual(int(confusion[2, 0]), 1)
        metrics = semantic_metrics(confusion)
        self.assertEqual(metrics["stay_cable_IoU"], 0.0)
        self.assertAlmostEqual(metrics["mIoU_with_background"], 0.7)

    def test_reference_edge_strength(self) -> None:
        rgb = np.zeros((2, 2, 3), dtype=np.float32)
        rgb[:, 1] = 1.0
        edge = reference_edge_strength(rgb)
        self.assertTrue(np.allclose(edge, 0.5))


if __name__ == "__main__":
    unittest.main()

