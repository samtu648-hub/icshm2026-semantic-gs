import copy
import unittest
import numpy as np

try:
    from scripts.check_sem311_quality import check_metrics
    from src.semantic2d import confusion_metrics
except ImportError:
    check_metrics = None


@unittest.skipIf(check_metrics is None, "optional audit runtime unavailable")
class IndependentQualityTest(unittest.TestCase):
    def test_recomputes_counts_and_scores(self):
        matrix = np.eye(5, dtype=np.int64)*17
        matrix[0,4] = 7
        matrix[4,0] = 3
        check_metrics(matrix, confusion_metrics(matrix))

    def test_rejects_tampered_score_and_count(self):
        matrix = np.eye(5, dtype=np.int64)*17
        expected = confusion_metrics(matrix)
        for key in ("tp", "iou"):
            bad = copy.deepcopy(expected)
            bad["per_class"]["foundation"][key] += .01
            with self.assertRaises(ValueError):
                check_metrics(matrix,bad)
