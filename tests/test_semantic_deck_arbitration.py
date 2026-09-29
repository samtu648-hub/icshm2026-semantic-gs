import unittest

import numpy as np

from src.semantic_deck_arbitration import (
    arbitrate_deck,
    arbitration_gate,
    select_arbitration_threshold,
)


def metrics(*, iou, precision, recall, miou, background_fp):
    confusion = [[0] * 5 for _ in range(5)]
    confusion[0][1] = background_fp
    return {
        "mIoU_with_background": miou,
        "per_class": {"deck": {"iou": iou, "precision": precision, "recall": recall}},
        "confusion_matrix_rows_gt_columns_prediction": confusion,
    }


class DeckArbitrationTests(unittest.TestCase):
    def test_only_authorised_zero_one_lane_changes(self):
        base = np.asarray([[0, 1, 2], [3, 4, 0]], dtype=np.uint8)
        pre = np.asarray([[0, 1, 2], [3, 4, 4]], dtype=np.uint8)
        probability = np.asarray(
            [[0.9, 0.1, 0.9], [0.9, 0.9, 0.9]], dtype=np.float32
        )
        candidate, mutable, additions, removals = arbitrate_deck(
            base, pre, probability, threshold=0.5
        )
        np.testing.assert_array_equal(
            candidate, np.asarray([[1, 0, 2], [3, 4, 0]], dtype=np.uint8)
        )
        self.assertTrue(mutable[0, 0] and mutable[0, 1])
        self.assertFalse(mutable[1, 2])
        self.assertTrue(additions[0, 0] and removals[0, 1])

    def test_invalid_probability_or_threshold_is_rejected(self):
        base = np.zeros((2, 2), dtype=np.uint8)
        with self.assertRaises(ValueError):
            arbitrate_deck(base, base, np.zeros((2, 3), np.float32), threshold=0.5)
        with self.assertRaises(ValueError):
            arbitrate_deck(base, base, np.zeros((2, 2), np.float32), threshold=1.0)

    def test_gate_and_selection_require_all_registered_gains(self):
        baseline = metrics(
            iou=0.90, precision=0.95, recall=0.96, miou=0.82, background_fp=100
        )
        candidate = metrics(
            iou=0.91, precision=0.96, recall=0.97, miou=0.83, background_fp=90
        )
        gate = arbitration_gate(
            candidate=candidate,
            baseline=baseline,
            protected_predictions_exact=True,
            minimum_iou_gain=0.002,
            minimum_precision_gain=0.002,
            minimum_recall_gain=0.0,
            minimum_miou_gain=0.0,
            maximum_background_fp_ratio=1.0,
        )
        self.assertTrue(gate["passed"])
        selected = select_arbitration_threshold(
            [
                {"threshold": 0.5, "metrics": candidate, "gate": gate},
                {
                    "threshold": 0.9,
                    "metrics": metrics(
                        iou=0.92,
                        precision=0.97,
                        recall=0.95,
                        miou=0.84,
                        background_fp=80,
                    ),
                    "gate": {"passed": False},
                },
            ]
        )
        self.assertEqual(selected["threshold"], 0.5)


if __name__ == "__main__":
    unittest.main()
