from __future__ import annotations

import unittest

import numpy as np

from src.semantic_foundation_verifier import (
    confusion_after_foundation_rejection,
    foundation_gate,
    select_foundation_threshold,
    verify_foundation,
)
from src.semantic_protected_fusion import local_confusion, metrics_from_confusion


class FoundationVerifierTests(unittest.TestCase):
    def test_only_low_probability_foundation_can_change(self) -> None:
        base = np.asarray([[0, 1, 2], [3, 4, 4]], dtype=np.uint8)
        probability = np.asarray(
            [[0.0, 0.0, 0.0], [0.0, 0.2, 0.9]], dtype=np.float32
        )
        verified, candidates, rejected = verify_foundation(
            base, probability, threshold=0.5
        )
        np.testing.assert_array_equal(
            verified, np.asarray([[0, 1, 2], [3, 0, 4]], dtype=np.uint8)
        )
        np.testing.assert_array_equal(candidates, base == 4)
        np.testing.assert_array_equal(
            rejected,
            np.asarray([[False, False, False], [False, True, False]]),
        )
        for class_id in (1, 2, 3):
            np.testing.assert_array_equal(verified == class_id, base == class_id)

    def test_confusion_transform_matches_direct_prediction(self) -> None:
        target = np.asarray([[0, 4, 4], [1, 2, 3]], dtype=np.uint8)
        base = np.asarray([[4, 4, 4], [1, 2, 3]], dtype=np.uint8)
        probability = np.asarray(
            [[0.1, 0.2, 0.9], [0.0, 0.0, 0.0]], dtype=np.float32
        )
        verified, _, rejected = verify_foundation(base, probability, threshold=0.5)
        baseline_confusion = local_confusion(target, base)
        rejected_by_gt = np.bincount(target[rejected], minlength=5)
        transformed = confusion_after_foundation_rejection(
            baseline_confusion, rejected_by_gt
        )
        np.testing.assert_array_equal(transformed, local_confusion(target, verified))

    def test_gate_and_selection_require_all_precision_first_conditions(self) -> None:
        baseline_confusion = np.asarray(
            [
                [100, 0, 0, 0, 20],
                [0, 100, 0, 0, 0],
                [0, 0, 100, 0, 0],
                [0, 0, 0, 100, 0],
                [0, 0, 0, 0, 100],
            ],
            dtype=np.int64,
        )
        candidate_confusion = confusion_after_foundation_rejection(
            baseline_confusion, np.asarray([15, 0, 0, 0, 2], dtype=np.int64)
        )
        baseline = metrics_from_confusion(baseline_confusion)
        candidate = metrics_from_confusion(candidate_confusion)
        passing = foundation_gate(
            candidate=candidate,
            baseline=baseline,
            immutable_predictions_exact=True,
            minimum_iou_gain=0.02,
            minimum_precision_gain=0.05,
            minimum_recall=0.95,
            minimum_miou_gain=0.0,
            minimum_background_fp_reduction=0.5,
        )
        self.assertTrue(passing["passed"])
        failing = foundation_gate(
            candidate=candidate,
            baseline=baseline,
            immutable_predictions_exact=False,
            minimum_iou_gain=0.02,
            minimum_precision_gain=0.05,
            minimum_recall=0.95,
            minimum_miou_gain=0.0,
            minimum_background_fp_reduction=0.5,
        )
        self.assertFalse(failing["passed"])
        records = [
            {"threshold": 0.2, "metrics": candidate, "gate": passing},
            {"threshold": 0.8, "metrics": baseline, "gate": failing},
        ]
        self.assertEqual(select_foundation_threshold(records)["threshold"], 0.2)

    def test_invalid_probability_is_rejected(self) -> None:
        base = np.asarray([[4]], dtype=np.uint8)
        with self.assertRaises(ValueError):
            verify_foundation(
                base, np.asarray([[1.2]], dtype=np.float32), threshold=0.5
            )


if __name__ == "__main__":
    unittest.main()
