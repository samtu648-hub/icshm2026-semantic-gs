from __future__ import annotations

import unittest

import numpy as np

from src.semantic_deck_binary import (
    deck_gate,
    recover_deck,
    select_calibration_threshold,
)
from src.semantic_protected_fusion import local_confusion, metrics_from_confusion


class DeckBinaryTests(unittest.TestCase):
    def test_only_original_background_can_become_deck(self) -> None:
        base = np.asarray([[0, 0, 1], [2, 3, 4]], dtype=np.uint8)
        pre_foundation = np.asarray([[0, 4, 1], [2, 3, 4]], dtype=np.uint8)
        probability = np.asarray(
            [[0.9, 0.9, 0.9], [0.9, 0.9, 0.9]], dtype=np.float32
        )
        recovered, eligible, additions = recover_deck(
            base, pre_foundation, probability, threshold=0.8
        )
        np.testing.assert_array_equal(
            recovered, np.asarray([[1, 0, 1], [2, 3, 4]], dtype=np.uint8)
        )
        np.testing.assert_array_equal(
            eligible,
            np.asarray([[True, False, False], [False, False, False]]),
        )
        np.testing.assert_array_equal(additions, eligible)

    def test_existing_classes_and_rejected_foundation_are_exact(self) -> None:
        base = np.asarray([[0, 1, 2], [3, 4, 0]], dtype=np.uint8)
        pre_foundation = np.asarray([[0, 1, 2], [3, 4, 4]], dtype=np.uint8)
        probability = np.ones(base.shape, dtype=np.float32)
        recovered, _, _ = recover_deck(
            base, pre_foundation, probability, threshold=0.5
        )
        np.testing.assert_array_equal(
            recovered, np.asarray([[1, 1, 2], [3, 4, 0]], dtype=np.uint8)
        )
        for class_id in (2, 3, 4):
            np.testing.assert_array_equal(recovered == class_id, base == class_id)
        self.assertTrue(np.all((base == 1) <= (recovered == 1)))

    def test_gate_and_selection_require_quality_and_protection(self) -> None:
        target = np.asarray([[0, 0, 1, 1, 4], [0, 0, 2, 3, 4]], dtype=np.uint8)
        baseline_prediction = np.asarray(
            [[0, 1, 1, 0, 4], [0, 0, 2, 3, 4]], dtype=np.uint8
        )
        candidate_prediction = np.asarray(
            [[0, 1, 1, 1, 4], [0, 0, 2, 3, 4]], dtype=np.uint8
        )
        baseline = metrics_from_confusion(local_confusion(target, baseline_prediction))
        candidate = metrics_from_confusion(local_confusion(target, candidate_prediction))
        gate = deck_gate(
            candidate=candidate,
            baseline=baseline,
            protected_predictions_exact=True,
            minimum_iou_gain=0.1,
            minimum_recall_gain=0.1,
            minimum_precision=0.6,
            minimum_miou_gain=0.0,
            maximum_background_fp_ratio=1.0,
        )
        self.assertTrue(gate["passed"])
        rejected = {**gate, "passed": False}
        records = [
            {"threshold": 0.8, "metrics": candidate, "gate": gate},
            {"threshold": 0.9, "metrics": candidate, "gate": rejected},
        ]
        self.assertEqual(select_calibration_threshold(records)["threshold"], 0.8)

    def test_invalid_probability_is_rejected(self) -> None:
        base = np.asarray([[0]], dtype=np.uint8)
        with self.assertRaises(ValueError):
            recover_deck(
                base,
                base,
                np.asarray([[1.2]], dtype=np.float32),
                threshold=0.5,
            )


if __name__ == "__main__":
    unittest.main()
