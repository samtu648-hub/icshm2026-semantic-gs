from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.semantic_cable_binary import (
    IMAGE_SHAPE,
    binary_target,
    fuse_cable_probability,
    relative_gate,
    score_metrics,
    select_calibration_threshold,
)


class CableBinaryTests(unittest.TestCase):
    def test_binary_target_maps_only_official_cable_to_positive(self) -> None:
        label = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
        label[0, :5] = np.asarray([0, 1, 2, 3, 4], dtype=np.uint8)
        target = binary_target(label)
        self.assertEqual(target[0, :5].tolist(), [0, 0, 1, 0, 0])
        self.assertEqual(int(target.sum()), 1)

    def test_fusion_preserves_every_structure_pixel(self) -> None:
        base = np.asarray([[0, 1, 2], [3, 4, 0]], dtype=np.uint8)
        probability = np.asarray(
            [[0.9, 0.99, 0.1], [0.99, 0.99, 0.2]], dtype=np.float32
        )
        fused, protected = fuse_cable_probability(
            base, probability, threshold=0.5
        )
        self.assertTrue(np.array_equal(fused[protected], base[protected]))
        self.assertEqual(fused.tolist(), [[2, 1, 0], [3, 4, 0]])
        self.assertTrue(set(np.unique(fused[~protected])).issubset({0, 2}))

    def test_relative_gate_and_selection_require_real_improvement(self) -> None:
        baseline = {
            "mIoU_with_background": 0.70,
            "background_to_stay_cable_pixels": 100,
            "per_class": {"stay_cable": {"iou": 0.60, "recall": 0.70}},
        }
        candidate = {
            "mIoU_with_background": 0.72,
            "background_to_stay_cable_pixels": 110,
            "per_class": {"stay_cable": {"iou": 0.63, "recall": 0.74}},
        }
        gate = relative_gate(
            candidate=candidate,
            baseline=baseline,
            protected_predictions_exact=True,
            minimum_iou_gain=0.02,
            minimum_recall_gain=0.03,
            minimum_miou_gain=0.0,
            maximum_background_fp_ratio=1.25,
        )
        self.assertTrue(gate["passed"])
        record = {
            "threshold": 0.6,
            "metrics": candidate,
            "score": score_metrics(candidate),
            "gate": gate,
        }
        failed = {
            **record,
            "threshold": 0.7,
            "score": record["score"] + 1.0,
            "gate": {**gate, "passed": False},
        }
        self.assertEqual(select_calibration_threshold([record, failed]), record)

    def test_invalid_probability_is_rejected(self) -> None:
        base = np.zeros((2, 2), dtype=np.uint8)
        probability = np.asarray([[0.0, 1.1], [0.5, 0.2]], dtype=np.float32)
        with self.assertRaises(ValueError):
            fuse_cable_probability(base, probability, threshold=0.5)


if __name__ == "__main__":
    unittest.main()
