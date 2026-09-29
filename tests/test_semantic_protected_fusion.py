from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.semantic_protected_fusion import (
    fuse_background_stay_cable,
    partition_training_names,
    protected_fusion_gate,
    select_calibration_candidate,
)


class ProtectedFusionTests(unittest.TestCase):
    def _teacher(self) -> np.ndarray:
        teacher = np.zeros((5, 2, 3), dtype=np.float32)
        teacher[0] = 0.9
        teacher[2] = 0.1
        teacher[2, 0, 0] = 0.95
        teacher[0, 0, 0] = 0.05
        return teacher

    def test_fusion_only_changes_background_stay_cable_lane(self) -> None:
        base = np.asarray([[0, 1, 2], [3, 4, 0]], dtype=np.uint8)
        fused, fixed = fuse_background_stay_cable(
            base, self._teacher(), stay_cable_threshold=0.5
        )
        self.assertTrue(np.array_equal(fused[fixed], base[fixed]))
        self.assertEqual(int(fused[0, 0]), 2)
        self.assertEqual(int(fused[1, 2]), 0)
        self.assertTrue(set(np.unique(fused[~fixed])).issubset({0, 2}))

    def test_partition_is_deterministic_and_complete(self) -> None:
        names = [f"{index:03d}.png" for index in range(1, 220)]
        first = partition_training_names(names, expected_count=219)
        second = partition_training_names(names, expected_count=219)
        self.assertEqual(first, second)
        self.assertEqual(set(first["calibration"]) | set(first["check"]), set(names))
        self.assertFalse(set(first["calibration"]) & set(first["check"]))

    def test_gate_requires_protected_exactness_and_selects_only_passing(self) -> None:
        baseline = {
            "mIoU_with_background": 0.5,
            "background_to_stay_cable_pixels": 100,
            "per_class": {
                "deck": {"tp": 1, "fp": 1},
                "stay_cable": {"iou": 0.1, "recall": 0.1},
                "tower": {"tp": 1, "fp": 1},
                "foundation": {"tp": 1, "fp": 1},
            },
        }
        fused = {
            "mIoU_with_background": 0.6,
            "background_to_stay_cable_pixels": 10,
            "per_class": {
                "deck": {"tp": 1, "fp": 1},
                "stay_cable": {"iou": 0.8, "recall": 0.9},
                "tower": {"tp": 1, "fp": 1},
                "foundation": {"tp": 1, "fp": 1},
            },
        }
        gate = protected_fusion_gate(
            baseline_metrics=baseline,
            fused_metrics=fused,
            protected_predictions_exact=True,
            gates={
                "stay_cable_iou_min": 0.7,
                "stay_cable_recall_min": 0.85,
                "background_to_stay_cable_reduction_min_fraction": 0.8,
            },
        )
        self.assertTrue(gate["passed"])
        selected = select_calibration_candidate(
            [
                {"stay_cable_threshold": 0.8, "gate": gate},
                {"stay_cable_threshold": 0.9, "gate": {**gate, "passed": False}},
            ]
        )
        self.assertEqual(selected["stay_cable_threshold"], 0.8)
