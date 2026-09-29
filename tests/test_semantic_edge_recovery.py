from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.semantic_edge_recovery import (
    edge_recovery_gate,
    recover_edge_connected_stay_cable,
    select_edge_recovery_candidate,
)


class EdgeRecoveryTests(unittest.TestCase):
    def _teacher(self) -> np.ndarray:
        teacher = np.zeros((5, 7, 7), dtype=np.float32)
        teacher[0] = 0.9
        teacher[2] = 0.1
        teacher[2, 3, 3] = 0.95
        teacher[0, 3, 3] = 0.05
        teacher[2, 3, 4] = 0.65
        teacher[0, 3, 4] = 0.35
        return teacher

    def test_recovery_requires_edge_connectivity_and_keeps_protected_pixels(self) -> None:
        base = np.zeros((7, 7), dtype=np.uint8)
        base[0, 0] = 1
        base[0, 1] = 3
        base[0, 2] = 4
        base[3, 4] = 2
        rgb = np.zeros((7, 7, 3), dtype=np.uint8)
        rgb[:, 4:] = 255
        fused, fixed, recovered, diagnostics = recover_edge_connected_stay_cable(
            base,
            self._teacher(),
            rgb,
            high_stay_cable_threshold=0.8,
            low_stay_cable_threshold=0.55,
            edge_quantile=0.8,
            adjacency_radius=1,
        )
        self.assertTrue(np.array_equal(fused[fixed], base[fixed]))
        self.assertTrue(bool(recovered[3, 4]))
        self.assertEqual(int(fused[3, 4]), 2)
        self.assertEqual(diagnostics["recovered_pixels"], 1)

    def test_recovery_does_not_restore_without_a_base_cable_label(self) -> None:
        base = np.zeros((7, 7), dtype=np.uint8)
        rgb = np.zeros((7, 7, 3), dtype=np.uint8)
        rgb[:, 4:] = 255
        fused, _fixed, recovered, _diagnostics = recover_edge_connected_stay_cable(
            base,
            self._teacher(),
            rgb,
            high_stay_cable_threshold=0.8,
            low_stay_cable_threshold=0.55,
            edge_quantile=0.8,
            adjacency_radius=1,
        )
        self.assertFalse(recovered.any())
        self.assertEqual(int(fused[3, 4]), 0)

    def test_gate_requires_an_actual_quality_improvement(self) -> None:
        raw = {"background_to_stay_cable_pixels": 1000}
        baseline = {
            "mIoU_with_background": 0.7,
            "background_to_stay_cable_pixels": 100,
            "per_class": {"stay_cable": {"iou": 0.7, "recall": 0.7}},
        }
        recovered = {
            "mIoU_with_background": 0.704,
            "background_to_stay_cable_pixels": 105,
            "per_class": {"stay_cable": {"iou": 0.704, "recall": 0.708}},
        }
        gate = edge_recovery_gate(
            raw_sem001_metrics=raw,
            baseline_metrics=baseline,
            recovered_metrics=recovered,
            protected_predictions_exact=True,
            recovered_pixels=9,
            gates={
                "mIoU_with_background_improvement_min": 0.002,
                "stay_cable_iou_improvement_min": 0.002,
                "stay_cable_recall_improvement_min": 0.005,
                "background_to_stay_cable_growth_max_fraction": 0.1,
                "raw_background_to_stay_cable_reduction_min_fraction": 0.8,
            },
        )
        self.assertTrue(gate["passed"])
        selected = select_edge_recovery_candidate(
            [
                {
                    "recovered_pixels": 8,
                    "recovered_metrics": recovered,
                    "gate": gate,
                },
                {
                    "recovered_pixels": 1,
                    "recovered_metrics": baseline,
                    "gate": {**gate, "passed": False},
                },
            ]
        )
        self.assertEqual(selected["recovered_pixels"], 8)
