from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_sem123_band_recovery_dev_sweep as sweep

from src.semantic_band_recovery import recover_selective_band
from src.semantic_protected_fusion import fuse_background_stay_cable, local_confusion


def teacher_from_cable_probability(value: np.ndarray) -> np.ndarray:
    teacher = np.zeros((5, *value.shape), dtype=np.float32)
    teacher[2] = value
    teacher[0] = 1.0 - value
    return teacher


class Sem123HistogramTests(unittest.TestCase):
    def test_histogram_confusion_transform_matches_direct_recovery(self) -> None:
        grid = {
            "lower_threshold": 0.20,
            "upper_threshold": 0.30,
            "probability_minima": [0.20, 0.24, 0.28],
            "maximum_seed_distances_px": [2.0, 4.0, 8.0],
        }
        base = np.asarray([[0, 2, 0, 1], [0, 0, 0, 3], [4, 0, 0, 0]], dtype=np.uint8)
        teacher = teacher_from_cable_probability(np.asarray([[0.28, 0.29, 0.31, 0.28], [0.22, 0.25, 0.15, 0.31], [0.25, 0.28, 0.24, 0.20]], dtype=np.float32))
        target = np.asarray([[2, 0, 2, 1], [0, 2, 0, 3], [4, 0, 2, 0]], dtype=np.uint8)
        spec = {"id": "test", "baseline": False, "probability_minimum": 0.24, "maximum_seed_distance": 4.0, "sem001_prior_mode": "any"}
        baseline, _fixed = fuse_background_stay_cable(base, teacher, stay_cable_threshold=0.30)
        direct, selected, _fixed = recover_selective_band(base, teacher, probability_minimum=0.24, maximum_seed_distance=4.0, prior_mode="any")
        histogram = sweep.recovery_target_histogram(base=base, teacher=teacher, target=target, grid=grid)
        selected_counts = sweep.selected_target_counts(histogram, spec, grid)
        transformed = local_confusion(target, baseline)
        transformed[:, 0] -= selected_counts
        transformed[:, 2] += selected_counts
        self.assertTrue(np.array_equal(selected_counts, np.bincount(target[selected].ravel(), minlength=5)))
        self.assertTrue(np.array_equal(transformed, local_confusion(target, direct)))


if __name__ == "__main__":
    unittest.main()
