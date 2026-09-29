from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.semantic_band_recovery import recover_selective_band, selective_recovery_mask


def teacher_from_cable_probability(value: np.ndarray) -> np.ndarray:
    teacher = np.zeros((5, *value.shape), dtype=np.float32)
    teacher[2] = value
    teacher[0] = 1.0 - value
    return teacher


class BandRecoveryTests(unittest.TestCase):
    def test_selects_only_near_high_band_pixels_with_optional_sem001_prior(self) -> None:
        base = np.asarray([[0, 2, 0, 1], [0, 0, 0, 3]], dtype=np.uint8)
        probability = np.asarray([[0.28, 0.28, 0.31, 0.28], [0.22, 0.29, 0.15, 0.31]], dtype=np.float32)
        selected, _seed = selective_recovery_mask(
            base, teacher_from_cable_probability(probability), probability_minimum=0.26,
            maximum_seed_distance=2.0, prior_mode="sem001_stay_cable",
        )
        self.assertTrue(selected[0, 1])
        self.assertFalse(selected[0, 0])
        self.assertFalse(selected[0, 3])

    def test_recovery_preserves_protected_classes_and_never_leaves_band(self) -> None:
        base = np.asarray([[0, 2, 1], [3, 4, 0]], dtype=np.uint8)
        probability = np.asarray([[0.28, 0.29, 0.31], [0.31, 0.25, 0.30]], dtype=np.float32)
        candidate, selected, fixed = recover_selective_band(
            base, teacher_from_cable_probability(probability), probability_minimum=0.20,
            maximum_seed_distance=8.0, prior_mode="any",
        )
        self.assertTrue(np.array_equal(candidate[fixed], base[fixed]))
        self.assertTrue(np.all((probability[selected] >= 0.20) & (probability[selected] < 0.30)))
        self.assertFalse(selected[1, 2])


if __name__ == "__main__":
    unittest.main()
