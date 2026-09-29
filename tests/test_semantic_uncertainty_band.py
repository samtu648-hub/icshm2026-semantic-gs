from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.semantic_uncertainty_band import (
    connectivity_to_seed,
    high_confidence_seed,
    summarise_uncertainty_band,
    uncertainty_band,
)


def teacher_from_cable_probability(value: np.ndarray) -> np.ndarray:
    teacher = np.zeros((5, *value.shape), dtype=np.float32)
    teacher[2] = value
    teacher[0] = 1.0 - value
    return teacher


class UncertaintyBandTests(unittest.TestCase):
    def test_band_is_exactly_lower_inclusive_upper_exclusive_and_mutable(self) -> None:
        base = np.asarray([[0, 0, 1, 2], [3, 4, 0, 0]], dtype=np.uint8)
        probability = np.asarray([[0.19, 0.20, 0.25, 0.29], [0.22, 0.21, 0.30, 0.31]], dtype=np.float32)
        band, mutable = uncertainty_band(base, teacher_from_cable_probability(probability), lower=0.20, upper=0.30)
        expected = np.asarray([[False, True, False, True], [False, False, False, False]])
        self.assertTrue(np.array_equal(band, expected))
        self.assertFalse(mutable[0, 2])
        self.assertFalse(mutable[1, 0])

    def test_connectivity_distinguishes_seed_touching_and_detached_components(self) -> None:
        band = np.asarray([[False, True, False, False], [False, False, False, False], [False, False, True, True]], dtype=bool)
        seed = np.asarray([[True, False, False, False], [False, False, False, False], [False, False, False, False]], dtype=bool)
        target = np.asarray([[0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2]], dtype=np.uint8)
        result = connectivity_to_seed(band, seed, target)
        self.assertEqual(result["component_count"], 2)
        self.assertEqual(result["touching_seed_component_count"], 1)
        self.assertEqual(result["touching_seed"]["gt_stay_cable"], 1)
        self.assertEqual(result["detached"]["gt_background"], 1)

    def test_summary_reports_probability_and_distance_bins_without_writing_predictions(self) -> None:
        base = np.zeros((3, 4), dtype=np.uint8)
        probability = np.asarray([[0.20, 0.24, 0.30, 0.10], [0.28, 0.29, 0.35, 0.22], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        target = np.asarray([[2, 0, 2, 0], [0, 2, 0, 1], [0, 0, 0, 0]], dtype=np.uint8)
        band, seed, summary = summarise_uncertainty_band(
            base,
            teacher_from_cable_probability(probability),
            target,
            lower=0.20,
            upper=0.30,
            probability_bin_edges=(0.20, 0.25, 0.30),
            distance_bin_edges=(0.0, 1.0, 4.0, float("inf")),
        )
        self.assertEqual(int(band.sum()), 5)
        self.assertEqual(int(seed.sum()), 2)
        self.assertEqual(summary["band"]["counts"]["stay_cable"], 2)
        self.assertEqual(len(summary["probability_bins"]), 2)
        self.assertEqual(len(summary["seed_distance_bins"]), 3)


if __name__ == "__main__":
    unittest.main()
