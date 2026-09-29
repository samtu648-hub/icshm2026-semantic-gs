from __future__ import annotations

import unittest

import numpy as np

from src.semantic_cable_network_cleanup import (
    apply_cable_network_probability_transfer,
    cable_network_keep_masks,
    validate_cable_network_rule,
)
from src.semantic_postprocess import BACKGROUND, STAY_CABLE

RULE = {
    "hough_threshold": 8,
    "minimum_line_length_px": 12,
    "maximum_line_gap_px": 2,
    "structure_contact_radius_px": 6,
    "retained_line_half_width_px": 1,
    "main_max_verticality": 0.8,
    "stay_min_verticality": 0.2,
    "hanger_min_verticality": 0.85,
    "main_tower_top_fraction": 0.5,
    "main_boundary_band_px": 4,
    "hanger_main_contact_radius_px": 4,
}


def probabilities_with_labels(labels: np.ndarray) -> np.ndarray:
    probabilities = np.full((5, *labels.shape), 0.01, dtype=np.float32)
    for index in range(5):
        probabilities[index, labels == index] = 0.96
    probabilities /= probabilities.sum(axis=0, keepdims=True)
    return probabilities


class CableNetworkCleanupTests(unittest.TestCase):
    def test_all_three_normal_cable_families_are_retained_but_water_line_is_not(
        self,
    ) -> None:
        labels = np.zeros((100, 100), dtype=np.uint8)
        labels[10:30, 15:25] = 3
        labels[10:30, 75:85] = 3
        labels[80:90, 10:90] = 1
        labels[20:22, 20:80] = STAY_CABLE  # main cable, tower top to tower top
        labels[22:80, 49:51] = STAY_CABLE  # hanger, main cable to deck
        for row in range(26, 80):  # stay cable, tower to deck
            column = 20 + (row - 26) // 4
            labels[row, column : column + 2] = STAY_CABLE
        labels[35:75, 5:7] = STAY_CABLE  # off-structure water/background line
        families, diagnostics = cable_network_keep_masks(
            probabilities_with_labels(labels), RULE
        )
        self.assertGreater(int(families["main_cable"].sum()), 0)
        self.assertGreater(int(families["hanger"].sum()), 0)
        self.assertGreater(int(families["stay_cable"].sum()), 0)
        self.assertEqual(
            int(np.logical_or.reduce(tuple(families.values()))[35:75, 3:9].sum()), 0
        )
        self.assertGreater(diagnostics["main_cable_line_count"], 0)
        self.assertGreater(diagnostics["hanger_line_count"], 0)
        self.assertGreater(diagnostics["stay_cable_line_count"], 0)

    def test_rejected_pixels_transfer_to_background_and_preserve_simplex(self) -> None:
        labels = np.zeros((100, 100), dtype=np.uint8)
        labels[10:30, 15:25] = 3
        labels[10:30, 75:85] = 3
        labels[80:90, 10:90] = 1
        labels[20:22, 20:80] = STAY_CABLE
        labels[35:75, 5:7] = STAY_CABLE
        raw = probabilities_with_labels(labels)
        cleaned, families, _diagnostics = apply_cable_network_probability_transfer(
            raw, RULE
        )
        retained = np.logical_or.reduce(tuple(families.values()))
        rejected = (raw.argmax(axis=0) == STAY_CABLE) & ~retained
        self.assertTrue(np.allclose(cleaned.sum(axis=0), 1.0))
        self.assertTrue(np.all(cleaned[STAY_CABLE, rejected] == 0.0))
        self.assertTrue(
            np.allclose(
                cleaned[BACKGROUND, rejected],
                raw[BACKGROUND, rejected] + raw[STAY_CABLE, rejected],
            )
        )

    def test_rule_schema_rejects_incomplete_or_inconsistent_thresholds(self) -> None:
        with self.assertRaises(ValueError):
            validate_cable_network_rule({"hough_threshold": 1})
        invalid = dict(RULE)
        invalid["hanger_min_verticality"] = 0.1
        with self.assertRaises(ValueError):
            validate_cable_network_rule(invalid)
