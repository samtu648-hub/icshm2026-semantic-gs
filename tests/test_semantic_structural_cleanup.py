from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from src.semantic_postprocess import BACKGROUND, STAY_CABLE
from src.semantic_structural_cleanup import (
    apply_structural_probability_transfer,
    structural_keep_mask,
    validate_structural_rule,
)

RULE = {
    "hough_threshold": 8,
    "minimum_line_length_px": 20,
    "maximum_line_gap_px": 2,
    "minimum_verticality": 0.8,
    "structure_contact_radius_px": 3,
    "retained_line_half_width_px": 1,
}

ROOT = Path(__file__).resolve().parents[1]


def probabilities_with_labels(labels: np.ndarray) -> np.ndarray:
    probabilities = np.full((5, *labels.shape), 0.01, dtype=np.float32)
    for index in range(5):
        probabilities[index, labels == index] = 0.96
    probabilities /= probabilities.sum(axis=0, keepdims=True)
    return probabilities


class StructuralCleanupTests(unittest.TestCase):
    def test_tower_to_deck_line_is_retained_and_off_structure_line_is_removed(
        self,
    ) -> None:
        labels = np.zeros((100, 100), dtype=np.uint8)
        labels[10:20, 45:55] = 3
        labels[80:90, 45:55] = 1
        labels[20:80, 49:51] = STAY_CABLE
        labels[20:80, 10:12] = STAY_CABLE
        retained, diagnostics = structural_keep_mask(
            probabilities_with_labels(labels), RULE
        )
        self.assertGreater(int(retained[20:80, 45:55].sum()), 0)
        self.assertEqual(int(retained[20:80, 8:14].sum()), 0)
        self.assertGreater(diagnostics["tower_deck_supported_line_count"], 0)

    def test_transfer_preserves_simplex_and_changes_only_rejected_cable_pixels(
        self,
    ) -> None:
        labels = np.zeros((100, 100), dtype=np.uint8)
        labels[10:20, 45:55] = 3
        labels[80:90, 45:55] = 1
        labels[20:80, 49:51] = STAY_CABLE
        labels[20:80, 10:12] = STAY_CABLE
        raw = probabilities_with_labels(labels)
        cleaned, retained, _diagnostics = apply_structural_probability_transfer(
            raw, RULE
        )
        rejected = (raw.argmax(axis=0) == STAY_CABLE) & ~retained
        self.assertTrue(np.allclose(cleaned.sum(axis=0), 1.0))
        self.assertTrue(np.all(cleaned[STAY_CABLE, rejected] == 0.0))
        self.assertTrue(
            np.allclose(
                cleaned[BACKGROUND, rejected],
                raw[BACKGROUND, rejected] + raw[STAY_CABLE, rejected],
            )
        )
        self.assertTrue(np.array_equal(cleaned[:, ~rejected], raw[:, ~rejected]))

    def test_rule_requires_exact_schema(self) -> None:
        with self.assertRaises(ValueError):
            validate_structural_rule({"hough_threshold": 1})
        invalid = dict(RULE)
        invalid["minimum_verticality"] = 0.0
        with self.assertRaises(ValueError):
            validate_structural_rule(invalid)

    def test_registered_diagnostic_forbids_validation_and_has_one_complete_rule(
        self,
    ) -> None:
        config = json.loads(
            (ROOT / "experiments/runs/SEM-080-STRUCTURAL-DIAG/config.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["experiment_id"], "SEM-080-STRUCTURAL-DIAG")
        self.assertEqual(config["data"]["training_images"], 219)
        self.assertEqual(config["data"]["validation_images_opened"], 0)
        self.assertFalse(config["data"]["uses_unlabeled_rgb"])
        self.assertFalse(config["data"]["uses_hidden_test_rgb_or_labels"])
        self.assertEqual(
            validate_structural_rule(config["structural_rule"]),
            config["structural_rule"],
        )
