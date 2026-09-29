from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.semantic_continuous_fusion import (
    calibrated_cable_probability,
    fit_logistic_calibrator,
    fuse_continuous_background_stay_cable,
    two_stream_features,
)


class ContinuousFusionTests(unittest.TestCase):
    def _teacher(self) -> np.ndarray:
        value = np.full((5, 2, 3), 0.01, dtype=np.float32)
        value[0] = np.asarray([[0.60, 0.20, 0.55], [0.20, 0.55, 0.60]], dtype=np.float32)
        value[2] = np.asarray([[0.30, 0.70, 0.35], [0.70, 0.35, 0.30]], dtype=np.float32)
        value /= value.sum(axis=0, keepdims=True)
        return value

    def test_features_encode_teacher_odds_and_binary_sem001_prior(self) -> None:
        base = np.asarray([[0, 2, 1], [3, 4, 0]], dtype=np.uint8)
        features = two_stream_features(base, self._teacher())
        self.assertEqual(features.shape, (2, 3, 2))
        self.assertGreater(features[0, 1, 0], 0.0)
        self.assertEqual(features[0, 1, 1], 1.0)
        self.assertEqual(features[0, 0, 1], -1.0)

    def test_fit_and_fuse_keep_protected_sem001_pixels_exact(self) -> None:
        features = np.asarray([[-3.0, -1.0], [-2.0, -1.0], [2.0, 1.0], [3.0, 1.0]], dtype=np.float32)
        targets = np.asarray([0, 0, 1, 1], dtype=np.uint8)
        calibrator = fit_logistic_calibrator(features, targets, l2=0.01, maximum_iterations=32)
        base = np.asarray([[0, 2, 1], [3, 4, 0]], dtype=np.uint8)
        fused, fixed, probability = fuse_continuous_background_stay_cable(base, self._teacher(), calibrator)
        self.assertTrue(np.array_equal(base[fixed], fused[fixed]))
        self.assertTrue(np.all((0.0 <= probability) & (probability <= 1.0)))
        self.assertEqual(fused.dtype, np.uint8)

    def test_calibrator_schema_is_strict(self) -> None:
        base = np.zeros((2, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            calibrated_cable_probability(base, self._teacher(), {"intercept": 0.0})
