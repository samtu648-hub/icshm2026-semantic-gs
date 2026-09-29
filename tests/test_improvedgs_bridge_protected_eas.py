import json
import unittest
from pathlib import Path

import numpy as np

from src.improvedgs_bridge_protected_eas import build_reliability_map, quantize_reliability


ROOT = Path(__file__).resolve().parents[1]


class BridgeProtectedEASTest(unittest.TestCase):
    def test_bridge_pixels_and_dilated_boundary_are_hard_protected(self):
        photo = np.zeros((33, 41, 3), dtype=np.uint8)
        parent = photo.copy()
        parent[:, 20:] = 180
        semantic = np.zeros((33, 41), dtype=np.uint8)
        semantic[16, 8] = 2
        reliability, stats = build_reliability_map(
            photo,
            parent,
            semantic,
            residual_sigma=0.0,
            bridge_dilation_pixels=2,
            minimum_reliability=0.25,
        )
        self.assertEqual(float(reliability[16, 8]), 1.0)
        self.assertEqual(float(reliability[16, 10]), 1.0)
        self.assertLessEqual(float(reliability[16, 30]), 0.250001)
        self.assertEqual(stats["protected_minimum_reliability"], 1.0)

    def test_nonbridge_reliability_is_monotone_with_parent_residual(self):
        photo = np.zeros((9, 101, 3), dtype=np.uint8)
        ramp = np.arange(101, dtype=np.uint8)
        parent = np.repeat(ramp[None, :, None], 9, axis=0)
        parent = np.repeat(parent, 3, axis=2)
        reliability, _ = build_reliability_map(
            photo,
            parent,
            None,
            residual_sigma=0.0,
            lower_quantile=0.5,
            upper_quantile=0.9,
            minimum_reliability=0.25,
        )
        self.assertTrue(np.all(np.diff(reliability[4]) <= 1e-7))
        self.assertAlmostEqual(float(reliability[4, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(reliability[4, 100]), 0.25, places=6)

    def test_quantization_preserves_unit_bridge_and_floor(self):
        value = np.asarray([[1.0, 0.25, 0.5]], dtype=np.float32)
        quantized = quantize_reliability(value)
        self.assertEqual(quantized.tolist(), [[255, 64, 128]])

    def test_extra_projection_protects_unlabeled_pixels(self):
        photo = np.zeros((13, 17, 3), dtype=np.uint8)
        parent = np.full_like(photo, 180)
        projected = np.zeros((13, 17), dtype=bool)
        projected[4, 6] = True
        reliability, stats = build_reliability_map(
            photo,
            parent,
            None,
            extra_protected_mask=projected,
            residual_sigma=0.0,
            minimum_reliability=0.25,
        )
        self.assertEqual(float(reliability[4, 6]), 1.0)
        self.assertAlmostEqual(stats["extra_protected_fraction"], 1.0 / (13 * 17))

    def test_registered_candidate_keeps_rgb170_controls(self):
        parent = json.loads((ROOT / "experiments/runs/RGB-170-IMPROVEDGS-BUDGET15M-H50K/config.yaml").read_text())
        candidate = json.loads((ROOT / "experiments/runs/RGB-174-IMPROVEDGS-BRIDGE-PROTECTED-EAS/config.yaml").read_text())
        for key in ("iterations", "densify_from_iter", "densify_until_iter", "densification_interval",
                    "use_eas", "use_las", "use_rap", "use_mu", "edge_sample_cams"):
            self.assertEqual(candidate["training"][key], parent["training"][key])
        self.assertEqual(candidate["training"]["budget"], parent["training"]["candidate_budget"])
        self.assertEqual(candidate["seed"], parent["seed"])
        self.assertEqual(candidate["data"]["initial_ply_sha256"], parent["data"]["initial_ply_sha256"])


if __name__ == "__main__":
    unittest.main()
