from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fusion.mapanything_hybrid import merge_baseline_and_supplement, select_hybrid_map_points  # noqa: E402


class MapAnythingHybridTest(unittest.TestCase):
    def test_only_high_confidence_ordinary_low_coverage_points_are_added(self) -> None:
        baseline_xyz = np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)
        baseline_rgb = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
        points = {
            "xyz": np.asarray([[0.2, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float32),
            "rgb": np.asarray([[10, 0, 0], [20, 0, 0], [30, 0, 0], [40, 0, 0]], dtype=np.uint8),
            "calibrated_confidence": np.asarray([0.95, 0.90, 0.70, 0.95], dtype=np.float32),
            "support_views": np.asarray([3, 3, 3, 2], dtype=np.int16),
            "line_structure": np.asarray([0, 0, 1, 0], dtype=np.uint8),
            "source_image_index": np.asarray([0, 1, 2, 3], dtype=np.int16),
        }
        supplement, details = select_hybrid_map_points(
            baseline_xyz,
            points,
            r0=1.0,
            minimum_confidence=0.75,
            minimum_support=3,
            coverage_radius_r0=1.5,
            maximum_added_points=3,
        )
        self.assertEqual(details["quality_points"], 2)
        self.assertEqual(details["added_points"], 1)
        self.assertTrue(np.allclose(supplement["xyz"], [[2.0, 0.0, 0.0]]))
        merged_xyz, merged_rgb = merge_baseline_and_supplement(baseline_xyz, baseline_rgb, supplement)
        self.assertTrue(np.array_equal(merged_xyz[:2], baseline_xyz))
        self.assertTrue(np.array_equal(merged_rgb[:2], baseline_rgb))
        self.assertEqual(len(merged_xyz), 3)

    def test_capacity_tie_break_is_deterministic(self) -> None:
        baseline = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
        points = {
            "xyz": np.asarray([[3.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float32),
            "rgb": np.zeros((2, 3), dtype=np.uint8),
            "calibrated_confidence": np.asarray([0.9, 0.9], dtype=np.float32),
            "support_views": np.asarray([3, 3], dtype=np.int16),
            "line_structure": np.asarray([0, 0], dtype=np.uint8),
            "source_image_index": np.asarray([3, 2], dtype=np.int16),
        }
        selected, details = select_hybrid_map_points(
            baseline,
            points,
            r0=1.0,
            minimum_confidence=0.8,
            minimum_support=3,
            coverage_radius_r0=1.0,
            maximum_added_points=1,
        )
        self.assertEqual(details["added_points"], 1)
        self.assertTrue(np.allclose(selected["xyz"], [[5.0, 0.0, 0.0]]))


if __name__ == "__main__":
    unittest.main()
