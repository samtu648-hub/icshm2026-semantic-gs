import json
import unittest
from pathlib import Path

import numpy as np

from scripts.create_sem400_refinement_split import build
from src.semantic_boundary_direction import deterministic_control_lines, ellipsoid_line_radius


ROOT = Path(__file__).resolve().parents[1]


class Sem400DirectionSplitTests(unittest.TestCase):
    def test_registered_refinement_roles_are_disjoint_and_complete(self):
        split = build()
        fold = split["primary_fold"]
        holdout = set(fold["holdout_all_cameras"])
        buffer = set(fold["buffer_excluded_all_cameras"])
        train = set(fold["train_rgb_400"])
        parent = json.loads((ROOT / split["parent_split"]).read_text())["primary_fold"]
        self.assertEqual((len(train), len(fold["semantic_train_labeled"]), len(holdout), len(buffer)), (292, 201, 30, 31))
        self.assertFalse(holdout & buffer or holdout & train or buffer & train)
        self.assertEqual(holdout | buffer | train, set(parent["train_rgb_400"]))

    def test_control_lines_are_repeatable_unit_vectors(self):
        first = deterministic_control_lines(4096, 400)
        second = deterministic_control_lines(4096, 400)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0)

    def test_ellipsoid_axis_and_rotation_radius(self):
        scales = np.array([[2.0, 1.0, 0.5], [2.0, 1.0, 0.5]])
        identity = np.eye(3)[None].repeat(2, axis=0)
        lines = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(ellipsoid_line_radius(scales, identity, lines), [2.0, 0.5])
        rotation = np.array([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
        np.testing.assert_allclose(
            ellipsoid_line_radius(scales[:1], rotation, np.array([[0.0, 1.0, 0.0]])),
            [2.0],
        )


if __name__ == "__main__":
    unittest.main()
