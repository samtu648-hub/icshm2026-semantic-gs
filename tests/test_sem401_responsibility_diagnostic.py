import unittest

import numpy as np

from src.semantic_responsibility_diagnostic import (
    jensen_shannon_rows,
    normalized_quaternion_rotations,
    project_world_points,
    symmetric_ellipsoid_children,
)


class ResponsibilityDiagnosticTest(unittest.TestCase):
    def test_identity_quaternion_and_ellipsoid_children(self):
        rotation = normalized_quaternion_rotations(np.array([[1.0, 0.0, 0.0, 0.0]]))
        np.testing.assert_allclose(rotation, np.eye(3)[None], atol=1e-12)
        plus, minus, radius = symmetric_ellipsoid_children(
            np.array([[1.0, 2.0, 3.0]]),
            np.array([[2.0, 1.0, 0.5]]),
            rotation,
            np.array([[1.0, 0.0, 0.0]]),
        )
        np.testing.assert_allclose(radius, [2.0])
        np.testing.assert_allclose(plus, [[3.0, 2.0, 3.0]])
        np.testing.assert_allclose(minus, [[-1.0, 2.0, 3.0]])

    def test_projection_uses_camera_from_world(self):
        pixels, depth = project_world_points(
            np.array([[1.0, 2.0, 4.0]]),
            np.eye(3),
            np.zeros(3),
            np.array([[10.0, 0.0, 5.0], [0.0, 20.0, 6.0], [0.0, 0.0, 1.0]]),
        )
        np.testing.assert_allclose(pixels, [[7.5, 16.0]])
        np.testing.assert_allclose(depth, [4.0])

    def test_jensen_shannon_is_zero_for_equal_and_positive_for_distinct(self):
        result = jensen_shannon_rows(
            np.array([[2, 2], [4, 0]], dtype=np.float64),
            np.array([[2, 2], [0, 4]], dtype=np.float64),
        )
        self.assertAlmostEqual(float(result[0]), 0.0)
        self.assertAlmostEqual(float(result[1]), float(np.log(2.0)))


if __name__ == "__main__":
    unittest.main()
