import unittest

import numpy as np

from src.geometry_prior.camera_io import ColmapCamera, ColmapImage
from src.semantic_multiview_deck import DepthObservation
from src.semantic_xyz_deck_field import (
    coordinate_normalization,
    normalize_coordinates,
    surface_points_from_depth,
    surface_samples_from_view,
)


class SemanticXYZDeckFieldTest(unittest.TestCase):
    def test_surface_samples_follow_depth_and_deck_labels(self) -> None:
        camera = ColmapCamera(1, "PINHOLE", 3, 3, (1.0, 1.0, 1.0, 1.0))
        image = ColmapImage(
            1,
            "a.png",
            1,
            np.asarray([1.0, 0.0, 0.0, 0.0]),
            np.zeros(3),
            (),
        )
        labels = np.zeros((3, 3), dtype=np.uint8)
        labels[1, 1] = 1
        observation = DepthObservation(
            np.full((3, 3), 2.0, dtype=np.float32),
            np.ones((3, 3), dtype=np.float32),
        )
        points, binary = surface_samples_from_view(
            labels,
            observation,
            camera=camera,
            image=image,
            stride=1,
            minimum_foreground_mass=0.5,
        )
        self.assertEqual(points.shape, (9, 3))
        self.assertEqual(int(binary.sum()), 1)
        np.testing.assert_allclose(points[binary.astype(bool)][0], [0.0, 0.0, 2.0])
        unlabelled, y, x = surface_points_from_depth(
            observation,
            camera=camera,
            image=image,
            minimum_foreground_mass=0.5,
        )
        self.assertEqual(unlabelled.shape, (9, 3))
        self.assertEqual((len(y), len(x)), (9, 9))

    def test_robust_coordinate_normalization(self) -> None:
        points = np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        center, scale = coordinate_normalization(points)
        normalized = normalize_coordinates(points, center, scale)
        self.assertEqual(center.shape, (3,))
        self.assertGreater(scale, 0.0)
        self.assertTrue(np.isfinite(normalized).all())


if __name__ == "__main__":
    unittest.main()
