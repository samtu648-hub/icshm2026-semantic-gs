import random
import unittest

import numpy as np

from src.geometry_prior.camera_io import ColmapCamera
from src.semantic_multiview_deck import DepthObservation
from src.semantic_rgbd_deck import frozen_surface_features, random_native_rgbd_pair


class SemanticRGBDDeckTest(unittest.TestCase):
    def test_flat_plane_normal_and_depth_are_finite(self):
        depth = np.full((989, 1320), 4.0, dtype=np.float32)
        foreground = np.ones_like(depth)
        camera = ColmapCamera(1, "PINHOLE", 1320, 989, (900.0, 900.0, 660.0, 494.5))
        features = frozen_surface_features(
            DepthObservation(depth, foreground), camera, minimum_foreground_mass=0.5
        )
        self.assertEqual(features.shape, (989, 1320, 4))
        self.assertTrue(np.isfinite(features).all())
        self.assertTrue(np.allclose(features[..., 0], 0.0))
        self.assertLess(float(features[494, 660, 3]), -0.99)

    def test_rgb_geometry_label_crop_remains_aligned(self):
        y, x = np.indices((989, 1320))
        label = ((x + y) % 5).astype(np.uint8)
        image = np.repeat(label[..., None], 3, axis=2)
        geometry = np.zeros((989, 1320, 4), dtype=np.float32)
        geometry[..., 0] = label
        rgb, geo, target = random_native_rgbd_pair(
            image, geometry, label, 960, random.Random(7)
        )
        self.assertTrue(np.array_equal(rgb[..., 0], target))
        self.assertTrue(np.array_equal(geo[..., 0], target))


if __name__ == "__main__":
    unittest.main()
