import unittest
import numpy as np
from src.camera_adapter import CameraFromWorld
from src.quality_backbone import CameraRequest
from src.quality_native_gaussian import projection_matrix


class NativeProjectionTests(unittest.TestCase):
    def test_off_center_principal_point_and_non_square_focal(self):
        k = np.array([[700., 0, 350], [0, 620, 240], [0, 0, 1]])
        camera = CameraRequest(800, 600, k, CameraFromWorld(np.eye(3), np.zeros(3)))
        xyz = np.array([[.3, -.2, 4, 1], [-.2, .8, 7, 1]])
        clip = xyz @ projection_matrix(camera).T
        ndc = clip[:, :2] / clip[:, 3:4]
        pixels_edge_origin = (ndc + 1) * np.array([800, 600]) / 2
        expected = xyz[:, :3] @ k.T
        expected = expected[:, :2] / expected[:, 2:3]
        np.testing.assert_allclose(pixels_edge_origin, expected, atol=2e-5)

    def test_centered_projection_matches_native_fov_formula(self):
        k = np.array([[925., 0, 660], [0, 925, 494.5], [0, 0, 1]])
        c = CameraRequest(1320, 989, k, CameraFromWorld(np.eye(3), np.zeros(3)))
        p = projection_matrix(c)
        self.assertEqual(p[0, 2], 0)
        self.assertEqual(p[1, 2], 0)
        self.assertAlmostEqual(p[0, 0], 2 * 925 / 1320, places=6)


if __name__ == "__main__":
    unittest.main()
