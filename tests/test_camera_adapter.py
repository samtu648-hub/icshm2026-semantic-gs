from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import numpy as np

    from camera_adapter import (  # noqa: E402
        CameraFromWorld,
        Sim3,
        SimpleRadialCamera,
        official_to_model_camera,
        redistort_semantic,
        redistort_semantic_probabilities,
        undistort_rgb,
    )
except ModuleNotFoundError:
    HAS_CAMERA_PREPROCESSING_DEPS = False
else:
    HAS_CAMERA_PREPROCESSING_DEPS = True


@unittest.skipUnless(HAS_CAMERA_PREPROCESSING_DEPS, "NumPy and OpenCV are optional before R1 environment setup")
class CameraAdapterTest(unittest.TestCase):
    def test_official_to_model_camera_respects_sim3(self) -> None:
        official = CameraFromWorld(np.eye(3), np.array([-1.0, 2.0, -3.0]))
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        alignment = Sim3(2.0, rotation, np.array([4.0, 5.0, 6.0]))
        model = official_to_model_camera(official, alignment)
        expected_center = 2.0 * rotation @ official.center + np.array([4.0, 5.0, 6.0])
        self.assertTrue(np.allclose(model.center, expected_center))
        self.assertTrue(np.allclose(model.rotation, rotation.T))

    def test_semantic_redistortion_preserves_integer_class_ids(self) -> None:
        camera = SimpleRadialCamera(16, 12, 13.0, 8.0, 6.0, 0.02)
        labels = np.zeros((12, 16), dtype=np.uint8)
        labels[:, 5:10] = 2
        output = redistort_semantic(labels, camera)
        self.assertEqual(output.shape, labels.shape)
        self.assertTrue(set(np.unique(output)).issubset({0, 2}))

    def test_rgb_full_frame_undistortion_keeps_size(self) -> None:
        camera = SimpleRadialCamera(16, 12, 13.0, 8.0, 6.0, 0.02)
        image = np.arange(16 * 12 * 3, dtype=np.uint8).reshape(12, 16, 3)
        self.assertEqual(undistort_rgb(image, camera).shape, image.shape)

    def test_probability_redistortion_preserves_five_class_simplex(self) -> None:
        camera = SimpleRadialCamera(16, 12, 13.0, 8.0, 6.0, 0.02)
        probabilities = np.zeros((5, 12, 16), dtype=np.float32)
        probabilities[0] = 0.25
        probabilities[1] = 0.25
        probabilities[2] = 0.20
        probabilities[3] = 0.15
        probabilities[4] = 0.15
        output = redistort_semantic_probabilities(probabilities, camera)
        self.assertEqual(output.shape, probabilities.shape)
        self.assertTrue(np.all(output >= 0.0))
        self.assertTrue(np.allclose(output.sum(axis=0), 1.0, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
