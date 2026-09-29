from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geometry_prior.camera_io import (  # noqa: E402
    ColmapCamera,
    ColmapImage,
    backproject_pixels,
    pose_roundtrip,
    project_world_points,
    qvec_to_rotation,
    read_frames_txt,
    read_rigs_txt,
)
from geometry_prior.sim3 import umeyama_similarity  # noqa: E402


class GeometryPriorTest(unittest.TestCase):
    def test_qvec_pose_roundtrip_and_center(self) -> None:
        qvec = np.array([0.9238795325, 0.0, 0.3826834324, 0.0])
        rotation = qvec_to_rotation(qvec)
        translation = np.array([1.2, -0.4, 2.3])
        recovered_rotation, recovered_translation = pose_roundtrip(rotation, translation)
        self.assertTrue(np.allclose(recovered_rotation, rotation, atol=1e-8))
        self.assertTrue(np.allclose(recovered_translation, translation, atol=1e-8))
        self.assertTrue(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8))

    def test_projection_backprojection_roundtrip(self) -> None:
        image = ColmapImage(1, "001.png", 1, np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), tuple())
        camera = ColmapCamera(1, "SIMPLE_RADIAL", 1320, 989, (925.7, 660.0, 494.5, 0.0089))
        points = np.array([[0.0, 0.0, 4.0], [0.4, -0.2, 6.0], [-0.7, 0.3, 5.0]])
        pixels, depth = project_world_points(points, image, camera)
        recovered = backproject_pixels(pixels, depth, camera, image)
        self.assertTrue(np.allclose(recovered, points, atol=1e-5))

    def test_umeyama_similarity(self) -> None:
        source = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        target = 2.5 * (source @ rotation.T) + np.array([4.0, -2.0, 1.0])
        fit = umeyama_similarity(source, target)
        self.assertAlmostEqual(fit.scale, 2.5, places=6)
        self.assertTrue(np.allclose(fit.rotation, rotation, atol=1e-6))
        self.assertTrue(np.allclose(fit.apply(source), target, atol=1e-6))

    def test_minimal_rig_and_frame_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rigs.txt").write_text("1 1 CAMERA 1\n", encoding="utf-8")
            (root / "frames.txt").write_text("1 1 1 0 0 0 0 0 0 1 CAMERA 1 7\n", encoding="utf-8")
            rig = read_rigs_txt(root / "rigs.txt")[1]
            frame = read_frames_txt(root / "frames.txt")[1]
            self.assertEqual(rig.ref_sensor_id, 1)
            self.assertEqual(frame.data_ids, (("CAMERA", 1, 7),))


if __name__ == "__main__":
    unittest.main()
