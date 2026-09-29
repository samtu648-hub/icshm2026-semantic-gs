import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_external_colmap_camera.py"
SPEC = importlib.util.spec_from_file_location("render_external_colmap_camera", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ExternalColmapCameraParsingTest(unittest.TestCase):
    def test_reads_official_simple_radial_and_camera_from_world_pose(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cameras = root / "cameras.txt"
            images = root / "images.txt"
            cameras.write_text("# header\n1 SIMPLE_RADIAL 1320 989 1000 660 494.5 0.01\n", encoding="utf-8")
            images.write_text(
                "# header\n"
                "7 1 0 0 0 1 2 3 1 view.png\n"
                "\n",
                encoding="utf-8",
            )
            camera_id, camera = MODULE.read_simple_radial_camera(cameras, None)
            image_name, pose = MODULE.read_pose(images, "view.png", camera_id)

        self.assertEqual(camera_id, 1)
        self.assertEqual((camera.width, camera.height), (1320, 989))
        self.assertEqual(image_name, "view.png")
        np.testing.assert_allclose(pose.rotation, np.eye(3))
        np.testing.assert_allclose(pose.translation, [1, 2, 3])
        np.testing.assert_allclose(pose.center, [-1, -2, -3])

    def test_rejects_ambiguous_pose_without_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            images = Path(temporary) / "images.txt"
            images.write_text(
                "1 1 0 0 0 0 0 0 1 a.png\n\n"
                "2 1 0 0 0 0 0 0 1 b.png\n\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Expected exactly one selected pose"):
                MODULE.read_pose(images, None, 1)


if __name__ == "__main__":
    unittest.main()
