from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import cv2
    import numpy as np

    from prepare_rgb_colmap_scene import prepare_scene  # noqa: E402
except ModuleNotFoundError:
    HAS_PREPROCESSING_DEPS = False
else:
    HAS_PREPROCESSING_DEPS = True


@unittest.skipUnless(HAS_PREPROCESSING_DEPS, "NumPy and OpenCV are optional before R1 environment setup")
class PrepareRgbColmapSceneTest(unittest.TestCase):
    def test_prepares_pinhole_scene_and_retriangulates_track(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "dataset"
            (root / "images").mkdir(parents=True)
            (root / "unlabeled_Images").mkdir()
            (root / "camera_parameters").mkdir()
            image = np.full((8, 8, 3), (20, 100, 200), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(root / "images" / "1.png"), image))
            self.assertTrue(cv2.imwrite(str(root / "images" / "2.png"), image))
            self.assertTrue(cv2.imwrite(str(root / "images" / "3.png"), image))
            (root / "camera_parameters" / "cameras.txt").write_text(
                "1 SIMPLE_RADIAL 8 8 10 4 4 0\n", encoding="utf-8"
            )
            (root / "camera_parameters" / "images.txt").write_text(
                "1 1 0 0 0 0 0 0 1 1.png\n4 4 7\n"
                "2 1 0 0 0 -1 0 0 1 2.png\n2 4 7\n"
                "3 1 0 0 0 -2 0 0 1 3.png\n2 4 -1\n",
                encoding="utf-8",
            )
            split = Path(temp) / "split.json"
            split.write_text(
                json.dumps(
                    {
                        "split_id": "synthetic",
                        "primary_fold": {
                            "train_rgb_400": ["1.png", "2.png"],
                            "validation_labeled": ["3.png"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            output = Path(temp) / "prepared"
            metadata = prepare_scene(root, split, output, "train_rgb_400", 0.01, "validation_labeled")
            self.assertEqual(metadata["counts"]["training_images"], 2)
            self.assertEqual(metadata["counts"]["evaluation_images"], 1)
            self.assertFalse(metadata["evaluation"]["initialization_uses_evaluation_images"])
            self.assertEqual(metadata["counts"]["retriangulated_points"], 1)
            self.assertTrue((output / "images" / "1.png").is_file())
            self.assertTrue((output / "images" / "3.png").is_file())
            self.assertIn("PINHOLE", (output / "sparse" / "0" / "cameras.txt").read_text(encoding="utf-8"))
            self.assertIn("7 ", (output / "sparse" / "0" / "points3D.txt").read_text(encoding="utf-8"))
            self.assertEqual((output / "sparse" / "0" / "test.txt").read_text(encoding="utf-8"), "3.png\n")


if __name__ == "__main__":
    unittest.main()
