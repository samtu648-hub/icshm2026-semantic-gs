from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from create_rgb_split import CameraRecord, camera_center_from_world_to_camera, create_split  # noqa: E402


class RgbSplitTest(unittest.TestCase):
    def test_camera_center_uses_colmap_world_to_camera_convention(self) -> None:
        rotation = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        self.assertEqual(camera_center_from_world_to_camera(rotation, (2.0, -3.0, 4.0)), (-2.0, 3.0, -4.0))

    def test_split_keeps_holdout_and_guard_cameras_out_of_training(self) -> None:
        records = [
            CameraRecord("1.png", 1, (0.0, 0.0, 0.0)),
            CameraRecord("2.png", 2, (0.1, 0.0, 0.0)),
            CameraRecord("301.png", 301, (0.2, 0.0, 0.0)),
            CameraRecord("140.png", 140, (10.0, 0.0, 0.0)),
            CameraRecord("141.png", 141, (10.1, 0.0, 0.0)),
            CameraRecord("302.png", 302, (10.2, 0.0, 0.0)),
            CameraRecord("3.png", 3, (20.0, 0.0, 0.0)),
            CameraRecord("4.png", 4, (20.1, 0.0, 0.0)),
            CameraRecord("303.png", 303, (20.2, 0.0, 0.0)),
        ]
        payload = create_split(
            records,
            labeled_names={"1.png", "2.png", "3.png", "4.png", "140.png", "141.png"},
            unlabeled_names={"301.png", "302.png", "303.png"},
            clusters=3,
            seed=7,
            guard_multiplier=1.0,
            challenge_views={"140.png", "141.png"},
        )
        primary = payload["primary_fold"]
        train = set(primary["train_rgb_400"])
        self.assertFalse(train & set(primary["holdout_all_cameras"]))
        self.assertFalse(train & set(primary["buffer_excluded_all_cameras"]))
        self.assertTrue(primary["checks"]["no_train_camera_within_guard_radius"])
        self.assertEqual(set(primary["challenge_views"]), {"140.png", "141.png"})


if __name__ == "__main__":
    unittest.main()
