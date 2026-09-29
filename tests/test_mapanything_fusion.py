from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fusion.mapanything_p0 import read_binary_ply_xyz, voxel_downsample, write_scaffold_init_ply  # noqa: E402
from fusion.scaffold_init import build_scaffold_init_scene  # noqa: E402


class MapAnythingFusionTest(unittest.TestCase):
    def test_scaffold_ply_roundtrip_and_confidence_voxel_choice(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            xyz = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
            rgb = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
            path = root / "init.ply"
            write_scaffold_init_ply(path, xyz, rgb)
            self.assertTrue(np.allclose(read_binary_ply_xyz(path), xyz))
            fused = voxel_downsample(
                {"xyz": xyz, "rgb": rgb, "calibrated_confidence": np.array([0.1, 0.9, 0.5], dtype=np.float32), "support_views": np.array([3, 3, 3]), "line_structure": np.array([0, 0, 0], dtype=np.uint8)},
                0.1,
                maximum_points=10,
            )
            self.assertEqual(len(fused["xyz"]), 2)
            self.assertTrue(any(np.allclose(point, [0.01, 0.0, 0.0]) for point in fused["xyz"]))

    def test_scene_builder_never_overwrites_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            (source / "images").mkdir(parents=True)
            sparse = source / "sparse" / "0"
            sparse.mkdir(parents=True)
            (sparse / "cameras.txt").write_text("camera", encoding="utf-8")
            (sparse / "images.txt").write_text("images", encoding="utf-8")
            init = root / "candidate.ply"
            write_scaffold_init_ply(init, np.zeros((1, 3), dtype=np.float32), np.zeros((1, 3), dtype=np.uint8))
            destination = root / "candidate_scene"
            manifest = build_scaffold_init_scene(source, init, destination)
            self.assertEqual(manifest["candidate_init_ply_sha256"], manifest["scene_init_ply_sha256"])
            self.assertTrue((destination / "images").is_symlink())
            self.assertTrue((destination / "sparse" / "0" / "points3D.ply").is_file())
            with self.assertRaises(FileExistsError):
                build_scaffold_init_scene(source, init, destination)


if __name__ == "__main__":
    unittest.main()
