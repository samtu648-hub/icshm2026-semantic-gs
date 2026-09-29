from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geometry_prior.camera_io import ColmapImage  # noqa: E402
from geometry_prior.mapanything_adapter import (  # noqa: E402
    _scale_depth_to_sparse,
    _world_points_from_p0,
    build_window_manifest,
    training_names_from_split,
)
from geometry_prior.windowing import covisibility_windows  # noqa: E402


def complete_graph(image_count: int) -> dict:
    names = [f"{index:03d}.png" for index in range(1, image_count + 1)]
    adjacency = {}
    for source in names:
        adjacency[source] = [
            {"source": source, "target": target, "shared_point3d": image_count - index, "center_distance": float(index)}
            for index, target in enumerate(names)
            if target != source
        ]
    return {"schema_version": 1, "adjacency": adjacency, "nodes": [{"name": name} for name in names]}


class MapAnythingPreflightTest(unittest.TestCase):
    def test_covisibility_windows_are_training_only_and_overlap(self) -> None:
        graph = complete_graph(24)
        names = [f"{index:03d}.png" for index in range(1, 25)]
        windows = covisibility_windows(graph, names, window_size=8, overlap=3, max_windows=2)
        self.assertEqual(len(windows), 2)
        self.assertTrue(all(len(window["images"]) == 8 for window in windows))
        self.assertTrue(all(len(window["images"]) == len(set(window["images"])) for window in windows))
        self.assertGreaterEqual(len(set(windows[0]["images"]) & set(windows[1]["images"])), 3)
        self.assertTrue(set().union(*(set(window["images"]) for window in windows)).issubset(names))

    def test_split_rejects_holdout_or_guard_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            path.write_text(
                json.dumps({"primary_fold": {"train_rgb_400": ["001.png"], "holdout_all_cameras": ["001.png"]}}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                training_names_from_split(path)

    def test_manifest_contains_only_training_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph_path = root / "graph.json"
            split_path = root / "split.json"
            graph_path.write_text(json.dumps(complete_graph(20)), encoding="utf-8")
            names = [f"{index:03d}.png" for index in range(1, 21)]
            split_path.write_text(
                json.dumps({"primary_fold": {"train_rgb_400": names, "holdout_all_cameras": ["900.png"], "buffer_excluded_all_cameras": ["901.png"]}}),
                encoding="utf-8",
            )
            manifest = build_window_manifest(
                graph_path=graph_path,
                split_path=split_path,
                window_size=8,
                overlap=3,
                max_windows=2,
                experiment_id="GEO-010-PF-R1",
                source_commit_value="a" * 40,
                checkpoint_sha256="b" * 64,
                checkpoint_revision="c" * 40,
            )
            self.assertEqual(manifest["training_image_count"], 20)
            self.assertEqual(manifest["geometry_output_schema"]["schema_id"], "geometry_prior_npz_v2")
            self.assertTrue(set().union(*(set(item["images"]) for item in manifest["windows"])).issubset(names))

    def test_sparse_scale_and_p0_world_points(self) -> None:
        image = ColmapImage(1, "001.png", 1, np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), ((1.0, 1.0, 1),))
        depth = np.full((3, 3), 2.0, dtype=np.float32)
        intrinsic = np.eye(3, dtype=np.float64)
        sparse = {1: np.array([2.0, 2.0, 4.0])}
        scaled, ratio, count = _scale_depth_to_sparse(
            depth, intrinsic, image, sparse, minimum_correspondences=1, minimum_depth=1e-6
        )
        self.assertEqual(count, 1)
        self.assertAlmostEqual(ratio, 2.0)
        self.assertTrue(np.allclose(scaled, 4.0))
        points = _world_points_from_p0(scaled, intrinsic, np.eye(4))
        self.assertTrue(np.allclose(points[1, 1], [4.0, 4.0, 4.0]))


if __name__ == "__main__":
    unittest.main()
