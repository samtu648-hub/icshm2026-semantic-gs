from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geometry_prior.camera_io import ColmapImage  # noqa: E402
from geometry_prior.output_schema import validate_geometry_manifest_v2  # noqa: E402
from geometry_prior.vggt_omega import (  # noqa: E402
    SMOKE_ANCHORS,
    build_vggt_omega_manifest,
    training_names_from_split,
)
from geometry_prior.vggt_omega_adapter import (  # noqa: E402
    audit_geometry_npz,
    balanced_preprocess_transform,
    repeatability_metrics,
    scale_depth_to_sparse,
    scale_pinhole_intrinsics,
    valid_depth_mask,
    world_points_from_official_p0,
    write_geometry_npz,
)


def complete_graph() -> dict:
    names = [f"{index:03d}.png" for index in range(100, 131)]
    edges = []
    for source_index, source in enumerate(names):
        for target_index in range(source_index + 1, len(names)):
            edges.append(
                {
                    "source": source,
                    "target": names[target_index],
                    "shared_point3d": 10_000 - source_index * len(names) - target_index,
                    "center_distance": 1.0,
                    "optical_axis_angle_deg": 1.0,
                }
            )
    adjacency = {
        source: [
            {
                "source": source,
                "target": target,
                "shared_point3d": 20_000 - target_index,
            }
            for target_index, target in enumerate(names)
            if target != source
        ][:3]
        for source in names
    }
    return {
        "schema_version": 1,
        "pose_convention": "CAMERA_FROM_WORLD; centers C=-R^T t",
        "nodes": [{"name": name} for name in names],
        "adjacency": adjacency,
        "edges": edges,
    }


def split_for(names: list[str]) -> dict:
    return {
        "split_id": "unit-split",
        "primary_fold": {
            "train_rgb_400": names,
            "holdout_all_cameras": [],
            "validation_labeled": [],
            "withheld_unlabeled": [],
            "buffer_excluded_all_cameras": [],
            "buffer_excluded_labeled": [],
            "buffer_excluded_unlabeled": [],
        },
    }


class VggtOmegaWindowTest(unittest.TestCase):
    def test_fixed_anchors_and_16_view_exact_prefix(self) -> None:
        graph = complete_graph()
        names = [node["name"] for node in graph["nodes"]]
        manifest_25 = build_vggt_omega_manifest(
            graph, names, view_count=25, split_id="unit-split"
        )
        manifest_16 = build_vggt_omega_manifest(
            graph, names, view_count=16, split_id="unit-split"
        )
        self.assertEqual(
            [window["anchor_image"] for window in manifest_25["windows"]],
            list(SMOKE_ANCHORS),
        )
        by_anchor_25 = {
            window["anchor_image"]: window for window in manifest_25["windows"]
        }
        for window in manifest_16["windows"]:
            canonical = by_anchor_25[window["anchor_image"]]["images"]
            self.assertEqual(window["images"], canonical[:16])
            self.assertEqual(window["canonical_25_images"], canonical)
            self.assertTrue(
                any(
                    fact["edge_source"] == "complete_edges"
                    for fact in window["co_visibility_facts"]
                )
            )
        validate_geometry_manifest_v2(manifest_25)
        validate_geometry_manifest_v2(manifest_16)

    def test_window_rejects_weak_padding(self) -> None:
        graph = complete_graph()
        graph["nodes"] = [
            *graph["nodes"][:23],
            next(node for node in graph["nodes"] if node["name"] == "126.png"),
        ]
        names = [node["name"] for node in graph["nodes"]]
        with self.assertRaisesRegex(ValueError, "without weak padding"):
            build_vggt_omega_manifest(graph, names, view_count=25)

    def test_split_rejects_validation_or_guard_overlap(self) -> None:
        split = split_for(["115.png", "126.png"])
        split["primary_fold"]["validation_labeled"] = ["115.png"]
        with self.assertRaisesRegex(ValueError, "overlaps validation or guard"):
            training_names_from_split(split)

    def test_repository_split_and_graph_build_both_windows(self) -> None:
        split = json.loads(
            (ROOT / "experiments" / "splits" / "rgb_r1_spatial_v1.json").read_text(
                encoding="utf-8"
            )
        )
        graph = json.loads(
            (ROOT / "experiments" / "geometry_camera_graph.json").read_text(
                encoding="utf-8"
            )
        )
        manifest = build_vggt_omega_manifest(
            graph,
            training_names_from_split(split),
            view_count=25,
            split_id=split["split_id"],
        )
        self.assertEqual(manifest["window_count"], 2)
        self.assertTrue(
            all(len(window["images"]) == 25 for window in manifest["windows"])
        )
        self.assertEqual(
            manifest["windows"][0]["images"][:16],
            [
                "115.png",
                "116.png",
                "114.png",
                "113.png",
                "117.png",
                "112.png",
                "111.png",
                "110.png",
                "118.png",
                "109.png",
                "108.png",
                "119.png",
                "107.png",
                "106.png",
                "120.png",
                "121.png",
            ],
        )
        self.assertEqual(
            manifest["windows"][1]["images"][:16],
            [
                "126.png",
                "127.png",
                "125.png",
                "124.png",
                "128.png",
                "123.png",
                "122.png",
                "129.png",
                "121.png",
                "120.png",
                "130.png",
                "119.png",
                "131.png",
                "118.png",
                "132.png",
                "133.png",
            ],
        )

    def test_repository_full_training_plan_covers_all_303_anchors(self) -> None:
        split = json.loads(
            (ROOT / "experiments" / "splits" / "rgb_r1_spatial_v1.json").read_text(
                encoding="utf-8"
            )
        )
        graph = json.loads(
            (ROOT / "experiments" / "geometry_camera_graph.json").read_text(
                encoding="utf-8"
            )
        )
        names = training_names_from_split(split)
        manifest = build_vggt_omega_manifest(
            graph,
            names,
            view_count=25,
            anchors=names,
            split_id=split["split_id"],
            scope="full_training",
        )
        self.assertEqual(manifest["window_count"], 303)
        self.assertEqual(manifest["unique_image_coverage"]["images"], names)
        self.assertTrue(
            all(len(window["images"]) == 25 for window in manifest["windows"])
        )


class VggtOmegaGeometryTest(unittest.TestCase):
    def test_balanced_transform_and_independent_intrinsic_scaling(self) -> None:
        transform = balanced_preprocess_transform(1320, 989)
        self.assertFalse(transform["crop_applied"])
        self.assertEqual(transform["target_size_wh"], [592, 448])
        intrinsics = np.asarray(
            [[1000.0, 0.0, 660.0], [0.0, 900.0, 494.5], [0.0, 0.0, 1.0]]
        )
        scaled = scale_pinhole_intrinsics(intrinsics, transform)
        self.assertAlmostEqual(scaled[0, 0], 1000.0 * 592.0 / 1320.0)
        self.assertAlmostEqual(scaled[1, 1], 900.0 * 448.0 / 989.0)
        self.assertAlmostEqual(scaled[0, 2], 660.0 * 592.0 / 1320.0)
        self.assertAlmostEqual(scaled[1, 2], 494.5 * 448.0 / 989.0)

    def test_sparse_depth_scale_requires_64_correspondences(self) -> None:
        image = ColmapImage(
            image_id=1,
            name="115.png",
            camera_id=1,
            qvec=np.asarray([1.0, 0.0, 0.0, 0.0]),
            tvec=np.zeros(3),
            points2d=tuple((2.0, 2.0, point_id) for point_id in range(64)),
        )
        sparse = {point_id: np.asarray([0.0, 0.0, 4.0]) for point_id in range(64)}
        depth = np.full((5, 5), 2.0, dtype=np.float32)
        intrinsics = np.asarray([[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]])
        scaled, ratio, count = scale_depth_to_sparse(depth, intrinsics, image, sparse)
        self.assertEqual(count, 64)
        self.assertEqual(ratio, 2.0)
        np.testing.assert_array_equal(scaled, np.full((5, 5), 4.0, dtype=np.float32))

        image_63 = ColmapImage(
            image_id=1,
            name="115.png",
            camera_id=1,
            qvec=image.qvec,
            tvec=image.tvec,
            points2d=image.points2d[:63],
        )
        with self.assertRaisesRegex(ValueError, "only 63 usable"):
            scale_depth_to_sparse(depth, intrinsics, image_63, sparse)

    def test_valid_mask_has_no_confidence_threshold(self) -> None:
        depth = np.asarray([[1.0, 0.0], [np.nan, 2.0]], dtype=np.float32)
        confidence = np.asarray([[-100.0, 1.0], [1.0, np.nan]], dtype=np.float32)
        np.testing.assert_array_equal(
            valid_depth_mask(depth, confidence),
            np.asarray([[True, False], [False, False]]),
        )

    def test_npz_audit_enforces_p0_roundtrip_dtype_and_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "geometry.npz"
            depth = np.asarray([[1.0, 2.0], [3.0, 0.0]], dtype=np.float32)
            confidence = np.asarray([[1.0, -2.0], [-100.0, 0.0]], dtype=np.float32)
            valid = valid_depth_mask(depth, confidence)
            intrinsics = np.asarray(
                [[100.0, 0.0, 0.5], [0.0, 110.0, 0.5], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            cam2world = np.asarray(
                [
                    [0.0, -1.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 2.0],
                    [0.0, 0.0, 1.0, 3.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            world = world_points_from_official_p0(depth, intrinsics, cam2world)
            write_geometry_npz(
                path,
                depth=depth,
                confidence=confidence,
                valid_mask=valid,
                world_points=world,
                pinhole_intrinsics=intrinsics,
                official_cam2world=cam2world,
                pred_cam2world=np.eye(4),
                depth_scale_ratio=2.0,
            )
            audit = audit_geometry_npz(path, expected_resolution_wh=(2, 2))
            self.assertEqual(audit["valid_count"], 3)
            self.assertLess(audit["max_project_roundtrip_pixel_error"], 1e-3)
            self.assertEqual(audit["finite_fractions"]["depth"], 1.0)

            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            arrays["depth"] = arrays["depth"].astype(np.float64)
            wrong_dtype = root / "wrong_dtype.npz"
            np.savez(wrong_dtype, **arrays)
            with self.assertRaisesRegex(ValueError, "wrong array dtypes"):
                audit_geometry_npz(wrong_dtype)

            arrays["depth"] = depth
            arrays["unexpected"] = np.asarray(1)
            unexpected = root / "unexpected.npz"
            np.savez(unexpected, **arrays)
            with self.assertRaisesRegex(ValueError, "unexpected arrays"):
                audit_geometry_npz(unexpected)
            del arrays["unexpected"]

            arrays["valid_mask"] = np.zeros_like(valid, dtype=np.uint8)
            wrong_mask = root / "wrong_mask.npz"
            np.savez(wrong_mask, **arrays)
            with self.assertRaisesRegex(
                ValueError, "valid_mask does not exactly match"
            ):
                audit_geometry_npz(wrong_mask)

            arrays["valid_mask"] = valid.astype(np.uint8)
            arrays["world_points"] = world.copy()
            arrays["world_points"][0, 0, 0] += 1.0
            wrong_world = root / "wrong_world.npz"
            np.savez(wrong_world, **arrays)
            with self.assertRaisesRegex(ValueError, "not official-P0 backprojections"):
                audit_geometry_npz(wrong_world)

            arrays["world_points"] = world
            arrays["confidence"] = confidence.copy()
            arrays["confidence"][0, 0] = np.nan
            arrays["valid_mask"] = valid_depth_mask(depth, arrays["confidence"]).astype(
                np.uint8
            )
            nonfinite = root / "nonfinite.npz"
            np.savez(nonfinite, **arrays)
            with self.assertRaisesRegex(ValueError, "nonfinite full-grid outputs"):
                audit_geometry_npz(nonfinite)

    def test_repeatability_metrics_remain_strict_json(self) -> None:
        first = {
            "depth": np.ones((2, 2), dtype=np.float32),
            "confidence": np.ones((2, 2), dtype=np.float32),
            "pred_cam2world": np.eye(4),
        }
        second = dict(first)
        second["depth"] = np.ones((3, 2), dtype=np.float32)
        metrics = repeatability_metrics(first, second)
        self.assertFalse(metrics["all_exact_equal"])
        json.dumps(metrics, allow_nan=False)


class VggtOmegaCliTest(unittest.TestCase):
    def test_dry_run_uses_only_training_manifest_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            image_dir.mkdir()
            graph = complete_graph()
            names = [node["name"] for node in graph["nodes"]]
            for name in names:
                (image_dir / name).touch()
            graph_path = root / "camera_graph.json"
            graph_path.write_text(json.dumps(graph), encoding="utf-8")
            split_path = root / "split.json"
            split_path.write_text(json.dumps(split_for(names)), encoding="utf-8")
            output_dir = root / "output"
            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_vggt_omega.py"),
                "--dry-run",
                "--image-dir",
                str(image_dir),
                "--camera-graph",
                str(graph_path),
                "--split",
                str(split_path),
                "--output-dir",
                str(output_dir),
                "--view-count",
                "25",
            ]
            completed = subprocess.run(
                command, capture_output=True, text=True, check=True
            )
            self.assertIn("dry_run", completed.stdout)
            payload = json.loads(
                (output_dir / "window_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["model_execution"], "not_run")
            self.assertEqual(payload["window_count"], 2)
            self.assertTrue(
                all(len(window["images"]) == 25 for window in payload["windows"])
            )

    def test_cli_requires_exactly_one_mode(self) -> None:
        script = str(ROOT / "scripts" / "run_vggt_omega.py")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            missing = subprocess.run(
                [sys.executable, script, "--output-dir", str(output)],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(missing.returncode, 0)
            conflicting = subprocess.run(
                [
                    sys.executable,
                    script,
                    "--dry-run",
                    "--validate-load",
                    "--output-dir",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(conflicting.returncode, 0)

    def test_cli_rejects_nonempty_attempt_directory_without_overwrite(self) -> None:
        script = str(ROOT / "scripts" / "run_vggt_omega.py")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("preserve", encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, script, "--dry-run", "--output-dir", str(output)],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")
            self.assertFalse((output / "failure.json").exists())

    def test_real_mode_precondition_failure_is_atomic_json(self) -> None:
        script = str(ROOT / "scripts" / "run_vggt_omega.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            completed = subprocess.run(
                [
                    sys.executable,
                    script,
                    "--validate-load",
                    "--output-dir",
                    str(output),
                    "--source-root",
                    str(root / "missing-source"),
                    "--checkpoint",
                    str(root / "missing-checkpoint.pt"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["status"], "failed_preserved")
            self.assertEqual(failure["stage"], "fixed_artifact_gates")

    def test_dry_run_entrypoint_has_no_model_or_network_import(self) -> None:
        source = (ROOT / "scripts" / "run_vggt_omega.py").read_text(encoding="utf-8")
        self.assertNotIn("import torch", source)
        self.assertNotIn("import requests", source)
        self.assertNotIn("import urllib", source)


if __name__ == "__main__":
    unittest.main()
