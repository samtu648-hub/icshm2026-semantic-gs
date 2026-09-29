#!/usr/bin/env python3
"""Independently replay RGB117 evidence and explain leave-out triangulation conditioning."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256, verify_hashes, write_new_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "data-root", "artifact-root", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT / "outputs/quality"):
        raise ValueError("Use gitignored quality output")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    parent = json.loads((args.run / "result.json").read_text())
    config_path = ROOT / "experiments/runs/RGB-117-CAMERA-CONSISTENCY-DIAG/audit_config.yaml"
    config = json.loads(config_path.read_text())
    record = {"experiment_id": config["experiment_id"], "status": "running_evidence_audit", "seed": config["seed"],
              "parent_result_sha256": sha256(args.run / "result.json"), "audit_config_sha256": sha256(config_path),
              "code_sha256": {"scripts/audit_quality_camera_evidence.py": sha256(Path(__file__))},
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "camera_updates": False, "labels_decoded": False, "developer_rgb_decoded": False}
    write_new_json(args.output / "started.json", record)
    try:
        import cv2
        import numpy as np
        from collections import defaultdict
        from scripts.prepare_rgb_colmap_scene import parse_single_simple_radial_camera, undistort_observations
        from src.quality_pose_diagnostics import (load_training_records, mutual_matches, evaluate_pair, stable_number,
            fundamental_from_poses, triangulate, reprojection, summary)
        if parent["experiment_id"] != config["experiment_id"] or parent["status"] != "completed_camera_consistency_diagnosis_not_pose_truth":
            raise ValueError("Unaccepted diagnostic parent")
        verify_hashes(args.run, parent["artifacts"])
        verify_hashes(ROOT, parent["code_sha256"])
        verify_hashes(ROOT, parent["input_hashes"])
        verify_hashes(args.data_root / "camera_parameters", parent["camera_input_hashes"])
        original = json.loads((args.run / "config.yaml").read_text())
        if sha256(args.run / "config.yaml") != parent["config_sha256"]:
            raise ValueError("Saved configuration mismatch")
        protocol = json.loads((ROOT / original["protocol"]).read_text())
        frozen = json.loads((ROOT / protocol["frozen_manifest"]).read_text())
        frozen_hashes = {a["path"]: a["sha256"] for a in frozen["artifacts"]}
        verify_hashes(args.artifact_root, frozen_hashes)
        access = json.loads((args.run / "access.json").read_text())
        train = access["train"]
        pairs = json.loads((args.run / "pairs.json").read_text())["pairs"]
        rows = json.loads((args.run / "independent_matches.json").read_text())["per_pair"]
        _, w, h, f, cx, cy, radial = parse_single_simple_radial_camera(args.data_root / "camera_parameters/cameras.txt")
        k = np.array([[f, 0., cx], [0., f, cy], [0., 0., 1.]])
        coeff = np.array([radial, 0., 0., 0., 0.])
        cv2.setNumThreads(original["execution"]["opencv_threads"])
        mc = original["independent_matching"]
        feature_views = [train[i] for i in np.linspace(0, len(train) - 1, config["feature_replays"], dtype=int)]
        rgb_hashes = json.loads((args.run / "rgb_input_hashes.json").read_text())
        sift = cv2.SIFT_create(nfeatures=mc["sift_nfeatures"], contrastThreshold=mc["sift_contrast_threshold"], enable_precise_upscale=mc["sift_precise_upscale"])
        for n in feature_views:
            paths = [args.data_root / directory / n for directory in ("images", "unlabeled_Images")]
            paths = [p for p in paths if p.is_file()]
            if len(paths) != 1 or sha256(paths[0]) != rgb_hashes[n]:
                raise ValueError("Changed feature-replay RGB")
            points, desc = sift.detectAndCompute(cv2.imread(str(paths[0]), cv2.IMREAD_GRAYSCALE), None)
            raw = np.asarray([p.pt for p in points], np.float64).reshape(-1, 2) + .5
            xy = cv2.undistortPoints(raw[:, None], k, coeff, P=k).reshape(-1, 2)
            with np.load(args.run / "features" / (Path(n).stem + ".npz")) as cached:
                for key, value in (("raw_xy", raw), ("xy", xy), ("descriptors", desc)):
                    np.testing.assert_array_equal(value, cached[key])
        print(json.dumps({"stage": "fresh_feature_replay_exact", "views": feature_views}), flush=True)
        records = load_training_records(args.data_root / "camera_parameters/images.txt", train)
        undistort_observations(list(records.values()), k, radial)
        sampled_pairs = [pairs[i] for i in np.linspace(0, len(pairs) - 1, config["pair_replays"], dtype=int)]
        for pair in sampled_pairs:
            a, b = pair
            stem = f"{Path(a).stem}-{Path(b).stem}"
            with np.load(args.run / "features" / (Path(a).stem + ".npz")) as first, np.load(args.run / "features" / (Path(b).stem + ".npz")) as second:
                matches = mutual_matches(first["descriptors"], second["descriptors"], mc, stable_number(original["seed"], *pair))
                official = fundamental_from_poses(records[a].rotation, records[a].translation, records[b].rotation, records[b].translation, k)
                row, arrays = evaluate_pair(first["xy"][matches[:, 0]], second["xy"][matches[:, 1]], official, pair, mc, original["seed"], (w, h))
            if row != json.loads((args.run / "matches" / (stem + ".json")).read_text()):
                raise ValueError(f"Pair summary replay differs: {pair}")
            with np.load(args.run / "matches" / (stem + ".npz")) as cached:
                if set(arrays) != set(cached.files):
                    raise ValueError("Pair artifact fields differ")
                for key, value in arrays.items():
                    np.testing.assert_array_equal(value, cached[key])
        print(json.dumps({"stage": "fresh_pair_replay_exact", "pairs": sampled_pairs}), flush=True)
        # Different arithmetic formulation from the production vectorized evaluator.
        def independent_sampson(matrix, a, b):
            x, y, u, v = a[:, 0], a[:, 1], b[:, 0], b[:, 1]
            l0 = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
            l1 = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
            l2 = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
            m0 = matrix[0, 0] * u + matrix[1, 0] * v + matrix[2, 0]
            m1 = matrix[0, 1] * u + matrix[1, 1] * v + matrix[2, 1]
            return abs(u * l0 + v * l1 + l2) / np.sqrt(l0*l0 + l1*l1 + m0*m0 + m1*m1)
        supported_views, ambiguous_views = set(), set()
        per_view = defaultdict(list)
        checked = 0
        for row in rows:
            if row["status"] != "completed_correspondence_crossfit_not_pose_truth":
                continue
            stem = "-".join(Path(n).stem for n in row["pair"])
            with np.load(args.run / "matches" / (stem + ".npz")) as data:
                a, b = data["a"][data["check"]], data["b"][data["check"]]
                for key, matrix in (("official", "official_f"), ("image_f", "image_f")):
                    actual = independent_sampson(data[matrix], a, b)
                    np.testing.assert_allclose(actual, data[key + "_check_error"], rtol=1e-9, atol=1e-8)
                checked += len(a)
            for n in row["pair"]:
                per_view[n].append({"pair": row["pair"], "official_median_px": row["official_sampson_px"]["median"],
                                   "image_f_median_px": row["image_f_sampson_px"]["median"], "check_matches": row["check_matches"],
                                   "possible_planar_or_low_parallax": row["possible_planar_or_low_parallax"]})
            if row["image_f_support"] >= .5 and min(row["occupied_check_cells"]) >= mc["minimum_check_occupied_cells"]:
                (ambiguous_views if row["possible_planar_or_low_parallax"] else supported_views).update(row["pair"])
        write_new_json(args.output / "per_view_pairs.json", dict(per_view))
        # Posthoc all-track conditioning, never camera adjustment or a scoring filter.
        tracks = defaultdict(list)
        for index, n in enumerate(train):
            for obs in records[n].observations:
                if obs.point_id >= 0:
                    tracks[obs.point_id].append((index, obs.undistorted_xy))
        with np.load(args.run / "official_tracks.npz") as data:
            cross = data["cross_prediction"].copy()
        values = []
        for counter, row in enumerate(cross):
            pid = int(row[0])
            track = tracks[pid]
            held = stable_number(original["seed"], pid) % len(track)
            xy = np.asarray([p for _, p in track])
            poses = [(records[train[i]].rotation, records[train[i]].translation) for i, _ in track]
            fit = [j for j in range(len(track)) if j != held]
            point = triangulate([poses[j] for j in fit], xy[fit], k)
            residual = reprojection(point, poses, xy, k)[held]
            np.testing.assert_allclose(residual, row[4:], atol=1e-8, rtol=1e-9)
            centres = np.asarray([-poses[j][0].T @ poses[j][1] for j in fit])
            rays = point - centres
            rays /= np.linalg.norm(rays, axis=1, keepdims=True)
            max_angle = float(np.degrees(np.arccos(np.clip((rays @ rays.T).min(), -1, 1))))
            values.append((pid, row[1], np.linalg.norm(residual), max_angle, len(fit)))
            if counter % 15000 == 0:
                print(json.dumps({"stage": "all_track_conditioning", "completed": counter, "total": len(cross)}), flush=True)
        values = np.asarray(values)
        strata = []
        edges = [0., *config["source_ray_angle_bins_deg"], float("inf")]
        for low, high in zip(edges[:-1], edges[1:]):
            selected = (values[:, 3] >= low) & (values[:, 3] < high)
            errors = values[selected, 2]
            strata.append({"source_max_ray_angle_deg": [low, None if np.isinf(high) else high], "error_px": summary(errors),
                           "above_5px_count": int((errors > config["large_cross_prediction_error_px"]).sum())})
        np.savez_compressed(args.output / "track_conditioning.npz", point_id=values[:, 0].astype(np.int64), view_index=values[:, 1].astype(np.int64),
                            error_px=values[:, 2], source_max_ray_angle_deg=values[:, 3], source_count=values[:, 4].astype(np.int64))
        conditioning = {"posthoc_not_causal": True, "all_successful_cross_predictions_reproduced": len(values), "strata": strata,
                        "source_count_two": int((values[:, 4] == 2).sum()),
                        "above_5px_with_source_angle_below_point1_deg": int(((values[:, 2] > 5) & (values[:, 3] < .1)).sum())}
        write_new_json(args.output / "conditioning.json", conditioning)
        verify_hashes(ROOT, parent["code_sha256"])
        verify_hashes(ROOT, parent["input_hashes"])
        verify_hashes(ROOT, record["code_sha256"])
        verify_hashes(args.data_root / "camera_parameters", parent["camera_input_hashes"])
        verify_hashes(args.artifact_root, frozen_hashes)
        if sha256(config_path) != record["audit_config_sha256"]:
            raise ValueError("Audit config changed")
        shutil.copy2(Path(__file__), args.output / "audit_source.py")
        shutil.copy2(config_path, args.output / "audit_config.yaml")
        record.update(status="completed_independent_camera_evidence_audit", feature_replay_views=feature_views, pair_replays=sampled_pairs,
                      independent_formula_check_matches=checked, pair_covered_training_views=len(per_view),
                      well_supported_nonplanar_training_views=len(supported_views), training_views_without_supported_nonplanar_pair=sorted(set(train)-supported_views),
                      conditioning=conditioning, frozen_verified=len(frozen_hashes), seconds=time.monotonic() - started)
        record["artifacts"] = {str(p.relative_to(args.output)): sha256(p) for p in sorted(args.output.rglob("*")) if p.is_file()}
        write_new_json(args.output / "result.json", record)
        print(json.dumps({k:v for k,v in record.items() if k in ("status", "independent_formula_check_matches", "pair_covered_training_views", "well_supported_nonplanar_training_views", "conditioning", "seconds")}), flush=True)
    except BaseException as exc:
        record.update(status="failed_camera_evidence_audit", error=f"{type(exc).__name__}: {exc}", seconds=time.monotonic() - started)
        write_new_json(args.output / "failure.json", record)
        raise


if __name__ == "__main__":
    main()
