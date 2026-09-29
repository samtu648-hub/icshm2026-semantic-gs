#!/usr/bin/env python3
"""RGB117: released-track and fresh-image correspondence consistency, without fitting poses."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256, verify_hashes, write_new_json

CODE = ("scripts/audit_quality_camera_poses.py", "src/quality_pose_diagnostics.py", "src/quality_protocol.py", "scripts/prepare_rgb_colmap_scene.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "artifact-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    config_path = ROOT / "experiments/runs/RGB-117-CAMERA-CONSISTENCY-DIAG/config.yaml"
    config = json.loads(config_path.read_text())
    protocol = json.loads((ROOT / config["protocol"]).read_text())
    frozen = json.loads((ROOT / protocol["frozen_manifest"]).read_text())
    frozen_hashes = {a["path"]: a["sha256"] for a in frozen["artifacts"]}
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "outputs/quality"):
        raise ValueError("Diagnostic outputs must be inside gitignored outputs/quality")
    if shutil.disk_usage(ROOT / "outputs/quality").free / 2**30 < config["execution"]["minimum_free_gib"]:
        raise ValueError("Disk reserve not met")
    with (ROOT / "outputs/quality/pose_diagnostic.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(parents=True, exist_ok=False)
        begin = time.monotonic()
        record = {"experiment_id": config["experiment_id"], "parent_experiment": config["parent_experiment"], "seed": config["seed"],
                  "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "config_sha256": sha256(config_path), "code_sha256": {p: sha256(ROOT / p) for p in CODE},
                  "input_hashes": {p: sha256(ROOT / p) for p in (config["dataset_manifest"], config["split"], config["protocol"], protocol["frozen_manifest"])},
                  "training": False, "camera_updates": False, "labels_decoded": False, "developer_rgb_decoded": False,
                  "status": "running_camera_diagnostics"}
        write_new_json(output / "started.json", record)
        try:
            import cv2
            import numpy as np
            from scripts.prepare_rgb_colmap_scene import parse_single_simple_radial_camera, undistort_observations
            from src.quality_pose_diagnostics import (training_access, load_training_records, audit_released_tracks, select_pairs,
                stable_number, mutual_matches, evaluate_pair, fundamental_from_poses, summary)
            cv2.setNumThreads(config["execution"]["opencv_threads"])
            record["runtime"] = {"python": sys.version, "opencv": cv2.__version__, "numpy": np.__version__, "opencv_threads": cv2.getNumThreads()}
            for relative in CODE:
                dest = output / "source" / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, dest)
            shutil.copy2(config_path, output / "config.yaml")
            verify_hashes(args.artifact_root, frozen_hashes)
            train, excluded = training_access(json.loads((ROOT / config["split"]).read_text()))
            if len(train) != 303:
                raise ValueError("Registered RGB117 requires the exact 303 training views")
            write_new_json(output / "access.json", {"train": train, "excluded": excluded,
                "all_camera_metadata_allowed": True, "observation_decode_views": train,
                "original_sfm_used_all_images": "unknown; official metadata was historically produced using released images; not full SfM blind isolation"})
            camera_root = args.data_root / "camera_parameters"
            record["camera_input_hashes"] = {n: sha256(camera_root / n) for n in ("cameras.txt", "images.txt")}
            cid, w, h, f, cx, cy, radial = parse_single_simple_radial_camera(camera_root / "cameras.txt")
            if [w, h] != config["resolution"]:
                raise ValueError("Native camera size changed")
            k = np.array([[f, 0., cx], [0., f, cy], [0., 0., 1.]])
            records = load_training_records(camera_root / "images.txt", train)
            if any(r.camera_id != cid for r in records.values()):
                raise ValueError("Camera IDs do not match intrinsics")
            undistort_observations(list(records.values()), k, radial)
            track_report, track_arrays = audit_released_tracks(records, k, config["seed"])
            write_new_json(output / "official_tracks.json", track_report)
            np.savez_compressed(output / "official_tracks.npz", **track_arrays)
            del track_arrays
            print(json.dumps({"stage": "official_tracks_completed", "summary": {k: track_report[k] for k in ("all_fit", "cross_prediction", "rejection_counts")}}), flush=True)
            for r in records.values():
                r.observations.clear()
            match_config = config["independent_matching"]
            pairs, missing = select_pairs(records, match_config["neighbor_ranks"], match_config["maximum_optical_axis_angle_deg"], match_config["minimum_baseline"])
            write_new_json(output / "pairs.json", {"pairs": pairs, "missing_neighbor_ranks": missing, "selected_before_rgb": True})
            (output / "features").mkdir()
            (output / "matches").mkdir()
            photo_hashes, photo_paths, feature_rows = {}, {}, []
            sift = cv2.SIFT_create(nfeatures=match_config["sift_nfeatures"], contrastThreshold=match_config["sift_contrast_threshold"],
                                   enable_precise_upscale=match_config["sift_precise_upscale"])
            coefficients = np.array([radial, 0., 0., 0., 0.])
            for index, n in enumerate(train):
                candidates = [args.data_root / folder / n for folder in ("images", "unlabeled_Images")]
                candidates = [p for p in candidates if p.is_file()]
                if len(candidates) != 1:
                    raise ValueError(f"Missing or duplicate RGB {n}")
                path = candidates[0]
                photo_paths[n] = path
                photo_hashes[n] = sha256(path)
                gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if gray is None or gray.shape != (h, w):
                    raise ValueError(f"Unexpected original RGB shape {n}")
                points, descriptors = sift.detectAndCompute(gray, None)
                raw_xy = np.array([p.pt for p in points], dtype=np.float64).reshape(-1, 2) + .5
                xy = cv2.undistortPoints(raw_xy[:, None], k, coefficients, P=k).reshape(-1, 2) if len(raw_xy) else raw_xy
                if descriptors is None:
                    descriptors = np.empty((0, 128), np.float32)
                np.savez_compressed(output / "features" / (Path(n).stem + ".npz"), raw_xy=raw_xy, xy=xy, descriptors=descriptors)
                feature_rows.append({"view": n, "features": len(xy)})
                if index % 25 == 0 or index == len(train) - 1:
                    print(json.dumps({"stage": "features", "completed": index + 1, "total": len(train)}), flush=True)
            write_new_json(output / "rgb_input_hashes.json", photo_hashes)
            write_new_json(output / "features.json", feature_rows)
            reports, pooled_official, pooled_image = [], [], []
            for index, pair in enumerate(pairs):
                a, b = pair
                with np.load(output / "features" / (Path(a).stem + ".npz")) as first, np.load(output / "features" / (Path(b).stem + ".npz")) as second:
                    matches = mutual_matches(first["descriptors"], second["descriptors"], match_config, stable_number(config["seed"], *pair))
                    xy_a, xy_b = first["xy"][matches[:, 0]], second["xy"][matches[:, 1]]
                f_official = fundamental_from_poses(records[a].rotation, records[a].translation, records[b].rotation, records[b].translation, k)
                row, arrays = evaluate_pair(xy_a, xy_b, f_official, pair, match_config, config["seed"], (w, h))
                stem = f"{Path(a).stem}-{Path(b).stem}"
                write_new_json(output / "matches" / (stem + ".json"), row)
                np.savez_compressed(output / "matches" / (stem + ".npz"), **arrays)
                reports.append(row)
                if "official_check_error" in arrays:
                    pooled_official.extend(arrays["official_check_error"])
                    pooled_image.extend(arrays["image_f_check_error"])
                if index % 25 == 0 or index == len(pairs) - 1:
                    print(json.dumps({"stage": "image_pairs", "completed": index + 1, "total": len(pairs)}), flush=True)
            successful = [r for r in reports if r["status"] == "completed_correspondence_crossfit_not_pose_truth"]
            summary_rows = {"pairs": len(pairs), "completed_crossfits": len(successful), "insufficient_or_failed_pairs": len(pairs) - len(successful),
                "possible_planar_or_low_parallax_pairs": sum(r["possible_planar_or_low_parallax"] for r in successful),
                "candidate_pairs": [r["pair"] for r in reports if r["pose_inconsistency_candidate"]],
                "pooled_all_check_matches_official_sampson_px": summary(pooled_official),
                "pooled_all_check_matches_image_f_sampson_px": summary(pooled_image),
                "aggregation_limit": "unequal texture/match density; pooled mean is not mean pose accuracy; failed pairs remain in per-pair inventory"}
            write_new_json(output / "independent_matches.json", {"summary": summary_rows, "per_pair": reports})
            verify_hashes(ROOT, record["code_sha256"])
            verify_hashes(ROOT, record["input_hashes"])
            verify_hashes(camera_root, record["camera_input_hashes"])
            if sha256(config_path) != record["config_sha256"]:
                raise ValueError("Diagnostic configuration changed during execution")
            if any(sha256(photo_paths[n]) != photo_hashes[n] for n in train):
                raise ValueError("Training photos changed during execution")
            verify_hashes(args.artifact_root, frozen_hashes)
            record.update(status="completed_camera_consistency_diagnosis_not_pose_truth", frozen_verified=len(frozen_hashes),
                train_rgb_decoded=len(train), excluded_observations_decoded=False,
                official_tracks_summary={k: track_report[k] for k in ("all_fit", "cross_prediction", "rejection_counts")},
                independent_matches_summary=summary_rows, seconds=time.monotonic() - begin)
            record["artifacts"] = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}
            write_new_json(output / "result.json", record)
            print(json.dumps({"status": record["status"], "seconds": record["seconds"], "summary": summary_rows}), flush=True)
        except BaseException as exc:
            record.update(status="failed_camera_diagnosis_not_method_rejection", error=f"{type(exc).__name__}: {exc}", seconds=time.monotonic() - begin)
            write_new_json(output / "failure.json", record)
            raise


if __name__ == "__main__":
    main()
