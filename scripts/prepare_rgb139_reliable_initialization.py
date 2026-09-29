#!/usr/bin/env python3
"""Select a fixed-budget, geometry-reliable subset of RGB137 all400 tracks."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_all400_initialization import REGIONS, write_ply
from scripts.prepare_rgb_colmap_scene import (
    parse_colmap_images,
    parse_single_simple_radial_camera,
    projection_matrix,
    undistort_observations,
)
from src.quality_protocol import sha256, write_new_json


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--source-scene", type=Path, required=True)
    p.add_argument("--baseline-scene", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--target-points", type=int, default=72_292)
    return p.parse_args()


def read_points(path):
    points = {}
    for raw in path.read_text().splitlines():
        row = raw.split()
        if not row or row[0].startswith("#"):
            continue
        point_id = int(row[0])
        points[point_id] = {
            "point_id": point_id,
            "xyz": np.asarray([float(x) for x in row[1:4]], dtype=np.float64),
            "rgb": tuple(int(x) for x in row[4:7]),
            "mean_reprojection_error_px": float(row[7]),
        }
    return points


def dlt_matrix(track, camera_matrix):
    rows = []
    matrices = []
    for record, observation in track:
        matrix = projection_matrix(record, camera_matrix)
        u, v = observation.undistorted_xy
        rows.extend((u * matrix[2] - matrix[0], v * matrix[2] - matrix[1]))
        matrices.append(matrix)
    return np.stack(rows), matrices


def triangulate_subset(track, camera_matrix):
    if len(track) < 2:
        return None
    matrix, projections = dlt_matrix(track, camera_matrix)
    _, _, right = np.linalg.svd(matrix, full_matrices=False)
    homogeneous = right[-1]
    if abs(homogeneous[3]) < 1e-10:
        return None
    xyz = homogeneous[:3] / homogeneous[3]
    if not np.isfinite(xyz).all():
        return None
    if any((projection @ np.append(xyz, 1.0))[2] <= 1e-8 for projection in projections):
        return None
    return xyz


def percentile_score(values, higher_is_better=True, missing=0.5):
    values = np.asarray(values, dtype=np.float64)
    score = np.full(len(values), missing, dtype=np.float64)
    valid = np.isfinite(values)
    if not valid.any():
        return score
    indices = np.flatnonzero(valid)
    order = np.argsort(values[valid], kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(len(order), dtype=np.float64)
    if len(order) > 1:
        ranks /= len(order) - 1
    else:
        ranks[:] = 0.5
    if not higher_is_better:
        ranks = 1.0 - ranks
    score[indices] = ranks
    return score


def quantiles(values):
    return {str(q): float(np.quantile(values, q)) for q in (0, .1, .25, .5, .75, .9, 1)}


def allocate(strata, target):
    total = sum(len(indices) for indices in strata.values())
    ideals = {key: len(indices) * target / total for key, indices in strata.items()}
    counts = {key: math.floor(value) for key, value in ideals.items()}
    remainder = target - sum(counts.values())
    order = sorted(strata, key=lambda key: (-(ideals[key] - counts[key]), key))
    for key in order[:remainder]:
        counts[key] += 1
    if sum(counts.values()) != target or any(counts[key] > len(strata[key]) for key in strata):
        raise RuntimeError("Invalid proportional stratum allocation")
    return counts


def main():
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    source_scene = args.source_scene.resolve()
    baseline_scene = args.baseline_scene.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    config = ROOT / "experiments/runs/RGB-139-STANDARD-RELIABLE-INIT400/config.yaml"
    record = {
        "schema_version": 1,
        "experiment_id": "RGB-139-STANDARD-RELIABLE-INIT400",
        "status": "starting",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config_sha256": sha256(config),
        "code_sha256": sha256(Path(__file__)),
        "source_scene": str(source_scene),
        "source_ply_sha256": sha256(source_scene / "sparse/0/points3D.ply"),
        "baseline_ply_sha256": sha256(baseline_scene / "sparse/0/points3D.ply"),
        "target_points": args.target_points,
    }
    try:
        source_points = read_points(source_scene / "sparse/0/points3D.txt")
        baseline_points = read_points(baseline_scene / "sparse/0/points3D.txt")
        if len(source_points) != 85_985 or len(baseline_points) != 72_292 or args.target_points != 72_292:
            raise ValueError("Registered source/baseline/target point counts changed")
        records = parse_colmap_images(dataset_root / "camera_parameters/images.txt")
        _, width, height, focal, cx, cy, radial = parse_single_simple_radial_camera(
            dataset_root / "camera_parameters/cameras.txt")
        camera_matrix = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1.]], dtype=np.float64)
        undistort_observations(records, camera_matrix, radial)
        tracks = defaultdict(list)
        for record_image in records:
            for observation in record_image.observations:
                if observation.point_id in source_points:
                    tracks[observation.point_id].append((record_image, observation))
        if set(tracks) != set(source_points):
            raise RuntimeError("All source points must retain their released observations")

        features = []
        for index, point_id in enumerate(sorted(source_points)):
            point = source_points[point_id]
            track = sorted(tracks[point_id], key=lambda item: (item[0].image_id, item[1].point_index))
            xyz = point["xyz"]
            centers = np.asarray([-image.rotation.T @ image.translation for image, _ in track])
            rays = xyz - centers
            rays /= np.linalg.norm(rays, axis=1, keepdims=True)
            dots = np.clip(rays @ rays.T, -1, 1)
            angle = float(np.degrees(np.arccos(dots.min())))
            matrix, _ = dlt_matrix(track, camera_matrix)
            singular = np.linalg.svd(matrix, compute_uv=False, full_matrices=False)
            condition_gap = float(np.log10(max(singular[-2] / max(singular[-1], 1e-15), 1.0)))
            split_stability = float("nan")
            if len(track) >= 4:
                first = triangulate_subset(track[::2], camera_matrix)
                second = triangulate_subset(track[1::2], camera_matrix)
                if first is not None and second is not None:
                    scale = float(np.median(np.linalg.norm(xyz - centers, axis=1)))
                    if scale > 1e-12:
                        split_stability = float(np.linalg.norm(first - second) / scale)
            features.append({
                "point_id": point_id,
                "track_length": len(track),
                "maximum_triangulation_angle_deg": angle,
                "mean_reprojection_error_px": point["mean_reprojection_error_px"],
                "dlt_log10_nullspace_gap": condition_gap,
                "split_half_relative_3d_difference": split_stability,
            })
            if index and index % 10_000 == 0:
                print(json.dumps({"scored": index, "total": len(source_points)}), flush=True)

        lengths = np.asarray([row["track_length"] for row in features], dtype=np.float64)
        angles = np.asarray([row["maximum_triangulation_angle_deg"] for row in features], dtype=np.float64)
        length_edges = np.quantile(lengths, (0, .25, .5, .75, 1))
        angle_edges = np.quantile(angles, (0, .25, .5, .75, 1))
        strata = defaultdict(list)
        for index, row in enumerate(features):
            length_bin = int(np.searchsorted(length_edges[1:-1], row["track_length"], side="right"))
            angle_bin = int(np.searchsorted(angle_edges[1:-1], row["maximum_triangulation_angle_deg"], side="right"))
            key = f"length_q{length_bin}_angle_q{angle_bin}"
            row["stratum"] = key
            strata[key].append(index)
        allocations = allocate(strata, args.target_points)
        selected_indices = set()
        stratum_summary = {}
        for key, indices in sorted(strata.items()):
            errors = [features[index]["mean_reprojection_error_px"] for index in indices]
            conditions = [features[index]["dlt_log10_nullspace_gap"] for index in indices]
            stability = [features[index]["split_half_relative_3d_difference"] for index in indices]
            error_score = percentile_score(errors, higher_is_better=False)
            condition_score = percentile_score(conditions, higher_is_better=True)
            stability_score = percentile_score(stability, higher_is_better=False, missing=.5)
            for local, index in enumerate(indices):
                features[index]["reliability_score"] = float(
                    (error_score[local] + condition_score[local] + stability_score[local]) / 3)
            ranked = sorted(indices, key=lambda index: (-features[index]["reliability_score"],
                                                        features[index]["point_id"]))
            kept = ranked[:allocations[key]]
            selected_indices.update(kept)
            stratum_summary[key] = {"source": len(indices), "selected": len(kept),
                                    "retained_fraction": len(kept) / len(indices)}
        if len(selected_indices) != args.target_points:
            raise RuntimeError("Selected point count differs from fixed target")
        selected_ids = {features[index]["point_id"] for index in selected_indices}
        selected_points = [source_points[point_id] for point_id in sorted(selected_ids)]

        # Preserve the source scene layout while creating a new, append-only
        # provenance record for this selection.  Source manifests describe
        # RGB137 and must not be inherited by RGB139.
        shutil.copytree(
            source_scene,
            output,
            symlinks=True,
            ignore=shutil.ignore_patterns("initialization_manifest.json", "selection_records.jsonl"),
        )
        sparse = output / "sparse/0"
        (sparse / "points3D.txt").unlink()
        (sparse / "points3D.ply").unlink()
        with (sparse / "points3D.txt").open("x") as handle:
            handle.write("# RGB139 reliability-selected subset of RGB137 all400 DLT tracks.\n")
            for point in selected_points:
                x, y, z = point["xyz"]
                red, green, blue = point["rgb"]
                handle.write(f"{point['point_id']} {x:.17g} {y:.17g} {z:.17g} "
                             f"{red} {green} {blue} {point['mean_reprojection_error_px']:.17g}\n")
        write_ply(sparse / "points3D.ply", selected_points)

        selected_features = [features[index] for index in sorted(selected_indices)]
        with (output / "selection_records.jsonl").open("x") as handle:
            for row in features:
                saved = dict(row)
                if not np.isfinite(saved["split_half_relative_3d_difference"]):
                    saved["split_half_relative_3d_difference"] = None
                saved["selected"] = saved["point_id"] in selected_ids
                handle.write(json.dumps(saved, sort_keys=True, allow_nan=False) + "\n")

        region_counts = {}
        for key, (view, boxes) in REGIONS.items():
            count = 0
            for point_id in selected_ids:
                xyz = source_points[point_id]["xyz"]
                for image, _ in tracks[point_id]:
                    if image.name != view:
                        continue
                    projected = projection_matrix(image, camera_matrix) @ np.append(xyz, 1.)
                    u, v = projected[:2] / projected[2]
                    if any(x0 <= u < x1 and y0 <= v < y1 for x0, y0, x1, y1 in boxes):
                        count += 1
                    break
            region_counts[key] = count

        def distribution(rows):
            return {
                "track_length": quantiles([row["track_length"] for row in rows]),
                "maximum_triangulation_angle_deg": quantiles(
                    [row["maximum_triangulation_angle_deg"] for row in rows]),
                "mean_reprojection_error_px": quantiles(
                    [row["mean_reprojection_error_px"] for row in rows]),
                "dlt_log10_nullspace_gap": quantiles(
                    [row["dlt_log10_nullspace_gap"] for row in rows]),
                "split_half_relative_3d_difference": quantiles(
                    [row["split_half_relative_3d_difference"] for row in rows
                     if np.isfinite(row["split_half_relative_3d_difference"])]),
            }

        baseline_ids = set(baseline_points)
        manifest = {
            "schema_version": 1,
            "experiment_id": "RGB-139-STANDARD-RELIABLE-INIT400",
            "status": "completed_initialization",
            "selection_policy": {
                "source": "RGB137 all400 accepted DLT tracks",
                "source_points": len(source_points),
                "target_points": len(selected_points),
                "stratification": "global empirical quartiles of track length x maximum triangulation angle",
                "length_edges": length_edges.tolist(),
                "angle_edges_deg": angle_edges.tolist(),
                "allocation": "same retained fraction per stratum with largest-remainder exact budget",
                "ranking": "equal mean of within-stratum percentile scores: lower mean reprojection error, higher DLT nullspace gap, lower split-half relative 3D difference; missing split-half gets neutral 0.5",
                "tie_break": "ascending released POINT3D_ID",
                "labels_semantics_or_depth": "none",
                "new_absolute_filter": "none",
            },
            "strata": stratum_summary,
            "source_distribution": distribution(features),
            "selected_distribution": distribution(selected_features),
            "selected_overlap": {
                "rgb130_point_ids": len(selected_ids & baseline_ids),
                "new_vs_rgb130_point_ids": len(selected_ids - baseline_ids),
                "rgb130_ids_not_selected": len(baseline_ids - selected_ids),
            },
            "region_projected_point_counts": region_counts,
            "camera_hashes": {name: sha256(sparse / name) for name in ("cameras.txt", "images.txt")},
            "source_camera_hashes": {name: sha256(source_scene / "sparse/0" / name)
                                     for name in ("cameras.txt", "images.txt")},
            "selected_points3D_txt_sha256": sha256(sparse / "points3D.txt"),
            "selected_ply_sha256": sha256(sparse / "points3D.ply"),
            "selection_records_sha256": sha256(output / "selection_records.jsonl"),
        }
        if manifest["camera_hashes"] != manifest["source_camera_hashes"]:
            raise RuntimeError("Camera files changed during point selection")
        write_new_json(output / "initialization_manifest.json", manifest)
        record.update(status="completed_initialization", manifest=manifest,
                      output=str(output), output_manifest_sha256=sha256(output / "initialization_manifest.json"))
        print(json.dumps(manifest, ensure_ascii=False), flush=True)
    except BaseException as exc:
        record.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["seconds"] = time.monotonic() - started
        write_new_json(output.parent / f"{output.name}-preparation-result.json", record)


if __name__ == "__main__":
    main()
