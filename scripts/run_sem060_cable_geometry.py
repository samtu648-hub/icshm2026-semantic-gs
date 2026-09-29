#!/usr/bin/env python3
"""Run the training-only SEM-060 Cable3D fit and independent geometry check."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geometry_prior.camera_io import read_cameras_txt, read_images_txt  # noqa: E402
from semantic.cable_geometry import (  # noqa: E402
    CableCurve,
    centerline_metrics,
    extract_centerline,
    fit_piecewise_lines,
    nearest_skeleton,
    rasterize_projected_curves,
    voxel_representatives,
    write_case_overlay,
)
from semantic.cable_triangulation import (  # noqa: E402
    CableObservation,
    fundamental_matrix,
    pinhole_matrix,
    reproject,
    robust_triangulate,
    symmetric_epipolar_distance,
    triangulate_dlt,
    triangulation_angle_deg,
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _resolve(value: str) -> Path:
    expanded = os.path.expandvars(value)
    if "$" in expanded:
        raise ValueError(f"Unresolved environment variable in path: {value}")
    return Path(expanded)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _git_evidence() -> dict:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    diff = subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)
    status = subprocess.check_output(["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=ROOT)
    return {
        "head": head,
        "tracked_diff_sha256": _hash_bytes(diff),
        "porcelain_sha256": _hash_bytes(status),
        "worktree_clean": not bool(status),
    }


def _manifest(output_root: Path) -> dict:
    records = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            records.append({"path": str(path.relative_to(output_root)), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    payload = {"schema_version": 1, "files": records, "total_bytes": sum(record["bytes"] for record in records)}
    _write_json(output_root / "artifact_manifest.json", payload)
    return payload


def _camera_setup(data_root: Path):
    camera_dir = data_root / "camera_parameters"
    cameras = read_cameras_txt(camera_dir / "cameras.txt")
    if len(cameras) != 1:
        raise ValueError("SEM-060 expects the released single-camera calibration")
    camera = next(iter(cameras.values()))
    if camera.model != "SIMPLE_RADIAL" or len(camera.params) != 4:
        raise ValueError("SEM-060 expects the released SIMPLE_RADIAL calibration")
    focal, cx, cy, radial = camera.params
    intrinsic = pinhole_matrix(focal, cx, cy)
    distortion = np.array([radial, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    images = {image.name: image for image in read_images_txt(camera_dir / "images.txt").values()}
    return camera, intrinsic, distortion, images


def _verify_label(label_path: Path, expected_hash: str) -> None:
    if not label_path.is_file() or _sha256(label_path) != expected_hash:
        raise ValueError(f"SEM-001 training label hash mismatch: {label_path.name}")


def _extract_many(names: Iterable[str], label_root: Path, manifest_records: dict, config: dict) -> tuple[dict, list[dict]]:
    centerlines, rows = {}, []
    for name in names:
        record = manifest_records[name]
        path = label_root / name
        _verify_label(path, record["label_png_sha256"])
        centerline = extract_centerline(path, int(config["class_id"]), config)
        centerlines[name] = centerline
        rows.append(
            {
                "image": name,
                "components": centerline.component_count,
                "skeleton_components": centerline.skeleton_component_count,
                "mask_pixels": int(np.count_nonzero(centerline.mask)),
                "skeleton_pixels": int(np.count_nonzero(centerline.skeleton)),
                "samples": len(centerline.sample_pixels_xy),
                "endpoints": len(centerline.endpoints_xy),
                "branches": len(centerline.branches_xy),
                "median_width_px": float(np.median(centerline.sample_widths_px)) if len(centerline.sample_widths_px) else 0.0,
            }
        )
    return centerlines, rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _valid_graph_edges(graph: dict, names: set[str], config: dict) -> tuple[dict[frozenset[str], dict], list[dict]]:
    valid = []
    for edge in graph["edges"]:
        if edge["source"] not in names or edge["target"] not in names:
            continue
        if int(edge["shared_point3d"]) < int(config["minimum_camera_graph_shared_points"]):
            continue
        baseline = float(edge["center_distance"])
        if baseline < float(config["minimum_camera_baseline"]) or baseline > float(config["maximum_camera_baseline"]):
            continue
        if float(edge["optical_axis_angle_deg"]) > float(config["maximum_optical_axis_angle_deg"]):
            continue
        valid.append(edge)
    lookup = {frozenset((edge["source"], edge["target"])): edge for edge in valid}
    by_name: dict[str, list[dict]] = defaultdict(list)
    for edge in valid:
        by_name[edge["source"]].append(edge)
        by_name[edge["target"]].append(edge)
    selected_keys: set[frozenset[str]] = set()
    maximum = int(config["maximum_pair_neighbors_per_camera"])
    for name, edges in by_name.items():
        edges.sort(key=lambda edge: (-int(edge["shared_point3d"]), str(edge["source"]), str(edge["target"])))
        for edge in edges[:maximum]:
            selected_keys.add(frozenset((edge["source"], edge["target"])))
    selected = [lookup[key] for key in sorted(selected_keys, key=lambda key: tuple(sorted(key)))]
    return lookup, selected


def _undistort_points(raw_xy: np.ndarray, intrinsic: np.ndarray, distortion: np.ndarray) -> np.ndarray:
    if not len(raw_xy):
        return np.empty((0, 2), dtype=np.float64)
    return cv2.undistortPoints(np.asarray(raw_xy, dtype=np.float64).reshape(-1, 1, 2), intrinsic, distortion, P=intrinsic).reshape(-1, 2)


def _track_observations(
    names: list[str], centerlines: dict, images: dict, intrinsic: np.ndarray, distortion: np.ndarray, maximum_distance: float
) -> tuple[dict[int, list[CableObservation]], dict]:
    tracks: dict[int, list[CableObservation]] = defaultdict(list)
    selected = 0
    all_tracked = 0
    for name in names:
        image = images[name]
        distance, indices = nearest_skeleton(centerlines[name].skeleton)
        width_map = 2.0 * cv2.distanceTransform(centerlines[name].mask.astype(np.uint8), cv2.DIST_L2, 5)
        raw, point_ids = [], []
        for x, y, point_id in image.points2d:
            if point_id >= 0:
                raw.append((x, y))
                point_ids.append(point_id)
        all_tracked += len(raw)
        converted = _undistort_points(np.asarray(raw), intrinsic, distortion)
        for pixel, point_id in zip(converted, point_ids):
            x, y = np.rint(pixel).astype(int)
            if y < 0 or y >= distance.shape[0] or x < 0 or x >= distance.shape[1] or distance[y, x] > maximum_distance:
                continue
            nearest_y, nearest_x = int(indices[0, y, x]), int(indices[1, y, x])
            width = float(width_map[nearest_y, nearest_x])
            tracks[int(point_id)].append(
                CableObservation(name, np.array([nearest_x, nearest_y], dtype=np.float64), width, image, "released_track")
            )
            selected += 1
    return tracks, {"tracked_observations": all_tracked, "near_centerline_track_observations": selected, "near_centerline_track_ids": len(tracks)}


def _track_components(
    observations: list[CableObservation], edge_lookup: dict, intrinsic: np.ndarray, epipolar_threshold: float
) -> list[list[CableObservation]]:
    parent = list(range(len(observations)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for left in range(len(observations)):
        for right in range(left + 1, len(observations)):
            a, b = observations[left], observations[right]
            if frozenset((a.image_name, b.image_name)) not in edge_lookup:
                continue
            fundamental = fundamental_matrix(a.image, b.image, intrinsic)
            distance = symmetric_epipolar_distance(a.pixel_xy[None], b.pixel_xy[None], fundamental)[0]
            if distance <= epipolar_threshold:
                union(left, right)
    groups: dict[int, list[CableObservation]] = defaultdict(list)
    for index, observation in enumerate(observations):
        groups[find(index)].append(observation)
    return list(groups.values())


def _match_with_fundamental(left, right, fundamental: np.ndarray, threshold: float, width_ratio: float) -> list[tuple[int, int, float]]:
    left_h = np.column_stack([left.sample_pixels_xy, np.ones(len(left.sample_pixels_xy))])
    right_h = np.column_stack([right.sample_pixels_xy, np.ones(len(right.sample_pixels_xy))])
    right_lines = left_h @ fundamental.T
    left_lines = right_h @ fundamental
    numerator = np.abs(right_h @ right_lines.T)
    distance = 0.5 * numerator * (
        1.0 / np.maximum(np.linalg.norm(right_lines[:, :2], axis=1)[None], np.finfo(float).eps)
        + 1.0 / np.maximum(np.linalg.norm(left_lines[:, :2], axis=1)[:, None], np.finfo(float).eps)
    )
    nearest_right = np.argmin(distance, axis=0)
    nearest_left = np.argmin(distance, axis=1)
    matches = []
    for left_index, right_index in enumerate(nearest_right):
        right_index = int(right_index)
        if int(nearest_left[right_index]) != left_index:
            continue
        value = float(distance[right_index, left_index])
        if value > threshold:
            continue
        a, b = float(left.sample_widths_px[left_index]), float(right.sample_widths_px[right_index])
        ratio = max(a, b) / max(min(a, b), np.finfo(float).eps)
        if ratio <= width_ratio:
            matches.append((left_index, right_index, value))
    return matches


def _project_many(points: np.ndarray, image, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera = points @ image.rotation_world_to_camera.T + image.tvec
    depth = camera[:, 2]
    pixels = camera @ intrinsic.T
    pixels = pixels[:, :2] / pixels[:, 2:3]
    return pixels, depth


def _limit_observations(observations: list[CableObservation], maximum: int) -> list[CableObservation]:
    unique = {observation.image_name: observation for observation in observations}
    pool = [unique[name] for name in sorted(unique)]
    if len(pool) <= maximum:
        return pool
    selected = [pool.pop(0)]
    while pool and len(selected) < maximum:
        best = max(
            range(len(pool)),
            key=lambda index: (
                min(float(np.linalg.norm(pool[index].image.center - item.image.center)) for item in selected),
                pool[index].image_name,
            ),
        )
        selected.append(pool.pop(best))
    return selected


def run_fit(config_path: Path, config: dict, split: dict, data_root: Path, label_root: Path, output_root: Path) -> dict:
    start = time.perf_counter()
    camera, intrinsic, distortion, images = _camera_setup(data_root)
    fit_names = list(split["geometry_fit"])
    prohibited = set(split["geometry_check"]) | set(split["local_guard_band"]) | set(split["fixed_validation_read_only_names"]) | set(split["unlabeled_never_used"])
    if set(fit_names) & prohibited:
        raise ValueError("Fit inputs overlap a prohibited cohort")
    manifest = _read_json(ROOT / config["data"]["label_manifest"])
    records = {record["image"]: record for record in manifest["records"]}
    centerlines, rows = _extract_many(fit_names, label_root, records, config["centerline"])
    _write_csv(output_root / "fit" / "centerlines.csv", rows)
    graph = _read_json(ROOT / config["data"]["camera_graph"])
    edge_lookup, pair_edges = _valid_graph_edges(graph, set(fit_names), config["matching"])
    tracks, track_summary = _track_observations(
        fit_names,
        centerlines,
        images,
        intrinsic,
        distortion,
        float(config["matching"]["track_to_centerline_distance_px"]),
    )
    seeds: list[dict] = []
    track_groups_considered = 0
    for track_id in sorted(tracks):
        if len({observation.image_name for observation in tracks[track_id]}) < int(config["triangulation"]["minimum_independent_support_views"]):
            continue
        for component in _track_components(
            tracks[track_id], edge_lookup, intrinsic, float(config["matching"]["epipolar_distance_px"])
        ):
            if len(component) < int(config["triangulation"]["minimum_independent_support_views"]):
                continue
            track_groups_considered += 1
            candidate = robust_triangulate(
                component,
                intrinsic,
                float(config["matching"]["minimum_camera_baseline"]),
                float(config["matching"]["maximum_camera_baseline"]),
                float(config["triangulation"]["minimum_triangulation_angle_deg"]),
                float(config["triangulation"]["ransac_reprojection_threshold_px"]),
                int(config["triangulation"]["minimum_independent_support_views"]),
            )
            if candidate is not None:
                seeds.append({"xyz": candidate.xyz, "observations": list(candidate.observations), "score": candidate.confidence + 10.0, "source": f"track:{track_id}"})
    track_seed_count = len(seeds)
    if config["matching"]["trackless_epipolar_matching"]:
        for edge in pair_edges:
            left_name, right_name = edge["source"], edge["target"]
            left, right = centerlines[left_name], centerlines[right_name]
            fundamental = fundamental_matrix(images[left_name], images[right_name], intrinsic)
            matches = _match_with_fundamental(
                left,
                right,
                fundamental,
                float(config["matching"]["epipolar_distance_px"]),
                float(config["matching"]["maximum_width_ratio"]),
            )
            for left_index, right_index, epipolar_error in matches:
                observations = [
                    CableObservation(left_name, left.sample_pixels_xy[left_index], float(left.sample_widths_px[left_index]), images[left_name], "epipolar"),
                    CableObservation(right_name, right.sample_pixels_xy[right_index], float(right.sample_widths_px[right_index]), images[right_name], "epipolar"),
                ]
                try:
                    point = triangulate_dlt(observations, intrinsic)
                except ValueError:
                    continue
                reprojections = [reproject(point, observation.image, intrinsic) for observation in observations]
                if any(depth <= 0 for _, depth in reprojections):
                    continue
                errors = [float(np.linalg.norm(pixel - observation.pixel_xy)) for observation, (pixel, _) in zip(observations, reprojections)]
                angle = triangulation_angle_deg(point, observations[0].image, observations[1].image)
                if max(errors) > float(config["triangulation"]["ransac_reprojection_threshold_px"]) or angle < float(config["triangulation"]["minimum_triangulation_angle_deg"]):
                    continue
                seeds.append({"xyz": point, "observations": observations, "score": angle / (1.0 + epipolar_error), "source": f"epipolar:{left_name}:{right_name}"})
    if not seeds:
        raise RuntimeError("No track or epipolar seed survived the frozen fit-side rules")
    seed_points = np.asarray([seed["xyz"] for seed in seeds], dtype=np.float64)
    seed_scores = np.asarray([seed["score"] for seed in seeds], dtype=np.float64)
    selected = voxel_representatives(
        seed_points,
        seed_scores,
        float(config["triangulation"]["candidate_voxel_size_official"]),
        int(config["triangulation"]["maximum_candidates_before_support"]),
    )
    seeds = [seeds[int(index)] for index in selected]
    points = np.asarray([seed["xyz"] for seed in seeds], dtype=np.float64)
    support_lists = [list(seed["observations"]) for seed in seeds]
    support_threshold = float(config["triangulation"]["support_centerline_distance_px"])
    for name in fit_names:
        distance, indices = nearest_skeleton(centerlines[name].skeleton)
        width_map = 2.0 * cv2.distanceTransform(centerlines[name].mask.astype(np.uint8), cv2.DIST_L2, 5)
        pixels, depth = _project_many(points, images[name], intrinsic)
        rounded = np.rint(pixels).astype(np.int64)
        valid = (
            (depth > 0)
            & np.isfinite(pixels).all(axis=1)
            & (rounded[:, 0] >= 0)
            & (rounded[:, 0] < camera.width)
            & (rounded[:, 1] >= 0)
            & (rounded[:, 1] < camera.height)
        )
        valid_indices = np.flatnonzero(valid)
        if not len(valid_indices):
            continue
        x, y = rounded[valid_indices, 0], rounded[valid_indices, 1]
        close = distance[y, x] <= support_threshold
        for candidate_index, sample_x, sample_y in zip(valid_indices[close], x[close], y[close]):
            nearest_y, nearest_x = int(indices[0, sample_y, sample_x]), int(indices[1, sample_y, sample_x])
            width = float(width_map[nearest_y, nearest_x])
            support_lists[int(candidate_index)].append(
                CableObservation(name, np.array([nearest_x, nearest_y], dtype=np.float64), width, images[name], "support")
            )
    candidates = []
    for seed, observations in zip(seeds, support_lists):
        observations = _limit_observations(observations, int(config["triangulation"]["maximum_support_observations_per_candidate"]))
        candidate = robust_triangulate(
            observations,
            intrinsic,
            float(config["matching"]["minimum_camera_baseline"]),
            float(config["matching"]["maximum_camera_baseline"]),
            float(config["triangulation"]["minimum_triangulation_angle_deg"]),
            float(config["triangulation"]["ransac_reprojection_threshold_px"]),
            int(config["triangulation"]["minimum_independent_support_views"]),
        )
        if candidate is None:
            continue
        if float(np.median(candidate.reprojection_errors_px)) > float(config["triangulation"]["maximum_fit_median_reprojection_error_px"]):
            continue
        if float(np.quantile(candidate.reprojection_errors_px, 0.9)) > float(config["triangulation"]["maximum_fit_p90_reprojection_error_px"]):
            continue
        candidates.append((candidate, seed["source"]))
    if candidates:
        final_points = np.asarray([candidate.xyz for candidate, _ in candidates])
        final_scores = np.asarray([candidate.confidence for candidate, _ in candidates])
        final_selected = voxel_representatives(
            final_points,
            final_scores,
            float(config["triangulation"]["candidate_voxel_size_official"]),
            len(candidates),
        )
        candidates = [candidates[int(index)] for index in final_selected]
    candidate_points = np.asarray([candidate.xyz for candidate, _ in candidates], dtype=np.float64).reshape(-1, 3)
    confidences = np.asarray([candidate.confidence for candidate, _ in candidates], dtype=np.float64)
    widths = np.asarray([np.median([observation.width_px for observation in candidate.observations]) for candidate, _ in candidates], dtype=np.float64)
    support_views = [tuple(observation.image_name for observation in candidate.observations) for candidate, _ in candidates]
    curves = fit_piecewise_lines(candidate_points, confidences, support_views, widths, config["curves"], int(config["seed"])) if len(candidates) else []
    candidate_records = []
    for candidate, source in candidates:
        candidate_records.append(
            {
                "xyz": candidate.xyz.tolist(),
                "source": source,
                "support_views": [observation.image_name for observation in candidate.observations],
                "source_pixels_xy": [observation.pixel_xy.tolist() for observation in candidate.observations],
                "source_kinds": [observation.source for observation in candidate.observations],
                "support_count": len(candidate.observations),
                "reprojection_errors_px": candidate.reprojection_errors_px.tolist(),
                "minimum_baseline": candidate.minimum_baseline,
                "maximum_baseline": candidate.maximum_baseline,
                "minimum_triangulation_angle_deg": candidate.minimum_triangulation_angle_deg,
                "local_width_px": float(np.median([observation.width_px for observation in candidate.observations])),
                "confidence": candidate.confidence,
            }
        )
    curve_records = [
        {
            "start": curve.start.tolist(),
            "end": curve.end.tolist(),
            "tangent": curve.tangent.tolist(),
            "length": curve.length,
            "inlier_candidate_indices": curve.inlier_indices.tolist(),
            "confidence": curve.confidence,
            "support_views": list(curve.support_views),
            "local_width_px": curve.local_width_px,
        }
        for curve in curves
    ]
    _write_json(output_root / "fit" / "candidates.json", candidate_records)
    _write_json(output_root / "fit" / "curves.json", curve_records)
    pose_errors = []
    for name in fit_names:
        image = images[name]
        recovered_t = -image.rotation_world_to_camera @ image.center
        pose_errors.append(float(np.max(np.abs(recovered_t - image.tvec))))
    centers = np.asarray([images[name].center for name in fit_names])
    diameter = float(np.max(np.linalg.norm(centers[:, None] - centers[None], axis=2)))
    maximum_candidate_distance = 0.0
    for candidate, _ in candidates:
        maximum_candidate_distance = max(
            maximum_candidate_distance,
            min(float(np.linalg.norm(candidate.xyz - images[name].center)) for name in fit_names),
        )
    coordinate_checks = {
        "pose_roundtrip_max_abs_error": max(pose_errors),
        "pose_roundtrip_pass": max(pose_errors) <= float(config["coordinate_audit"]["maximum_pose_roundtrip_absolute_error"]),
        "fit_camera_diameter": diameter,
        "maximum_candidate_distance_to_fit_camera": maximum_candidate_distance,
        "scale_pass": maximum_candidate_distance <= float(config["coordinate_audit"]["maximum_candidate_distance_in_fit_camera_diameters"]) * diameter,
        "all_candidate_coordinates_finite": bool(np.isfinite(candidate_points).all()),
        "all_curve_bounds_finite": bool(all(np.isfinite(curve.start).all() and np.isfinite(curve.end).all() for curve in curves)),
    }
    summary = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "stage": "fit",
        "fit_views": len(fit_names),
        "fit_view_names_sha256": _hash_bytes("\n".join(fit_names).encode()),
        "geometry_check_loaded": False,
        "fixed_74_view_loaded": False,
        "camera_model": "PINHOLE converted from released SIMPLE_RADIAL with unchanged K",
        "pose_convention": "CAMERA_FROM_WORLD; C=-R^T t",
        "valid_strong_graph_edges": len(edge_lookup),
        "selected_trackless_pair_edges": len(pair_edges),
        "centerline_totals": {key: sum(int(row[key]) for row in rows) for key in ("components", "skeleton_components", "mask_pixels", "skeleton_pixels", "samples", "endpoints", "branches")},
        "track_summary": track_summary,
        "track_groups_considered": track_groups_considered,
        "track_seed_candidates": track_seed_count,
        "all_seed_candidates_before_voxel": len(seed_points),
        "seed_candidates_after_voxel_and_cap": len(seeds),
        "retained_candidates": len(candidates),
        "support_view_distribution": dict(sorted(Counter(record["support_count"] for record in candidate_records).items())),
        "curves": len(curves),
        "curve_total_length": float(sum(curve.length for curve in curves)),
        "curve_lengths": [curve.length for curve in curves],
        "curve_bounds_min": candidate_points.min(axis=0).tolist() if len(candidate_points) else None,
        "curve_bounds_max": candidate_points.max(axis=0).tolist() if len(candidate_points) else None,
        "coordinate_checks": coordinate_checks,
        "elapsed_seconds": time.perf_counter() - start,
    }
    _write_json(output_root / "fit" / "result.json", summary)
    freeze = {
        "config_sha256": _sha256(config_path),
        "split_sha256": _sha256(ROOT / config["data"]["geometry_split"]),
        "candidates_sha256": _sha256(output_root / "fit" / "candidates.json"),
        "curves_sha256": _sha256(output_root / "fit" / "curves.json"),
        "git": _git_evidence(),
        "geometry_check_opened": False,
        "fixed_74_view_opened": False,
    }
    _write_json(output_root / "fit" / "freeze.json", freeze)
    _manifest(output_root)
    return summary


def _load_curves(path: Path) -> list[CableCurve]:
    records = _read_json(path)
    return [
        CableCurve(
            start=np.asarray(record["start"], dtype=np.float64),
            end=np.asarray(record["end"], dtype=np.float64),
            tangent=np.asarray(record["tangent"], dtype=np.float64),
            length=float(record["length"]),
            inlier_indices=np.asarray(record["inlier_candidate_indices"], dtype=np.int64),
            confidence=float(record["confidence"]),
            support_views=tuple(record["support_views"]),
            local_width_px=float(record["local_width_px"]),
        )
        for record in records
    ]


def run_check(config_path: Path, config: dict, split: dict, data_root: Path, label_root: Path, output_root: Path) -> dict:
    start = time.perf_counter()
    freeze_path = output_root / "fit" / "freeze.json"
    freeze = _read_json(freeze_path)
    if freeze["config_sha256"] != _sha256(config_path) or freeze["split_sha256"] != _sha256(ROOT / config["data"]["geometry_split"]):
        raise ValueError("SEM-060 config or split changed after fit freeze")
    if freeze["curves_sha256"] != _sha256(output_root / "fit" / "curves.json") or freeze["candidates_sha256"] != _sha256(output_root / "fit" / "candidates.json"):
        raise ValueError("SEM-060 frozen fit artifacts changed before geometry-check")
    fit_result = _read_json(output_root / "fit" / "result.json")
    candidate_records = _read_json(output_root / "fit" / "candidates.json")
    curves = _load_curves(output_root / "fit" / "curves.json")
    camera, intrinsic, _distortion, images = _camera_setup(data_root)
    check_names = list(split["geometry_check"])
    prohibited = set(split["geometry_fit"]) | set(split["local_guard_band"]) | set(split["fixed_validation_read_only_names"]) | set(split["unlabeled_never_used"])
    if set(check_names) & prohibited:
        raise ValueError("Check inputs overlap a prohibited cohort")
    manifest = _read_json(ROOT / config["data"]["label_manifest"])
    records = {record["image"]: record for record in manifest["records"]}
    centerlines, rows = _extract_many(check_names, label_root, records, config["centerline"])
    _write_csv(output_root / "check" / "centerlines.csv", rows)
    per_view, predictions = [], {}
    aggregate = {
        "predicted_pixels": 0,
        "target_pixels": 0,
        "exact_tp": 0,
        "exact_union": 0,
        "far_false_positive_pixels": 0,
        "tolerant_prediction_tp": Counter(),
        "tolerant_target_tp": Counter(),
    }
    errors = []
    tolerances = [int(value) for value in config["evaluation"]["tolerance_pixels"]]
    for name in check_names:
        image = images[name]

        def project(points, image=image):
            return _project_many(points, image, intrinsic)

        prediction = rasterize_projected_curves(
            curves,
            project,
            camera.width,
            camera.height,
            float(config["curves"]["sampling_spacing_official"]),
        )
        target = centerlines[name].skeleton
        predictions[name] = prediction
        metrics = centerline_metrics(
            prediction,
            target,
            tolerances,
            float(config["evaluation"]["coverage_minimum_recall_at_2px"]),
        )
        metrics["image"] = name
        per_view.append(metrics)
        pred_count, target_count = int(np.count_nonzero(prediction)), int(np.count_nonzero(target))
        target_distance = ndimage.distance_transform_edt(~target) if target_count else np.full(target.shape, np.inf)
        pred_distance = ndimage.distance_transform_edt(~prediction) if pred_count else np.full(prediction.shape, np.inf)
        aggregate["predicted_pixels"] += pred_count
        aggregate["target_pixels"] += target_count
        aggregate["exact_tp"] += int(np.count_nonzero(prediction & target))
        aggregate["exact_union"] += int(np.count_nonzero(prediction | target))
        aggregate["far_false_positive_pixels"] += int(np.count_nonzero(prediction & (target_distance > int(config["evaluation"]["far_false_positive_distance_px"]))))
        errors.extend(target_distance[prediction].tolist())
        for tolerance in tolerances:
            aggregate["tolerant_prediction_tp"][tolerance] += int(np.count_nonzero(prediction & (target_distance <= tolerance)))
            aggregate["tolerant_target_tp"][tolerance] += int(np.count_nonzero(target & (pred_distance <= tolerance)))
    _write_csv(output_root / "check" / "per_view.csv", per_view)
    pred_total, target_total = aggregate["predicted_pixels"], aggregate["target_pixels"]
    metrics = {
        "exact_precision": aggregate["exact_tp"] / pred_total if pred_total else 0.0,
        "exact_recall": aggregate["exact_tp"] / target_total if target_total else 0.0,
        "exact_iou": aggregate["exact_tp"] / aggregate["exact_union"] if aggregate["exact_union"] else 0.0,
        "prediction_error_median_px": float(np.median(errors)) if errors else 1.0e9,
        "prediction_error_p90_px": float(np.quantile(errors, 0.9)) if errors else 1.0e9,
        "far_false_positive_pixels": aggregate["far_false_positive_pixels"],
        "far_false_positive_fraction": aggregate["far_false_positive_pixels"] / pred_total if pred_total else 1.0,
        "covered_check_views": sum(bool(row["covered_at_2px"]) for row in per_view),
        "covered_check_view_fraction": sum(bool(row["covered_at_2px"]) for row in per_view) / len(per_view),
        "predicted_pixels": pred_total,
        "target_pixels": target_total,
    }
    for tolerance in tolerances:
        metrics[f"precision_at_{tolerance}px"] = aggregate["tolerant_prediction_tp"][tolerance] / pred_total if pred_total else 0.0
        metrics[f"recall_at_{tolerance}px"] = aggregate["tolerant_target_tp"][tolerance] / target_total if target_total else 0.0
    scores = sorted(
        per_view,
        key=lambda row: (
            -(2 * row["precision_at_2px"] * row["recall_at_2px"] / max(row["precision_at_2px"] + row["recall_at_2px"], np.finfo(float).eps)),
            row["image"],
        ),
    )
    success_count = int(config["evaluation"]["representative_success_count"])
    failure_count = int(config["evaluation"]["representative_failure_count"])
    representative_success = [row["image"] for row in scores[:success_count]]
    representative_failure = [row["image"] for row in reversed(scores[-failure_count:])]
    for category, names in (("success", representative_success), ("failure", representative_failure)):
        for name in names:
            write_case_overlay(centerlines[name].skeleton, predictions[name], output_root / "check" / "cases" / f"{category}_{name}")
    gate_config = config["geometry_gate"]
    support_minimum = min((int(record["support_count"]) for record in candidate_records), default=0)
    coordinate_checks = fit_result["coordinate_checks"]
    coordinate_pass = bool(
        coordinate_checks["pose_roundtrip_pass"]
        and coordinate_checks["scale_pass"]
        and coordinate_checks["all_candidate_coordinates_finite"]
        and coordinate_checks["all_curve_bounds_finite"]
        and split["checks"]["nearby_camera_graph_edges_fit_to_check"] == 0
    )
    gates = {
        "minimum_support_views": support_minimum >= int(gate_config["minimum_support_views_per_retained_candidate"]),
        "precision_at_2px": metrics["precision_at_2px"] >= float(gate_config["minimum_precision_at_2px"]),
        "recall_at_2px": metrics["recall_at_2px"] >= float(gate_config["minimum_recall_at_2px"]),
        "covered_check_view_fraction": metrics["covered_check_view_fraction"] >= float(gate_config["minimum_covered_check_view_fraction"]),
        "median_reprojection_error": metrics["prediction_error_median_px"] <= float(gate_config["maximum_median_reprojection_error_px"]),
        "p90_reprojection_error": metrics["prediction_error_p90_px"] <= float(gate_config["maximum_p90_reprojection_error_px"]),
        "coordinate_scale_depth_and_leakage": coordinate_pass,
        "far_false_positive_not_major": metrics["far_false_positive_fraction"] <= float(gate_config["maximum_far_false_positive_fraction"]),
    }
    passed = bool(gates and all(gates.values()))
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "stage": "geometry_check",
        "decision": "pass_geometry_gate" if passed else "drop_geometry_not_reliable",
        "geometry_gate_passed": passed,
        "geometry_fit_views": len(split["geometry_fit"]),
        "geometry_check_views": len(check_names),
        "local_guard_views": len(split["local_guard_band"]),
        "fixed_74_view_loaded": False,
        "unlabeled_or_hidden_inputs_loaded": False,
        "curves": len(curves),
        "curve_total_length": fit_result["curve_total_length"],
        "retained_candidates": len(candidate_records),
        "minimum_candidate_support_views": support_minimum,
        "metrics": metrics,
        "gates": gates,
        "coordinate_checks": coordinate_checks,
        "representative_success": representative_success,
        "representative_failure": representative_failure,
        "config_sha256": _sha256(config_path),
        "split_sha256": _sha256(ROOT / config["data"]["geometry_split"]),
        "fit_freeze_sha256": _sha256(freeze_path),
        "git": _git_evidence(),
        "elapsed_seconds": time.perf_counter() - start,
    }
    _write_json(output_root / "check" / "result.json", result)
    _manifest(output_root)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("fit", "check"), required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = _read_json(config_path)
    data_root = _resolve("$ICSHM_DATA_ROOT")
    output_root = _resolve(config["output"]["root"])
    label_root = _resolve(config["data"]["label_root"])
    split = _read_json(ROOT / config["data"]["geometry_split"])
    if split["counts"] != {
        "fit": int(config["data"]["expected_fit_views"]),
        "check": int(config["data"]["expected_check_views"]),
        "guard": int(config["data"]["expected_guard_views"]),
    }:
        raise ValueError("SEM-060 split counts do not match the preregistered config")
    output_root.mkdir(parents=True, exist_ok=True)
    invocation = {
        "argv": ["python3", "scripts/run_sem060_cable_geometry.py", "--config", str(args.config), "--stage", args.stage],
        "config": str(args.config),
        "stage": args.stage,
        "data_root_from": "ICSHM_DATA_ROOT",
        "output_root_from": "ICSHM_OUTPUT_ROOT",
    }
    _write_json(output_root / f"invocation_{args.stage}.json", invocation)
    result = run_fit(config_path, config, split, data_root, label_root, output_root) if args.stage == "fit" else run_check(config_path, config, split, data_root, label_root, output_root)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
