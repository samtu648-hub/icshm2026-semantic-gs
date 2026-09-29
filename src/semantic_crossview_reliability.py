"""Frozen-depth cross-view reliability helpers for SEM-395.

Selection and geometry functions in this module deliberately have no semantic
ground-truth input. Ground truth is only consumed by the runner's later score
stage after the camera plan and evidence artifacts have been sealed.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

PAIR_STAGE_NAMES = (
    "target_invalid_depth",
    "target_low_opacity",
    "target_depth_discontinuity",
    "source_outside_or_behind",
    "source_invalid_depth",
    "source_low_opacity",
    "source_depth_discontinuity",
    "forward_depth_conflict",
    "backward_outside_or_behind",
    "backward_depth_conflict",
    "cycle_reprojection",
    "source_low_confidence",
    "reliable",
)


def depth_proxy(inverse_depth_sum: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Return alpha / sum(T alpha / z), not geometric ground-truth depth."""
    inverse = np.asarray(inverse_depth_sum, dtype=np.float32)
    opacity = np.asarray(alpha, dtype=np.float32)
    if inverse.shape != opacity.shape or inverse.ndim != 2:
        raise ValueError("Depth and opacity must be matching HxW arrays")
    valid = np.isfinite(inverse) & np.isfinite(opacity) & (inverse > 0) & (opacity > 0)
    result = np.full(inverse.shape, np.nan, dtype=np.float32)
    np.divide(opacity, inverse, out=result, where=valid)
    result[~np.isfinite(result) | (result <= 0)] = np.nan
    return result


def depth_continuity_mask(
    depth: np.ndarray,
    alpha: np.ndarray,
    *,
    alpha_minimum: float,
    relative_range_maximum: float,
) -> np.ndarray:
    """Require a complete, opaque 3x3 neighbourhood with limited depth range."""
    z = np.asarray(depth, dtype=np.float32)
    opacity = np.asarray(alpha, dtype=np.float32)
    if z.shape != opacity.shape or z.ndim != 2:
        raise ValueError("Depth and opacity must be matching HxW arrays")
    valid = np.isfinite(z) & (z > 0) & np.isfinite(opacity) & (opacity >= alpha_minimum)
    low = minimum_filter(
        np.where(valid, z, -np.inf), size=3, mode="constant", cval=-np.inf
    )
    high = maximum_filter(
        np.where(valid, z, np.inf), size=3, mode="constant", cval=np.inf
    )
    complete = (
        minimum_filter(valid.astype(np.uint8), size=3, mode="constant", cval=0) == 1
    )
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    relative = np.subtract(high, low) / np.maximum(z, np.finfo(np.float32).tiny)
    return complete & np.isfinite(relative) & (relative <= relative_range_maximum)


def project_world(
    points: np.ndarray, camera: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points to array coordinates under CAMERA_FROM_WORLD.

    COLMAP intrinsics use edge-origin pixel coordinates. Array coordinate zero
    denotes the first pixel centre, so projected edge coordinates are shifted
    by -0.5 before bilinear sampling.
    """
    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("Expected Nx3 world points")
    rotation = np.asarray(camera.pose.rotation, dtype=np.float64)
    translation = np.asarray(camera.pose.translation, dtype=np.float64)
    camera_points = xyz @ rotation.T + translation
    z = camera_points[:, 2]
    edge_x = camera.matrix[0, 0] * camera_points[:, 0] / z + camera.matrix[0, 2]
    edge_y = camera.matrix[1, 1] * camera_points[:, 1] / z + camera.matrix[1, 2]
    return edge_x - 0.5, edge_y - 0.5, z


def backproject_pixels(
    x: np.ndarray, y: np.ndarray, depth: np.ndarray, camera: Any
) -> np.ndarray:
    """Backproject array-coordinate pixel centres to world points."""
    px = np.asarray(x, dtype=np.float64)
    py = np.asarray(y, dtype=np.float64)
    z = np.asarray(depth, dtype=np.float64)
    if px.shape != py.shape or px.shape != z.shape:
        raise ValueError("Pixel coordinates and depth must have matching shape")
    edge_x, edge_y = px + 0.5, py + 0.5
    camera_points = np.column_stack(
        (
            (edge_x - camera.matrix[0, 2]) * z / camera.matrix[0, 0],
            (edge_y - camera.matrix[1, 2]) * z / camera.matrix[1, 1],
            z,
        )
    )
    translation = np.asarray(camera.pose.translation, dtype=np.float64)
    rotation = np.asarray(camera.pose.rotation, dtype=np.float64)
    return (camera_points - translation) @ rotation


def bilinear_sample(values: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample HxW or CxHxW values at array-centre coordinates."""
    array = np.asarray(values)
    px, py = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if px.shape != py.shape or px.ndim != 1:
        raise ValueError("Sample coordinates must be matching vectors")
    if array.ndim == 2:
        channels = array[None]
        scalar = True
    elif array.ndim == 3:
        channels = array
        scalar = False
    else:
        raise ValueError("Expected HxW or CxHxW values")
    _, height, width = channels.shape
    inside = (
        np.isfinite(px)
        & np.isfinite(py)
        & (px >= 0)
        & (py >= 0)
        & (px <= width - 1)
        & (py <= height - 1)
    )
    safe_x = np.clip(np.nan_to_num(px), 0, width - 1)
    safe_y = np.clip(np.nan_to_num(py), 0, height - 1)
    x0, y0 = np.floor(safe_x).astype(np.int64), np.floor(safe_y).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    wx, wy = safe_x - x0, safe_y - y0
    sampled = (
        channels[:, y0, x0] * ((1 - wx) * (1 - wy))[None]
        + channels[:, y0, x1] * (wx * (1 - wy))[None]
        + channels[:, y1, x0] * ((1 - wx) * wy)[None]
        + channels[:, y1, x1] * (wx * wy)[None]
    )
    sampled = np.asarray(sampled, dtype=np.float32)
    sampled[:, ~inside] = np.nan
    return sampled[0] if scalar else sampled


def nearest_pixel_index(
    x: np.ndarray, y: np.ndarray, width: int, height: int
) -> np.ndarray:
    """Return nearest source pixel indices, or -1 outside the source image."""
    px, py = np.asarray(x), np.asarray(y)
    safe_x = np.nan_to_num(px, nan=-1.0, posinf=-1.0, neginf=-1.0)
    safe_y = np.nan_to_num(py, nan=-1.0, posinf=-1.0, neginf=-1.0)
    ix = np.floor(safe_x + 0.5).astype(np.int64)
    iy = np.floor(safe_y + 0.5).astype(np.int64)
    valid = (
        np.isfinite(px)
        & np.isfinite(py)
        & (ix >= 0)
        & (ix < width)
        & (iy >= 0)
        & (iy < height)
    )
    return np.where(valid, iy * width + ix, -1)


def _reject(
    stage: np.ndarray, active: np.ndarray, condition: np.ndarray, code: int
) -> np.ndarray:
    rejected = active & ~condition
    stage[rejected] = code
    return active & condition


def evaluate_pair(
    target_depth: np.ndarray,
    target_alpha: np.ndarray,
    source_depth: np.ndarray,
    source_alpha: np.ndarray,
    source_probability: np.ndarray,
    target_camera: Any,
    source_camera: Any,
    thresholds: dict[str, float],
    *,
    chunk_rows: int = 64,
) -> dict[str, Any]:
    """Apply the single frozen two-way geometry and confidence filter."""
    target_z = np.asarray(target_depth, dtype=np.float32)
    target_opacity = np.asarray(target_alpha, dtype=np.float32)
    source_z = np.asarray(source_depth, dtype=np.float32)
    source_opacity = np.asarray(source_alpha, dtype=np.float32)
    probability = np.asarray(source_probability, dtype=np.float32)
    height, width = target_z.shape
    if target_opacity.shape != (height, width):
        raise ValueError("Target depth/opacity shape mismatch")
    if source_z.shape != (source_camera.height, source_camera.width):
        raise ValueError("Source depth shape differs from camera")
    if source_opacity.shape != source_z.shape or probability.shape != (
        5,
        *source_z.shape,
    ):
        raise ValueError("Source evidence has incompatible shape")
    if (height, width) != (target_camera.height, target_camera.width):
        raise ValueError("Target depth shape differs from camera")
    if not np.isfinite(probability).all() or np.any(probability < 0):
        raise ValueError("Invalid source probability values")
    if not np.allclose(probability.sum(0), 1, atol=2e-5, rtol=0):
        raise ValueError("Source probabilities do not sum to one")

    continuity = depth_continuity_mask(
        target_z,
        target_opacity,
        alpha_minimum=thresholds["alpha_minimum"],
        relative_range_maximum=thresholds["depth_discontinuity_relative_maximum"],
    )
    source_continuity = depth_continuity_mask(
        source_z,
        source_opacity,
        alpha_minimum=thresholds["alpha_minimum"],
        relative_range_maximum=thresholds["depth_discontinuity_relative_maximum"],
    )
    stage_counts = np.zeros(len(PAIR_STAGE_NAMES), dtype=np.int64)
    sparse: dict[str, list[np.ndarray]] = defaultdict(list)
    occlusion_counts = {
        "projected_behind_source_surface": 0,
        "projected_in_front_of_source_surface": 0,
    }

    for row0 in range(0, height, chunk_rows):
        row1 = min(row0 + chunk_rows, height)
        yy, xx = np.meshgrid(np.arange(row0, row1), np.arange(width), indexing="ij")
        flat_y, flat_x = yy.ravel(), xx.ravel()
        index = flat_y * width + flat_x
        z = target_z[row0:row1].ravel()
        opacity = target_opacity[row0:row1].ravel()
        stage = np.full(index.shape, -1, dtype=np.int8)
        active = np.ones(index.shape, dtype=bool)
        active = _reject(
            stage, active, np.isfinite(z) & (z > thresholds["minimum_depth"]), 0
        )
        active = _reject(
            stage,
            active,
            np.isfinite(opacity) & (opacity >= thresholds["alpha_minimum"]),
            1,
        )
        active = _reject(stage, active, continuity[row0:row1].ravel(), 2)

        world = backproject_pixels(
            flat_x, flat_y, np.where(np.isfinite(z), z, 1), target_camera
        )
        source_x, source_y, projected_source_z = project_world(world, source_camera)
        inside_source = (
            np.isfinite(source_x)
            & np.isfinite(source_y)
            & np.isfinite(projected_source_z)
            & (projected_source_z > thresholds["minimum_depth"])
            & (source_x >= 0)
            & (source_x <= source_camera.width - 1)
            & (source_y >= 0)
            & (source_y <= source_camera.height - 1)
        )
        active = _reject(stage, active, inside_source, 3)
        sampled_source_z = bilinear_sample(source_z, source_x, source_y)
        sampled_source_alpha = bilinear_sample(source_opacity, source_x, source_y)
        valid_source_depth = np.isfinite(sampled_source_z) & (
            sampled_source_z > thresholds["minimum_depth"]
        )
        active = _reject(stage, active, valid_source_depth, 4)
        valid_source_alpha = np.isfinite(sampled_source_alpha) & (
            sampled_source_alpha >= thresholds["alpha_minimum"]
        )
        active = _reject(stage, active, valid_source_alpha, 5)
        source_nearest = nearest_pixel_index(
            source_x, source_y, source_camera.width, source_camera.height
        )
        continuous_source = np.zeros(source_nearest.shape, dtype=bool)
        valid_nearest = source_nearest >= 0
        continuous_source[valid_nearest] = source_continuity.ravel()[
            source_nearest[valid_nearest]
        ]
        active = _reject(stage, active, continuous_source, 6)
        source_sample_valid = (
            np.isfinite(sampled_source_z)
            & (sampled_source_z > thresholds["minimum_depth"])
            & np.isfinite(sampled_source_alpha)
            & (sampled_source_alpha >= thresholds["alpha_minimum"])
        )
        if np.any(active & ~source_sample_valid):
            raise RuntimeError("Source validity stages are internally inconsistent")
        relative_forward = np.abs(projected_source_z - sampled_source_z) / np.maximum(
            sampled_source_z, thresholds["minimum_depth"]
        )
        forward_consistent = np.isfinite(relative_forward) & (
            relative_forward <= thresholds["relative_depth_maximum"]
        )
        forward_rejected = active & ~forward_consistent
        occlusion_counts["projected_behind_source_surface"] += int(
            np.count_nonzero(forward_rejected & (projected_source_z > sampled_source_z))
        )
        occlusion_counts["projected_in_front_of_source_surface"] += int(
            np.count_nonzero(forward_rejected & (projected_source_z < sampled_source_z))
        )
        active = _reject(stage, active, forward_consistent, 7)

        source_world = backproject_pixels(
            source_x,
            source_y,
            np.where(np.isfinite(sampled_source_z), sampled_source_z, 1),
            source_camera,
        )
        target_x2, target_y2, target_z2 = project_world(source_world, target_camera)
        inside_target = (
            np.isfinite(target_x2)
            & np.isfinite(target_y2)
            & np.isfinite(target_z2)
            & (target_z2 > thresholds["minimum_depth"])
            & (target_x2 >= 0)
            & (target_x2 <= width - 1)
            & (target_y2 >= 0)
            & (target_y2 <= height - 1)
        )
        active = _reject(stage, active, inside_target, 8)
        relative_backward = np.abs(target_z2 - z) / np.maximum(
            z, thresholds["minimum_depth"]
        )
        backward_consistent = np.isfinite(relative_backward) & (
            relative_backward <= thresholds["relative_depth_maximum"]
        )
        active = _reject(stage, active, backward_consistent, 9)
        cycle = np.hypot(target_x2 - flat_x, target_y2 - flat_y)
        cycle_consistent = np.isfinite(cycle) & (
            cycle <= thresholds["cycle_reprojection_maximum_px"]
        )
        active = _reject(stage, active, cycle_consistent, 10)

        sampled_probability = bilinear_sample(probability, source_x, source_y)
        normalizer = np.nansum(sampled_probability, axis=0)
        sampled_probability = np.divide(
            sampled_probability,
            normalizer[None],
            out=np.full_like(sampled_probability, np.nan),
            where=normalizer[None] > 0,
        )
        source_class = np.argmax(
            np.nan_to_num(sampled_probability, nan=-1), axis=0
        ).astype(np.uint8)
        confidence = np.max(np.nan_to_num(sampled_probability, nan=-np.inf), axis=0)
        confident = np.isfinite(confidence) & (
            confidence >= thresholds["source_confidence_minimum"]
        )
        active = _reject(stage, active, confident, 11)
        stage[active] = 12
        if np.any(stage < 0):
            raise RuntimeError("Pair stage assignment is incomplete")
        stage_counts += np.bincount(stage, minlength=len(PAIR_STAGE_NAMES))

        use = stage == 12
        if np.any(use):
            source_index = nearest_pixel_index(
                source_x[use], source_y[use], source_camera.width, source_camera.height
            )
            if np.any(source_index < 0):
                raise RuntimeError("Reliable source sample has no nearest source pixel")
            sparse["target_index"].append(index[use].astype(np.int32))
            sparse["source_index"].append(source_index.astype(np.int32))
            sparse["source_class"].append(source_class[use])
            sparse["source_confidence"].append(confidence[use].astype(np.float16))
            sparse["target_alpha"].append(opacity[use].astype(np.float16))
            sparse["source_alpha"].append(sampled_source_alpha[use].astype(np.float16))
            sparse["relative_depth_error"].append(
                relative_forward[use].astype(np.float16)
            )
            sparse["cycle_reprojection_px"].append(cycle[use].astype(np.float16))

    if int(stage_counts.sum()) != height * width:
        raise RuntimeError("Pair stage counts do not cover the target frame")
    packed = {
        key: np.concatenate(parts) if parts else np.empty(0, dtype=dtype)
        for key, parts, dtype in (
            ("target_index", sparse["target_index"], np.int32),
            ("source_index", sparse["source_index"], np.int32),
            ("source_class", sparse["source_class"], np.uint8),
            ("source_confidence", sparse["source_confidence"], np.float16),
            ("target_alpha", sparse["target_alpha"], np.float16),
            ("source_alpha", sparse["source_alpha"], np.float16),
            ("relative_depth_error", sparse["relative_depth_error"], np.float16),
            ("cycle_reprojection_px", sparse["cycle_reprojection_px"], np.float16),
        )
    }
    if packed["target_index"].size and np.any(np.diff(packed["target_index"]) <= 0):
        raise RuntimeError("Reliable target indices must be unique and sorted")
    return {
        "arrays": packed,
        "stage_counts": {
            name: int(stage_counts[i]) for i, name in enumerate(PAIR_STAGE_NAMES)
        },
        "occlusion_counts": occlusion_counts,
        "total_pixels": int(height * width),
    }


def combine_source_evidence(
    first: dict[str, np.ndarray], second: dict[str, np.ndarray], total_pixels: int
) -> dict[str, Any]:
    """Intersect two reliable sources and require semantic agreement."""
    a, b = first["target_index"], second["target_index"]
    common, ia, ib = np.intersect1d(a, b, assume_unique=True, return_indices=True)
    agreement = first["source_class"][ia] == second["source_class"][ib]
    use_a, use_b = ia[agreement], ib[agreement]
    reliable = common[agreement]
    return {
        "target_index": reliable.astype(np.int32),
        "consensus_class": first["source_class"][use_a].astype(np.uint8),
        "first_source_index": first["source_index"][use_a].astype(np.int32),
        "second_source_index": second["source_index"][use_b].astype(np.int32),
        "first_source_confidence": first["source_confidence"][use_a],
        "second_source_confidence": second["source_confidence"][use_b],
        "target_alpha": np.minimum(
            first["target_alpha"][use_a], second["target_alpha"][use_b]
        ),
        "source_alpha": np.minimum(
            first["source_alpha"][use_a], second["source_alpha"][use_b]
        ),
        "counts": {
            "total_pixels": int(total_pixels),
            "first_source_reliable": int(a.size),
            "second_source_reliable": int(b.size),
            "both_sources_reliable": int(common.size),
            "source_semantic_disagreement": int(np.count_nonzero(~agreement)),
            "reliable_consensus": int(reliable.size),
            "insufficient_two_source_support": int(total_pixels - common.size),
        },
    }


def confusion_on_indices(
    target: np.ndarray, prediction: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    truth = np.asarray(target, dtype=np.uint8).ravel()[indices]
    estimate = np.asarray(prediction, dtype=np.uint8).ravel()[indices]
    if truth.shape != estimate.shape or np.any(truth > 4) or np.any(estimate > 4):
        raise ValueError("Invalid five-class indexed labels")
    return np.bincount(truth.astype(np.int64) * 5 + estimate, minlength=25).reshape(
        5, 5
    )


def replacement_counts(
    target: np.ndarray,
    baseline: np.ndarray,
    indices: np.ndarray,
    consensus: np.ndarray,
) -> dict[str, int]:
    truth = np.asarray(target, dtype=np.uint8).ravel()[indices]
    old = np.asarray(baseline, dtype=np.uint8).ravel()[indices]
    new = np.asarray(consensus, dtype=np.uint8)
    if not (truth.shape == old.shape == new.shape):
        raise ValueError("Replacement arrays differ")
    corrected = (old != truth) & (new == truth)
    damaged = (old == truth) & (new != truth)
    return {
        "eligible_pixels": int(indices.size),
        "changed_pixels": int(np.count_nonzero(old != new)),
        "corrected_pixels": int(np.count_nonzero(corrected)),
        "damaged_pixels": int(np.count_nonzero(damaged)),
        "net_correct_pixels": int(
            np.count_nonzero(corrected) - np.count_nonzero(damaged)
        ),
        "sources_agree_target_gt_disagrees": int(np.count_nonzero(new != truth)),
    }


def _seeded_rank(seed: int, *parts: str) -> str:
    return hashlib.sha256((str(seed) + "\0" + "\0".join(parts)).encode()).hexdigest()


def segment_membership(segments: dict[str, Any]) -> dict[str, str]:
    membership: dict[str, str] = {}
    for segment in segments["segments"]:
        for name in segment["images"]:
            if name in membership:
                raise ValueError("Image appears in multiple trajectory segments")
            membership[name] = segment["segment_id"]
    return membership


def select_diagnostic_plan(
    graph: dict[str, Any],
    segments: dict[str, Any],
    train_names: list[str],
    *,
    seed: int,
    target_count: int,
    sources_per_target: int,
    source_minimum_shared_points: int,
    source_maximum_axis_angle_deg: float,
    source_minimum_center_distance: float,
    source_maximum_center_distance: float,
    second_source_minimum_separation: float,
) -> dict[str, Any]:
    """Select targets and source pairs from metadata only, without labels or pixels."""
    train = set(train_names)
    if len(train) != len(train_names) or target_count <= 0 or sources_per_target != 2:
        raise ValueError("Invalid training universe or fixed diagnostic budget")
    nodes = {node["name"]: node for node in graph["nodes"]}
    membership = segment_membership(segments)
    if not train <= nodes.keys() or not train <= membership.keys():
        raise ValueError("Missing graph or segment metadata for training views")
    adjacency = graph["adjacency"]

    def candidates(name: str) -> list[dict[str, Any]]:
        rows = []
        for edge in adjacency[name]:
            other = edge["target"] if edge["source"] == name else edge["source"]
            if other not in train or other == name:
                continue
            if edge["shared_point3d"] < source_minimum_shared_points:
                continue
            if edge["optical_axis_angle_deg"] > source_maximum_axis_angle_deg:
                continue
            if (
                not source_minimum_center_distance
                <= edge["center_distance"]
                <= source_maximum_center_distance
            ):
                continue
            rows.append({**edge, "name": other})
        return rows

    eligible = {name: candidates(name) for name in sorted(train)}
    eligible = {
        name: rows for name, rows in eligible.items() if len(rows) >= sources_per_target
    }
    by_segment: dict[str, list[str]] = defaultdict(list)
    for name in eligible:
        by_segment[membership[name]].append(name)
    active_segments = sorted(by_segment)
    if target_count < len(active_segments):
        raise ValueError("Target budget cannot cover every eligible trajectory segment")
    quotas = {segment: 1 for segment in active_segments}
    remaining = target_count - len(active_segments)
    allocation_order = sorted(
        active_segments, key=lambda s: (-len(by_segment[s]), _seeded_rank(seed, s))
    )
    cursor = 0
    while remaining:
        segment = allocation_order[cursor % len(allocation_order)]
        if quotas[segment] < len(by_segment[segment]):
            quotas[segment] += 1
            remaining -= 1
        cursor += 1
        if cursor > target_count * len(allocation_order) * 2:
            raise ValueError("Unable to allocate target budget across segments")

    targets = []
    for segment in active_segments:
        names = sorted(by_segment[segment])
        quota = quotas[segment]
        used: set[str] = set()
        for slot in range(quota):
            ideal = (slot + 1) * (len(names) - 1) / (quota + 1)
            order = sorted(
                range(len(names)),
                key=lambda index: (
                    abs(index - ideal),
                    _seeded_rank(seed, segment, str(slot), names[index]),
                ),
            )
            chosen = next(names[index] for index in order if names[index] not in used)
            used.add(chosen)
            rows = eligible[chosen]
            first = min(
                rows,
                key=lambda row: (
                    -row["shared_point3d"],
                    row["center_distance"],
                    row["optical_axis_angle_deg"],
                    row["name"],
                ),
            )
            first_center = np.asarray(nodes[first["name"]]["center"], dtype=np.float64)
            diverse = [
                row
                for row in rows
                if row["name"] != first["name"]
                and np.linalg.norm(
                    np.asarray(nodes[row["name"]]["center"]) - first_center
                )
                >= second_source_minimum_separation
            ]
            fallback_used = not diverse
            pool = diverse or [row for row in rows if row["name"] != first["name"]]
            second = min(
                pool,
                key=lambda row: (
                    -row["shared_point3d"] * row["center_distance"],
                    -row["shared_point3d"],
                    row["optical_axis_angle_deg"],
                    row["name"],
                ),
            )
            targets.append(
                {
                    "target": chosen,
                    "segment_id": segment,
                    "selection_slot": slot,
                    "sources": [
                        {
                            key: first[key]
                            for key in (
                                "name",
                                "shared_point3d",
                                "center_distance",
                                "optical_axis_angle_deg",
                            )
                        },
                        {
                            key: second[key]
                            for key in (
                                "name",
                                "shared_point3d",
                                "center_distance",
                                "optical_axis_angle_deg",
                            )
                        },
                    ],
                    "second_source_diversity_fallback": fallback_used,
                }
            )
    targets.sort(
        key=lambda row: (row["segment_id"], row["selection_slot"], row["target"])
    )
    if (
        len(targets) != target_count
        or len({row["target"] for row in targets}) != target_count
    ):
        raise RuntimeError("Diagnostic target selection did not meet the sealed budget")
    return {
        "targets": targets,
        "target_count": len(targets),
        "pair_count": sum(len(row["sources"]) for row in targets),
        "eligible_target_count": len(eligible),
        "covered_segments": sorted({row["segment_id"] for row in targets}),
        "segment_quotas": quotas,
    }


def camera_graph_connected(
    names: set[str], adjacency: dict[str, list[dict[str, Any]]]
) -> bool:
    if not names:
        return False
    start = min(names)
    seen, stack = {start}, [start]
    while stack:
        current = stack.pop()
        for edge in adjacency[current]:
            other = edge["target"] if edge["source"] == current else edge["source"]
            if other in names and other not in seen:
                seen.add(other)
                stack.append(other)
    return seen == names


def camera_graph_component_count(
    names: set[str], adjacency: dict[str, list[dict[str, Any]]]
) -> int:
    remaining = set(names)
    components = 0
    while remaining:
        components += 1
        start = min(remaining)
        seen, stack = {start}, [start]
        while stack:
            current = stack.pop()
            for edge in adjacency[current]:
                other = edge["target"] if edge["source"] == current else edge["source"]
                if other in names and other not in seen:
                    seen.add(other)
                    stack.append(other)
        remaining -= seen
    return components


def build_confirmation_candidate(
    graph: dict[str, Any],
    segments: dict[str, Any],
    labeled_names: list[str],
    *,
    seed: int,
    region_count: int,
    views_per_region: int,
    guard_radius: float,
    minimum_region_separation: float,
    minimum_rgb_train: int,
    minimum_semantic_train: int,
) -> dict[str, Any]:
    """Build a camera-only future split candidate with strict spatial guards."""
    labeled = set(labeled_names)
    nodes = {node["name"]: node for node in graph["nodes"]}
    all_names = set(nodes)
    windows = []
    for segment in segments["segments"]:
        names = [name for name in segment["images"] if name in labeled]
        for start in range(len(names) - views_per_region + 1):
            selected = names[start : start + views_per_region]
            centres = np.asarray([nodes[name]["center"] for name in selected])
            centroid = centres.mean(0)
            radius = float(np.linalg.norm(centres - centroid, axis=1).max())
            if radius <= 2.0:
                windows.append(
                    {
                        "segment_id": segment["segment_id"],
                        "start": start,
                        "images": selected,
                        "centroid": centroid,
                        "radius": radius,
                        "rank": _seeded_rank(seed, segment["segment_id"], str(start)),
                    }
                )
    windows.sort(key=lambda row: row["rank"])
    baseline_component_count = camera_graph_component_count(
        all_names, graph["adjacency"]
    )
    selected_windows: list[dict[str, Any]] = []
    for window in windows:
        if any(window["segment_id"] == old["segment_id"] for old in selected_windows):
            continue
        if any(
            np.linalg.norm(window["centroid"] - old["centroid"])
            < minimum_region_separation
            for old in selected_windows
        ):
            continue
        proposed = selected_windows + [window]
        if len(proposed) < region_count:
            selected_windows = proposed
            continue
        confirmation = {name for row in proposed for name in row["images"]}
        buffer = {
            name
            for name in all_names - confirmation
            if min(
                math.dist(nodes[name]["center"], nodes[target]["center"])
                for target in confirmation
            )
            < guard_radius
        }
        rgb_train = all_names - confirmation - buffer
        semantic_train = labeled - confirmation - buffer
        if (
            len(rgb_train) >= minimum_rgb_train
            and len(semantic_train) >= minimum_semantic_train
            and camera_graph_component_count(rgb_train, graph["adjacency"])
            <= baseline_component_count
        ):
            selected_windows = proposed
            break
    if len(selected_windows) != region_count:
        raise ValueError(
            "No camera-only confirmation partition satisfies the fixed feasibility gates"
        )
    confirmation = {name for row in selected_windows for name in row["images"]}
    buffer = {
        name
        for name in all_names - confirmation
        if min(
            math.dist(nodes[name]["center"], nodes[target]["center"])
            for target in confirmation
        )
        < guard_radius
    }
    rgb_train = all_names - confirmation - buffer
    semantic_train = labeled - confirmation - buffer
    train_edges = sum(
        edge["source"] in rgb_train and edge["target"] in rgb_train
        for edge in graph["edges"]
    )
    full_observations = sum(node["point3d_count"] for node in graph["nodes"])
    retained_observations = sum(
        node["point3d_count"] for node in graph["nodes"] if node["name"] in rgb_train
    )
    membership = segment_membership(segments)
    return {
        "schema_version": 1,
        "split_id": "sem395_spatial_confirmation_candidate_v1",
        "status": "candidate_only_not_executed",
        "selection_uses": "official camera centres, trajectory segments, and >=8-point camera graph only",
        "selection_uses_model_outputs_or_labels": False,
        "seed": seed,
        "guard_radius_strict_less_than": guard_radius,
        "confirmation_regions": [
            {
                "region_id": f"region-{index + 1}",
                "segment_id": row["segment_id"],
                "images": row["images"],
                "centroid": row["centroid"].tolist(),
                "maximum_radius": row["radius"],
            }
            for index, row in enumerate(selected_windows)
        ],
        "confirmation_labeled": sorted(confirmation),
        "buffer_excluded_all_cameras": sorted(buffer),
        "buffer_excluded_labeled": sorted(buffer & labeled),
        "buffer_excluded_unlabeled": sorted(buffer - labeled),
        "train_rgb_400": sorted(rgb_train),
        "train_rgb_300": sorted(semantic_train),
        "checks": {
            "disjoint_roles": not (
                confirmation & buffer or confirmation & rgb_train or buffer & rgb_train
            ),
            "all_400_accounted": len(confirmation | buffer | rgb_train) == 400,
            "full_camera_graph_component_count": baseline_component_count,
            "rgb_train_camera_graph_component_count": camera_graph_component_count(
                rgb_train, graph["adjacency"]
            ),
            "camera_graph_component_count_not_increased": camera_graph_component_count(
                rgb_train, graph["adjacency"]
            )
            <= baseline_component_count,
            "minimum_rgb_train_gate": len(rgb_train) >= minimum_rgb_train,
            "minimum_semantic_train_gate": len(semantic_train)
            >= minimum_semantic_train,
            "retained_camera_graph_edges": int(train_edges),
            "retained_camera_graph_edge_fraction": float(
                train_edges / len(graph["edges"])
            ),
            "retained_point3d_observation_count_proxy": int(retained_observations),
            "retained_point3d_observation_fraction_proxy": float(
                retained_observations / full_observations
            ),
            "retained_trajectory_segment_count": len(
                {membership[name] for name in rgb_train}
            ),
        },
        "counts": {
            "confirmation": len(confirmation),
            "buffer_all": len(buffer),
            "buffer_labeled": len(buffer & labeled),
            "buffer_unlabeled": len(buffer - labeled),
            "rgb_train": len(rgb_train),
            "semantic_train": len(semantic_train),
        },
        "future_requirement": "Rebuild RGB initialization and all RGB/semantic models from scratch using only train roles; never use confirmation or buffer RGB, labels, tracks, point colours, features, or pseudo-labels before the final once-only evaluation.",
    }
