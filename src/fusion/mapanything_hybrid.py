"""Deterministic MapAnything supplementation of an immutable Scaffold PLY.

This module deliberately does *not* replace or downsample the original DLT
initialization.  It selects only a small, high-confidence subset of the
already-audited MapAnything P0 fusion output for a pre-registered diagnostic.
"""

from __future__ import annotations

from typing import Any

import numpy as np


REQUIRED_MAP_ARRAYS = {
    "xyz",
    "rgb",
    "calibrated_confidence",
    "support_views",
    "line_structure",
    "source_image_index",
}


def select_hybrid_map_points(
    baseline_xyz: np.ndarray,
    map_points: dict[str, np.ndarray],
    *,
    r0: float,
    minimum_confidence: float,
    minimum_support: int,
    coverage_radius_r0: float,
    maximum_added_points: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Select extra Map points without changing a single baseline point.

    Map points must be ordinary (not the two-view thin-line exception), have
    sufficient confidence/support, and be farther than the recorded coverage
    radius from every original initialization point.  One highest-ranked point
    per ``r0`` voxel is retained before applying the explicit capacity cap.
    """
    from scipy.spatial import cKDTree

    missing = REQUIRED_MAP_ARRAYS.difference(map_points)
    if missing:
        raise ValueError(f"Map point archive is missing arrays: {sorted(missing)}")
    if baseline_xyz.ndim != 2 or baseline_xyz.shape[1] != 3 or not np.isfinite(baseline_xyz).all():
        raise ValueError("Baseline XYZ must be finite Nx3")
    if r0 <= 0 or not np.isfinite(r0):
        raise ValueError("r0 must be finite and positive")
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence must be in [0, 1]")
    if minimum_support < 1 or coverage_radius_r0 <= 0 or maximum_added_points < 1:
        raise ValueError("Invalid support, coverage radius, or capacity")

    xyz = np.asarray(map_points["xyz"], dtype=np.float32)
    rgb = np.asarray(map_points["rgb"], dtype=np.uint8)
    confidence = np.asarray(map_points["calibrated_confidence"], dtype=np.float32)
    support = np.asarray(map_points["support_views"], dtype=np.int16)
    line = np.asarray(map_points["line_structure"], dtype=bool)
    source = np.asarray(map_points["source_image_index"], dtype=np.int64)
    count = len(xyz)
    if xyz.shape != (count, 3) or rgb.shape != (count, 3):
        raise ValueError("Map XYZ and RGB must both be Nx3")
    if any(len(values) != count for values in (confidence, support, line, source)):
        raise ValueError("Map point arrays must have a common first dimension")

    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(confidence)
    quality = finite & (support >= minimum_support) & ~line & (confidence >= minimum_confidence)
    quality_indices = np.flatnonzero(quality)
    if not len(quality_indices):
        raise ValueError("No Map points pass the ordinary-surface quality gate")

    nearest_distance = cKDTree(baseline_xyz).query(xyz[quality_indices], k=1, workers=-1)[0]
    coverage = nearest_distance >= coverage_radius_r0 * r0
    coverage_indices = quality_indices[coverage]
    if not len(coverage_indices):
        raise ValueError("No Map points lie in the pre-registered low-coverage region")

    coverage_xyz = xyz[coverage_indices]
    keys = np.floor(coverage_xyz / r0).astype(np.int64)
    # Primary sort is voxel key, then descending confidence, then source and
    # original index.  This eliminates set/dict order and platform variance.
    voxel_order = np.lexsort(
        (
            coverage_indices,
            source[coverage_indices],
            -confidence[coverage_indices],
            keys[:, 2],
            keys[:, 1],
            keys[:, 0],
        )
    )
    ordered_keys = keys[voxel_order]
    first_per_voxel = np.ones(len(voxel_order), dtype=bool)
    first_per_voxel[1:] = np.any(ordered_keys[1:] != ordered_keys[:-1], axis=1)
    selected_indices = coverage_indices[voxel_order[first_per_voxel]]

    if len(selected_indices) > maximum_added_points:
        capacity_order = np.lexsort(
            (
                selected_indices,
                source[selected_indices],
                -confidence[selected_indices],
            )
        )
        selected_indices = selected_indices[capacity_order[:maximum_added_points]]

    selected = {
        name: (value if name == "source_images" else np.asarray(value)[selected_indices])
        for name, value in map_points.items()
    }
    details: dict[str, Any] = {
        "baseline_points": int(len(baseline_xyz)),
        "map_points_input": int(count),
        "quality_points": int(len(quality_indices)),
        "low_coverage_points_before_voxel": int(len(coverage_indices)),
        "low_coverage_points_after_voxel": int(first_per_voxel.sum()),
        "added_points": int(len(selected_indices)),
        "minimum_nearest_baseline_distance": float(nearest_distance[coverage].min()),
        "median_nearest_baseline_distance": float(np.median(nearest_distance[coverage])),
        "filters": {
            "minimum_confidence": float(minimum_confidence),
            "minimum_support": int(minimum_support),
            "line_structure_allowed": False,
            "coverage_radius_r0": float(coverage_radius_r0),
            "coverage_radius": float(coverage_radius_r0 * r0),
            "voxel_size": float(r0),
            "maximum_added_points": int(maximum_added_points),
        },
    }
    return selected, details


def merge_baseline_and_supplement(
    baseline_xyz: np.ndarray,
    baseline_rgb: np.ndarray,
    supplement: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Append selected Map points while preserving baseline byte-level values."""
    if baseline_rgb.shape != baseline_xyz.shape:
        raise ValueError("Baseline RGB must match baseline XYZ")
    xyz = np.concatenate((baseline_xyz, np.asarray(supplement["xyz"], dtype=np.float32)), axis=0)
    rgb = np.concatenate((baseline_rgb, np.asarray(supplement["rgb"], dtype=np.uint8)), axis=0)
    if not np.array_equal(xyz[: len(baseline_xyz)], baseline_xyz) or not np.array_equal(rgb[: len(baseline_rgb)], baseline_rgb):
        raise AssertionError("Baseline points were not preserved during hybrid merge")
    return xyz, rgb
