"""Canonical-depth float64 audit for preserved VGGT-Omega P0 outputs.

The original GEO-025-P0 gate projected the serialized float32 ``world_points``
array.  Fusion does not consume that redundant array: it reconstructs P0 points
from ``depth`` plus the float64 official intrinsics and pose.  This module keeps
the serialized array as a fidelity diagnostic while making that downstream
canonical representation the coordinate hard gate for a separately registered
experiment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_ARRAYS = {
    "depth",
    "confidence",
    "valid_mask",
    "world_points",
    "pinhole_intrinsics",
    "official_cam2world",
    "depth_scale_ratio",
}
EXPECTED_DTYPES = {
    "depth": np.dtype(np.float32),
    "confidence": np.dtype(np.float32),
    "valid_mask": np.dtype(np.uint8),
    "world_points": np.dtype(np.float32),
    "pinhole_intrinsics": np.dtype(np.float64),
    "official_cam2world": np.dtype(np.float64),
    "depth_scale_ratio": np.dtype(np.float64),
}
MINIMUM_DEPTH = 1e-6
CANONICAL_PIXEL_LIMIT = 1e-7
CANONICAL_DEPTH_ATOL = 1e-10
CANONICAL_DEPTH_RTOL = 1e-10
STORED_WORLD_ATOL = 2e-5
STORED_WORLD_RTOL = 2e-5
HISTORICAL_STORED_PIXEL_LIMIT = 1e-3


def canonical_world_points_float64(
    depth: np.ndarray, intrinsics: np.ndarray, cam2world: np.ndarray
) -> np.ndarray:
    """Backproject the canonical depth representation without float32 storage."""
    depth64 = np.asarray(depth, dtype=np.float64)
    matrix = np.asarray(intrinsics, dtype=np.float64)
    pose = np.asarray(cam2world, dtype=np.float64)
    height, width = depth64.shape
    columns, rows = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
        indexing="xy",
    )
    camera = np.stack(
        (
            (columns - matrix[0, 2]) * depth64 / matrix[0, 0],
            (rows - matrix[1, 2]) * depth64 / matrix[1, 1],
            depth64,
        ),
        axis=-1,
    )
    return camera @ pose[:3, :3].T + pose[:3, 3]


def project_world_float64(
    world: np.ndarray, intrinsics: np.ndarray, cam2world: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project official-frame points with the same row-vector convention."""
    points = np.asarray(world, dtype=np.float64)
    matrix = np.asarray(intrinsics, dtype=np.float64)
    pose = np.asarray(cam2world, dtype=np.float64)
    rotation = pose[:3, :3]
    camera = (points - pose[:3, 3]) @ rotation
    x = matrix[0, 0] * camera[..., 0] / camera[..., 2] + matrix[0, 2]
    y = matrix[1, 1] * camera[..., 1] / camera[..., 2] + matrix[1, 2]
    return x, y, camera[..., 2]


def audit_npz_float64(
    path: Path, *, expected_resolution_wh: tuple[int, int]
) -> dict[str, Any]:
    """Audit one preserved archive under the preregistered GEO-026 contract."""
    if not path.is_file():
        raise FileNotFoundError(f"Geometry NPZ does not exist: {path}")
    with np.load(path, allow_pickle=False) as archive:
        names = set(archive.files)
        missing = REQUIRED_ARRAYS - names
        unexpected = names - (REQUIRED_ARRAYS | {"pred_cam2world"})
        if missing or unexpected:
            raise ValueError(
                f"{path}: schema mismatch missing={sorted(missing)} unexpected={sorted(unexpected)}"
            )
        arrays = {name: archive[name] for name in archive.files}

    wrong_dtypes = {
        name: str(arrays[name].dtype)
        for name, expected in EXPECTED_DTYPES.items()
        if arrays[name].dtype != expected
    }
    if "pred_cam2world" in arrays and arrays["pred_cam2world"].dtype != np.float64:
        wrong_dtypes["pred_cam2world"] = str(arrays["pred_cam2world"].dtype)
    if wrong_dtypes:
        raise ValueError(f"{path}: wrong array dtypes {wrong_dtypes}")

    depth = arrays["depth"]
    confidence = arrays["confidence"]
    valid_mask = arrays["valid_mask"]
    stored_world = arrays["world_points"]
    intrinsics = arrays["pinhole_intrinsics"]
    cam2world = arrays["official_cam2world"]
    scale = arrays["depth_scale_ratio"]
    expected_shape = (expected_resolution_wh[1], expected_resolution_wh[0])
    if depth.shape != expected_shape:
        raise ValueError(f"{path}: depth shape {depth.shape} != {expected_shape}")
    if confidence.shape != depth.shape or valid_mask.shape != depth.shape:
        raise ValueError(f"{path}: depth/confidence/valid_mask shapes differ")
    if stored_world.shape != depth.shape + (3,):
        raise ValueError(f"{path}: world_points shape is invalid")
    if intrinsics.shape != (3, 3) or cam2world.shape != (4, 4) or scale.shape != ():
        raise ValueError(f"{path}: camera or scale shape is invalid")
    if "pred_cam2world" in arrays and arrays["pred_cam2world"].shape != (4, 4):
        raise ValueError(f"{path}: pred_cam2world shape is invalid")
    if not np.isin(valid_mask, (0, 1)).all():
        raise ValueError(f"{path}: valid_mask is not binary")
    expected_valid = (
        np.isfinite(depth)
        & (depth > MINIMUM_DEPTH)
        & np.isfinite(confidence)
    )
    valid = valid_mask.astype(bool)
    if not np.array_equal(valid, expected_valid) or not valid.any():
        raise ValueError(f"{path}: valid_mask does not match the fixed rule")
    for name in ("depth", "confidence", "world_points", "pinhole_intrinsics", "official_cam2world"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"{path}: {name} contains nonfinite values")
    if "pred_cam2world" in arrays and not np.isfinite(arrays["pred_cam2world"]).all():
        raise ValueError(f"{path}: pred_cam2world contains nonfinite values")
    if not np.isfinite(float(scale)) or float(scale) <= 0:
        raise ValueError(f"{path}: depth_scale_ratio must be finite and positive")
    if (
        intrinsics[0, 0] <= 0
        or intrinsics[1, 1] <= 0
        or not np.allclose(intrinsics[2], [0.0, 0.0, 1.0])
        or not np.isclose(intrinsics[0, 1], 0.0)
        or not np.isclose(intrinsics[1, 0], 0.0)
    ):
        raise ValueError(f"{path}: invalid PINHOLE intrinsics")
    rotation = cam2world[:3, :3]
    if (
        not np.allclose(cam2world[3], [0.0, 0.0, 0.0, 1.0])
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
    ):
        raise ValueError(f"{path}: invalid official cam2world")

    canonical_world = canonical_world_points_float64(depth, intrinsics, cam2world)
    stored_error = np.linalg.norm(
        stored_world.astype(np.float64) - canonical_world, axis=2
    )
    if not np.allclose(
        stored_world.astype(np.float64),
        canonical_world,
        rtol=STORED_WORLD_RTOL,
        atol=STORED_WORLD_ATOL,
    ):
        raise ValueError(f"{path}: stored world_points exceed the fixed fidelity bound")

    rows, columns = np.nonzero(valid)
    canonical_x, canonical_y, canonical_z = project_world_float64(
        canonical_world[valid], intrinsics, cam2world
    )
    canonical_pixel_error = np.maximum(
        np.abs(canonical_x - columns), np.abs(canonical_y - rows)
    )
    canonical_depth_error = np.abs(canonical_z - depth[valid].astype(np.float64))
    canonical_pixel_max = float(canonical_pixel_error.max(initial=0.0))
    canonical_depth_max = float(canonical_depth_error.max(initial=0.0))
    if canonical_pixel_max > CANONICAL_PIXEL_LIMIT or not np.allclose(
        canonical_z,
        depth[valid].astype(np.float64),
        rtol=CANONICAL_DEPTH_RTOL,
        atol=CANONICAL_DEPTH_ATOL,
    ):
        raise ValueError(f"{path}: canonical float64 P0 roundtrip failed")

    stored_x, stored_y, stored_z = project_world_float64(
        stored_world[valid], intrinsics, cam2world
    )
    stored_pixel_error = np.maximum(
        np.abs(stored_x - columns), np.abs(stored_y - rows)
    )
    stored_depth_error = np.abs(stored_z - depth[valid].astype(np.float64))
    return {
        "valid_count": int(valid.sum()),
        "valid_fraction": float(valid.mean()),
        "minimum_valid_depth": float(depth[valid].min()),
        "canonical_pixel_error_max": canonical_pixel_max,
        "canonical_depth_error_max": canonical_depth_max,
        "stored_world_error_max": float(stored_error.max(initial=0.0)),
        "stored_pixel_error_max": float(stored_pixel_error.max(initial=0.0)),
        "stored_depth_error_max": float(stored_depth_error.max(initial=0.0)),
        "stored_pixels_over_historical_1e_3": int(
            (stored_pixel_error > HISTORICAL_STORED_PIXEL_LIMIT).sum()
        ),
    }
