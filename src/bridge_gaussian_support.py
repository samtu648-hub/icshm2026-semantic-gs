"""Geometry and summary helpers for bridge-Gaussian support diagnostics.

These helpers operate on existing Gaussian centers and official cameras.  They
do not infer or create 3D structure from semantic masks.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np


def project_gaussians(
    xyz_world: np.ndarray,
    scales: np.ndarray,
    rotations_world: np.ndarray,
    camera: Any,
    *,
    sigma: float = 3.0,
) -> dict[str, np.ndarray]:
    """Project Gaussian centers and covariance ellipses with an official camera."""
    xyz = np.asarray(xyz_world, dtype=np.float64)
    scl = np.asarray(scales, dtype=np.float64)
    rot = np.asarray(rotations_world, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz_world must have shape [N,3]")
    if scl.shape != xyz.shape or rot.shape != (len(xyz), 3, 3):
        raise ValueError("scale/rotation shape mismatch")
    if not np.isfinite(xyz).all() or not np.isfinite(scl).all() or not np.isfinite(rot).all():
        raise FloatingPointError("nonfinite Gaussian geometry")
    if np.any(scl <= 0):
        raise ValueError("Gaussian scales must be positive")

    camera_rotation = np.asarray(camera.pose.rotation, dtype=np.float64)
    camera_translation = np.asarray(camera.pose.translation, dtype=np.float64)
    camera_xyz = xyz @ camera_rotation.T + camera_translation
    z = camera_xyz[:, 2]
    matrix = np.asarray(camera.matrix, dtype=np.float64)
    fx, fy, cx, cy = matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]
    safe_z = np.where(np.abs(z) > 1e-12, z, np.nan)
    u = fx * camera_xyz[:, 0] / safe_z + cx
    v = fy * camera_xyz[:, 1] / safe_z + cy

    # R diag(s^2) R^T; multiplying columns of R by s^2 avoids materializing
    # an N x 3 x 3 diagonal tensor.
    covariance_world = (rot * np.square(scl)[:, None, :]) @ np.swapaxes(rot, 1, 2)
    covariance_camera = camera_rotation[None] @ covariance_world @ camera_rotation.T[None]
    jacobian = np.zeros((len(xyz), 2, 3), dtype=np.float64)
    jacobian[:, 0, 0] = fx / safe_z
    jacobian[:, 0, 2] = -fx * camera_xyz[:, 0] / np.square(safe_z)
    jacobian[:, 1, 1] = fy / safe_z
    jacobian[:, 1, 2] = -fy * camera_xyz[:, 1] / np.square(safe_z)
    covariance_image = jacobian @ covariance_camera @ np.swapaxes(jacobian, 1, 2)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_image)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    minor = sigma * np.sqrt(eigenvalues[:, 0])
    major = sigma * np.sqrt(eigenvalues[:, 1])
    major_vector = eigenvectors[:, :, 1]
    angle = np.arctan2(major_vector[:, 1], major_vector[:, 0])
    valid = (
        (z > 0)
        & np.isfinite(u)
        & np.isfinite(v)
        & np.isfinite(major)
        & np.isfinite(minor)
    )
    return {
        "u": u,
        "v": v,
        "depth": z,
        "major_radius": major,
        "minor_radius": minor,
        "angle_rad": angle,
        "valid": valid,
    }


def undirected_angle_error(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the smallest angle between unoriented 2D axes, in radians."""
    delta = np.abs(np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64))
    delta = np.mod(delta, math.pi)
    return np.minimum(delta, math.pi - delta)


def quantiles(values: Iterable[float]) -> dict[str, float | None]:
    """JSON-safe fixed quantiles for finite values."""
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {key: None for key in ("min", "p10", "p25", "p50", "p75", "p90", "max")}
    levels = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    keys = ("min", "p10", "p25", "p50", "p75", "p90", "max")
    return {key: float(value) for key, value in zip(keys, np.quantile(array, levels))}


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize true-like, mixed/artifact-like, and ambiguous contributors."""
    output: dict[str, Any] = {"records": len(records)}
    for kind in ("true_like", "artifact_like", "ambiguous"):
        selected = [row for row in records if row["kind"] == kind]
        output[kind] = {
            "count": len(selected),
            "fraction": float(len(selected) / max(len(records), 1)),
            "opacity": quantiles(row["opacity"] for row in selected),
            "projected_major_radius_px": quantiles(row["major_radius_px"] for row in selected),
            "projected_minor_radius_px": quantiles(row["minor_radius_px"] for row in selected),
            "projected_area_proxy_px2": quantiles(row["area_proxy_px2"] for row in selected),
            "anisotropy": quantiles(row["anisotropy"] for row in selected),
            "near_background_fraction": quantiles(row["near_background_fraction"] for row in selected),
            "cross_view_top_count": quantiles(row["cross_view_top_count"] for row in selected),
            "orientation_error_deg": quantiles(row["orientation_error_deg"] for row in selected),
        }
    return output
