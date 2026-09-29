"""Pure geometry and evidence helpers for the SEM-401 responsibility diagnostic."""
from __future__ import annotations

import numpy as np


def normalized_quaternion_rotations(quaternions: np.ndarray) -> np.ndarray:
    """Convert scalar-first quaternions to proper 3x3 rotation matrices."""
    values = np.asarray(quaternions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("quaternions must have shape [N,4]")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("quaternions must be nonzero")
    w, x, y, z = (values / norms).T
    rotations = np.empty((len(values), 3, 3), dtype=np.float64)
    rotations[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rotations[:, 0, 1] = 2 * (x * y - z * w)
    rotations[:, 0, 2] = 2 * (x * z + y * w)
    rotations[:, 1, 0] = 2 * (x * y + z * w)
    rotations[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rotations[:, 1, 2] = 2 * (y * z - x * w)
    rotations[:, 2, 0] = 2 * (x * z - y * w)
    rotations[:, 2, 1] = 2 * (y * z + x * w)
    rotations[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rotations


def symmetric_ellipsoid_children(
    xyz: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    world_lines: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return plus/minus one-sigma ellipsoid intersections and their radii."""
    xyz = np.asarray(xyz, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    lines = np.asarray(world_lines, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape [N,3]")
    if scales.shape != xyz.shape or lines.shape != xyz.shape:
        raise ValueError("scales and world_lines must match xyz")
    if rotations.shape != (len(xyz), 3, 3):
        raise ValueError("rotations must have shape [N,3,3]")
    if np.any(scales <= 0):
        raise ValueError("scales must be positive")
    norms = np.linalg.norm(lines, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("world lines must be nonzero")
    lines = lines / norms
    local = np.einsum("nij,nj->ni", np.swapaxes(rotations, 1, 2), lines)
    radius = 1.0 / np.sqrt(np.sum((local / scales) ** 2, axis=1))
    offset = radius[:, None] * lines
    return xyz + offset, xyz - offset, radius


def project_world_points(
    xyz: np.ndarray,
    camera_rotation: np.ndarray,
    camera_translation: np.ndarray,
    camera_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points with the COLMAP camera-from-world convention."""
    xyz = np.asarray(xyz, dtype=np.float64)
    rotation = np.asarray(camera_rotation, dtype=np.float64)
    translation = np.asarray(camera_translation, dtype=np.float64)
    matrix = np.asarray(camera_matrix, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape [N,3]")
    if rotation.shape != (3, 3) or translation.shape != (3,) or matrix.shape != (3, 3):
        raise ValueError("invalid camera shapes")
    camera_xyz = xyz @ rotation.T + translation
    depth = camera_xyz[:, 2]
    homogeneous = camera_xyz @ matrix.T
    pixels = homogeneous[:, :2] / depth[:, None]
    return pixels, depth


def jensen_shannon_rows(left_counts: np.ndarray, right_counts: np.ndarray) -> np.ndarray:
    """Natural-log Jensen-Shannon divergence for paired count rows."""
    left = np.asarray(left_counts, dtype=np.float64)
    right = np.asarray(right_counts, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("count arrays must have equal [N,C] shape")
    if np.any(left < 0) or np.any(right < 0):
        raise ValueError("counts must be non-negative")
    left_sum = left.sum(axis=1, keepdims=True)
    right_sum = right.sum(axis=1, keepdims=True)
    if np.any(left_sum == 0) or np.any(right_sum == 0):
        raise ValueError("every row needs evidence on both sides")
    left = left / left_sum
    right = right / right_sum
    mixture = 0.5 * (left + right)

    def kl(probability: np.ndarray) -> np.ndarray:
        terms = np.zeros_like(probability)
        active = probability > 0
        terms[active] = probability[active] * np.log(probability[active] / mixture[active])
        return terms.sum(axis=1)

    return 0.5 * (kl(left) + kl(right))
