"""Pure SEM-400 helpers for deterministic child displacement lines."""
from __future__ import annotations

import numpy as np


def deterministic_control_lines(count: int, seed: int) -> np.ndarray:
    if count < 0:
        raise ValueError("count must be non-negative")
    rng = np.random.default_rng(seed)
    lines = rng.normal(size=(count, 3))
    norms = np.linalg.norm(lines, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise RuntimeError("Generated a zero control direction")
    return lines / norms


def ellipsoid_line_radius(
    scales: np.ndarray, rotations: np.ndarray, world_lines: np.ndarray
) -> np.ndarray:
    """Distance from centre to the one-sigma ellipsoid along each world line."""
    scales = np.asarray(scales, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    lines = np.asarray(world_lines, dtype=np.float64)
    if scales.shape != lines.shape or scales.ndim != 2 or scales.shape[1] != 3:
        raise ValueError("scales and lines must both have shape [N,3]")
    if rotations.shape != (len(scales), 3, 3):
        raise ValueError("rotations must have shape [N,3,3]")
    if np.any(scales <= 0):
        raise ValueError("ellipsoid scales must be positive")
    lines = lines / np.linalg.norm(lines, axis=1, keepdims=True)
    local = np.einsum("nij,nj->ni", np.swapaxes(rotations, 1, 2), lines)
    return 1.0 / np.sqrt(np.sum((local / scales) ** 2, axis=1))
