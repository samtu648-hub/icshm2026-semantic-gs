"""Similarity alignment helpers for geometry-prior proposals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Similarity3:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        return self.scale * (values @ self.rotation.T) + self.translation

    def inverse(self) -> "Similarity3":
        rotation = self.rotation.T
        scale = 1.0 / self.scale
        return Similarity3(scale, rotation, -scale * (self.translation @ rotation.T))


def umeyama_similarity(source: np.ndarray, target: np.ndarray, *, allow_scale: bool = True) -> Similarity3:
    """Fit ``target ~= s * R * source + t`` with a proper rotation."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or source.shape[0] < 3:
        raise ValueError("source and target must be matching Nx3 arrays with N>=3")
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    centered_source, centered_target = source - source_mean, target - target_mean
    covariance = centered_target.T @ centered_source / source.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vt
    variance = float(np.sum(centered_source * centered_source) / source.shape[0])
    if variance <= np.finfo(float).eps:
        raise ValueError("source points are degenerate for Sim(3) alignment")
    scale = float(np.sum(singular_values * np.diag(correction)) / variance) if allow_scale else 1.0
    translation = target_mean - scale * (rotation @ source_mean)
    return Similarity3(scale, rotation, translation)


def rotation_angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first) @ np.asarray(second).T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))
