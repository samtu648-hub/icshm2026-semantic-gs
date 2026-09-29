"""Official-camera adaptation shared by R1 candidates and the final submission path.

R1 trains in a full-frame PINHOLE image space. Blind-test cameras remain official COLMAP
SIMPLE_RADIAL CAMERA_FROM_WORLD records. This module converts camera poses between an official
and optional model frame, and maps renderings back to the official distorted pixel grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraFromWorld:
    """COLMAP world-to-camera pose x_c = R x_w + t."""

    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self) -> None:
        if self.rotation.shape != (3, 3) or self.translation.shape != (3,):
            raise ValueError("rotation must be 3x3 and translation must contain three values")

    @property
    def center(self) -> np.ndarray:
        return -self.rotation.T @ self.translation


@dataclass(frozen=True)
class Sim3:
    """Official-to-model transform x_model = scale * rotation * x_official + translation."""

    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError("Sim(3) scale must be positive")
        if self.rotation.shape != (3, 3) or self.translation.shape != (3,):
            raise ValueError("Sim(3) rotation must be 3x3 and translation must contain three values")
        if not np.allclose(self.rotation.T @ self.rotation, np.eye(3), atol=1e-6):
            raise ValueError("Sim(3) rotation must be orthonormal")
        if not np.isclose(np.linalg.det(self.rotation), 1.0, atol=1e-6):
            raise ValueError("Sim(3) rotation must be proper (determinant +1)")


@dataclass(frozen=True)
class SimpleRadialCamera:
    width: int
    height: int
    focal: float
    cx: float
    cy: float
    radial_k: float

    @property
    def camera_matrix(self) -> np.ndarray:
        return np.array(
            [[self.focal, 0.0, self.cx], [0.0, self.focal, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def distortion(self) -> np.ndarray:
        return np.array([self.radial_k, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)


def official_to_model_camera(official_camera: CameraFromWorld, official_to_model: Sim3) -> CameraFromWorld:
    """Convert an official CAMERA_FROM_WORLD pose into the model world frame.

    Given x_m = s Q x_o + d and x_c = R_o x_o + t_o, the model-frame pose is
    R_m = R_o Q^T and t_m = -R_m (s Q C_o + d), where C_o = -R_o^T t_o.
    """
    model_rotation = official_camera.rotation @ official_to_model.rotation.T
    model_center = (
        official_to_model.scale * official_to_model.rotation @ official_camera.center
        + official_to_model.translation
    )
    return CameraFromWorld(model_rotation, -model_rotation @ model_center)


def undistort_rgb(raw_rgb: np.ndarray, camera: SimpleRadialCamera) -> np.ndarray:
    """Return the R1 full-frame PINHOLE image using unchanged K and no crop."""
    _validate_image_size(raw_rgb, camera)
    return cv2.undistort(raw_rgb, camera.camera_matrix, camera.distortion, None, camera.camera_matrix)


def redistort_rgb(pinhole_rgb: np.ndarray, camera: SimpleRadialCamera) -> np.ndarray:
    """Map a PINHOLE RGB rendering into the official SIMPLE_RADIAL output grid."""
    return _redistort(pinhole_rgb, camera, interpolation=cv2.INTER_LINEAR)


def redistort_semantic(pinhole_labels: np.ndarray, camera: SimpleRadialCamera) -> np.ndarray:
    """Map a single-channel semantic rendering using nearest-neighbor interpolation only."""
    if pinhole_labels.ndim != 2:
        raise ValueError("Semantic labels must be a single-channel image")
    result = _redistort(pinhole_labels, camera, interpolation=cv2.INTER_NEAREST)
    if result.ndim != 2:
        raise ValueError("Semantic remapping unexpectedly changed the channel count")
    return result


def redistort_semantic_probabilities(pinhole_probabilities: np.ndarray, camera: SimpleRadialCamera) -> np.ndarray:
    """Map CxHxW semantic probabilities to the official grid and renormalize.

    The RGB submission adapter uses linear remapping, while class-ID PNGs use
    nearest neighbour.  A logits interface needs neither: remap probabilities
    linearly, normalize their five-class simplex, then take log probabilities
    at the caller.  This prevents interpolation from creating invalid scores.
    """
    if pinhole_probabilities.ndim != 3:
        raise ValueError("Semantic probabilities must have CxHxW shape")
    channels, height, width = pinhole_probabilities.shape
    if channels != 5:
        raise ValueError("SEM-001 expects exactly five semantic probability channels")
    _validate_image_size(np.moveaxis(pinhole_probabilities, 0, -1), camera)
    mapped = _redistort(np.moveaxis(pinhole_probabilities, 0, -1), camera, interpolation=cv2.INTER_LINEAR)
    mapped = np.moveaxis(np.asarray(mapped, dtype=np.float32), -1, 0)
    mapped = np.clip(mapped, 0.0, None)
    normalizer = np.maximum(mapped.sum(axis=0, keepdims=True), np.finfo(np.float32).tiny)
    return mapped / normalizer


def _validate_image_size(image: np.ndarray, camera: SimpleRadialCamera) -> None:
    if image.shape[0] != camera.height or image.shape[1] != camera.width:
        raise ValueError(
            f"Image size {image.shape[1]}x{image.shape[0]} does not match camera {camera.width}x{camera.height}"
        )


def _redistortion_map(camera: SimpleRadialCamera) -> tuple[np.ndarray, np.ndarray]:
    raw_grid = np.stack(
        np.meshgrid(np.arange(camera.width), np.arange(camera.height), indexing="xy"), axis=-1
    ).reshape(-1, 1, 2).astype(np.float64)
    pinhole_grid = cv2.undistortPoints(
        raw_grid, camera.camera_matrix, camera.distortion, P=camera.camera_matrix
    ).reshape(camera.height, camera.width, 2)
    return pinhole_grid[..., 0].astype(np.float32), pinhole_grid[..., 1].astype(np.float32)


def _redistort(image: np.ndarray, camera: SimpleRadialCamera, interpolation: int) -> np.ndarray:
    _validate_image_size(image, camera)
    map_x, map_y = _redistortion_map(camera)
    return cv2.remap(image, map_x, map_y, interpolation, borderMode=cv2.BORDER_CONSTANT)
