"""Continuous world-coordinate deck field used by SEM-210.

The field is deliberately small and view independent.  Labelled pixels are
lifted to the frozen RGB geometry, encoded with fixed Fourier features and
classified as deck/non-deck.  Novel-camera inference therefore needs only the
camera and the frozen renderer depth, never a target photograph.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.geometry_prior.camera_io import (
    ColmapCamera,
    ColmapImage,
    backproject_pixels,
)
from src.semantic_multiview_deck import DepthObservation


def surface_samples_from_view(
    labels: np.ndarray,
    observation: DepthObservation,
    *,
    camera: ColmapCamera,
    image: ColmapImage,
    stride: int,
    minimum_foreground_mass: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return frozen-surface world points and binary deck targets."""
    target = np.asarray(labels)
    if target.shape != observation.shape or target.ndim != 2:
        raise ValueError("labels and depth observation must be same-shaped HxW")
    if int(stride) < 1:
        raise ValueError("stride must be positive")
    sampled = np.zeros(target.shape, dtype=bool)
    sampled[:: int(stride), :: int(stride)] = True
    valid = sampled & observation.valid(minimum_foreground_mass)
    y, x = np.nonzero(valid)
    if not len(x):
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.uint8)
    points = backproject_pixels(
        np.column_stack((x, y)).astype(np.float64),
        observation.depth[y, x].astype(np.float64),
        camera,
        image,
    ).astype(np.float32)
    binary = (target[y, x] == 1).astype(np.uint8)
    return points, binary


def surface_points_from_depth(
    observation: DepthObservation,
    *,
    camera: ColmapCamera,
    image: ColmapImage,
    minimum_foreground_mass: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backproject every renderer-supported pixel without accepting a label."""
    valid = observation.valid(minimum_foreground_mass)
    y, x = np.nonzero(valid)
    if not len(x):
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )
    points = backproject_pixels(
        np.column_stack((x, y)).astype(np.float64),
        observation.depth[y, x].astype(np.float64),
        camera,
        image,
    ).astype(np.float32)
    return points, y.astype(np.int64), x.astype(np.int64)


def coordinate_normalization(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit a robust isotropic normalization using training points only."""
    value = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(value) < 2 or not np.isfinite(value).all():
        raise ValueError("normalization requires at least two finite 3D points")
    center = np.median(value, axis=0)
    distances = np.linalg.norm(value - center[None, :], axis=1)
    scale = float(np.quantile(distances, 0.99))
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError("coordinate normalization scale is degenerate")
    return center.astype(np.float32), scale


def normalize_coordinates(
    points: np.ndarray, center: np.ndarray, scale: float
) -> np.ndarray:
    value = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    origin = np.asarray(center, dtype=np.float32)
    if origin.shape != (3,) or not np.isfinite(origin).all() or not float(scale) > 0:
        raise ValueError("invalid coordinate normalization")
    return (value - origin[None, :]) / np.float32(scale)


def build_xyz_deck_field(*, fourier_levels: int, width: int, hidden_layers: int):
    """Construct the fixed Fourier-feature MLP without importing torch globally."""
    import torch

    if int(fourier_levels) < 1 or int(width) < 1 or int(hidden_layers) < 1:
        raise ValueError("field dimensions must be positive")

    class XYZDeckField(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fourier_levels = int(fourier_levels)
            input_width = 3 + 6 * self.fourier_levels
            layers: list[torch.nn.Module] = [
                torch.nn.Linear(input_width, int(width)),
                torch.nn.ReLU(inplace=True),
            ]
            for _ in range(int(hidden_layers) - 1):
                layers.extend(
                    [torch.nn.Linear(int(width), int(width)), torch.nn.ReLU(inplace=True)]
                )
            layers.append(torch.nn.Linear(int(width), 1))
            self.network = torch.nn.Sequential(*layers)

        def encode(self, coordinates):
            frequencies = (2.0 ** torch.arange(
                self.fourier_levels,
                dtype=coordinates.dtype,
                device=coordinates.device,
            )) * torch.pi
            angles = coordinates[..., None] * frequencies
            return torch.cat(
                [coordinates, torch.sin(angles).flatten(-2), torch.cos(angles).flatten(-2)],
                dim=-1,
            )

        def forward(self, coordinates):
            return self.network(self.encode(coordinates)).squeeze(-1)

    return XYZDeckField()


def balanced_batch_indices(
    positive_count: int,
    negative_count: int,
    *,
    batch_size: int,
    generator: Any,
    device: Any,
):
    """Return deterministic half-positive/half-negative local indices."""
    import torch

    if positive_count < 1 or negative_count < 1:
        raise ValueError("balanced sampling requires both classes")
    if int(batch_size) < 2 or int(batch_size) % 2:
        raise ValueError("batch_size must be an even integer >= 2")
    half = int(batch_size) // 2
    positive = torch.randint(positive_count, (half,), generator=generator, device=device)
    negative = torch.randint(negative_count, (half,), generator=generator, device=device)
    return positive, negative
