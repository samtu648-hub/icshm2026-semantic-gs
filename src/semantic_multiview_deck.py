"""Deployable multi-view deck geometry utilities for SEM-200.

The module turns labelled deck pixels from *other* training cameras into a
world-space surface proposal using a frozen renderer depth map.  A proposal is
accepted only when it is visible in confirming source cameras and in the
target camera according to the same frozen renderer depth convention.

No target label is accepted by any prediction function in this module.  This
keeps OOF evidence generation mechanically separate from later evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import cv2
import numpy as np

from src.geometry_prior.camera_io import (
    ColmapCamera,
    ColmapImage,
    backproject_pixels,
    project_world_points,
)


@dataclass(frozen=True)
class DepthObservation:
    """Frozen camera-space depth plus exact rasterizer foreground support."""

    depth: np.ndarray
    foreground_mass: np.ndarray

    def __post_init__(self) -> None:
        if self.depth.ndim != 2 or self.depth.shape != self.foreground_mass.shape:
            raise ValueError("depth and foreground_mass must be same-shaped HxW arrays")

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.depth.shape[0]), int(self.depth.shape[1])

    def valid(self, minimum_foreground_mass: float) -> np.ndarray:
        if not 0.0 <= float(minimum_foreground_mass) <= 1.0:
            raise ValueError("minimum_foreground_mass must be in [0, 1]")
        return (
            np.isfinite(self.depth)
            & (self.depth > 0.0)
            & np.isfinite(self.foreground_mass)
            & (self.foreground_mass >= float(minimum_foreground_mass))
        )


def depth_consistent(
    projected_depth: np.ndarray,
    observed_depth: np.ndarray,
    *,
    relative_tolerance: float,
) -> np.ndarray:
    """Return the symmetric relative camera-depth agreement predicate."""
    if not 0.0 < float(relative_tolerance) < 1.0:
        raise ValueError("relative_tolerance must be in (0, 1)")
    projected = np.asarray(projected_depth, dtype=np.float64)
    observed = np.asarray(observed_depth, dtype=np.float64)
    if projected.shape != observed.shape:
        raise ValueError("projected and observed depth shapes differ")
    denominator = np.maximum(np.maximum(projected, observed), 1e-8)
    return (
        np.isfinite(projected)
        & np.isfinite(observed)
        & (projected > 0.0)
        & (observed > 0.0)
        & (np.abs(projected - observed) / denominator <= float(relative_tolerance))
    )


def _sample_projection(
    points_world: np.ndarray,
    *,
    camera: ColmapCamera,
    image: ColmapImage,
    observation: DepthObservation,
    minimum_foreground_mass: float,
    relative_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project points and return (visible, x, y) in the target pixel grid."""
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    if not len(points):
        empty = np.empty(0, dtype=np.int64)
        return np.empty(0, dtype=bool), empty, empty
    pixels, projected_depth = project_world_points(points, image, camera)
    x = np.rint(pixels[:, 0]).astype(np.int64)
    y = np.rint(pixels[:, 1]).astype(np.int64)
    height, width = observation.shape
    inside = (
        (projected_depth > 0.0)
        & (x >= 0)
        & (x < width)
        & (y >= 0)
        & (y < height)
    )
    sampled_depth = np.full(len(points), np.nan, dtype=np.float64)
    sampled_foreground = np.zeros(len(points), dtype=np.float32)
    indices = np.flatnonzero(inside)
    sampled_depth[indices] = observation.depth[y[indices], x[indices]]
    sampled_foreground[indices] = observation.foreground_mass[y[indices], x[indices]]
    visible = inside & (sampled_foreground >= float(minimum_foreground_mass))
    visible &= depth_consistent(
        projected_depth,
        sampled_depth,
        relative_tolerance=relative_tolerance,
    )
    return visible, x, y


def deck_points_from_view(
    deck_mask: np.ndarray,
    observation: DepthObservation,
    *,
    camera: ColmapCamera,
    image: ColmapImage,
    stride: int,
    minimum_foreground_mass: float,
) -> np.ndarray:
    """Backproject a deterministic strided sample of one labelled deck mask."""
    mask = np.asarray(deck_mask, dtype=bool)
    if mask.shape != observation.shape:
        raise ValueError("deck mask and depth observation shapes differ")
    if int(stride) < 1:
        raise ValueError("stride must be positive")
    sample = np.zeros(mask.shape, dtype=bool)
    sample[:: int(stride), :: int(stride)] = True
    selected = mask & sample & observation.valid(minimum_foreground_mass)
    y, x = np.nonzero(selected)
    if not len(x):
        return np.empty((0, 3), dtype=np.float64)
    pixels = np.column_stack((x, y)).astype(np.float64)
    return backproject_pixels(
        pixels,
        observation.depth[y, x].astype(np.float64),
        camera,
        image,
    )


def confirm_deck_points(
    points_world: np.ndarray,
    confirmations: Iterable[
        tuple[np.ndarray, DepthObservation, ColmapCamera, ColmapImage]
    ],
    *,
    minimum_confirming_views: int,
    minimum_foreground_mass: float,
    relative_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep points visible on deck in enough independent source cameras."""
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    if int(minimum_confirming_views) < 0:
        raise ValueError("minimum_confirming_views cannot be negative")
    support = np.zeros(len(points), dtype=np.uint16)
    confirmation_count = 0
    for deck_mask, observation, camera, image in confirmations:
        mask = np.asarray(deck_mask, dtype=bool)
        if mask.shape != observation.shape:
            raise ValueError("confirmation deck mask and depth shape differ")
        visible, x, y = _sample_projection(
            points,
            camera=camera,
            image=image,
            observation=observation,
            minimum_foreground_mass=minimum_foreground_mass,
            relative_tolerance=relative_tolerance,
        )
        indices = np.flatnonzero(visible)
        agreed = np.zeros(len(points), dtype=bool)
        agreed[indices] = mask[y[indices], x[indices]]
        support += agreed.astype(np.uint16)
        confirmation_count += 1
    if int(minimum_confirming_views) > confirmation_count:
        return np.zeros(len(points), dtype=bool), support
    return support >= int(minimum_confirming_views), support


def project_points_to_target_mask(
    points_world: np.ndarray,
    *,
    target_observation: DepthObservation,
    target_camera: ColmapCamera,
    target_image: ColmapImage,
    minimum_foreground_mass: float,
    relative_tolerance: float,
    splat_radius: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Project one source camera's stable points as one binary target vote."""
    if int(splat_radius) < 0:
        raise ValueError("splat_radius cannot be negative")
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    visible, x, y = _sample_projection(
        points,
        camera=target_camera,
        image=target_image,
        observation=target_observation,
        minimum_foreground_mass=minimum_foreground_mass,
        relative_tolerance=relative_tolerance,
    )
    mask = np.zeros(target_observation.shape, dtype=np.uint8)
    indices = np.flatnonzero(visible)
    mask[y[indices], x[indices]] = 1
    if int(splat_radius) > 0 and mask.any():
        diameter = 2 * int(splat_radius) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
        mask = cv2.dilate(mask, kernel, iterations=1)
        # Dilation must not escape the renderer-supported target surface.
        mask &= target_observation.valid(minimum_foreground_mass).astype(np.uint8)
    return mask.astype(bool), {
        "input_points": int(len(points)),
        "visible_points": int(indices.size),
        "vote_pixels": int(mask.sum()),
    }


def strong_neighbours(
    graph: dict[str, Any],
    name: str,
    allowed: set[str],
    *,
    count: int,
) -> list[str]:
    """Select deterministic highest-covisibility neighbours from an allowlist."""
    if int(count) < 1:
        raise ValueError("count must be positive")
    ranked: list[tuple[int, float, str]] = []
    for edge in graph.get("adjacency", {}).get(name, []):
        target = edge.get("target") if edge.get("source") == name else edge.get("source")
        if target in allowed and target != name:
            ranked.append(
                (
                    -int(edge.get("shared_point3d", 0)),
                    float(edge.get("center_distance", float("inf"))),
                    str(target),
                )
            )
    ranked.sort()
    return [target for _negative_shared, _distance, target in ranked[: int(count)]]


def accumulate_source_votes(source_masks: Iterable[np.ndarray]) -> np.ndarray:
    """Count at most one vote per source camera and target pixel."""
    masks = [np.asarray(mask, dtype=bool) for mask in source_masks]
    if not masks:
        raise ValueError("at least one source mask is required")
    shape = masks[0].shape
    if any(mask.shape != shape for mask in masks):
        raise ValueError("source vote masks have inconsistent shapes")
    votes = np.zeros(shape, dtype=np.uint16)
    for mask in masks:
        votes += mask.astype(np.uint16)
    return votes


def add_deck_with_protection(
    baseline: np.ndarray,
    eligible: np.ndarray,
    votes: np.ndarray,
    *,
    minimum_source_votes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply only background-to-deck additions inside a frozen eligibility mask."""
    base = np.asarray(baseline)
    allowed = np.asarray(eligible, dtype=bool)
    support = np.asarray(votes)
    if base.dtype != np.uint8 or base.ndim != 2:
        raise ValueError("baseline must be a uint8 HxW class map")
    if allowed.shape != base.shape or support.shape != base.shape:
        raise ValueError("eligibility/votes shape differs from baseline")
    if int(minimum_source_votes) < 1:
        raise ValueError("minimum_source_votes must be positive")
    additions = (base == 0) & allowed & (support >= int(minimum_source_votes))
    result = base.copy()
    result[additions] = 1
    if not np.array_equal(result[~additions], base[~additions]):
        raise RuntimeError("multi-view deck fusion changed a protected pixel")
    return result, additions
