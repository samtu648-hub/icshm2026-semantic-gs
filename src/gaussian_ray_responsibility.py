"""Decode native 3DGS raster buffers and replay per-pixel contributions.

The helpers mirror the buffer layout and forward compositing equations of the
vendored diff-gaussian-rasterization source. They are diagnostic-only and do
not modify the renderer or model state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _take_aligned(buffer: Any, cursor: int, count: int, dtype: Any, alignment: int = 128):
    """Return a typed view at the renderer's absolute-address alignment."""
    if cursor < 0 or count < 0:
        raise ValueError("cursor and count must be non-negative")
    base = int(buffer.data_ptr())
    start = ((base + cursor + alignment - 1) & ~(alignment - 1)) - base
    element_size = int(dtype.itemsize if hasattr(dtype, "itemsize") else 0)
    if not element_size:
        # torch.dtype exposes element size through a zero-length tensor only.
        element_size = int(buffer.new_empty(0).view(dtype).element_size())
    end = start + count * element_size
    if end > int(buffer.numel()):
        raise ValueError("decoded field exceeds renderer buffer")
    return buffer[start:end].view(dtype), end


@dataclass(frozen=True)
class RasterBuffers:
    depths: Any
    means2d: Any
    conic_opacity: Any
    final_transmittance: Any
    n_contrib: Any
    ranges: Any
    point_list: Any


def decode_raster_buffers(torch: Any, geom: Any, binning: Any, image: Any, points: int, instances: int, pixels: int) -> RasterBuffers:
    """Decode fields needed to replay the native rasterizer's ray compositing."""
    cursor = 0
    depths, cursor = _take_aligned(geom, cursor, points, torch.float32)
    _, cursor = _take_aligned(geom, cursor, points * 3, torch.bool)
    _, cursor = _take_aligned(geom, cursor, points, torch.int32)
    means2d, cursor = _take_aligned(geom, cursor, points * 2, torch.float32)
    _, cursor = _take_aligned(geom, cursor, points * 6, torch.float32)
    conic_opacity, cursor = _take_aligned(geom, cursor, points * 4, torch.float32)
    # Remaining geometry fields are not needed for ray replay.

    cursor = 0
    point_list, cursor = _take_aligned(binning, cursor, instances, torch.int32)

    cursor = 0
    final_transmittance, cursor = _take_aligned(image, cursor, pixels, torch.float32)
    n_contrib, cursor = _take_aligned(image, cursor, pixels, torch.int32)
    ranges, cursor = _take_aligned(image, cursor, pixels * 2, torch.int32)
    return RasterBuffers(
        depths=depths,
        means2d=means2d.reshape(points, 2),
        conic_opacity=conic_opacity.reshape(points, 4),
        final_transmittance=final_transmittance,
        n_contrib=n_contrib,
        ranges=ranges.reshape(pixels, 2),
        point_list=point_list,
    )


@dataclass(frozen=True)
class PixelTrace:
    gaussian_ids: np.ndarray
    depths: np.ndarray
    weights: np.ndarray
    final_transmittance: float
    checked_entries: int


def replay_pixel(
    x: int,
    y: int,
    width: int,
    height: int,
    means2d: np.ndarray,
    conic_opacity: np.ndarray,
    depths: np.ndarray,
    point_list: np.ndarray,
    ranges: np.ndarray,
    n_contrib: np.ndarray,
    *,
    block_x: int = 16,
    block_y: int = 16,
    alpha_minimum: float = 1.0 / 255.0,
    transmittance_minimum: float = 1e-4,
) -> PixelTrace:
    """Replay exactly the accepted alpha-compositing events for one pixel."""
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("pixel lies outside image")
    horizontal_blocks = (width + block_x - 1) // block_x
    tile_id = (y // block_y) * horizontal_blocks + (x // block_x)
    start, tile_end = (int(value) for value in ranges[tile_id])
    checked = int(n_contrib[y * width + x])
    end = min(tile_end, start + checked)
    # Mirror CUDA float arithmetic closely; double-precision replay can cross
    # the native alpha/termination thresholds on rare boundary pixels.
    transmittance = np.float32(1.0)
    alpha_cutoff = np.float32(alpha_minimum)
    transmittance_cutoff = np.float32(transmittance_minimum)
    accepted_ids: list[int] = []
    accepted_depths: list[float] = []
    accepted_weights: list[float] = []
    for raw_id in point_list[start:end]:
        gaussian_id = int(raw_id)
        dx = np.float32(means2d[gaussian_id, 0]) - np.float32(x)
        dy = np.float32(means2d[gaussian_id, 1]) - np.float32(y)
        conic = conic_opacity[gaussian_id]
        power = np.float32(-0.5) * (
            np.float32(conic[0]) * dx * dx + np.float32(conic[2]) * dy * dy
        ) - np.float32(conic[1]) * dx * dy
        if power > np.float32(0.0):
            continue
        alpha = np.minimum(
            np.float32(0.99), np.float32(conic[3]) * np.exp(power, dtype=np.float32)
        )
        if alpha < alpha_cutoff:
            continue
        next_transmittance = transmittance * (np.float32(1.0) - alpha)
        if next_transmittance < transmittance_cutoff:
            break
        accepted_ids.append(gaussian_id)
        accepted_depths.append(float(depths[gaussian_id]))
        accepted_weights.append(float(alpha * transmittance))
        transmittance = next_transmittance
    return PixelTrace(
        gaussian_ids=np.asarray(accepted_ids, dtype=np.int64),
        depths=np.asarray(accepted_depths, dtype=np.float64),
        weights=np.asarray(accepted_weights, dtype=np.float64),
        final_transmittance=float(transmittance),
        checked_entries=checked,
    )


def trace_statistics(trace: PixelTrace) -> dict[str, float | int]:
    """Summarize concentration and depth spread without semantic labels."""
    mass = float(trace.weights.sum())
    if mass <= 0.0:
        return {
            "accepted_contributors": 0,
            "opacity_mass": 0.0,
            "top_share": 0.0,
            "effective_contributors": 0.0,
            "depth_mean": 0.0,
            "depth_std": 0.0,
            "prefix80_contributors": 0,
        }
    probability = trace.weights / mass
    cumulative = np.cumsum(trace.weights)
    prefix80 = int(np.searchsorted(cumulative, 0.8 * mass, side="left") + 1)
    mean = float(np.sum(probability * trace.depths))
    variance = float(np.sum(probability * (trace.depths - mean) ** 2))
    return {
        "accepted_contributors": int(len(trace.weights)),
        "opacity_mass": mass,
        "top_share": float(probability.max()),
        "effective_contributors": float(1.0 / np.sum(probability**2)),
        "depth_mean": mean,
        "depth_std": float(np.sqrt(max(variance, 0.0))),
        "prefix80_contributors": prefix80,
    }


@dataclass(frozen=True)
class ResponsibilityPrediction:
    standard_probability: np.ndarray
    surface_probability: np.ndarray
    tail_probability: np.ndarray | None
    surface_contributors: int
    surface_mass: float


@dataclass(frozen=True)
class DepthLayerSplit:
    """A geometry-defined split between two substantial ray layers."""

    eligible: bool
    clear: bool
    split_index: int | None
    front_mass_fraction: float
    back_mass_fraction: float
    depth_gap: float
    relative_depth_gap: float
    front_probability: np.ndarray | None
    back_probability: np.ndarray | None


@dataclass(frozen=True)
class DepthModeSplit:
    """A mass-constrained, variance-optimal two-mode partition of one ray."""

    eligible: bool
    clear: bool
    split_index: int | None
    front_mass_fraction: float
    back_mass_fraction: float
    front_depth_mean: float
    back_depth_mean: float
    explained_depth_variance: float
    front_probability: np.ndarray | None
    back_probability: np.ndarray | None


def split_depth_modes(
    trace: PixelTrace,
    gaussian_probability: np.ndarray,
    *,
    minimum_side_mass_fraction: float = 0.2,
    minimum_explained_depth_variance: float = 0.5,
) -> DepthModeSplit:
    """Find the ordered two-mode split maximizing explained depth variance.

    Unlike :func:`split_depth_layers`, this objective uses the full weighted
    depth distribution instead of one local adjacent gap. Labels and semantic
    probabilities do not influence the selected boundary or clarity test.
    """
    probability = np.asarray(gaussian_probability, dtype=np.float64)
    if probability.ndim != 2 or probability.shape[1] < 2:
        raise ValueError("gaussian_probability must have shape [N, C]")
    if not 0.0 < minimum_side_mass_fraction <= 0.5:
        raise ValueError("minimum_side_mass_fraction must lie in (0, 0.5]")
    if not 0.0 <= minimum_explained_depth_variance <= 1.0:
        raise ValueError("minimum_explained_depth_variance must lie in [0, 1]")
    if len(trace.gaussian_ids) != len(trace.weights) or len(trace.depths) != len(trace.weights):
        raise ValueError("trace arrays differ in length")

    empty = DepthModeSplit(False, False, None, 0.0, 0.0, 0.0, 0.0, 0.0, None, None)
    if len(trace.weights) < 2:
        return empty
    if np.any(np.diff(trace.depths) < -1e-6):
        raise ValueError("trace depths must be front-to-back ordered")
    total_mass = float(trace.weights.sum())
    if total_mass <= np.finfo(np.float64).tiny:
        return empty

    cumulative_mass = np.cumsum(trace.weights)
    cumulative_depth = np.cumsum(trace.weights * trace.depths)
    front_fraction = cumulative_mass[:-1] / total_mass
    fraction_tolerance = 16.0 * np.finfo(np.float64).eps
    admissible = np.flatnonzero(
        (front_fraction + fraction_tolerance >= minimum_side_mass_fraction)
        & ((1.0 - front_fraction) + fraction_tolerance >= minimum_side_mass_fraction)
    )
    if not len(admissible):
        return empty

    global_mean = float(cumulative_depth[-1] / total_mass)
    total_variance_mass = float(np.sum(trace.weights * (trace.depths - global_mean) ** 2))
    scores = []
    means: list[tuple[float, float]] = []
    for boundary in admissible:
        front_mass = float(cumulative_mass[boundary])
        back_mass = total_mass - front_mass
        front_mean = float(cumulative_depth[boundary] / front_mass)
        back_mean = float((cumulative_depth[-1] - cumulative_depth[boundary]) / back_mass)
        between = front_mass * (front_mean - global_mean) ** 2 + back_mass * (back_mean - global_mean) ** 2
        score = between / total_variance_mass if total_variance_mass > np.finfo(np.float64).tiny else 0.0
        scores.append(float(min(max(score, 0.0), 1.0)))
        means.append((front_mean, back_mean))
    choice = int(np.argmax(np.asarray(scores)))
    boundary = int(admissible[choice])
    split_index = boundary + 1
    front_mass = float(cumulative_mass[boundary])
    back_mass = total_mass - front_mass
    selected = probability[trace.gaussian_ids]
    front_probability = trace.weights[:split_index] @ selected[:split_index] / front_mass
    back_probability = trace.weights[split_index:] @ selected[split_index:] / back_mass
    score = scores[choice]
    front_mean, back_mean = means[choice]
    return DepthModeSplit(
        eligible=True,
        clear=score >= minimum_explained_depth_variance,
        split_index=split_index,
        front_mass_fraction=front_mass / total_mass,
        back_mass_fraction=back_mass / total_mass,
        front_depth_mean=front_mean,
        back_depth_mean=back_mean,
        explained_depth_variance=score,
        front_probability=front_probability,
        back_probability=back_probability,
    )


def split_depth_layers(
    trace: PixelTrace,
    gaussian_probability: np.ndarray,
    *,
    minimum_side_mass_fraction: float = 0.2,
    minimum_relative_depth_gap: float = 0.01,
) -> DepthLayerSplit:
    """Split a ray at its largest admissible adjacent depth discontinuity.

    A split is admissible only when each side owns at least the fixed fraction
    of accepted alpha-transmittance mass. The split is ``clear`` when its depth
    gap, normalized by the ray's contribution-weighted mean depth, reaches the
    fixed minimum. Labels are deliberately absent from this calculation.
    """
    probability = np.asarray(gaussian_probability, dtype=np.float64)
    if probability.ndim != 2 or probability.shape[1] < 2:
        raise ValueError("gaussian_probability must have shape [N, C]")
    if not 0.0 < minimum_side_mass_fraction <= 0.5:
        raise ValueError("minimum_side_mass_fraction must lie in (0, 0.5]")
    if minimum_relative_depth_gap < 0.0:
        raise ValueError("minimum_relative_depth_gap must be non-negative")
    if len(trace.gaussian_ids) != len(trace.weights) or len(trace.depths) != len(trace.weights):
        raise ValueError("trace arrays differ in length")

    empty = DepthLayerSplit(False, False, None, 0.0, 0.0, 0.0, 0.0, None, None)
    if len(trace.weights) < 2:
        return empty
    if np.any(np.diff(trace.depths) < -1e-6):
        raise ValueError("trace depths must be front-to-back ordered")
    total_mass = float(trace.weights.sum())
    if total_mass <= np.finfo(np.float64).tiny:
        return empty

    cumulative = np.cumsum(trace.weights)
    front_fraction = cumulative[:-1] / total_mass
    fraction_tolerance = 16.0 * np.finfo(np.float64).eps
    admissible = np.flatnonzero(
        (front_fraction + fraction_tolerance >= minimum_side_mass_fraction)
        & ((1.0 - front_fraction) + fraction_tolerance >= minimum_side_mass_fraction)
    )
    if not len(admissible):
        return empty

    gaps = trace.depths[1:] - trace.depths[:-1]
    # np.argmax is stable: equal gaps retain the nearest/front-most split.
    boundary = int(admissible[int(np.argmax(gaps[admissible]))])
    split_index = boundary + 1
    front_mass = float(cumulative[boundary])
    back_mass = total_mass - front_mass
    depth_mean = float(np.sum(trace.weights * trace.depths) / total_mass)
    gap = float(gaps[boundary])
    relative_gap = gap / max(abs(depth_mean), np.finfo(np.float64).tiny)
    selected = probability[trace.gaussian_ids]
    front_probability = trace.weights[:split_index] @ selected[:split_index] / front_mass
    back_probability = trace.weights[split_index:] @ selected[split_index:] / back_mass
    return DepthLayerSplit(
        eligible=True,
        clear=relative_gap >= minimum_relative_depth_gap,
        split_index=split_index,
        front_mass_fraction=front_mass / total_mass,
        back_mass_fraction=back_mass / total_mass,
        depth_gap=gap,
        relative_depth_gap=relative_gap,
        front_probability=front_probability,
        back_probability=back_probability,
    )


def responsibility_prediction(
    trace: PixelTrace,
    gaussian_probability: np.ndarray,
    *,
    surface_mass_fraction: float = 0.8,
    background_class: int = 0,
) -> ResponsibilityPrediction:
    """Compare native semantic composition with a front surface prefix.

    The surface prefix is the smallest front-to-back set reaching the fixed
    fraction of accepted non-background contribution mass. Its semantic
    distribution is renormalized within the prefix; this is a training-side
    responsibility diagnostic, not a changed production renderer.
    """
    probability = np.asarray(gaussian_probability, dtype=np.float64)
    if probability.ndim != 2 or probability.shape[1] < 2:
        raise ValueError("gaussian_probability must have shape [N, C]")
    if not 0.0 < surface_mass_fraction <= 1.0:
        raise ValueError("surface_mass_fraction must lie in (0, 1]")
    if not 0 <= background_class < probability.shape[1]:
        raise ValueError("background class out of range")
    if len(trace.gaussian_ids) != len(trace.weights):
        raise ValueError("trace ids and weights differ in length")
    background = np.zeros(probability.shape[1], dtype=np.float64)
    background[background_class] = 1.0
    if not len(trace.weights):
        return ResponsibilityPrediction(background, background, None, 0, 0.0)
    selected_probability = probability[trace.gaussian_ids]
    standard = trace.weights @ selected_probability + trace.final_transmittance * background
    mass = float(trace.weights.sum())
    prefix = int(np.searchsorted(np.cumsum(trace.weights), surface_mass_fraction * mass, side="left") + 1)
    surface_weights = trace.weights[:prefix]
    surface_mass = float(surface_weights.sum())
    surface = surface_weights @ selected_probability[:prefix] / surface_mass
    tail_weights = trace.weights[prefix:]
    tail = None
    if len(tail_weights) and float(tail_weights.sum()) > np.finfo(np.float64).tiny:
        tail = tail_weights @ selected_probability[prefix:] / float(tail_weights.sum())
    return ResponsibilityPrediction(standard, surface, tail, prefix, surface_mass)
