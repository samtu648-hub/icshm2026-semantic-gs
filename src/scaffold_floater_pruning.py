"""Deterministic training-support mask for the RGB-073 frozen renderer."""

from __future__ import annotations

from typing import Mapping

import numpy as np


REQUIRED_ARRAYS = (
    "dominant_pixels",
    "dominant_views",
    "mean_RGB_MAE",
    "high_error_pixel_fraction",
)


def residual_support_prune_mask(
    arrays: Mapping[str, np.ndarray],
    *,
    minimum_dominant_pixels: int,
    maximum_dominant_views: int,
    minimum_mean_rgb_mae: float,
    minimum_high_error_fraction: float,
) -> np.ndarray:
    """Return the single pre-registered low-support/high-residual anchor mask."""

    if minimum_dominant_pixels <= 0 or maximum_dominant_views < 0:
        raise ValueError("pixel/view support thresholds are invalid")
    if not 0.0 <= minimum_mean_rgb_mae <= 1.0:
        raise ValueError("RGB MAE threshold must be in [0,1]")
    if not 0.0 <= minimum_high_error_fraction <= 1.0:
        raise ValueError("high-error fraction threshold must be in [0,1]")
    missing = [name for name in REQUIRED_ARRAYS if name not in arrays]
    if missing:
        raise ValueError(f"anchor statistics are missing arrays: {missing}")
    values = [np.asarray(arrays[name]) for name in REQUIRED_ARRAYS]
    if any(value.ndim != 1 or value.shape != values[0].shape for value in values):
        raise ValueError("anchor statistics must be equal-length 1D arrays")
    dominant_pixels, dominant_views, mean_mae, high_error_fraction = values
    return (
        (dominant_pixels >= int(minimum_dominant_pixels))
        & (dominant_views <= int(maximum_dominant_views))
        & (mean_mae >= float(minimum_mean_rgb_mae))
        & (high_error_fraction >= float(minimum_high_error_fraction))
    )

