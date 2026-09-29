"""Read-only responsibility summaries for asymmetric bridge cloning."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from src.bridge_gaussian_support import quantiles
from src.improvedgs_responsibility_clone import alpha_conserving_partition


def summarize_responsibility(
    bridge_fractions: Iterable[float],
    opacity: float,
    *,
    minimum_bridge_fraction: float,
    maximum_bridge_fraction: float,
    maximum_iqr: float,
) -> dict[str, Any]:
    values = np.asarray(list(bridge_fractions), dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("at least one finite responsibility observation is required")
    median = float(np.median(values))
    iqr = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
    context_opacity, bridge_opacity = alpha_conserving_partition(
        np.asarray([float(opacity)]), np.asarray([median])
    )
    combined = 1.0 - (1.0 - context_opacity) * (1.0 - bridge_opacity)
    return {
        "observations": int(len(values)),
        "bridge_fraction": quantiles(values),
        "bridge_fraction_iqr": iqr,
        "stable_two_responsibility": bool(
            minimum_bridge_fraction <= median <= maximum_bridge_fraction and iqr <= maximum_iqr
        ),
        "parent_opacity": float(opacity),
        "context_child_opacity": float(context_opacity[0]),
        "bridge_child_opacity": float(bridge_opacity[0]),
        "alpha_conservation_abs_error": float(abs(combined[0] - opacity)),
    }


def summarize_population(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "gaussians": len(rows),
        "stable_two_responsibility": sum(bool(row["stable_two_responsibility"]) for row in rows),
        "bridge_fraction_p50": quantiles(row["bridge_fraction"]["p50"] for row in rows),
        "bridge_fraction_iqr": quantiles(row["bridge_fraction_iqr"] for row in rows),
        "parent_opacity": quantiles(row["parent_opacity"] for row in rows),
        "context_child_opacity": quantiles(row["context_child_opacity"] for row in rows),
        "bridge_child_opacity": quantiles(row["bridge_child_opacity"] for row in rows),
        "alpha_conservation_abs_error_max": max(
            (float(row["alpha_conservation_abs_error"]) for row in rows), default=0.0
        ),
    }

