"""Conservative rendered-RGB edge recovery for the SEM-112 OOF experiment.

SEM-112 starts from the frozen SEM-110 high-confidence protected fusion.  It
may only restore a low-confidence ``stay_cable`` pixel when all of the
following are true: the frozen SEM-001 map already calls it ``stay_cable``,
the frozen rendered RGB has a strong local luminance edge, and it is close to a
high-confidence rendered-domain cable seed.  The SEM-001 deck, tower and
foundation predictions are immutable.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from src.semantic_protected_fusion import (
    fuse_background_stay_cable,
    validate_prediction,
    validate_probabilities,
)


def validate_rgb(rgb: np.ndarray, *, shape: tuple[int, int]) -> np.ndarray:
    """Validate a frozen full-frame RGB render without accepting reference RGB."""
    if rgb.dtype != np.uint8 or rgb.shape != (*shape, 3):
        raise ValueError(
            "SEM-112 rendered RGB must be uint8 HxWx3 with the semantic-map shape"
        )
    return rgb


def luminance_edge_strength(rgb: np.ndarray) -> np.ndarray:
    """Return deterministic central-difference luminance edge strength in [0, 1]."""
    height, width = rgb.shape[:2]
    validate_rgb(rgb, shape=(height, width))
    luminance = (
        0.2126 * rgb[..., 0].astype(np.float32)
        + 0.7152 * rgb[..., 1].astype(np.float32)
        + 0.0722 * rgb[..., 2].astype(np.float32)
    )
    horizontal = np.zeros_like(luminance)
    vertical = np.zeros_like(luminance)
    horizontal[:, 1:-1] = (luminance[:, 2:] - luminance[:, :-2]) / (2.0 * 255.0)
    vertical[1:-1, :] = (luminance[2:, :] - luminance[:-2, :]) / (2.0 * 255.0)
    return np.hypot(horizontal, vertical)


def binary_dilate_square(mask: np.ndarray, *, radius: int) -> np.ndarray:
    """Dilate a boolean image with a bounded square kernel using only NumPy."""
    if mask.dtype != np.bool_ or mask.ndim != 2:
        raise ValueError("SEM-112 dilation expects a boolean HxW mask")
    if not isinstance(radius, int) or radius < 0 or radius > 8:
        raise ValueError("SEM-112 adjacency radius must be an integer in 0..8")
    if radius == 0:
        return mask.copy()
    height, width = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    result = np.zeros_like(mask)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            result |= padded[dy : dy + height, dx : dx + width]
    return result


def recover_edge_connected_stay_cable(
    base_prediction: np.ndarray,
    teacher_probabilities: np.ndarray,
    rendered_rgb: np.ndarray,
    *,
    high_stay_cable_threshold: float,
    low_stay_cable_threshold: float,
    edge_quantile: float,
    adjacency_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Restore only edge-connected low-confidence cable evidence.

    The returned prediction begins exactly as SEM-110's high-threshold fusion.
    ``recovered`` is the exact set of additional class-2 pixels and is always a
    subset of mutable SEM-001 class-2 pixels.  No protected class is consulted
    or altered by the recovery predicate.
    """
    if not 0.0 < float(low_stay_cable_threshold) < float(high_stay_cable_threshold) < 1.0:
        raise ValueError("SEM-112 requires 0 < low threshold < high threshold < 1")
    if not 0.0 < float(edge_quantile) < 1.0:
        raise ValueError("SEM-112 edge quantile must lie strictly between zero and one")
    base = validate_prediction(base_prediction, context="SEM-112 base prediction")
    teacher = validate_probabilities(teacher_probabilities, shape=base.shape)
    rgb = validate_rgb(rendered_rgb, shape=base.shape)
    baseline, fixed = fuse_background_stay_cable(
        base,
        teacher,
        stay_cable_threshold=high_stay_cable_threshold,
    )
    mutable = ~fixed
    edge = luminance_edge_strength(rgb)
    edge_threshold = float(np.quantile(edge[mutable], edge_quantile))
    seeds = mutable & (teacher[2] >= float(high_stay_cable_threshold))
    candidate = (
        mutable
        & (base == 2)
        & (teacher[2] >= float(low_stay_cable_threshold))
        & (teacher[2] < float(high_stay_cable_threshold))
        & (edge >= edge_threshold)
        & (edge > 0.0)
    )
    recovered = candidate & binary_dilate_square(seeds, radius=adjacency_radius)
    fused = baseline.copy()
    fused[recovered] = 2
    if not np.array_equal(fused[fixed], base[fixed]):
        raise RuntimeError("SEM-112 modified a protected SEM-001 prediction")
    if not np.all(np.isin(fused[mutable], (0, 2))):
        raise RuntimeError("SEM-112 emitted a non-mutable class outside the protected mask")
    if np.any(recovered & (base != 2)) or np.any(recovered & (baseline == 2)):
        raise RuntimeError("SEM-112 recovery escaped its low-confidence cable contract")
    return fused, fixed, recovered, {
        "edge_quantile": float(edge_quantile),
        "edge_threshold": edge_threshold,
        "high_confidence_seed_pixels": int(seeds.sum()),
        "candidate_pixels_before_connectivity": int(candidate.sum()),
        "recovered_pixels": int(recovered.sum()),
    }


def edge_recovery_gate(
    *,
    raw_sem001_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    recovered_metrics: dict[str, Any],
    protected_predictions_exact: bool,
    recovered_pixels: int,
    gates: dict[str, Any],
) -> dict[str, Any]:
    """Apply SEM-112's fixed training-only optimisation gate."""
    baseline_cable = baseline_metrics["per_class"]["stay_cable"]
    recovered_cable = recovered_metrics["per_class"]["stay_cable"]
    raw_fp = int(raw_sem001_metrics["background_to_stay_cable_pixels"])
    baseline_fp = int(baseline_metrics["background_to_stay_cable_pixels"])
    recovered_fp = int(recovered_metrics["background_to_stay_cable_pixels"])
    checks = {
        "protected_prediction_pixels_exact": bool(protected_predictions_exact),
        "recovered_pixels_nonzero": int(recovered_pixels) > 0,
        "mIoU_with_background_improvement": float(
            recovered_metrics["mIoU_with_background"]
        )
        >= float(baseline_metrics["mIoU_with_background"])
        + float(gates["mIoU_with_background_improvement_min"]),
        "stay_cable_iou_improvement": float(recovered_cable["iou"])
        >= float(baseline_cable["iou"])
        + float(gates["stay_cable_iou_improvement_min"]),
        "stay_cable_recall_improvement": float(recovered_cable["recall"])
        >= float(baseline_cable["recall"])
        + float(gates["stay_cable_recall_improvement_min"]),
        "background_to_stay_cable_growth": recovered_fp
        <= baseline_fp
        * (1.0 + float(gates["background_to_stay_cable_growth_max_fraction"])),
        "raw_background_to_stay_cable_reduction": recovered_fp
        <= raw_fp
        * (1.0 - float(gates["raw_background_to_stay_cable_reduction_min_fraction"])),
    }
    return {
        "raw_sem001_background_to_stay_cable_pixels": raw_fp,
        "baseline_background_to_stay_cable_pixels": baseline_fp,
        "recovered_background_to_stay_cable_pixels": recovered_fp,
        "background_to_stay_cable_growth_vs_baseline_fraction": float(
            recovered_fp / baseline_fp - 1.0
        )
        if baseline_fp
        else 0.0,
        "background_to_stay_cable_reduction_vs_raw_fraction": float(
            1.0 - recovered_fp / raw_fp
        )
        if raw_fp
        else 0.0,
        "gates": {key: float(value) for key, value in gates.items()},
        "checks": checks,
        "passed": all(checks.values()),
    }


def select_edge_recovery_candidate(
    records: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    """Choose only calibration-passing candidates, favouring cable IoU then restraint."""
    passing = [record for record in records if record.get("gate", {}).get("passed")]
    if not passing:
        return None
    return max(
        passing,
        key=lambda record: (
            float(record["recovered_metrics"]["per_class"]["stay_cable"]["iou"]),
            float(record["recovered_metrics"]["per_class"]["stay_cable"]["recall"]),
            -int(record["recovered_metrics"]["background_to_stay_cable_pixels"]),
            -int(record["recovered_pixels"]),
        ),
    )
