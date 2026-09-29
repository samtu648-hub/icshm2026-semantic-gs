"""Read-only diagnostics for the SEM-120 0.20--0.30 cable uncertainty band.

The band is exactly the set of mutable SEM-001 pixels whose frozen rendered-RGB
teacher has ``0.20 <= p(stay_cable) < 0.30``.  It is the pixels that hard 0.20
adds over hard 0.30, so this module never changes a prediction, a probability,
or a model parameter.  Its purpose is to measure whether proposed future
evidence separates genuine bridge cables from the added yellow background.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import ndimage

from src.semantic_protected_fusion import (
    CLASS_COUNT,
    CLASS_NAMES,
    protected_mask,
    validate_prediction,
    validate_probabilities,
)


def uncertainty_band(
    base_prediction: np.ndarray,
    teacher_probabilities: np.ndarray,
    *,
    lower: float,
    upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the mutable pixels admitted by hard ``lower`` but not ``upper``."""
    if not 0.0 < float(lower) < float(upper) < 1.0:
        raise ValueError("SEM-122 requires thresholds strictly inside (0, 1)")
    base = validate_prediction(base_prediction, context="SEM-122 base prediction")
    teacher = validate_probabilities(teacher_probabilities, shape=base.shape)
    mutable = ~protected_mask(base)
    band = mutable & (teacher[2] >= float(lower)) & (teacher[2] < float(upper))
    return band, mutable


def high_confidence_seed(
    base_prediction: np.ndarray,
    teacher_probabilities: np.ndarray,
    *,
    threshold: float,
) -> np.ndarray:
    """Return the mutable hard-0.30 cable seed without modifying its class map."""
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("SEM-122 seed threshold must lie strictly inside (0, 1)")
    base = validate_prediction(base_prediction, context="SEM-122 base prediction")
    teacher = validate_probabilities(teacher_probabilities, shape=base.shape)
    return (~protected_mask(base)) & (teacher[2] >= float(threshold))


def _composition(mask: np.ndarray, classes: np.ndarray) -> dict[str, Any]:
    if mask.shape != classes.shape or mask.dtype != bool:
        raise ValueError("SEM-122 composition inputs must be aligned boolean/class maps")
    values = np.bincount(classes[mask].ravel(), minlength=CLASS_COUNT)
    total = int(values.sum())
    counts = {name: int(values[index]) for index, name in enumerate(CLASS_NAMES)}
    return {
        "pixels": total,
        "counts": counts,
        "fractions": {
            name: float(count / total) if total else 0.0
            for name, count in counts.items()
        },
    }


def _three_way(mask: np.ndarray, target: np.ndarray) -> dict[str, int]:
    return {
        "gt_stay_cable": int(np.count_nonzero(mask & (target == 2))),
        "gt_background": int(np.count_nonzero(mask & (target == 0))),
        "gt_other": int(np.count_nonzero(mask & (target != 2) & (target != 0))),
    }


def probability_bin_composition(
    band: np.ndarray,
    teacher_probabilities: np.ndarray,
    target: np.ndarray,
    *,
    edges: Sequence[float],
) -> list[dict[str, Any]]:
    """Report fixed probability bins; the bins are descriptive, not a search."""
    if len(edges) < 2 or any(float(a) >= float(b) for a, b in zip(edges, edges[1:])):
        raise ValueError("SEM-122 probability-bin edges must be strictly increasing")
    target = validate_prediction(target, context="SEM-122 target")
    teacher = validate_probabilities(teacher_probabilities, shape=target.shape)
    records: list[dict[str, Any]] = []
    for index, (lower, upper) in enumerate(zip(edges, edges[1:])):
        inclusive_upper = index == len(edges) - 2
        in_bin = band & (teacher[2] >= float(lower))
        in_bin &= teacher[2] <= float(upper) if inclusive_upper else teacher[2] < float(upper)
        records.append(
            {
                "lower_inclusive": float(lower),
                "upper_inclusive" if inclusive_upper else "upper_exclusive": float(upper),
                "composition": _three_way(in_bin, target),
            }
        )
    return records


def connectivity_to_seed(
    band: np.ndarray, seed: np.ndarray, target: np.ndarray
) -> dict[str, Any]:
    """Measure 8-connected band components that touch a hard-0.30 seed."""
    target = validate_prediction(target, context="SEM-122 target")
    if band.shape != target.shape or seed.shape != target.shape:
        raise ValueError("SEM-122 connectivity inputs must share one image shape")
    if band.dtype != bool or seed.dtype != bool:
        raise ValueError("SEM-122 connectivity masks must be boolean")
    structure = np.ones((3, 3), dtype=bool)
    labels, component_count = ndimage.label(band, structure=structure)
    neighbor_seed = ndimage.binary_dilation(seed, structure=structure)
    touching_ids = np.unique(labels[band & neighbor_seed])
    touching_ids = touching_ids[touching_ids != 0]
    touching = np.isin(labels, touching_ids)
    detached = band & ~touching
    areas = np.bincount(labels.ravel())[1:]
    touching_areas = areas[touching_ids - 1] if touching_ids.size else np.empty(0, dtype=np.int64)
    return {
        "component_count": int(component_count),
        "component_area_median": float(np.median(areas)) if areas.size else 0.0,
        "touching_seed_component_count": int(touching_ids.size),
        "touching_seed_component_area_median": float(np.median(touching_areas)) if touching_areas.size else 0.0,
        "touching_seed": _three_way(touching, target),
        "detached": _three_way(detached, target),
    }


def seed_distance_bins(
    band: np.ndarray,
    seed: np.ndarray,
    target: np.ndarray,
    *,
    edges: Sequence[float],
) -> list[dict[str, Any]]:
    """Bin the band by Euclidean distance to its nearest hard-0.30 seed."""
    if len(edges) < 2 or any(float(a) >= float(b) for a, b in zip(edges, edges[1:])):
        raise ValueError("SEM-122 distance-bin edges must be strictly increasing")
    target = validate_prediction(target, context="SEM-122 target")
    if band.shape != target.shape or seed.shape != target.shape:
        raise ValueError("SEM-122 distance inputs must share one image shape")
    if not np.any(seed):
        distance = np.full(target.shape, np.inf, dtype=np.float32)
    else:
        distance = ndimage.distance_transform_edt(~seed)
    records: list[dict[str, Any]] = []
    for index, (lower, upper) in enumerate(zip(edges, edges[1:])):
        inclusive_upper = index == len(edges) - 2
        in_bin = band & (distance >= float(lower))
        in_bin &= distance <= float(upper) if inclusive_upper else distance < float(upper)
        records.append(
            {
                "lower_inclusive": float(lower),
                "upper_inclusive" if inclusive_upper else "upper_exclusive": float(upper),
                "composition": _three_way(in_bin, target),
            }
        )
    return records


def summarise_uncertainty_band(
    base_prediction: np.ndarray,
    teacher_probabilities: np.ndarray,
    target: np.ndarray,
    *,
    lower: float,
    upper: float,
    probability_bin_edges: Sequence[float],
    distance_bin_edges: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return masks and a compact label-only diagnostic summary for one view."""
    target = validate_prediction(target, context="SEM-122 target")
    band, mutable = uncertainty_band(
        base_prediction, teacher_probabilities, lower=lower, upper=upper
    )
    seed = high_confidence_seed(
        base_prediction, teacher_probabilities, threshold=upper
    )
    return band, seed, {
        "band": _composition(band, target),
        "sem001_mutable_prediction": _composition(band, base_prediction),
        "probability_bins": probability_bin_composition(
            band, teacher_probabilities, target, edges=probability_bin_edges
        ),
        "seed_connectivity": connectivity_to_seed(band, seed, target),
        "seed_distance_bins": seed_distance_bins(
            band, seed, target, edges=distance_bin_edges
        ),
        "mutable_pixels": int(np.count_nonzero(mutable)),
        "hard_030_seed_pixels": int(np.count_nonzero(seed)),
    }
