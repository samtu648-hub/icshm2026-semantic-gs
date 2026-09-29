"""Selective, protected recovery from the fixed SEM-120 0.20--0.30 band."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import ndimage

from src.semantic_protected_fusion import fuse_background_stay_cable, protected_mask, validate_prediction, validate_probabilities
from src.semantic_uncertainty_band import high_confidence_seed, uncertainty_band

PriorMode = Literal["any", "sem001_stay_cable"]


def selective_recovery_mask(
    base_prediction: np.ndarray,
    teacher_probabilities: np.ndarray,
    *,
    probability_minimum: float,
    maximum_seed_distance: float,
    prior_mode: PriorMode,
    lower_threshold: float = 0.20,
    upper_threshold: float = 0.30,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a fixed subset of the 0.20--0.30 band without touching structure."""
    if not lower_threshold <= probability_minimum < upper_threshold:
        raise ValueError("SEM-123 probability minimum must lie inside the uncertainty band")
    if float(maximum_seed_distance) <= 0.0:
        raise ValueError("SEM-123 maximum seed distance must be positive")
    if prior_mode not in ("any", "sem001_stay_cable"):
        raise ValueError("SEM-123 prior mode is invalid")
    base = validate_prediction(base_prediction, context="SEM-123 base prediction")
    teacher = validate_probabilities(teacher_probabilities, shape=base.shape)
    band, _mutable = uncertainty_band(base, teacher, lower=lower_threshold, upper=upper_threshold)
    seed = high_confidence_seed(base, teacher, threshold=upper_threshold)
    distance = ndimage.distance_transform_edt(~seed) if np.any(seed) else np.full(base.shape, np.inf, dtype=np.float32)
    selected = band & (teacher[2] >= float(probability_minimum)) & (distance <= float(maximum_seed_distance))
    if prior_mode == "sem001_stay_cable":
        selected &= base == 2
    return selected, seed


def recover_selective_band(
    base_prediction: np.ndarray,
    teacher_probabilities: np.ndarray,
    *,
    probability_minimum: float,
    maximum_seed_distance: float,
    prior_mode: PriorMode,
    lower_threshold: float = 0.20,
    upper_threshold: float = 0.30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Start from hard 0.30 and recover only the registered band subset."""
    base = validate_prediction(base_prediction, context="SEM-123 base prediction")
    candidate, fixed = fuse_background_stay_cable(base, teacher_probabilities, stay_cable_threshold=upper_threshold)
    selected, _seed = selective_recovery_mask(
        base,
        teacher_probabilities,
        probability_minimum=probability_minimum,
        maximum_seed_distance=maximum_seed_distance,
        prior_mode=prior_mode,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
    )
    candidate[selected] = 2
    if not np.array_equal(candidate[fixed], base[fixed]) or np.any(selected & protected_mask(base)):
        raise RuntimeError("SEM-123 modified a protected SEM-001 pixel")
    return candidate, selected, fixed
