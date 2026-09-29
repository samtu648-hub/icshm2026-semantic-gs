"""Continuous two-stream background/stay_cable fusion for SEM-121.

The module never changes a frozen SEM-001 deck, tower or foundation pixel.
Within the mutable background/stay_cable lane it combines two independent
signals: the frozen rendered-RGB teacher's cable-vs-background log odds and
the frozen SEM-001 binary cable prior.  The three scalar calibration weights
are fitted once on a registered training calibration half; no threshold grid
or model training is involved.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.semantic_protected_fusion import (
    CLASS_COUNT,
    protected_mask,
    validate_prediction,
    validate_probabilities,
)


def two_stream_features(
    base_prediction: np.ndarray, teacher_probabilities: np.ndarray
) -> np.ndarray:
    """Return [teacher cable-vs-background log odds, SEM-001 cable prior]."""
    base = validate_prediction(base_prediction, context="SEM-121 base prediction")
    teacher = validate_probabilities(teacher_probabilities, shape=base.shape)
    epsilon = np.finfo(np.float32).eps
    teacher_log_odds = np.log(
        np.clip(teacher[2], epsilon, 1.0) / np.clip(teacher[0], epsilon, 1.0)
    )
    base_cable_prior = np.where(base == 2, 1.0, -1.0).astype(np.float32)
    return np.stack((teacher_log_odds.astype(np.float32), base_cable_prior), axis=-1)


def fit_logistic_calibrator(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    l2: float,
    maximum_iterations: int,
) -> dict[str, Any]:
    """Fit fixed-size L2 logistic calibration by deterministic Newton updates."""
    if features.ndim != 2 or features.shape[1] != 2:
        raise ValueError("SEM-121 features must have shape [N, 2]")
    if targets.shape != (features.shape[0],) or targets.dtype != np.uint8:
        raise ValueError("SEM-121 targets must be a uint8 vector matching features")
    if features.shape[0] < 2 or not np.isfinite(features).all():
        raise ValueError("SEM-121 calibration data is empty or non-finite")
    if not 0.0 < float(l2) <= 1.0 or int(maximum_iterations) < 1:
        raise ValueError("SEM-121 requires positive bounded L2 and iterations")
    if not set(np.unique(targets)).issubset({0, 1}) or len(np.unique(targets)) != 2:
        raise ValueError("SEM-121 calibration needs both cable and non-cable targets")

    values = features.astype(np.float64, copy=False)
    design = np.column_stack((np.ones(values.shape[0], dtype=np.float64), values))
    truth = targets.astype(np.float64, copy=False)
    weights = np.zeros(3, dtype=np.float64)
    penalty = np.diag((0.0, float(l2), float(l2)))
    for iteration in range(int(maximum_iterations)):
        logits = np.clip(design @ weights, -40.0, 40.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        gradient = (design.T @ (probability - truth)) / len(truth) + penalty @ weights
        curvature = probability * (1.0 - probability)
        hessian = (design.T * curvature) @ design / len(truth) + penalty
        update = np.linalg.solve(hessian, gradient)
        weights -= update
        if float(np.linalg.norm(update)) < 1e-9:
            break
    logits = np.clip(design @ weights, -40.0, 40.0)
    probability = 1.0 / (1.0 + np.exp(-logits))
    epsilon = np.finfo(np.float64).eps
    loss = -np.mean(
        truth * np.log(np.clip(probability, epsilon, 1.0))
        + (1.0 - truth) * np.log(np.clip(1.0 - probability, epsilon, 1.0))
    )
    return {
        "intercept": float(weights[0]),
        "teacher_cable_background_log_odds_weight": float(weights[1]),
        "sem001_cable_prior_weight": float(weights[2]),
        "iterations": iteration + 1,
        "balanced_sample_log_loss": float(loss),
    }


def calibrated_cable_probability(
    base_prediction: np.ndarray,
    teacher_probabilities: np.ndarray,
    calibrator: dict[str, Any],
) -> np.ndarray:
    """Calculate calibrated cable probability for every official pixel."""
    required = (
        "intercept",
        "teacher_cable_background_log_odds_weight",
        "sem001_cable_prior_weight",
    )
    if any(key not in calibrator for key in required):
        raise ValueError("SEM-121 calibrator has an incomplete schema")
    coefficients = np.asarray([calibrator[key] for key in required], dtype=np.float64)
    if not np.isfinite(coefficients).all():
        raise ValueError("SEM-121 calibrator contains non-finite weights")
    features = two_stream_features(base_prediction, teacher_probabilities)
    logits = (
        coefficients[0]
        + coefficients[1] * features[..., 0]
        + coefficients[2] * features[..., 1]
    )
    logits = np.clip(logits, -40.0, 40.0)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def fuse_continuous_background_stay_cable(
    base_prediction: np.ndarray,
    teacher_probabilities: np.ndarray,
    calibrator: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the fixed 0.5 calibrated decision in the mutable two-class lane."""
    base = validate_prediction(base_prediction, context="SEM-121 base prediction")
    probability = calibrated_cable_probability(base, teacher_probabilities, calibrator)
    if probability.shape != base.shape or not np.isfinite(probability).all():
        raise RuntimeError("SEM-121 calibrated probability has an invalid shape")
    fixed = protected_mask(base)
    fused = base.copy()
    mutable = ~fixed
    fused[mutable] = np.where(probability[mutable] >= 0.5, 2, 0).astype(np.uint8)
    if not np.array_equal(fused[fixed], base[fixed]) or int(fused.max()) >= CLASS_COUNT:
        raise RuntimeError("SEM-121 modified protected or official class values")
    return fused, fixed, probability
