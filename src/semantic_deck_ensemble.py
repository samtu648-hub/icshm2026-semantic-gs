"""Frozen probability ensemble contracts for SEM-173."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from src.semantic_deck_binary import validate_deck_probability


def blend_deck_probability(
    sem160_probability: np.ndarray,
    sem172_probability: np.ndarray,
    *,
    sem160_weight: float,
) -> np.ndarray:
    """Return one registered global convex blend of deck probabilities."""
    weight = float(sem160_weight)
    if not 0.0 < weight < 1.0:
        raise ValueError("SEM-160 ensemble weight must lie inside (0, 1)")
    first = validate_deck_probability(
        sem160_probability, shape=sem160_probability.shape
    )
    second = validate_deck_probability(
        sem172_probability, shape=sem160_probability.shape
    )
    return (weight * first + (1.0 - weight) * second).astype(
        np.float32, copy=False
    )


def select_ensemble_candidate(
    records: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    """Select only calibration passers using the registered ordering."""
    passing = [record for record in records if record["gate"]["passed"]]
    if not passing:
        return None
    return max(
        passing,
        key=lambda record: (
            float(record["metrics"]["per_class"]["deck"]["iou"]),
            float(record["metrics"]["mIoU_with_background"]),
            float(record["metrics"]["per_class"]["deck"]["recall"]),
            float(record["metrics"]["per_class"]["deck"]["precision"]),
            float(record["sem160_weight"]),
            float(record["threshold"]),
        ),
    )


__all__ = ["blend_deck_probability", "select_ensemble_candidate"]
