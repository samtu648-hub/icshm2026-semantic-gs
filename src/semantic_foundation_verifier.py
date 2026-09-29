"""Contracts for conservative foundation verification.

The verifier is deliberately one-way: it may reject an existing foundation
prediction to background, but it can never introduce foundation or modify
deck, stay_cable, or tower predictions.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from src.semantic2d import CLASS_COUNT
from src.semantic_protected_fusion import validate_prediction

BACKGROUND_ID = 0
FOUNDATION_ID = 4
IMMUTABLE_CLASS_IDS = (1, 2, 3)


def validate_foundation_probability(
    probability: np.ndarray, *, shape: tuple[int, int]
) -> np.ndarray:
    """Validate one full-frame foundation probability map."""
    if probability.dtype not in (np.float16, np.float32, np.float64):
        raise ValueError("Foundation probability must be floating point")
    if probability.shape != shape:
        raise ValueError(
            f"Foundation probability must have shape {shape}, got {probability.shape}"
        )
    if not np.isfinite(probability).all():
        raise ValueError("Foundation probability contains a non-finite value")
    if np.any(probability < -1e-5) or np.any(probability > 1.0 + 1e-5):
        raise ValueError("Foundation probability lies outside [0, 1]")
    return probability.astype(np.float32, copy=False)


def verify_foundation(
    base_prediction: np.ndarray,
    foundation_probability: np.ndarray,
    *,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reject low-confidence class-4 pixels to background only."""
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("Foundation threshold must lie strictly between zero and one")
    base = validate_prediction(base_prediction, context="foundation verifier base")
    probability = validate_foundation_probability(
        foundation_probability, shape=base.shape
    )
    candidates = base == FOUNDATION_ID
    rejected = candidates & (probability < float(threshold))
    verified = base.copy()
    verified[rejected] = BACKGROUND_ID
    for class_id in IMMUTABLE_CLASS_IDS:
        if not np.array_equal(verified == class_id, base == class_id):
            raise RuntimeError(
                f"Foundation verifier changed immutable class {class_id}"
            )
    if np.any((verified != base) & ~rejected):
        raise RuntimeError("Foundation verifier changed a non-rejected pixel")
    return verified, candidates, rejected


def confusion_after_foundation_rejection(
    baseline_confusion: np.ndarray, rejected_by_gt: np.ndarray
) -> np.ndarray:
    """Transform a confusion matrix for class-4 to class-0 rejections."""
    confusion = np.asarray(baseline_confusion)
    rejected = np.asarray(rejected_by_gt)
    if confusion.shape != (CLASS_COUNT, CLASS_COUNT) or not np.issubdtype(
        confusion.dtype, np.integer
    ):
        raise ValueError("Foundation baseline confusion must be a 5x5 integer array")
    if rejected.shape != (CLASS_COUNT,) or not np.issubdtype(
        rejected.dtype, np.integer
    ):
        raise ValueError("Foundation rejected-by-GT counts must be five integers")
    if np.any(rejected < 0) or np.any(rejected > confusion[:, FOUNDATION_ID]):
        raise ValueError("Foundation rejection counts exceed baseline class-4 counts")
    updated = confusion.astype(np.int64, copy=True)
    updated[:, FOUNDATION_ID] -= rejected
    updated[:, BACKGROUND_ID] += rejected
    return updated


def foundation_gate(
    *,
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    immutable_predictions_exact: bool,
    minimum_iou_gain: float,
    minimum_precision_gain: float,
    minimum_recall: float,
    minimum_miou_gain: float,
    minimum_background_fp_reduction: float,
) -> dict[str, Any]:
    """Apply the registered precision-first foundation gate."""
    candidate_foundation = candidate["per_class"]["foundation"]
    baseline_foundation = baseline["per_class"]["foundation"]
    baseline_bg_fp = int(baseline["confusion_matrix_rows_gt_columns_prediction"][0][4])
    candidate_bg_fp = int(candidate["confusion_matrix_rows_gt_columns_prediction"][0][4])
    reduction = (
        float(1.0 - candidate_bg_fp / baseline_bg_fp) if baseline_bg_fp else 0.0
    )
    checks = {
        "immutable_deck_stay_cable_tower_predictions_exact": bool(
            immutable_predictions_exact
        ),
        "mIoU_gain": float(candidate["mIoU_with_background"])
        >= float(baseline["mIoU_with_background"]) + float(minimum_miou_gain),
        "foundation_iou_gain": float(candidate_foundation["iou"])
        >= float(baseline_foundation["iou"]) + float(minimum_iou_gain),
        "foundation_precision_gain": float(candidate_foundation["precision"])
        >= float(baseline_foundation["precision"])
        + float(minimum_precision_gain),
        "foundation_recall_floor": float(candidate_foundation["recall"])
        >= float(minimum_recall),
        "background_to_foundation_reduction": reduction
        >= float(minimum_background_fp_reduction),
    }
    return {
        "requirements": {
            "minimum_mIoU_gain": float(minimum_miou_gain),
            "minimum_foundation_iou_gain": float(minimum_iou_gain),
            "minimum_foundation_precision_gain": float(minimum_precision_gain),
            "minimum_foundation_recall": float(minimum_recall),
            "minimum_background_to_foundation_reduction": float(
                minimum_background_fp_reduction
            ),
            "immutable_deck_stay_cable_tower_predictions_exact": True,
        },
        "observed": {
            "mIoU_gain": float(candidate["mIoU_with_background"])
            - float(baseline["mIoU_with_background"]),
            "foundation_iou_gain": float(candidate_foundation["iou"])
            - float(baseline_foundation["iou"]),
            "foundation_precision_gain": float(candidate_foundation["precision"])
            - float(baseline_foundation["precision"]),
            "foundation_recall": float(candidate_foundation["recall"]),
            "background_to_foundation_reduction": reduction,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def select_foundation_threshold(
    records: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    """Maximise foundation IoU among calibration-gate passers."""
    passing = [record for record in records if record["gate"]["passed"]]
    if not passing:
        return None
    return max(
        passing,
        key=lambda record: (
            float(record["metrics"]["per_class"]["foundation"]["iou"]),
            float(record["metrics"]["per_class"]["foundation"]["precision"]),
            float(record["metrics"]["mIoU_with_background"]),
            float(record["metrics"]["per_class"]["foundation"]["recall"]),
            -float(record["threshold"]),
        ),
    )
