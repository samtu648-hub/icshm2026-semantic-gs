"""Protected bidirectional background/deck arbitration for SEM-170/171."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from src.semantic_deck_binary import validate_deck_probability
from src.semantic_protected_fusion import validate_prediction


def arbitrate_deck(
    base_prediction: np.ndarray,
    pre_foundation_prediction: np.ndarray,
    deck_probability: np.ndarray,
    *,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Let the frozen binary model decide only the authorised class-0/1 lane."""
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("Deck arbitration threshold must lie inside (0, 1)")
    base = validate_prediction(base_prediction, context="deck arbitration base")
    pre = validate_prediction(
        pre_foundation_prediction, context="deck arbitration pre-foundation"
    )
    if pre.shape != base.shape:
        raise ValueError("Deck arbitration parent grids do not match")
    probability = validate_deck_probability(deck_probability, shape=base.shape)
    mutable = (base == 1) | ((base == 0) & (pre == 0))
    candidate = base.copy()
    candidate[mutable] = (probability[mutable] >= float(threshold)).astype(np.uint8)
    additions = (base == 0) & (candidate == 1)
    removals = (base == 1) & (candidate == 0)
    if not np.array_equal(candidate[~mutable], base[~mutable]):
        raise RuntimeError("Deck arbitration modified a protected pixel")
    for class_id in (2, 3, 4):
        if not np.array_equal(candidate == class_id, base == class_id):
            raise RuntimeError(
                f"Deck arbitration changed immutable class {class_id}"
            )
    if np.any(additions & (pre != 0)):
        raise RuntimeError("Deck arbitration relabelled a foundation rejection")
    return candidate, mutable, additions, removals


def arbitration_gate(
    *,
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    protected_predictions_exact: bool,
    minimum_iou_gain: float,
    minimum_precision_gain: float,
    minimum_recall_gain: float,
    minimum_miou_gain: float,
    maximum_background_fp_ratio: float,
) -> dict[str, Any]:
    """Apply SEM-170's registered two-sided deck quality gate."""
    after = candidate["per_class"]["deck"]
    before = baseline["per_class"]["deck"]
    base_fp = int(baseline["confusion_matrix_rows_gt_columns_prediction"][0][1])
    candidate_fp = int(candidate["confusion_matrix_rows_gt_columns_prediction"][0][1])
    fp_ratio = float(candidate_fp / base_fp) if base_fp else float("inf")
    checks = {
        "protected_predictions_exact": bool(protected_predictions_exact),
        "mIoU_gain": float(candidate["mIoU_with_background"])
        >= float(baseline["mIoU_with_background"]) + float(minimum_miou_gain),
        "deck_iou_gain": float(after["iou"])
        >= float(before["iou"]) + float(minimum_iou_gain),
        "deck_precision_gain": float(after["precision"])
        >= float(before["precision"]) + float(minimum_precision_gain),
        "deck_recall_gain": float(after["recall"])
        >= float(before["recall"]) + float(minimum_recall_gain),
        "background_to_deck_fp_ratio": fp_ratio
        <= float(maximum_background_fp_ratio),
    }
    return {
        "requirements": {
            "minimum_mIoU_gain": float(minimum_miou_gain),
            "minimum_deck_iou_gain": float(minimum_iou_gain),
            "minimum_deck_precision_gain": float(minimum_precision_gain),
            "minimum_deck_recall_gain": float(minimum_recall_gain),
            "maximum_background_to_deck_fp_ratio": float(
                maximum_background_fp_ratio
            ),
            "protected_predictions_exact": True,
        },
        "observed": {
            "mIoU_gain": float(candidate["mIoU_with_background"])
            - float(baseline["mIoU_with_background"]),
            "deck_iou_gain": float(after["iou"]) - float(before["iou"]),
            "deck_precision_gain": float(after["precision"])
            - float(before["precision"]),
            "deck_recall_gain": float(after["recall"])
            - float(before["recall"]),
            "background_to_deck_fp_ratio": fp_ratio,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def select_arbitration_threshold(
    records: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    """Select only among calibration passers using the registered order."""
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
            float(record["threshold"]),
        ),
    )


__all__ = ["arbitrate_deck", "arbitration_gate", "select_arbitration_threshold"]
