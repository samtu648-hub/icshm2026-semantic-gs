"""Protected spatial deck-gap recovery for SEM-171."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import cv2
import numpy as np

from src.semantic_deck_binary import validate_deck_probability
from src.semantic_protected_fusion import validate_prediction


def closed_deck_mask(prediction: np.ndarray, *, radius: int) -> np.ndarray:
    """Close narrow gaps in a class-1 mask with a registered ellipse."""
    if int(radius) != radius or int(radius) < 1:
        raise ValueError("Deck closing radius must be a positive integer")
    base = validate_prediction(prediction, context="deck continuity base")
    radius = int(radius)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    closed = cv2.morphologyEx(
        np.ascontiguousarray(base == 1, dtype=np.uint8),
        cv2.MORPH_CLOSE,
        kernel,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return closed.astype(bool, copy=False)


def recover_deck_continuity(
    baseline_prediction: np.ndarray,
    pre_foundation_prediction: np.ndarray,
    deck_probability: np.ndarray,
    *,
    radius: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add only probability-supported pixels enclosed by the deck closing."""
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("Deck continuity threshold must lie inside (0, 1)")
    baseline = validate_prediction(
        baseline_prediction, context="deck continuity baseline"
    )
    pre = validate_prediction(
        pre_foundation_prediction, context="deck continuity pre-foundation"
    )
    if pre.shape != baseline.shape:
        raise ValueError("Deck continuity parent grids do not match")
    probability = validate_deck_probability(deck_probability, shape=baseline.shape)
    enclosed = closed_deck_mask(baseline, radius=radius) & (baseline != 1)
    eligible = enclosed & (baseline == 0) & (pre == 0)
    additions = eligible & (probability >= float(threshold))
    candidate = baseline.copy()
    candidate[additions] = 1

    if not np.array_equal(candidate[baseline == 1], baseline[baseline == 1]):
        raise RuntimeError("Deck continuity modified an existing deck pixel")
    if not np.array_equal(candidate[~additions], baseline[~additions]):
        raise RuntimeError("Deck continuity modified a non-addition pixel")
    for class_id in (2, 3, 4):
        if not np.array_equal(candidate == class_id, baseline == class_id):
            raise RuntimeError(
                f"Deck continuity changed immutable class {class_id}"
            )
    if np.any(additions & (pre != 0)):
        raise RuntimeError("Deck continuity relabelled a foundation rejection")
    return candidate, enclosed, additions


def continuity_gate(
    *,
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    protected_predictions_exact: bool,
    existing_deck_predictions_exact: bool,
    minimum_iou_gain: float,
    minimum_recall_gain: float,
    minimum_precision: float,
    minimum_miou_gain: float,
    maximum_background_fp_ratio: float,
) -> dict[str, Any]:
    """Apply SEM-171's registered additive continuity gate."""
    after = candidate["per_class"]["deck"]
    before = baseline["per_class"]["deck"]
    base_fp = int(baseline["confusion_matrix_rows_gt_columns_prediction"][0][1])
    candidate_fp = int(candidate["confusion_matrix_rows_gt_columns_prediction"][0][1])
    fp_ratio = float(candidate_fp / base_fp) if base_fp else float("inf")
    checks = {
        "protected_predictions_exact": bool(protected_predictions_exact),
        "existing_deck_predictions_exact": bool(existing_deck_predictions_exact),
        "mIoU_gain": float(candidate["mIoU_with_background"])
        >= float(baseline["mIoU_with_background"]) + float(minimum_miou_gain),
        "deck_iou_gain": float(after["iou"])
        >= float(before["iou"]) + float(minimum_iou_gain),
        "deck_recall_gain": float(after["recall"])
        >= float(before["recall"]) + float(minimum_recall_gain),
        "deck_precision_floor": float(after["precision"])
        >= float(minimum_precision),
        "background_to_deck_fp_ratio": fp_ratio
        <= float(maximum_background_fp_ratio),
    }
    return {
        "requirements": {
            "minimum_mIoU_gain": float(minimum_miou_gain),
            "minimum_deck_iou_gain": float(minimum_iou_gain),
            "minimum_deck_recall_gain": float(minimum_recall_gain),
            "minimum_deck_precision": float(minimum_precision),
            "maximum_background_to_deck_fp_ratio": float(
                maximum_background_fp_ratio
            ),
            "existing_deck_predictions_exact": True,
            "protected_predictions_exact": True,
        },
        "observed": {
            "mIoU_gain": float(candidate["mIoU_with_background"])
            - float(baseline["mIoU_with_background"]),
            "deck_iou_gain": float(after["iou"]) - float(before["iou"]),
            "deck_recall_gain": float(after["recall"])
            - float(before["recall"]),
            "deck_precision": float(after["precision"]),
            "background_to_deck_fp_ratio": fp_ratio,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def select_continuity_candidate(
    records: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    """Select only calibration passers in the preregistered order."""
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
            -int(record["radius"]),
            float(record["threshold"]),
        ),
    )


__all__ = [
    "closed_deck_mask",
    "continuity_gate",
    "recover_deck_continuity",
    "select_continuity_candidate",
]
