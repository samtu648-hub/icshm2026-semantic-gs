"""Contracts for the SEM-160/161 high-resolution deck recovery branch.

The learned branch predicts ``non_deck`` versus ``deck`` from frozen rendered
RGB.  Fusion is one-way: only pixels that were background before the frozen
foundation gate may become deck.  Existing deck, stay_cable, tower and
foundation predictions, including foundation pixels rejected to background,
remain immutable.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from src.semantic2d import CLASS_COUNT, label_array
from src.semantic_cable_binary import (
    IMAGE_SHAPE,
    normalize_rgb,
    random_native_pair,
    read_rgb,
    sliding_cable_probability,
)
from src.semantic_protected_fusion import validate_prediction

BINARY_CLASS_NAMES = ("non_deck", "deck")
BACKGROUND_ID = 0
DECK_ID = 1
IMMUTABLE_BASE_CLASS_IDS = (1, 2, 3, 4)


def binary_target(label: np.ndarray) -> np.ndarray:
    """Map an official five-class label to non-deck/deck IDs."""
    if label.dtype != np.uint8 or label.shape != IMAGE_SHAPE:
        raise ValueError("Deck binary target requires a uint8 989x1320 label")
    if label.size == 0 or int(label.max()) >= CLASS_COUNT:
        raise ValueError("Deck binary target contains an invalid official class ID")
    return (label == DECK_ID).astype(np.uint8)


def binary_class_weights_from_labels(
    label_root: Path,
    names: Iterable[str],
    *,
    minimum: float = 0.05,
    maximum: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute inverse-frequency binary weights from authorised labels only."""
    counts = np.zeros(2, dtype=np.int64)
    ordered = list(names)
    if not ordered:
        raise ValueError("Deck binary class weights require at least one label")
    for name in ordered:
        counts += np.bincount(
            binary_target(label_array(label_root / name)).ravel(), minlength=2
        )
    if np.any(counts <= 0):
        raise ValueError(f"Deck binary training is missing a class: {counts.tolist()}")
    weights = np.median(counts.astype(np.float64)) / counts
    return np.clip(weights, minimum, maximum).astype(np.float32), counts


def load_binary_model(model_dir: Path, seed: int):
    """Load the frozen SegFormer-B2 source with a fresh deck-binary head."""
    import torch
    from transformers import SegformerForSemanticSegmentation

    torch.manual_seed(seed)
    return SegformerForSemanticSegmentation.from_pretrained(
        model_dir,
        local_files_only=True,
        num_labels=2,
        id2label={0: BINARY_CLASS_NAMES[0], 1: BINARY_CLASS_NAMES[1]},
        label2id={name: index for index, name in enumerate(BINARY_CLASS_NAMES)},
        ignore_mismatched_sizes=True,
    )


def sliding_deck_probability(
    model: Any,
    image: np.ndarray,
    *,
    crop: int,
    stride: int,
    device: Any,
    amp_dtype: Any,
) -> np.ndarray:
    """Predict full-frame deck probability using the frozen native tiling."""
    return sliding_cable_probability(
        model,
        image,
        crop=crop,
        stride=stride,
        device=device,
        amp_dtype=amp_dtype,
    )


def validate_deck_probability(
    probability: np.ndarray, *, shape: tuple[int, int]
) -> np.ndarray:
    """Validate one scalar deck probability map."""
    if probability.dtype not in (np.float16, np.float32, np.float64):
        raise ValueError("Deck probability must be floating point")
    if probability.shape != shape:
        raise ValueError(
            f"Deck probability must have shape {shape}, got {probability.shape}"
        )
    if not np.isfinite(probability).all():
        raise ValueError("Deck probability contains a non-finite value")
    if np.any(probability < -1e-5) or np.any(probability > 1.0 + 1e-5):
        raise ValueError("Deck probability lies outside [0, 1]")
    return probability.astype(np.float32, copy=False)


def recover_deck(
    base_prediction: np.ndarray,
    pre_foundation_prediction: np.ndarray,
    deck_probability: np.ndarray,
    *,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover deck only from pixels that were and remain background."""
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("Deck threshold must lie strictly between zero and one")
    base = validate_prediction(base_prediction, context="deck recovery base")
    pre_foundation = validate_prediction(
        pre_foundation_prediction, context="deck pre-foundation prediction"
    )
    if pre_foundation.shape != base.shape:
        raise ValueError("Deck recovery parent grids do not match")
    probability = validate_deck_probability(deck_probability, shape=base.shape)
    eligible = (base == BACKGROUND_ID) & (pre_foundation == BACKGROUND_ID)
    additions = eligible & (probability >= float(threshold))
    recovered = base.copy()
    recovered[additions] = DECK_ID
    if np.any((recovered != base) & ~additions):
        raise RuntimeError("Deck recovery changed a non-addition pixel")
    if not np.array_equal(recovered[~eligible], base[~eligible]):
        raise RuntimeError("Deck recovery changed a protected pixel")
    for class_id in (2, 3, 4):
        if not np.array_equal(recovered == class_id, base == class_id):
            raise RuntimeError(f"Deck recovery changed immutable class {class_id}")
    if not np.all((base == DECK_ID) <= (recovered == DECK_ID)):
        raise RuntimeError("Deck recovery deleted an existing deck pixel")
    return recovered, eligible, additions


def deck_gate(
    *,
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    protected_predictions_exact: bool,
    minimum_iou_gain: float,
    minimum_recall_gain: float,
    minimum_precision: float,
    minimum_miou_gain: float,
    maximum_background_fp_ratio: float,
) -> dict[str, Any]:
    """Apply the registered precision-aware one-way deck recovery gate."""
    candidate_deck = candidate["per_class"]["deck"]
    baseline_deck = baseline["per_class"]["deck"]
    base_fp = int(baseline["confusion_matrix_rows_gt_columns_prediction"][0][1])
    candidate_fp = int(candidate["confusion_matrix_rows_gt_columns_prediction"][0][1])
    fp_ratio = float(candidate_fp / base_fp) if base_fp else float("inf")
    checks = {
        "protected_predictions_exact": bool(protected_predictions_exact),
        "mIoU_gain": float(candidate["mIoU_with_background"])
        >= float(baseline["mIoU_with_background"]) + float(minimum_miou_gain),
        "deck_iou_gain": float(candidate_deck["iou"])
        >= float(baseline_deck["iou"]) + float(minimum_iou_gain),
        "deck_recall_gain": float(candidate_deck["recall"])
        >= float(baseline_deck["recall"]) + float(minimum_recall_gain),
        "deck_precision_floor": float(candidate_deck["precision"])
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
            "protected_predictions_exact": True,
        },
        "observed": {
            "mIoU_gain": float(candidate["mIoU_with_background"])
            - float(baseline["mIoU_with_background"]),
            "deck_iou_gain": float(candidate_deck["iou"])
            - float(baseline_deck["iou"]),
            "deck_recall_gain": float(candidate_deck["recall"])
            - float(baseline_deck["recall"]),
            "deck_precision": float(candidate_deck["precision"]),
            "background_to_deck_fp_ratio": fp_ratio,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def select_calibration_threshold(
    records: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    """Choose the best deck IoU among calibration-gate passers."""
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


__all__ = [
    "BINARY_CLASS_NAMES",
    "binary_class_weights_from_labels",
    "binary_target",
    "deck_gate",
    "load_binary_model",
    "normalize_rgb",
    "random_native_pair",
    "read_rgb",
    "recover_deck",
    "select_calibration_threshold",
    "sliding_deck_probability",
]
