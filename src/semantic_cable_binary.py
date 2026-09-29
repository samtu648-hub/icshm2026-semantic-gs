"""Shared contracts for the SEM-140/141 specialised cable segmenter.

The learned branch predicts only ``non_stay_cable`` versus ``stay_cable`` on
native-resolution frozen Scaffold RGB.  Fusion is deliberately asymmetric:
SEM-001 predictions for deck, tower and foundation are immutable, while every
other pixel is decided in the binary background/stay_cable lane.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.semantic2d import CLASS_COUNT, label_array, tile_starts

BINARY_CLASS_NAMES = ("non_stay_cable", "stay_cable")
PROTECTED_CLASS_IDS = (1, 3, 4)
STAY_CABLE_ID = 2
IMAGE_SHAPE = (989, 1320)


def binary_target(label: np.ndarray) -> np.ndarray:
    """Map an official five-class label to the registered binary target."""
    if label.dtype != np.uint8 or label.shape != IMAGE_SHAPE:
        raise ValueError("Cable binary target requires a uint8 989x1320 label")
    if label.size == 0 or int(label.max()) >= CLASS_COUNT:
        raise ValueError("Cable binary target contains an invalid official class ID")
    return (label == STAY_CABLE_ID).astype(np.uint8)


def binary_class_weights_from_labels(
    label_root: Path,
    names: Iterable[str],
    *,
    minimum: float = 0.05,
    maximum: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return inverse-frequency weights computed only from authorised labels."""
    counts = np.zeros(2, dtype=np.int64)
    ordered = list(names)
    if not ordered:
        raise ValueError("Cable binary class weights require at least one label")
    for name in ordered:
        target = binary_target(label_array(label_root / name))
        counts += np.bincount(target.ravel(), minlength=2)
    if np.any(counts <= 0):
        raise ValueError(f"Cable binary training is missing a class: {counts.tolist()}")
    weights = np.median(counts.astype(np.float64)) / counts
    return np.clip(weights, minimum, maximum).astype(np.float32), counts


def read_rgb(path: Path) -> np.ndarray:
    """Read one frozen RGB rendering without resizing it."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Could not read frozen RGB image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape != (*IMAGE_SHAPE, 3):
        raise ValueError(
            f"{path}: expected 1320x989 RGB, got {tuple(image.shape)}"
        )
    return image


def normalize_rgb(image: np.ndarray):
    """Apply the frozen ImageNet normalisation without spatial resampling."""
    import torch

    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Cable binary RGB input must be uint8 HxWx3")
    array = torch.from_numpy(
        np.ascontiguousarray(image.transpose(2, 0, 1))
    ).float().div_(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return (array - mean) / std


def random_native_pair(
    image: np.ndarray,
    label: np.ndarray,
    crop: int,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a native-pixel square crop and paired horizontal flip."""
    if image.shape[:2] != label.shape or label.shape != IMAGE_SHAPE:
        raise ValueError("Cable binary RGB and label grids do not match")
    height, width = label.shape
    if crop <= 0 or crop > min(height, width):
        raise ValueError("Cable binary crop does not fit the native image")
    top = rng.randint(0, height - crop)
    left = rng.randint(0, width - crop)
    image_crop = image[top : top + crop, left : left + crop]
    label_crop = label[top : top + crop, left : left + crop]
    if rng.random() < 0.5:
        image_crop = image_crop[:, ::-1]
        label_crop = label_crop[:, ::-1]
    return np.ascontiguousarray(image_crop), np.ascontiguousarray(label_crop)


def load_binary_model(model_dir: Path, seed: int):
    """Load the frozen SegFormer-B2 source with a fresh two-class head."""
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


def sliding_cable_probability(
    model: Any,
    image: np.ndarray,
    *,
    crop: int,
    stride: int,
    device: Any,
    amp_dtype: Any,
) -> np.ndarray:
    """Predict a full native-grid cable probability with overlapping tiles."""
    import torch
    import torch.nn.functional as functional

    score_sum = np.zeros(image.shape[:2], dtype=np.float32)
    count = np.zeros(image.shape[:2], dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for top in tile_starts(image.shape[0], crop, stride):
            for left in tile_starts(image.shape[1], crop, stride):
                tile = normalize_rgb(
                    image[top : top + crop, left : left + crop]
                ).unsqueeze(0).to(device)
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(pixel_values=tile).logits
                    logits = functional.interpolate(
                        logits,
                        size=(crop, crop),
                        mode="bilinear",
                        align_corners=False,
                    )
                    probability = torch.softmax(logits.float(), dim=1)[0, 1]
                score_sum[top : top + crop, left : left + crop] += (
                    probability.cpu().numpy()
                )
                count[top : top + crop, left : left + crop] += 1.0
    if np.any(count == 0):
        raise RuntimeError("Cable binary sliding inference left uncovered pixels")
    return score_sum / count


def validate_cable_probability(
    probability: np.ndarray, *, shape: tuple[int, int]
) -> np.ndarray:
    """Validate one scalar stay_cable probability map."""
    if probability.dtype not in (np.float16, np.float32, np.float64):
        raise ValueError("Cable probability must be floating point")
    if probability.shape != shape:
        raise ValueError(
            f"Cable probability must have shape {shape}, got {probability.shape}"
        )
    if not np.isfinite(probability).all():
        raise ValueError("Cable probability contains a non-finite value")
    if np.any(probability < -1e-5) or np.any(probability > 1.0 + 1e-5):
        raise ValueError("Cable probability lies outside [0, 1]")
    return probability.astype(np.float32, copy=False)


def fuse_cable_probability(
    base_prediction: np.ndarray,
    cable_probability: np.ndarray,
    *,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse binary cable evidence while preserving SEM-001 structure pixels."""
    if base_prediction.dtype != np.uint8 or base_prediction.ndim != 2:
        raise ValueError("SEM-001 base prediction must be a uint8 HxW class map")
    if base_prediction.size == 0 or int(base_prediction.max()) >= CLASS_COUNT:
        raise ValueError("SEM-001 base prediction contains an invalid class ID")
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("Cable threshold must lie strictly between zero and one")
    probability = validate_cable_probability(
        cable_probability, shape=base_prediction.shape
    )
    protected = np.isin(base_prediction, PROTECTED_CLASS_IDS)
    fused = base_prediction.copy()
    mutable = ~protected
    fused[mutable] = np.where(probability[mutable] >= threshold, 2, 0).astype(
        np.uint8
    )
    if not np.array_equal(fused[protected], base_prediction[protected]):
        raise RuntimeError("Cable binary fusion changed a protected SEM-001 pixel")
    if not set(np.unique(fused[mutable])).issubset({0, 2}):
        raise RuntimeError("Cable binary fusion emitted a protected class")
    return fused, protected


def score_metrics(metrics: dict[str, Any]) -> float:
    """Reuse the documented developer score without introducing new weights."""
    cable = metrics["per_class"]["stay_cable"]
    return float(
        0.55 * float(metrics["mIoU_with_background"])
        + 0.30 * float(cable["iou"])
        + 0.15 * float(cable["recall"])
    )


def relative_gate(
    *,
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    protected_predictions_exact: bool,
    minimum_iou_gain: float,
    minimum_recall_gain: float,
    minimum_miou_gain: float,
    maximum_background_fp_ratio: float,
) -> dict[str, Any]:
    """Apply a fixed cable-specialisation gate relative to protected fusion."""
    candidate_cable = candidate["per_class"]["stay_cable"]
    baseline_cable = baseline["per_class"]["stay_cable"]
    base_fp = int(baseline["background_to_stay_cable_pixels"])
    candidate_fp = int(candidate["background_to_stay_cable_pixels"])
    checks = {
        "protected_predictions_exact": bool(protected_predictions_exact),
        "mIoU_gain": float(candidate["mIoU_with_background"])
        >= float(baseline["mIoU_with_background"]) + float(minimum_miou_gain),
        "stay_cable_iou_gain": float(candidate_cable["iou"])
        >= float(baseline_cable["iou"]) + float(minimum_iou_gain),
        "stay_cable_recall_gain": float(candidate_cable["recall"])
        >= float(baseline_cable["recall"]) + float(minimum_recall_gain),
        "background_false_positive_ratio": candidate_fp
        <= base_fp * float(maximum_background_fp_ratio),
    }
    return {
        "baseline": baseline,
        "requirements": {
            "minimum_mIoU_gain": float(minimum_miou_gain),
            "minimum_stay_cable_iou_gain": float(minimum_iou_gain),
            "minimum_stay_cable_recall_gain": float(minimum_recall_gain),
            "maximum_background_false_positive_ratio": float(
                maximum_background_fp_ratio
            ),
            "protected_predictions_exact": True,
        },
        "observed": {
            "mIoU_gain": float(candidate["mIoU_with_background"])
            - float(baseline["mIoU_with_background"]),
            "stay_cable_iou_gain": float(candidate_cable["iou"])
            - float(baseline_cable["iou"]),
            "stay_cable_recall_gain": float(candidate_cable["recall"])
            - float(baseline_cable["recall"]),
            "background_false_positive_ratio": float(candidate_fp / base_fp)
            if base_fp
            else 0.0,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def select_calibration_threshold(
    records: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    """Select the best documented score only among calibration gate passers."""
    passing = [record for record in records if record["gate"]["passed"]]
    if not passing:
        return None
    return max(
        passing,
        key=lambda record: (
            float(record["score"]),
            float(record["metrics"]["per_class"]["stay_cable"]["iou"]),
            float(record["metrics"]["per_class"]["stay_cable"]["recall"]),
            float(record["threshold"]),
        ),
    )
