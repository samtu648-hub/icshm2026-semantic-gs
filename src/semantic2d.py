"""Shared, model-agnostic utilities for the SEM-005 2D OOF audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


CLASS_NAMES = ("background", "deck", "stay_cable", "tower", "foundation")
CLASS_COUNT = len(CLASS_NAMES)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def label_array(path: Path) -> np.ndarray:
    from PIL import Image

    array = np.asarray(Image.open(path), dtype=np.uint8)
    if array.shape != (989, 1320):
        raise ValueError(f"{path}: expected a 989x1320 single-channel label, got {array.shape}")
    if int(array.max()) >= CLASS_COUNT:
        raise ValueError(f"{path}: class IDs must be in 0..{CLASS_COUNT - 1}")
    return array


def update_confusion(confusion: np.ndarray, target: np.ndarray, prediction: np.ndarray) -> None:
    if target.shape != prediction.shape:
        raise ValueError(f"Confusion inputs differ: {target.shape} vs {prediction.shape}")
    encoded = target.astype(np.int64).ravel() * CLASS_COUNT + prediction.astype(np.int64).ravel()
    confusion += np.bincount(encoded, minlength=CLASS_COUNT * CLASS_COUNT).reshape(CLASS_COUNT, CLASS_COUNT)


def confusion_metrics(confusion: np.ndarray) -> dict[str, Any]:
    per_class: dict[str, Any] = {}
    ious: list[float] = []
    for class_id, name in enumerate(CLASS_NAMES):
        tp = int(confusion[class_id, class_id])
        fp = int(confusion[:, class_id].sum() - tp)
        fn = int(confusion[class_id, :].sum() - tp)
        denominator = tp + fp + fn
        precision = float(tp / (tp + fp)) if tp + fp else None
        recall = float(tp / (tp + fn)) if tp + fn else None
        iou = float(tp / denominator) if denominator else None
        per_class[name] = {"id": class_id, "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "iou": iou}
        if iou is not None:
            ious.append(iou)
    foreground = [per_class[name]["iou"] for name in CLASS_NAMES[1:] if per_class[name]["iou"] is not None]
    return {
        "confusion_matrix_rows_gt_columns_prediction": confusion.astype(int).tolist(),
        "per_class": per_class,
        "mIoU_with_background": float(np.mean(ious)) if ious else None,
        "mIoU_without_background": float(np.mean(foreground)) if foreground else None,
        "background_to_stay_cable_pixels": int(confusion[0, 2]),
        "background_to_stay_cable_fraction": float(confusion[0, 2] / confusion[0].sum()) if confusion[0].sum() else None,
    }


def boundary_metrics(target: np.ndarray, prediction: np.ndarray, tolerances: Iterable[int] = (2, 4, 8)) -> dict[str, Any]:
    from scipy.ndimage import binary_erosion, distance_transform_edt

    output: dict[str, Any] = {}
    structure = np.ones((3, 3), dtype=bool)
    for class_id, name in enumerate(CLASS_NAMES):
        target_mask = target == class_id
        prediction_mask = prediction == class_id
        target_boundary = target_mask ^ binary_erosion(target_mask, structure=structure, border_value=0)
        prediction_boundary = prediction_mask ^ binary_erosion(prediction_mask, structure=structure, border_value=0)
        target_distance = distance_transform_edt(~target_boundary)
        prediction_distance = distance_transform_edt(~prediction_boundary)
        class_result: dict[str, Any] = {}
        for tolerance in tolerances:
            precision = float(np.mean(target_distance[prediction_boundary] <= tolerance)) if prediction_boundary.any() else 0.0
            recall = float(np.mean(prediction_distance[target_boundary] <= tolerance)) if target_boundary.any() else 0.0
            f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
            class_result[str(tolerance)] = {"precision": precision, "recall": recall, "f1": f1}
        output[name] = class_result
    return output


def stay_cable_structure(target: np.ndarray, prediction: np.ndarray, remote_distance: int = 8) -> dict[str, Any]:
    from scipy.ndimage import distance_transform_edt, label

    target_mask = target == 2
    prediction_mask = prediction == 2
    false_positive = prediction_mask & ~target_mask
    distance = distance_transform_edt(~target_mask)
    remote = false_positive & (distance > remote_distance)
    predicted_components, predicted_count = label(prediction_mask)
    target_components, target_count = label(target_mask)
    predicted_areas = np.bincount(predicted_components.ravel())[1:]
    target_areas = np.bincount(target_components.ravel())[1:]
    return {
        "predicted_components": int(predicted_count),
        "target_components": int(target_count),
        "predicted_component_area_median": float(np.median(predicted_areas)) if predicted_areas.size else 0.0,
        "target_component_area_median": float(np.median(target_areas)) if target_areas.size else 0.0,
        "false_positive_pixels": int(false_positive.sum()),
        "remote_false_positive_pixels_gt_8px": int(remote.sum()),
        "remote_false_positive_fraction": float(remote.sum() / false_positive.sum()) if false_positive.any() else 0.0,
        "true_stay_cable_pixels_covered": int((prediction_mask & target_mask).sum()),
    }


def tile_starts(length: int, crop: int, stride: int) -> list[int]:
    if length < crop:
        raise ValueError(f"Input length {length} is smaller than crop {crop}")
    starts = list(range(0, length - crop + 1, stride))
    if starts[-1] != length - crop:
        starts.append(length - crop)
    return starts


def class_weights_from_labels(label_root: Path, names: list[str], minimum: float = 0.05, maximum: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros(CLASS_COUNT, dtype=np.int64)
    for name in names:
        counts += np.bincount(label_array(label_root / name).ravel(), minlength=CLASS_COUNT)
    if np.any(counts <= 0):
        raise ValueError(f"A fold training set is missing an official class: {counts.tolist()}")
    weights = np.median(counts.astype(np.float64)) / counts
    return np.clip(weights, minimum, maximum).astype(np.float32), counts


def fold_for_index(split: dict[str, Any], fold_index: int) -> dict[str, Any]:
    for fold in split["folds"]:
        if int(fold["fold"]) == fold_index:
            return fold
    raise KeyError(f"OOF split has no fold {fold_index}")
