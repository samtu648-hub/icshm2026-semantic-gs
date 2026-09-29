#!/usr/bin/env python3
"""Independently recompute all saved long-training endpoint metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quality_protocol import sha256, write_new_json


CLASSES = ("background", "deck", "stay_cable", "tower", "foundation")
ENDPOINTS = (6000, 8000, 10000, 12000)


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def confusion(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    if target.shape != prediction.shape or target.dtype != np.uint8 or prediction.dtype != np.uint8:
        raise ValueError("Expected matched uint8 target and prediction")
    if target.max() > 4 or prediction.max() > 4:
        raise ValueError("Prediction or target is outside the five-class contract")
    return np.bincount((target.astype(np.int64) * 5 + prediction).ravel(), minlength=25).reshape(5, 5)


def metric_summary(matrix: np.ndarray) -> dict[str, Any]:
    per_class: dict[str, Any] = {}
    ious = []
    for index, name in enumerate(CLASSES):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        iou = tp / (tp + fp + fn) if tp + fp + fn else None
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        per_class[name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "IoU": iou,
            "precision": precision,
            "recall": recall,
        }
        ious.append(iou)
    valid_all = [value for value in ious if value is not None]
    valid_foreground = [value for value in ious[1:] if value is not None]
    return {
        "per_class": per_class,
        "confusion": matrix.tolist(),
        "mIoU_with_background": float(np.mean(valid_all)) if valid_all else None,
        "mIoU_without_background": (
            float(np.mean(valid_foreground)) if valid_foreground else None
        ),
    }


def boundary_counts(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    cable = target == 2
    gt_boundary = cable ^ binary_erosion(cable)
    gt_distance = distance_transform_edt(~gt_boundary)
    predicted_cable = prediction == 2
    predicted_boundary = predicted_cable ^ binary_erosion(predicted_cable)
    predicted_distance = distance_transform_edt(~predicted_boundary)
    return np.asarray(
        [
            np.sum(predicted_boundary & (gt_distance <= 2)),
            np.sum(predicted_boundary),
            np.sum(gt_boundary & (predicted_distance <= 2)),
            np.sum(gt_boundary),
        ],
        dtype=np.int64,
    )


def boundary_summary(values: np.ndarray) -> dict[str, Any]:
    precision = float(values[0] / values[1]) if values[1] else None
    recall = float(values[2] / values[3]) if values[3] else None
    f1 = (
        float(2 * precision * recall / (precision + recall))
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "tolerance_px": 2,
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
        "counts": values.tolist(),
    }


def close(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(close(left[key], right[key], tolerance) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(close(a, b, tolerance) for a, b in zip(left, right))
    if isinstance(left, float) or isinstance(right, float):
        return abs(float(left) - float(right)) <= tolerance
    return left == right


def audit(experiment_id: str) -> None:
    config_path = ROOT / "experiments/runs" / experiment_id / "config.yaml"
    config = read(config_path)
    batch = ROOT / config["output_path"]
    inference_path = batch / "inference/result.json"
    inference = read(inference_path)
    if inference["status"] != "completed":
        raise ValueError("Completed inference is required")
    split = read(ROOT / config["split_path"])
    fold = split["primary_fold"]
    train_names = set(fold["semantic_train_labeled"])
    holdout_names = set(fold["semantic_validation_labeled"])
    group_by_name = {
        name: group["group_id"] for group in split["groups"] for name in group["images"]
    }
    labels = ROOT / config["label_root"]
    recomputed: dict[str, Any] = {}
    checks = 0
    for step in ENDPOINTS:
        keys = ["train", "holdout", *[group["group_id"] for group in split["groups"]]]
        matrices = {key: np.zeros((5, 5), dtype=np.int64) for key in keys}
        boundaries = {key: np.zeros(4, dtype=np.int64) for key in keys}
        per_view = []
        prediction_dir = batch / f"inference/predictions/{step}"
        for name in sorted(train_names | holdout_names):
            target = np.asarray(Image.open(labels / name), dtype=np.uint8)
            prediction = np.asarray(Image.open(prediction_dir / name), dtype=np.uint8)
            matrix = confusion(target, prediction)
            counts = boundary_counts(target, prediction)
            role = "train" if name in train_names else "holdout"
            aggregate_keys = [role]
            if role == "holdout":
                aggregate_keys.append(group_by_name[name])
            for key in aggregate_keys:
                matrices[key] += matrix
                boundaries[key] += counts
            per_view.append({"name": name, "role": role, "group": group_by_name.get(name), **metric_summary(matrix)})
        metrics = {key: metric_summary(value) for key, value in matrices.items()}
        boundary_metrics = {key: boundary_summary(value) for key, value in boundaries.items()}
        saved = inference["endpoints"][str(step)]
        if not close(metrics, saved["metrics"]):
            raise ValueError(f"Endpoint {step} metric recomputation differs")
        checks += len(metrics)
        if not close(boundary_metrics, saved["boundary_metrics"]):
            raise ValueError(f"Endpoint {step} boundary recomputation differs")
        checks += len(boundary_metrics)
        saved_by_name = {row["name"]: row for row in saved["per_view"]}
        if not close({row["name"]: row for row in per_view}, saved_by_name):
            raise ValueError(f"Endpoint {step} per-view recomputation differs")
        checks += len(per_view)
        recomputed[str(step)] = {
            "metrics": metrics,
            "boundary_metrics": boundary_metrics,
            "worst_holdout_view": min(
                (row for row in per_view if row["role"] == "holdout"),
                key=lambda row: row["mIoU_without_background"],
            ),
        }
    write_new_json(
        batch / "audit-result.json",
        {
            "schema_version": 1,
            "status": "completed_independent_recalculation",
            "experiment_id": experiment_id,
            "config_sha256": sha256(config_path),
            "inference_result_sha256": sha256(inference_path),
            "checks": checks,
            "endpoints": recomputed,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args()
    audit(args.experiment_id)


if __name__ == "__main__":
    main()
