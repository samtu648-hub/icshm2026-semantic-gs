#!/usr/bin/env python3
"""Independently audit the sealed SEM-428 training and spatial evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quality_protocol import sha256  # noqa: E402


EID = "SEM-428-RGB152-DIR5-SPATIAL256"
CONFIG = ROOT / "experiments/runs" / EID / "config.yaml"
OUTPUT = ROOT / "outputs/quality/SEM-428/r1"
NAMES = ("background", "deck", "stay_cable", "tower", "foundation")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def confusion(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    if target.shape != prediction.shape:
        raise ValueError(f"shape mismatch: {target.shape} != {prediction.shape}")
    if target.min() < 0 or target.max() >= 5 or prediction.min() < 0 or prediction.max() >= 5:
        raise ValueError("semantic IDs must be in [0, 4]")
    return np.bincount((target.ravel() * 5 + prediction.ravel()), minlength=25).reshape(5, 5)


def summary(matrix: np.ndarray) -> dict:
    rows = {}
    ious = []
    for index, name in enumerate(NAMES):
        tp = int(matrix[index, index])
        fn = int(matrix[index, :].sum() - tp)
        fp = int(matrix[:, index].sum() - tp)
        denom = tp + fn + fp
        iou = float(tp / denom) if denom else None
        precision = float(tp / (tp + fp)) if tp + fp else None
        recall = float(tp / (tp + fn)) if tp + fn else None
        rows[name] = {"IoU": iou, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}
        ious.append(iou)
    return {
        "confusion": matrix.tolist(),
        "mIoU_with_background": float(np.mean(ious)),
        "mIoU_without_background": float(np.mean(ious[1:])),
        "per_class": rows,
    }


def close(left, right, atol=1e-12):
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(close(left[key], right[key], atol) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(close(a, b, atol) for a, b in zip(left, right))
    if left is None or right is None or isinstance(left, (str, bool)):
        return left == right
    return bool(np.isclose(left, right, rtol=0, atol=atol))


def main() -> None:
    config = read(CONFIG)
    binding = read(OUTPUT / "binding.json")
    training = read(OUTPUT / "training/result.json")
    result_path = OUTPUT / "inference/result.json"
    result = read(result_path)
    split = read(ROOT / config["split_path"])
    fold = split["primary_fold"]
    train_names = set(fold["semantic_train_labeled"])
    holdout_names = set(fold["semantic_validation_labeled"])
    buffer_names = set(fold["buffer_excluded_all_cameras"])
    group_by_name = {name: group["group_id"] for group in split["groups"] for name in group["images"]}

    matrices = {
        "train": np.zeros((5, 5), dtype=np.int64),
        "holdout": np.zeros((5, 5), dtype=np.int64),
        **{group["group_id"]: np.zeros((5, 5), dtype=np.int64) for group in split["groups"]},
    }
    prediction_hashes = {}
    label_root = ROOT / config["label_root"]
    for name in sorted(train_names | holdout_names):
        prediction_path = OUTPUT / "inference/predictions" / name
        prediction_hashes[name] = sha256(prediction_path)
        prediction = np.asarray(Image.open(prediction_path), dtype=np.uint8)
        target = np.asarray(Image.open(label_root / name), dtype=np.uint8)
        matrix = confusion(target, prediction)
        role = "train" if name in train_names else "holdout"
        matrices[role] += matrix
        if role == "holdout":
            matrices[group_by_name[name]] += matrix
    recomputed = {key: summary(value) for key, value in matrices.items()}

    checkpoint = OUTPUT / "training" / f"directional-logits-{config['primary_checkpoint']}.pt"
    checks = {
        "identity": config["experiment_id"] == EID == result["experiment_id"],
        "roles": (len(train_names), len(holdout_names), len(buffer_names)) == (256, 30, 17),
        "roles_disjoint": not (train_names & holdout_names or train_names & buffer_names or holdout_names & buffer_names),
        "config_binding": sha256(CONFIG) == binding["config_sha256"],
        "split_binding": sha256(ROOT / config["split_path"]) == binding["split_sha256"],
        "local_plan_binding": sha256(OUTPUT / "sampling-plan.json") == binding["local_sampling_plan_sha256"],
        "training_complete": training["status"] == "completed_sealed_6000" and training["successful_steps"] == 6000,
        "all_training_views_accessed": set(training["training_views_accessed"]) == train_names,
        "pre_confirmation_isolation": all(
            training[key] == 0
            for key in ("confirmation_rgb_accesses", "confirmation_label_accesses", "buffer_rgb_accesses", "buffer_label_accesses")
        ),
        "rgb_isolation": training["rgb_state_hashes_equal"] and training["fresh_rgb_exact"] and training["frozen_rgb_max_absolute_pixel_difference"] == 0,
        "checkpoint_replay": training["checkpoint_replay_exact"],
        "checkpoint_hash": sha256(checkpoint) == training["checkpoint_sha256s"]["6000"],
        "checkpoint_unchanged": result["checkpoint_unchanged"] and result["checkpoint_sha256_before_confirmation_access"] == result["checkpoint_sha256_after_confirmation_access"] == sha256(checkpoint),
        "inference_complete": result["status"] == "completed",
        "prediction_count": len(prediction_hashes) == 286,
        "prediction_hashes": prediction_hashes == result["prediction_hashes"],
        "metric_recount": close(recomputed, result["metrics"]),
        "access_counts": (
            result["train_label_accesses"],
            result["confirmation_label_accesses"],
            result["buffer_label_accesses"],
            result["unlabeled_label_accesses"],
        ) == (256, 30, 0, 0),
        "resource_gate": training["peak_reserved_mib"] < 12 * 1024 and result["peak_reserved_mib"] < 12 * 1024,
        "comparison_metrics_finite": all(np.isfinite(value) for value in result["comparison_sem396"].values()),
    }
    audit = {
        "schema_version": 1,
        "experiment_id": EID,
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "passed": sum(bool(value) for value in checks.values()),
        "total": len(checks),
        "result_sha256": sha256(result_path),
        "checkpoint_sha256": sha256(checkpoint),
        "recomputed_metrics": recomputed,
        "conclusion": "Artifact, isolation and metric recount audit only; promotion is decided from the registered SEM426 and SEM396 comparisons.",
    }
    audit_path = OUTPUT / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if audit["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

