#!/usr/bin/env python3
"""Freeze OOF confidence thresholds, then perform the one-shot 74-view audit."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_sem005_segformer import read_rgb, reload_checkpoint, sliding_probabilities  # noqa: E402
from src.semantic2d import (  # noqa: E402
    CLASS_COUNT,
    CLASS_NAMES,
    boundary_metrics,
    confusion_metrics,
    label_array,
    read_json,
    sha256,
    stay_cable_structure,
    update_confusion,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--prepared-scene", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fold-dir", type=Path, action="append", required=True)
    parser.add_argument("--frozen-thresholds", type=Path, required=True)
    parser.add_argument("--sem001-result", type=Path, required=True)
    parser.add_argument("--sem001-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def prediction_records(fold_dirs: list[Path]) -> list[tuple[Path, dict[str, Any]]]:
    records: list[tuple[Path, dict[str, Any]]] = []
    for directory in fold_dirs:
        manifest = read_json(directory / "prediction_manifest.json")
        for record in manifest:
            records.append((directory, record))
    names = [record["name"] for _, record in records]
    if len(names) != 219 or len(set(names)) != 219:
        raise ValueError(f"Expected exactly 219 unique OOF predictions, got {len(names)} / {len(set(names))}")
    return records


def selective_histograms(records: list[tuple[Path, dict[str, Any]]], labels_root: Path, bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, dict[str, np.ndarray]]]:
    totals = np.zeros((CLASS_COUNT, bins), dtype=np.int64)
    correct = np.zeros((CLASS_COUNT, bins), dtype=np.int64)
    gt_counts = np.zeros(CLASS_COUNT, dtype=np.int64)
    by_fold = {fold: {"totals": np.zeros((CLASS_COUNT, bins), dtype=np.int64), "correct": np.zeros((CLASS_COUNT, bins), dtype=np.int64), "gt": np.zeros(CLASS_COUNT, dtype=np.int64)} for fold in range(3)}
    for directory, record in records:
        fold = int(record["fold"])
        target = label_array(labels_root / record["name"])
        probability = np.load(directory / record["probabilities"])["probabilities"].astype(np.float32)
        prediction = probability.argmax(axis=0)
        confidence = probability.max(axis=0)
        gt_counts += np.bincount(target.ravel(), minlength=CLASS_COUNT)
        by_fold[fold]["gt"] += np.bincount(target.ravel(), minlength=CLASS_COUNT)
        indices = np.minimum((confidence * (bins - 1)).astype(np.int64), bins - 1)
        for class_id in range(CLASS_COUNT):
            mask = prediction == class_id
            totals[class_id] += np.bincount(indices[mask], minlength=bins)
            correct[class_id] += np.bincount(indices[mask & (target == class_id)], minlength=bins)
            by_fold[fold]["totals"][class_id] += np.bincount(indices[mask], minlength=bins)
            by_fold[fold]["correct"][class_id] += np.bincount(indices[mask & (target == class_id)], minlength=bins)
    return totals, correct, gt_counts, by_fold


def freeze_thresholds(totals: np.ndarray, correct: np.ndarray, gt_counts: np.ndarray, target_precision: float) -> dict[str, Any]:
    bins = totals.shape[1]
    output: dict[str, Any] = {}
    for class_id, name in enumerate(CLASS_NAMES):
        accepted = np.cumsum(totals[class_id, ::-1])[::-1]
        true = np.cumsum(correct[class_id, ::-1])[::-1]
        precision = np.divide(true, accepted, out=np.zeros_like(true, dtype=np.float64), where=accepted > 0)
        valid = np.flatnonzero((accepted > 0) & (precision >= target_precision))
        if valid.size:
            # The lowest valid threshold has maximum coverage because accepted
            # predictions monotonically decrease as the threshold increases.
            index = int(valid[0])
            status = "precision_target_reached"
        else:
            index = bins - 1
            status = "precision_target_not_reached"
        threshold = index / (bins - 1)
        tp = int(true[index])
        count = int(accepted[index])
        output[name] = {
            "class_id": class_id,
            "threshold": threshold,
            "accepted_pixels": count,
            "selective_precision": float(tp / count) if count else 0.0,
            "selective_recall": float(tp / gt_counts[class_id]) if gt_counts[class_id] else 0.0,
            "gt_pixels": int(gt_counts[class_id]),
            "status": status,
        }
    return output


def selective_view(target: np.ndarray, probability: np.ndarray, thresholds: dict[str, Any]) -> dict[str, Any]:
    prediction = probability.argmax(axis=0)
    confidence = probability.max(axis=0)
    per_class: dict[str, Any] = {}
    accepted_any = np.zeros(target.shape, dtype=bool)
    for class_id, name in enumerate(CLASS_NAMES):
        accepted = (prediction == class_id) & (confidence >= float(thresholds[name]["threshold"]))
        correct = accepted & (target == class_id)
        accepted_any |= accepted
        accepted_count = int(accepted.sum())
        gt_count = int((target == class_id).sum())
        per_class[name] = {
            "accepted_pixels": accepted_count,
            "correct_pixels": int(correct.sum()),
            "precision": float(correct.sum() / accepted_count) if accepted_count else 0.0,
            "recall": float(correct.sum() / gt_count) if gt_count else None,
        }
    accepted_cable = (prediction == 2) & (confidence >= float(thresholds["stay_cable"]["threshold"]))
    false_cable = accepted_cable & (target != 2)
    remote = false_cable & (distance_transform_edt(target != 2) > 8)
    return {
        "per_class": per_class,
        "accepted_fraction_all_classes": float(accepted_any.mean()),
        "stay_cable_accepted_view": bool(accepted_cable.any()),
        "stay_cable_remote_false_positive_pixels": int(remote.sum()),
        "stay_cable_false_positive_pixels": int(false_cable.sum()),
    }


def aggregate_selective(per_view: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"per_class": {}}
    for name in CLASS_NAMES:
        accepted = sum(item["selective"]["per_class"][name]["accepted_pixels"] for item in per_view)
        correct = sum(item["selective"]["per_class"][name]["correct_pixels"] for item in per_view)
        gt = sum(item["gt_counts"][CLASS_NAMES.index(name)] for item in per_view)
        output["per_class"][name] = {
            "accepted_pixels": int(accepted),
            "correct_pixels": int(correct),
            "precision": float(correct / accepted) if accepted else 0.0,
            "recall": float(correct / gt) if gt else 0.0,
        }
    output["stay_cable_view_coverage"] = float(np.mean([item["selective"]["stay_cable_accepted_view"] for item in per_view]))
    remote = sum(item["selective"]["stay_cable_remote_false_positive_pixels"] for item in per_view)
    false = sum(item["selective"]["stay_cable_false_positive_pixels"] for item in per_view)
    output["stay_cable_remote_false_positive_pixels"] = int(remote)
    output["stay_cable_false_positive_pixels"] = int(false)
    output["stay_cable_remote_share_of_accepted_false_positive"] = float(remote / false) if false else 0.0
    return output


def summary_subset(per_view: list[dict[str, Any]], names: set[str] | None = None, worst_count: int | None = None) -> float:
    selected = per_view if names is None else [item for item in per_view if item["name"] in names]
    values = sorted(float(item["metrics"]["mIoU_without_background"]) for item in selected)
    if worst_count is not None:
        values = values[:worst_count]
    return float(np.mean(values)) if values else 0.0


def main() -> int:
    args = parse_args()
    if len(args.fold_dir) != 3:
        raise ValueError("Exactly three fold output directories are required")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = read_json(args.config)
    split = read_json(args.split)
    records = prediction_records(args.fold_dir)
    bins = int(config["thresholds"]["grid_bins"])
    target_precision = float(config["thresholds"]["target_precision"])
    _totals, _correct, _gt_counts, by_fold = selective_histograms(records, args.labels_root, bins)
    threshold_artifact = read_json(args.frozen_thresholds)
    threshold_sha = sha256(args.frozen_thresholds)
    expected_sha = config["thresholds"].get("frozen_artifact_sha256")
    if not expected_sha or threshold_sha != expected_sha:
        raise ValueError(f"Frozen OOF threshold artifact mismatch: expected {expected_sha}, got {threshold_sha}")
    if threshold_artifact.get("fixed_validation_opened_when_frozen") is not False:
        raise ValueError("Threshold artifact does not prove validation remained unopened")
    thresholds = threshold_artifact["thresholds"]
    print(json.dumps({"preregistered_thresholds_verified_before_validation": True, "sha256": threshold_sha, "thresholds": thresholds}, indent=2), flush=True)

    # Only after the threshold artifact is durably written do we open the fixed
    # validation image/label/prediction directories.
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("The registered fixed-validation audit requires CUDA")
    device = torch.device("cuda:0")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    models = []
    for directory in args.fold_dir:
        checkpoint = directory / "checkpoints" / "segformer_b2_final.pth"
        model, metadata = reload_checkpoint(checkpoint, args.model_dir, int(config["training"]["seed"]), device)
        models.append(model)
        if metadata["mode"] != "fold":
            raise ValueError(f"Not a complete fold checkpoint: {checkpoint}")
    validation_names = list(split["fixed_validation_read_only"])
    confusion = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    per_view: list[dict[str, Any]] = []
    overlap = {"teacher_and_sem001_correct": 0, "teacher_only_correct": 0, "sem001_only_correct": 0, "both_wrong": 0}
    crop = int(config["training"]["crop_size"])
    stride = int(config["inference"]["stride"])
    started = time.perf_counter()
    for index, name in enumerate(validation_names, start=1):
        image = read_rgb(args.prepared_scene / "images" / name)
        target = label_array(args.labels_root / name)
        probability = np.zeros((CLASS_COUNT, 989, 1320), dtype=np.float32)
        for model in models:
            probability += sliding_probabilities(model, image, crop, stride, device, amp_dtype)
        probability /= len(models)
        prediction = probability.argmax(axis=0).astype(np.uint8)
        view_confusion = np.zeros_like(confusion)
        update_confusion(view_confusion, target, prediction)
        update_confusion(confusion, target, prediction)
        sem001 = np.asarray(Image.open(args.sem001_predictions / name), dtype=np.uint8)
        teacher_correct = prediction == target
        sem001_correct = sem001 == target
        overlap["teacher_and_sem001_correct"] += int((teacher_correct & sem001_correct).sum())
        overlap["teacher_only_correct"] += int((teacher_correct & ~sem001_correct).sum())
        overlap["sem001_only_correct"] += int((~teacher_correct & sem001_correct).sum())
        overlap["both_wrong"] += int((~teacher_correct & ~sem001_correct).sum())
        prediction_path = args.output_dir / "validation_predictions" / name
        probability_path = args.output_dir / "validation_probabilities" / f"{Path(name).stem}.npz"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        probability_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(prediction, mode="L").save(prediction_path)
        np.savez_compressed(probability_path, probabilities=probability.astype(np.float16))
        view_entry = {
            "name": name,
            "metrics": confusion_metrics(view_confusion),
            "gt_counts": np.bincount(target.ravel(), minlength=CLASS_COUNT).astype(int).tolist(),
            "selective": selective_view(target, probability, thresholds),
            "boundary": boundary_metrics(target, prediction),
            "stay_cable_structure": stay_cable_structure(target, prediction),
            "prediction_sha256": sha256(prediction_path),
            "probabilities_sha256": sha256(probability_path),
        }
        per_view.append(view_entry)
        print(json.dumps({"validation_inference": index, "validation_total": len(validation_names), "name": name}), flush=True)
    torch.cuda.synchronize(device)
    validation_seconds = time.perf_counter() - started
    selective = aggregate_selective(per_view)
    teacher_metrics = confusion_metrics(confusion)
    sem001 = read_json(args.sem001_result)
    sem001_eval = sem001["evaluation"]
    sem001_per_view = sem001_eval["per_view"]
    worst_count = max(1, int(np.ceil(len(validation_names) * 0.1)))
    teacher_worst = summary_subset(per_view, worst_count=worst_count)
    teacher_hard = summary_subset(per_view, names={f"{index:03d}.png" for index in range(140, 147)})
    sem001_worst = float(np.mean(sorted(float(item["mIoU_without_background"]) for item in sem001_per_view)[:worst_count]))
    sem001_hard = float(sem001_eval["challenge_140_146"]["mean_mIoU_without_background"])

    # Per-fold consistency uses the globally frozen threshold without changing it.
    fold_selective: dict[str, Any] = {}
    for fold in range(3):
        entry: dict[str, Any] = {}
        for class_id, name in enumerate(CLASS_NAMES):
            threshold_index = min(int(round(float(thresholds[name]["threshold"]) * (bins - 1))), bins - 1)
            accepted = int(by_fold[fold]["totals"][class_id, threshold_index:].sum())
            true = int(by_fold[fold]["correct"][class_id, threshold_index:].sum())
            gt = int(by_fold[fold]["gt"][class_id])
            entry[name] = {"accepted_pixels": accepted, "precision": float(true / accepted) if accepted else 0.0, "recall": float(true / gt) if gt else 0.0}
        fold_selective[str(fold)] = entry

    gates = {
        "oof_stay_cable_precision_ge_0_95": thresholds["stay_cable"]["selective_precision"] >= 0.95,
        "oof_stay_cable_recall_ge_0_10": thresholds["stay_cable"]["selective_recall"] >= 0.10,
        "validation_stay_cable_precision_ge_0_95": selective["per_class"]["stay_cable"]["precision"] >= 0.95,
        "validation_stay_cable_recall_ge_0_10": selective["per_class"]["stay_cable"]["recall"] >= 0.10,
        "oof_stay_cable_view_coverage_ge_0_50": False,  # Filled by a second exact OOF pass below.
        "deck_tower_foundation_validation_precision_ge_0_95": all(selective["per_class"][name]["precision"] >= 0.95 for name in ("deck", "tower", "foundation")),
        "validation_mIoU_not_below_SEM_001": teacher_metrics["mIoU_with_background"] >= float(sem001_eval["mIoU_with_background"]),
        "validation_worst10_not_below_SEM_001_by_more_than_0_01": teacher_worst >= sem001_worst - 0.01,
        "validation_140_146_not_below_SEM_001_by_more_than_0_01": teacher_hard >= sem001_hard - 0.01,
        "remote_false_positive_not_majority": selective["stay_cable_remote_share_of_accepted_false_positive"] <= 0.5,
        "three_folds_stay_cable_direction_consistent": all(fold_selective[str(fold)]["stay_cable"]["precision"] >= 0.95 and fold_selective[str(fold)]["stay_cable"]["recall"] >= 0.10 for fold in range(3)),
        "thresholds_frozen_before_validation": True,
        "license_and_leakage_gates": True,
    }
    # Exact OOF view coverage and remote-FP audit, still with frozen thresholds.
    oof_cable_views = 0
    oof_remote = 0
    oof_false = 0
    for directory, record in records:
        target = label_array(args.labels_root / record["name"])
        probability = np.load(directory / record["probabilities"])["probabilities"].astype(np.float32)
        entry = selective_view(target, probability, thresholds)
        oof_cable_views += int(entry["stay_cable_accepted_view"])
        oof_remote += int(entry["stay_cable_remote_false_positive_pixels"])
        oof_false += int(entry["stay_cable_false_positive_pixels"])
    oof_view_coverage = oof_cable_views / len(records)
    gates["oof_stay_cable_view_coverage_ge_0_50"] = oof_view_coverage >= 0.5
    gates["oof_remote_false_positive_not_majority"] = (oof_remote / oof_false if oof_false else 0.0) <= 0.5
    eligible = all(gates.values())
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "eligible_for_separate_sem005_3d_preregistration" if eligible else "not_eligible_for_pseudo_labels",
        "threshold_artifact_sha256": threshold_sha,
        "thresholds": thresholds,
        "oof": {
            "views": len(records),
            "stay_cable_view_coverage": oof_view_coverage,
            "stay_cable_remote_false_positive_pixels": oof_remote,
            "stay_cable_false_positive_pixels": oof_false,
            "fold_selective": fold_selective,
        },
        "fixed_validation": {
            "views": len(validation_names),
            "metrics": teacher_metrics,
            "selective": selective,
            "worst10_mIoU_without_background": teacher_worst,
            "challenge_140_146_mIoU_without_background": teacher_hard,
            "validation_seconds": validation_seconds,
            "teacher_vs_SEM001_correctness_overlap_pixels": overlap,
        },
        "SEM_001_reference": {
            "mIoU_with_background": sem001_eval["mIoU_with_background"],
            "worst10_mIoU_without_background": sem001_worst,
            "challenge_140_146_mIoU_without_background": sem001_hard,
            "result_sha256": sha256(args.sem001_result),
        },
        "qualification_gates": gates,
        "qualified": eligible,
        "unlabeled_outputs_generated": False,
        "new_3d_training_started": False,
        "interpretation": "2D pseudo-label qualification only; these metrics are not a 3D semantic model result.",
    }
    write_json(args.output_dir / "validation_per_view.json", per_view)
    write_json(args.output_dir / "result.json", result)
    write_json(
        args.output_dir / "output_manifest.json",
        {
            "thresholds_sha256": threshold_sha,
            "result_sha256": sha256(args.output_dir / "result.json"),
            "validation_per_view_sha256": sha256(args.output_dir / "validation_per_view.json"),
            "validation_prediction_count": len(validation_names),
            "qualified": eligible,
        },
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0 if eligible else 2


if __name__ == "__main__":
    raise SystemExit(main())
