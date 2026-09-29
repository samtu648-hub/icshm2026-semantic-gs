#!/usr/bin/env python3
"""Independently audit SEM382 checkpoints against SEM381 and SEM378."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quality_protocol import CLASSES, sha256, verify_hashes, write_new_json

EID = "SEM-382-DINOV3-VITB-SFP-LR1E4-ALL300"
BATCH = ROOT / "outputs/quality/SEM-382/r1"
PARENT = ROOT / "outputs/quality/SEM-381/r1"
BASE = ROOT / "outputs/quality/SEM-378/r1"
SEM = ROOT / "outputs/quality/all_data/SEM-370-r1"
LABELS = SEM / "labels"
ARMS = {
    "sem378": BASE / "inference/predictions",
    "sem381": PARENT / "inference/predictions",
    "sem382_3000": BATCH / "inference-3000/predictions",
    "sem382_6000": BATCH / "inference-6000/predictions",
}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def confusion(gt: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    if gt.shape != prediction.shape or gt.shape != (989, 1320):
        raise ValueError("Native 1320x989 labels are required")
    if int(gt.max()) >= 5 or int(prediction.max()) >= 5:
        raise ValueError("Expected class IDs 0..4")
    return np.bincount(
        5 * gt.astype(np.int16).ravel() + prediction.ravel(), minlength=25
    ).reshape(5, 5)


def optional_mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def scores(cm: np.ndarray) -> dict:
    per_class = {}
    ious = []
    foreground = []
    for index, name in enumerate(CLASSES):
        tp = int(cm[index, index])
        fp = int(cm[:, index].sum()) - tp
        fn = int(cm[index].sum()) - tp
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
        if iou is not None:
            ious.append(iou)
            if index:
                foreground.append(iou)
    return {
        "confusion": cm.tolist(),
        "per_class": per_class,
        "mIoU_with_background": optional_mean(ious),
        "mIoU_without_background": optional_mean(foreground),
    }


def close_optional(left: float | None, right: float | None) -> None:
    if left is None or right is None:
        if left is not right:
            raise ValueError("Defined/undefined metric mismatch")
    elif abs(left - right) >= 1e-15:
        raise ValueError(f"Metric mismatch: {left} != {right}")


def close_metrics(actual: dict, expected: dict) -> None:
    if actual["confusion"] != expected["confusion"]:
        raise ValueError("Confusion matrix mismatch")
    for key in ("mIoU_with_background", "mIoU_without_background"):
        close_optional(actual[key], expected[key])
    for name in CLASSES:
        for key in ("tp", "fp", "fn"):
            if actual["per_class"][name][key] != expected["per_class"][name][key]:
                raise ValueError(f"Count mismatch: {name}/{key}")
        for key in ("IoU", "precision", "recall"):
            close_optional(
                actual["per_class"][name][key],
                expected["per_class"][name][key],
            )


def pair_change(gt: np.ndarray, old: np.ndarray, new: np.ndarray) -> dict:
    change = old != new
    corrected = (old != gt) & (new == gt)
    harmed = (old == gt) & (new != gt)
    return {
        "change": change,
        "corrected_mask": corrected,
        "harmed_mask": harmed,
        "changed": int(change.sum()),
        "corrected": int(corrected.sum()),
        "harmed": int(harmed.sum()),
        "wrong_to_wrong": int((change & (old != gt) & (new != gt)).sum()),
    }


def boundary_summary(counts: np.ndarray, cm: np.ndarray, fn: np.ndarray) -> dict:
    precision = float(counts[0] / counts[1]) if counts[1] else None
    recall = float(counts[2] / counts[3]) if counts[3] else None
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
        "counts": counts.tolist(),
        "fiveclass_gt_cable_boundary_band": scores(cm),
        "cable_fn_boundary_interior": fn.tolist(),
    }


def main() -> None:
    started = time.monotonic()
    config = read(ROOT / "experiments/runs" / EID / "config.yaml")
    names = sorted(read(ROOT / config["split_path"])["semantic_train"])
    if len(names) != 300:
        raise ValueError("Expected all 300 training labels")
    records = {
        "sem378": read(BASE / "inference/result.json"),
        "sem381": read(PARENT / "inference/result.json"),
        "sem382_3000": read(BATCH / "inference-3000/result.json"),
        "sem382_6000": read(BATCH / "inference-6000/result.json"),
    }
    if any(record["status"] != "completed" for record in records.values()):
        raise ValueError("All four inference records must be complete")
    train_record = read(BATCH / "mask2former/result.json")
    export = read(BATCH / "export/result.json")
    binding = read(BATCH / "binding.json")
    verify_hashes(BATCH / "export/images", export["image_hashes"])
    verify_hashes(BATCH / "mask2former", train_record["artifacts"])
    verify_hashes(LABELS, binding["label_hashes"])
    verify_hashes(PARENT / "inference/predictions", binding["sem381_prediction_hashes"])

    record_rows = {
        arm: {row["name"]: row for row in record["per_view"]}
        for arm, record in records.items()
    }
    totals = {arm: np.zeros((5, 5), np.int64) for arm in ARMS}
    boundary_counts = {arm: np.zeros(4, np.int64) for arm in ARMS}
    boundary_cm = {arm: np.zeros((5, 5), np.int64) for arm in ARMS}
    cable_fn_boundary_interior = {arm: np.zeros(2, np.int64) for arm in ARMS}
    pair_specs = {
        "sem382_6000_vs_sem381": ("sem381", "sem382_6000"),
        "sem382_6000_vs_sem378": ("sem378", "sem382_6000"),
        "sem382_3000_vs_sem381": ("sem381", "sem382_3000"),
        "sem382_6000_vs_sem382_3000": ("sem382_3000", "sem382_6000"),
    }
    pair_totals = {
        key: {
            "changed": 0,
            "corrected": 0,
            "harmed": 0,
            "wrong_to_wrong": 0,
            "transition": np.zeros((5, 5, 5), np.int64),
            "per_gt_class": {name: {"corrected": 0, "harmed": 0} for name in CLASSES},
        }
        for key in pair_specs
    }
    error_roots = {
        "sem382_6000_vs_sem381": BATCH / "error-changes-vs-sem381",
        "sem382_6000_vs_sem382_3000": BATCH / "error-changes-3000-to-6000",
    }
    for root in error_roots.values():
        root.mkdir(exist_ok=False)
    per_view = []

    for index, name in enumerate(names):
        gt = np.asarray(Image.open(LABELS / name))
        predictions = {
            arm: np.asarray(Image.open(root / name)) for arm, root in ARMS.items()
        }
        image_scores = {}
        for arm, prediction in predictions.items():
            cm = confusion(gt, prediction)
            if cm.tolist() != record_rows[arm][name]["confusion"]:
                raise ValueError(f"Per-view confusion mismatch: {arm}/{name}")
            totals[arm] += cm
            image_scores[arm] = scores(cm)

            cable = gt == 2
            gt_boundary = cable ^ binary_erosion(cable)
            gt_distance = np.asarray(distance_transform_edt(~gt_boundary))
            boundary_band = gt_distance <= 2
            predicted_cable = prediction == 2
            predicted_boundary = predicted_cable ^ binary_erosion(predicted_cable)
            predicted_distance = np.asarray(distance_transform_edt(~predicted_boundary))
            boundary_counts[arm] += np.asarray(
                [
                    np.sum(predicted_boundary & (gt_distance <= 2)),
                    np.sum(predicted_boundary),
                    np.sum(gt_boundary & (predicted_distance <= 2)),
                    np.sum(gt_boundary),
                ],
                dtype=np.int64,
            )
            boundary_cm[arm] += np.bincount(
                5 * gt[boundary_band].astype(np.int16) + prediction[boundary_band],
                minlength=25,
            ).reshape(5, 5)
            cable_fn_boundary_interior[arm] += np.asarray(
                [
                    np.sum(cable & (prediction != 2) & boundary_band),
                    np.sum(cable & (prediction != 2) & ~boundary_band),
                ],
                dtype=np.int64,
            )

        row = {
            "name": name,
            **{
                f"{arm}_mIoU": score["mIoU_with_background"]
                for arm, score in image_scores.items()
            },
        }
        for pair_name, (old_arm, new_arm) in pair_specs.items():
            old, new = predictions[old_arm], predictions[new_arm]
            change = pair_change(gt, old, new)
            aggregate = pair_totals[pair_name]
            for key in ("changed", "corrected", "harmed", "wrong_to_wrong"):
                aggregate[key] += change[key]
            aggregate["transition"] += np.bincount(
                (25 * gt.astype(np.int16) + 5 * old + new).ravel(), minlength=125
            ).reshape(5, 5, 5)
            for class_index, class_name in enumerate(CLASSES):
                mask = gt == class_index
                aggregate["per_gt_class"][class_name]["corrected"] += int(
                    (change["corrected_mask"] & mask).sum()
                )
                aggregate["per_gt_class"][class_name]["harmed"] += int(
                    (change["harmed_mask"] & mask).sum()
                )
            row[f"{pair_name}_delta_pp"] = 100 * (
                image_scores[new_arm]["mIoU_with_background"]
                - image_scores[old_arm]["mIoU_with_background"]
            )
            row[f"{pair_name}_corrected"] = change["corrected"]
            row[f"{pair_name}_harmed"] = change["harmed"]
            if pair_name in error_roots:
                overlay = np.zeros((*gt.shape, 3), np.uint8)
                overlay[change["corrected_mask"]] = [40, 190, 100]
                overlay[change["harmed_mask"]] = [230, 70, 70]
                overlay[
                    change["change"]
                    & ~change["corrected_mask"]
                    & ~change["harmed_mask"]
                ] = [170, 100, 210]
                Image.fromarray(overlay).save(error_roots[pair_name] / name)
        per_view.append(row)
        if index % 50 == 0:
            print(json.dumps({"audited": index + 1, "total": len(names)}), flush=True)

    aggregate = {arm: scores(cm) for arm, cm in totals.items()}
    for arm in ARMS:
        close_metrics(aggregate[arm], records[arm]["metrics"])
    boundary = {
        arm: boundary_summary(
            boundary_counts[arm], boundary_cm[arm], cable_fn_boundary_interior[arm]
        )
        for arm in ARMS
    }

    non_cable = [0, 1, 3, 4]
    pair_results = {}
    for pair_name, (old_arm, new_arm) in pair_specs.items():
        row = pair_totals[pair_name]
        transition = row.pop("transition")
        cable_changes = {
            "fn_fixed": int(transition[2, non_cable, 2].sum()),
            "new_fn": int(transition[2, 2, non_cable].sum()),
            "fp_removed": int(transition[np.ix_(non_cable, [2], non_cable)].sum()),
            "new_fp": int(transition[np.ix_(non_cable, non_cable, [2])].sum()),
            "background_fp_removed": int(transition[0, 2, non_cable].sum()),
            "new_background_fp": int(transition[0, non_cable, 2].sum()),
        }
        deltas = np.asarray(
            [item[f"{pair_name}_delta_pp"] for item in per_view], dtype=np.float64
        )
        pair_results[pair_name] = {
            "old_arm": old_arm,
            "new_arm": new_arm,
            **row,
            "cable_changes": cable_changes,
            "delta_mIoU_pp": 100
            * (
                aggregate[new_arm]["mIoU_with_background"]
                - aggregate[old_arm]["mIoU_with_background"]
            ),
            "paired": {
                "improved": int((deltas > 0).sum()),
                "degraded": int((deltas < 0).sum()),
                "tied": int((deltas == 0).sum()),
            },
        }

    region_configs = [
        "experiments/runs/RGB-132-THREE-BACKBONE-DIAG/config.yaml",
        "experiments/runs/QUAL-005-PHOTO-INPUT-DIAG/config.yaml",
        "experiments/runs/QUAL-004-CURRENT-BOTTLENECK/config.yaml",
    ]
    regions = read(ROOT / region_configs[0])["regions"].copy()
    for prefix, key, path in (
        ("difficult", "rois", region_configs[1]),
        ("structure", "crops", region_configs[2]),
    ):
        for stem, box in read(ROOT / path)[key].items():
            regions[f"{prefix}-{stem}"] = {"view": f"{stem}.png", "boxes": [box]}
    region_rows = []
    for region_name, spec in regions.items():
        name = spec["view"]
        gt = np.asarray(Image.open(LABELS / name))
        mask = np.zeros(gt.shape, bool)
        for x0, y0, x1, y1 in spec["boxes"]:
            mask[y0:y1, x0:x1] = True
        arm_scores = {}
        for arm, root in ARMS.items():
            prediction = np.asarray(Image.open(root / name))
            cm = np.bincount(
                5 * gt[mask].astype(np.int16) + prediction[mask], minlength=25
            ).reshape(5, 5)
            arm_scores[arm] = scores(cm)
        region_rows.append(
            {
                "region": region_name,
                "view": name,
                "boxes": spec["boxes"],
                "pixels": int(mask.sum()),
                "metrics": arm_scores,
                "sem382_6000_vs_sem381_delta_pp": 100
                * (
                    arm_scores["sem382_6000"]["mIoU_with_background"]
                    - arm_scores["sem381"]["mIoU_with_background"]
                ),
                "sem382_6000_vs_sem378_delta_pp": 100
                * (
                    arm_scores["sem382_6000"]["mIoU_with_background"]
                    - arm_scores["sem378"]["mIoU_with_background"]
                ),
            }
        )

    trajectory = {}
    trajectories = {}
    for arm, root in ARMS.items():
        adjacent = []
        previous = None
        trajectories[arm] = {}
        for name in export["trajectory_names"]:
            current = np.asarray(Image.open(root / name))
            trajectories[arm][name] = current
            if previous is not None:
                adjacent.append(float(np.mean(current != previous)))
            previous = current
        trajectory[arm] = {
            "mean_adjacent_label_change": float(np.mean(adjacent)),
            "max_adjacent_label_change": float(np.max(adjacent)),
            "max_transition_index": int(np.argmax(adjacent) + 1),
            "per_transition": adjacent,
        }
    for reference in ("sem378", "sem381", "sem382_3000"):
        disagreements = [
            float(
                np.mean(
                    trajectories[reference][name] != trajectories["sem382_6000"][name]
                )
            )
            for name in export["trajectory_names"]
        ]
        trajectory[f"sem382_6000_vs_{reference}"] = {
            "mean_disagreement": float(np.mean(disagreements)),
            "max_disagreement": float(np.max(disagreements)),
            "max_frame_index": int(np.argmax(disagreements)),
            "per_frame": disagreements,
        }

    prediction_hashes = {
        arm: {name: sha256(root / name) for name in sorted(export["image_hashes"])}
        for arm, root in ARMS.items()
        if arm.startswith("sem382")
    }
    result = {
        "schema_version": 1,
        "status": "completed",
        "experiment_id": EID,
        "primary_checkpoint": 6000,
        "analysis_checkpoint": 3000,
        "quality_scope": "all300 training-view fit; no independent held-out GT",
        "metrics": aggregate,
        "comparisons": pair_results,
        "per_view": per_view,
        "boundary": boundary,
        "regions": region_rows,
        "trajectory_proxy": trajectory,
        "resources": {
            "training_seconds": train_record["training_seconds"],
            "training_peak_allocated_mib": train_record["peak_allocated_mib"],
            "training_peak_reserved_mib": train_record["peak_reserved_mib"],
            "inference": {
                str(step): {
                    "seconds": sum(records[f"sem382_{step}"]["inference_seconds"]),
                    "median_seconds": records[f"sem382_{step}"][
                        "inference_median_seconds"
                    ],
                }
                for step in (3000, 6000)
            },
            "checkpoint_bytes": {
                str(step): (BATCH / f"mask2former/model-{step}.pt").stat().st_size
                for step in (3000, 6000)
            },
            "model_parameters": train_record["pretrained_load"],
            "optimizer_contract": train_record["optimizer_contract"],
            "parameter_updates": train_record["parameter_updates"],
        },
        "prediction_hashes": prediction_hashes,
        "undefined_metrics_encoded_as_null": True,
        "seconds": time.monotonic() - started,
        "script_sha256": sha256(Path(__file__)),
    }
    write_new_json(BATCH / "audit.json", result)


if __name__ == "__main__":
    main()
