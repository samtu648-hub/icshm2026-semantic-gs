#!/usr/bin/env python3
"""Independent all300 and no-GT proxy audit for SEM381 against SEM378."""

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

EID = "SEM-381-DINOV3-VITB-SFP-ALL300"
BATCH = ROOT / "outputs/quality/SEM-381/r1"
BASE = ROOT / "outputs/quality/SEM-378/r1"
SEM = ROOT / "outputs/quality/all_data/SEM-370-r1"
LABELS = SEM / "labels"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def confusion(gt: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    assert gt.shape == prediction.shape == (989, 1320)
    assert int(gt.max()) < 5 and int(prediction.max()) < 5
    return np.bincount(
        5 * gt.astype(np.int16).ravel() + prediction.ravel(), minlength=25
    ).reshape(5, 5)


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
        "mIoU_with_background": float(np.mean(ious)),
        "mIoU_without_background": float(np.mean(foreground)),
    }


def close_metrics(actual: dict, expected: dict) -> None:
    assert actual["confusion"] == expected["confusion"]
    for key in ("mIoU_with_background", "mIoU_without_background"):
        assert abs(actual[key] - expected[key]) < 1e-15
    for name in CLASSES:
        for key in ("tp", "fp", "fn"):
            assert actual["per_class"][name][key] == expected["per_class"][name][key]
        for key in ("IoU", "precision", "recall"):
            left, right = (
                actual["per_class"][name][key],
                expected["per_class"][name][key],
            )
            if left is None or right is None:
                assert left is right
            else:
                assert abs(left - right) < 1e-15


def main() -> None:
    started = time.monotonic()
    config = read(ROOT / "experiments/runs" / EID / "config.yaml")
    split = read(ROOT / config["split_path"])
    names = sorted(split["semantic_train"])
    assert len(names) == 300
    candidate_record = read(BATCH / "inference/result.json")
    baseline_record = read(BASE / "inference/result.json")
    train_record = read(BATCH / "mask2former/result.json")
    export = read(BATCH / "export/result.json")
    binding = read(BATCH / "binding.json")
    assert candidate_record["status"] == baseline_record["status"] == "completed"
    verify_hashes(BATCH / "export/images", export["image_hashes"])
    verify_hashes(BATCH / "mask2former", train_record["artifacts"])
    verify_hashes(LABELS, binding["label_hashes"])
    candidate_rows = {row["name"]: row for row in candidate_record["per_view"]}
    baseline_rows = {row["name"]: row for row in baseline_record["per_view"]}
    totals = {
        "sem378": np.zeros((5, 5), np.int64),
        "sem381": np.zeros((5, 5), np.int64),
    }
    transition = np.zeros((5, 5, 5), np.int64)
    boundary_counts = {arm: np.zeros(4, np.int64) for arm in totals}
    boundary_cm = {arm: np.zeros((5, 5), np.int64) for arm in totals}
    cable_fn_boundary_interior = {arm: np.zeros(2, np.int64) for arm in totals}
    per_class_changes = {name: {"corrected": 0, "harmed": 0} for name in CLASSES}
    per_view = []
    changed = corrected = harmed = still_wrong = 0
    error_changes = BATCH / "error-changes"
    error_changes.mkdir(exist_ok=False)

    for index, name in enumerate(names):
        gt = np.asarray(Image.open(LABELS / name))
        old = np.asarray(Image.open(BASE / "inference/predictions" / name))
        new = np.asarray(Image.open(BATCH / "inference/predictions" / name))
        old_cm, new_cm = confusion(gt, old), confusion(gt, new)
        assert old_cm.tolist() == baseline_rows[name]["confusion"]
        assert new_cm.tolist() == candidate_rows[name]["confusion"]
        totals["sem378"] += old_cm
        totals["sem381"] += new_cm
        old_score, new_score = scores(old_cm), scores(new_cm)
        change = old != new
        fix = (old != gt) & (new == gt)
        harm = (old == gt) & (new != gt)
        changed += int(change.sum())
        corrected += int(fix.sum())
        harmed += int(harm.sum())
        still_wrong += int((change & (old != gt) & (new != gt)).sum())
        transition += np.bincount(
            (25 * gt.astype(np.int16) + 5 * old + new).ravel(), minlength=125
        ).reshape(5, 5, 5)
        for class_index, class_name in enumerate(CLASSES):
            mask = gt == class_index
            per_class_changes[class_name]["corrected"] += int((fix & mask).sum())
            per_class_changes[class_name]["harmed"] += int((harm & mask).sum())

        cable = gt == 2
        gt_boundary = cable ^ binary_erosion(cable)
        gt_distance = np.asarray(distance_transform_edt(~gt_boundary))
        boundary_band = gt_distance <= 2
        for arm, prediction in (("sem378", old), ("sem381", new)):
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

        overlay = np.zeros((*gt.shape, 3), np.uint8)
        overlay[fix] = [40, 190, 100]
        overlay[harm] = [230, 70, 70]
        overlay[change & ~fix & ~harm] = [170, 100, 210]
        Image.fromarray(overlay).save(error_changes / name)
        per_view.append(
            {
                "name": name,
                "sem378_mIoU": old_score["mIoU_with_background"],
                "sem381_mIoU": new_score["mIoU_with_background"],
                "delta_pp": 100
                * (
                    new_score["mIoU_with_background"]
                    - old_score["mIoU_with_background"]
                ),
                "changed": int(change.sum()),
                "corrected": int(fix.sum()),
                "harmed": int(harm.sum()),
                "stay_cable_fn_delta": int(
                    new_cm[2].sum() - new_cm[2, 2] - old_cm[2].sum() + old_cm[2, 2]
                ),
                "stay_cable_fp_delta": int(
                    new_cm[:, 2].sum()
                    - new_cm[2, 2]
                    - old_cm[:, 2].sum()
                    + old_cm[2, 2]
                ),
            }
        )
        if index % 50 == 0:
            print(json.dumps({"audited": index + 1, "total": len(names)}), flush=True)

    aggregate = {arm: scores(cm) for arm, cm in totals.items()}
    close_metrics(aggregate["sem378"], baseline_record["metrics"])
    close_metrics(aggregate["sem381"], candidate_record["metrics"])
    boundary = {}
    for arm, values in boundary_counts.items():
        precision, recall = values[0] / values[1], values[2] / values[3]
        boundary[arm] = {
            "tolerance_px": 2,
            "boundary_precision": float(precision),
            "boundary_recall": float(recall),
            "boundary_f1": float(2 * precision * recall / (precision + recall)),
            "counts": values.tolist(),
            "fiveclass_gt_cable_boundary_band": scores(boundary_cm[arm]),
            "cable_fn_boundary_interior": cable_fn_boundary_interior[arm].tolist(),
        }
    non_cable = [0, 1, 3, 4]
    cable_changes = {
        "fn_fixed": int(transition[2, non_cable, 2].sum()),
        "new_fn": int(transition[2, 2, non_cable].sum()),
        "fp_removed": int(transition[np.ix_(non_cable, [2], non_cable)].sum()),
        "new_fp": int(transition[np.ix_(non_cable, non_cable, [2])].sum()),
        "background_fp_removed": int(transition[0, 2, non_cable].sum()),
        "new_background_fp": int(transition[0, non_cable, 2].sum()),
    }
    assert (
        aggregate["sem378"]["per_class"]["stay_cable"]["fn"]
        - cable_changes["fn_fixed"]
        + cable_changes["new_fn"]
        == aggregate["sem381"]["per_class"]["stay_cable"]["fn"]
    )
    assert (
        aggregate["sem378"]["per_class"]["stay_cable"]["fp"]
        - cable_changes["fp_removed"]
        + cable_changes["new_fp"]
        == aggregate["sem381"]["per_class"]["stay_cable"]["fp"]
    )

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
        old = np.asarray(Image.open(BASE / "inference/predictions" / name))
        new = np.asarray(Image.open(BATCH / "inference/predictions" / name))
        mask = np.zeros(gt.shape, bool)
        for x0, y0, x1, y1 in spec["boxes"]:
            mask[y0:y1, x0:x1] = True
        old_cm = np.bincount(
            5 * gt[mask].astype(np.int16) + old[mask], minlength=25
        ).reshape(5, 5)
        new_cm = np.bincount(
            5 * gt[mask].astype(np.int16) + new[mask], minlength=25
        ).reshape(5, 5)
        fix, harm, change = (
            (old != gt) & (new == gt) & mask,
            (old == gt) & (new != gt) & mask,
            (old != new) & mask,
        )
        old_score, new_score = scores(old_cm), scores(new_cm)
        region_rows.append(
            {
                "region": region_name,
                "view": name,
                "boxes": spec["boxes"],
                "pixels": int(mask.sum()),
                "sem378": old_score,
                "sem381": new_score,
                "delta_pp": 100
                * (
                    new_score["mIoU_with_background"]
                    - old_score["mIoU_with_background"]
                ),
                "changed": int(change.sum()),
                "corrected": int(fix.sum()),
                "harmed": int(harm.sum()),
            }
        )

    trajectory = {}
    roots = {
        "sem378": BASE / "inference/predictions",
        "sem381": BATCH / "inference/predictions",
    }
    for arm, root in roots.items():
        adjacent, previous = [], None
        for name in export["trajectory_names"]:
            current = np.asarray(Image.open(root / name))
            if previous is not None:
                adjacent.append(float(np.mean(current != previous)))
            previous = current
        trajectory[arm] = {
            "mean_adjacent_label_change": float(np.mean(adjacent)),
            "max_adjacent_label_change": float(np.max(adjacent)),
            "max_transition_index": int(np.argmax(adjacent) + 1),
            "per_transition": adjacent,
        }
    disagreements = [
        float(
            np.mean(
                np.asarray(Image.open(roots["sem378"] / name))
                != np.asarray(Image.open(roots["sem381"] / name))
            )
        )
        for name in export["trajectory_names"]
    ]
    trajectory["candidate_vs_baseline"] = {
        "mean_disagreement": float(np.mean(disagreements)),
        "max_disagreement": float(np.max(disagreements)),
        "max_frame_index": int(np.argmax(disagreements)),
        "per_frame": disagreements,
    }
    deltas = np.asarray([row["delta_pp"] for row in per_view])
    result = {
        "schema_version": 1,
        "status": "completed",
        "experiment_id": EID,
        "quality_scope": "all300 training-view fit; no independent held-out GT",
        "metrics": aggregate,
        "delta_mIoU_pp": 100
        * (
            aggregate["sem381"]["mIoU_with_background"]
            - aggregate["sem378"]["mIoU_with_background"]
        ),
        "paired": {
            "improved": int((deltas > 0).sum()),
            "degraded": int((deltas < 0).sum()),
            "tied": int((deltas == 0).sum()),
            "per_view": per_view,
        },
        "pixel_changes": {
            "changed": changed,
            "corrected": corrected,
            "harmed": harmed,
            "wrong_to_wrong": still_wrong,
            "per_gt_class": per_class_changes,
        },
        "cable_changes": cable_changes,
        "boundary": boundary,
        "regions": region_rows,
        "trajectory_proxy": trajectory,
        "resources": {
            "training_seconds": train_record["training_seconds"],
            "training_peak_allocated_mib": train_record["peak_allocated_mib"],
            "training_peak_reserved_mib": train_record["peak_reserved_mib"],
            "inference_seconds": sum(candidate_record["inference_seconds"]),
            "inference_median_seconds": candidate_record["inference_median_seconds"],
            "checkpoint_6000_bytes": (BATCH / "mask2former/model-6000.pt")
            .stat()
            .st_size,
            "model_parameters": train_record["pretrained_load"],
        },
        "prediction_hashes": {
            name: sha256(BATCH / "inference/predictions" / name)
            for name in sorted(export["image_hashes"])
        },
        "seconds": time.monotonic() - started,
        "script_sha256": sha256(Path(__file__)),
    }
    write_new_json(BATCH / "audit.json", result)


if __name__ == "__main__":
    main()
