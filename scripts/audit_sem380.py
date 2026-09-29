#!/usr/bin/env python3
"""Independent system audit for RGB148+SEM380 against RGB140+SEM378."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
    maximum_filter,
    minimum_filter,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quality_protocol import CLASSES, sha256, verify_hashes, write_new_json

EID = "SEM-380-RGB148-GT-REGION-RGB-ALL300"
BATCH = ROOT / "outputs/quality/SEM-380/r2"
BASE = ROOT / "outputs/quality/SEM-378/r1"
SEM = ROOT / "outputs/quality/all_data/SEM-370-r1"
LABELS = SEM / "labels"
PHOTOS = ROOT / "outputs/quality/all_data/RGB-120-r2/scene/images"


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
        "mIoU_without_background": float(np.mean(foreground)) if foreground else None,
    }


def alignment_template() -> dict:
    return {
        direction: {
            outcome: 0
            for outcome in ("corrected", "harmed", "remains_correct", "remains_wrong")
        }
        for direction in ("rgb_improved", "rgb_worsened", "rgb_tied")
    }


def add_alignment(
    target: dict,
    rgb_old: np.ndarray,
    rgb_new: np.ndarray,
    gt: np.ndarray,
    old: np.ndarray,
    new: np.ndarray,
    mask: np.ndarray,
) -> None:
    directions = {
        "rgb_improved": rgb_new < rgb_old,
        "rgb_worsened": rgb_new > rgb_old,
        "rgb_tied": rgb_new == rgb_old,
    }
    outcomes = {
        "corrected": (old != gt) & (new == gt),
        "harmed": (old == gt) & (new != gt),
        "remains_correct": (old == gt) & (new == gt),
        "remains_wrong": (old != gt) & (new != gt),
    }
    for direction, rgb_mask in directions.items():
        for outcome, semantic_mask in outcomes.items():
            target[direction][outcome] += int((mask & rgb_mask & semantic_mask).sum())


def close_metrics(actual: dict, expected: dict) -> None:
    assert actual["confusion"] == expected["confusion"]
    for key in ("mIoU_with_background", "mIoU_without_background"):
        assert abs(actual[key] - expected[key]) < 1e-15
    for name in CLASSES:
        for key in ("tp", "fp", "fn"):
            assert actual["per_class"][name][key] == expected["per_class"][name][key]
        for key in ("IoU", "precision", "recall"):
            left = actual["per_class"][name][key]
            right = expected["per_class"][name][key]
            if left is None or right is None:
                assert left is right
            else:
                assert abs(left - right) < 1e-15


def main() -> None:
    started = time.monotonic()
    config_path = ROOT / "experiments/runs" / EID / "config.yaml"
    config = read(config_path)
    split = read(ROOT / config["split_path"])
    names = sorted(split["semantic_train"])
    assert len(names) == 300
    candidate_record = read(BATCH / "inference/result.json")
    baseline_record = read(BASE / "inference/result.json")
    train_record = read(BATCH / "mask2former/result.json")
    export = read(BATCH / "export/result.json")
    binding = read(BATCH / "binding.json")
    assert candidate_record["status"] == baseline_record["status"] == "completed"
    assert len(export["image_hashes"]) == 544
    verify_hashes(BATCH / "export/images", export["image_hashes"])
    verify_hashes(BATCH / "mask2former", train_record["artifacts"])
    verify_hashes(LABELS, binding["label_hashes"])

    candidate_rows = {row["name"]: row for row in candidate_record["per_view"]}
    baseline_rows = {row["name"]: row for row in baseline_record["per_view"]}
    totals = {
        "rgb140_sem378": np.zeros((5, 5), dtype=np.int64),
        "rgb148_sem380": np.zeros((5, 5), dtype=np.int64),
    }
    per_view = []
    changed = corrected = harmed = still_wrong = 0
    per_class_changes = {name: {"corrected": 0, "harmed": 0} for name in CLASSES}
    transition = np.zeros((5, 5, 5), dtype=np.int64)
    alignment = {
        "all": alignment_template(),
        **{name: alignment_template() for name in CLASSES},
    }
    rgb_error_sum = {arm: np.zeros(5, dtype=np.float64) for arm in ("rgb140", "rgb148")}
    rgb_pixel_count = np.zeros(5, dtype=np.int64)
    detail_strata = {
        key: {"pixels": 0, "sem378_fn": 0, "sem380_fn": 0, "fn_fixed": 0, "fn_added": 0}
        for key in (
            "photo_detail_rgb148_weak",
            "photo_and_rgb148_have_detail",
            "both_low_detail",
            "rgb148_only_detail",
        )
    }
    boundary_counts = {
        arm: np.zeros(4, dtype=np.int64) for arm in ("rgb140_sem378", "rgb148_sem380")
    }
    boundary_cm = {arm: np.zeros((5, 5), dtype=np.int64) for arm in boundary_counts}
    cable_fn_boundary_interior = {
        arm: np.zeros(2, dtype=np.int64) for arm in boundary_counts
    }
    error_changes = BATCH / "error-changes"
    error_changes.mkdir(exist_ok=False)

    for index, name in enumerate(names):
        gt = np.asarray(Image.open(LABELS / name))
        old = np.asarray(Image.open(BASE / "inference/predictions" / name))
        new = np.asarray(Image.open(BATCH / "inference/predictions" / name))
        old_cm = confusion(gt, old)
        new_cm = confusion(gt, new)
        assert old_cm.tolist() == baseline_rows[name]["confusion"]
        assert new_cm.tolist() == candidate_rows[name]["confusion"]
        totals["rgb140_sem378"] += old_cm
        totals["rgb148_sem380"] += new_cm
        old_scores = scores(old_cm)
        new_scores = scores(new_cm)
        delta_pp = 100 * (
            new_scores["mIoU_with_background"] - old_scores["mIoU_with_background"]
        )
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
            class_mask = gt == class_index
            per_class_changes[class_name]["corrected"] += int((fix & class_mask).sum())
            per_class_changes[class_name]["harmed"] += int((harm & class_mask).sum())

        photo = np.asarray(Image.open(PHOTOS / name).convert("RGB"), dtype=np.int16)
        rgb140 = np.asarray(
            Image.open(BASE / "export/images" / name).convert("RGB"), dtype=np.int16
        )
        rgb148 = np.asarray(
            Image.open(BATCH / "export/images" / name).convert("RGB"), dtype=np.int16
        )
        old_error = np.abs(rgb140 - photo).mean(axis=2)
        new_error = np.abs(rgb148 - photo).mean(axis=2)
        add_alignment(
            alignment["all"],
            old_error,
            new_error,
            gt,
            old,
            new,
            np.ones(gt.shape, bool),
        )
        for class_index, class_name in enumerate(CLASSES):
            class_mask = gt == class_index
            add_alignment(
                alignment[class_name], old_error, new_error, gt, old, new, class_mask
            )
            rgb_error_sum["rgb140"][class_index] += float(old_error[class_mask].sum())
            rgb_error_sum["rgb148"][class_index] += float(new_error[class_mask].sum())
            rgb_pixel_count[class_index] += int(class_mask.sum())

        gray = [
            np.asarray(array, dtype=np.float32).mean(axis=2)
            for array in (photo, rgb148)
        ]
        photo_detail, rgb148_detail = [
            np.asarray(maximum_filter(array, size=5), dtype=np.float32)
            - np.asarray(minimum_filter(array, size=5), dtype=np.float32)
            >= 12
            for array in gray
        ]
        cable = gt == 2
        strata_masks = {
            "photo_detail_rgb148_weak": cable & photo_detail & ~rgb148_detail,
            "photo_and_rgb148_have_detail": cable & photo_detail & rgb148_detail,
            "both_low_detail": cable & ~photo_detail & ~rgb148_detail,
            "rgb148_only_detail": cable & ~photo_detail & rgb148_detail,
        }
        for key, mask in strata_masks.items():
            detail_strata[key]["pixels"] += int(mask.sum())
            detail_strata[key]["sem378_fn"] += int((mask & (old != 2)).sum())
            detail_strata[key]["sem380_fn"] += int((mask & (new != 2)).sum())
            detail_strata[key]["fn_fixed"] += int(
                (mask & (old != 2) & (new == 2)).sum()
            )
            detail_strata[key]["fn_added"] += int(
                (mask & (old == 2) & (new != 2)).sum()
            )

        gt_boundary = cable ^ binary_erosion(cable)
        gt_distance = np.asarray(distance_transform_edt(~gt_boundary))
        boundary_band = gt_distance <= 2
        for arm, prediction in (("rgb140_sem378", old), ("rgb148_sem380", new)):
            predicted_cable = prediction == 2
            predicted_boundary = predicted_cable ^ binary_erosion(predicted_cable)
            predicted_distance = np.asarray(distance_transform_edt(~predicted_boundary))
            values = np.asarray(
                [
                    np.sum(predicted_boundary & (gt_distance <= 2)),
                    np.sum(predicted_boundary),
                    np.sum(gt_boundary & (predicted_distance <= 2)),
                    np.sum(gt_boundary),
                ],
                dtype=np.int64,
            )
            boundary_counts[arm] += values
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

        overlay = np.zeros((*gt.shape, 3), dtype=np.uint8)
        overlay[fix] = [40, 190, 100]
        overlay[harm] = [230, 70, 70]
        overlay[change & ~fix & ~harm] = [170, 100, 210]
        Image.fromarray(overlay).save(error_changes / name)
        per_view.append(
            {
                "name": name,
                "rgb140_sem378_mIoU": old_scores["mIoU_with_background"],
                "rgb148_sem380_mIoU": new_scores["mIoU_with_background"],
                "semantic_delta_pp": delta_pp,
                "rgb140_mae_255": float(old_error.mean()),
                "rgb148_mae_255": float(new_error.mean()),
                "rgb_mae_delta_255": float(new_error.mean() - old_error.mean()),
                "changed": int(change.sum()),
                "corrected": int(fix.sum()),
                "harmed": int(harm.sum()),
                "stay_cable_fn_delta": int(
                    new_cm[2].sum() - new_cm[2, 2] - (old_cm[2].sum() - old_cm[2, 2])
                ),
                "stay_cable_fp_delta": int(
                    new_cm[:, 2].sum()
                    - new_cm[2, 2]
                    - (old_cm[:, 2].sum() - old_cm[2, 2])
                ),
            }
        )
        if index % 50 == 0:
            print(json.dumps({"audited": index + 1, "total": len(names)}), flush=True)

    aggregate = {arm: scores(cm) for arm, cm in totals.items()}
    close_metrics(aggregate["rgb140_sem378"], baseline_record["metrics"])
    close_metrics(aggregate["rgb148_sem380"], candidate_record["metrics"])

    boundary = {}
    for arm, values in boundary_counts.items():
        precision = values[0] / values[1]
        recall = values[2] / values[3]
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
    old_cable = aggregate["rgb140_sem378"]["per_class"]["stay_cable"]
    new_cable = aggregate["rgb148_sem380"]["per_class"]["stay_cable"]
    assert (
        old_cable["fn"] - cable_changes["fn_fixed"] + cable_changes["new_fn"]
        == new_cable["fn"]
    )
    assert (
        old_cable["fp"] - cable_changes["fp_removed"] + cable_changes["new_fp"]
        == new_cable["fp"]
    )
    assert int(transition.sum()) == 300 * 989 * 1320

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
        photo = np.asarray(Image.open(PHOTOS / name).convert("RGB"), dtype=np.float32)
        rgb140 = np.asarray(
            Image.open(BASE / "export/images" / name).convert("RGB"), dtype=np.float32
        )
        rgb148 = np.asarray(
            Image.open(BATCH / "export/images" / name).convert("RGB"), dtype=np.float32
        )
        mask = np.zeros(gt.shape, dtype=bool)
        for x0, y0, x1, y1 in spec["boxes"]:
            mask[y0:y1, x0:x1] = True
        old_cm = np.bincount(
            5 * gt[mask].astype(np.int16) + old[mask], minlength=25
        ).reshape(5, 5)
        new_cm = np.bincount(
            5 * gt[mask].astype(np.int16) + new[mask], minlength=25
        ).reshape(5, 5)
        fix = (old != gt) & (new == gt) & mask
        harm = (old == gt) & (new != gt) & mask
        change = (old != new) & mask
        old_score = scores(old_cm)
        new_score = scores(new_cm)
        region_rows.append(
            {
                "region": region_name,
                "view": name,
                "boxes": spec["boxes"],
                "pixels": int(mask.sum()),
                "rgb140_sem378": old_score,
                "rgb148_sem380": new_score,
                "semantic_delta_pp": 100
                * (
                    new_score["mIoU_with_background"]
                    - old_score["mIoU_with_background"]
                ),
                "rgb140_mae_255": float(np.abs(rgb140 - photo)[mask].mean()),
                "rgb148_mae_255": float(np.abs(rgb148 - photo)[mask].mean()),
                "changed": int(change.sum()),
                "corrected": int(fix.sum()),
                "harmed": int(harm.sum()),
                "per_class": {
                    class_name: {
                        "corrected": int((fix & (gt == class_index)).sum()),
                        "harmed": int((harm & (gt == class_index)).sum()),
                    }
                    for class_index, class_name in enumerate(CLASSES)
                },
            }
        )

    trajectory_names = export["trajectory_names"]
    prediction_roots = {
        "rgb140_sem378": BASE / "inference/predictions",
        "rgb148_sem380": BATCH / "inference/predictions",
    }
    trajectory = {}
    for arm, root in prediction_roots.items():
        adjacent = []
        previous = None
        for name in trajectory_names:
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
                np.asarray(Image.open(prediction_roots["rgb140_sem378"] / name))
                != np.asarray(Image.open(prediction_roots["rgb148_sem380"] / name))
            )
        )
        for name in trajectory_names
    ]
    trajectory["candidate_vs_baseline"] = {
        "mean_label_disagreement": float(np.mean(disagreements)),
        "max_label_disagreement": float(np.max(disagreements)),
        "max_frame_index": int(np.argmax(disagreements)),
        "per_frame": disagreements,
        "interpretation": "no-GT motion-contaminated proxy; not accuracy or flicker",
    }

    unlabeled_names = sorted(
        name
        for name in export["image_hashes"]
        if not name.startswith("frame") and name not in set(names)
    )
    assert len(unlabeled_names) == 100
    unlabeled = []
    for name in unlabeled_names:
        old = np.asarray(Image.open(BASE / "inference/predictions" / name))
        new = np.asarray(Image.open(BATCH / "inference/predictions" / name))
        unlabeled.append(
            {
                "name": name,
                "changed_pixels": int((old != new).sum()),
                "old_background_to_cable": int(((old == 0) & (new == 2)).sum()),
                "old_cable_to_background": int(((old == 2) & (new == 0)).sum()),
            }
        )

    deltas = [row["semantic_delta_pp"] for row in per_view]
    fixed = read(ROOT / "experiments/runs/SEM-370-ALL300/config.yaml")[
        "diagnostic_views"
    ]
    per_view_lookup = {row["name"]: row for row in per_view}
    rgb140_run = read(ROOT / config["baseline_rgb_run"] / "result.json")
    rgb140_eval = read(ROOT / config["baseline_rgb_run"] / "evaluation/result.json")
    rgb148_run = read(ROOT / config["rgb_run"] / "result.json")
    rgb148_eval = read(ROOT / config["rgb_evaluation"] / "result.json")
    baseline_train = read(BASE / "mask2former/result.json")
    baseline_inference = read(BASE / "inference/result.json")
    output_bytes = sum(
        path.stat().st_size
        for path in BATCH.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    result = {
        "schema_version": 1,
        "status": "completed",
        "experiment_id": EID,
        "binding": config,
        "config_sha256": sha256(config_path),
        "script_sha256": sha256(Path(__file__)),
        "source_results_sha256": {
            "sem378": sha256(BASE / "inference/result.json"),
            "sem380": sha256(BATCH / "inference/result.json"),
            "rgb140": sha256(ROOT / config["baseline_rgb_run"] / "result.json"),
            "rgb148": sha256(ROOT / config["rgb_run"] / "result.json"),
        },
        "sampling6000_matches_sem378": train_record["sampling6000_matches_sem378"],
        "resolved_recipe_matches_sem378": train_record[
            "resolved_recipe_matches_sem378"
        ],
        "initial_state_matches_sem378": train_record["initial_state_matches_sem378"],
        "independent_confusion_checks": 600,
        "metrics": aggregate,
        "paired": {
            "global_mIoU_delta_pp": 100
            * (
                aggregate["rgb148_sem380"]["mIoU_with_background"]
                - aggregate["rgb140_sem378"]["mIoU_with_background"]
            ),
            "improved_views": int(sum(delta > 0 for delta in deltas)),
            "degraded_views": int(sum(delta < 0 for delta in deltas)),
            "tied_views": int(sum(delta == 0 for delta in deltas)),
            "changed_pixels": changed,
            "corrected_pixels": corrected,
            "harmed_pixels": harmed,
            "still_wrong_changed_pixels": still_wrong,
            "per_class": per_class_changes,
            "per_view": per_view,
        },
        "stay_cable_binary_changes": cable_changes,
        "boundary": boundary,
        "gt_old_new_transition_cube": transition.tolist(),
        "rgb_semantic_alignment": alignment,
        "class_rgb_mae_255_pixel_weighted": {
            arm: {
                class_name: float(rgb_error_sum[arm][index] / rgb_pixel_count[index])
                for index, class_name in enumerate(CLASSES)
            }
            for arm in ("rgb140", "rgb148")
        },
        "input_detail_proxy": {
            "method": "5x5 local luminance range >=12/255; not physical cable visibility",
            "warning": "stay_cable labels are filled sectors, not individual physical wires",
            "strata": detail_strata,
        },
        "regions": region_rows,
        "fixed_nine": [per_view_lookup[name] for name in fixed],
        "trajectory_proxy": trajectory,
        "unlabeled_proxy": {
            "scope": "100 views without GT; label changes only, not accuracy",
            "per_view": unlabeled,
        },
        "system_cost": {
            "rgb140_sem378": {
                "gaussians": rgb140_eval["final_gaussian_count"],
                "rgb_model_mib": rgb140_eval["model_mib"],
                "rgb_training_seconds": rgb140_run["worker"]["seconds"],
                "rgb_fixed9_render_median_ms": rgb140_eval["render_median_ms"],
                "rgb_peak_allocated_mib": rgb140_run["worker"]["peak_allocated_mib"],
                "semantic_training_seconds": baseline_train["training_seconds"],
                "semantic_peak_allocated_mib": baseline_train["peak_allocated_mib"],
                "semantic_peak_reserved_mib": baseline_train["peak_reserved_mib"],
                "semantic_inference_median_ms": 1000
                * baseline_inference["inference_median_seconds"],
                "semantic_checkpoint_bytes": (BASE / "mask2former/model-6000.pt")
                .stat()
                .st_size,
            },
            "rgb148_sem380": {
                "gaussians": rgb148_eval["final_gaussian_count"],
                "rgb_model_mib": rgb148_eval["model_mib"],
                "rgb_training_seconds": rgb148_run["worker"]["seconds"],
                "rgb_fixed9_render_median_ms": rgb148_eval[
                    "fixed9_cuda_render_median_ms"
                ],
                "rgb_all544_render_median_ms": rgb148_eval["render_median_ms"],
                "rgb_peak_allocated_mib": rgb148_run["worker"]["peak_allocated_mib"],
                "semantic_training_seconds": train_record["training_seconds"],
                "semantic_peak_allocated_mib": train_record["peak_allocated_mib"],
                "semantic_peak_reserved_mib": train_record["peak_reserved_mib"],
                "semantic_inference_median_ms": 1000
                * candidate_record["inference_median_seconds"],
                "semantic_checkpoint_bytes": (BATCH / "mask2former/model-6000.pt")
                .stat()
                .st_size,
                "experiment_output_bytes_before_review": output_bytes,
            },
        },
        "source_roi_configs_sha256": {
            path: sha256(ROOT / path) for path in region_configs
        },
        "raw_prediction_hashes": {
            arm: {path.name: sha256(path) for path in sorted(root.glob("*.png"))}
            for arm, root in (
                ("sem378", BASE / "inference/predictions"),
                ("sem380", BATCH / "inference/predictions"),
            )
        },
        "rgb544_hashes_exact": True,
        "rgb148_endpoint_unchanged": True,
        "sem378_unchanged": True,
        "seconds": time.monotonic() - started,
    }
    assert all(len(rows) == 544 for rows in result["raw_prediction_hashes"].values())
    write_new_json(BATCH / "audit.json", result)


if __name__ == "__main__":
    main()
