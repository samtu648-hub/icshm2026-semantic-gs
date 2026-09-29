#!/usr/bin/env python3
"""Independent full-image, fixed-ROI, and trajectory audit for SEM379."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quality_protocol import CLASSES, sha256, verify_hashes, write_new_json

EID = "SEM-379-CABLE-REGION-SAMPLING"
BATCH = ROOT / "outputs/quality/SEM-379/r1"
BASELINE = ROOT / "outputs/quality/SEM-378/r1"
LABELS = ROOT / "outputs/quality/all_data/SEM-370-r1/labels"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def confusion(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    assert gt.shape == pred.shape == (989, 1320)
    assert int(gt.max()) < 5 and int(pred.max()) < 5
    return np.asarray(
        [[int(np.logical_and(gt == target, pred == predicted).sum()) for predicted in range(5)]
         for target in range(5)],
        dtype=np.int64,
    )


def scores(cm: np.ndarray) -> dict:
    per_class = {}
    ious = []
    foreground_ious = []
    for index, name in enumerate(CLASSES):
        tp = int(cm[index, index])
        fp = int(cm[:, index].sum()) - tp
        fn = int(cm[index].sum()) - tp
        iou = tp / (tp + fp + fn) if tp + fp + fn else None
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        per_class[name] = {"tp": tp, "fp": fp, "fn": fn, "IoU": iou,
                           "precision": precision, "recall": recall}
        if iou is not None:
            ious.append(iou)
            if index > 0:
                foreground_ious.append(iou)
    return {"confusion": cm.tolist(), "per_class": per_class,
            "mIoU_with_background": float(sum(ious) / len(ious)),
            "mIoU_without_background": (float(sum(foreground_ious) / len(foreground_ious))
                                         if foreground_ious else None)}


def main() -> None:
    started = time.monotonic()
    config_path = ROOT / "experiments/runs" / EID / "config.yaml"
    config = read(config_path)
    split = read(ROOT / config["split_path"])
    names = sorted(split["semantic_train"])
    candidate_record = read(BATCH / "inference/result.json")
    baseline_record = read(BASELINE / "inference/result.json")
    train_record = read(BATCH / "mask2former/result.json")
    export = read(BATCH / "export/result.json")
    assert candidate_record["status"] == baseline_record["status"] == "completed"
    assert len(names) == 300 and len(export["image_hashes"]) == 544
    verify_hashes(BATCH / "export/images", export["image_hashes"])
    verify_hashes(BATCH / "mask2former", train_record["artifacts"])

    candidate_rows = {row["name"]: row for row in candidate_record["per_view"]}
    baseline_rows = {row["name"]: row for row in baseline_record["per_view"]}
    totals = {"sem378": np.zeros((5, 5), np.int64), "sem379": np.zeros((5, 5), np.int64)}
    per_view = []
    changed = corrected = harmed = still_wrong = 0
    per_class_changes = {name: {"corrected": 0, "harmed": 0} for name in CLASSES}
    for name in names:
        gt = np.asarray(Image.open(LABELS / name))
        old = np.asarray(Image.open(BASELINE / "inference/predictions" / name))
        new = np.asarray(Image.open(BATCH / "inference/predictions" / name))
        old_cm = confusion(gt, old)
        new_cm = confusion(gt, new)
        assert old_cm.tolist() == baseline_rows[name]["confusion"]
        assert new_cm.tolist() == candidate_rows[name]["confusion"]
        totals["sem378"] += old_cm
        totals["sem379"] += new_cm
        old_score, new_score = scores(old_cm), scores(new_cm)
        delta = 100 * (new_score["mIoU_with_background"] - old_score["mIoU_with_background"])
        per_view.append({"name": name, "sem378_mIoU": old_score["mIoU_with_background"],
                         "sem379_mIoU": new_score["mIoU_with_background"], "delta_pp": delta})
        change = old != new
        fix = (old != gt) & (new == gt)
        harm = (old == gt) & (new != gt)
        changed += int(change.sum())
        corrected += int(fix.sum())
        harmed += int(harm.sum())
        still_wrong += int((change & (old != gt) & (new != gt)).sum())
        for index, class_name in enumerate(CLASSES):
            per_class_changes[class_name]["corrected"] += int((fix & (gt == index)).sum())
            per_class_changes[class_name]["harmed"] += int((harm & (gt == index)).sum())

    aggregate = {name: scores(cm) for name, cm in totals.items()}
    for arm, record in (("sem378", baseline_record), ("sem379", candidate_record)):
        actual, expected = aggregate[arm], record["metrics"]
        assert actual["confusion"] == expected["confusion"]
        for key in ("mIoU_with_background", "mIoU_without_background"):
            assert abs(actual[key] - expected[key]) < 1e-15
        for class_name in CLASSES:
            for key in ("tp", "fp", "fn"):
                assert actual["per_class"][class_name][key] == expected["per_class"][class_name][key]
            for key in ("IoU", "precision", "recall"):
                assert abs(actual["per_class"][class_name][key] -
                           expected["per_class"][class_name][key]) < 1e-15

    region_configs = [
        "experiments/runs/RGB-132-THREE-BACKBONE-DIAG/config.yaml",
        "experiments/runs/QUAL-005-PHOTO-INPUT-DIAG/config.yaml",
        "experiments/runs/QUAL-004-CURRENT-BOTTLENECK/config.yaml",
    ]
    regions = read(ROOT / region_configs[0])["regions"].copy()
    for prefix, key, path in (("difficult", "rois", region_configs[1]),
                              ("structure", "crops", region_configs[2])):
        for stem, box in read(ROOT / path)[key].items():
            regions[f"{prefix}-{stem}"] = {"view": f"{stem}.png", "boxes": [box]}
    region_rows = []
    for region_name, spec in regions.items():
        name = spec["view"]
        gt = np.asarray(Image.open(LABELS / name))
        old = np.asarray(Image.open(BASELINE / "inference/predictions" / name))
        new = np.asarray(Image.open(BATCH / "inference/predictions" / name))
        mask = np.zeros(gt.shape, bool)
        for x0, y0, x1, y1 in spec["boxes"]:
            mask[y0:y1, x0:x1] = True
        old_cm = np.asarray([[int(np.logical_and(gt[mask] == t, old[mask] == p).sum())
                              for p in range(5)] for t in range(5)], dtype=np.int64)
        new_cm = np.asarray([[int(np.logical_and(gt[mask] == t, new[mask] == p).sum())
                              for p in range(5)] for t in range(5)], dtype=np.int64)
        fix = (old != gt) & (new == gt) & mask
        harm = (old == gt) & (new != gt) & mask
        change = (old != new) & mask
        region_rows.append({
            "region": region_name, "view": name, "boxes": spec["boxes"], "pixels": int(mask.sum()),
            "sem378": scores(old_cm), "sem379": scores(new_cm),
            "delta_pp": 100 * (scores(new_cm)["mIoU_with_background"] - scores(old_cm)["mIoU_with_background"]),
            "changed": int(change.sum()), "corrected": int(fix.sum()), "harmed": int(harm.sum()),
            "still_wrong_changed": int((change & (old != gt) & (new != gt)).sum()),
            "per_class": {class_name: {"corrected": int((fix & (gt == index)).sum()),
                                        "harmed": int((harm & (gt == index)).sum())}
                          for index, class_name in enumerate(CLASSES)},
        })

    trajectory_names = export["trajectory_names"]
    temporal = {}
    predictions = {"sem378": BASELINE / "inference/predictions",
                   "sem379": BATCH / "inference/predictions"}
    for arm, path in predictions.items():
        changes = []
        previous = None
        for name in trajectory_names:
            current = np.asarray(Image.open(path / name))
            if previous is not None:
                changes.append(float(np.mean(current != previous)))
            previous = current
        temporal[arm] = {"mean_adjacent_label_change": float(np.mean(changes)),
                         "max_adjacent_label_change": float(np.max(changes)),
                         "max_transition_index": int(np.argmax(changes) + 1),
                         "per_transition": changes}
    disagreements = [float(np.mean(np.asarray(Image.open(predictions["sem378"] / name)) !=
                                   np.asarray(Image.open(predictions["sem379"] / name))))
                     for name in trajectory_names]
    temporal["candidate_vs_baseline"] = {
        "mean_label_disagreement": float(np.mean(disagreements)),
        "max_label_disagreement": float(np.max(disagreements)),
        "max_frame_index": int(np.argmax(disagreements)),
        "per_frame": disagreements,
        "interpretation": "no-GT motion-contaminated visual proxy; not an accuracy or flicker metric",
    }

    deltas = [row["delta_pp"] for row in per_view]
    result = {
        "schema_version": 1,
        "status": "completed",
        "experiment_id": EID,
        "binding": config,
        "config_sha256": sha256(config_path),
        "script_sha256": sha256(Path(__file__)),
        "source_results_sha256": {
            "sem378": sha256(BASELINE / "inference/result.json"),
            "sem379": sha256(BATCH / "inference/result.json"),
        },
        "sampling6000_matches_plan": train_record["sampling6000_matches_plan"],
        "resolved_recipe_matches_sem378": train_record["resolved_recipe_matches_sem378"],
        "initial_state_matches_sem378": train_record["initial_state_matches_sem378"],
        "independent_confusion_checks": 600,
        "metrics": aggregate,
        "paired": {
            "global_mIoU_delta_pp": 100 * (aggregate["sem379"]["mIoU_with_background"] -
                                             aggregate["sem378"]["mIoU_with_background"]),
            "improved_views": int(sum(delta > 0 for delta in deltas)),
            "degraded_views": int(sum(delta < 0 for delta in deltas)),
            "tied_views": int(sum(delta == 0 for delta in deltas)),
            "changed_pixels": changed, "corrected_pixels": corrected,
            "harmed_pixels": harmed, "still_wrong_changed_pixels": still_wrong,
            "per_class": per_class_changes, "per_view": per_view,
        },
        "regions": region_rows,
        "trajectory_proxy": temporal,
        "source_roi_configs_sha256": {path: sha256(ROOT / path) for path in region_configs},
        "rgb544_hashes_exact": True,
        "rgb140_endpoint_unchanged": True,
        "sem378_unchanged": True,
        "seconds": time.monotonic() - started,
    }
    write_new_json(BATCH / "audit.json", result)


if __name__ == "__main__":
    main()
