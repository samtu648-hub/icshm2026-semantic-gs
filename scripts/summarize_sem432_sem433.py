#!/usr/bin/env python3
"""Independently audit and compare SEM432/433 on sealed v1 predictions."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "outputs/quality/SEM-432/r1"
CANDIDATE = ROOT / "outputs/quality/SEM-433/r1"
OUTPUT = ROOT / "outputs/quality/QUAL-019"
CONFIG = ROOT / "experiments/runs/SEM-433-PAIRED-PHOTO-DOMAIN/config.yaml"
CLASS_NAMES = ("background", "deck", "stay_cable", "tower", "foundation")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def confusion(target, prediction):
    return np.bincount((target * 5 + prediction).ravel(), minlength=25).reshape(5, 5)


def metric_delta(candidate, control):
    return 100.0 * (float(candidate) - float(control))


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    control_train = read(CONTROL / "mask2former/result.json")
    candidate_train = read(CANDIDATE / "mask2former/result.json")
    control = read(CONTROL / "inference/result.json")
    candidate = read(CANDIDATE / "inference/result.json")
    config = read(CONFIG)
    if any(row["status"] != "completed" for row in (control, candidate)):
        raise ValueError("Both sealed inference results are required")
    if any(row["status"] != "completed_sealed_6000" for row in (control_train, candidate_train)):
        raise ValueError("Both sealed training endpoints are required")
    plan_control = read(CONTROL / "sampling-plan.json")
    plan_candidate = read(CANDIDATE / "sampling-plan.json")
    source_control = read(CONTROL / "paired-source.json")
    source_candidate = read(CANDIDATE / "paired-source.json")
    split = read(ROOT / config["split_path"])["primary_fold"]
    holdout = sorted(split["semantic_validation_labeled"])
    label_root = ROOT / config["label_root"]
    matrices = {"control": np.zeros((5, 5), np.int64), "candidate": np.zeros((5, 5), np.int64)}
    changed = corrected = harmed = neutral = 0
    transitions = Counter()
    for name in holdout:
        target = np.asarray(Image.open(label_root / name), dtype=np.uint8)
        a = np.asarray(Image.open(CONTROL / "inference/predictions" / name), dtype=np.uint8)
        b = np.asarray(Image.open(CANDIDATE / "inference/predictions" / name), dtype=np.uint8)
        matrices["control"] += confusion(target, a)
        matrices["candidate"] += confusion(target, b)
        different = a != b
        changed += int(different.sum())
        corrected += int((different & (b == target) & (a != target)).sum())
        harmed += int((different & (a == target) & (b != target)).sum())
        neutral += int((different & (a != target) & (b != target)).sum())
        values, counts = np.unique(a[different] * 5 + b[different], return_counts=True)
        for value, count in zip(values, counts):
            transitions[f"{CLASS_NAMES[int(value)//5]}->{CLASS_NAMES[int(value)%5]}"] += int(count)
    if matrices["control"].tolist() != control["metrics"]["holdout"]["confusion"]:
        raise ValueError("Independent control confusion mismatch")
    if matrices["candidate"].tolist() != candidate["metrics"]["holdout"]["confusion"]:
        raise ValueError("Independent candidate confusion mismatch")
    cm = control["metrics"]["holdout"]
    pm = candidate["metrics"]["holdout"]
    class_delta = {
        name: metric_delta(pm["per_class"][name]["IoU"], cm["per_class"][name]["IoU"])
        for name in CLASS_NAMES
    }
    group_delta = {
        key: metric_delta(
            candidate["metrics"][key]["mIoU_without_background"],
            control["metrics"][key]["mIoU_without_background"],
        )
        for key in ("region-1", "region-2", "region-3")
    }
    deltas = {
        "five_class_miou_pp": metric_delta(pm["mIoU_with_background"], cm["mIoU_with_background"]),
        "foreground_miou_pp": metric_delta(pm["mIoU_without_background"], cm["mIoU_without_background"]),
        "stay_cable_boundary_f1_pp": metric_delta(
            candidate["boundary_metrics"]["holdout"]["boundary_f1"],
            control["boundary_metrics"]["holdout"]["boundary_f1"],
        ),
        "per_class_iou_pp": class_delta,
        "per_group_foreground_miou_pp": group_delta,
    }
    gates_cfg = config["promotion_gates_vs_sem432_pp"]
    gates = {
        "five_class_miou": deltas["five_class_miou_pp"] >= gates_cfg["minimum_five_class_miou"],
        "foreground_miou": deltas["foreground_miou_pp"] >= gates_cfg["minimum_foreground_miou"],
        "stay_cable_boundary_f1": deltas["stay_cable_boundary_f1_pp"] >= gates_cfg["minimum_stay_cable_boundary_f1"],
        "all_structural_classes": all(class_delta[name] >= gates_cfg["minimum_each_structural_class_iou"] for name in CLASS_NAMES[1:]),
        "all_spatial_groups": all(value >= gates_cfg["minimum_each_spatial_group_foreground_miou"] for value in group_delta.values()),
    }
    audit = {
        "initial_state_exact": control_train["initial_model_state_sha256"] == candidate_train["initial_model_state_sha256"],
        "sampling_rows_exact": plan_control["samples"] == plan_candidate["samples"],
        "render_hashes_exact": source_control["render_hashes"] == source_candidate["render_hashes"],
        "photo_hashes_exact": source_control["photo_hashes"] == source_candidate["photo_hashes"],
        "checkpoint_control_stable": control["checkpoint_sha256_before_holdout_access"] == control["checkpoint_sha256_after_holdout_access"],
        "checkpoint_candidate_stable": candidate["checkpoint_sha256_before_holdout_access"] == candidate["checkpoint_sha256_after_holdout_access"],
        "independent_control_confusion_exact": True,
        "independent_candidate_confusion_exact": True,
        "frozen_rgb_hashes_exact": control["frozen_rgb_hashes_exact"] and candidate["frozen_rgb_hashes_exact"],
    }
    OUTPUT.mkdir(parents=True)
    result = {
        "schema_version": 1,
        "status": "completed_passed_for_v2_replication" if all(gates.values()) and all(audit.values()) else "completed_not_promoted",
        "experiment_id": "QUAL-019-PAIRED-PHOTO-DOMAIN",
        "comparison": "SEM433_minus_SEM432",
        "scope": "historically exposed spatial_confirmation_v1; render-only inference; single seed; not blind",
        "deltas": deltas,
        "gates": gates,
        "all_promotion_gates_pass": all(gates.values()),
        "audit": audit,
        "changed_pixels": changed,
        "corrected_pixels": corrected,
        "harmed_pixels": harmed,
        "neutral_wrong_to_wrong_pixels": neutral,
        "net_corrected_pixels": corrected - harmed,
        "changed_precision": corrected / changed if changed else None,
        "transitions": dict(transitions.most_common()),
        "control_training_seconds": control_train["training_seconds"],
        "candidate_training_seconds": candidate_train["training_seconds"],
        "control_peak_reserved_mib": control_train["peak_reserved_mib"],
        "candidate_peak_reserved_mib": candidate_train["peak_reserved_mib"],
        "evidence": {
            "control_train": digest(CONTROL / "mask2former/result.json"),
            "candidate_train": digest(CANDIDATE / "mask2former/result.json"),
            "control_inference": digest(CONTROL / "inference/result.json"),
            "candidate_inference": digest(CANDIDATE / "inference/result.json"),
        },
    }
    (OUTPUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

