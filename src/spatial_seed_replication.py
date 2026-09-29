"""CPU contracts for the frozen spatial-v2 semantic seed replication."""

from __future__ import annotations

from typing import Any

CLASSES = ("background", "deck", "stay_cable", "tower", "foundation")
PAIR_CONFIG = "experiments/runs/QUAL-013-SPATIAL-V2-SEED-REPLICATION/config.yaml"


def validate_pair_configs(arms: dict[str, dict], split: dict) -> None:
    """Reject seed, data-role or training-recipe drift before any GPU work."""
    a, s = arms["absgs"], arms["standard"]
    for key in (
        "seed",
        "dataset_manifest",
        "split_id",
        "split_path",
        "method",
        "iterations",
        "crop",
        "sampling_trace_keys",
        "label_root",
        "label_manifest",
        "semantic_train_views",
        "semantic_confirmation_views",
        "buffer_excluded_views",
        "seed_aware_initialization",
        "confirmation_access",
        "protocol",
    ):
        if a[key] != s[key]:
            raise ValueError(f"Paired config mismatch: {key}")
    if (a["seed"], a["iterations"], a["crop"], a["split_id"]) != (
        20260919,
        6000,
        768,
        "spatial_confirmation_v2",
    ):
        raise ValueError("Unregistered seed, endpoint, crop or split")
    if a["seed_aware_initialization"]["reference_seed"] != a["seed"]:
        raise ValueError("Initialization reference must use the current seed")
    if (
        a["rgb_run"] != "outputs/quality/RGB-154/full-r1"
        or s["rgb_run"] != "outputs/quality/RGB-155/full-r1"
    ):
        raise ValueError("The frozen RGB pair changed")
    fold = split["primary_fold"]
    train = set(fold["train_rgb_400"])
    semantic = set(fold["semantic_train_labeled"])
    holdout = set(fold["semantic_validation_labeled"])
    buffer = set(fold["buffer_excluded_all_cameras"])
    if tuple(map(len, (train, semantic, holdout, buffer))) != (349, 255, 30, 21):
        raise ValueError("Spatial v2 role counts changed")
    if train & holdout or train & buffer or holdout & buffer or not semantic <= train:
        raise ValueError("Spatial v2 roles overlap")
    if len(train | holdout | buffer) != 400:
        raise ValueError("Incomplete camera roles")


def validate_pair_endpoints(a: dict, s: dict) -> None:
    """A single completed arm cannot authorize confirmation access."""
    for row in (a, s):
        if row["status"] != "completed_sealed_6000" or row["trained_steps"] != 6000:
            raise ValueError("Both 6000-step endpoints must be sealed")
        if row["seed"] != 20260919 or not row.get(
            "seed_aware_initialization_validated"
        ):
            raise ValueError("Missing seed-aware initialization evidence")
        if row["holdout_labels_accessed_during_fit"]:
            raise ValueError("Confirmation labels accessed during fitting")
        if not row["sampling_plan_exact"] or not row["frozen_rgb_hashes_exact"]:
            raise ValueError("Sampling or frozen RGB check failed")
    for key in (
        "initial_model_state_sha256",
        "pretrained_load",
        "split_sha256",
        "access",
    ):
        if a[key] != s[key]:
            raise ValueError(f"Paired endpoints differ in {key}")


def confusion_metrics(matrix: list[list[int]]) -> dict[str, Any]:
    """Independent scalar arithmetic with GT rows and prediction columns."""
    if len(matrix) != 5 or any(len(row) != 5 for row in matrix):
        raise ValueError("Expected a five-class confusion matrix")
    if any(not isinstance(v, int) or v < 0 for row in matrix for v in row):
        raise ValueError("Confusion counts must be nonnegative integers")
    classes = {}
    for i, name in enumerate(CLASSES):
        tp = matrix[i][i]
        fp = sum(row[i] for row in matrix) - tp
        fn = sum(matrix[i]) - tp
        classes[name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "IoU": tp / (tp + fp + fn) if tp + fp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
        }
    all_iou = [v["IoU"] for v in classes.values() if v["IoU"] is not None]
    fg = [classes[n]["IoU"] for n in CLASSES[1:] if classes[n]["IoU"] is not None]
    return {
        "per_class": classes,
        "mIoU_with_background": sum(all_iou) / len(all_iou) if all_iou else None,
        "mIoU_without_background": sum(fg) / len(fg) if fg else None,
    }


def flatten(result: dict, role: str) -> dict[str, float]:
    m = result["metrics"][role]
    cable = m["per_class"]["stay_cable"]
    return {
        "foreground_miou": m["mIoU_without_background"],
        "five_class_miou": m["mIoU_with_background"],
        "stay_cable_iou": cable["IoU"],
        "stay_cable_precision": cable["precision"],
        "stay_cable_recall": cable["recall"],
        "boundary_f1": result["boundary_metrics"][role]["boundary_f1"],
    }


def direction_gates(deltas: dict[str, dict[str, float]]) -> dict:
    if set(deltas) != {"20260805", "20260919"}:
        raise ValueError("Both registered semantic seeds are required")
    gates = {
        seed: {
            "foreground_positive": d["foreground_miou"] > 0,
            "stay_cable_nonnegative": d["stay_cable_iou"] >= 0,
            "boundary_positive": d["boundary_f1"] > 0,
        }
        for seed, d in deltas.items()
    }
    passed = all(flag for row in gates.values() for flag in row.values())
    return {"per_seed": gates, "all_direction_gates_pass": passed}
