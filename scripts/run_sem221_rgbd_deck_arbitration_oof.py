#!/usr/bin/env python3
"""Test whether SEM-220 can safely arbitrate the frozen background/deck lane."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_sem200_multiview_deck_diag as sem200

from src.semantic2d import confusion_metrics, label_array, read_json, sha256, write_json
from src.semantic_deck_arbitration import arbitration_gate
from src.semantic_protected_fusion import local_confusion, merge_confusions, metrics_from_confusion, partition_training_names
from src.semantic_render_domain import semantic_rgb
from src.semantic_rendered_training import select_rendered_training_views


EXPERIMENT_ID = "SEM-221-RGBD-DECK-ARBITRATION-OOF"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--sem001-training-root", type=Path, required=True)
    parser.add_argument("--sem150-root", type=Path, required=True)
    parser.add_argument("--sem160-root", type=Path, required=True)
    parser.add_argument("--sem220-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_probability(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        require(set(archive.files) == {"deck_probability"}, f"SEM-221 probability schema mismatch: {path}")
        value = np.asarray(archive["deck_probability"], dtype=np.float32)
    require(value.shape == (989, 1320) and np.isfinite(value).all(), f"SEM-221 invalid probability: {path}")
    return value


def candidate_for(
    *,
    baseline: np.ndarray,
    eligible: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mutable = (baseline == 1) | eligible
    candidate = baseline.copy()
    candidate[mutable] = (probability[mutable] >= float(threshold)).astype(np.uint8)
    additions = (baseline == 0) & (candidate == 1)
    removals = (baseline == 1) & (candidate == 0)
    require(np.array_equal(candidate[~mutable], baseline[~mutable]), "SEM-221 changed a protected pixel")
    for class_id in (2, 3, 4):
        require(np.array_equal(candidate == class_id, baseline == class_id), f"SEM-221 changed class {class_id}")
    return candidate, mutable, additions, removals


def evaluate(
    *,
    args: argparse.Namespace,
    names: list[str],
    thresholds: list[float],
    baseline_paths: dict[str, Path],
    probability_paths: dict[str, Path],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline_confusions: list[np.ndarray] = []
    candidate_confusions = {threshold: [] for threshold in thresholds}
    additions = {threshold: np.zeros(5, dtype=np.int64) for threshold in thresholds}
    removals = {threshold: np.zeros(5, dtype=np.int64) for threshold in thresholds}
    protected = {threshold: True for threshold in thresholds}
    for name in names:
        baseline = sem200._read_class_map(baseline_paths[name], "SEM-221 baseline")
        eligible = sem200._eligibility(name=name, sem001_root=args.sem001_training_root, sem150_root=args.sem150_root, baseline=baseline)
        probability = load_probability(probability_paths[name])
        candidates = {
            threshold: candidate_for(baseline=baseline, eligible=eligible, probability=probability, threshold=threshold)
            for threshold in thresholds
        }
        target = label_array(args.labels_root / name)
        baseline_confusions.append(local_confusion(target, baseline))
        for threshold, (candidate, mutable, added, removed) in candidates.items():
            candidate_confusions[threshold].append(local_confusion(target, candidate))
            additions[threshold] += np.bincount(target[added], minlength=5).astype(np.int64)
            removals[threshold] += np.bincount(target[removed], minlength=5).astype(np.int64)
            protected[threshold] = protected[threshold] and bool(np.array_equal(candidate[~mutable], baseline[~mutable]))
    baseline_metrics = metrics_from_confusion(merge_confusions(baseline_confusions))
    records = []
    for threshold in thresholds:
        candidate_metrics = metrics_from_confusion(merge_confusions(candidate_confusions[threshold]))
        evaluation = {
            "metrics": candidate_metrics,
            "protected_predictions_exact": protected[threshold],
            "additions_by_ground_truth": additions[threshold].astype(int).tolist(),
            "removals_by_ground_truth": removals[threshold].astype(int).tolist(),
            "added_deck_pixels": int(additions[threshold].sum()),
            "removed_deck_pixels": int(removals[threshold].sum()),
        }
        gate = arbitration_gate(
            candidate=candidate_metrics,
            baseline=baseline_metrics,
            protected_predictions_exact=protected[threshold],
            minimum_iou_gain=0.002,
            minimum_precision_gain=0.001,
            minimum_recall_gain=0.0,
            minimum_miou_gain=0.0,
            maximum_background_fp_ratio=1.0,
        )
        records.append({"threshold": threshold, **evaluation, "gate": gate})
    return baseline_metrics, records


def write_diagnostic_panels(
    *,
    args: argparse.Namespace,
    names: list[str],
    threshold: float,
    baseline_paths: dict[str, Path],
    probability_paths: dict[str, Path],
) -> None:
    root = args.output_dir / "calibration_panels"
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        baseline = sem200._read_class_map(baseline_paths[name], "SEM-221 panel baseline")
        eligible = sem200._eligibility(name=name, sem001_root=args.sem001_training_root, sem150_root=args.sem150_root, baseline=baseline)
        candidate, _, _, _ = candidate_for(
            baseline=baseline,
            eligible=eligible,
            probability=load_probability(probability_paths[name]),
            threshold=threshold,
        )
        target = label_array(args.labels_root / name)
        changed = candidate != baseline
        outcome = np.zeros((*target.shape, 3), dtype=np.uint8)
        outcome[changed & (candidate == target)] = np.asarray([48, 220, 96], dtype=np.uint8)
        outcome[changed & (candidate != target)] = np.asarray([255, 48, 48], dtype=np.uint8)
        Image.fromarray(np.concatenate((semantic_rgb(target), semantic_rgb(baseline), semantic_rgb(candidate), outcome), axis=1), mode="RGB").save(root / name)


def materialize_all(
    *,
    args: argparse.Namespace,
    names: list[str],
    threshold: float,
    baseline_paths: dict[str, Path],
    probability_paths: dict[str, Path],
) -> dict[str, Any]:
    root = args.output_dir / "selected_predictions"
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for name in names:
        baseline = sem200._read_class_map(baseline_paths[name], "SEM-221 baseline")
        eligible = sem200._eligibility(name=name, sem001_root=args.sem001_training_root, sem150_root=args.sem150_root, baseline=baseline)
        candidate, mutable, additions, removals = candidate_for(
            baseline=baseline,
            eligible=eligible,
            probability=load_probability(probability_paths[name]),
            threshold=threshold,
        )
        path = root / name
        Image.fromarray(candidate, mode="L").save(path)
        records.append({"name": name, "path": str(path.relative_to(args.output_dir)), "sha256": sha256(path), "added_deck_pixels": int(additions.sum()), "removed_deck_pixels": int(removals.sum()), "protected_exact": bool(np.array_equal(candidate[~mutable], baseline[~mutable]))})
    manifest = {"schema_version": 1, "experiment_id": EXPERIMENT_ID, "status": "completed_selected_predictions", "selected_threshold": threshold, "target_labels_opened_during_materialization": False, "fixed_validation_opened": False, "predictions": records}
    write_json(args.output_dir / "prediction_manifest.json", manifest)
    return manifest


def evaluate_selected_half(
    *,
    names: list[str],
    labels_root: Path,
    baseline_paths: dict[str, Path],
    candidate_root: Path,
) -> tuple[dict[str, Any], dict[str, int]]:
    before, after = [], []
    comparison = {"improved": 0, "equal": 0, "regressed": 0}
    for name in names:
        target = label_array(labels_root / name)
        baseline = sem200._read_class_map(baseline_paths[name], "SEM-221 check baseline")
        candidate = sem200._read_class_map(candidate_root / name, "SEM-221 check candidate")
        c0, c1 = local_confusion(target, baseline), local_confusion(target, candidate)
        before.append(c0); after.append(c1)
        m0, m1 = confusion_metrics(c0)["per_class"]["deck"]["iou"], confusion_metrics(c1)["per_class"]["deck"]["iou"]
        delta = float(m1) - float(m0)
        comparison["improved" if delta > 1e-12 else "regressed" if delta < -1e-12 else "equal"] += 1
    return metrics_from_confusion(merge_confusions(after)), comparison


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    require(config.get("experiment_id") == EXPERIMENT_ID and config.get("status") == "registered_not_run", "SEM-221 config mismatch")
    require(not args.output_dir.exists() or not any(args.output_dir.iterdir()), f"refusing to overwrite non-empty output: {args.output_dir}")
    require(sha256(args.split) == config["data"]["semantic_split_sha256"], "SEM-221 split hash differs")
    require(sha256(args.sem220_root / "oof_probability_manifest.json") == config["inputs"]["sem220_probability_manifest_sha256"], "SEM-221 SEM-220 manifest differs")
    require(sha256(args.sem160_root / "prediction_manifest.json") == config["inputs"]["sem160_prediction_manifest_sha256"], "SEM-221 baseline differs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = select_rendered_training_views(read_json(args.split), expected_count=219)
    partition = partition_training_names(names, expected_count=219)
    baseline_manifest = read_json(args.sem160_root / "prediction_manifest.json")
    baseline_paths = {record["name"]: args.sem160_root / record["prediction"] for record in baseline_manifest["predictions"]}
    probability_manifest = read_json(args.sem220_root / "oof_probability_manifest.json")
    probability_paths = {record["name"]: args.sem220_root / record["path"] for record in probability_manifest["predictions"]}
    require(set(names) == set(baseline_paths) == set(probability_paths), "SEM-221 input coverage differs")
    thresholds = [float(value) for value in config["arbitration"]["threshold_grid"]]
    baseline_metrics, records = evaluate(args=args, names=partition["calibration"], thresholds=thresholds, baseline_paths=baseline_paths, probability_paths=probability_paths)
    passing = [record for record in records if record["gate"]["passed"]]
    diagnostic = max(records, key=lambda record: (record["metrics"]["per_class"]["deck"]["iou"], record["metrics"]["mIoU_with_background"], record["threshold"]))
    panel_names = [name for name in config["visual_samples"] if name in partition["calibration"]]
    write_diagnostic_panels(args=args, names=panel_names, threshold=float(diagnostic["threshold"]), baseline_paths=baseline_paths, probability_paths=probability_paths)
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "baseline_calibration_metrics": baseline_metrics,
        "calibration_records": records,
        "calibration_thresholds_passing": len(passing),
        "diagnostic_best_threshold": float(diagnostic["threshold"]),
        "fixed_validation_opened": False,
        "check_labels_opened": False,
        "rgb_modified": False,
    }
    if not passing:
        result.update({"status": "completed_calibration_gate_failed", "selected_threshold": None, "next_step": "close_SEM220_deck_line_keep_SEM161"})
        write_json(args.output_dir / "result.json", result)
        print(json.dumps({"experiment_id": EXPERIMENT_ID, "status": result["status"], "calibration_thresholds_passing": 0, "diagnostic_best_threshold": result["diagnostic_best_threshold"], "check_labels_opened": False, "result_sha256": sha256(args.output_dir / "result.json")}, sort_keys=True))
        return 0
    selected = max(passing, key=lambda record: (record["metrics"]["per_class"]["deck"]["iou"], record["metrics"]["mIoU_with_background"], record["threshold"]))
    threshold = float(selected["threshold"])
    manifest = materialize_all(args=args, names=names, threshold=threshold, baseline_paths=baseline_paths, probability_paths=probability_paths)
    check_metrics, comparison = evaluate_selected_half(names=partition["check"], labels_root=args.labels_root, baseline_paths=baseline_paths, candidate_root=args.output_dir / "selected_predictions")
    check_gate = arbitration_gate(candidate=check_metrics, baseline=metrics_from_confusion(merge_confusions([local_confusion(label_array(args.labels_root / name), sem200._read_class_map(baseline_paths[name], "SEM-221 check baseline")) for name in partition["check"]])), protected_predictions_exact=all(record["protected_exact"] for record in manifest["predictions"]), minimum_iou_gain=0.002, minimum_precision_gain=0.001, minimum_recall_gain=0.0, minimum_miou_gain=0.0, maximum_background_fp_ratio=1.0)
    result.update({"status": "completed_oof_gate_passed" if check_gate["passed"] else "completed_check_gate_failed", "selected_threshold": threshold, "selected_calibration_record": selected, "check_labels_opened": True, "check_metrics": check_metrics, "check_gate": check_gate, "check_deck_iou_per_view_comparison": comparison, "prediction_manifest_sha256": sha256(args.output_dir / "prediction_manifest.json"), "next_step": "register_final_RGBD_deck_model" if check_gate["passed"] else "close_SEM220_deck_line_keep_SEM161"})
    write_json(args.output_dir / "result.json", result)
    print(json.dumps({"experiment_id": EXPERIMENT_ID, "status": result["status"], "selected_threshold": threshold, "check_labels_opened": True, "result_sha256": sha256(args.output_dir / "result.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
