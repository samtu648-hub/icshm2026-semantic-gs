#!/usr/bin/env python3
"""Qualify frozen bidirectional deck arbitration on SEM-160 spatial OOF."""

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

from src.semantic2d import confusion_metrics, read_json, sha256, write_json
from src.semantic_deck_arbitration import (
    arbitrate_deck,
    arbitration_gate,
    select_arbitration_threshold,
)
from src.semantic_protected_fusion import (
    local_confusion,
    merge_confusions,
    metrics_from_confusion,
    partition_training_names,
)
from src.semantic_render_domain import semantic_rgb
from src.semantic_rendered_training import select_rendered_training_views


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--oof-split", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--sem001-training-root", type=Path, required=True)
    parser.add_argument("--sem150-root", type=Path, required=True)
    parser.add_argument("--sem150-result", type=Path, required=True)
    parser.add_argument("--sem160-root", type=Path, required=True)
    parser.add_argument("--sem160-result", type=Path, required=True)
    parser.add_argument("--sem160-oof-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def class_map(path: Path, *, context: str) -> np.ndarray:
    value = np.asarray(Image.open(path), dtype=np.uint8)
    if value.shape != (989, 1320) or value.size == 0 or int(value.max()) >= 5:
        raise ValueError(f"{context} is not a 1320x989 five-class PNG: {path}")
    return value


def deck_probability(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        require(
            archive.files == ["deck_probability"],
            "SEM-170 deck probability NPZ has an unexpected schema",
        )
        return archive["deck_probability"]


def sem001_paths(
    root: Path, names: list[str], *, manifest_sha256: str
) -> dict[str, Path]:
    manifest_path = root / "training_sem001_render_manifest.json"
    require(
        manifest_path.is_file() and sha256(manifest_path) == manifest_sha256,
        "SEM-170 SEM-001 training manifest hash mismatch",
    )
    manifest = read_json(manifest_path)
    records = manifest["rendered"]
    require(
        manifest.get("status") == "completed_training_sem001_render"
        and [record["name"] for record in records] == names,
        "SEM-170 SEM-001 training predictions are incomplete",
    )
    output = {}
    for record in records:
        path = root / record["semantic_png"]
        require(path.is_file() and sha256(path) == record["sha256"], "SEM-170 SEM-001 prediction hash mismatch")
        output[record["name"]] = path
    return output


def prediction_paths(
    root: Path,
    names: list[str],
    *,
    manifest_sha256: str,
    experiment_id: str,
) -> dict[str, Path]:
    manifest_path = root / "prediction_manifest.json"
    require(
        manifest_path.is_file() and sha256(manifest_path) == manifest_sha256,
        f"SEM-170 {experiment_id} prediction manifest hash mismatch",
    )
    manifest = read_json(manifest_path)
    records = manifest["predictions"]
    require(
        manifest.get("experiment_id") == experiment_id
        and [record["name"] for record in records] == names,
        f"SEM-170 {experiment_id} prediction order mismatch",
    )
    output = {}
    for record in records:
        path = root / record["prediction"]
        require(
            path.is_file() and sha256(path) == record["prediction_sha256"],
            f"SEM-170 {experiment_id} prediction hash mismatch: {record['name']}",
        )
        output[record["name"]] = path
    return output


def oof_probability_paths(
    root: Path,
    oof_split: dict[str, Any],
    names: list[str],
    *,
    result_hashes: list[str],
    manifest_hashes: list[str],
) -> dict[str, Path]:
    output = {}
    for fold in oof_split["folds"]:
        index = int(fold["fold"])
        fold_root = root / f"fold-{index}"
        result_path = fold_root / "result.json"
        manifest_path = fold_root / "prediction_manifest.json"
        require(sha256(result_path) == result_hashes[index], f"SEM-170 fold {index} result hash mismatch")
        require(sha256(manifest_path) == manifest_hashes[index], f"SEM-170 fold {index} manifest hash mismatch")
        result = read_json(result_path)
        require(
            result.get("status") == "completed_fold_oof"
            and result.get("fixed_validation_opened") is False
            and result.get("unlabeled_rgb_opened") is False,
            f"SEM-170 fold {index} violated its training-only contract",
        )
        records = read_json(manifest_path)
        require([record["name"] for record in records] == list(fold["oof"]), f"SEM-170 fold {index} order mismatch")
        for record in records:
            path = fold_root / record["deck_probability"]
            require(path.is_file() and sha256(path) == record["deck_probability_sha256"], f"SEM-170 probability hash mismatch: {record['name']}")
            output[record["name"]] = path
    require(set(output) == set(names), "SEM-170 OOF probabilities do not cover training")
    return output


def pre_foundation_map(base: np.ndarray, sem001: np.ndarray) -> np.ndarray:
    restored = base.copy()
    restored[(base == 0) & (sem001 == 4)] = 4
    return restored


def evaluate(
    *,
    names: list[str],
    threshold: float,
    labels_root: Path,
    base: dict[str, Path],
    sem001: dict[str, Path],
    probability: dict[str, Path],
    prediction_root: Path | None = None,
) -> dict[str, Any]:
    confusions = []
    protected_exact = True
    additions_by_gt = np.zeros(5, dtype=np.int64)
    removals_by_gt = np.zeros(5, dtype=np.int64)
    per_view = []
    records = []
    for name in names:
        target = class_map(labels_root / name, context="SEM-170 training label")
        baseline = class_map(base[name], context="SEM-170 SEM-150 baseline")
        sem001_map = class_map(sem001[name], context="SEM-170 SEM-001 provenance")
        candidate, mutable, additions, removals = arbitrate_deck(
            baseline,
            pre_foundation_map(baseline, sem001_map),
            deck_probability(probability[name]),
            threshold=threshold,
        )
        protected_exact = protected_exact and bool(
            np.array_equal(candidate[~mutable], baseline[~mutable])
        )
        additions_by_gt += np.bincount(target[additions], minlength=5)
        removals_by_gt += np.bincount(target[removals], minlength=5)
        confusion = local_confusion(target, candidate)
        confusions.append(confusion)
        local = confusion_metrics(confusion)
        per_view.append({"view": name, "deck_iou": local["per_class"]["deck"]["iou"]})
        if prediction_root is not None:
            prediction_root.mkdir(parents=True, exist_ok=True)
            path = prediction_root / name
            Image.fromarray(candidate).save(path)
            records.append(
                {
                    "name": name,
                    "prediction": str(path.relative_to(prediction_root.parent)),
                    "prediction_sha256": sha256(path),
                    "added_deck_pixels": int(np.count_nonzero(additions)),
                    "removed_deck_pixels": int(np.count_nonzero(removals)),
                }
            )
    return {
        "metrics": metrics_from_confusion(merge_confusions(confusions)),
        "protected_predictions_exact": protected_exact,
        "additions_by_gt_class": additions_by_gt.astype(int).tolist(),
        "removals_by_gt_class": removals_by_gt.astype(int).tolist(),
        "added_deck_pixels": int(additions_by_gt.sum()),
        "removed_deck_pixels": int(removals_by_gt.sum()),
        "per_view": per_view,
        "prediction_records": records,
    }


def gate_for(evaluation: dict[str, Any], baseline: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    return arbitration_gate(
        candidate=evaluation["metrics"],
        baseline=baseline,
        protected_predictions_exact=evaluation["protected_predictions_exact"],
        minimum_iou_gain=float(gate["minimum_deck_iou_gain"]),
        minimum_precision_gain=float(gate["minimum_deck_precision_gain"]),
        minimum_recall_gain=float(gate["minimum_deck_recall_gain"]),
        minimum_miou_gain=float(gate["minimum_mIoU_gain"]),
        maximum_background_fp_ratio=float(gate["maximum_background_to_deck_fp_ratio"]),
    )


def write_panels(*, names, labels_root, baseline, candidate_root, output_root):
    output_root.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = class_map(labels_root / name, context="SEM-170 panel label")
        before = class_map(baseline[name], context="SEM-170 panel baseline")
        after = class_map(candidate_root / name, context="SEM-170 panel candidate")
        changed = before != after
        outcome = np.zeros((*target.shape, 3), dtype=np.uint8)
        outcome[changed & (after == target)] = np.asarray([48, 220, 96], np.uint8)
        outcome[changed & (after != target)] = np.asarray([255, 48, 48], np.uint8)
        panel = np.concatenate(
            [semantic_rgb(target), semantic_rgb(before), semantic_rgb(after), outcome],
            axis=1,
        )
        Image.fromarray(panel).save(output_root / name)


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    require(config.get("experiment_id") == "SEM-170-DECK-ARBITRATION-OOF", "SEM-170 received another config")
    require(config.get("status") == "registered_not_run", "SEM-170 requires registered config")
    require(not args.output_dir.exists() or not any(args.output_dir.iterdir()), "SEM-170 refuses to overwrite output")
    require(sha256(args.split) == config["data"]["semantic_split_sha256"], "SEM-170 semantic split hash mismatch")
    require(sha256(args.oof_split) == config["data"]["oof_split_sha256"], "SEM-170 OOF split hash mismatch")
    require(sha256(args.sem150_result) == config["inputs"]["sem150_result_sha256"], "SEM-170 SEM-150 result hash mismatch")
    require(sha256(args.sem160_result) == config["inputs"]["sem160_result_sha256"], "SEM-170 SEM-160 result hash mismatch")
    sem160 = read_json(args.sem160_result)
    require(sem160.get("status") == "completed_oof_gate_passed", "SEM-170 requires passing SEM-160")
    names = select_rendered_training_views(read_json(args.split), expected_count=int(config["data"]["training_images"]))
    partition = partition_training_names(names, expected_count=len(names))
    sem001 = sem001_paths(args.sem001_training_root, names, manifest_sha256=config["inputs"]["sem001_training_manifest_sha256"])
    base = prediction_paths(args.sem150_root, names, manifest_sha256=config["inputs"]["sem150_prediction_manifest_sha256"], experiment_id="SEM-150-FOUNDATION-GATE-OOF")
    baseline_paths = prediction_paths(args.sem160_root, names, manifest_sha256=config["inputs"]["sem160_prediction_manifest_sha256"], experiment_id="SEM-160-DECK-BINARY-OOF")
    probability = oof_probability_paths(
        args.sem160_oof_root,
        read_json(args.oof_split),
        names,
        result_hashes=list(config["inputs"]["sem160_fold_result_sha256"]),
        manifest_hashes=list(config["inputs"]["sem160_fold_prediction_manifest_sha256"]),
    )
    baseline_metrics = {
        "calibration": sem160["selected_calibration_record"]["metrics"],
        "check": sem160["check_record"]["metrics"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output_dir / "input_access_manifest.json",
        {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "training_views": names,
            "partition": partition,
            "all_inputs_validated_before_labels": True,
            "fixed_validation_opened": False,
            "unlabeled_images_opened": False,
            "real_rgb_input_opened": False,
        },
    )
    calibration_records = []
    for threshold in config["arbitration"]["threshold_grid"]:
        evaluation = evaluate(
            names=partition["calibration"],
            threshold=float(threshold),
            labels_root=args.labels_root,
            base=base,
            sem001=sem001,
            probability=probability,
        )
        record = {
            "threshold": float(threshold),
            **evaluation,
            "gate": gate_for(evaluation, baseline_metrics["calibration"], config["oof_gates"]["calibration"]),
        }
        calibration_records.append(record)
        print(json.dumps({"phase": "calibration", "threshold": threshold, "deck_iou": evaluation["metrics"]["per_class"]["deck"]["iou"], "deck_precision": evaluation["metrics"]["per_class"]["deck"]["precision"], "deck_recall": evaluation["metrics"]["per_class"]["deck"]["recall"], "passed": record["gate"]["passed"]}), flush=True)
    selected = select_arbitration_threshold(calibration_records)
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "threshold_grid": list(config["arbitration"]["threshold_grid"]),
        "baseline_calibration_metrics": baseline_metrics["calibration"],
        "baseline_check_metrics": baseline_metrics["check"],
        "calibration_records": calibration_records,
        "input_access_manifest_sha256": sha256(args.output_dir / "input_access_manifest.json"),
        "fixed_validation_opened": False,
        "unlabeled_images_opened": False,
        "real_rgb_input_opened": False,
        "rgb_modified": False,
    }
    if selected is None:
        result.update({"status": "completed_calibration_gate_failed", "selected_threshold": None, "check_labels_opened": False, "next_step": "Drop arbitration and retain SEM-161."})
        write_json(args.output_dir / "result.json", result)
        print(json.dumps({"status": result["status"]}), flush=True)
        return 0
    threshold = float(selected["threshold"])
    prediction_root = args.output_dir / "selected_predictions"
    check = evaluate(names=partition["check"], threshold=threshold, labels_root=args.labels_root, base=base, sem001=sem001, probability=probability, prediction_root=prediction_root)
    check["gate"] = gate_for(check, baseline_metrics["check"], config["oof_gates"]["check"])
    calibration_materialised = evaluate(names=partition["calibration"], threshold=threshold, labels_root=args.labels_root, base=base, sem001=sem001, probability=probability, prediction_root=prediction_root)
    records = sorted(check["prediction_records"] + calibration_materialised["prediction_records"], key=lambda value: names.index(value["name"]))
    write_json(args.output_dir / "prediction_manifest.json", {"schema_version": 1, "experiment_id": config["experiment_id"], "status": "completed_selected_predictions", "selected_threshold": threshold, "predictions": records, "fixed_validation_opened": False})
    write_panels(names=list(config["outputs"]["training_visual_samples"]), labels_root=args.labels_root, baseline=baseline_paths, candidate_root=prediction_root, output_root=args.output_dir / "training_panels")
    baseline_per_view = []
    for name in partition["check"]:
        target = class_map(args.labels_root / name, context="SEM-170 check label")
        baseline_prediction = class_map(
            baseline_paths[name], context="SEM-170 SEM-160 check baseline"
        )
        local = confusion_metrics(local_confusion(target, baseline_prediction))
        baseline_per_view.append(
            {"view": name, "deck_iou": local["per_class"]["deck"]["iou"]}
        )
    comparison = {"improved": 0, "equal": 0, "regressed": 0}
    for before, after in zip(baseline_per_view, check["per_view"], strict=True):
        delta = float(after["deck_iou"]) - float(before["deck_iou"])
        comparison["improved" if delta > 1e-12 else "regressed" if delta < -1e-12 else "equal"] += 1
    passed = bool(check["gate"]["passed"])
    result.update(
        {
            "status": "completed_oof_gate_passed" if passed else "completed_check_gate_failed",
            "selected_threshold": threshold,
            "selected_calibration_record": {**selected, "prediction_records": []},
            "check_record": check,
            "check_deck_iou_per_view_comparison": comparison,
            "check_labels_opened": True,
            "prediction_manifest_sha256": sha256(args.output_dir / "prediction_manifest.json"),
            "next_step": "Register one frozen final application over SEM-161." if passed else "Drop arbitration and retain SEM-161.",
        }
    )
    write_json(args.output_dir / "result.json", result)
    print(json.dumps({"status": result["status"], "selected_threshold": threshold, "check_gate": check["gate"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
