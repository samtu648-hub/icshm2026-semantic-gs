#!/usr/bin/env python3
"""Select and independently check SEM-172 against frozen SEM-160 OOF."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_sem170_deck_arbitration_oof import prediction_paths
from scripts.summarize_sem160_deck_binary_oof import (
    base_paths,
    class_map,
    evaluate,
    gate_for,
    require,
    sem001_paths,
    write_panels,
)
from src.semantic2d import confusion_metrics, read_json, sha256, write_json
from src.semantic_deck_binary import select_calibration_threshold
from src.semantic_protected_fusion import local_confusion, partition_training_names
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
    parser.add_argument("--sem172-oof-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def oof_probability_paths(
    root: Path, oof_split: dict[str, Any], names: list[str]
) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    output = {}
    fold_audit = []
    for fold in oof_split["folds"]:
        fold_index = int(fold["fold"])
        fold_root = root / f"fold-{fold_index}"
        result_path = fold_root / "result.json"
        manifest_path = fold_root / "prediction_manifest.json"
        require(
            result_path.is_file() and manifest_path.is_file(),
            f"SEM-172 fold {fold_index} artifacts are missing",
        )
        result = read_json(result_path)
        require(
            result.get("experiment_id") == "SEM-172-DECK-INTERIOR-OOF"
            and result.get("status") == "completed_fold_oof",
            f"SEM-172 fold {fold_index} is incomplete",
        )
        require(
            result.get("fixed_validation_opened") is False
            and result.get("unlabeled_rgb_opened") is False
            and result.get("real_rgb_input_opened") is False
            and result.get("checkpoint_reload") == "passed",
            f"SEM-172 fold {fold_index} violated its training-only boundary",
        )
        records = read_json(manifest_path)
        require(
            [record["name"] for record in records] == list(fold["oof"]),
            f"SEM-172 fold {fold_index} prediction order mismatch",
        )
        for record in records:
            path = fold_root / record["deck_probability"]
            require(
                path.is_file()
                and sha256(path) == record["deck_probability_sha256"],
                f"SEM-172 OOF probability hash mismatch: {record['name']}",
            )
            output[record["name"]] = path
        fold_audit.append(
            {
                "fold": fold_index,
                "result_sha256": sha256(result_path),
                "prediction_manifest_sha256": sha256(manifest_path),
                "checkpoint_sha256": result["checkpoint_sha256"],
                "training_seconds": result["training_seconds"],
                "inference_seconds": result["inference_seconds"],
                "gpu_hours_training": result["gpu_hours_training"],
                "peak_cuda_allocated_mib": result["peak_cuda_allocated_mib"],
                "checkpoint_reload": result["checkpoint_reload"],
            }
        )
    require(set(output) == set(names), "SEM-172 OOF does not cover all training views")
    return output, fold_audit


def comparison_to_sem160(
    *,
    names: list[str],
    labels_root: Path,
    sem160_paths: dict[str, Path],
    candidate_root: Path,
) -> dict[str, Any]:
    changed_to_deck_by_gt = np.zeros(5, dtype=np.int64)
    changed_from_deck_by_gt = np.zeros(5, dtype=np.int64)
    per_view = []
    improved = equal = regressed = 0
    for name in names:
        target = class_map(labels_root / name, context="SEM-172 comparison label")
        baseline = class_map(
            sem160_paths[name], context="SEM-172 SEM-160 comparison"
        )
        candidate = class_map(
            candidate_root / name, context="SEM-172 candidate comparison"
        )
        additions = (baseline != 1) & (candidate == 1)
        removals = (baseline == 1) & (candidate != 1)
        changed_to_deck_by_gt += np.bincount(target[additions], minlength=5)
        changed_from_deck_by_gt += np.bincount(target[removals], minlength=5)
        before_iou = confusion_metrics(local_confusion(target, baseline))["per_class"][
            "deck"
        ]["iou"]
        after_iou = confusion_metrics(local_confusion(target, candidate))["per_class"][
            "deck"
        ]["iou"]
        delta = float(after_iou) - float(before_iou)
        if delta > 1e-12:
            improved += 1
        elif delta < -1e-12:
            regressed += 1
        else:
            equal += 1
        per_view.append(
            {
                "view": name,
                "sem160_deck_iou": before_iou,
                "sem172_deck_iou": after_iou,
                "delta": delta,
            }
        )
    return {
        "changed_to_deck_by_gt_class": changed_to_deck_by_gt.astype(int).tolist(),
        "changed_from_deck_by_gt_class": changed_from_deck_by_gt.astype(int).tolist(),
        "changed_to_deck_pixels": int(changed_to_deck_by_gt.sum()),
        "changed_from_deck_pixels": int(changed_from_deck_by_gt.sum()),
        "deck_iou_improved_views": improved,
        "deck_iou_equal_views": equal,
        "deck_iou_regressed_views": regressed,
        "per_view": per_view,
    }


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    require(
        config.get("experiment_id") == "SEM-172-DECK-INTERIOR-OOF",
        "SEM-172 summary received another experiment config",
    )
    require(
        config.get("status") == "registered_not_run",
        "SEM-172 summary requires its registered configuration",
    )
    require(
        not args.output_dir.exists() or not any(args.output_dir.iterdir()),
        "SEM-172 refuses to overwrite its OOF summary",
    )
    require(
        sha256(args.split) == config["data"]["semantic_split_sha256"]
        and sha256(args.oof_split) == config["data"]["oof_split_sha256"],
        "SEM-172 split hash mismatch",
    )
    require(
        sha256(args.sem150_result) == config["inputs"]["sem150_result_sha256"],
        "SEM-172 SEM-150 result hash mismatch",
    )
    require(
        sha256(args.sem160_result) == config["inputs"]["sem160_result_sha256"],
        "SEM-172 SEM-160 result hash mismatch",
    )
    sem150 = read_json(args.sem150_result)
    sem160 = read_json(args.sem160_result)
    require(
        sem150.get("status") == "completed_oof_gate_passed"
        and sem160.get("status") == "completed_oof_gate_passed",
        "SEM-172 requires passing SEM-150 and SEM-160 parents",
    )
    names = select_rendered_training_views(
        read_json(args.split), expected_count=int(config["data"]["training_images"])
    )
    oof_split = read_json(args.oof_split)
    partition = partition_training_names(names, expected_count=len(names))
    sem001 = sem001_paths(
        args.sem001_training_root,
        names,
        manifest_sha256=config["inputs"]["sem001_training_manifest_sha256"],
    )
    base = base_paths(
        args.sem150_root,
        names,
        manifest_sha256=config["inputs"]["sem150_prediction_manifest_sha256"],
    )
    sem160_paths = prediction_paths(
        args.sem160_root,
        names,
        manifest_sha256=config["inputs"]["sem160_prediction_manifest_sha256"],
        experiment_id="SEM-160-DECK-BINARY-OOF",
    )
    probability, fold_audit = oof_probability_paths(
        args.sem172_oof_root, oof_split, names
    )
    baseline_calibration = sem160["selected_calibration_record"]["metrics"]
    baseline_check = sem160["check_record"]["metrics"]

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output_dir / "input_access_manifest.json",
        {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "training_views": names,
            "partition": partition,
            "sem150_result_sha256": sha256(args.sem150_result),
            "sem160_result_sha256": sha256(args.sem160_result),
            "fold_audit": fold_audit,
            "all_oof_inputs_validated_before_labels": True,
            "fixed_validation_opened": False,
            "unlabeled_images_opened": False,
            "real_rgb_input_opened": False,
        },
    )

    records = []
    for threshold in config["fusion"]["threshold_grid"]:
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
            "gate": gate_for(
                evaluation,
                baseline_calibration,
                config["oof_gates"]["calibration"],
            ),
        }
        records.append(record)
        print(
            json.dumps(
                {
                    "phase": "calibration",
                    "threshold": threshold,
                    "deck_iou": evaluation["metrics"]["per_class"]["deck"]["iou"],
                    "deck_precision": evaluation["metrics"]["per_class"]["deck"][
                        "precision"
                    ],
                    "deck_recall": evaluation["metrics"]["per_class"]["deck"][
                        "recall"
                    ],
                    "gate_passed": record["gate"]["passed"],
                }
            ),
            flush=True,
        )
    selected = select_calibration_threshold(records)
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "threshold_grid": list(config["fusion"]["threshold_grid"]),
        "calibration_records": records,
        "baseline_calibration_metrics": baseline_calibration,
        "baseline_check_metrics": baseline_check,
        "fold_audit": fold_audit,
        "total_training_seconds": float(
            sum(record["training_seconds"] for record in fold_audit)
        ),
        "total_inference_seconds": float(
            sum(record["inference_seconds"] for record in fold_audit)
        ),
        "total_gpu_hours_training": float(
            sum(record["gpu_hours_training"] for record in fold_audit)
        ),
        "peak_cuda_allocated_mib": max(
            int(record["peak_cuda_allocated_mib"]) for record in fold_audit
        ),
        "input_access_manifest_sha256": sha256(
            args.output_dir / "input_access_manifest.json"
        ),
        "fixed_validation_opened": False,
        "unlabeled_images_opened": False,
        "real_rgb_input_opened": False,
        "rgb_modified": False,
    }
    if selected is None:
        result.update(
            {
                "status": "completed_calibration_gate_failed",
                "selected_threshold": None,
                "check_labels_opened": False,
                "next_step": "Drop interior-weighted training and retain SEM-160/161.",
            }
        )
        write_json(args.output_dir / "result.json", result)
        print(json.dumps({"status": result["status"]}), flush=True)
        return 0

    threshold = float(selected["threshold"])
    prediction_root = args.output_dir / "selected_predictions"
    check = evaluate(
        names=partition["check"],
        threshold=threshold,
        labels_root=args.labels_root,
        base=base,
        sem001=sem001,
        probability=probability,
        prediction_root=prediction_root,
    )
    check_gate = gate_for(check, baseline_check, config["oof_gates"]["check"])
    calibration_materialised = evaluate(
        names=partition["calibration"],
        threshold=threshold,
        labels_root=args.labels_root,
        base=base,
        sem001=sem001,
        probability=probability,
        prediction_root=prediction_root,
    )
    records_by_name = {
        record["name"]: record
        for record in check["prediction_records"]
        + calibration_materialised["prediction_records"]
    }
    prediction_records = [records_by_name[name] for name in names]
    write_json(
        args.output_dir / "prediction_manifest.json",
        {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "status": "completed_selected_predictions",
            "selected_threshold": threshold,
            "predictions": prediction_records,
            "fixed_validation_opened": False,
        },
    )
    write_panels(
        names=list(config["outputs"]["training_visual_samples"]),
        labels_root=args.labels_root,
        base=sem160_paths,
        prediction_root=prediction_root,
        output_root=args.output_dir / "training_panels",
    )
    comparison = comparison_to_sem160(
        names=names,
        labels_root=args.labels_root,
        sem160_paths=sem160_paths,
        candidate_root=prediction_root,
    )
    passed = bool(check_gate["passed"])
    result.update(
        {
            "status": (
                "completed_oof_gate_passed"
                if passed
                else "completed_oof_gate_failed"
            ),
            "selected_threshold": threshold,
            "selected_calibration_record": selected,
            "check_labels_opened": True,
            "check_record": {**check, "gate": check_gate},
            "comparison_to_sem160_all_training": comparison,
            "prediction_manifest_sha256": sha256(
                args.output_dir / "prediction_manifest.json"
            ),
            "next_step": (
                "Register one frozen full-data final model."
                if passed
                else "Drop interior-weighted training and retain SEM-160/161."
            ),
        }
    )
    write_json(args.output_dir / "result.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_threshold": threshold,
                "check_gate": check_gate,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
