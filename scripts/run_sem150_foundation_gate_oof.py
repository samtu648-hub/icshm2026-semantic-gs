#!/usr/bin/env python3
"""Qualify a one-way foundation verifier using frozen training OOF evidence."""

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

from src.semantic2d import read_json, sha256, write_json
from src.semantic_cable_binary import fuse_cable_probability
from src.semantic_foundation_verifier import (
    confusion_after_foundation_rejection,
    foundation_gate,
    select_foundation_threshold,
    verify_foundation,
)
from src.semantic_protected_fusion import (
    local_confusion,
    merge_confusions,
    metrics_from_confusion,
    partition_training_names,
    validate_probabilities,
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
    parser.add_argument("--sem140-oof-root", type=Path, required=True)
    parser.add_argument("--sem140-result", type=Path, required=True)
    parser.add_argument("--sem101-oof-root", type=Path, required=True)
    parser.add_argument("--sem101-result", type=Path, required=True)
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


def scalar_cable_probability(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        require(
            archive.files == ["cable_probability"],
            "SEM-150 cable probability NPZ has an unexpected schema",
        )
        return archive["cable_probability"]


def five_class_probabilities(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        require(
            archive.files == ["probabilities"],
            "SEM-150 foundation teacher NPZ has an unexpected schema",
        )
        value = archive["probabilities"]
    return validate_probabilities(value, shape=(989, 1320))


def sem001_paths(
    root: Path, names: list[str], *, manifest_sha256: str, checkpoint_sha256: str
) -> dict[str, Path]:
    manifest_path = root / "training_sem001_render_manifest.json"
    require(
        manifest_path.is_file() and sha256(manifest_path) == manifest_sha256,
        "SEM-150 SEM-001 training manifest hash mismatch",
    )
    manifest = read_json(manifest_path)
    require(
        manifest.get("status") == "completed_training_sem001_render"
        and manifest.get("sem001_checkpoint_sha256") == checkpoint_sha256,
        "SEM-150 SEM-001 training render is not the frozen parent",
    )
    require(
        [record["name"] for record in manifest["rendered"]] == names,
        "SEM-150 SEM-001 maps do not exactly match training views",
    )
    output: dict[str, Path] = {}
    for record in manifest["rendered"]:
        path = root / record["semantic_png"]
        require(
            path.is_file() and sha256(path) == record["sha256"],
            f"SEM-150 SEM-001 hash mismatch: {record['name']}",
        )
        output[record["name"]] = path
    return output


def oof_paths(
    *,
    root: Path,
    oof_split: dict[str, Any],
    names: list[str],
    experiment_id: str,
    probability_key: str,
) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for fold in oof_split["folds"]:
        fold_index = int(fold["fold"])
        fold_root = root / f"fold-{fold_index}"
        result = read_json(fold_root / "result.json")
        require(
            result.get("experiment_id") == experiment_id
            and result.get("status") == "completed_fold_oof",
            f"SEM-150 {experiment_id} fold {fold_index} is incomplete",
        )
        require(
            result.get("fixed_validation_opened") is False
            and result.get("unlabeled_rgb_opened") is False,
            f"SEM-150 {experiment_id} fold {fold_index} leaked forbidden inputs",
        )
        records = read_json(fold_root / "prediction_manifest.json")
        require(
            [record["name"] for record in records] == list(fold["oof"]),
            f"SEM-150 {experiment_id} fold {fold_index} OOF order mismatch",
        )
        for record in records:
            path = fold_root / record[probability_key]
            require(
                path.is_file()
                and sha256(path) == record[f"{probability_key}_sha256"],
                f"SEM-150 {experiment_id} probability hash mismatch: {record['name']}",
            )
            output[record["name"]] = path
    require(set(output) == set(names), f"SEM-150 {experiment_id} OOF coverage mismatch")
    return output


def build_baseline(
    sem001: np.ndarray, cable_path: Path, *, cable_threshold: float
) -> np.ndarray:
    baseline, _ = fuse_cable_probability(
        sem001,
        scalar_cable_probability(cable_path),
        threshold=cable_threshold,
    )
    return baseline


def evaluate_thresholds(
    *,
    names: list[str],
    thresholds: list[float],
    labels_root: Path,
    sem001: dict[str, Path],
    cable: dict[str, Path],
    foundation: dict[str, Path],
    cable_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline_confusions: list[np.ndarray] = []
    rejected_by_gt = {
        threshold: np.zeros(5, dtype=np.int64) for threshold in thresholds
    }
    candidate_pixels = 0
    for name in names:
        target = class_map(labels_root / name, context="SEM-150 training label")
        base = build_baseline(
            class_map(sem001[name], context="SEM-150 SEM-001 prediction"),
            cable[name],
            cable_threshold=cable_threshold,
        )
        baseline_confusions.append(local_confusion(target, base))
        probability = five_class_probabilities(foundation[name])[4]
        candidates = base == 4
        candidate_pixels += int(np.count_nonzero(candidates))
        for threshold in thresholds:
            rejected = candidates & (probability < threshold)
            rejected_by_gt[threshold] += np.bincount(
                target[rejected], minlength=5
            ).astype(np.int64)
    baseline_confusion = merge_confusions(baseline_confusions)
    baseline_metrics = metrics_from_confusion(baseline_confusion)
    records: list[dict[str, Any]] = []
    for threshold in thresholds:
        confusion = confusion_after_foundation_rejection(
            baseline_confusion, rejected_by_gt[threshold]
        )
        records.append(
            {
                "threshold": threshold,
                "metrics": metrics_from_confusion(confusion),
                "candidate_foundation_pixels": candidate_pixels,
                "rejected_pixels": int(rejected_by_gt[threshold].sum()),
                "rejected_by_gt_class": rejected_by_gt[threshold].astype(int).tolist(),
            }
        )
    return baseline_metrics, records


def attach_gates(
    records: list[dict[str, Any]], baseline: dict[str, Any], gate: dict[str, Any]
) -> None:
    for record in records:
        record["gate"] = foundation_gate(
            candidate=record["metrics"],
            baseline=baseline,
            immutable_predictions_exact=True,
            minimum_iou_gain=float(gate["minimum_foundation_iou_gain"]),
            minimum_precision_gain=float(
                gate["minimum_foundation_precision_gain"]
            ),
            minimum_recall=float(gate["minimum_foundation_recall"]),
            minimum_miou_gain=float(gate["minimum_mIoU_gain"]),
            minimum_background_fp_reduction=float(
                gate["minimum_background_to_foundation_reduction"]
            ),
        )


def materialise(
    *,
    names: list[str],
    threshold: float,
    sem001: dict[str, Path],
    cable: dict[str, Path],
    foundation: dict[str, Path],
    cable_threshold: float,
    output_root: Path,
) -> list[dict[str, Any]]:
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for name in names:
        base = build_baseline(
            class_map(sem001[name], context="SEM-150 SEM-001 prediction"),
            cable[name],
            cable_threshold=cable_threshold,
        )
        verified, _, rejected = verify_foundation(
            base, five_class_probabilities(foundation[name])[4], threshold=threshold
        )
        path = output_root / name
        Image.fromarray(verified).save(path)
        records.append(
            {
                "name": name,
                "prediction": str(path.relative_to(output_root.parent)),
                "prediction_sha256": sha256(path),
                "rejected_foundation_pixels": int(np.count_nonzero(rejected)),
            }
        )
    return records


def write_panels(
    *,
    names: list[str],
    labels_root: Path,
    sem001: dict[str, Path],
    cable: dict[str, Path],
    cable_threshold: float,
    prediction_root: Path,
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = class_map(labels_root / name, context="SEM-150 panel target")
        baseline = build_baseline(
            class_map(sem001[name], context="SEM-150 panel SEM-001"),
            cable[name],
            cable_threshold=cable_threshold,
        )
        prediction = class_map(
            prediction_root / name, context="SEM-150 panel prediction"
        )
        changed = np.zeros((*target.shape, 3), dtype=np.uint8)
        changed[baseline != prediction] = np.asarray([255, 255, 255], dtype=np.uint8)
        panel = np.concatenate(
            [semantic_rgb(target), semantic_rgb(baseline), semantic_rgb(prediction), changed],
            axis=1,
        )
        Image.fromarray(panel).save(output_root / name)


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    require(
        config.get("experiment_id") == "SEM-150-FOUNDATION-GATE-OOF",
        "SEM-150 received another experiment config",
    )
    require(
        config.get("status") == "registered_not_run",
        "SEM-150 may run only from its registered configuration",
    )
    require(
        not args.output_dir.exists() or not any(args.output_dir.iterdir()),
        "SEM-150 refuses to overwrite an output directory",
    )
    require(
        sha256(args.split) == config["data"]["semantic_split_sha256"]
        and sha256(args.oof_split) == config["data"]["oof_split_sha256"],
        "SEM-150 split hash mismatch",
    )
    require(
        sha256(args.sem140_result) == config["inputs"]["sem140_result_sha256"],
        "SEM-150 SEM-140 parent result hash mismatch",
    )
    require(
        sha256(args.sem101_result) == config["inputs"]["sem101_result_sha256"],
        "SEM-150 SEM-101 parent result hash mismatch",
    )
    sem140_result = read_json(args.sem140_result)
    sem101_result = read_json(args.sem101_result)
    require(
        sem140_result.get("status") == "completed_oof_gate_passed"
        and sem101_result.get("status") == "completed_oof_gate_passed",
        "SEM-150 requires passing SEM-140 and SEM-101 OOF parents",
    )
    cable_threshold = float(sem140_result["selected_threshold"])
    require(
        cable_threshold == float(config["fusion"]["frozen_cable_threshold"]),
        "SEM-150 cable threshold differs from frozen SEM-140",
    )
    semantic_split = read_json(args.split)
    oof_split = read_json(args.oof_split)
    names = select_rendered_training_views(
        semantic_split, expected_count=int(config["data"]["training_images"])
    )
    partition = partition_training_names(names, expected_count=len(names))
    sem001 = sem001_paths(
        args.sem001_training_root,
        names,
        manifest_sha256=config["inputs"]["sem001_training_manifest_sha256"],
        checkpoint_sha256=config["inputs"]["sem001_checkpoint_sha256"],
    )
    cable = oof_paths(
        root=args.sem140_oof_root,
        oof_split=oof_split,
        names=names,
        experiment_id="SEM-140-CABLE-BINARY-OOF",
        probability_key="cable_probability",
    )
    foundation = oof_paths(
        root=args.sem101_oof_root,
        oof_split=oof_split,
        names=names,
        experiment_id="SEM-101-RENDERED-SEG-OOF",
        probability_key="probabilities",
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output_dir / "input_access_manifest.json",
        {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "training_views": names,
            "partition": partition,
            "sem140_result_sha256": sha256(args.sem140_result),
            "sem101_result_sha256": sha256(args.sem101_result),
            "all_oof_inputs_validated_before_labels": True,
            "fixed_validation_opened": False,
            "unlabeled_images_opened": False,
            "real_rgb_input_opened": False,
            "external_camera_mode_present": False,
        },
    )

    thresholds = [float(value) for value in config["fusion"]["threshold_grid"]]
    baseline_calibration, calibration_records = evaluate_thresholds(
        names=partition["calibration"],
        thresholds=thresholds,
        labels_root=args.labels_root,
        sem001=sem001,
        cable=cable,
        foundation=foundation,
        cable_threshold=cable_threshold,
    )
    attach_gates(
        calibration_records, baseline_calibration, config["oof_gates"]["calibration"]
    )
    for record in calibration_records:
        print(
            json.dumps(
                {
                    "phase": "calibration",
                    "threshold": record["threshold"],
                    "gate_passed": record["gate"]["passed"],
                    "foundation_iou": record["metrics"]["per_class"]["foundation"]["iou"],
                    "foundation_precision": record["metrics"]["per_class"]["foundation"]["precision"],
                    "foundation_recall": record["metrics"]["per_class"]["foundation"]["recall"],
                }
            ),
            flush=True,
        )
    selected = select_foundation_threshold(calibration_records)
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "threshold_grid": thresholds,
        "calibration_records": calibration_records,
        "baseline_calibration_metrics": baseline_calibration,
        "input_access_manifest_sha256": sha256(
            args.output_dir / "input_access_manifest.json"
        ),
        "fixed_validation_opened": False,
        "unlabeled_images_opened": False,
        "real_rgb_input_opened": False,
        "external_camera_mode_present": False,
        "rgb_modified": False,
    }
    if selected is None:
        result.update(
            {
                "status": "completed_calibration_gate_failed",
                "selected_threshold": None,
                "check_labels_opened": False,
                "next_step": "Stop the reused-teacher gate and register a specialised foundation verifier only under a new ID.",
            }
        )
        write_json(args.output_dir / "result.json", result)
        print(json.dumps({"status": result["status"]}), flush=True)
        return 0

    threshold = float(selected["threshold"])
    baseline_check, check_records = evaluate_thresholds(
        names=partition["check"],
        thresholds=[threshold],
        labels_root=args.labels_root,
        sem001=sem001,
        cable=cable,
        foundation=foundation,
        cable_threshold=cable_threshold,
    )
    attach_gates(check_records, baseline_check, config["oof_gates"]["check"])
    check = check_records[0]
    prediction_root = args.output_dir / "selected_predictions"
    prediction_records = materialise(
        names=names,
        threshold=threshold,
        sem001=sem001,
        cable=cable,
        foundation=foundation,
        cable_threshold=cable_threshold,
        output_root=prediction_root,
    )
    write_json(
        args.output_dir / "prediction_manifest.json",
        {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "selected_threshold": threshold,
            "predictions": prediction_records,
            "fixed_validation_opened": False,
        },
    )
    write_panels(
        names=list(config["outputs"]["training_visual_samples"]),
        labels_root=args.labels_root,
        sem001=sem001,
        cable=cable,
        cable_threshold=cable_threshold,
        prediction_root=prediction_root,
        output_root=args.output_dir / "training_panels",
    )
    result.update(
        {
            "status": "completed_oof_gate_passed"
            if check["gate"]["passed"]
            else "completed_oof_gate_failed",
            "selected_threshold": threshold,
            "check_labels_opened": True,
            "baseline_check_metrics": baseline_check,
            "check_record": check,
            "prediction_manifest_sha256": sha256(
                args.output_dir / "prediction_manifest.json"
            ),
            "panels": [
                str((args.output_dir / "training_panels" / name).resolve())
                for name in config["outputs"]["training_visual_samples"]
            ],
            "next_step": "A pass permits one separately registered final developer audit with the frozen threshold; a failure permits only a new specialised verifier experiment.",
        }
    )
    write_json(args.output_dir / "result.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_threshold": threshold,
                "check_gate_passed": check["gate"]["passed"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
