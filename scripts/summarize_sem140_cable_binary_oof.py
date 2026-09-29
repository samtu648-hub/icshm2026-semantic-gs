#!/usr/bin/env python3
"""Select and check SEM-140 cable-binary protected fusion on training OOF."""

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
from src.semantic_cable_binary import (
    fuse_cable_probability,
    relative_gate,
    score_metrics,
    select_calibration_threshold,
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
    parser.add_argument("--sem140-oof-root", type=Path, required=True)
    parser.add_argument("--sem110-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def class_map(path: Path, *, context: str) -> np.ndarray:
    value = np.asarray(Image.open(path), dtype=np.uint8)
    if value.shape != (989, 1320) or int(value.max()) >= 5:
        raise ValueError(f"{context} is not a 1320x989 five-class PNG: {path}")
    return value


def cable_probability(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        require(
            archive.files == ["cable_probability"],
            "SEM-140 cable probability NPZ has an unexpected schema",
        )
        return archive["cable_probability"]


def sem001_training_paths(
    root: Path, names: list[str], *, checkpoint_sha256: str
) -> dict[str, Path]:
    manifest_path = root / "training_sem001_render_manifest.json"
    require(manifest_path.is_file(), "SEM-140 SEM-001 training manifest is missing")
    manifest = read_json(manifest_path)
    require(
        manifest.get("status") == "completed_training_sem001_render",
        "SEM-140 SEM-001 training render is incomplete",
    )
    require(
        manifest.get("sem001_checkpoint_sha256") == checkpoint_sha256,
        "SEM-140 received another SEM-001 checkpoint",
    )
    require(
        [record["name"] for record in manifest["rendered"]] == names,
        "SEM-140 SEM-001 maps do not exactly match training views",
    )
    output: dict[str, Path] = {}
    for record in manifest["rendered"]:
        path = root / record["semantic_png"]
        require(
            path.is_file() and sha256(path) == record["sha256"],
            f"SEM-140 SEM-001 prediction hash mismatch: {record['name']}",
        )
        output[record["name"]] = path
    return output


def oof_probability_paths(
    root: Path, oof_split: dict[str, Any], names: list[str]
) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for fold in oof_split["folds"]:
        fold_index = int(fold["fold"])
        fold_root = root / f"fold-{fold_index}"
        result = read_json(fold_root / "result.json")
        require(
            result.get("experiment_id") == "SEM-140-CABLE-BINARY-OOF"
            and result.get("status") == "completed_fold_oof",
            f"SEM-140 fold {fold_index} is incomplete",
        )
        require(
            result.get("fixed_validation_opened") is False
            and result.get("unlabeled_rgb_opened") is False,
            f"SEM-140 fold {fold_index} violated its training-only boundary",
        )
        records = read_json(fold_root / "prediction_manifest.json")
        require(
            [record["name"] for record in records] == list(fold["oof"]),
            f"SEM-140 fold {fold_index} predictions do not match the OOF split",
        )
        for record in records:
            path = fold_root / record["cable_probability"]
            require(
                path.is_file()
                and sha256(path) == record["cable_probability_sha256"],
                f"SEM-140 OOF probability hash mismatch: {record['name']}",
            )
            output[record["name"]] = path
    require(set(output) == set(names), "SEM-140 OOF does not cover all training views")
    return output


def evaluate(
    *,
    names: list[str],
    threshold: float,
    labels_root: Path,
    base_paths: dict[str, Path],
    probability_paths: dict[str, Path],
    prediction_root: Path | None = None,
) -> dict[str, Any]:
    confusions: list[np.ndarray] = []
    protected_exact = True
    changed_pixels = 0
    for name in names:
        target = class_map(labels_root / name, context="SEM-140 training label")
        base = class_map(base_paths[name], context="SEM-140 SEM-001 training output")
        fused, protected = fuse_cable_probability(
            base, cable_probability(probability_paths[name]), threshold=threshold
        )
        protected_exact = protected_exact and bool(
            np.array_equal(fused[protected], base[protected])
        )
        changed_pixels += int(np.count_nonzero(fused != base))
        confusions.append(local_confusion(target, fused))
        if prediction_root is not None:
            prediction_root.mkdir(parents=True, exist_ok=True)
            Image.fromarray(fused).save(prediction_root / name)
    return {
        "metrics": metrics_from_confusion(merge_confusions(confusions)),
        "protected_predictions_exact": protected_exact,
        "changed_pixels": changed_pixels,
    }


def parent_baselines(sem110: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    threshold = float(sem110["selected_threshold"])
    calibration = [
        record
        for record in sem110["calibration_records"]
        if float(record["stay_cable_threshold"]) == threshold
    ]
    require(len(calibration) == 1, "SEM-140 could not resolve SEM-110 calibration")
    return calibration[0]["fused_metrics"], sem110["check_record"]["fused_metrics"]


def gate_for(
    evaluation: dict[str, Any], baseline: dict[str, Any], gate: dict[str, Any]
) -> dict[str, Any]:
    return relative_gate(
        candidate=evaluation["metrics"],
        baseline=baseline,
        protected_predictions_exact=evaluation["protected_predictions_exact"],
        minimum_iou_gain=float(gate["minimum_stay_cable_iou_gain"]),
        minimum_recall_gain=float(gate["minimum_stay_cable_recall_gain"]),
        minimum_miou_gain=float(gate["minimum_mIoU_gain"]),
        maximum_background_fp_ratio=float(
            gate["maximum_background_false_positive_ratio"]
        ),
    )


def write_panels(
    *,
    names: list[str],
    labels_root: Path,
    base_paths: dict[str, Path],
    prediction_root: Path,
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = class_map(labels_root / name, context="SEM-140 panel target")
        base = class_map(base_paths[name], context="SEM-140 panel SEM-001")
        fused = class_map(prediction_root / name, context="SEM-140 panel fused")
        error = np.zeros((*target.shape, 3), dtype=np.uint8)
        error[target != fused] = np.asarray([255, 48, 48], dtype=np.uint8)
        panel = np.concatenate(
            [semantic_rgb(target), semantic_rgb(base), semantic_rgb(fused), error],
            axis=1,
        )
        Image.fromarray(panel).save(output_root / name)


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    require(
        config.get("experiment_id") == "SEM-140-CABLE-BINARY-OOF",
        "SEM-140 summary received another experiment",
    )
    require(
        config.get("status") == "registered_not_run",
        "SEM-140 summary requires its registered configuration",
    )
    require(
        not args.output_dir.exists() or not any(args.output_dir.iterdir()),
        "SEM-140 refuses to overwrite an OOF summary",
    )
    require(
        sha256(args.split) == config["data"]["semantic_split_sha256"]
        and sha256(args.oof_split) == config["data"]["oof_split_sha256"],
        "SEM-140 summary split hash mismatch",
    )
    require(
        sha256(args.sem110_result) == config["inputs"]["sem110_result_sha256"],
        "SEM-140 SEM-110 result differs from the registered parent",
    )
    semantic_split = read_json(args.split)
    oof_split = read_json(args.oof_split)
    names = select_rendered_training_views(
        semantic_split, expected_count=int(config["data"]["training_images"])
    )
    partition = partition_training_names(names, expected_count=len(names))
    base_paths = sem001_training_paths(
        args.sem001_training_root,
        names,
        checkpoint_sha256=config["inputs"]["sem001_checkpoint_sha256"],
    )
    probability_paths = oof_probability_paths(args.sem140_oof_root, oof_split, names)
    sem110 = read_json(args.sem110_result)
    require(
        sem110.get("status") == "completed_oof_gate_passed",
        "SEM-140 requires the completed SEM-110 baseline",
    )
    baseline_calibration, baseline_check = parent_baselines(sem110)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output_dir / "input_access_manifest.json",
        {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "training_views": names,
            "partition": partition,
            "sem110_result_sha256": sha256(args.sem110_result),
            "all_oof_inputs_validated_before_labels": True,
            "fixed_validation_opened": False,
            "unlabeled_images_opened": False,
            "real_rgb_input_opened": False,
        },
    )

    records: list[dict[str, Any]] = []
    for threshold in config["fusion"]["threshold_grid"]:
        evaluation = evaluate(
            names=partition["calibration"],
            threshold=float(threshold),
            labels_root=args.labels_root,
            base_paths=base_paths,
            probability_paths=probability_paths,
        )
        record = {
            "threshold": float(threshold),
            **evaluation,
            "score": score_metrics(evaluation["metrics"]),
            "gate": gate_for(
                evaluation, baseline_calibration, config["oof_gates"]["calibration"]
            ),
        }
        records.append(record)
        print(
            json.dumps(
                {
                    "phase": "calibration",
                    "threshold": threshold,
                    "score": record["score"],
                    "gate_passed": record["gate"]["passed"],
                }
            ),
            flush=True,
        )
    selected = select_calibration_threshold(records)
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "calibration_records": records,
        "baseline_calibration_metrics": baseline_calibration,
        "baseline_check_metrics": baseline_check,
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
                "next_step": "Stop before SEM-141; no calibration threshold improved the protected SEM-110 baseline.",
            }
        )
        write_json(args.output_dir / "result.json", result)
        print(json.dumps({"status": result["status"]}), flush=True)
        return 0

    threshold = float(selected["threshold"])
    prediction_root = args.output_dir / "selected_fused_predictions"
    check = evaluate(
        names=partition["check"],
        threshold=threshold,
        labels_root=args.labels_root,
        base_paths=base_paths,
        probability_paths=probability_paths,
        prediction_root=prediction_root,
    )
    check_gate = gate_for(check, baseline_check, config["oof_gates"]["check"])
    evaluate(
        names=partition["calibration"],
        threshold=threshold,
        labels_root=args.labels_root,
        base_paths=base_paths,
        probability_paths=probability_paths,
        prediction_root=prediction_root,
    )
    write_panels(
        names=list(config["outputs"]["training_visual_samples"]),
        labels_root=args.labels_root,
        base_paths=base_paths,
        prediction_root=prediction_root,
        output_root=args.output_dir / "training_panels",
    )
    result.update(
        {
            "status": "completed_oof_gate_passed"
            if check_gate["passed"]
            else "completed_oof_gate_failed",
            "selected_threshold": threshold,
            "selected_calibration_record": selected,
            "check_labels_opened": True,
            "check_record": {**check, "gate": check_gate},
            "selected_predictions": "selected_fused_predictions",
            "next_step": "A passing OOF gate permits SEM-141 final training; a failure stops this binary recipe without validation tuning.",
        }
    )
    write_json(args.output_dir / "result.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_threshold": threshold,
                "check_gate_passed": check_gate["passed"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
