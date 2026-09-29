#!/usr/bin/env python3
"""Run SEM-110's fixed protected background/stay_cable fusion qualification.

All input images are already-produced semantic arrays for the exact 219 labelled
training views.  The OOF teacher was not trained on the view it predicts.  The
fixed 74-view set, real RGB, unlabelled imagery, Scaffold training and external
camera inputs are intentionally absent from this interface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_sem080_structural_diag as common

from src.semantic_protected_fusion import (
    fuse_background_stay_cable,
    local_confusion,
    merge_confusions,
    metrics_from_confusion,
    partition_training_names,
    protected_fusion_gate,
    select_calibration_candidate,
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
    parser.add_argument("--sem101-oof-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    require(
        config.get("experiment_id") == "SEM-110-PROTECTED-FUSION-OOF",
        "SEM-110 received another config",
    )
    require(
        config.get("status") == "registered_not_run",
        "SEM-110 may run only from its registered config",
    )


def image(path: Path, *, context: str) -> np.ndarray:
    value = np.asarray(Image.open(path), dtype=np.uint8)
    if value.shape != (989, 1320) or int(value.max()) >= 5:
        raise ValueError(f"{context} is not a 1320x989 official five-class PNG: {path}")
    return value


def validate_base_manifest(
    root: Path, names: list[str], checkpoint_sha256: str
) -> dict[str, Path]:
    path = root / "training_sem001_render_manifest.json"
    require(path.is_file(), "SEM-110 SEM-001 training render manifest is missing")
    manifest = common.load_json(path)
    require(
        manifest.get("experiment_id") == "SEM-110-PROTECTED-FUSION-OOF",
        "SEM-110 base manifest has another experiment ID",
    )
    require(
        manifest.get("status") == "completed_training_sem001_render",
        "SEM-110 base render is incomplete",
    )
    require(
        manifest.get("sem001_checkpoint_sha256") == checkpoint_sha256,
        "SEM-110 base render used another SEM-001 checkpoint",
    )
    access = manifest.get("training_only_access", {})
    require(
        all(
            access.get(key) is False
            for key in (
                "labels_opened",
                "fixed_validation_views_opened",
                "unlabeled_images_opened",
                "real_rgb_input_opened",
                "external_camera_mode_present",
            )
        ),
        "SEM-110 base renderer violated its training-only access boundary",
    )
    records = manifest.get("rendered", [])
    require(
        [record.get("name") for record in records] == names,
        "SEM-110 base render does not exactly match training views",
    )
    result: dict[str, Path] = {}
    for record in records:
        value = root / str(record["semantic_png"])
        require(
            value.is_file() and sha256(value) == record.get("sha256"),
            f"SEM-110 base render hash mismatch: {record.get('name')}",
        )
        result[str(record["name"])] = value
    return result


def validate_oof_teacher(
    root: Path, oof_split: dict[str, Any], names: list[str]
) -> dict[str, Path]:
    expected_by_fold = {
        int(fold["fold"]): list(fold["oof"]) for fold in oof_split["folds"]
    }
    require(
        set(expected_by_fold) == {0, 1, 2},
        "SEM-110 requires SEM-101's three frozen OOF folds",
    )
    teacher: dict[str, Path] = {}
    for fold, expected in expected_by_fold.items():
        result = common.load_json(root / f"fold-{fold}" / "result.json")
        require(
            result.get("experiment_id") == "SEM-101-RENDERED-SEG-OOF",
            "SEM-110 teacher has another experiment ID",
        )
        require(
            result.get("status") == "completed_fold_oof",
            "SEM-110 teacher fold is incomplete",
        )
        require(
            result.get("fixed_validation_opened") is False
            and result.get("unlabeled_rgb_opened") is False,
            "SEM-110 teacher violated its training-only boundary",
        )
        manifest = common.load_json(root / f"fold-{fold}" / "prediction_manifest.json")
        require(
            [record.get("name") for record in manifest] == expected,
            f"SEM-110 teacher fold {fold} does not exactly match its OOF list",
        )
        for record in manifest:
            probability = root / f"fold-{fold}" / str(record["probabilities"])
            require(
                probability.is_file(),
                f"SEM-110 teacher probability is missing: {record['name']}",
            )
            require(
                sha256(probability) == record.get("probabilities_sha256"),
                f"SEM-110 teacher probability hash mismatch: {record['name']}",
            )
            teacher[str(record["name"])] = probability
    require(
        set(teacher) == set(names),
        "SEM-110 teacher does not cover exactly the training list",
    )
    return teacher


def teacher_probabilities(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        require(
            archive.files == ["probabilities"],
            "SEM-110 teacher NPZ has an unexpected schema",
        )
        return archive["probabilities"]


def evaluate(
    *,
    names: list[str],
    threshold: float | None,
    labels_root: Path,
    base_paths: dict[str, Path],
    teacher_paths: dict[str, Path],
    write_predictions: Path | None = None,
) -> dict[str, Any]:
    baseline_confusions: list[np.ndarray] = []
    fused_confusions: list[np.ndarray] = []
    protected_exact = True
    changed_pixels = 0
    per_view: list[dict[str, Any]] = []
    for name in names:
        target = image(labels_root / name, context="SEM-110 training label")
        base = image(base_paths[name], context="SEM-110 frozen SEM-001 output")
        baseline_confusions.append(local_confusion(target, base))
        if threshold is None:
            continue
        fused, fixed = fuse_background_stay_cable(
            base,
            teacher_probabilities(teacher_paths[name]),
            stay_cable_threshold=threshold,
        )
        protected_exact = protected_exact and bool(
            np.array_equal(base[fixed], fused[fixed])
        )
        changed_pixels += int(np.count_nonzero(base != fused))
        fused_confusions.append(local_confusion(target, fused))
        if write_predictions is not None:
            write_predictions.mkdir(parents=True, exist_ok=True)
            Image.fromarray(fused).save(write_predictions / name)
        per_view.append(
            {"view": name, "changed_pixels": int(np.count_nonzero(base != fused))}
        )
    result = {
        "baseline_metrics": metrics_from_confusion(
            merge_confusions(baseline_confusions)
        )
    }
    if threshold is not None:
        result.update(
            {
                "fused_metrics": metrics_from_confusion(
                    merge_confusions(fused_confusions)
                ),
                "protected_predictions_exact": protected_exact,
                "changed_pixels": changed_pixels,
                "per_view": per_view,
            }
        )
    return result


def panel(target: np.ndarray, base: np.ndarray, fused: np.ndarray) -> np.ndarray:
    error = np.zeros((*target.shape, 3), dtype=np.uint8)
    error[target != fused] = np.asarray([255, 48, 48], dtype=np.uint8)
    return np.concatenate(
        [semantic_rgb(target), semantic_rgb(base), semantic_rgb(fused), error], axis=1
    )


def write_panels(
    *,
    names: list[str],
    labels_root: Path,
    base_paths: dict[str, Path],
    fused_root: Path,
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = image(labels_root / name, context="SEM-110 training label")
        base = image(base_paths[name], context="SEM-110 frozen SEM-001 output")
        fused = image(fused_root / name, context="SEM-110 fused prediction")
        Image.fromarray(panel(target, base, fused)).save(output / name)


def gate_record(
    evaluation: dict[str, Any], gates: dict[str, Any], threshold: float
) -> dict[str, Any]:
    gate = protected_fusion_gate(
        baseline_metrics=evaluation["baseline_metrics"],
        fused_metrics=evaluation["fused_metrics"],
        protected_predictions_exact=bool(evaluation["protected_predictions_exact"]),
        gates=gates,
    )
    return {
        "stay_cable_threshold": threshold,
        "baseline_metrics": evaluation["baseline_metrics"],
        "fused_metrics": evaluation["fused_metrics"],
        "changed_pixels": evaluation["changed_pixels"],
        "gate": gate,
    }


def main() -> int:
    args = parse_args()
    config = common.load_json(args.config)
    split = common.load_json(args.split)
    oof_split = common.load_json(args.oof_split)
    validate_config(config)
    require(args.labels_root.is_dir(), "SEM-110 authorised training labels are missing")
    existing_files = list(args.output_dir.iterdir()) if args.output_dir.exists() else []
    resumable_input_manifest = args.output_dir / "input_access_manifest.json"
    require(
        not existing_files or existing_files == [resumable_input_manifest],
        "SEM-110 refuses to overwrite fusion output",
    )
    names = select_rendered_training_views(
        split, expected_count=int(config["data"]["training_images"])
    )
    partition = partition_training_names(names, expected_count=len(names))
    base_paths = validate_base_manifest(
        args.sem001_training_root, names, config["inputs"]["sem001_checkpoint_sha256"]
    )
    teacher_paths = validate_oof_teacher(args.sem101_oof_root, oof_split, names)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "partition": partition,
        "training_views": len(names),
        "inputs_validated_before_labels": {
            "sem001_training_predictions": True,
            "sem101_oof_probabilities": True,
            "fixed_validation_views_opened": False,
            "unlabeled_images_opened": False,
            "real_rgb_input_opened": False,
            "external_camera_mode_present": False,
        },
    }
    if resumable_input_manifest.exists():
        require(
            common.load_json(resumable_input_manifest) == input_manifest,
            "SEM-110 resumable input manifest differs from the registered inputs",
        )
    else:
        common.write_json(resumable_input_manifest, input_manifest)

    # Calibration labels are the sole source of threshold selection.
    thresholds = [
        float(value) for value in config["fusion"]["stay_cable_threshold_grid"]
    ]
    records: list[dict[str, Any]] = []
    for threshold in thresholds:
        record = gate_record(
            evaluate(
                names=partition["calibration"],
                threshold=threshold,
                labels_root=args.labels_root,
                base_paths=base_paths,
                teacher_paths=teacher_paths,
            ),
            config["training_oof_gates"],
            threshold,
        )
        records.append(record)
        print(
            json.dumps(
                {
                    "phase": "calibration",
                    "threshold": threshold,
                    "gate_passed": record["gate"]["passed"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    selected = select_calibration_candidate(records)
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "threshold_grid": thresholds,
        "calibration_records": records,
        "selection_rule": config["fusion"]["selection_rule"],
        "input_access_manifest_sha256": sha256(
            args.output_dir / "input_access_manifest.json"
        ),
        "fixed_validation_views_opened": False,
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
                "next_step": "Stop: no fixed calibration candidate qualified; do not read check labels or open 74-view.",
            }
        )
        common.write_json(args.output_dir / "result.json", result)
        print(
            json.dumps(
                {"status": result["status"], "selected_threshold": None}, sort_keys=True
            )
        )
        return 0

    # Check labels are read only after calibration has selected a single rule.
    threshold = float(selected["stay_cable_threshold"])
    check = gate_record(
        evaluate(
            names=partition["check"],
            threshold=threshold,
            labels_root=args.labels_root,
            base_paths=base_paths,
            teacher_paths=teacher_paths,
            write_predictions=args.output_dir / "selected_fused_predictions",
        ),
        config["training_oof_gates"],
        threshold,
    )
    # Complete selected output after the threshold is frozen.  Calibration labels
    # were already legitimately opened for selection; no validation path exists.
    evaluate(
        names=partition["calibration"],
        threshold=threshold,
        labels_root=args.labels_root,
        base_paths=base_paths,
        teacher_paths=teacher_paths,
        write_predictions=args.output_dir / "selected_fused_predictions",
    )
    write_panels(
        names=list(config["outputs"]["training_visual_samples"]),
        labels_root=args.labels_root,
        base_paths=base_paths,
        fused_root=args.output_dir / "selected_fused_predictions",
        output=args.output_dir / "training_panels",
    )
    result.update(
        {
            "status": "completed_oof_gate_passed"
            if check["gate"]["passed"]
            else "completed_oof_gate_failed",
            "selected_threshold": threshold,
            "check_labels_opened": True,
            "check_record": check,
            "selected_predictions": "selected_fused_predictions",
            "panels": [
                str((args.output_dir / "training_panels" / name).resolve())
                for name in config["outputs"]["training_visual_samples"]
            ],
            "next_step": "A passed training-only gate may justify a separately registered final candidate; either result forbids threshold changes in SEM-110.",
        }
    )
    common.write_json(args.output_dir / "result.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_threshold": threshold,
                "check_gate_passed": check["gate"]["passed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
