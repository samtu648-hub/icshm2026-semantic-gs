#!/usr/bin/env python3
"""Run SEM-112's training-only rendered-RGB edge-connected cable recovery.

This script has no fixed-validation, unlabelled, real-RGB or external-camera
argument.  It validates every frozen SEM-110/SEM-101 input before opening an
authorised training label.  Calibration selects exactly one edge quantile; the
check half is opened only after that selection and can never choose a rule.
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
import run_sem110_protected_fusion_oof as sem110

from src.semantic_edge_recovery import (
    edge_recovery_gate,
    recover_edge_connected_stay_cable,
    select_edge_recovery_candidate,
)
from src.semantic_protected_fusion import (
    fuse_background_stay_cable,
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
    parser.add_argument("--sem101-oof-root", type=Path, required=True)
    parser.add_argument("--rendered-rgb-root", type=Path, required=True)
    parser.add_argument("--sem110-root", type=Path, required=True)
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


def semantic_image(path: Path, *, context: str) -> np.ndarray:
    value = np.asarray(Image.open(path), dtype=np.uint8)
    if value.shape != (989, 1320) or int(value.max()) >= 5:
        raise ValueError(f"{context} is not an official 1320x989 five-class PNG: {path}")
    return value


def rgb_image(path: Path, *, context: str) -> np.ndarray:
    value = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    if value.shape != (989, 1320, 3):
        raise ValueError(f"{context} is not an official 1320x989 RGB PNG: {path}")
    return value


def validate_config(config: dict[str, Any]) -> None:
    require(
        config.get("experiment_id") == "SEM-112-EDGE-RECOVERY-OOF",
        "SEM-112 received another config",
    )
    require(
        config.get("status") == "registered_not_run",
        "SEM-112 may run only from its registered config",
    )


def validate_rendered_rgb(
    root: Path, names: list[str], expected_manifest_sha256: str
) -> dict[str, Path]:
    manifest_path = root / "rendered_dataset_manifest.json"
    require(manifest_path.is_file(), "SEM-112 rendered-RGB manifest is missing")
    require(
        sha256(manifest_path) == expected_manifest_sha256,
        "SEM-112 rendered-RGB manifest hash differs from SEM-101",
    )
    manifest = common.load_json(manifest_path)
    require(
        manifest.get("experiment_id") == "SEM-101-RENDERED-SEG-OOF",
        "SEM-112 rendered RGB belongs to another experiment",
    )
    records = manifest.get("rendered_images", [])
    require(
        [record.get("name") for record in records] == names,
        "SEM-112 rendered RGB does not exactly match authorised training views",
    )
    result: dict[str, Path] = {}
    for record in records:
        path = root / str(record["path"])
        require(
            path.is_file() and sha256(path) == record.get("sha256"),
            f"SEM-112 rendered RGB hash mismatch: {record.get('name')}",
        )
        result[str(record["name"])] = path
    return result


def validate_sem110_baseline(
    *,
    root: Path,
    names: list[str],
    expected_result_sha256: str,
    base_paths: dict[str, Path],
    teacher_paths: dict[str, Path],
    threshold: float,
) -> dict[str, Path]:
    result_path = root / "result.json"
    require(result_path.is_file(), "SEM-112 SEM-110 result is missing")
    require(
        sha256(result_path) == expected_result_sha256,
        "SEM-112 SEM-110 result hash differs from the registered result",
    )
    result = common.load_json(result_path)
    require(
        result.get("experiment_id") == "SEM-110-PROTECTED-FUSION-OOF"
        and result.get("status") == "completed_oof_gate_passed",
        "SEM-112 requires the completed passing SEM-110 training result",
    )
    require(
        float(result.get("selected_threshold")) == threshold,
        "SEM-112 requires SEM-110's exact high-confidence threshold",
    )
    prediction_root = root / "selected_fused_predictions"
    paths: dict[str, Path] = {}
    for name in names:
        saved = prediction_root / name
        require(saved.is_file(), f"SEM-112 SEM-110 baseline is missing: {name}")
        base = semantic_image(base_paths[name], context="SEM-112 frozen SEM-001")
        expected, _fixed = fuse_background_stay_cable(
            base,
            sem110.teacher_probabilities(teacher_paths[name]),
            stay_cable_threshold=threshold,
        )
        require(
            np.array_equal(expected, semantic_image(saved, context="SEM-112 SEM-110 baseline")),
            f"SEM-112 SEM-110 baseline differs from the frozen reconstruction: {name}",
        )
        paths[name] = saved
    return paths


def evaluate(
    *,
    names: list[str],
    edge_quantile: float,
    labels_root: Path,
    base_paths: dict[str, Path],
    teacher_paths: dict[str, Path],
    rgb_paths: dict[str, Path],
    sem110_paths: dict[str, Path],
    fusion: dict[str, Any],
    write_predictions: Path | None = None,
) -> dict[str, Any]:
    raw_confusions: list[np.ndarray] = []
    baseline_confusions: list[np.ndarray] = []
    recovered_confusions: list[np.ndarray] = []
    protected_exact = True
    recovered_pixels = 0
    edge_thresholds: list[float] = []
    candidates_before_connectivity = 0
    seed_pixels = 0
    per_view: list[dict[str, Any]] = []
    for name in names:
        target = semantic_image(labels_root / name, context="SEM-112 authorised training label")
        base = semantic_image(base_paths[name], context="SEM-112 frozen SEM-001")
        teacher = sem110.teacher_probabilities(teacher_paths[name])
        rgb = rgb_image(rgb_paths[name], context="SEM-112 frozen rendered RGB")
        baseline = semantic_image(sem110_paths[name], context="SEM-112 frozen SEM-110")
        recovered, fixed, rescue, diagnostics = recover_edge_connected_stay_cable(
            base,
            teacher,
            rgb,
            high_stay_cable_threshold=float(fusion["high_stay_cable_threshold"]),
            low_stay_cable_threshold=float(fusion["low_stay_cable_threshold"]),
            edge_quantile=edge_quantile,
            adjacency_radius=int(fusion["adjacency_radius"]),
        )
        expected_baseline, _expected_fixed = fuse_background_stay_cable(
            base,
            teacher,
            stay_cable_threshold=float(fusion["high_stay_cable_threshold"]),
        )
        require(
            np.array_equal(expected_baseline, baseline),
            f"SEM-112 baseline drifted while evaluating {name}",
        )
        protected_exact = protected_exact and bool(
            np.array_equal(recovered[fixed], base[fixed])
        )
        raw_confusions.append(local_confusion(target, base))
        baseline_confusions.append(local_confusion(target, baseline))
        recovered_confusions.append(local_confusion(target, recovered))
        recovered_pixels += int(rescue.sum())
        edge_thresholds.append(float(diagnostics["edge_threshold"]))
        candidates_before_connectivity += int(diagnostics["candidate_pixels_before_connectivity"])
        seed_pixels += int(diagnostics["high_confidence_seed_pixels"])
        if write_predictions is not None:
            write_predictions.mkdir(parents=True, exist_ok=True)
            Image.fromarray(recovered).save(write_predictions / name)
        per_view.append(
            {
                "view": name,
                "recovered_pixels": int(rescue.sum()),
                "edge_threshold": float(diagnostics["edge_threshold"]),
            }
        )
    return {
        "raw_sem001_metrics": metrics_from_confusion(merge_confusions(raw_confusions)),
        "baseline_metrics": metrics_from_confusion(merge_confusions(baseline_confusions)),
        "recovered_metrics": metrics_from_confusion(merge_confusions(recovered_confusions)),
        "protected_predictions_exact": protected_exact,
        "recovered_pixels": recovered_pixels,
        "high_confidence_seed_pixels": seed_pixels,
        "candidate_pixels_before_connectivity": candidates_before_connectivity,
        "edge_threshold_summary": {
            "minimum": float(min(edge_thresholds)),
            "mean": float(np.mean(edge_thresholds)),
            "maximum": float(max(edge_thresholds)),
        },
        "per_view": per_view,
    }


def record_for(
    evaluation: dict[str, Any], gates: dict[str, Any], edge_quantile: float
) -> dict[str, Any]:
    gate = edge_recovery_gate(
        raw_sem001_metrics=evaluation["raw_sem001_metrics"],
        baseline_metrics=evaluation["baseline_metrics"],
        recovered_metrics=evaluation["recovered_metrics"],
        protected_predictions_exact=bool(evaluation["protected_predictions_exact"]),
        recovered_pixels=int(evaluation["recovered_pixels"]),
        gates=gates,
    )
    return {"edge_quantile": edge_quantile, **evaluation, "gate": gate}


def panel(target: np.ndarray, baseline: np.ndarray, recovered: np.ndarray) -> np.ndarray:
    error = np.zeros((*target.shape, 3), dtype=np.uint8)
    error[target != recovered] = np.asarray([255, 48, 48], dtype=np.uint8)
    return np.concatenate(
        [semantic_rgb(target), semantic_rgb(baseline), semantic_rgb(recovered), error],
        axis=1,
    )


def write_panels(
    *,
    names: list[str],
    labels_root: Path,
    baseline_paths: dict[str, Path],
    recovered_root: Path,
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = semantic_image(labels_root / name, context="SEM-112 training label")
        baseline = semantic_image(baseline_paths[name], context="SEM-112 baseline")
        recovered = semantic_image(recovered_root / name, context="SEM-112 recovered")
        Image.fromarray(panel(target, baseline, recovered)).save(output / name)


def main() -> int:
    args = parse_args()
    config = common.load_json(args.config)
    split = common.load_json(args.split)
    oof_split = common.load_json(args.oof_split)
    validate_config(config)
    require(args.labels_root.is_dir(), "SEM-112 authorised training labels are missing")
    existing = list(args.output_dir.iterdir()) if args.output_dir.exists() else []
    resumable = args.output_dir / "input_access_manifest.json"
    require(
        not existing or existing == [resumable],
        "SEM-112 refuses to overwrite an existing output",
    )
    names = select_rendered_training_views(
        split, expected_count=int(config["data"]["training_images"])
    )
    partition = partition_training_names(names, expected_count=len(names))
    base_paths = sem110.validate_base_manifest(
        args.sem001_training_root,
        names,
        config["inputs"]["sem001_checkpoint_sha256"],
    )
    teacher_paths = sem110.validate_oof_teacher(args.sem101_oof_root, oof_split, names)
    rgb_paths = validate_rendered_rgb(
        args.rendered_rgb_root,
        names,
        config["inputs"]["sem101_rendered_rgb_manifest_sha256"],
    )
    baseline_paths = validate_sem110_baseline(
        root=args.sem110_root,
        names=names,
        expected_result_sha256=config["inputs"]["sem110_result_sha256"],
        base_paths=base_paths,
        teacher_paths=teacher_paths,
        threshold=float(config["fusion"]["high_stay_cable_threshold"]),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    access = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "partition": partition,
        "training_views": len(names),
        "inputs_validated_before_labels": {
            "sem001_training_predictions": True,
            "sem101_oof_probabilities": True,
            "sem101_frozen_rendered_rgb": True,
            "sem110_high_confidence_baseline": True,
            "fixed_validation_views_opened": False,
            "unlabeled_images_opened": False,
            "real_rgb_input_opened": False,
            "external_camera_mode_present": False,
        },
    }
    if resumable.exists():
        require(common.load_json(resumable) == access, "SEM-112 resumable inputs differ")
    else:
        common.write_json(resumable, access)

    records: list[dict[str, Any]] = []
    for edge_quantile in config["fusion"]["edge_quantile_grid"]:
        record = record_for(
            evaluate(
                names=partition["calibration"],
                edge_quantile=float(edge_quantile),
                labels_root=args.labels_root,
                base_paths=base_paths,
                teacher_paths=teacher_paths,
                rgb_paths=rgb_paths,
                sem110_paths=baseline_paths,
                fusion=config["fusion"],
            ),
            config["training_oof_gates"],
            float(edge_quantile),
        )
        records.append(record)
        print(json.dumps({"phase": "calibration", "edge_quantile": edge_quantile, "passed": record["gate"]["passed"]}, sort_keys=True), flush=True)
    selected = select_edge_recovery_candidate(records)
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "calibration_records": records,
        "selection_rule": config["fusion"]["selection_rule"],
        "input_access_manifest_sha256": sha256(resumable),
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
                "selected_edge_quantile": None,
                "check_labels_opened": False,
                "next_step": "Stop: no training calibration edge rule qualified; fixed 74-view remains closed.",
            }
        )
        common.write_json(args.output_dir / "result.json", result)
        return 0

    edge_quantile = float(selected["edge_quantile"])
    check = record_for(
        evaluate(
            names=partition["check"],
            edge_quantile=edge_quantile,
            labels_root=args.labels_root,
            base_paths=base_paths,
            teacher_paths=teacher_paths,
            rgb_paths=rgb_paths,
            sem110_paths=baseline_paths,
            fusion=config["fusion"],
            write_predictions=args.output_dir / "selected_edge_recovery_predictions",
        ),
        config["training_oof_gates"],
        edge_quantile,
    )
    evaluate(
        names=partition["calibration"],
        edge_quantile=edge_quantile,
        labels_root=args.labels_root,
        base_paths=base_paths,
        teacher_paths=teacher_paths,
        rgb_paths=rgb_paths,
        sem110_paths=baseline_paths,
        fusion=config["fusion"],
        write_predictions=args.output_dir / "selected_edge_recovery_predictions",
    )
    write_panels(
        names=list(config["outputs"]["training_visual_samples"]),
        labels_root=args.labels_root,
        baseline_paths=baseline_paths,
        recovered_root=args.output_dir / "selected_edge_recovery_predictions",
        output=args.output_dir / "training_panels",
    )
    result.update(
        {
            "status": "completed_oof_gate_passed" if check["gate"]["passed"] else "completed_oof_gate_failed",
            "selected_edge_quantile": edge_quantile,
            "check_labels_opened": True,
            "check_record": check,
            "selected_predictions": "selected_edge_recovery_predictions",
            "panels": [
                str((args.output_dir / "training_panels" / name).resolve())
                for name in config["outputs"]["training_visual_samples"]
            ],
            "next_step": "A pass may justify one separately registered fixed-rule final audit; neither outcome allows SEM-112 retuning.",
        }
    )
    common.write_json(args.output_dir / "result.json", result)
    print(json.dumps({"status": result["status"], "selected_edge_quantile": edge_quantile, "check_passed": check["gate"]["passed"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
