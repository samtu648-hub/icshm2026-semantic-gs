#!/usr/bin/env python3
"""Fit one continuous two-stream fusion rule on training OOF evidence.

SEM-121 changes only the mutable background/stay_cable decision from SEM-110's
hard teacher threshold to a fixed-form logistic calibrator.  It first fits on
the existing training calibration half, checks on the existing training check
half, then (only if that check dominates the visual 0.30 hard-fusion baseline)
evaluates the already-authorised 74-view development set.
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
import run_sem120_dev_threshold_sweep as sem120

from src.semantic_continuous_fusion import (
    fit_logistic_calibrator,
    fuse_continuous_background_stay_cable,
    two_stream_features,
)
from src.semantic_development import development_score
from src.semantic2d import confusion_metrics
from src.semantic_protected_fusion import (
    fuse_background_stay_cable,
    local_confusion,
    merge_confusions,
    metrics_from_confusion,
    protected_mask,
)
from src.semantic_render_domain import semantic_rgb
from src.semantic_rendered_final import require_exact_view_set

CLASS_COUNT = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--oof-split", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--sem001-training-root", type=Path, required=True)
    parser.add_argument("--sem101-oof-root", type=Path, required=True)
    parser.add_argument("--sem101-oof-summary", type=Path, required=True)
    parser.add_argument("--sem110-result", type=Path, required=True)
    parser.add_argument("--sem001-result", type=Path, required=True)
    parser.add_argument("--sem001-validation-predictions", type=Path, required=True)
    parser.add_argument("--sem102-validation-manifest", type=Path, required=True)
    parser.add_argument("--sem102-validation-root", type=Path, required=True)
    parser.add_argument("--sem120-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Resume only the registered immutable SEM-121 output")
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


def class_map(path: Path, *, context: str) -> np.ndarray:
    value = np.asarray(Image.open(path), dtype=np.uint8)
    if value.shape != (989, 1320) or int(value.max()) >= CLASS_COUNT:
        raise ValueError(f"{context} must be an official 1320x989 five-class PNG")
    return value


def teacher_probabilities(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        require(archive.files == ["probabilities"], "SEM-121 teacher NPZ schema is invalid")
        value = archive["probabilities"]
    if value.shape != (CLASS_COUNT, 989, 1320):
        raise ValueError("SEM-121 teacher probability shape is invalid")
    return value


def validate_config(config: dict[str, Any]) -> None:
    require(config.get("experiment_id") == "SEM-121-CONTINUOUS-FUSION-OOF", "SEM-121 received another config")
    require(config.get("status") == "registered_not_run", "SEM-121 may run only from its registered config")
    rule = config["continuous_fusion"]
    require(rule["output_cable_probability_threshold"] == 0.5, "SEM-121 output threshold is not fixed")
    require(rule["sample_per_class_per_calibration_view"] >= 1, "SEM-121 sample count is invalid")


def validation_teacher_paths(
    *, manifest_path: Path, root: Path, names: list[str], expected_sha256: str
) -> dict[str, Path]:
    require(sha256(manifest_path) == expected_sha256, "SEM-121 SEM-102 validation manifest hash differs")
    manifest = common.load_json(manifest_path)
    require([record.get("name") for record in manifest] == names, "SEM-121 teacher does not exactly cover development views")
    paths: dict[str, Path] = {}
    for record in manifest:
        path = root / str(record["probabilities"])
        require(path.is_file() and sha256(path) == record.get("probabilities_sha256"), f"SEM-121 teacher hash mismatch: {record.get('name')}")
        paths[str(record["name"])] = path
    return paths


def sample_calibration_features(
    *,
    names: list[str],
    labels_root: Path,
    base_paths: dict[str, Path],
    teacher_paths: dict[str, Path],
    sample_per_class_per_view: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Take equal positive/negative samples per view without a threshold grid."""
    rng = np.random.default_rng(seed)
    feature_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    per_view: list[dict[str, Any]] = []
    for name in names:
        target = class_map(labels_root / name, context="SEM-121 calibration label")
        base = class_map(base_paths[name], context="SEM-121 base prediction")
        teacher = teacher_probabilities(teacher_paths[name])
        mutable = ~protected_mask(base)
        positive = np.flatnonzero(mutable & (target == 2))
        negative = np.flatnonzero(mutable & (target != 2))
        positive_count = min(int(sample_per_class_per_view), len(positive))
        negative_count = min(int(sample_per_class_per_view), len(negative))
        if not positive_count or not negative_count:
            continue
        selected_positive = rng.choice(positive, size=positive_count, replace=False)
        selected_negative = rng.choice(negative, size=negative_count, replace=False)
        flat_features = two_stream_features(base, teacher).reshape(-1, 2)
        feature_parts.extend((flat_features[selected_positive], flat_features[selected_negative]))
        target_parts.extend((np.ones(positive_count, dtype=np.uint8), np.zeros(negative_count, dtype=np.uint8)))
        per_view.append({"view": name, "positive_samples": positive_count, "negative_samples": negative_count})
    require(feature_parts, "SEM-121 calibration sampling produced no usable examples")
    features = np.concatenate(feature_parts, axis=0).astype(np.float32, copy=False)
    targets = np.concatenate(target_parts, axis=0).astype(np.uint8, copy=False)
    return features, targets, {
        "views_with_both_classes": len(per_view),
        "positive_samples": int(np.count_nonzero(targets)),
        "negative_samples": int(len(targets) - np.count_nonzero(targets)),
        "per_view": per_view,
    }


def evaluate_continuous(
    *,
    names: list[str],
    labels_root: Path,
    base_paths: dict[str, Path],
    teacher_paths: dict[str, Path],
    calibrator: dict[str, Any],
    write_predictions: Path | None = None,
) -> dict[str, Any]:
    confusions: list[np.ndarray] = []
    per_view: list[dict[str, Any]] = []
    protected_exact = True
    for name in names:
        target = class_map(labels_root / name, context="SEM-121 target")
        base = class_map(base_paths[name], context="SEM-121 base prediction")
        fused, fixed, _probability = fuse_continuous_background_stay_cable(base, teacher_probabilities(teacher_paths[name]), calibrator)
        protected_exact = protected_exact and bool(np.array_equal(base[fixed], fused[fixed]))
        confusion = local_confusion(target, fused)
        confusions.append(confusion)
        per_view.append({"view": name, **metrics_from_confusion(confusion)})
        if write_predictions is not None:
            write_predictions.mkdir(parents=True, exist_ok=True)
            Image.fromarray(fused, mode="L").save(write_predictions / name)
    metrics = metrics_from_confusion(merge_confusions(confusions))
    worst_count = max(1, int(np.ceil(0.10 * len(per_view))))
    worst = sorted(per_view, key=lambda row: row["mIoU_without_background"])[:worst_count]
    challenge = [row for row in per_view if 140 <= int(Path(row["view"]).stem) <= 146]
    metrics.update(
        {
            "view_count": len(names),
            "worst_10_percent_mIoU_without_background": float(np.mean([row["mIoU_without_background"] for row in worst])),
            "worst_10_percent_views": [row["view"] for row in worst],
            "challenge_140_146_mIoU_without_background": float(np.mean([row["mIoU_without_background"] for row in challenge])) if challenge else None,
            "per_view": per_view,
        }
    )
    return {"metrics": metrics, "protected_prediction_pixels_exact": protected_exact, "developer_score": development_score(metrics)}


def evaluate_hard_threshold(
    *,
    names: list[str],
    labels_root: Path,
    base_paths: dict[str, Path],
    teacher_paths: dict[str, Path],
    threshold: float,
) -> dict[str, Any]:
    confusions: list[np.ndarray] = []
    for name in names:
        target = class_map(labels_root / name, context="SEM-121 hard baseline target")
        base = class_map(base_paths[name], context="SEM-121 hard baseline base")
        fused, fixed = fuse_background_stay_cable(base, teacher_probabilities(teacher_paths[name]), stay_cable_threshold=threshold)
        require(np.array_equal(base[fixed], fused[fixed]), "SEM-121 hard baseline changed protected pixels")
        confusions.append(local_confusion(target, fused))
    metrics = metrics_from_confusion(merge_confusions(confusions))
    return {"stay_cable_threshold": threshold, "metrics": metrics, "developer_score": development_score(metrics)}


def dominates_visual_baseline(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    candidate_metrics = candidate["metrics"]
    baseline_metrics = baseline["metrics"]
    candidate_cable = candidate_metrics["per_class"]["stay_cable"]
    baseline_cable = baseline_metrics["per_class"]["stay_cable"]
    checks = {
        "mIoU_with_background": candidate_metrics["mIoU_with_background"] >= baseline_metrics["mIoU_with_background"],
        "stay_cable_iou": candidate_cable["iou"] >= baseline_cable["iou"],
        "stay_cable_recall": candidate_cable["recall"] >= baseline_cable["recall"],
        "background_to_stay_cable_pixels": candidate_metrics["background_to_stay_cable_pixels"] <= baseline_metrics["background_to_stay_cable_pixels"],
        "protected_prediction_pixels_exact": candidate["protected_prediction_pixels_exact"] is True,
    }
    return {"checks": checks, "passed": all(checks.values())}


def panel(target: np.ndarray, hard030: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    error = np.zeros((*target.shape, 3), dtype=np.uint8)
    error[target != candidate] = np.asarray([255, 48, 48], dtype=np.uint8)
    return np.concatenate((semantic_rgb(target), semantic_rgb(hard030), semantic_rgb(candidate), error), axis=1)


def write_development_panels(
    *, names: list[str], labels_root: Path, base_paths: dict[str, Path], teacher_paths: dict[str, Path], calibrator: dict[str, Any], output: Path
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = class_map(labels_root / name, context="SEM-121 panel target")
        base = class_map(base_paths[name], context="SEM-121 panel base")
        teacher = teacher_probabilities(teacher_paths[name])
        hard030, _fixed = fuse_background_stay_cable(base, teacher, stay_cable_threshold=0.3)
        candidate, _fixed, _probability = fuse_continuous_background_stay_cable(base, teacher, calibrator)
        Image.fromarray(panel(target, hard030, candidate)).save(output / name)


def _load_confusion(path: Path, key: str) -> np.ndarray:
    with np.load(path) as archive:
        require(key in archive.files, f"SEM-121 partial evaluation lacks {key}: {path.name}")
        value = archive[key]
    require(value.shape == (CLASS_COUNT, CLASS_COUNT), f"SEM-121 partial confusion shape is invalid: {path.name}")
    return value.astype(np.int64, copy=False)


def _load_flag(path: Path, key: str) -> bool:
    with np.load(path) as archive:
        require(key in archive.files, f"SEM-121 partial evaluation lacks {key}: {path.name}")
        value = archive[key]
    require(value.shape == (1, 1) and int(value[0, 0]) in (0, 1), f"SEM-121 partial flag is invalid: {path.name}")
    return bool(value[0, 0])


def _aggregate_records(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    confusions = [record[key] for record in records]
    metrics = metrics_from_confusion(merge_confusions(confusions))
    per_view = [{"view": record["view"], **confusion_metrics(record[key])} for record in records]
    worst_count = max(1, int(np.ceil(0.10 * len(per_view))))
    worst = sorted(per_view, key=lambda row: row["mIoU_without_background"])[:worst_count]
    challenge = [row for row in per_view if 140 <= int(Path(row["view"]).stem) <= 146]
    metrics.update(
        {
            "view_count": len(per_view),
            "worst_10_percent_mIoU_without_background": float(np.mean([row["mIoU_without_background"] for row in worst])),
            "worst_10_percent_views": [row["view"] for row in worst],
            "challenge_140_146_mIoU_without_background": float(np.mean([row["mIoU_without_background"] for row in challenge])) if challenge else None,
            "per_view": per_view,
        }
    )
    return metrics


def evaluate_triplet_resumable(
    *,
    names: list[str],
    labels_root: Path,
    base_paths: dict[str, Path],
    teacher_paths: dict[str, Path],
    calibrator: dict[str, Any],
    output: Path,
    write_predictions: Path | None = None,
) -> dict[str, Any]:
    """Persist per-view confusions so a short execution window can safely resume."""
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, name in enumerate(names, start=1):
        record_path = output / f"{Path(name).stem}.npz"
        prediction_path = write_predictions / name if write_predictions is not None else None
        if record_path.is_file() and (prediction_path is None or prediction_path.is_file()):
            record = {
                "view": name,
                "hard020": _load_confusion(record_path, "hard020"),
                "hard030": _load_confusion(record_path, "hard030"),
                "candidate": _load_confusion(record_path, "candidate"),
                "protected_exact": _load_flag(record_path, "protected_exact"),
            }
            records.append(record)
            continue
        target = class_map(labels_root / name, context="SEM-121 resumable target")
        base = class_map(base_paths[name], context="SEM-121 resumable base")
        teacher = teacher_probabilities(teacher_paths[name])
        hard020, fixed020 = fuse_background_stay_cable(base, teacher, stay_cable_threshold=0.2)
        hard030, fixed030 = fuse_background_stay_cable(base, teacher, stay_cable_threshold=0.3)
        candidate, fixed_candidate, _probability = fuse_continuous_background_stay_cable(base, teacher, calibrator)
        protected_exact = bool(
            np.array_equal(base[fixed020], hard020[fixed020])
            and np.array_equal(base[fixed030], hard030[fixed030])
            and np.array_equal(base[fixed_candidate], candidate[fixed_candidate])
        )
        record = {
            "view": name,
            "hard020": local_confusion(target, hard020),
            "hard030": local_confusion(target, hard030),
            "candidate": local_confusion(target, candidate),
            "protected_exact": protected_exact,
        }
        np.savez_compressed(
            record_path,
            hard020=record["hard020"],
            hard030=record["hard030"],
            candidate=record["candidate"],
            protected_exact=np.asarray([[int(protected_exact)]], dtype=np.uint8),
        )
        if prediction_path is not None:
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(candidate, mode="L").save(prediction_path)
        records.append(record)
        if index == 1 or index % 12 == 0 or index == len(names):
            print(json.dumps({"phase": output.name, "completed_views": index, "total_views": len(names), "view": name}), flush=True)
    hard020_metrics = _aggregate_records(records, "hard020")
    hard030_metrics = _aggregate_records(records, "hard030")
    candidate_metrics = _aggregate_records(records, "candidate")
    protected_exact = all(record["protected_exact"] for record in records)
    return {
        "hard020": {"stay_cable_threshold": 0.2, "metrics": hard020_metrics, "developer_score": development_score(hard020_metrics)},
        "hard030": {"stay_cable_threshold": 0.3, "metrics": hard030_metrics, "developer_score": development_score(hard030_metrics)},
        "candidate": {"metrics": candidate_metrics, "protected_prediction_pixels_exact": protected_exact, "developer_score": development_score(candidate_metrics)},
    }


def main() -> int:
    args = parse_args()
    config = common.load_json(args.config)
    split = common.load_json(args.split)
    oof_split = common.load_json(args.oof_split)
    validate_config(config)
    require(args.labels_root.is_dir(), "SEM-121 labels root is missing")

    training_names = sem110.select_rendered_training_views(split, expected_count=int(config["data"]["training_images"]))
    partition = sem110.partition_training_names(training_names, expected_count=len(training_names))
    base_training = sem110.validate_base_manifest(args.sem001_training_root, training_names, config["inputs"]["sem001_checkpoint_sha256"])
    teacher_training = sem110.validate_oof_teacher(args.sem101_oof_root, oof_split, training_names)
    require(sha256(args.sem101_oof_summary) == config["inputs"]["sem101_oof_summary_sha256"], "SEM-121 SEM-101 OOF summary hash differs")
    require(common.load_json(args.sem101_oof_summary).get("status") == "completed_oof_gate_passed", "SEM-121 requires SEM-101 OOF gate evidence")
    require(sha256(args.sem110_result) == config["inputs"]["sem110_oof_result_sha256"], "SEM-121 SEM-110 result hash differs")
    require(common.load_json(args.sem110_result).get("status") == "completed_oof_gate_passed", "SEM-121 requires SEM-110 OOF gate evidence")
    require(sha256(args.sem120_result) == config["inputs"]["sem120_result_sha256"], "SEM-121 SEM-120 result hash differs")
    sem001_result = common.load_json(args.sem001_result)
    require(sha256(args.sem001_result) == config["inputs"]["sem001_result_sha256"], "SEM-121 SEM-001 result hash differs")
    require(sem001_result.get("checkpoint_reloaded") is True, "SEM-121 requires a reloaded SEM-001 checkpoint")
    input_manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "training_partition": partition,
        "inputs_validated_before_labels": {
            "sem001_training_predictions": True,
            "sem101_oof_probabilities": True,
            "fixed_validation_views_opened": False,
            "unlabeled_images_opened": False,
            "real_rgb_input_opened": False,
            "external_camera_mode_present": False,
        },
    }
    input_manifest_path = args.output_dir / "input_access_manifest.json"
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        require(args.resume, "SEM-121 refuses to overwrite non-empty output without --resume")
        require(input_manifest_path.is_file(), "SEM-121 resume is missing its input manifest")
        require(common.load_json(input_manifest_path) == input_manifest, "SEM-121 resume inputs differ from the registered inputs")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        common.write_json(input_manifest_path, input_manifest)

    rule = config["continuous_fusion"]
    calibrator_path = args.output_dir / "calibrator.json"
    if calibrator_path.is_file():
        require(args.resume, "SEM-121 found an existing calibrator without --resume")
        calibrator = common.load_json(calibrator_path)
        require(calibrator.get("output_cable_probability_threshold") == 0.5, "SEM-121 resumed calibrator differs")
    else:
        features, targets, sample_summary = sample_calibration_features(
            names=partition["calibration"], labels_root=args.labels_root, base_paths=base_training,
            teacher_paths=teacher_training, sample_per_class_per_view=int(rule["sample_per_class_per_calibration_view"]), seed=int(config["seed"])
        )
        calibrator = fit_logistic_calibrator(features, targets, l2=float(rule["l2"]), maximum_iterations=int(rule["maximum_newton_iterations"]))
        calibrator.update({"feature_order": ["teacher_cable_vs_background_log_odds", "sem001_is_stay_cable_prior"], "output_cable_probability_threshold": 0.5, "sample_summary": sample_summary})
        common.write_json(calibrator_path, calibrator)

    check = evaluate_triplet_resumable(
        names=partition["check"], labels_root=args.labels_root, base_paths=base_training,
        teacher_paths=teacher_training, calibrator=calibrator, output=args.output_dir / "check_evaluations"
    )
    check_hard030 = check["hard030"]
    check_hard020 = check["hard020"]
    check_candidate = check["candidate"]
    check_gate = dominates_visual_baseline(check_candidate, check_hard030)
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "development_data_notice": "The 74 views are an explicitly reused development set and are not blind-test evidence.",
        "calibrator": calibrator,
        "training_partition": partition,
        "check_hard_threshold_0_30": check_hard030,
        "check_hard_threshold_0_20": check_hard020,
        "check_continuous_candidate": check_candidate,
        "check_visual_baseline_dominance_gate": check_gate,
        "fixed_validation_views_opened": False,
        "real_rgb_input_opened": False,
        "unlabeled_images_opened": False,
        "external_camera_mode_present": False,
        "rgb_modified": False,
    }
    if not check_gate["passed"]:
        result["status"] = "completed_oof_dominance_gate_failed"
        common.write_json(args.output_dir / "result.json", result)
        print(json.dumps({"status": result["status"], "check_gate": check_gate}, sort_keys=True))
        return 0

    development_names = require_exact_view_set(split["semantic_validation_labeled"], split["semantic_validation_labeled"], context="SEM-121 development plan")
    require(len(development_names) == int(config["data"]["development_images"]), "SEM-121 development view count differs")
    base_development = {name: args.sem001_validation_predictions / name for name in development_names}
    for name, path in base_development.items():
        require(path.is_file(), f"SEM-121 missing SEM-001 development prediction: {name}")
    teacher_development = validation_teacher_paths(manifest_path=args.sem102_validation_manifest, root=args.sem102_validation_root, names=development_names, expected_sha256=config["inputs"]["sem102_validation_prediction_manifest_sha256"])
    result["fixed_validation_views_opened"] = True
    development = evaluate_triplet_resumable(
        names=development_names, labels_root=args.labels_root, base_paths=base_development,
        teacher_paths=teacher_development, calibrator=calibrator,
        output=args.output_dir / "development_evaluations", write_predictions=args.output_dir / "development_predictions"
    )
    development_candidate = development["candidate"]
    development_hard030 = development["hard030"]
    write_development_panels(names=[str(name) for name in config["outputs"]["development_visual_samples"]], labels_root=args.labels_root, base_paths=base_development, teacher_paths=teacher_development, calibrator=calibrator, output=args.output_dir / "development_panels")
    result.update({"status": "completed_development_evaluated", "development_hard_threshold_0_30": development_hard030, "development_continuous_candidate": development_candidate})
    common.write_json(args.output_dir / "result.json", result)
    print(json.dumps({"status": result["status"], "check_gate": check_gate, "development_score": development_candidate["developer_score"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
