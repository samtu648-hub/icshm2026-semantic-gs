#!/usr/bin/env python3
"""Developer sweep for selective recovery from the fixed 0.20--0.30 band."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_sem080_structural_diag as common

from src.semantic_band_recovery import recover_selective_band
from src.semantic_development import development_score
from src.semantic_protected_fusion import (
    CLASS_COUNT,
    fuse_background_stay_cable,
    local_confusion,
    merge_confusions,
    metrics_from_confusion,
)
from src.semantic_render_domain import semantic_rgb
from src.semantic_rendered_final import require_exact_view_set
from src.semantic_uncertainty_band import high_confidence_seed, uncertainty_band


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--sem001-result", type=Path, required=True)
    parser.add_argument("--sem001-predictions", type=Path, required=True)
    parser.add_argument("--sem102-manifest", type=Path, required=True)
    parser.add_argument("--sem102-audit-root", type=Path, required=True)
    parser.add_argument("--sem120-result", type=Path, required=True)
    parser.add_argument("--sem122-result", type=Path, required=True)
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


def class_map(path: Path, *, context: str) -> np.ndarray:
    value = np.asarray(Image.open(path), dtype=np.uint8)
    require(value.shape == (989, 1320) and int(value.max()) < CLASS_COUNT, f"{context} must be an official five-class PNG")
    return value


def probabilities(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        require(archive.files == ["probabilities"], "SEM-123 teacher NPZ schema is invalid")
        value = archive["probabilities"]
    require(value.shape == (CLASS_COUNT, 989, 1320), "SEM-123 teacher probability shape is invalid")
    return value


def specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    grid = config["recovery_grid"]
    result: list[dict[str, Any]] = [{"id": "hard_030_baseline", "baseline": True}]
    for probability in grid["probability_minima"]:
        for distance in grid["maximum_seed_distances_px"]:
            for prior in grid["sem001_prior_modes"]:
                result.append({
                    "id": f"p{int(round(100 * float(probability))):02d}_d{int(distance):02d}_{prior}",
                    "probability_minimum": float(probability),
                    "maximum_seed_distance": float(distance),
                    "sem001_prior_mode": str(prior),
                    "baseline": False,
                })
    require(len({record["id"] for record in result}) == len(result), "SEM-123 candidate IDs are not unique")
    return result


def validate_config(config: dict[str, Any]) -> None:
    require(config.get("experiment_id") == "SEM-123-BAND-RECOVERY-DEV-SWEEP", "SEM-123 received another config")
    require(config.get("status") == "registered_not_run", "SEM-123 may run only from its registered config")
    candidates = specs(config)
    require(len(candidates) == 19, "SEM-123 requires exactly 18 registered recovery predicates plus baseline")
    require(config["recovery_grid"]["lower_threshold"] == 0.2 and config["recovery_grid"]["upper_threshold"] == 0.3, "SEM-123 band thresholds are fixed")


def validate_sources(*, args: argparse.Namespace, config: dict[str, Any], names: list[str]) -> dict[str, Path]:
    require(sha256(args.sem001_result) == config["inputs"]["sem001_result_sha256"], "SEM-123 SEM-001 result hash differs")
    require(common.load_json(args.sem001_result).get("checkpoint_reloaded") is True, "SEM-123 requires reloaded SEM-001")
    require(sha256(args.sem120_result) == config["inputs"]["sem120_result_sha256"], "SEM-123 SEM-120 result hash differs")
    sem120 = common.load_json(args.sem120_result)
    require(sem120.get("status") == "completed_development_sweep", "SEM-123 requires SEM-120")
    require(sha256(args.sem122_result) == config["inputs"]["sem122_result_sha256"], "SEM-123 SEM-122 result hash differs")
    sem122 = common.load_json(args.sem122_result)
    require(sem122.get("status") == "completed_development_read_only_diagnostic", "SEM-123 requires completed SEM-122")
    require(sem122.get("band_equals_hard020_minus_hard030_exact") is True, "SEM-123 requires the exact SEM-122 difference-band proof")
    require(sha256(args.sem102_manifest) == config["inputs"]["sem102_prediction_manifest_sha256"], "SEM-123 SEM-102 manifest hash differs")
    manifest = common.load_json(args.sem102_manifest)
    require([record.get("name") for record in manifest] == names, "SEM-123 teacher does not cover the development set exactly")
    paths: dict[str, Path] = {}
    for record in manifest:
        path = args.sem102_audit_root / str(record["probabilities"])
        require(path.is_file() and sha256(path) == record.get("probabilities_sha256"), f"SEM-123 teacher hash mismatch: {record.get('name')}")
        paths[str(record["name"])] = path
    return paths


def select(records: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        records,
        key=lambda record: (
            float(record["developer_score"]),
            float(record["metrics"]["per_class"]["stay_cable"]["iou"]),
            -int(record["metrics"]["background_to_stay_cable_pixels"]),
            -int(record["recovered_pixels"]),
        ),
    )


def write_leaderboard(path: Path, records: list[dict[str, Any]]) -> None:
    fields = ["id", "probability_minimum", "maximum_seed_distance", "sem001_prior_mode", "developer_score", "mIoU_with_background", "stay_cable_iou", "stay_cable_recall", "background_to_stay_cable_pixels", "recovered_pixels", "recovered_gt_stay_cable_pixels", "recovered_gt_background_pixels", "protected_prediction_pixels_exact"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            cable = record["metrics"]["per_class"]["stay_cable"]
            writer.writerow({
                "id": record["id"],
                "probability_minimum": record.get("probability_minimum"),
                "maximum_seed_distance": record.get("maximum_seed_distance"),
                "sem001_prior_mode": record.get("sem001_prior_mode"),
                "developer_score": record["developer_score"],
                "mIoU_with_background": record["metrics"]["mIoU_with_background"],
                "stay_cable_iou": cable["iou"],
                "stay_cable_recall": cable["recall"],
                "background_to_stay_cable_pixels": record["metrics"]["background_to_stay_cable_pixels"],
                "recovered_pixels": record["recovered_pixels"],
                "recovered_gt_stay_cable_pixels": record["recovered_target_counts"][2],
                "recovered_gt_background_pixels": record["recovered_target_counts"][0],
                "protected_prediction_pixels_exact": record["protected_prediction_pixels_exact"],
            })


def error_panel(target: np.ndarray, baseline: np.ndarray, selected: np.ndarray) -> np.ndarray:
    changed = selected == 2
    added = changed & (baseline != 2)
    image = np.zeros((*target.shape, 3), dtype=np.uint8)
    image[added & (target == 2)] = np.asarray([56, 220, 80], dtype=np.uint8)
    image[added & (target == 0)] = np.asarray([240, 56, 56], dtype=np.uint8)
    image[added & (target != 2) & (target != 0)] = np.asarray([255, 172, 35], dtype=np.uint8)
    return image


def recovery_target_histogram(
    *,
    base: np.ndarray,
    teacher: np.ndarray,
    target: np.ndarray,
    grid: dict[str, Any],
) -> np.ndarray:
    """Count target classes for every predicate without re-rasterising maps.

    Every selected pixel starts as hard-0.30 background and changes only to
    stay_cable. A [probability-bin, distance-bin, SEM-001-prior, target-class]
    histogram is therefore exactly sufficient to transform the hard-0.30
    confusion matrix for all 18 predicates.
    """
    lower = float(grid["lower_threshold"])
    upper = float(grid["upper_threshold"])
    probability_minima = [float(value) for value in grid["probability_minima"]]
    distances = [float(value) for value in grid["maximum_seed_distances_px"]]
    band, _mutable = uncertainty_band(base, teacher, lower=lower, upper=upper)
    seed = high_confidence_seed(base, teacher, threshold=upper)
    distance = ndimage.distance_transform_edt(~seed) if np.any(seed) else np.full(base.shape, np.inf, dtype=np.float32)
    indices = np.flatnonzero(band)
    shape = (len(probability_minima), len(distances) + 1, 2, CLASS_COUNT)
    if not indices.size:
        return np.zeros(shape, dtype=np.int64)
    # Match NumPy's float32 comparison in ``teacher[2] >= probability_minimum``
    # exactly; a float64 bin edge can otherwise classify the represented 0.24
    # just below a Python literal 0.24.
    probability_values = teacher[2].ravel()[indices]
    probability_bins = np.digitize(
        probability_values,
        np.asarray(probability_minima[1:], dtype=probability_values.dtype),
        right=False,
    )
    distance_bins = np.digitize(distance.ravel()[indices], distances, right=True)
    priors = (base.ravel()[indices] == 2).astype(np.int64)
    classes = target.ravel()[indices].astype(np.int64)
    flat = (((probability_bins * shape[1] + distance_bins) * 2 + priors) * CLASS_COUNT + classes)
    return np.bincount(flat, minlength=int(np.prod(shape))).reshape(shape).astype(np.int64, copy=False)


def selected_target_counts(histogram: np.ndarray, spec: dict[str, Any], grid: dict[str, Any]) -> np.ndarray:
    """Read exact target counts for one registered predicate from the histogram."""
    if spec["baseline"]:
        return np.zeros(CLASS_COUNT, dtype=np.int64)
    probability_minima = [float(value) for value in grid["probability_minima"]]
    distances = [float(value) for value in grid["maximum_seed_distances_px"]]
    probability_index = probability_minima.index(float(spec["probability_minimum"]))
    distance_index = distances.index(float(spec["maximum_seed_distance"]))
    prior_slice = slice(None) if spec["sem001_prior_mode"] == "any" else slice(1, 2)
    return histogram[probability_index:, : distance_index + 1, prior_slice, :].sum(axis=(0, 1, 2), dtype=np.int64)


def main() -> int:
    args = parse_args()
    config = common.load_json(args.config)
    split = common.load_json(args.split)
    validate_config(config)
    require(args.labels_root.is_dir(), "SEM-123 developer labels are missing")
    require(not args.output_dir.exists() or not any(args.output_dir.iterdir()), "SEM-123 refuses to overwrite developer output")
    names = require_exact_view_set(split["semantic_validation_labeled"], split["semantic_validation_labeled"], context="SEM-123 developer view plan")
    require(len(names) == int(config["data"]["development_images"]), "SEM-123 expects 74 developer views")
    teacher_paths = validate_sources(args=args, config=config, names=names)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common.write_json(args.output_dir / "input_access_manifest.json", {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "inputs_verified_before_labels": {"sem001_result_and_reloaded_checkpoint": True, "sem102_development_probabilities": True, "sem120_source_record": True, "sem122_exact_band_diagnostic": True, "fixed_validation_views_opened": False, "real_rgb_input_opened": False, "unlabeled_images_opened": False, "hidden_test_input_opened": False, "external_camera_mode_present": False},
    })
    candidate_specs = specs(config)
    states = {record["id"]: {"spec": record, "confusions": [], "per_view": [], "recovered_pixels": 0, "recovered_target_counts": np.zeros(CLASS_COUNT, dtype=np.int64), "protected_prediction_pixels_exact": True} for record in candidate_specs}
    grid = config["recovery_grid"]
    for name in names:
        base = class_map(args.sem001_predictions / name, context="SEM-123 SEM-001 prediction")
        teacher = probabilities(teacher_paths[name])
        hard030, hard_fixed = fuse_background_stay_cable(base, teacher, stay_cable_threshold=float(grid["upper_threshold"]))
        target = class_map(args.labels_root / name, context="SEM-123 developer label")
        baseline_confusion = local_confusion(target, hard030)
        histogram = recovery_target_histogram(base=base, teacher=teacher, target=target, grid=grid)
        for spec in candidate_specs:
            state = states[spec["id"]]
            selected_counts = selected_target_counts(histogram, spec, grid)
            confusion = baseline_confusion.copy()
            confusion[:, 0] -= selected_counts
            confusion[:, 2] += selected_counts
            require(np.all(confusion >= 0), "SEM-123 recovery histogram is inconsistent with hard-0.30")
            state["protected_prediction_pixels_exact"] = state["protected_prediction_pixels_exact"] and bool(np.array_equal(hard030[hard_fixed], base[hard_fixed]))
            state["confusions"].append(confusion)
            state["per_view"].append({"view": name, **metrics_from_confusion(state["confusions"][-1])})
            state["recovered_pixels"] += int(selected_counts.sum())
            state["recovered_target_counts"] += selected_counts
    records: list[dict[str, Any]] = []
    for spec in candidate_specs:
        state = states[spec["id"]]
        metrics = metrics_from_confusion(merge_confusions(state["confusions"]))
        records.append({
            **spec,
            "metrics": metrics,
            "developer_score": development_score(metrics),
            "recovered_pixels": state["recovered_pixels"],
            "recovered_target_counts": state["recovered_target_counts"].tolist(),
            "protected_prediction_pixels_exact": state["protected_prediction_pixels_exact"],
        })
    selected_record = select(records)
    selected_predictions = args.output_dir / "selected_predictions"
    selected_predictions.mkdir(parents=True, exist_ok=True)
    panel_root = args.output_dir / "development_panels"
    for name in names:
        base = class_map(args.sem001_predictions / name, context="SEM-123 materialised SEM-001 prediction")
        teacher = probabilities(teacher_paths[name])
        baseline, _fixed = fuse_background_stay_cable(base, teacher, stay_cable_threshold=float(grid["upper_threshold"]))
        if selected_record["baseline"]:
            candidate = baseline
        else:
            candidate, _selected, _fixed = recover_selective_band(
                base, teacher,
                probability_minimum=selected_record["probability_minimum"],
                maximum_seed_distance=selected_record["maximum_seed_distance"],
                prior_mode=selected_record["sem001_prior_mode"],
                lower_threshold=float(grid["lower_threshold"]), upper_threshold=float(grid["upper_threshold"]),
            )
        Image.fromarray(candidate).save(selected_predictions / name)
        if name in config["outputs"]["visual_samples"]:
            target = class_map(args.labels_root / name, context="SEM-123 panel label")
            panel_root.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.concatenate((semantic_rgb(target), semantic_rgb(baseline), semantic_rgb(candidate), error_panel(target, baseline, candidate)), axis=1)).save(panel_root / name)
    selected_confusions = states[selected_record["id"]]["confusions"]
    materialised_confusions = []
    for name in names:
        target = class_map(args.labels_root / name, context="SEM-123 materialised label")
        candidate = class_map(selected_predictions / name, context="SEM-123 materialised prediction")
        materialised_confusions.append(local_confusion(target, candidate))
    require(np.array_equal(merge_confusions(materialised_confusions), merge_confusions(selected_confusions)), "SEM-123 materialised selected maps drifted")
    write_leaderboard(args.output_dir / "leaderboard.csv", records)
    baseline_record = next(record for record in records if record["baseline"])
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "completed_development_sweep",
        "development_data_notice": "All 74 labels are an explicitly reusable development set. The selected recovery predicate is not blind, formal, or a replacement claim.",
        "candidate_records": records,
        "selected_record": selected_record,
        "baseline_record": baseline_record,
        "selected_minus_baseline": {
            "mIoU_with_background": selected_record["metrics"]["mIoU_with_background"] - baseline_record["metrics"]["mIoU_with_background"],
            "stay_cable_iou": selected_record["metrics"]["per_class"]["stay_cable"]["iou"] - baseline_record["metrics"]["per_class"]["stay_cable"]["iou"],
            "stay_cable_recall": selected_record["metrics"]["per_class"]["stay_cable"]["recall"] - baseline_record["metrics"]["per_class"]["stay_cable"]["recall"],
            "background_to_stay_cable_pixels": selected_record["metrics"]["background_to_stay_cable_pixels"] - baseline_record["metrics"]["background_to_stay_cable_pixels"],
        },
        "fixed_validation_views_opened": True,
        "real_rgb_input_opened": False,
        "unlabeled_images_opened": False,
        "hidden_test_input_opened": False,
        "external_camera_mode_present": False,
        "rgb_modified": False,
        "outputs": {"leaderboard_csv": "leaderboard.csv", "selected_predictions": "selected_predictions", "panel_order": ["GT", "hard_0.30", "selected", "selected_additions: green=true cable/red=background/orange=other"], "development_panels": [str((panel_root / name).resolve()) for name in config["outputs"]["visual_samples"]]},
    }
    common.write_json(args.output_dir / "result.json", result)
    print(json.dumps({"status": result["status"], "selected_id": selected_record["id"], "selected_developer_score": selected_record["developer_score"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
