#!/usr/bin/env python3
"""Qualify spatially constrained additive deck recovery on SEM-160 OOF."""

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

from scripts.run_sem170_deck_arbitration_oof import (
    class_map,
    deck_probability,
    oof_probability_paths,
    pre_foundation_map,
    prediction_paths,
    require,
    sem001_paths,
)
from src.semantic2d import confusion_metrics, read_json, sha256, write_json
from src.semantic_deck_continuity import (
    closed_deck_mask,
    continuity_gate,
    recover_deck_continuity,
    select_continuity_candidate,
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


def adjusted_confusion(
    baseline: np.ndarray, additions_by_gt: np.ndarray
) -> np.ndarray:
    candidate = baseline.copy()
    candidate[:, 0] -= additions_by_gt
    candidate[:, 1] += additions_by_gt
    if np.any(candidate < 0):
        raise RuntimeError("SEM-171 derived a negative confusion count")
    return candidate


def gate_for(
    evaluation: dict[str, Any], baseline: dict[str, Any], gate: dict[str, Any]
) -> dict[str, Any]:
    return continuity_gate(
        candidate=evaluation["metrics"],
        baseline=baseline,
        protected_predictions_exact=evaluation["protected_predictions_exact"],
        existing_deck_predictions_exact=evaluation[
            "existing_deck_predictions_exact"
        ],
        minimum_iou_gain=float(gate["minimum_deck_iou_gain"]),
        minimum_recall_gain=float(gate["minimum_deck_recall_gain"]),
        minimum_precision=float(gate["minimum_deck_precision"]),
        minimum_miou_gain=float(gate["minimum_mIoU_gain"]),
        maximum_background_fp_ratio=float(
            gate["maximum_background_to_deck_fp_ratio"]
        ),
    )


def evaluate_grid(
    *,
    names: list[str],
    radii: list[int],
    thresholds: list[float],
    labels_root: Path,
    baseline_paths: dict[str, Path],
    sem001: dict[str, Path],
    probability: dict[str, Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    keys = [(int(radius), float(threshold)) for radius in radii for threshold in thresholds]
    states = {
        key: {
            "confusion": np.zeros((5, 5), dtype=np.int64),
            "additions_by_gt": np.zeros(5, dtype=np.int64),
            "per_view": [],
        }
        for key in keys
    }
    baseline_confusions = []
    for name in names:
        target = class_map(labels_root / name, context="SEM-171 training label")
        baseline = class_map(
            baseline_paths[name], context="SEM-171 SEM-160 baseline"
        )
        provenance = class_map(
            sem001[name], context="SEM-171 SEM-001 provenance"
        )
        pre = pre_foundation_map(baseline, provenance)
        probability_map = deck_probability(probability[name])
        base_confusion = local_confusion(target, baseline)
        baseline_confusions.append(base_confusion)

        for radius in radii:
            enclosed = closed_deck_mask(baseline, radius=int(radius))
            spatial = enclosed & (baseline == 0) & (pre == 0)
            spatial_probability = probability_map[spatial]
            spatial_target = target[spatial]
            for threshold in thresholds:
                accepted_target = spatial_target[
                    spatial_probability >= float(threshold)
                ]
                counts = np.bincount(accepted_target, minlength=5).astype(
                    np.int64, copy=False
                )
                key = (int(radius), float(threshold))
                candidate_confusion = adjusted_confusion(base_confusion, counts)
                states[key]["confusion"] += candidate_confusion
                states[key]["additions_by_gt"] += counts
                local = confusion_metrics(candidate_confusion)
                states[key]["per_view"].append(
                    {
                        "view": name,
                        "deck_iou": local["per_class"]["deck"]["iou"],
                    }
                )

    baseline_metrics = metrics_from_confusion(merge_confusions(baseline_confusions))
    records = []
    for radius, threshold in keys:
        state = states[(radius, threshold)]
        additions = state["additions_by_gt"]
        records.append(
            {
                "radius": radius,
                "threshold": threshold,
                "metrics": metrics_from_confusion(state["confusion"]),
                "protected_predictions_exact": True,
                "existing_deck_predictions_exact": True,
                "additions_by_gt_class": additions.astype(int).tolist(),
                "added_deck_pixels": int(additions.sum()),
                "per_view": state["per_view"],
            }
        )
    return records, baseline_metrics


def evaluate_one(
    *,
    names: list[str],
    radius: int,
    threshold: float,
    labels_root: Path,
    baseline_paths: dict[str, Path],
    sem001: dict[str, Path],
    probability: dict[str, Path],
    prediction_root: Path | None = None,
) -> dict[str, Any]:
    confusions = []
    additions_by_gt = np.zeros(5, dtype=np.int64)
    enclosed_pixels = 0
    protected_exact = True
    existing_deck_exact = True
    per_view = []
    prediction_records = []
    for name in names:
        target = class_map(labels_root / name, context="SEM-171 training label")
        baseline = class_map(
            baseline_paths[name], context="SEM-171 SEM-160 baseline"
        )
        provenance = class_map(
            sem001[name], context="SEM-171 SEM-001 provenance"
        )
        candidate, enclosed, additions = recover_deck_continuity(
            baseline,
            pre_foundation_map(baseline, provenance),
            deck_probability(probability[name]),
            radius=radius,
            threshold=threshold,
        )
        existing_deck_exact = existing_deck_exact and bool(
            np.array_equal(candidate[baseline == 1], baseline[baseline == 1])
        )
        protected = (baseline == 2) | (baseline == 3) | (baseline == 4)
        protected_exact = protected_exact and bool(
            np.array_equal(candidate[protected], baseline[protected])
        )
        additions_by_gt += np.bincount(target[additions], minlength=5)
        enclosed_pixels += int(np.count_nonzero(enclosed))
        confusion = local_confusion(target, candidate)
        confusions.append(confusion)
        local = confusion_metrics(confusion)
        per_view.append(
            {"view": name, "deck_iou": local["per_class"]["deck"]["iou"]}
        )
        if prediction_root is not None:
            prediction_root.mkdir(parents=True, exist_ok=True)
            path = prediction_root / name
            Image.fromarray(candidate).save(path)
            prediction_records.append(
                {
                    "name": name,
                    "prediction": str(path.relative_to(prediction_root.parent)),
                    "prediction_sha256": sha256(path),
                    "enclosed_non_deck_pixels": int(np.count_nonzero(enclosed)),
                    "added_deck_pixels": int(np.count_nonzero(additions)),
                }
            )
    return {
        "metrics": metrics_from_confusion(merge_confusions(confusions)),
        "protected_predictions_exact": protected_exact,
        "existing_deck_predictions_exact": existing_deck_exact,
        "additions_by_gt_class": additions_by_gt.astype(int).tolist(),
        "added_deck_pixels": int(additions_by_gt.sum()),
        "enclosed_non_deck_pixels": enclosed_pixels,
        "per_view": per_view,
        "prediction_records": prediction_records,
    }


def write_panels(
    *,
    names: list[str],
    labels_root: Path,
    baseline_paths: dict[str, Path],
    candidate_root: Path,
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = class_map(labels_root / name, context="SEM-171 panel label")
        before = class_map(
            baseline_paths[name], context="SEM-171 panel baseline"
        )
        after = class_map(candidate_root / name, context="SEM-171 panel candidate")
        changed = before != after
        outcome = np.zeros((*target.shape, 3), dtype=np.uint8)
        outcome[changed & (after == target)] = np.asarray([48, 220, 96], np.uint8)
        outcome[changed & (after != target)] = np.asarray([255, 48, 48], np.uint8)
        panel = np.concatenate(
            [semantic_rgb(target), semantic_rgb(before), semantic_rgb(after), outcome],
            axis=1,
        )
        Image.fromarray(panel).save(output_root / name)


def require_baseline_match(
    observed: dict[str, Any], expected: dict[str, Any], *, phase: str
) -> None:
    require(
        observed["confusion_matrix_rows_gt_columns_prediction"]
        == expected["confusion_matrix_rows_gt_columns_prediction"],
        f"SEM-171 {phase} baseline confusion does not match SEM-160",
    )


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    require(
        config.get("experiment_id") == "SEM-171-DECK-CONTINUITY-OOF",
        "SEM-171 received another config",
    )
    require(
        config.get("status") == "registered_not_run",
        "SEM-171 requires registered config",
    )
    require(
        not args.output_dir.exists() or not any(args.output_dir.iterdir()),
        "SEM-171 refuses to overwrite output",
    )
    require(
        sha256(args.split) == config["data"]["semantic_split_sha256"],
        "SEM-171 semantic split hash mismatch",
    )
    require(
        sha256(args.oof_split) == config["data"]["oof_split_sha256"],
        "SEM-171 OOF split hash mismatch",
    )
    require(
        sha256(args.sem150_result) == config["inputs"]["sem150_result_sha256"],
        "SEM-171 SEM-150 result hash mismatch",
    )
    require(
        sha256(args.sem160_result) == config["inputs"]["sem160_result_sha256"],
        "SEM-171 SEM-160 result hash mismatch",
    )
    sem160 = read_json(args.sem160_result)
    require(
        sem160.get("status") == "completed_oof_gate_passed",
        "SEM-171 requires passing SEM-160",
    )
    names = select_rendered_training_views(
        read_json(args.split), expected_count=int(config["data"]["training_images"])
    )
    partition = partition_training_names(names, expected_count=len(names))
    sem001 = sem001_paths(
        args.sem001_training_root,
        names,
        manifest_sha256=config["inputs"]["sem001_training_manifest_sha256"],
    )
    prediction_paths(
        args.sem150_root,
        names,
        manifest_sha256=config["inputs"]["sem150_prediction_manifest_sha256"],
        experiment_id="SEM-150-FOUNDATION-GATE-OOF",
    )
    baseline_paths = prediction_paths(
        args.sem160_root,
        names,
        manifest_sha256=config["inputs"]["sem160_prediction_manifest_sha256"],
        experiment_id="SEM-160-DECK-BINARY-OOF",
    )
    probability = oof_probability_paths(
        args.sem160_oof_root,
        read_json(args.oof_split),
        names,
        result_hashes=list(config["inputs"]["sem160_fold_result_sha256"]),
        manifest_hashes=list(
            config["inputs"]["sem160_fold_prediction_manifest_sha256"]
        ),
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

    records, observed_baseline = evaluate_grid(
        names=partition["calibration"],
        radii=[int(value) for value in config["recovery"]["radius_grid_pixels"]],
        thresholds=[
            float(value)
            for value in config["recovery"]["probability_threshold_grid"]
        ],
        labels_root=args.labels_root,
        baseline_paths=baseline_paths,
        sem001=sem001,
        probability=probability,
    )
    require_baseline_match(
        observed_baseline, baseline_metrics["calibration"], phase="calibration"
    )
    for record in records:
        record["gate"] = gate_for(
            record, baseline_metrics["calibration"], config["oof_gates"]["calibration"]
        )
        print(
            json.dumps(
                {
                    "phase": "calibration",
                    "radius": record["radius"],
                    "threshold": record["threshold"],
                    "deck_iou": record["metrics"]["per_class"]["deck"]["iou"],
                    "deck_precision": record["metrics"]["per_class"]["deck"][
                        "precision"
                    ],
                    "deck_recall": record["metrics"]["per_class"]["deck"][
                        "recall"
                    ],
                    "added_deck_pixels": record["added_deck_pixels"],
                    "passed": record["gate"]["passed"],
                }
            ),
            flush=True,
        )

    selected = select_continuity_candidate(records)
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "radius_grid_pixels": list(config["recovery"]["radius_grid_pixels"]),
        "probability_threshold_grid": list(
            config["recovery"]["probability_threshold_grid"]
        ),
        "baseline_calibration_metrics": baseline_metrics["calibration"],
        "baseline_check_metrics": baseline_metrics["check"],
        "calibration_records": records,
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
                "selected_radius": None,
                "selected_threshold": None,
                "check_labels_opened": False,
                "next_step": "Drop spatial continuity recovery and retain SEM-161.",
            }
        )
        write_json(args.output_dir / "result.json", result)
        print(json.dumps({"status": result["status"]}), flush=True)
        return 0

    radius = int(selected["radius"])
    threshold = float(selected["threshold"])
    prediction_root = args.output_dir / "selected_predictions"
    check = evaluate_one(
        names=partition["check"],
        radius=radius,
        threshold=threshold,
        labels_root=args.labels_root,
        baseline_paths=baseline_paths,
        sem001=sem001,
        probability=probability,
        prediction_root=prediction_root,
    )
    check["gate"] = gate_for(
        check, baseline_metrics["check"], config["oof_gates"]["check"]
    )
    calibration_materialised = evaluate_one(
        names=partition["calibration"],
        radius=radius,
        threshold=threshold,
        labels_root=args.labels_root,
        baseline_paths=baseline_paths,
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
            "selected_radius": radius,
            "selected_threshold": threshold,
            "predictions": prediction_records,
            "fixed_validation_opened": False,
        },
    )
    write_panels(
        names=list(config["outputs"]["training_visual_samples"]),
        labels_root=args.labels_root,
        baseline_paths=baseline_paths,
        candidate_root=prediction_root,
        output_root=args.output_dir / "training_panels",
    )

    selected_by_name = {record["view"]: record for record in selected["per_view"]}
    comparison = {"improved": 0, "equal": 0, "regressed": 0}
    for after in check["per_view"]:
        target = class_map(
            args.labels_root / after["view"], context="SEM-171 check label"
        )
        baseline = class_map(
            baseline_paths[after["view"]], context="SEM-171 check baseline"
        )
        before = confusion_metrics(local_confusion(target, baseline))
        delta = float(after["deck_iou"]) - float(
            before["per_class"]["deck"]["iou"]
        )
        key = "improved" if delta > 1e-12 else "regressed" if delta < -1e-12 else "equal"
        comparison[key] += 1
    require(
        set(selected_by_name) == set(partition["calibration"]),
        "SEM-171 selected calibration record is incomplete",
    )

    passed = bool(check["gate"]["passed"])
    result.update(
        {
            "status": (
                "completed_oof_gate_passed"
                if passed
                else "completed_check_gate_failed"
            ),
            "selected_radius": radius,
            "selected_threshold": threshold,
            "selected_calibration_record": selected,
            "check_record": check,
            "check_deck_iou_per_view_comparison": comparison,
            "check_labels_opened": True,
            "prediction_manifest_sha256": sha256(
                args.output_dir / "prediction_manifest.json"
            ),
            "next_step": (
                "Register one frozen final application over SEM-161."
                if passed
                else "Drop spatial continuity recovery and retain SEM-161."
            ),
        }
    )
    write_json(args.output_dir / "result.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_radius": radius,
                "selected_threshold": threshold,
                "check_gate": check["gate"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
