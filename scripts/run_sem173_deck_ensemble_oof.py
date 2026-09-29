#!/usr/bin/env python3
"""Qualify the frozen SEM-160/172 deck-probability ensemble."""

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
from scripts.run_sem171_deck_continuity_oof import adjusted_confusion
from src.semantic2d import confusion_metrics, read_json, sha256, write_json
from src.semantic_deck_binary import deck_gate, recover_deck
from src.semantic_deck_ensemble import (
    blend_deck_probability,
    select_ensemble_candidate,
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
    parser.add_argument("--sem172-result", type=Path, required=True)
    parser.add_argument("--sem172-oof-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def gate_for(
    evaluation: dict[str, Any], baseline: dict[str, Any], gate: dict[str, Any]
) -> dict[str, Any]:
    return deck_gate(
        candidate=evaluation["metrics"],
        baseline=baseline,
        protected_predictions_exact=evaluation["protected_predictions_exact"],
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
    weights: list[float],
    thresholds: list[float],
    labels_root: Path,
    base_paths: dict[str, Path],
    sem001_paths_by_name: dict[str, Path],
    sem160_probability_paths: dict[str, Path],
    sem172_probability_paths: dict[str, Path],
) -> list[dict[str, Any]]:
    keys = [(float(weight), float(threshold)) for weight in weights for threshold in thresholds]
    states = {
        key: {
            "confusion": np.zeros((5, 5), dtype=np.int64),
            "additions_by_gt": np.zeros(5, dtype=np.int64),
            "per_view": [],
        }
        for key in keys
    }
    for name in names:
        target = class_map(labels_root / name, context="SEM-173 training label")
        base = class_map(base_paths[name], context="SEM-173 SEM-150 baseline")
        provenance = class_map(
            sem001_paths_by_name[name], context="SEM-173 SEM-001 provenance"
        )
        eligible = (base == 0) & (pre_foundation_map(base, provenance) == 0)
        eligible_target = target[eligible]
        base_confusion = local_confusion(target, base)
        p160 = deck_probability(sem160_probability_paths[name])
        p172 = deck_probability(sem172_probability_paths[name])
        for weight in weights:
            blended = blend_deck_probability(
                p160, p172, sem160_weight=float(weight)
            )[eligible]
            for threshold in thresholds:
                counts = np.bincount(
                    eligible_target[blended >= float(threshold)], minlength=5
                ).astype(np.int64, copy=False)
                key = (float(weight), float(threshold))
                candidate_confusion = adjusted_confusion(base_confusion, counts)
                states[key]["confusion"] += candidate_confusion
                states[key]["additions_by_gt"] += counts
                local = confusion_metrics(candidate_confusion)
                states[key]["per_view"].append(
                    {"view": name, "deck_iou": local["per_class"]["deck"]["iou"]}
                )
    records = []
    for weight, threshold in keys:
        state = states[(weight, threshold)]
        additions = state["additions_by_gt"]
        records.append(
            {
                "sem160_weight": weight,
                "threshold": threshold,
                "metrics": metrics_from_confusion(state["confusion"]),
                "protected_predictions_exact": True,
                "additions_by_gt_class": additions.astype(int).tolist(),
                "added_deck_pixels": int(additions.sum()),
                "per_view": state["per_view"],
            }
        )
    return records


def evaluate_one(
    *,
    names: list[str],
    weight: float,
    threshold: float,
    labels_root: Path,
    base_paths: dict[str, Path],
    sem001_paths_by_name: dict[str, Path],
    sem160_probability_paths: dict[str, Path],
    sem172_probability_paths: dict[str, Path],
    prediction_root: Path | None = None,
) -> dict[str, Any]:
    confusions = []
    additions_by_gt = np.zeros(5, dtype=np.int64)
    protected_exact = True
    per_view = []
    prediction_records = []
    for name in names:
        target = class_map(labels_root / name, context="SEM-173 training label")
        base = class_map(base_paths[name], context="SEM-173 SEM-150 baseline")
        provenance = class_map(
            sem001_paths_by_name[name], context="SEM-173 SEM-001 provenance"
        )
        blended = blend_deck_probability(
            deck_probability(sem160_probability_paths[name]),
            deck_probability(sem172_probability_paths[name]),
            sem160_weight=weight,
        )
        candidate, eligible, additions = recover_deck(
            base,
            pre_foundation_map(base, provenance),
            blended,
            threshold=threshold,
        )
        protected_exact = protected_exact and bool(
            np.array_equal(candidate[~eligible], base[~eligible])
        )
        additions_by_gt += np.bincount(target[additions], minlength=5)
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
                    "added_deck_pixels": int(np.count_nonzero(additions)),
                }
            )
    return {
        "metrics": metrics_from_confusion(merge_confusions(confusions)),
        "protected_predictions_exact": protected_exact,
        "additions_by_gt_class": additions_by_gt.astype(int).tolist(),
        "added_deck_pixels": int(additions_by_gt.sum()),
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
        target = class_map(labels_root / name, context="SEM-173 panel label")
        before = class_map(baseline_paths[name], context="SEM-173 panel baseline")
        after = class_map(candidate_root / name, context="SEM-173 panel candidate")
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
    require(
        config.get("experiment_id") == "SEM-173-DECK-ENSEMBLE-OOF",
        "SEM-173 received another config",
    )
    require(config.get("status") == "registered_not_run", "SEM-173 requires registered config")
    require(
        not args.output_dir.exists() or not any(args.output_dir.iterdir()),
        "SEM-173 refuses to overwrite output",
    )
    require(
        sha256(args.split) == config["data"]["semantic_split_sha256"]
        and sha256(args.oof_split) == config["data"]["oof_split_sha256"],
        "SEM-173 split hash mismatch",
    )
    require(
        sha256(args.sem150_result) == config["inputs"]["sem150_result_sha256"],
        "SEM-173 SEM-150 result hash mismatch",
    )
    require(
        sha256(args.sem160_result) == config["inputs"]["sem160_result_sha256"],
        "SEM-173 SEM-160 result hash mismatch",
    )
    require(
        sha256(args.sem172_result) == config["inputs"]["sem172_result_sha256"],
        "SEM-173 SEM-172 result hash mismatch",
    )
    sem160_result = read_json(args.sem160_result)
    sem172_result = read_json(args.sem172_result)
    require(
        sem160_result.get("status") == "completed_oof_gate_passed"
        and sem172_result.get("status") == "completed_calibration_gate_failed",
        "SEM-173 parent status mismatch",
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
    base = prediction_paths(
        args.sem150_root,
        names,
        manifest_sha256=config["inputs"]["sem150_prediction_manifest_sha256"],
        experiment_id="SEM-150-FOUNDATION-GATE-OOF",
    )
    sem160_predictions = prediction_paths(
        args.sem160_root,
        names,
        manifest_sha256=config["inputs"]["sem160_prediction_manifest_sha256"],
        experiment_id="SEM-160-DECK-BINARY-OOF",
    )
    p160 = oof_probability_paths(
        args.sem160_oof_root,
        oof_split,
        names,
        result_hashes=list(config["inputs"]["sem160_fold_result_sha256"]),
        manifest_hashes=list(
            config["inputs"]["sem160_fold_prediction_manifest_sha256"]
        ),
    )
    p172 = oof_probability_paths(
        args.sem172_oof_root,
        oof_split,
        names,
        result_hashes=list(config["inputs"]["sem172_fold_result_sha256"]),
        manifest_hashes=list(
            config["inputs"]["sem172_fold_prediction_manifest_sha256"]
        ),
    )
    baseline_metrics = {
        "calibration": sem160_result["selected_calibration_record"]["metrics"],
        "check": sem160_result["check_record"]["metrics"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output_dir / "input_access_manifest.json",
        {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "training_views": names,
            "partition": partition,
            "all_probability_and_parent_inputs_validated_before_labels": True,
            "fixed_validation_opened": False,
            "unlabeled_images_opened": False,
            "real_rgb_input_opened": False,
        },
    )
    records = evaluate_grid(
        names=partition["calibration"],
        weights=[float(value) for value in config["ensemble"]["sem160_weight_grid"]],
        thresholds=[float(value) for value in config["ensemble"]["threshold_grid"]],
        labels_root=args.labels_root,
        base_paths=base,
        sem001_paths_by_name=sem001,
        sem160_probability_paths=p160,
        sem172_probability_paths=p172,
    )
    for record in records:
        record["gate"] = gate_for(
            record, baseline_metrics["calibration"], config["oof_gates"]["calibration"]
        )
        print(
            json.dumps(
                {
                    "phase": "calibration",
                    "sem160_weight": record["sem160_weight"],
                    "threshold": record["threshold"],
                    "deck_iou": record["metrics"]["per_class"]["deck"]["iou"],
                    "deck_precision": record["metrics"]["per_class"]["deck"]["precision"],
                    "deck_recall": record["metrics"]["per_class"]["deck"]["recall"],
                    "passed": record["gate"]["passed"],
                }
            ),
            flush=True,
        )
    selected = select_ensemble_candidate(records)
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "sem160_weight_grid": list(config["ensemble"]["sem160_weight_grid"]),
        "threshold_grid": list(config["ensemble"]["threshold_grid"]),
        "baseline_calibration_metrics": baseline_metrics["calibration"],
        "baseline_check_metrics": baseline_metrics["check"],
        "calibration_records": records,
        "input_access_manifest_sha256": sha256(args.output_dir / "input_access_manifest.json"),
        "fixed_validation_opened": False,
        "unlabeled_images_opened": False,
        "real_rgb_input_opened": False,
        "rgb_modified": False,
    }
    if selected is None:
        result.update(
            {
                "status": "completed_calibration_gate_failed",
                "selected_sem160_weight": None,
                "selected_threshold": None,
                "check_labels_opened": False,
                "next_step": "End current deck refinement series and retain SEM-161.",
            }
        )
        write_json(args.output_dir / "result.json", result)
        print(json.dumps({"status": result["status"]}), flush=True)
        return 0
    weight = float(selected["sem160_weight"])
    threshold = float(selected["threshold"])
    prediction_root = args.output_dir / "selected_predictions"
    check = evaluate_one(
        names=partition["check"],
        weight=weight,
        threshold=threshold,
        labels_root=args.labels_root,
        base_paths=base,
        sem001_paths_by_name=sem001,
        sem160_probability_paths=p160,
        sem172_probability_paths=p172,
        prediction_root=prediction_root,
    )
    check["gate"] = gate_for(
        check, baseline_metrics["check"], config["oof_gates"]["check"]
    )
    calibration_materialised = evaluate_one(
        names=partition["calibration"],
        weight=weight,
        threshold=threshold,
        labels_root=args.labels_root,
        base_paths=base,
        sem001_paths_by_name=sem001,
        sem160_probability_paths=p160,
        sem172_probability_paths=p172,
        prediction_root=prediction_root,
    )
    by_name = {
        record["name"]: record
        for record in check["prediction_records"]
        + calibration_materialised["prediction_records"]
    }
    write_json(
        args.output_dir / "prediction_manifest.json",
        {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "status": "completed_selected_predictions",
            "selected_sem160_weight": weight,
            "selected_threshold": threshold,
            "predictions": [by_name[name] for name in names],
            "fixed_validation_opened": False,
        },
    )
    write_panels(
        names=list(config["outputs"]["training_visual_samples"]),
        labels_root=args.labels_root,
        baseline_paths=sem160_predictions,
        candidate_root=prediction_root,
        output_root=args.output_dir / "training_panels",
    )
    passed = bool(check["gate"]["passed"])
    result.update(
        {
            "status": "completed_oof_gate_passed" if passed else "completed_check_gate_failed",
            "selected_sem160_weight": weight,
            "selected_threshold": threshold,
            "selected_calibration_record": selected,
            "check_labels_opened": True,
            "check_record": check,
            "prediction_manifest_sha256": sha256(args.output_dir / "prediction_manifest.json"),
            "next_step": "Register frozen full-data ensemble." if passed else "End current deck refinement series and retain SEM-161.",
        }
    )
    write_json(args.output_dir / "result.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_sem160_weight": weight,
                "selected_threshold": threshold,
                "check_gate": check["gate"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
