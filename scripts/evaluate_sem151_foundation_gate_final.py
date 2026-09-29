#!/usr/bin/env python3
"""Apply SEM-150's frozen foundation gate to SEM-141 developer predictions."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.semantic2d import confusion_metrics, read_json, sha256, write_json
from src.semantic_foundation_verifier import foundation_gate, verify_foundation
from src.semantic_protected_fusion import (
    local_confusion,
    merge_confusions,
    metrics_from_confusion,
    validate_probabilities,
)
from src.semantic_render_domain import semantic_rgb
from src.semantic_rendered_final import require_exact_view_set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--sem150-result", type=Path, required=True)
    parser.add_argument("--sem141-result", type=Path, required=True)
    parser.add_argument("--sem141-root", type=Path, required=True)
    parser.add_argument("--sem102-result", type=Path, required=True)
    parser.add_argument("--sem102-root", type=Path, required=True)
    parser.add_argument("--rendered-rgb-root", type=Path, required=True)
    parser.add_argument("--sem102-external-logits", type=Path, required=True)
    parser.add_argument("--sem141-external-root", type=Path, required=True)
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


def rgb_image(path: Path) -> np.ndarray:
    value = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    if value.shape != (989, 1320, 3):
        raise ValueError(f"SEM-151 RGB is not 1320x989: {path}")
    return value


def five_class_probabilities(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        require(
            archive.files == ["probabilities"],
            "SEM-151 probability NPZ has an unexpected schema",
        )
        value = archive["probabilities"]
    return validate_probabilities(value, shape=(989, 1320))


def softmax(logits: np.ndarray) -> np.ndarray:
    value = np.asarray(logits, dtype=np.float32)
    require(value.shape == (5, 989, 1320), "SEM-151 external logits shape mismatch")
    value = value - value.max(axis=0, keepdims=True)
    exponent = np.exp(value)
    return exponent / exponent.sum(axis=0, keepdims=True)


def one_hot_logits(prediction: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    probability = np.full((5, *prediction.shape), epsilon, dtype=np.float32)
    for class_id in range(5):
        probability[class_id, prediction == class_id] = 1.0 - epsilon * 4
    return np.log(np.clip(probability, np.finfo(np.float32).tiny, 1.0)).astype(
        np.float32
    )


def validate_sem141_inputs(
    root: Path, names: list[str], *, manifest_sha256: str
) -> dict[str, Path]:
    manifest_path = root / "prediction_manifest.json"
    require(
        manifest_path.is_file() and sha256(manifest_path) == manifest_sha256,
        "SEM-151 SEM-141 prediction manifest hash mismatch",
    )
    manifest = read_json(manifest_path)
    require(
        manifest.get("status") == "completed_predictions_before_developer_labels"
        and manifest.get("developer_labels_opened_before_manifest") is False,
        "SEM-151 SEM-141 parent predictions violated access order",
    )
    records = manifest["predictions"]
    require(
        [record["name"] for record in records] == names,
        "SEM-151 SEM-141 prediction order mismatch",
    )
    output: dict[str, Path] = {}
    for record in records:
        path = root / record["prediction"]
        require(
            path.is_file() and sha256(path) == record["prediction_sha256"],
            f"SEM-151 SEM-141 prediction hash mismatch: {record['name']}",
        )
        output[record["name"]] = path
    return output


def validate_sem102_inputs(
    root: Path, names: list[str], *, manifest_sha256: str
) -> dict[str, Path]:
    manifest_path = root / "prediction_manifest.json"
    require(
        manifest_path.is_file() and sha256(manifest_path) == manifest_sha256,
        "SEM-151 SEM-102 prediction manifest hash mismatch",
    )
    records = read_json(manifest_path)
    require(
        [record["name"] for record in records] == names,
        "SEM-151 SEM-102 probability order mismatch",
    )
    output: dict[str, Path] = {}
    for record in records:
        path = root / record["probabilities"]
        require(
            path.is_file() and sha256(path) == record["probabilities_sha256"],
            f"SEM-151 SEM-102 probability hash mismatch: {record['name']}",
        )
        output[record["name"]] = path
    return output


def extended_metrics(
    confusions: list[np.ndarray], per_view: list[dict[str, Any]]
) -> dict[str, Any]:
    metrics = metrics_from_confusion(merge_confusions(confusions))
    worst_count = max(1, int(np.ceil(len(per_view) * 0.10)))
    worst = sorted(per_view, key=lambda value: value["mIoU_without_background"])[
        :worst_count
    ]
    challenge = [
        value for value in per_view if 140 <= int(Path(value["view"]).stem) <= 146
    ]
    metrics.update(
        {
            "view_count": len(per_view),
            "per_view": per_view,
            "worst_10_percent": {
                "count": worst_count,
                "views": [value["view"] for value in worst],
                "mean_mIoU_without_background": float(
                    np.mean([value["mIoU_without_background"] for value in worst])
                ),
            },
            "challenge_140_146": {
                "views": [value["view"] for value in challenge],
                "mean_mIoU_without_background": float(
                    np.mean(
                        [value["mIoU_without_background"] for value in challenge]
                    )
                ),
            },
        }
    )
    return metrics


def metrics_for(
    names: list[str], labels_root: Path, predictions: dict[str, Path]
) -> dict[str, Any]:
    confusions: list[np.ndarray] = []
    per_view: list[dict[str, Any]] = []
    for name in names:
        target = class_map(labels_root / name, context="SEM-151 developer label")
        prediction = class_map(predictions[name], context="SEM-151 prediction")
        confusion = local_confusion(target, prediction)
        confusions.append(confusion)
        per_view.append({"view": name, **confusion_metrics(confusion)})
    return extended_metrics(confusions, per_view)


def write_panels(
    *,
    names: list[str],
    labels_root: Path,
    baseline: dict[str, Path],
    candidate: dict[str, Path],
    rgb_root: Path,
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = class_map(labels_root / name, context="SEM-151 panel target")
        before = class_map(baseline[name], context="SEM-151 panel baseline")
        after = class_map(candidate[name], context="SEM-151 panel candidate")
        outcome = np.zeros((*target.shape, 3), dtype=np.uint8)
        changed = before != after
        outcome[changed & (target != 4)] = np.asarray([48, 220, 96], dtype=np.uint8)
        outcome[changed & (target == 4)] = np.asarray([255, 48, 48], dtype=np.uint8)
        panel = np.concatenate(
            [
                rgb_image(rgb_root / name),
                semantic_rgb(target),
                semantic_rgb(before),
                semantic_rgb(after),
                outcome,
            ],
            axis=1,
        )
        Image.fromarray(panel).save(output_root / name)


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    require(
        config.get("experiment_id") == "SEM-151-FOUNDATION-GATE-FINAL",
        "SEM-151 received another experiment config",
    )
    require(
        config.get("status") == "registered_not_run",
        "SEM-151 may run only from its registered configuration",
    )
    require(
        not args.output_dir.exists() or not any(args.output_dir.iterdir()),
        "SEM-151 refuses to overwrite developer output",
    )
    require(
        sha256(args.split) == config["data"]["semantic_split_sha256"],
        "SEM-151 semantic split hash mismatch",
    )
    require(
        sha256(args.sem150_result) == config["inputs"]["sem150_result_sha256"],
        "SEM-151 SEM-150 result hash mismatch",
    )
    require(
        sha256(args.sem141_result) == config["inputs"]["sem141_result_sha256"],
        "SEM-151 SEM-141 result hash mismatch",
    )
    require(
        sha256(args.sem102_result) == config["inputs"]["sem102_result_sha256"],
        "SEM-151 SEM-102 result hash mismatch",
    )
    sem150 = read_json(args.sem150_result)
    sem141 = read_json(args.sem141_result)
    require(
        sem150.get("status") == "completed_oof_gate_passed",
        "SEM-151 requires a passing SEM-150 OOF gate",
    )
    require(
        sem141.get("status") == "completed_developer_target_gate_passed",
        "SEM-151 requires the frozen SEM-141 developer baseline",
    )
    threshold = float(config["fusion"]["foundation_threshold"])
    require(
        threshold == float(sem150["selected_threshold"]),
        "SEM-151 threshold differs from SEM-150",
    )
    split = read_json(args.split)
    names = require_exact_view_set(
        split["semantic_validation_labeled"],
        split["semantic_validation_labeled"],
        context="SEM-151 developer validation plan",
    )
    baseline_paths = validate_sem141_inputs(
        args.sem141_root,
        names,
        manifest_sha256=config["inputs"]["sem141_prediction_manifest_sha256"],
    )
    probability_paths = validate_sem102_inputs(
        args.sem102_root,
        names,
        manifest_sha256=config["inputs"]["sem102_prediction_manifest_sha256"],
    )
    require(
        args.sem102_external_logits.is_file()
        and sha256(args.sem102_external_logits)
        == config["inputs"]["sem102_external_logits_sha256"],
        "SEM-151 external foundation logits hash mismatch",
    )
    external_base_path = args.sem141_external_root / "semantic.png"
    external_rgb_path = args.sem141_external_root / "rgb.png"
    require(
        sha256(external_base_path)
        == config["inputs"]["sem141_external_semantic_sha256"]
        and sha256(external_rgb_path)
        == config["inputs"]["sem141_external_rgb_sha256"],
        "SEM-151 SEM-141 external artifact hash mismatch",
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    prediction_root = args.output_dir / "predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)
    immutable_exact = True
    records: list[dict[str, Any]] = []
    # Materialise all predictions before opening any developer label.
    for name in names:
        baseline = class_map(baseline_paths[name], context="SEM-151 SEM-141 baseline")
        probability = five_class_probabilities(probability_paths[name])[4]
        candidate, _, rejected = verify_foundation(
            baseline, probability, threshold=threshold
        )
        immutable_exact = immutable_exact and all(
            np.array_equal(candidate == class_id, baseline == class_id)
            for class_id in (1, 2, 3)
        )
        output_path = prediction_root / name
        Image.fromarray(candidate).save(output_path)
        records.append(
            {
                "name": name,
                "sem141_prediction_sha256": sha256(baseline_paths[name]),
                "sem102_probability_sha256": sha256(probability_paths[name]),
                "prediction": str(output_path.relative_to(args.output_dir)),
                "prediction_sha256": sha256(output_path),
                "rejected_foundation_pixels": int(np.count_nonzero(rejected)),
            }
        )
    write_json(
        args.output_dir / "prediction_manifest.json",
        {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "status": "completed_predictions_before_developer_labels",
            "selected_threshold": threshold,
            "predictions": records,
            "immutable_deck_stay_cable_tower_predictions_exact": immutable_exact,
            "developer_labels_opened_before_manifest": False,
            "hidden_test_opened": False,
            "real_rgb_input_opened": False,
        },
    )
    candidate_paths = {name: prediction_root / name for name in names}

    baseline_metrics = metrics_for(names, args.labels_root, baseline_paths)
    candidate_metrics = metrics_for(names, args.labels_root, candidate_paths)
    require(
        np.isclose(
            baseline_metrics["mIoU_with_background"],
            sem141["metrics"]["mIoU_with_background"],
            rtol=0.0,
            atol=1e-12,
        ),
        "SEM-151 did not reproduce the frozen SEM-141 baseline",
    )
    gate_config = config["developer_target_gate"]
    target_gate = foundation_gate(
        candidate=candidate_metrics,
        baseline=baseline_metrics,
        immutable_predictions_exact=immutable_exact,
        minimum_iou_gain=float(gate_config["minimum_foundation_iou_gain"]),
        minimum_precision_gain=float(
            gate_config["minimum_foundation_precision_gain"]
        ),
        minimum_recall=float(gate_config["minimum_foundation_recall"]),
        minimum_miou_gain=float(gate_config["minimum_mIoU_gain"]),
        minimum_background_fp_reduction=float(
            gate_config["minimum_background_to_foundation_reduction"]
        ),
    )
    retention = {
        "worst_10_percent": float(
            candidate_metrics["worst_10_percent"]["mean_mIoU_without_background"]
        )
        >= float(
            baseline_metrics["worst_10_percent"]["mean_mIoU_without_background"]
        )
        - float(gate_config["difficult_subset_drop_max"]),
        "challenge_140_146": float(
            candidate_metrics["challenge_140_146"]["mean_mIoU_without_background"]
        )
        >= float(
            baseline_metrics["challenge_140_146"]["mean_mIoU_without_background"]
        )
        - float(gate_config["difficult_subset_drop_max"]),
    }
    write_panels(
        names=list(config["outputs"]["validation_visual_samples"]),
        labels_root=args.labels_root,
        baseline=baseline_paths,
        candidate=candidate_paths,
        rgb_root=args.rendered_rgb_root,
        output_root=args.output_dir / "validation_panels",
    )

    external_base = class_map(external_base_path, context="SEM-151 external SEM-141")
    external_probabilities = softmax(np.load(args.sem102_external_logits))
    external_prediction, _, _ = verify_foundation(
        external_base, external_probabilities[4], threshold=threshold
    )
    external_root = args.output_dir / "external_camera"
    external_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(external_rgb_path, external_root / "rgb.png")
    Image.fromarray(external_prediction).save(external_root / "semantic.png")
    np.save(external_root / "semantic_logits.npy", one_hot_logits(external_prediction))
    external = {
        "input_pose_convention": "CAMERA_FROM_WORLD",
        "input_camera_model": "SIMPLE_RADIAL",
        "output_grid": "official_SIMPLE_RADIAL",
        "rgb_shape": list(rgb_image(external_root / "rgb.png").shape),
        "rgb_bytes_identical": external_rgb_path.read_bytes()
        == (external_root / "rgb.png").read_bytes(),
        "semantic_logits_shape": [5, 989, 1320],
        "semantic_png_shape": list(external_prediction.shape),
        "semantic_png_min": int(external_prediction.min()),
        "semantic_png_max": int(external_prediction.max()),
        "immutable_deck_stay_cable_tower_predictions_exact": all(
            np.array_equal(external_prediction == class_id, external_base == class_id)
            for class_id in (1, 2, 3)
        ),
        "uses_reference_rgb_or_labels": False,
    }
    engineering = {
        "parent_checkpoint_reload": sem141["engineering_checks"][
            "checkpoint_reload"
        ],
        "immutable_deck_stay_cable_tower_predictions_exact": immutable_exact,
        "rgb_bytes_identical": external["rgb_bytes_identical"],
        "official_external_camera": external["uses_reference_rgb_or_labels"] is False
        and external["semantic_png_shape"] == [989, 1320]
        and external["semantic_logits_shape"] == [5, 989, 1320],
    }
    passed = bool(
        target_gate["passed"]
        and all(retention.values())
        and all(engineering.values())
    )
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "completed_developer_target_gate_passed"
        if passed
        else "completed_developer_target_gate_failed",
        "developer_only": True,
        "selected_threshold": threshold,
        "prediction_manifest_sha256": sha256(
            args.output_dir / "prediction_manifest.json"
        ),
        "developer_labels_opened_after_all_predictions": True,
        "baseline_metrics": baseline_metrics,
        "metrics": candidate_metrics,
        "developer_target_gate": target_gate,
        "difficult_subset_retention_checks": retention,
        "engineering_checks": engineering,
        "external_camera": external,
        "recommended_for_hidden_evaluation": passed,
        "fallback": config["fallback"],
        "panels": [
            str((args.output_dir / "validation_panels" / name).resolve())
            for name in config["outputs"]["validation_visual_samples"]
        ],
        "claim_boundary": "The 74-view split is an explicitly reused development set; only hidden evaluation can establish final generalisation.",
    }
    write_json(args.output_dir / "result.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "recommended_for_hidden_evaluation": passed,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
