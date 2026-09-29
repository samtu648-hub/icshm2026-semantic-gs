#!/usr/bin/env python3
"""Run the frozen SEM-161 deck model and its one-time developer audit."""

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
from src.semantic_deck_binary import (
    deck_gate,
    load_binary_model,
    read_rgb,
    recover_deck,
    sliding_deck_probability,
)
from src.semantic_protected_fusion import (
    local_confusion,
    merge_confusions,
    metrics_from_confusion,
)
from src.semantic_render_domain import semantic_rgb
from src.semantic_rendered_final import require_exact_view_set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--sem160-result", type=Path, required=True)
    parser.add_argument("--final-result", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--validation-render-root", type=Path, required=True)
    parser.add_argument("--sem151-root", type=Path, required=True)
    parser.add_argument("--sem151-result", type=Path, required=True)
    parser.add_argument("--sem141-root", type=Path, required=True)
    parser.add_argument("--sem151-external-root", type=Path, required=True)
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
        raise ValueError(f"SEM-161 RGB is not 1320x989: {path}")
    return value


def one_hot_logits(prediction: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    probability = np.full((5, *prediction.shape), epsilon, dtype=np.float32)
    for class_id in range(5):
        probability[class_id, prediction == class_id] = 1.0 - epsilon * 4
    require(
        np.allclose(probability.sum(axis=0), 1.0, rtol=0.0, atol=1e-6),
        "SEM-161 external probabilities are not a simplex",
    )
    return np.log(np.clip(probability, np.finfo(np.float32).tiny, 1.0)).astype(
        np.float32
    )


def validate_prediction_manifest(
    root: Path,
    names: list[str],
    *,
    manifest_sha256: str,
    context: str,
) -> dict[str, Path]:
    manifest_path = root / "prediction_manifest.json"
    require(
        manifest_path.is_file() and sha256(manifest_path) == manifest_sha256,
        f"SEM-161 {context} prediction manifest hash mismatch",
    )
    manifest = read_json(manifest_path)
    require(
        manifest.get("status") == "completed_predictions_before_developer_labels"
        and manifest.get("developer_labels_opened_before_manifest") is False,
        f"SEM-161 {context} parent predictions violated access order",
    )
    records = manifest["predictions"]
    require(
        [record["name"] for record in records] == names,
        f"SEM-161 {context} prediction order mismatch",
    )
    output: dict[str, Path] = {}
    for record in records:
        path = root / record["prediction"]
        require(
            path.is_file() and sha256(path) == record["prediction_sha256"],
            f"SEM-161 {context} prediction hash mismatch: {record['name']}",
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
    deck_worst = sorted(
        per_view,
        key=lambda value: (
            -1.0
            if value["per_class"]["deck"]["iou"] is None
            else value["per_class"]["deck"]["iou"]
        ),
    )[:worst_count]
    challenge = [
        value for value in per_view if 140 <= int(Path(value["view"]).stem) <= 146
    ]
    oblique = [
        value for value in per_view if 158 <= int(Path(value["view"]).stem) <= 161
    ]
    require(len(oblique) == 4, "SEM-161 developer set lacks oblique views 158--161")
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
            "deck_worst_10_percent": {
                "count": worst_count,
                "views": [value["view"] for value in deck_worst],
                "mean_deck_iou": float(
                    np.mean(
                        [
                            0.0
                            if value["per_class"]["deck"]["iou"] is None
                            else value["per_class"]["deck"]["iou"]
                            for value in deck_worst
                        ]
                    )
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
            "oblique_158_161": {
                "views": [value["view"] for value in oblique],
                "mean_deck_iou": float(
                    np.mean(
                        [value["per_class"]["deck"]["iou"] for value in oblique]
                    )
                ),
                "mean_deck_recall": float(
                    np.mean(
                        [
                            value["per_class"]["deck"]["recall"]
                            for value in oblique
                        ]
                    )
                ),
            },
        }
    )
    return metrics


def metrics_for(
    names: list[str], labels_root: Path, predictions: dict[str, Path]
) -> tuple[dict[str, Any], list[np.ndarray]]:
    confusions: list[np.ndarray] = []
    per_view: list[dict[str, Any]] = []
    targets: list[np.ndarray] = []
    for name in names:
        target = class_map(labels_root / name, context="SEM-161 developer label")
        prediction = class_map(predictions[name], context="SEM-161 prediction")
        confusion = local_confusion(target, prediction)
        confusions.append(confusion)
        targets.append(target)
        per_view.append({"view": name, **confusion_metrics(confusion)})
    return extended_metrics(confusions, per_view), targets


def protection_exact(
    base: np.ndarray,
    pre_foundation: np.ndarray,
    candidate: np.ndarray,
    eligible: np.ndarray,
) -> bool:
    return bool(
        np.array_equal(candidate[~eligible], base[~eligible])
        and all(
            np.array_equal(candidate == class_id, base == class_id)
            for class_id in (2, 3, 4)
        )
        and np.all((base == 1) <= (candidate == 1))
        and np.all((candidate != base) <= ((base == 0) & (pre_foundation == 0)))
    )


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
        target = class_map(labels_root / name, context="SEM-161 panel target")
        before = class_map(baseline[name], context="SEM-161 panel baseline")
        after = class_map(candidate[name], context="SEM-161 panel candidate")
        outcome = np.zeros((*target.shape, 3), dtype=np.uint8)
        changed = before != after
        outcome[changed & (target == 1)] = np.asarray([48, 220, 96], dtype=np.uint8)
        outcome[changed & (target != 1)] = np.asarray([255, 48, 48], dtype=np.uint8)
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
        config.get("experiment_id") == "SEM-161-DECK-BINARY-FINAL",
        "SEM-161 evaluator received another experiment",
    )
    require(
        config.get("status") == "registered_not_run",
        "SEM-161 evaluator requires its registered configuration",
    )
    require(
        config["final_evidence"]["status"]
        == "completed_final_checkpoint_frozen",
        "SEM-161 evaluator requires frozen final evidence",
    )
    require(
        not args.output_dir.exists() or not any(args.output_dir.iterdir()),
        "SEM-161 refuses to overwrite developer audit output",
    )
    require(
        sha256(args.split) == config["data"]["semantic_split_sha256"],
        "SEM-161 semantic split hash mismatch",
    )
    require(
        sha256(args.sem160_result)
        == config["parent_evidence"]["sem160_result_sha256"],
        "SEM-161 SEM-160 result hash mismatch",
    )
    sem160 = read_json(args.sem160_result)
    require(
        sem160.get("status") == "completed_oof_gate_passed",
        "SEM-161 requires a passing SEM-160 OOF gate",
    )
    threshold = float(config["fusion"]["deck_threshold"])
    require(
        threshold == float(sem160["selected_threshold"]),
        "SEM-161 threshold differs from SEM-160",
    )
    require(
        sha256(args.final_result)
        == config["final_evidence"]["training_result_sha256"],
        "SEM-161 final result hash mismatch",
    )
    final_result = read_json(args.final_result)
    require(
        final_result.get("status") == "completed_final_checkpoint_frozen"
        and final_result.get("checkpoint_reload") == "passed",
        "SEM-161 final checkpoint is not frozen and reload-verified",
    )
    require(
        sha256(args.checkpoint)
        == config["final_evidence"]["checkpoint_sha256"]
        == final_result["checkpoint_sha256"],
        "SEM-161 final checkpoint hash mismatch",
    )
    require(
        sha256(args.sem151_result) == config["inputs"]["sem151_result_sha256"],
        "SEM-161 SEM-151 result hash mismatch",
    )
    sem151 = read_json(args.sem151_result)
    require(
        sem151.get("status") == "completed_developer_target_gate_passed",
        "SEM-161 requires the passing SEM-151 baseline",
    )

    split = read_json(args.split)
    names = require_exact_view_set(
        split["semantic_validation_labeled"],
        split["semantic_validation_labeled"],
        context="SEM-161 developer validation plan",
    )
    render_manifest_path = args.validation_render_root / "rendered_validation_manifest.json"
    require(
        render_manifest_path.is_file()
        and sha256(render_manifest_path)
        == config["inputs"]["rendered_validation_manifest_sha256"],
        "SEM-161 validation RGB manifest hash mismatch",
    )
    render_records = read_json(render_manifest_path)["rendered_images"]
    require(
        [record["name"] for record in render_records] == names,
        "SEM-161 rendered validation order mismatch",
    )
    for record in render_records:
        path = args.validation_render_root / record["path"]
        require(
            path.is_file() and sha256(path) == record["sha256"],
            f"SEM-161 validation RGB hash mismatch: {record['name']}",
        )
    baseline_paths = validate_prediction_manifest(
        args.sem151_root,
        names,
        manifest_sha256=config["inputs"]["sem151_prediction_manifest_sha256"],
        context="SEM-151",
    )
    pre_foundation_paths = validate_prediction_manifest(
        args.sem141_root,
        names,
        manifest_sha256=config["inputs"]["sem141_prediction_manifest_sha256"],
        context="SEM-141",
    )
    external_rgb_path = args.sem151_external_root / "rgb.png"
    external_base_path = args.sem151_external_root / "semantic.png"
    external_pre_path = args.sem141_external_root / "semantic.png"
    require(
        external_rgb_path.is_file()
        and sha256(external_rgb_path)
        == config["inputs"]["sem151_external_rgb_sha256"],
        "SEM-161 external RGB hash mismatch",
    )
    require(
        external_base_path.is_file()
        and sha256(external_base_path)
        == config["inputs"]["sem151_external_semantic_sha256"],
        "SEM-161 external SEM-151 semantic hash mismatch",
    )
    require(
        external_pre_path.is_file()
        and sha256(external_pre_path)
        == config["inputs"]["sem141_external_semantic_sha256"],
        "SEM-161 external SEM-141 semantic hash mismatch",
    )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("SEM-161 developer audit requires a visible CUDA GPU")
    device = torch.device("cuda:0")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    require(
        checkpoint.get("metadata", {}).get("experiment_id")
        == config["experiment_id"],
        "SEM-161 checkpoint metadata belongs to another experiment",
    )
    model = load_binary_model(args.model_dir, int(config["training"]["seed"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device).eval()
    crop = int(config["inference"]["crop_size"])
    stride = int(config["inference"]["stride"])

    args.output_dir.mkdir(parents=True, exist_ok=False)
    probability_root = args.output_dir / "deck_probabilities"
    prediction_root = args.output_dir / "predictions"
    records: list[dict[str, Any]] = []
    protected_exact = True
    # Materialise all 74 predictions before opening any developer label.
    for index, name in enumerate(names, start=1):
        rgb_path = args.validation_render_root / "images" / name
        probability = sliding_deck_probability(
            model,
            read_rgb(rgb_path),
            crop=crop,
            stride=stride,
            device=device,
            amp_dtype=amp_dtype,
        )
        base_path = baseline_paths[name]
        pre_path = pre_foundation_paths[name]
        base = class_map(base_path, context="SEM-161 SEM-151 baseline")
        pre = class_map(pre_path, context="SEM-161 SEM-141 pre-foundation")
        candidate, eligible, additions = recover_deck(
            base, pre, probability, threshold=threshold
        )
        protected_exact = protected_exact and protection_exact(
            base, pre, candidate, eligible
        )
        probability_path = probability_root / f"{Path(name).stem}.npz"
        prediction_path = prediction_root / name
        probability_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            probability_path, deck_probability=probability.astype(np.float16)
        )
        Image.fromarray(candidate).save(prediction_path)
        records.append(
            {
                "name": name,
                "frozen_rgb_sha256": sha256(rgb_path),
                "sem151_prediction_sha256": sha256(base_path),
                "sem141_prediction_sha256": sha256(pre_path),
                "deck_probability": str(
                    probability_path.relative_to(args.output_dir)
                ),
                "deck_probability_sha256": sha256(probability_path),
                "prediction": str(prediction_path.relative_to(args.output_dir)),
                "prediction_sha256": sha256(prediction_path),
                "eligible_pixels": int(np.count_nonzero(eligible)),
                "added_deck_pixels": int(np.count_nonzero(additions)),
            }
        )
        print(
            json.dumps(
                {"validation_inference": index, "total": len(names), "name": name}
            ),
            flush=True,
        )
    manifest_path = args.output_dir / "prediction_manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "status": "completed_predictions_before_developer_labels",
            "threshold": threshold,
            "predictions": records,
            "protected_predictions_exact": protected_exact,
            "developer_labels_opened_before_manifest": False,
            "hidden_test_opened": False,
            "unlabeled_images_opened": False,
            "real_rgb_input_opened": False,
        },
    )
    candidate_paths = {name: prediction_root / name for name in names}

    baseline_metrics, targets = metrics_for(names, args.labels_root, baseline_paths)
    candidate_metrics, _ = metrics_for(names, args.labels_root, candidate_paths)
    require(
        np.isclose(
            baseline_metrics["mIoU_with_background"],
            sem151["metrics"]["mIoU_with_background"],
            rtol=0.0,
            atol=1e-12,
        )
        and np.isclose(
            baseline_metrics["per_class"]["deck"]["iou"],
            sem151["metrics"]["per_class"]["deck"]["iou"],
            rtol=0.0,
            atol=1e-12,
        ),
        "SEM-161 did not reproduce the frozen SEM-151 baseline",
    )
    additions_by_target = np.zeros(5, dtype=np.int64)
    for name, target in zip(names, targets, strict=True):
        before = class_map(baseline_paths[name], context="SEM-161 addition baseline")
        after = class_map(candidate_paths[name], context="SEM-161 addition candidate")
        additions_by_target += np.bincount(
            target[(before == 0) & (after == 1)], minlength=5
        )
    per_view_comparison = {"improved": 0, "equal": 0, "regressed": 0}
    for before, after in zip(
        baseline_metrics["per_view"], candidate_metrics["per_view"], strict=True
    ):
        delta = float(after["per_class"]["deck"]["iou"]) - float(
            before["per_class"]["deck"]["iou"]
        )
        if delta > 1e-12:
            per_view_comparison["improved"] += 1
        elif delta < -1e-12:
            per_view_comparison["regressed"] += 1
        else:
            per_view_comparison["equal"] += 1

    gate_config = config["developer_target_gate"]
    target_gate = deck_gate(
        candidate=candidate_metrics,
        baseline=baseline_metrics,
        protected_predictions_exact=protected_exact,
        minimum_iou_gain=float(gate_config["minimum_deck_iou_gain"]),
        minimum_recall_gain=float(gate_config["minimum_deck_recall_gain"]),
        minimum_precision=float(gate_config["minimum_deck_precision"]),
        minimum_miou_gain=float(gate_config["minimum_mIoU_gain"]),
        maximum_background_fp_ratio=float(
            gate_config["maximum_background_to_deck_fp_ratio"]
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
            candidate_metrics["challenge_140_146"][
                "mean_mIoU_without_background"
            ]
        )
        >= float(
            baseline_metrics["challenge_140_146"][
                "mean_mIoU_without_background"
            ]
        )
        - float(gate_config["difficult_subset_drop_max"]),
        "oblique_158_161": float(
            candidate_metrics["oblique_158_161"]["mean_deck_iou"]
        )
        >= float(baseline_metrics["oblique_158_161"]["mean_deck_iou"])
        + float(gate_config["oblique_158_161_minimum_deck_iou_gain"]),
    }
    write_panels(
        names=list(config["outputs"]["validation_visual_samples"]),
        labels_root=args.labels_root,
        baseline=baseline_paths,
        candidate=candidate_paths,
        rgb_root=args.validation_render_root / "images",
        output_root=args.output_dir / "validation_panels",
    )

    external_base = class_map(external_base_path, context="SEM-161 external SEM-151")
    external_pre = class_map(external_pre_path, context="SEM-161 external SEM-141")
    external_probability = sliding_deck_probability(
        model,
        read_rgb(external_rgb_path),
        crop=crop,
        stride=stride,
        device=device,
        amp_dtype=amp_dtype,
    )
    external_prediction, external_eligible, external_additions = recover_deck(
        external_base, external_pre, external_probability, threshold=threshold
    )
    external_protected_exact = protection_exact(
        external_base,
        external_pre,
        external_prediction,
        external_eligible,
    )
    external_root = args.output_dir / "external_camera"
    external_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(external_rgb_path, external_root / "rgb.png")
    Image.fromarray(external_prediction).save(external_root / "semantic.png")
    np.save(external_root / "semantic_logits.npy", one_hot_logits(external_prediction))
    np.savez_compressed(
        external_root / "deck_probability.npz",
        deck_probability=external_probability.astype(np.float16),
    )
    external = {
        "input_pose_convention": "CAMERA_FROM_WORLD",
        "input_camera_model": "SIMPLE_RADIAL",
        "output_grid": "official_SIMPLE_RADIAL",
        "rgb_shape": list(rgb_image(external_root / "rgb.png").shape),
        "rgb_bytes_identical": external_rgb_path.read_bytes()
        == (external_root / "rgb.png").read_bytes(),
        "semantic_logits_shape": list(
            np.load(external_root / "semantic_logits.npy", mmap_mode="r").shape
        ),
        "semantic_png_shape": list(external_prediction.shape),
        "semantic_png_min": int(external_prediction.min()),
        "semantic_png_max": int(external_prediction.max()),
        "deck_probability_shape": list(external_probability.shape),
        "added_deck_pixels": int(np.count_nonzero(external_additions)),
        "protected_predictions_exact": external_protected_exact,
        "uses_reference_rgb_or_labels": False,
    }
    del model
    torch.cuda.empty_cache()
    engineering = {
        "checkpoint_reload": final_result["checkpoint_reload"] == "passed",
        "protected_predictions_exact": protected_exact,
        "external_protected_predictions_exact": external_protected_exact,
        "rgb_bytes_identical": external["rgb_bytes_identical"],
        "official_external_camera": external["uses_reference_rgb_or_labels"] is False
        and external["semantic_png_shape"] == [989, 1320]
        and external["semantic_logits_shape"] == [5, 989, 1320]
        and external["deck_probability_shape"] == [989, 1320],
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
        "final_training_result_sha256": sha256(args.final_result),
        "checkpoint_sha256": sha256(args.checkpoint),
        "prediction_manifest_sha256": sha256(manifest_path),
        "developer_labels_opened_after_all_predictions": True,
        "baseline_metrics": baseline_metrics,
        "metrics": candidate_metrics,
        "added_deck_pixels_by_ground_truth": {
            name: int(additions_by_target[index])
            for index, name in enumerate(
                ("background", "deck", "stay_cable", "tower", "foundation")
            )
        },
        "deck_iou_per_view_comparison": per_view_comparison,
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
                "mIoU": candidate_metrics["mIoU_with_background"],
                "deck": candidate_metrics["per_class"]["deck"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
