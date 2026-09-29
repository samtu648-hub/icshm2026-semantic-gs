#!/usr/bin/env python3
"""Run the frozen SEM-141 model and its 74-view developer-only audit."""

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

from src.semantic2d import read_json, sha256, write_json
from src.semantic_cable_binary import (
    fuse_cable_probability,
    load_binary_model,
    read_rgb,
    relative_gate,
    sliding_cable_probability,
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
    parser.add_argument("--sem140-result", type=Path, required=True)
    parser.add_argument("--final-result", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--validation-render-root", type=Path, required=True)
    parser.add_argument("--sem001-predictions", type=Path, required=True)
    parser.add_argument("--sem111-result", type=Path, required=True)
    parser.add_argument("--sem001-external-root", type=Path, required=True)
    parser.add_argument("--external-rgb-root", type=Path, required=True)
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


def one_hot_logits(prediction: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    probability = np.full((5, *prediction.shape), epsilon, dtype=np.float32)
    for class_id in range(5):
        probability[class_id, prediction == class_id] = 1.0 - epsilon * 4
    require(
        np.allclose(probability.sum(axis=0), 1.0, rtol=0.0, atol=1e-6),
        "SEM-141 external probabilities are not a simplex",
    )
    return np.log(np.clip(probability, np.finfo(np.float32).tiny, 1.0)).astype(
        np.float32
    )


def extended_metrics(
    confusions: list[np.ndarray], per_view: list[dict[str, Any]]
) -> dict[str, Any]:
    metrics = metrics_from_confusion(merge_confusions(confusions))
    worst_count = max(1, int(np.ceil(len(per_view) * 0.10)))
    worst = sorted(per_view, key=lambda record: record["mIoU_without_background"])[
        :worst_count
    ]
    challenge = [
        record for record in per_view if 140 <= int(Path(record["view"]).stem) <= 146
    ]
    metrics.update(
        {
            "view_count": len(per_view),
            "per_view": per_view,
            "worst_10_percent": {
                "count": worst_count,
                "views": [record["view"] for record in worst],
                "mean_mIoU_without_background": float(
                    np.mean([record["mIoU_without_background"] for record in worst])
                ),
            },
            "challenge_140_146": {
                "views": [record["view"] for record in challenge],
                "mean_mIoU_without_background": float(
                    np.mean(
                        [record["mIoU_without_background"] for record in challenge]
                    )
                ),
            },
        }
    )
    return metrics


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    require(
        config.get("experiment_id") == "SEM-141-CABLE-BINARY-FINAL",
        "SEM-141 evaluator received another experiment",
    )
    require(
        config.get("status") == "registered_not_run",
        "SEM-141 evaluator requires its registered configuration",
    )
    require(
        not args.output_dir.exists() or not any(args.output_dir.iterdir()),
        "SEM-141 refuses to overwrite developer audit output",
    )
    require(
        sha256(args.split) == config["data"]["semantic_split_sha256"],
        "SEM-141 validation split hash mismatch",
    )
    require(
        sha256(args.sem140_result) == config["parent_evidence"]["sem140_result_sha256"],
        "SEM-141 SEM-140 result hash mismatch",
    )
    sem140 = read_json(args.sem140_result)
    require(
        sem140.get("status") == "completed_oof_gate_passed",
        "SEM-141 requires a passing SEM-140 OOF gate",
    )
    threshold = float(config["fusion"]["stay_cable_threshold"])
    require(
        threshold == float(sem140["selected_threshold"]),
        "SEM-141 threshold is not the OOF-frozen threshold",
    )
    require(
        sha256(args.sem111_result) == config["inputs"]["sem111_result_sha256"],
        "SEM-141 SEM-111 baseline hash mismatch",
    )
    sem111 = read_json(args.sem111_result)
    require(
        sem111.get("status") == "completed_formal_retention_gate_passed",
        "SEM-141 requires the historical SEM-111 baseline",
    )
    require(
        sha256(args.final_result) == config["inputs"]["final_training_result_sha256"],
        "SEM-141 final training result hash mismatch",
    )
    final_result = read_json(args.final_result)
    require(
        final_result.get("status") == "completed_final_checkpoint_frozen"
        and final_result.get("checkpoint_reload") == "passed",
        "SEM-141 final checkpoint is not frozen and reload-verified",
    )
    require(
        sha256(args.checkpoint) == final_result["checkpoint_sha256"],
        "SEM-141 final checkpoint hash mismatch",
    )
    render_manifest_path = args.validation_render_root / "rendered_validation_manifest.json"
    require(
        render_manifest_path.is_file()
        and sha256(render_manifest_path)
        == config["inputs"]["rendered_validation_manifest_sha256"],
        "SEM-141 validation RGB manifest differs from the frozen parent",
    )
    split = read_json(args.split)
    names = require_exact_view_set(
        split["semantic_validation_labeled"],
        split["semantic_validation_labeled"],
        context="SEM-141 developer validation plan",
    )
    render_manifest = read_json(render_manifest_path)
    rendered_records = render_manifest["rendered_images"]
    rendered_names = [record["name"] for record in rendered_records]
    require(rendered_names == names, "SEM-141 rendered RGB view order mismatch")
    for record in rendered_records:
        path = args.validation_render_root / record["path"]
        require(
            path.is_file() and sha256(path) == record["sha256"],
            f"SEM-141 validation RGB hash mismatch: {record['name']}",
        )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("SEM-141 developer audit requires a visible CUDA GPU")
    device = torch.device("cuda:0")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = load_binary_model(args.model_dir, int(config["training"]["seed"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device).eval()
    crop = int(config["inference"]["crop_size"])
    stride = int(config["inference"]["stride"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    probability_root = args.output_dir / "cable_probabilities"
    prediction_root = args.output_dir / "predictions"
    manifest_records: list[dict[str, Any]] = []
    protected_exact = True
    # Materialise every prediction before opening any developer label.
    for index, name in enumerate(names, start=1):
        rgb_path = args.validation_render_root / "images" / name
        probability = sliding_cable_probability(
            model,
            read_rgb(rgb_path),
            crop=crop,
            stride=stride,
            device=device,
            amp_dtype=amp_dtype,
        )
        base_path = args.sem001_predictions / name
        base = class_map(base_path, context="SEM-141 SEM-001 validation output")
        fused, protected = fuse_cable_probability(
            base, probability, threshold=threshold
        )
        protected_exact = protected_exact and bool(
            np.array_equal(fused[protected], base[protected])
        )
        probability_path = probability_root / f"{Path(name).stem}.npz"
        prediction_path = prediction_root / name
        probability_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            probability_path, cable_probability=probability.astype(np.float16)
        )
        Image.fromarray(fused).save(prediction_path)
        manifest_records.append(
            {
                "name": name,
                "frozen_rgb_sha256": sha256(rgb_path),
                "sem001_prediction_sha256": sha256(base_path),
                "cable_probability": str(
                    probability_path.relative_to(args.output_dir)
                ),
                "cable_probability_sha256": sha256(probability_path),
                "prediction": str(prediction_path.relative_to(args.output_dir)),
                "prediction_sha256": sha256(prediction_path),
            }
        )
        print(
            json.dumps({"validation_inference": index, "total": len(names), "name": name}),
            flush=True,
        )
    write_json(
        args.output_dir / "prediction_manifest.json",
        {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "status": "completed_predictions_before_developer_labels",
            "threshold": threshold,
            "predictions": manifest_records,
            "protected_predictions_exact": protected_exact,
            "developer_labels_opened_before_manifest": False,
            "hidden_test_opened": False,
            "unlabeled_images_opened": False,
            "real_rgb_input_opened": False,
        },
    )

    confusions: list[np.ndarray] = []
    per_view: list[dict[str, Any]] = []
    panel_root = args.output_dir / "validation_panels"
    for name in names:
        target = class_map(args.labels_root / name, context="SEM-141 developer label")
        prediction = class_map(prediction_root / name, context="SEM-141 prediction")
        confusion = local_confusion(target, prediction)
        confusions.append(confusion)
        local = metrics_from_confusion(confusion)
        per_view.append({"view": name, **local})
        if name in config["outputs"]["validation_visual_samples"]:
            panel_root.mkdir(parents=True, exist_ok=True)
            rgb = read_rgb(args.validation_render_root / "images" / name)
            error = np.zeros((*target.shape, 3), dtype=np.uint8)
            error[target != prediction] = np.asarray([255, 48, 48], dtype=np.uint8)
            panel = np.concatenate(
                [rgb, semantic_rgb(target), semantic_rgb(prediction), error], axis=1
            )
            Image.fromarray(panel).save(panel_root / name)
    metrics = extended_metrics(confusions, per_view)
    target_gate = relative_gate(
        candidate=metrics,
        baseline=sem111["metrics"],
        protected_predictions_exact=protected_exact,
        minimum_iou_gain=float(config["developer_target_gate"]["minimum_stay_cable_iou_gain"]),
        minimum_recall_gain=float(
            config["developer_target_gate"]["minimum_stay_cable_recall_gain"]
        ),
        minimum_miou_gain=float(config["developer_target_gate"]["minimum_mIoU_gain"]),
        maximum_background_fp_ratio=float(
            config["developer_target_gate"]["maximum_background_false_positive_ratio"]
        ),
    )
    retention_checks = {
        "worst_10_percent": float(
            metrics["worst_10_percent"]["mean_mIoU_without_background"]
        )
        >= float(
            sem111["metrics"]["worst_10_percent"]["mean_mIoU_without_background"]
        )
        - float(config["developer_target_gate"]["difficult_subset_drop_max"]),
        "challenge_140_146": float(
            metrics["challenge_140_146"]["mean_mIoU_without_background"]
        )
        >= float(
            sem111["metrics"]["challenge_140_146"]["mean_mIoU_without_background"]
        )
        - float(config["developer_target_gate"]["difficult_subset_drop_max"]),
    }

    external_rgb_path = args.external_rgb_root / "rgb.png"
    external_base_path = args.sem001_external_root / "semantic.png"
    external_probability = sliding_cable_probability(
        model,
        read_rgb(external_rgb_path),
        crop=crop,
        stride=stride,
        device=device,
        amp_dtype=amp_dtype,
    )
    external_base = class_map(
        external_base_path, context="SEM-141 external SEM-001 output"
    )
    external_prediction, external_protected = fuse_cable_probability(
        external_base, external_probability, threshold=threshold
    )
    external_root = args.output_dir / "external_camera"
    external_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(external_rgb_path, external_root / "rgb.png")
    Image.fromarray(external_prediction).save(external_root / "semantic.png")
    np.save(external_root / "semantic_logits.npy", one_hot_logits(external_prediction))
    np.savez_compressed(
        external_root / "cable_probability.npz",
        cable_probability=external_probability.astype(np.float16),
    )
    external = {
        "input_pose_convention": "CAMERA_FROM_WORLD",
        "input_camera_model": "SIMPLE_RADIAL",
        "output_grid": "official_SIMPLE_RADIAL",
        "rgb_shape": list(read_rgb(external_root / "rgb.png").shape),
        "rgb_bytes_identical": external_rgb_path.read_bytes()
        == (external_root / "rgb.png").read_bytes(),
        "semantic_logits_shape": [5, 989, 1320],
        "semantic_png_shape": list(external_prediction.shape),
        "semantic_png_min": int(external_prediction.min()),
        "semantic_png_max": int(external_prediction.max()),
        "protected_predictions_exact": bool(
            np.array_equal(
                external_prediction[external_protected],
                external_base[external_protected],
            )
        ),
        "uses_reference_rgb_or_labels": False,
    }
    engineering = {
        "checkpoint_reload": final_result["checkpoint_reload"] == "passed",
        "protected_predictions_exact": protected_exact,
        "rgb_bytes_identical": external["rgb_bytes_identical"],
        "official_external_camera": external["uses_reference_rgb_or_labels"] is False
        and external["semantic_png_shape"] == [989, 1320]
        and external["semantic_logits_shape"] == [5, 989, 1320],
    }
    passed = bool(
        target_gate["passed"]
        and all(retention_checks.values())
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
        "validation_prediction_manifest_sha256": sha256(
            args.output_dir / "prediction_manifest.json"
        ),
        "developer_labels_opened_after_all_predictions": True,
        "metrics": metrics,
        "developer_target_gate": target_gate,
        "difficult_subset_retention_checks": retention_checks,
        "engineering_checks": engineering,
        "external_camera": external,
        "recommended_for_hidden_evaluation": passed,
        "fallback": config["fallback"],
        "panels": [
            str((panel_root / name).resolve())
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
