#!/usr/bin/env python3
"""Run SEM-102's once-only fixed-74-view 2D audit after checkpoint freeze.

All 74 predictions are written from frozen rendered RGB before this program
opens a validation label. Metrics are then a read-only aggregation over the
saved predictions and fixed labels; no checkpoint selection is possible here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_sem005_segformer as segformer
import run_sem080_structural_diag as s80

from src.semantic2d import (
    CLASS_COUNT,
    confusion_metrics,
    label_array,
    update_confusion,
)
from src.semantic_rendered_final import (
    require_exact_view_set,
    validate_final_training_result,
)
from src.semantic_rendered_training import rendered_segmentation_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--final-result", type=Path, required=True)
    parser.add_argument("--final-checkpoint", type=Path, required=True)
    parser.add_argument("--validation-rendered-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("frozen-74-audit",), required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_validation_render_manifest(
    manifest: dict[str, Any], *, expected_names: list[str], checkpoint_sha256: str
) -> None:
    if manifest.get("experiment_id") != "SEM-102-RENDERED-SEG-FINAL":
        raise ValueError(
            "SEM-102 validation RGB manifest belongs to another experiment"
        )
    if manifest.get("status") != "completed_frozen_validation_rgb_render":
        raise ValueError("SEM-102 validation RGB render is not finalized")
    names = [record.get("name") for record in manifest.get("rendered_images", [])]
    require_exact_view_set(
        names, expected_names, context="SEM-102 validation RGB manifest"
    )
    access = manifest.get("validation_render_access", {})
    if access.get("validation_labels_opened") is not False:
        raise ValueError("SEM-102 validation RGB renderer opened labels")
    if access.get("released_or_reference_rgb_opened") is not False:
        raise ValueError("SEM-102 validation RGB renderer opened reference RGB")
    if (
        access.get("final_checkpoint", {}).get("final_checkpoint_sha256")
        != checkpoint_sha256
    ):
        raise ValueError(
            "SEM-102 validation RGB was not unlocked by this final checkpoint"
        )
    isolation = manifest.get("rgb_isolation", {})
    if not (
        isolation.get("rgb_artifact_hashes_equal")
        and isolation.get("rgb_png_hashes_equal")
        and isolation.get("rgb_max_absolute_pixel_difference") == 0
    ):
        raise ValueError("SEM-102 validation RGB isolation evidence failed")


def per_view_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    local = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    update_confusion(local, target, prediction)
    return confusion_metrics(local)


def aggregate_metrics(
    *,
    names: list[str],
    prediction_root: Path,
    labels_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    combined = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    per_view: list[dict[str, Any]] = []
    for name in names:
        target = label_array(labels_root / name)
        prediction = np.asarray(Image.open(prediction_root / name), dtype=np.uint8)
        local = np.zeros_like(combined)
        update_confusion(local, target, prediction)
        update_confusion(combined, target, prediction)
        per_view.append({"view": name, **confusion_metrics(local)})
    metrics = confusion_metrics(combined)
    worst_count = max(1, math.ceil(len(per_view) * 0.10))
    worst = sorted(per_view, key=lambda item: item["mIoU_without_background"])[
        :worst_count
    ]
    challenge = [
        item for item in per_view if 140 <= int(Path(item["view"]).stem) <= 146
    ]
    metrics.update(
        {
            "view_count": len(names),
            "confusion_target_rows_prediction_columns": combined.astype(int).tolist(),
            "worst_10_percent": {
                "count": worst_count,
                "views": [item["view"] for item in worst],
                "mean_mIoU_without_background": float(
                    np.mean([item["mIoU_without_background"] for item in worst])
                ),
            },
            "challenge_140_146": {
                "views": [item["view"] for item in challenge],
                "mean_mIoU_without_background": float(
                    np.mean([item["mIoU_without_background"] for item in challenge])
                )
                if challenge
                else None,
            },
            "local_protocol_note": "Fixed full-image five-class confusion protocol, matching SEM-001; organizer aggregation is not yet published.",
        }
    )
    return metrics, per_view


def main() -> int:
    args = parse_args()
    config = s80.load_json(args.config)
    require(
        config.get("experiment_id") == "SEM-102-RENDERED-SEG-FINAL",
        "SEM-102 evaluator received another config",
    )
    require(
        config.get("status") == "registered_not_run",
        "SEM-102 evaluator requires its registered-not-run config",
    )
    require(args.mode == "frozen-74-audit", "SEM-102 only permits its fixed audit")
    require(
        not args.output_dir.exists() or not any(args.output_dir.iterdir()),
        "Refusing to overwrite an existing SEM-102 validation audit",
    )
    require(args.final_checkpoint.is_file(), "SEM-102 final checkpoint is missing")
    final_result = s80.load_json(args.final_result)
    checkpoint_sha256 = s80.sha256(args.final_checkpoint)
    validate_final_training_result(
        final_result,
        checkpoint_sha256=checkpoint_sha256,
        config_sha256=s80.sha256(args.config),
    )
    split_path = REPO_ROOT / config["data"]["semantic_split"]
    require(
        s80.sha256(split_path) == config["data"]["semantic_split_sha256"],
        "SEM-102 semantic split differs from its registered hash",
    )
    split = s80.load_json(split_path)
    validation_names = list(split["semantic_validation_labeled"])
    require(
        len(validation_names) == int(config["data"]["validation_images"])
        and len(validation_names) == len(set(validation_names)),
        "SEM-102 fixed validation audit view count differs from its registered plan",
    )
    require(args.labels_root.is_dir(), "SEM-102 validation labels root is missing")
    render_manifest_path = (
        args.validation_rendered_root / "rendered_validation_manifest.json"
    )
    require(
        render_manifest_path.is_file(), "SEM-102 validation RGB manifest is missing"
    )
    validate_validation_render_manifest(
        s80.load_json(render_manifest_path),
        expected_names=validation_names,
        checkpoint_sha256=checkpoint_sha256,
    )
    image_root = args.validation_rendered_root / "images"
    for name in validation_names:
        require(
            (image_root / name).is_file(),
            f"SEM-102 validation render is missing: {name}",
        )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("SEM-102 evaluator requires a visible CUDA GPU")
    device = torch.device("cuda:0")
    if torch.cuda.is_bf16_supported():
        amp_name, amp_dtype = "bf16", torch.bfloat16
    else:
        amp_name, amp_dtype = "fp16", torch.float16
    crop = int(config["inference"]["crop_size"])
    stride = int(config["inference"]["stride"])
    model, metadata = segformer.reload_checkpoint(
        args.final_checkpoint,
        args.model_dir,
        int(config["training"]["seed"]),
        device,
    )
    require(
        metadata.get("experiment_id") == config["experiment_id"],
        "SEM-102 loaded checkpoint metadata belongs to another experiment",
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    prediction_root = args.output_dir / "predictions"
    probability_root = args.output_dir / "probabilities"
    predictions: list[dict[str, Any]] = []
    started = time.perf_counter()
    # No labels are read in this loop. The final checkpoint is already frozen.
    for index, name in enumerate(validation_names, start=1):
        image = segformer.read_rgb(image_root / name)
        probability = segformer.sliding_probabilities(
            model, image, crop, stride, device, amp_dtype
        )
        require(
            np.all(np.isfinite(probability))
            and np.all(probability >= 0.0)
            and np.allclose(probability.sum(axis=0), 1.0, atol=2e-5),
            "SEM-102 produced invalid five-class probabilities",
        )
        prediction = probability.argmax(axis=0).astype(np.uint8)
        probability_path = probability_root / f"{Path(name).stem}.npz"
        prediction_path = prediction_root / name
        probability_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            probability_path, probabilities=probability.astype(np.float16)
        )
        Image.fromarray(prediction, mode="L").save(prediction_path)
        predictions.append(
            {
                "name": name,
                "probabilities": str(probability_path.relative_to(args.output_dir)),
                "probabilities_sha256": s80.sha256(probability_path),
                "prediction": str(prediction_path.relative_to(args.output_dir)),
                "prediction_sha256": s80.sha256(prediction_path),
            }
        )
        if index == 1 or index % 10 == 0 or index == len(validation_names):
            print(
                json.dumps(
                    {
                        "validation_inference": index,
                        "validation_total": len(validation_names),
                        "name": name,
                    }
                ),
                flush=True,
            )
    torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - started
    del model
    torch.cuda.empty_cache()
    s80.write_json(args.output_dir / "prediction_manifest.json", predictions)

    # Validation labels become accessible only after all model predictions exist.
    metrics, per_view = aggregate_metrics(
        names=validation_names,
        prediction_root=prediction_root,
        labels_root=args.labels_root,
    )
    panels: list[dict[str, str]] = []
    for name in config["outputs"]["validation_visual_samples"]:
        require(
            name in validation_names, "SEM-102 panel view is not in fixed validation"
        )
        rendered = np.asarray(
            Image.open(image_root / name).convert("RGB"), dtype=np.uint8
        )
        target = label_array(args.labels_root / name)
        prediction = np.asarray(Image.open(prediction_root / name), dtype=np.uint8)
        panel = rendered_segmentation_panel(rendered, target, prediction)
        panel_path = args.output_dir / "validation_panels" / name
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(panel).save(panel_path)
        panels.append(
            {
                "name": name,
                "path": str(panel_path.relative_to(args.output_dir)),
                "sha256": s80.sha256(panel_path),
            }
        )
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "completed_once_only_74_view_audit_pending_external_gate",
        "mode": args.mode,
        "config_sha256": s80.sha256(args.config),
        "final_training_result_sha256": s80.sha256(args.final_result),
        "final_checkpoint_sha256": checkpoint_sha256,
        "validation_render_manifest_sha256": s80.sha256(render_manifest_path),
        "validation_view_count": len(validation_names),
        "prediction_manifest_sha256": s80.sha256(
            args.output_dir / "prediction_manifest.json"
        ),
        "inference_seconds": inference_seconds,
        "amp": amp_name,
        "metrics": metrics,
        "per_view": per_view,
        "panels": panels,
        "validation_access": {
            "validation_labels_opened_after_all_predictions": True,
            "released_or_reference_rgb_opened": False,
            "unlabeled_images_opened": False,
            "hidden_test_rgb_or_labels_opened": False,
            "external_camera_mode_present": False,
        },
        "next_step": "Complete the independently callable official external-camera output gate, then aggregate fixed metrics and engineering evidence without retraining.",
    }
    s80.write_json(args.output_dir / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
