#!/usr/bin/env python3
"""Run SEM-100's training-only real-RGB versus frozen-rendered-RGB preflight.

This runner has no validation, unlabeled-image, external-camera, optimizer, or
checkpoint-writing mode.  It reuses each selected training view's frozen OOF
SegFormer checkpoint and compares its existing real-RGB OOF output with a new
prediction from the frozen Scaffold rendering at the same official camera.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_sem001 as s0
import run_sem005_segformer as s5
import run_sem080_structural_diag as s80

from src.scaffold_semantic_sidecar import (
    ScaffoldSettings,
    configure_scaffold_import,
    frozen_rgb_hashes,
    frozen_rgb_named_parameters,
    load_frozen_scaffold,
    pipeline,
    render_frozen_rgb,
)
from src.semantic2d import (
    CLASS_COUNT,
    CLASS_NAMES,
    confusion_metrics,
    label_array,
    update_confusion,
)
from src.semantic_render_domain import (
    domain_metric_delta,
    render_domain_panel,
    rgb_psnr_db,
    select_preflight_oof_views,
    validate_probability_simplex,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--prepared-scene", type=Path, required=True)
    parser.add_argument("--rgb-run-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight",), required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def oof_record(teacher_root: Path, fold: int, name: str) -> tuple[dict[str, Any], Path]:
    fold_root = teacher_root / f"fold-{fold}"
    manifest = s80.load_json(fold_root / "prediction_manifest.json")
    matches = [
        record
        for record in manifest
        if record["name"] == name and int(record["fold"]) == fold
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one OOF record for fold {fold} / {name}")
    record = matches[0]
    probability_path = fold_root / record["probabilities"]
    if (
        not probability_path.is_file()
        or s80.sha256(probability_path) != record["probabilities_sha256"]
    ):
        raise ValueError(f"OOF probability hash mismatch for fold {fold} / {name}")
    return record, probability_path


def load_oof_teacher(
    *,
    teacher_root: Path,
    model_dir: Path,
    fold: int,
    expected_sha256: str,
    seed: int,
    device: Any,
) -> tuple[Any, dict[str, Any], Path]:
    checkpoint = (
        teacher_root / f"fold-{fold}" / "checkpoints" / "segformer_b2_final.pth"
    )
    if not checkpoint.is_file() or s80.sha256(checkpoint) != expected_sha256:
        raise ValueError(f"Frozen OOF checkpoint hash mismatch for fold {fold}")
    model, metadata = s5.reload_checkpoint(checkpoint, model_dir, seed, device)
    require(
        int(metadata["fold"]) == fold,
        f"OOF checkpoint metadata fold mismatch for {fold}",
    )
    require(int(metadata["seed"]) == seed, f"OOF checkpoint seed mismatch for {fold}")
    require(
        metadata["class_names"] == list(CLASS_NAMES),
        "OOF checkpoint class mapping changed",
    )
    return model, metadata, checkpoint


def save_probability(path: Path, probabilities: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, probabilities=probabilities.astype(np.float16))
    return s80.sha256(path)


def save_semantic_png(path: Path, prediction: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(prediction.astype(np.uint8), mode="L").save(path)
    return s80.sha256(path)


def main() -> int:
    args = parse_args()
    config = s80.load_json(args.config)
    require(
        config.get("experiment_id") == "SEM-100-RENDER-SEG-PF",
        "SEM-100 runner received another config",
    )
    require(
        config.get("status") == "registered_not_run",
        "SEM-100 preflight status was changed before execution",
    )
    require(args.mode == "preflight", "SEM-100 only has a registered preflight mode")
    require(
        not args.output_dir.exists() or not any(args.output_dir.iterdir()),
        "Refusing to overwrite a non-empty output directory",
    )
    split = s80.load_json(REPO_ROOT / config["data"]["semantic_split"])
    oof_split = s80.load_json(REPO_ROOT / config["data"]["oof_split"])
    train_names = set(split["semantic_train_labeled"])
    validation_names = set(split["semantic_validation_labeled"])
    plan = select_preflight_oof_views(
        oof_split,
        config["runtime"]["preflight_oof_views_by_fold"],
        expected_count=int(config["runtime"]["preflight_view_count"]),
    )
    require(
        {record["name"] for record in plan} <= train_names,
        "SEM-100 plan includes a non-training image",
    )
    require(
        not ({record["name"] for record in plan} & validation_names),
        "SEM-100 plan leaks validation",
    )
    require(
        not (
            {record["name"] for record in plan}
            & set(split.get("excluded_unlabeled", []))
        ),
        "SEM-100 plan leaks unlabeled images",
    )
    image_root = args.prepared_scene / "images"
    require(
        image_root.is_dir() and args.labels_root.is_dir(),
        "SEM-100 requires paired training RGB and labels",
    )
    for record in plan:
        require(
            (image_root / record["name"]).is_file(),
            f"Missing training RGB: {record['name']}",
        )
        require(
            (args.labels_root / record["name"]).is_file(),
            f"Missing training label: {record['name']}",
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    s80.write_json(
        args.output_dir / "training_only_access.json",
        {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "mode": args.mode,
            "planned_oof_training_views": plan,
            "fixed_validation_views_opened": False,
            "unlabeled_images_opened": False,
            "hidden_test_rgb_or_labels_opened": False,
            "external_camera_mode_present": False,
            "labels_read_after_all_real_and_rendered_predictions": True,
            "new_parameters_trained": False,
        },
    )

    import torch

    frozen = config["frozen_rgb"]
    settings = ScaffoldSettings(
        train_camera_count=int(frozen["train_camera_count"]),
        n_offsets=int(frozen["n_offsets"]),
    )
    point_cloud = (
        args.rgb_run_dir
        / "point_cloud"
        / f"iteration_{frozen['iteration']}"
        / "point_cloud.ply"
    )
    artifact_before = frozen_rgb_hashes(args.rgb_run_dir, int(frozen["iteration"]))
    configure_scaffold_import(args.source_root)
    scaffold = load_frozen_scaffold(args.source_root, point_cloud, settings)
    require(
        not any(
            parameter.requires_grad
            for _name, parameter in frozen_rgb_named_parameters(scaffold)
        ),
        "Frozen RGB model unexpectedly has trainable tensors",
    )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "SEM-100 requires CUDA for frozen Scaffold rendering and SegFormer inference"
        )
    device = torch.device("cuda:0")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    crop = int(config["frozen_teacher"]["inference"]["crop_size"])
    stride = int(config["frozen_teacher"]["inference"]["stride"])
    teacher_seed = int(config["frozen_teacher"]["seed"])
    checkpoints = config["frozen_teacher"]["fold_checkpoints_sha256"]
    pipe = pipeline()
    contexts: list[dict[str, Any]] = []
    loaded_checkpoints: dict[str, Any] = {}
    started = time.perf_counter()

    for fold in sorted({int(record["fold"]) for record in plan}):
        teacher, metadata, checkpoint_path = load_oof_teacher(
            teacher_root=args.teacher_root,
            model_dir=args.model_dir,
            fold=fold,
            expected_sha256=checkpoints[str(fold)],
            seed=teacher_seed,
            device=device,
        )
        loaded_checkpoints[str(fold)] = {
            "path": str(checkpoint_path),
            "sha256": checkpoints[str(fold)],
            "metadata": metadata,
        }
        for record in (entry for entry in plan if int(entry["fold"]) == fold):
            name = record["name"]
            manifest_record, real_probability_path = oof_record(
                args.teacher_root, fold, name
            )
            with np.load(real_probability_path) as archive:
                real_probabilities = archive["probabilities"].astype(np.float32)
            validate_probability_simplex(real_probabilities)
            real_prediction = real_probabilities.argmax(axis=0).astype(np.uint8)
            _selected, _camera, view = s0.official_view(
                args.data_root, name, int(frozen["appearance_camera_id"])
            )
            with torch.inference_mode():
                rendered_rgb = s0.tensor_to_rgb(render_frozen_rgb(view, scaffold, pipe))
            rendered_probabilities = s5.sliding_probabilities(
                teacher, rendered_rgb, crop, stride, device, amp_dtype
            )
            validate_probability_simplex(rendered_probabilities)
            rendered_prediction = rendered_probabilities.argmax(axis=0).astype(np.uint8)
            rendered_rgb_path = args.output_dir / "rendered_rgb" / name
            rendered_rgb_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rendered_rgb, mode="RGB").save(rendered_rgb_path)
            rendered_probability_path = (
                args.output_dir / "rendered_probabilities" / f"{Path(name).stem}.npz"
            )
            rendered_prediction_path = args.output_dir / "rendered_predictions" / name
            contexts.append(
                {
                    "name": name,
                    "fold": fold,
                    "real_probability_path": str(real_probability_path),
                    "real_probability_sha256": manifest_record["probabilities_sha256"],
                    "real_prediction": real_prediction,
                    "rendered_rgb": rendered_rgb,
                    "rendered_rgb_path": str(rendered_rgb_path),
                    "rendered_rgb_sha256": s80.sha256(rendered_rgb_path),
                    "rendered_probability_path": str(rendered_probability_path),
                    "rendered_probability_sha256": save_probability(
                        rendered_probability_path, rendered_probabilities
                    ),
                    "rendered_prediction": rendered_prediction,
                    "rendered_prediction_path": str(rendered_prediction_path),
                    "rendered_prediction_sha256": save_semantic_png(
                        rendered_prediction_path, rendered_prediction
                    ),
                }
            )
        del teacher
        torch.cuda.empty_cache()

    # All model predictions above intentionally complete before any label is read.
    real_confusion = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    rendered_confusion = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    rows: list[dict[str, Any]] = []
    visual_samples = set(config["outputs"]["training_visual_samples"])
    require(
        visual_samples <= {entry["name"] for entry in contexts},
        "Visual samples must be selected preflight views",
    )
    for context in contexts:
        name = context["name"]
        target = label_array(args.labels_root / name)
        real_rgb = s5.read_rgb(image_root / name)
        update_confusion(real_confusion, target, context["real_prediction"])
        update_confusion(rendered_confusion, target, context["rendered_prediction"])
        rows.append(
            {
                "view": name,
                "fold": context["fold"],
                "real_to_rendered_rgb_psnr_db": rgb_psnr_db(
                    real_rgb, context["rendered_rgb"]
                ),
                "real_stay_cable_pixels": int((context["real_prediction"] == 2).sum()),
                "rendered_stay_cable_pixels": int(
                    (context["rendered_prediction"] == 2).sum()
                ),
            }
        )
        if name in visual_samples:
            panel_path = args.output_dir / "training_render_domain_panels" / name
            panel_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(
                render_domain_panel(
                    real_rgb,
                    context["rendered_rgb"],
                    target,
                    context["real_prediction"],
                    context["rendered_prediction"],
                ),
                mode="RGB",
            ).save(panel_path)
            context["panel_path"] = str(panel_path)
            context["panel_sha256"] = s80.sha256(panel_path)

    real_metrics = confusion_metrics(real_confusion)
    rendered_metrics = confusion_metrics(rendered_confusion)
    rgb = s80._rgb_isolation(
        list(config["isolation"]["training_views"]),
        data_root=args.data_root,
        appearance_camera_id=int(frozen["appearance_camera_id"]),
        model=scaffold,
        pipe=pipe,
        rgb_run_dir=args.rgb_run_dir,
        iteration=int(frozen["iteration"]),
        output_dir=args.output_dir / "rgb_isolation",
    )
    require(rgb["rgb_artifact_hashes_equal"], "Frozen RGB artifact hash changed")
    require(rgb["rgb_png_hashes_equal"], "Frozen RGB rendering changed")
    require(
        rgb["rgb_max_absolute_pixel_difference"] == 0,
        "Frozen RGB pixel isolation failed",
    )
    artifact_after = frozen_rgb_hashes(args.rgb_run_dir, int(frozen["iteration"]))
    require(
        artifact_before == artifact_after, "Frozen RGB artifact changed during SEM-100"
    )
    s80._write_rows(args.output_dir / "training_per_view_render_domain.csv", rows)
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "completed_preflight_not_quality_evidence",
        "mode": args.mode,
        "git_head": s80.git_head(),
        "training_only_access": s80.load_json(
            args.output_dir / "training_only_access.json"
        ),
        "frozen_rgb_artifact_hashes_before": artifact_before,
        "frozen_rgb_artifact_hashes_after": artifact_after,
        "frozen_teacher_checkpoints": loaded_checkpoints,
        "real_rgb_oof_metrics": real_metrics,
        "frozen_rendered_rgb_metrics": rendered_metrics,
        "rendered_minus_real_metric_delta": domain_metric_delta(
            real_metrics, rendered_metrics
        ),
        "per_view": [
            {
                key: value
                for key, value in context.items()
                if key not in {"real_prediction", "rendered_prediction", "rendered_rgb"}
            }
            for context in contexts
        ],
        "rgb_isolation": rgb,
        "validation_metrics": None,
        "external_camera_evidence": None,
        "new_checkpoint_written": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    s80.write_json(args.output_dir / "result.json", result)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
