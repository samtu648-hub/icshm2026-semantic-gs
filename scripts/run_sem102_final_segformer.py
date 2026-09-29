#!/usr/bin/env python3
"""Freeze one all-training-view rendered-RGB SegFormer checkpoint for SEM-102.

This runner has no validation, unlabelled, external-camera, or real-RGB input.
It is deliberately separate from the subsequent 74-view evaluator so that the
checkpoint is written and reload-verified before any fixed validation data can
be opened.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_sem005_segformer as segformer
import run_sem080_structural_diag as s80

from src.semantic2d import class_weights_from_labels
from src.semantic_rendered_final import (
    require_exact_view_set,
    validate_sem101_parent_summary,
)
from src.semantic_rendered_training import (
    select_rendered_training_views,
    validate_rendered_dataset_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sem101-summary", type=Path, required=True)
    parser.add_argument("--rendered-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("final",), required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def config_sha256(path: Path) -> str:
    return s80.sha256(path)


def final_checkpoint_metadata(
    *,
    config: dict[str, Any],
    config_hash: str,
    rendered_manifest_hash: str,
    sem101_summary_hash: str,
    training_names: list[str],
    iterations: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "mode": "final",
        "checkpoint_selection": "final_3000_only_no_validation_selection",
        "iterations": iterations,
        "seed": int(config["training"]["seed"]),
        "class_names": list(segformer.CLASS_NAMES),
        "config_sha256": config_hash,
        "rendered_dataset_manifest_sha256": rendered_manifest_hash,
        "sem101_summary_sha256": sem101_summary_hash,
        "training_view_count": len(training_names),
        "training_view_names": training_names,
        "model_source": config["model"],
    }


def main() -> int:
    args = parse_args()
    config = s80.load_json(args.config)
    require(
        config.get("experiment_id") == "SEM-102-RENDERED-SEG-FINAL",
        "SEM-102 final trainer received another config",
    )
    require(
        config.get("status") == "registered_not_run",
        "SEM-102 final trainer requires its registered-not-run config",
    )
    require(args.mode == "final", "SEM-102 only permits final checkpoint training")
    require(
        not args.output_dir.exists() or not any(args.output_dir.iterdir()),
        "Refusing to overwrite an existing SEM-102 final-training output",
    )

    parent_summary = s80.load_json(args.sem101_summary)
    parent = config["parent_evidence"]
    sem101_summary_hash = s80.sha256(args.sem101_summary)
    require(
        sem101_summary_hash == parent["sem101_summary_result_sha256"],
        "SEM-102 received a different SEM-101 summary",
    )
    validate_sem101_parent_summary(
        parent_summary,
        expected_rendered_manifest_sha256=parent[
            "sem101_rendered_dataset_manifest_sha256"
        ],
    )

    split_path = REPO_ROOT / config["data"]["semantic_split"]
    require(
        s80.sha256(split_path) == config["data"]["semantic_split_sha256"],
        "SEM-102 semantic split differs from its registered hash",
    )
    split = s80.load_json(split_path)
    training_names = select_rendered_training_views(
        split, expected_count=int(config["data"]["training_images"])
    )
    require_exact_view_set(
        training_names,
        split["semantic_train_labeled"],
        context="SEM-102 final training views",
    )
    validation_names = set(split["semantic_validation_labeled"])
    require(
        not (set(training_names) & validation_names),
        "SEM-102 final training includes a fixed validation view",
    )
    require(args.labels_root.is_dir(), "SEM-102 labels root is missing")
    manifest_path = args.rendered_root / "rendered_dataset_manifest.json"
    require(manifest_path.is_file(), "SEM-102 rendered training manifest is missing")
    rendered_manifest_hash = s80.sha256(manifest_path)
    require(
        rendered_manifest_hash == parent["sem101_rendered_dataset_manifest_sha256"],
        "SEM-102 rendered training manifest differs from qualified SEM-101 input",
    )
    validate_rendered_dataset_manifest(s80.load_json(manifest_path), training_names)
    image_root = args.rendered_root / "images"
    for name in training_names:
        require(
            (image_root / name).is_file() and (args.labels_root / name).is_file(),
            f"SEM-102 lacks an authorised rendered RGB/label pair: {name}",
        )

    import torch
    from torch.nn import functional

    if not torch.cuda.is_available():
        raise RuntimeError("SEM-102 final trainer requires a visible CUDA GPU")
    config_hash = config_sha256(args.config)
    iterations = int(config["training"]["iterations"])
    crop = int(config["training"]["crop_size"])
    require(iterations == 3000, "SEM-102 final budget must remain 3,000 iterations")
    segformer.seed_everything(int(config["training"]["seed"]))
    device = torch.device("cuda:0")
    if torch.cuda.is_bf16_supported():
        amp_name, amp_dtype = "bf16", torch.bfloat16
    else:
        amp_name, amp_dtype = "fp16", torch.float16
    args.output_dir.mkdir(parents=True, exist_ok=False)
    class_weights, pixel_counts = class_weights_from_labels(
        args.labels_root, training_names
    )
    model = segformer.load_model(args.model_dir, int(config["training"]["seed"])).to(
        device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    weight_tensor = torch.as_tensor(class_weights, device=device)
    rng = random.Random(int(config["training"]["seed"]))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    torch.cuda.reset_peak_memory_stats(device)
    losses: list[float] = []
    started = time.perf_counter()
    model.train()
    for iteration in range(1, iterations + 1):
        name = training_names[rng.randrange(len(training_names))]
        image = segformer.read_rgb(image_root / name)
        label = segformer.label_array(args.labels_root / name)
        image_crop, label_crop = segformer.random_pair(image, label, crop, rng)
        pixel_values = segformer.normalize_crop(image_crop).unsqueeze(0).to(device)
        target = torch.from_numpy(label_crop.astype(np.int64)).unsqueeze(0).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits = model(pixel_values=pixel_values).logits
            logits = functional.interpolate(
                logits, size=(crop, crop), mode="bilinear", align_corners=False
            )
            loss = functional.cross_entropy(
                logits.float(), target, weight=weight_tensor
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"SEM-102 produced a non-finite loss at iteration {iteration}"
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
        if iteration == 1 or iteration % 100 == 0 or iteration == iterations:
            print(
                json.dumps(
                    {
                        "mode": "final",
                        "iteration": iteration,
                        "loss": losses[-1],
                    }
                ),
                flush=True,
            )
    torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    peak_mib = int(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
    metadata = final_checkpoint_metadata(
        config=config,
        config_hash=config_hash,
        rendered_manifest_hash=rendered_manifest_hash,
        sem101_summary_hash=sem101_summary_hash,
        training_names=training_names,
        iterations=iterations,
    )
    checkpoint_path = args.output_dir / "checkpoints" / "segformer_b2_final.pth"
    segformer.save_checkpoint(checkpoint_path, model, metadata)
    del model, optimizer
    torch.cuda.empty_cache()
    reloaded, reload_metadata = segformer.reload_checkpoint(
        checkpoint_path, args.model_dir, int(config["training"]["seed"]), device
    )
    del reloaded
    torch.cuda.empty_cache()
    require(reload_metadata == metadata, "SEM-102 reload changed checkpoint metadata")
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "completed_final_checkpoint_frozen",
        "mode": "final",
        "config_sha256": config_hash,
        "iterations": iterations,
        "seed": int(config["training"]["seed"]),
        "training_view_count": len(training_names),
        "training_view_names": training_names,
        "rendered_dataset_manifest_sha256": rendered_manifest_hash,
        "sem101_summary_sha256": sem101_summary_hash,
        "class_pixel_counts_training": pixel_counts.astype(int).tolist(),
        "class_weights": class_weights.astype(float).tolist(),
        "loss": {
            "first": losses[0],
            "last": losses[-1],
            "mean_last_100": float(np.mean(losses[-100:])),
        },
        "training_seconds": training_seconds,
        "gpu_hours_training": training_seconds / 3600.0,
        "peak_cuda_allocated_mib": peak_mib,
        "amp": amp_name,
        "checkpoint_reload": "passed",
        "checkpoint_sha256": s80.sha256(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "environment": segformer.environment_summary(),
        "fixed_validation_opened": False,
        "unlabeled_rgb_opened": False,
        "real_rgb_input_opened": False,
        "hidden_test_rgb_or_labels_opened": False,
        "external_camera_mode_present": False,
        "scaffold_imported_or_trained": False,
    }
    s80.write_json(
        args.output_dir / "training_input_manifest.json",
        {
            "experiment_id": config["experiment_id"],
            "rendered_dataset_manifest_sha256": rendered_manifest_hash,
            "training_view_names": training_names,
            "training_view_count": len(training_names),
            "fixed_validation_views_opened": False,
            "unlabeled_images_opened": False,
            "real_rgb_input_opened": False,
        },
    )
    s80.write_json(args.output_dir / "result.json", result)
    s80.write_json(
        args.output_dir / "final_checkpoint_contract.json",
        {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "config_sha256": config_hash,
            "final_training_result_sha256": s80.sha256(args.output_dir / "result.json"),
            "checkpoint_sha256": result["checkpoint_sha256"],
            "checkpoint_reload": "passed",
            "validation_may_open_after_this_record": True,
            "fixed_validation_opened_during_training": False,
        },
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
