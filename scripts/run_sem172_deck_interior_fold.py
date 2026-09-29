#!/usr/bin/env python3
"""Train one SEM-172 fixed-interior-loss deck-binary spatial OOF fold."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_sem160_deck_binary_fold import (
    environment_summary,
    reload_checkpoint,
    require,
    save_checkpoint,
)
from src.semantic2d import fold_for_index, label_array, read_json, sha256, write_json
from src.semantic_deck_binary import (
    BINARY_CLASS_NAMES,
    binary_class_weights_from_labels,
    binary_target,
    load_binary_model,
    normalize_rgb,
    random_native_pair,
    read_rgb,
    sliding_deck_probability,
)
from src.semantic_deck_interior import interior_weighted_cross_entropy
from src.semantic_rendered_training import (
    select_rendered_training_views,
    validate_rendered_dataset_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--rendered-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "fold"), required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    require(
        config.get("experiment_id") == "SEM-172-DECK-INTERIOR-OOF",
        "SEM-172 trainer received another experiment config",
    )
    require(
        config.get("status") == "registered_not_run",
        "SEM-172 may run only from its registered configuration",
    )
    require(
        not args.output_dir.exists() or not any(args.output_dir.iterdir()),
        "SEM-172 refuses to overwrite a fold output",
    )
    require(
        sha256(args.split) == config["data"]["oof_split_sha256"],
        "SEM-172 OOF split differs from the registered hash",
    )
    oof_split = read_json(args.split)
    fold = fold_for_index(oof_split, args.fold)
    semantic_split_path = REPO_ROOT / config["data"]["semantic_split"]
    require(
        sha256(semantic_split_path) == config["data"]["semantic_split_sha256"],
        "SEM-172 semantic split differs from the registered hash",
    )
    semantic_split = read_json(semantic_split_path)
    training_names = select_rendered_training_views(
        semantic_split, expected_count=int(config["data"]["training_images"])
    )
    require(
        set(fold["train"]) | set(fold["oof"]) | set(fold["guard"])
        == set(training_names),
        "SEM-172 fold sets do not cover authorised training views",
    )
    require(
        not (set(fold["train"]) & set(fold["oof"]))
        and not (set(fold["train"]) & set(fold["guard"]))
        and not (set(fold["oof"]) & set(fold["guard"])),
        "SEM-172 fold train/oof/guard sets overlap",
    )
    manifest_path = args.rendered_root / "rendered_dataset_manifest.json"
    require(
        manifest_path.is_file()
        and sha256(manifest_path)
        == config["inputs"]["rendered_training_manifest_sha256"],
        "SEM-172 rendered training manifest differs from the frozen parent",
    )
    validate_rendered_dataset_manifest(read_json(manifest_path), training_names)
    require(args.labels_root.is_dir(), "SEM-172 labels root is missing")

    import torch
    import torch.nn.functional as functional

    if not torch.cuda.is_available():
        raise RuntimeError("SEM-172 requires a visible CUDA GPU")
    seed = int(config["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    if torch.cuda.is_bf16_supported():
        amp_name, amp_dtype = "bf16", torch.bfloat16
    else:
        amp_name, amp_dtype = "fp16", torch.float16
    crop = int(config["training"]["crop_size"])
    stride = int(config["inference"]["stride"])
    radius = int(config["training"]["deck_interior_erosion_radius_pixels"])
    multiplier = float(config["training"]["deck_interior_loss_multiplier"])
    iterations = int(
        config["training"][
            "smoke_iterations" if args.mode == "smoke" else "iterations_per_fold"
        ]
    )
    train_names = list(fold["train"])
    if args.mode == "smoke":
        train_names = train_names[: min(8, len(train_names))]
    eval_names = list(fold["oof"] if args.mode == "fold" else fold["oof"][:1])
    image_root = args.rendered_root / "images"
    for name in train_names + eval_names:
        require(
            (image_root / name).is_file() and (args.labels_root / name).is_file(),
            f"SEM-172 lacks an authorised RGB/label pair: {name}",
        )

    class_weights, class_counts = binary_class_weights_from_labels(
        args.labels_root, train_names
    )
    model = load_binary_model(args.model_dir, seed).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    weight_tensor = torch.as_tensor(class_weights, device=device)
    rng = random.Random(seed + args.fold * 100003)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    torch.cuda.reset_peak_memory_stats(device)
    losses: list[float] = []
    interior_fractions: list[float] = []
    started = time.perf_counter()
    model.train()
    for iteration in range(1, iterations + 1):
        name = train_names[rng.randrange(len(train_names))]
        target = binary_target(label_array(args.labels_root / name))
        rgb_crop, target_crop = random_native_pair(
            read_rgb(image_root / name), target, crop, rng
        )
        pixel_values = normalize_rgb(rgb_crop).unsqueeze(0).to(device)
        targets = torch.from_numpy(target_crop.astype(np.int64)).unsqueeze(0).to(
            device
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits = model(pixel_values=pixel_values).logits
            logits = functional.interpolate(
                logits,
                size=(crop, crop),
                mode="bilinear",
                align_corners=False,
            )
            loss, interior = interior_weighted_cross_entropy(
                logits,
                targets,
                class_weights=weight_tensor,
                radius=radius,
                multiplier=multiplier,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"SEM-172 produced a non-finite loss at iteration {iteration}"
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
        interior_fractions.append(float(interior.mean().detach().cpu()))
        if iteration == 1 or iteration % 100 == 0 or iteration == iterations:
            print(
                json.dumps(
                    {
                        "fold": args.fold,
                        "mode": args.mode,
                        "iteration": iteration,
                        "loss": losses[-1],
                        "deck_interior_fraction": interior_fractions[-1],
                    }
                ),
                flush=True,
            )
    torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    peak_mib = int(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
    metadata = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "mode": args.mode,
        "fold": args.fold,
        "iterations": iterations,
        "seed": seed,
        "binary_class_names": list(BINARY_CLASS_NAMES),
        "crop_size": crop,
        "deck_interior_erosion_radius_pixels": radius,
        "deck_interior_loss_multiplier": multiplier,
        "oof_split_sha256": sha256(args.split),
        "rendered_training_manifest_sha256": sha256(manifest_path),
        "model_source": config["model"],
    }
    checkpoint_path = args.output_dir / "checkpoints" / "deck_binary_final.pth"
    save_checkpoint(checkpoint_path, model, metadata)
    del model, optimizer
    torch.cuda.empty_cache()
    model, reload_metadata = reload_checkpoint(
        checkpoint_path, args.model_dir, seed, device
    )
    require(reload_metadata == metadata, "SEM-172 checkpoint metadata changed")

    prediction_entries: list[dict[str, Any]] = []
    inference_started = time.perf_counter()
    for index, name in enumerate(eval_names, start=1):
        probability = sliding_deck_probability(
            model,
            read_rgb(image_root / name),
            crop=crop,
            stride=stride,
            device=device,
            amp_dtype=amp_dtype,
        )
        probability_path = (
            args.output_dir / "oof_deck_probabilities" / f"{Path(name).stem}.npz"
        )
        raw_path = args.output_dir / "oof_binary_predictions" / name
        probability_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            probability_path, deck_probability=probability.astype(np.float16)
        )
        Image.fromarray(
            np.where(probability >= 0.5, 1, 0).astype(np.uint8)
        ).save(raw_path)
        prediction_entries.append(
            {
                "name": name,
                "fold": args.fold,
                "deck_probability": str(
                    probability_path.relative_to(args.output_dir)
                ),
                "deck_probability_sha256": sha256(probability_path),
                "raw_binary_prediction": str(raw_path.relative_to(args.output_dir)),
                "raw_binary_prediction_sha256": sha256(raw_path),
            }
        )
        print(
            json.dumps(
                {
                    "fold": args.fold,
                    "inference": index,
                    "total": len(eval_names),
                    "name": name,
                }
            ),
            flush=True,
        )
    torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_started
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": (
            "completed_preflight_not_quality_result"
            if args.mode == "smoke"
            else "completed_fold_oof"
        ),
        "mode": args.mode,
        "fold": args.fold,
        "iterations": iterations,
        "seed": seed,
        "actual_training_views": len(train_names),
        "oof_views_evaluated": len(eval_names),
        "binary_class_pixel_counts_training": class_counts.astype(int).tolist(),
        "binary_class_weights": class_weights.astype(float).tolist(),
        "deck_interior_erosion_radius_pixels": radius,
        "deck_interior_loss_multiplier": multiplier,
        "loss": {
            "first": losses[0],
            "last": losses[-1],
            "mean_last_100": float(np.mean(losses[-100:])),
        },
        "deck_interior_fraction": {
            "mean": float(np.mean(interior_fractions)),
            "last": interior_fractions[-1],
        },
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "gpu_hours_training": training_seconds / 3600.0,
        "peak_cuda_allocated_mib": peak_mib,
        "amp": amp_name,
        "checkpoint_reload": "passed",
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "environment": environment_summary(),
        "fixed_validation_opened": False,
        "unlabeled_rgb_opened": False,
        "real_rgb_input_opened": False,
        "scaffold_imported_or_trained": False,
    }
    write_json(args.output_dir / "prediction_manifest.json", prediction_entries)
    write_json(args.output_dir / "result.json", result)
    write_json(
        args.output_dir / "output_manifest.json",
        {
            "result_sha256": sha256(args.output_dir / "result.json"),
            "checkpoint_sha256": sha256(checkpoint_path),
            "prediction_manifest_sha256": sha256(
                args.output_dir / "prediction_manifest.json"
            ),
            "prediction_count": len(prediction_entries),
        },
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
