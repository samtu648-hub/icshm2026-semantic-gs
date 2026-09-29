#!/usr/bin/env python3
"""Train one frozen-protocol SegFormer-B2 OOF fold for SEM-005.

The script has no Scaffold imports.  It consumes the R1 prepared PINHOLE RGB
grid and SEM-001 label grid, trains one fold-specific checkpoint, and writes
only OOF probabilities and metrics to a Git-ignored output directory.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.semantic2d import (  # noqa: E402
    CLASS_COUNT,
    CLASS_NAMES,
    boundary_metrics,
    class_weights_from_labels,
    confusion_metrics,
    fold_for_index,
    label_array,
    read_json,
    sha256,
    stay_cable_structure,
    tile_starts,
    update_confusion,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--prepared-scene", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "fold"), required=True)
    parser.add_argument("--iterations", type=int, help="Registered-budget override; recorded in the result")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def load_model(model_dir: Path, seed: int):
    import torch
    from transformers import SegformerForSemanticSegmentation

    torch.manual_seed(seed)
    return SegformerForSemanticSegmentation.from_pretrained(
        model_dir,
        local_files_only=True,
        num_labels=CLASS_COUNT,
        id2label={index: name for index, name in enumerate(CLASS_NAMES)},
        label2id={name: index for index, name in enumerate(CLASS_NAMES)},
        ignore_mismatched_sizes=True,
    )


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Could not read RGB image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape != (989, 1320, 3):
        raise ValueError(f"{path}: expected 1320x989 RGB, got {image.shape}")
    return image


def normalize_crop(image: np.ndarray):
    import torch

    array = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return (array - mean) / std


def random_pair(image: np.ndarray, label: np.ndarray, crop: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    height, width = label.shape
    top = rng.randint(0, height - crop)
    left = rng.randint(0, width - crop)
    image_crop = image[top : top + crop, left : left + crop]
    label_crop = label[top : top + crop, left : left + crop]
    if rng.random() < 0.5:
        image_crop = image_crop[:, ::-1]
        label_crop = label_crop[:, ::-1]
    return np.ascontiguousarray(image_crop), np.ascontiguousarray(label_crop)


def sliding_probabilities(model: Any, image: np.ndarray, crop: int, stride: int, device: Any, amp_dtype: Any) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    score_sum = np.zeros((CLASS_COUNT, image.shape[0], image.shape[1]), dtype=np.float32)
    count = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for top in tile_starts(image.shape[0], crop, stride):
            for left in tile_starts(image.shape[1], crop, stride):
                tile = normalize_crop(image[top : top + crop, left : left + crop]).unsqueeze(0).to(device)
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(pixel_values=tile).logits
                    logits = functional.interpolate(logits, size=(crop, crop), mode="bilinear", align_corners=False)
                    probability = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
                score_sum[:, top : top + crop, left : left + crop] += probability
                count[top : top + crop, left : left + crop] += 1.0
    if np.any(count == 0):
        raise RuntimeError("Sliding-window inference left uncovered output pixels")
    probability = score_sum / count[None]
    probability /= np.maximum(probability.sum(axis=0, keepdims=True), np.finfo(np.float32).tiny)
    return probability


def save_checkpoint(path: Path, model: Any, metadata: dict[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": 1, "metadata": metadata, "model_state_dict": model.state_dict()}, path)


def reload_checkpoint(path: Path, model_dir: Path, seed: int, device: Any):
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = load_model(model_dir, seed)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval(), checkpoint["metadata"]


def environment_summary() -> dict[str, Any]:
    import torch
    import transformers

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
    }


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    split = read_json(args.split)
    fold = fold_for_index(split, args.fold)
    seed = int(config["training"]["seed"])
    iterations = args.iterations or int(config["training"]["smoke_iterations" if args.mode == "smoke" else "iterations_per_fold"])
    crop = int(config["training"]["crop_size"])
    stride = int(config["inference"]["stride"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")

    import torch
    import torch.nn.functional as functional

    if not torch.cuda.is_available():
        raise RuntimeError("SEM-005 requires a visible CUDA GPU; CPU fallback is not registered")
    seed_everything(seed)
    device = torch.device("cuda:0")
    if torch.cuda.is_bf16_supported():
        amp_name, amp_dtype = "bf16", torch.bfloat16
    else:
        amp_name, amp_dtype = "fp16", torch.float16
    train_names = list(fold["train"])
    if args.mode == "smoke":
        train_names = train_names[: min(8, len(train_names))]
    image_root = args.prepared_scene / "images"
    for name in train_names + list(fold["oof"][:1]):
        if not (image_root / name).is_file() or not (args.labels_root / name).is_file():
            raise FileNotFoundError(f"Missing paired PINHOLE RGB/label for {name}")
    class_weights, pixel_counts = class_weights_from_labels(args.labels_root, train_names)
    model = load_model(args.model_dir, seed).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    weight_tensor = torch.as_tensor(class_weights, device=device)
    rng = random.Random(seed + args.fold * 100003)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    torch.cuda.reset_peak_memory_stats(device)
    loss_history: list[float] = []
    started = time.perf_counter()
    model.train()
    for iteration in range(1, iterations + 1):
        name = train_names[rng.randrange(len(train_names))]
        image = read_rgb(image_root / name)
        label = label_array(args.labels_root / name)
        image_crop, label_crop = random_pair(image, label, crop, rng)
        pixel_values = normalize_crop(image_crop).unsqueeze(0).to(device)
        targets = torch.from_numpy(label_crop.astype(np.int64)).unsqueeze(0).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits = model(pixel_values=pixel_values).logits
            logits = functional.interpolate(logits, size=(crop, crop), mode="bilinear", align_corners=False)
            loss = functional.cross_entropy(logits.float(), targets, weight=weight_tensor)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at iteration {iteration}: {float(loss)}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_history.append(float(loss.detach().cpu()))
        if iteration == 1 or iteration % 100 == 0 or iteration == iterations:
            print(json.dumps({"fold": args.fold, "mode": args.mode, "iteration": iteration, "loss": loss_history[-1]}), flush=True)
    torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    peak_mib = int(torch.cuda.max_memory_allocated(device) / (1024 * 1024))

    metadata = {
        "experiment_id": config["experiment_id"],
        "mode": args.mode,
        "fold": args.fold,
        "iterations": iterations,
        "seed": seed,
        "class_names": list(CLASS_NAMES),
        "parent_split_sha256": sha256(Path(config["inputs"]["parent_split_repo_path"])),
        "oof_split_sha256": sha256(args.split),
        "model_source": config["model"],
    }
    checkpoint_path = args.output_dir / "checkpoints" / "segformer_b2_final.pth"
    save_checkpoint(checkpoint_path, model, metadata)
    del model
    torch.cuda.empty_cache()
    model, reload_metadata = reload_checkpoint(checkpoint_path, args.model_dir, seed, device)
    if reload_metadata != metadata:
        raise RuntimeError("Reloaded checkpoint metadata changed")

    eval_names = list(fold["oof"] if args.mode == "fold" else fold["oof"][:1])
    confusion = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    per_view: list[dict[str, Any]] = []
    prediction_entries: list[dict[str, Any]] = []
    inference_started = time.perf_counter()
    for index, name in enumerate(eval_names, start=1):
        image = read_rgb(image_root / name)
        target = label_array(args.labels_root / name)
        probability = sliding_probabilities(model, image, crop, stride, device, amp_dtype)
        prediction = probability.argmax(axis=0).astype(np.uint8)
        view_confusion = np.zeros_like(confusion)
        update_confusion(view_confusion, target, prediction)
        update_confusion(confusion, target, prediction)
        probability_path = args.output_dir / "oof_probabilities" / f"{Path(name).stem}.npz"
        prediction_path = args.output_dir / "oof_predictions" / name
        probability_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(probability_path, probabilities=probability.astype(np.float16))
        Image.fromarray(prediction, mode="L").save(prediction_path)
        per_view.append(
            {
                "name": name,
                "fold": args.fold,
                "metrics": confusion_metrics(view_confusion),
                "boundary": boundary_metrics(target, prediction),
                "stay_cable_structure": stay_cable_structure(target, prediction),
            }
        )
        prediction_entries.append(
            {
                "name": name,
                "fold": args.fold,
                "probabilities": str(probability_path.relative_to(args.output_dir)),
                "probabilities_sha256": sha256(probability_path),
                "prediction": str(prediction_path.relative_to(args.output_dir)),
                "prediction_sha256": sha256(prediction_path),
            }
        )
        print(json.dumps({"fold": args.fold, "oof_inference": index, "oof_total": len(eval_names), "name": name}), flush=True)
    torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_started
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "mode": args.mode,
        "fold": args.fold,
        "status": "completed_preflight_not_quality_result" if args.mode == "smoke" else "completed_fold_oof",
        "iterations": iterations,
        "seed": seed,
        "split_id": split["split_id"],
        "split_sha256": sha256(args.split),
        "fold_counts": fold["counts"],
        "actual_training_views": len(train_names),
        "oof_views_evaluated": len(eval_names),
        "class_pixel_counts_training": pixel_counts.astype(int).tolist(),
        "class_weights": class_weights.astype(float).tolist(),
        "loss": {"first": loss_history[0], "last": loss_history[-1], "mean_last_100": float(np.mean(loss_history[-100:]))},
        "metrics": confusion_metrics(confusion),
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
        "scaffold_imported_or_trained": False,
    }
    write_json(args.output_dir / "per_view_metrics.json", per_view)
    write_json(args.output_dir / "prediction_manifest.json", prediction_entries)
    write_json(args.output_dir / "result.json", result)
    write_json(
        args.output_dir / "output_manifest.json",
        {
            "result_sha256": sha256(args.output_dir / "result.json"),
            "checkpoint_sha256": sha256(checkpoint_path),
            "prediction_manifest_sha256": sha256(args.output_dir / "prediction_manifest.json"),
            "per_view_metrics_sha256": sha256(args.output_dir / "per_view_metrics.json"),
            "prediction_count": len(prediction_entries),
        },
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
