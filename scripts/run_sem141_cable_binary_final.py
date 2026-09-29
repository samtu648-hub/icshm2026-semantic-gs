#!/usr/bin/env python3
"""Train the all-219 final cable-binary checkpoint after SEM-140 qualifies."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.semantic2d import label_array, read_json, sha256, write_json
from src.semantic_cable_binary import (
    BINARY_CLASS_NAMES,
    binary_class_weights_from_labels,
    binary_target,
    load_binary_model,
    normalize_rgb,
    random_native_pair,
    read_rgb,
)
from src.semantic_rendered_training import (
    select_rendered_training_views,
    validate_rendered_dataset_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sem140-result", type=Path, required=True)
    parser.add_argument("--rendered-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    require(
        config.get("experiment_id") == "SEM-141-CABLE-BINARY-FINAL",
        "SEM-141 trainer received another experiment",
    )
    require(
        config.get("status") == "registered_not_run",
        "SEM-141 may run only from its registered configuration",
    )
    require(
        not args.output_dir.exists() or not any(args.output_dir.iterdir()),
        "SEM-141 refuses to overwrite final training output",
    )
    require(
        sha256(args.sem140_result) == config["parent_evidence"]["sem140_result_sha256"],
        "SEM-141 SEM-140 parent hash mismatch",
    )
    sem140 = read_json(args.sem140_result)
    require(
        sem140.get("status") == "completed_oof_gate_passed",
        "SEM-141 requires a passing SEM-140 OOF gate",
    )
    require(
        float(sem140["selected_threshold"])
        == float(config["fusion"]["stay_cable_threshold"]),
        "SEM-141 threshold differs from the OOF-frozen threshold",
    )
    split_path = REPO_ROOT / config["data"]["semantic_split"]
    require(
        sha256(split_path) == config["data"]["semantic_split_sha256"],
        "SEM-141 semantic split hash mismatch",
    )
    split = read_json(split_path)
    names = select_rendered_training_views(
        split, expected_count=int(config["data"]["training_images"])
    )
    manifest_path = args.rendered_root / "rendered_dataset_manifest.json"
    require(
        manifest_path.is_file()
        and sha256(manifest_path)
        == config["inputs"]["rendered_training_manifest_sha256"],
        "SEM-141 rendered training dataset differs from SEM-140",
    )
    validate_rendered_dataset_manifest(read_json(manifest_path), names)
    image_root = args.rendered_root / "images"
    for name in names:
        require(
            (image_root / name).is_file() and (args.labels_root / name).is_file(),
            f"SEM-141 lacks an authorised RGB/label pair: {name}",
        )

    import torch
    import torch.nn.functional as functional

    if not torch.cuda.is_available():
        raise RuntimeError("SEM-141 requires a visible CUDA GPU")
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
    iterations = int(config["training"]["iterations"])
    weights, counts = binary_class_weights_from_labels(args.labels_root, names)
    model = load_binary_model(args.model_dir, seed).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    weight_tensor = torch.as_tensor(weights, device=device)
    rng = random.Random(seed)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    losses: list[float] = []
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model.train()
    for iteration in range(1, iterations + 1):
        name = names[rng.randrange(len(names))]
        rgb = read_rgb(image_root / name)
        target = binary_target(label_array(args.labels_root / name))
        rgb_crop, target_crop = random_native_pair(rgb, target, crop, rng)
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
            loss = functional.cross_entropy(
                logits.float(), targets, weight=weight_tensor
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"SEM-141 produced a non-finite loss at iteration {iteration}"
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
        if iteration == 1 or iteration % 100 == 0 or iteration == iterations:
            print(
                json.dumps(
                    {"mode": "final", "iteration": iteration, "loss": losses[-1]}
                ),
                flush=True,
            )
    torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    peak_mib = int(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
    metadata = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "mode": "final",
        "iterations": iterations,
        "seed": seed,
        "binary_class_names": list(BINARY_CLASS_NAMES),
        "crop_size": crop,
        "sem140_result_sha256": sha256(args.sem140_result),
        "rendered_training_manifest_sha256": sha256(manifest_path),
        "training_view_count": len(names),
        "training_view_names": names,
        "model_source": config["model"],
    }
    checkpoint_path = args.output_dir / "checkpoints" / "cable_binary_final.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "metadata": metadata,
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    del model, optimizer
    torch.cuda.empty_cache()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    reloaded = load_binary_model(args.model_dir, seed)
    reloaded.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del reloaded
    require(checkpoint["metadata"] == metadata, "SEM-141 reload changed metadata")
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "completed_final_checkpoint_frozen",
        "iterations": iterations,
        "seed": seed,
        "training_view_count": len(names),
        "binary_class_pixel_counts_training": counts.astype(int).tolist(),
        "binary_class_weights": weights.astype(float).tolist(),
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
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "sem140_result_sha256": sha256(args.sem140_result),
        "rendered_training_manifest_sha256": sha256(manifest_path),
        "fixed_validation_opened": False,
        "unlabeled_rgb_opened": False,
        "real_rgb_input_opened": False,
        "hidden_test_opened": False,
        "scaffold_imported_or_trained": False,
    }
    write_json(
        args.output_dir / "training_input_manifest.json",
        {
            "experiment_id": config["experiment_id"],
            "training_view_names": names,
            "training_view_count": len(names),
            "rendered_training_manifest_sha256": sha256(manifest_path),
            "fixed_validation_opened": False,
            "unlabeled_images_opened": False,
            "real_rgb_input_opened": False,
        },
    )
    write_json(args.output_dir / "result.json", result)
    write_json(
        args.output_dir / "final_checkpoint_contract.json",
        {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "training_result_sha256": sha256(args.output_dir / "result.json"),
            "checkpoint_sha256": result["checkpoint_sha256"],
            "checkpoint_reload": "passed",
            "development_validation_may_open_after_this_record": True,
        },
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
