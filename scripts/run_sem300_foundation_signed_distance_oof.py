#!/usr/bin/env python3
"""Matched five-class CE versus a fixed two-sided foundation SDF objective."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_sem280_foundation_boundary_oof as base
from audit_sem260_error_budget import png
from src.semantic2d import CLASS_NAMES, read_json, sha256, write_json
from src.semantic_cable_binary import normalize_rgb
from src.semantic_foundation_signed_distance import (
    ARMS,
    foundation_signed_distance,
    five_class_signed_distance_loss,
)

EXP = "SEM-300-FOUNDATION-SIGNED-DISTANCE-OOF"


class FiveClassSignedDistanceModel(nn.Module):
    """A five-class SegFormer with one auxiliary distance head over its logits."""

    def __init__(self, model_dir: Path, seed: int):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation

        torch.manual_seed(seed)
        self.segmenter = SegformerForSemanticSegmentation.from_pretrained(
            model_dir,
            local_files_only=True,
            num_labels=5,
            id2label=dict(enumerate(CLASS_NAMES)),
            label2id={name: index for index, name in enumerate(CLASS_NAMES)},
            ignore_mismatched_sizes=True,
        )
        self.signed_distance_head = nn.Conv2d(5, 1, kernel_size=1)

    def forward(self, *, pixel_values: torch.Tensor) -> SimpleNamespace:
        logits = self.segmenter(pixel_values=pixel_values).logits
        return SimpleNamespace(logits=logits, signed_distance=self.signed_distance_head(logits))


def initialize(model_dir: Path, seed: int) -> FiveClassSignedDistanceModel:
    return FiveClassSignedDistanceModel(model_dir, seed)


def fit(output_dir, artifact_root, config, fold, arm, plan, rgb_paths, labels_root, smoke, started):
    folder = output_dir / f"fold{fold['fold']}_{arm}"
    folder.mkdir()
    names = sorted({row["name"] for row in plan}) if smoke else sorted(fold["train"])
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

    counts = np.zeros(5, np.int64)
    cache = {}
    access = []
    truncation = int(config["training"]["distance_truncation_pixels"])
    auxiliary_weight = float(config["training"]["arm_distance_loss_weights"][arm])
    beta = float(config["training"]["smooth_l1_beta"])
    for name in names:
        if name not in fold["train"] or name in fold["oof"] or name in fold["guard"]:
            raise ValueError("Forbidden training label access")
        rgb, target = png(rgb_paths[name][0]), png(labels_root / name)
        if rgb.shape != (989, 1320, 3) or target.shape != (989, 1320):
            raise ValueError("Native training grid mismatch")
        distance = foundation_signed_distance(target, truncation_pixels=truncation)
        counts += np.bincount(target.ravel(), minlength=5)
        cache[name] = (rgb, target, distance)
        access.append(
            {
                "name": name,
                "label_sha256": sha256(labels_root / name),
                "rgb_sha256": rgb_paths[name][1],
                "signed_distance_min": float(distance.min()),
                "signed_distance_max": float(distance.max()),
            }
        )
    if np.any(counts <= 0):
        raise ValueError("Missing training class")
    weights = np.clip(np.median(counts.astype(np.float64)) / counts, 0.05, 20).astype(np.float32)
    model = initialize(artifact_root / config["parents"]["model_config"], seed).cuda().train()
    initial_state_sha256 = base.state_hash(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    class_weights = torch.from_numpy(weights).cuda()
    crop = int(config["training"]["crop"])
    losses, ce_losses, distance_losses = [], [], []
    begin = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step, row in enumerate(plan, 1):
        rgb, target, distance = cache[row["name"]]
        y0, x0 = int(row["top"]), int(row["left"])
        image_crop = rgb[y0 : y0 + crop, x0 : x0 + crop]
        target_crop = target[y0 : y0 + crop, x0 : x0 + crop]
        distance_crop = distance[y0 : y0 + crop, x0 : x0 + crop]
        if row["flip"]:
            image_crop = image_crop[:, ::-1]
            target_crop = target_crop[:, ::-1]
            distance_crop = distance_crop[:, ::-1]
        pixels = normalize_rgb(np.ascontiguousarray(image_crop)).unsqueeze(0).cuda()
        targets = torch.from_numpy(target_crop.copy().astype(np.int64)).unsqueeze(0).cuda()
        distance_targets = torch.from_numpy(distance_crop.copy()).unsqueeze(0).unsqueeze(0).cuda()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(pixel_values=pixels)
            logits = functional.interpolate(output.logits, (crop, crop), mode="bilinear", align_corners=False)
            signed_distance = functional.interpolate(
                output.signed_distance, (crop, crop), mode="bilinear", align_corners=False
            )
            loss, parts = five_class_signed_distance_loss(
                logits,
                signed_distance,
                targets,
                distance_targets,
                class_weights=class_weights,
                auxiliary_weight=auxiliary_weight,
                smooth_l1_beta=beta,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        ce_losses.append(float(parts["cross_entropy"]))
        distance_losses.append(float(parts["signed_distance"]))
        if step == 1 or step % 250 == 0 or step == len(plan):
            print(
                json.dumps(
                    {
                        "stage": "training",
                        "fold": fold["fold"],
                        "arm": arm,
                        "step": step,
                        "total": len(plan),
                        "loss": losses[-1],
                        "cross_entropy": ce_losses[-1],
                        "signed_distance": distance_losses[-1],
                        "seconds": time.perf_counter() - begin,
                    }
                ),
                flush=True,
            )
            base.resource_gate(output_dir, config, started)

    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - begin
    peak_mib = torch.cuda.max_memory_allocated() / 2**20
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        replay = model(pixel_values=pixels)
        replay_logits, replay_distance = replay.logits.cpu(), replay.signed_distance.cpu()
    metadata = {
        "experiment_id": EXP,
        "fold": fold["fold"],
        "arm": arm,
        "seed": seed,
        "iterations": len(plan),
        "train": names,
        "outer_train": fold["train"],
        "oof": fold["oof"],
        "guard": fold["guard"],
        "actual_label_access": access,
        "actual_optimization_names": sorted({row["name"] for row in plan}),
        "sampling_plan_sha256": sha256(output_dir / f"fold{fold['fold']}_sampling_plan.json"),
        "config_sha256": config["_sha256"],
        "oof_split_sha256": config["oof_split_sha256"],
        "initialization": config["_initialization"],
        "initial_state_sha256": initial_state_sha256,
        "class_counts": counts.tolist(),
        "class_weights": weights.tolist(),
        "distance_truncation_pixels": truncation,
        "distance_loss_weight": auxiliary_weight,
        "smooth_l1_beta": beta,
        "mode": "smoke" if smoke else "full",
        "training_task": "five_class_with_optional_signed_distance_auxiliary",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }
    checkpoint = folder / "model.pth"
    torch.save({"metadata": metadata, "model_state_dict": model.state_dict()}, checkpoint)
    del model, optimizer, cache, logits, signed_distance, loss
    torch.cuda.empty_cache()

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    fresh = initialize(artifact_root / config["parents"]["model_config"], seed).cuda().eval().requires_grad_(False)
    fresh.load_state_dict(payload["model_state_dict"], strict=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        actual = fresh(pixel_values=pixels)
        actual_logits, actual_distance = actual.logits.cpu(), actual.signed_distance.cpu()
    if not torch.equal(actual_logits, replay_logits) or not torch.equal(actual_distance, replay_distance):
        raise ValueError("Strict checkpoint reload mismatch")
    if payload["metadata"] != metadata:
        raise ValueError("Checkpoint metadata mismatch")
    result = {
        "path": str(checkpoint),
        "sha256": sha256(checkpoint),
        "metadata": metadata,
        "reload_exact": True,
        "seconds": training_seconds,
        "peak_cuda_allocated_mib": peak_mib,
        "loss": {
            "first": losses[0],
            "last": losses[-1],
            "last100": float(np.mean(losses[-100:])),
            "cross_entropy_last100": float(np.mean(ce_losses[-100:])),
            "signed_distance_last100": float(np.mean(distance_losses[-100:])),
        },
    }
    write_json(folder / "training_result.json", result)
    del payload, pixels, targets, distance_targets
    torch.cuda.empty_cache()
    return fresh, result


def run(args, output_dir: Path, started: float) -> None:
    original = (base.EXP, base.ARMS, base.fit, base.__file__)
    try:
        base.EXP = EXP
        base.ARMS = ARMS
        base.fit = fit
        base.__file__ = str(Path(__file__).resolve())
        base.run(args, output_dir, started)
        provenance = {
            "experiment_id": EXP,
            "sources": {
                "scripts/run_sem300_foundation_signed_distance_oof.py": sha256(Path(__file__)),
                "src/semantic_foundation_signed_distance.py": sha256(ROOT / "src/semantic_foundation_signed_distance.py"),
            },
            "note": "The SEM-280 runner supplies shared OOF/fusion infrastructure; this manifest records the SEM-300-specific model and loss sources.",
        }
        write_json(output_dir / "sem300_source_manifest.json", provenance)
        result = read_json(output_dir / "result.json")
        result["sem300_source_manifest_sha256"] = sha256(output_dir / "sem300_source_manifest.json")
        write_json(output_dir / "result.json", result)
    finally:
        base.EXP, base.ARMS, base.fit, base.__file__ = original


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/runs/SEM-300-FOUNDATION-SIGNED-DISTANCE-OOF/config.yaml",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        run(args, output_dir, started)
    except Exception as error:
        write_json(
            output_dir / "failure.json",
            {
                "experiment_id": EXP,
                "status": "incomplete_not_quality_result",
                "error": repr(error),
                "seconds": time.perf_counter() - started,
            },
        )
        raise


if __name__ == "__main__":
    main()
