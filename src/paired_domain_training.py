"""Matched two-branch Mask2Former training for render/photo domain supervision.

The control evaluates two independently sampled Mask2Former loss graphs on the
same reconstruction crop.  The candidate changes only the second branch input
to the registered real-photo crop.  Both branches use the same label, crop and
horizontal flip, and their losses are averaged before one optimizer update.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


IMAGE_SHAPE = (989, 1320)
VALID_ARMS = {"render_render", "render_photo"}


def paired_crop(image: np.ndarray, trace: dict[str, Any], crop: int) -> np.ndarray:
    """Apply a frozen native crop/flip trace to a second aligned RGB source."""
    if image.dtype != np.uint8 or image.shape != (*IMAGE_SHAPE, 3):
        raise ValueError("Paired source must be native uint8 989x1320 RGB")
    top, left = int(trace["top"]), int(trace["left"])
    if (
        crop <= 0
        or top < 0
        or left < 0
        or top + crop > IMAGE_SHAPE[0]
        or left + crop > IMAGE_SHAPE[1]
    ):
        raise ValueError("Frozen paired crop does not fit the native image")
    out = image[top : top + crop, left : left + crop]
    if bool(trace["flip"]):
        out = out[:, ::-1]
    return np.ascontiguousarray(out)


def validate_domain_sources(
    render_images: dict[str, np.ndarray],
    photo_images: dict[str, np.ndarray],
    names: list[str],
) -> dict[str, Any]:
    """Validate exact name/grid pairing without treating the photo as a label."""
    expected = set(names)
    if not names or len(names) != len(expected) or names != sorted(names):
        raise ValueError("Training names must be sorted, unique and non-empty")
    if set(render_images) != expected or set(photo_images) != expected:
        raise ValueError("Render/photo source names do not match the training role")
    differing = 0
    for name in names:
        render, photo = render_images[name], photo_images[name]
        if render.dtype != np.uint8 or render.shape != (*IMAGE_SHAPE, 3):
            raise ValueError(f"Invalid render source for {name}")
        if photo.dtype != np.uint8 or photo.shape != (*IMAGE_SHAPE, 3):
            raise ValueError(f"Invalid photo source for {name}")
        differing += int(not np.array_equal(render, photo))
    if differing != len(names):
        raise ValueError("Every registered photo must differ from its reconstruction")
    return {
        "paired_names": len(names),
        "native_shape": [IMAGE_SHAPE[1], IMAGE_SHAPE[0]],
        "all_pairs_pixel_different": True,
    }


def _rng_sha256(state: Any) -> str:
    return hashlib.sha256(state.detach().cpu().numpy().tobytes()).hexdigest()


def paired_amp_update(
    model: Any,
    optimizer: Any,
    scaler: Any,
    first_image: Any,
    second_image: Any,
    instances: Any,
    *,
    max_attempts: int = 12,
    clip: float = 0.01,
) -> dict[str, Any]:
    """Average two stochastic author-loss graphs in one bounded AMP update."""
    import torch

    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    skips: list[dict[str, Any]] = []
    for attempt in range(max_attempts):
        optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        branch_rows: list[dict[str, float]] = []
        branch_totals = []
        for image in (first_image, second_image):
            with torch.autocast("cuda", dtype=torch.float16):
                losses = model([{"image": image, "instances": instances}])
                total = sum(losses.values())
            if not torch.isfinite(total):
                raise FloatingPointError("Nonfinite unscaled paired native loss")
            scaler.scale(total * 0.5).backward()
            branch_rows.append({key: float(value.detach()) for key, value in losses.items()})
            branch_totals.append(float(total.detach()))
            del losses, total
        scaler.unscale_(optimizer)
        bad = [
            name
            for name, value in model.named_parameters()
            if value.grad is not None and not torch.isfinite(value.grad).all()
        ]
        if bad:
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skips.append(
                {
                    "attempt": attempt + 1,
                    "scale_before": before,
                    "scale_after": scaler.get_scale(),
                    "bad_gradient_count": len(bad),
                    "first_bad_parameters": bad[:8],
                }
            )
            if scaler.get_scale() >= before:
                raise FloatingPointError("AMP did not skip/back off paired nonfinite gradient")
            continue
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), clip, error_if_nonfinite=True
        )
        scaler.step(optimizer)
        scaler.update()
        return {
            "loss": float(sum(branch_totals) * 0.5),
            "branch_loss": branch_totals,
            "branch_losses": branch_rows,
            "grad_norm": float(norm),
            "scale": scaler.get_scale(),
            "skips": skips,
            "attempt_start_cpu_rng_sha256": _rng_sha256(cpu_rng),
            "attempt_start_cuda_rng_sha256": _rng_sha256(cuda_rng),
        }
    raise FloatingPointError("AMP exhausted 12 paired attempts; preserve failed run")


def _save_probability(out: Path, name: str, probability: np.ndarray) -> None:
    from PIL import Image
    from src.quality_experts import validate_probability

    validate_probability(probability, probability.shape[0], IMAGE_SHAPE)
    (out / "probabilities").mkdir(exist_ok=True, parents=True)
    (out / "predictions").mkdir(exist_ok=True)
    np.savez_compressed(
        out / "probabilities" / (Path(name).stem + ".npz"),
        probabilities=probability,
    )
    Image.fromarray(probability.argmax(0).astype(np.uint8)).save(
        out / "predictions" / name
    )


def paired_mask2former(a: Any, out: Path, result: dict[str, Any], labels: Any, cfg: dict[str, Any]) -> None:
    """Train the matched control or photo-domain candidate."""
    import torch
    from detectron2.structures import Instances
    from src.quality_mask2former import (
        author_optimizer,
        class_masks,
        config_for_model,
        load_official_model,
        native_predict,
    )
    from src.quality_mask2former_training import poly_factor, sampled_crop, save_checkpoint
    from src.quality_protocol import sha256
    from src.semantic_cable_binary import read_rgb

    arm = cfg["dual_domain_arm"]
    if arm not in VALID_ARMS:
        raise ValueError(f"Unknown paired-domain arm: {arm}")
    steps = int(cfg.get("paired_steps", 6000))
    checkpoint_steps = tuple(int(value) for value in cfg.get("checkpoint_steps", [3000, 6000]))
    crop = int(cfg.get("paired_crop", 768))
    root = Path(__file__).resolve().parents[1]
    runtime = root / "outputs/quality/mask2former/SEM-340"
    model_config = config_for_model(
        runtime / "vendor/Mask2Former",
        "configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml",
        False,
    )
    model_config.defrost()
    model_config.DATASETS.TRAIN = ("sem341_rendered_training",)
    model_config.SOLVER.MAX_ITER = 6000
    model_config.SOLVER.POLY_LR_POWER = 0.9
    model_config.freeze()
    (out / "resolved_model.yaml").write_text(model_config.dump(), encoding="utf-8")
    model, loaded = load_official_model(
        model_config, runtime / "weights/model_final_6b4a3a.pkl"
    )
    result["pretrained_load"] = loaded
    if sum(parameter.numel() for parameter in model.parameters()) != 215451514:
        raise ValueError("Unexpected Mask2Former parameter count")
    model.cuda()
    optimizer = author_optimizer(model, model_config)
    rates = [group["lr"] for group in optimizer.param_groups]
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    names = sorted(labels.allowed)
    render_images = {
        name: read_rgb(Path(a.batch) / "export/images" / name) for name in names
    }
    photo_root = Path(cfg["paired_photo_root"])
    if not photo_root.is_absolute():
        photo_root = root / photo_root
    photo_images = {name: read_rgb(photo_root / name) for name in names}
    source_validation = validate_domain_sources(render_images, photo_images, names)
    targets = {name: labels.read(name) for name in names}
    source_record = Path(a.batch) / "paired-source.json"
    if source_record.is_file():
        result["binding"]["paired_source_sha256"] = sha256(source_record)
    rng = random.Random(cfg["seed"])
    model.train()
    started = time.monotonic()
    result.update(
        amp_skips=0,
        paired_domain={
            "arm": arm,
            "first_branch": "render",
            "second_branch": "render" if arm == "render_render" else "photo",
            "branch_loss_weights": [0.5, 0.5],
            "shared_view_crop_flip_and_label": True,
            "independent_author_loss_rng_per_branch": True,
            **source_validation,
        },
    )
    with (out / "training.jsonl").open("x", encoding="utf-8") as journal:
        for step in range(1, steps + 1):
            render_crop, target_crop, trace = sampled_crop(
                rng, names, render_images, targets, crop
            )
            second_source = render_images[trace["name"]] if arm == "render_render" else photo_images[trace["name"]]
            second_crop = paired_crop(second_source, trace, crop)
            if arm == "render_render" and not np.array_equal(render_crop, second_crop):
                raise ValueError("Matched control branches must receive identical pixels")
            if arm == "render_photo" and np.array_equal(render_crop, second_crop):
                raise ValueError("Photo candidate branches unexpectedly receive identical pixels")
            ids, masks = class_masks(target_crop)
            instances = Instances(tuple(target_crop.shape))
            instances.gt_classes = torch.from_numpy(ids)
            instances.gt_masks = torch.from_numpy(masks)
            first_tensor = torch.from_numpy(
                render_crop.transpose(2, 0, 1).copy()
            ).float()
            second_tensor = torch.from_numpy(
                second_crop.transpose(2, 0, 1).copy()
            ).float()
            factor = poly_factor(step - 1)
            for group, rate in zip(optimizer.param_groups, rates):
                group["lr"] = rate * factor
            row = paired_amp_update(
                model,
                optimizer,
                scaler,
                first_tensor,
                second_tensor,
                instances,
                max_attempts=12,
                clip=0.01,
            )
            row.update(
                step=step,
                sample=trace,
                domain_arm=arm,
                lr_factor=factor,
                elapsed_seconds=time.monotonic() - started,
            )
            journal.write(json.dumps(row) + "\n")
            result["trained_steps"] = step
            result["amp_skips"] += len(row["skips"])
            if step % 100 == 0 or step == 1 or step == steps:
                journal.flush()
                print(
                    json.dumps(
                        {
                            "stage": "paired_mask2former",
                            "arm": arm,
                            "step": step,
                            "amp_skips": result["amp_skips"],
                            "seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
            if step in checkpoint_steps:
                save_checkpoint(
                    model,
                    SimpleNamespace(binding=result["binding"], access=result["access"]),
                    step,
                    out / f"model-{step}.pt",
                )
    result["training_seconds"] = time.monotonic() - started
    result["paired_steps"] = steps
    if checkpoint_steps:
        del optimizer, scaler, render_images, photo_images, targets, model
        torch.cuda.empty_cache()
        model, _ = load_official_model(
            model_config, runtime / "weights/model_final_6b4a3a.pkl"
        )
        final_checkpoint = out / f"model-{max(checkpoint_steps)}.pt"
        model.load_state_dict(
            torch.load(final_checkpoint, map_location="cpu", weights_only=True)["model"]
        )
        model.cuda()
        for name in cfg.get("diagnostic_views", []):
            _save_probability(
                out,
                name,
                native_predict(model, read_rgb(Path(a.batch) / "export/images" / name)),
            )

