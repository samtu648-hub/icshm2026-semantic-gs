"""Fixed stay-cable boundary Dice auxiliary objective for Mask2Former.

This module deliberately leaves the native Mask2Former criterion, optimizer,
sampling order, and inference path unchanged.  The sole candidate variable is a
small Dice loss on the differentiable boundary of the dense stay-cable score.
"""

from __future__ import annotations

from pathlib import Path
import json
import random
import time


def dense_semantic_probabilities(outputs, *, size):
    """Convert native query outputs to normalized dense class probabilities."""
    import torch
    import torch.nn.functional as F

    logits = outputs["pred_logits"]
    masks = outputs["pred_masks"]
    if logits.ndim != 3 or masks.ndim != 4 or logits.shape[:2] != masks.shape[:2]:
        raise ValueError("Unexpected Mask2Former query output shapes")
    class_probability = logits.softmax(dim=-1)[..., :-1]
    mask_probability = masks.sigmoid()
    scores = torch.einsum("bqc,bqhw->bchw", class_probability, mask_probability)
    probability = scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-6)
    probability = F.interpolate(
        probability, size=size, mode="bilinear", align_corners=False
    )
    return probability / probability.sum(dim=1, keepdim=True).clamp_min(1e-6)


def soft_morphological_boundary(probability):
    """Return a differentiable one-pixel-radius morphological boundary."""
    import torch.nn.functional as F

    if probability.ndim != 4 or probability.shape[1] != 1:
        raise ValueError("Boundary probability must have shape Bx1xHxW")
    maximum = F.max_pool2d(probability, kernel_size=3, stride=1, padding=1)
    minimum = -F.max_pool2d(-probability, kernel_size=3, stride=1, padding=1)
    return (maximum - minimum).clamp(0.0, 1.0)


def stay_cable_boundary_dice(probability, target, *, class_id=2, epsilon=1.0):
    """Dice loss between soft and binary morphological stay-cable boundaries."""
    import torch

    if probability.ndim != 4 or target.ndim != 3:
        raise ValueError("Expected BxCxHxW probabilities and BxHxW targets")
    if probability.shape[0] != target.shape[0] or probability.shape[2:] != target.shape[1:]:
        raise ValueError("Boundary probability and target grids do not align")
    if not 0 <= int(class_id) < probability.shape[1]:
        raise ValueError("Boundary class ID is outside the semantic vocabulary")
    predicted = soft_morphological_boundary(probability[:, int(class_id) : int(class_id) + 1])
    binary = (target == int(class_id)).to(dtype=probability.dtype).unsqueeze(1)
    expected = soft_morphological_boundary(binary)
    dims = (1, 2, 3)
    numerator = 2.0 * (predicted * expected).sum(dim=dims) + float(epsilon)
    denominator = predicted.sum(dim=dims) + expected.sum(dim=dims) + float(epsilon)
    loss = 1.0 - numerator / denominator
    if not torch.isfinite(loss).all():
        raise FloatingPointError("Nonfinite stay-cable boundary Dice loss")
    return loss.mean(), {
        "predicted_boundary_mass": predicted.detach().sum(),
        "target_boundary_mass": expected.detach().sum(),
    }


def boundary_losses(model, image, instances, target, *, weight, class_id=2):
    """Run the unchanged native criterion and append one boundary Dice term."""
    from detectron2.structures import ImageList

    images = [(image.to(model.device) - model.pixel_mean) / model.pixel_std]
    images = ImageList.from_tensors(images, model.size_divisibility)
    features = model.backbone(images.tensor)
    outputs = model.sem_seg_head(features)
    prepared = model.prepare_targets([instances.to(model.device)], images)
    losses = model.criterion(outputs, prepared)
    for key in list(losses):
        if key in model.criterion.weight_dict:
            losses[key] *= model.criterion.weight_dict[key]
        else:
            losses.pop(key)
    probability = dense_semantic_probabilities(outputs, size=tuple(target.shape[-2:]))
    boundary, diagnostics = stay_cable_boundary_dice(
        probability, target.to(model.device), class_id=class_id
    )
    losses["loss_stay_cable_boundary"] = boundary * float(weight)
    return losses, boundary, diagnostics


def boundary_amp_update(
    model,
    optimizer,
    scaler,
    image,
    instances,
    target,
    *,
    weight,
    class_id=2,
    max_attempts=12,
    clip=0.01,
):
    """Use the baseline AMP retry contract with the fixed auxiliary loss."""
    import torch

    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    skips = []
    for attempt in range(max_attempts):
        optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        with torch.autocast("cuda", dtype=torch.float16):
            losses, raw_boundary, diagnostics = boundary_losses(
                model,
                image,
                instances,
                target,
                weight=weight,
                class_id=class_id,
            )
            loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite augmented Mask2Former loss")
        scaler.scale(loss).backward()
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
                raise FloatingPointError("AMP did not skip/back off nonfinite gradient")
            del losses, loss, raw_boundary, diagnostics
            continue
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), clip, error_if_nonfinite=True
        )
        scaler.step(optimizer)
        scaler.update()
        return {
            "loss": float(loss.detach()),
            "grad_norm": float(norm),
            "scale": scaler.get_scale(),
            "losses": {key: float(value.detach()) for key, value in losses.items()},
            "raw_stay_cable_boundary_dice": float(raw_boundary.detach()),
            "predicted_boundary_mass": float(
                diagnostics["predicted_boundary_mass"].cpu()
            ),
            "target_boundary_mass": float(diagnostics["target_boundary_mass"].cpu()),
            "skips": skips,
        }
    raise FloatingPointError("AMP exhausted 12 attempts without finite update")


def train_mask2former_boundary(args, out, result, labels, cfg, boundary_cfg):
    """Train the SEM396 recipe with only the registered boundary term added."""
    import torch
    from detectron2.structures import Instances
    from types import SimpleNamespace

    from src.quality_mask2former import (
        author_optimizer,
        class_masks,
        config_for_model,
        load_official_model,
        native_predict,
    )
    from src.quality_mask2former_training import (
        poly_factor,
        sampled_crop,
        save_checkpoint,
    )
    from src.quality_protocol import write_new_json
    from src.semantic_cable_binary import read_rgb

    root = Path(__file__).resolve().parents[1]
    runtime = root / "outputs/quality/mask2former/SEM-340"
    model_cfg = config_for_model(
        runtime / "vendor/Mask2Former",
        "configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml",
        False,
    )
    model_cfg.defrost()
    model_cfg.DATASETS.TRAIN = ("sem341_rendered_training",)
    model_cfg.SOLVER.MAX_ITER = 6000
    model_cfg.SOLVER.POLY_LR_POWER = 0.9
    model_cfg.freeze()
    (out / "resolved_model.yaml").write_text(model_cfg.dump())
    model, loaded = load_official_model(
        model_cfg, runtime / "weights/model_final_6b4a3a.pkl"
    )
    result["pretrained_load"] = loaded
    if sum(parameter.numel() for parameter in model.parameters()) != 215451514:
        raise ValueError("Mask2Former parameter count changed")
    model.cuda()
    optimizer = author_optimizer(model, model_cfg)
    rates = [group["lr"] for group in optimizer.param_groups]
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    names = sorted(labels.allowed)
    images = {name: read_rgb(args.batch / "export/images" / name) for name in names}
    targets = {name: labels.read(name) for name in names}
    rng = random.Random(cfg["seed"])
    model.train()
    started = time.monotonic()
    result["amp_skips"] = 0
    result["boundary_objective"] = dict(boundary_cfg)
    with (out / "training.jsonl").open("x") as journal:
        for step in range(1, 6001):
            rgb, target_array, trace = sampled_crop(
                rng, names, images, targets, 768
            )
            ids, masks = class_masks(target_array)
            instances = Instances(tuple(target_array.shape))
            instances.gt_classes = torch.from_numpy(ids)
            instances.gt_masks = torch.from_numpy(masks)
            image = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()
            target = torch.from_numpy(target_array.astype("int64", copy=True))[None]
            factor = poly_factor(step - 1)
            for group, rate in zip(optimizer.param_groups, rates):
                group["lr"] = rate * factor
            row = boundary_amp_update(
                model,
                optimizer,
                scaler,
                image,
                instances,
                target,
                weight=float(boundary_cfg["weight"]),
                class_id=int(boundary_cfg["class_id"]),
                max_attempts=12,
                clip=0.01,
            )
            row.update(
                step=step,
                sample=trace,
                lr_factor=factor,
                elapsed_seconds=time.monotonic() - started,
            )
            journal.write(json.dumps(row) + "\n")
            result["trained_steps"] = step
            result["amp_skips"] += len(row["skips"])
            if step % 100 == 0 or step == 1:
                journal.flush()
                print(
                    json.dumps(
                        {
                            "stage": "mask2former_boundary",
                            "step": step,
                            "amp_skips": result["amp_skips"],
                            "seconds": time.monotonic() - started,
                            "raw_boundary_dice": row[
                                "raw_stay_cable_boundary_dice"
                            ],
                        }
                    ),
                    flush=True,
                )
            if step in (3000, 6000):
                save_checkpoint(
                    model,
                    SimpleNamespace(binding=result["binding"], access=result["access"]),
                    step,
                    out / f"model-{step}.pt",
                )
    result["training_seconds"] = time.monotonic() - started
    del optimizer, scaler, images, targets, model
    torch.cuda.empty_cache()
    model, _ = load_official_model(
        model_cfg, runtime / "weights/model_final_6b4a3a.pkl"
    )
    model.load_state_dict(
        torch.load(out / "model-6000.pt", map_location="cpu", weights_only=True)[
            "model"
        ]
    )
    model.cuda()
    for name in cfg["diagnostic_views"]:
        probability = native_predict(
            model, read_rgb(args.batch / "export/images" / name)
        )
        destination = out / "probabilities"
        destination.mkdir(exist_ok=True)
        import numpy as np

        np.savez_compressed(destination / f"{Path(name).stem}.npz", probabilities=probability)


__all__ = [
    "boundary_amp_update",
    "boundary_losses",
    "dense_semantic_probabilities",
    "soft_morphological_boundary",
    "stay_cable_boundary_dice",
    "train_mask2former_boundary",
]
