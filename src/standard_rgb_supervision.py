"""GT-region photometric supervision for the RGB140 Standard recipe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter


def load_structure_mask(label_path: Path, dilation_px: int = 4) -> torch.Tensor:
    """Return the dilated non-background GT region as a 1xHxW float tensor."""
    label = np.asarray(Image.open(label_path))
    if label.ndim != 2 or label.shape != (989, 1320):
        raise ValueError(f"Unexpected label shape for {label_path}: {label.shape}")
    values = set(np.unique(label).tolist())
    if not values.issubset({0, 1, 2, 3, 4}):
        raise ValueError(f"Unexpected class IDs for {label_path}: {sorted(values)}")
    region = Image.fromarray(((label != 0) * 255).astype(np.uint8))
    if dilation_px:
        region = region.filter(ImageFilter.MaxFilter(2 * dilation_px + 1))
    return torch.from_numpy((np.asarray(region) > 0).astype(np.float32))[None]


def masked_rgb_l1(
    rendered: torch.Tensor, photo: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Area-normalized RGB L1 inside a spatial supervision region."""
    _validate_inputs(rendered, photo, mask)
    denominator = mask.sum().clamp_min(1.0) * rendered.shape[0]
    return ((rendered - photo).abs() * mask).sum() / denominator


def multiscale_gradient_l1(
    rendered: torch.Tensor,
    photo: torch.Tensor,
    mask: torch.Tensor,
    scales: Iterable[int] = (1, 4),
) -> torch.Tensor:
    """Compare full-image RGB finite differences, then weight them by the region.

    The images are never masked before gradients are computed. At coarse scales,
    area interpolation preserves the fractional support of the common GT region.
    """
    _validate_inputs(rendered, photo, mask)
    losses = []
    for scale in scales:
        if scale < 1:
            raise ValueError(f"Invalid gradient scale: {scale}")
        if scale == 1:
            pred, target, weights = rendered, photo, mask
        else:
            size = (rendered.shape[-2] // scale, rendered.shape[-1] // scale)
            pred = F.interpolate(rendered[None], size=size, mode="area")[0]
            target = F.interpolate(photo[None], size=size, mode="area")[0]
            weights = F.interpolate(mask[None], size=size, mode="area")[0]

        pred_dx, target_dx = (
            pred[..., 1:] - pred[..., :-1],
            target[..., 1:] - target[..., :-1],
        )
        pred_dy, target_dy = (
            pred[..., 1:, :] - pred[..., :-1, :],
            target[..., 1:, :] - target[..., :-1, :],
        )
        weight_x = torch.maximum(weights[..., 1:], weights[..., :-1])
        weight_y = torch.maximum(weights[..., 1:, :], weights[..., :-1, :])
        channels = rendered.shape[0]
        loss_x = ((pred_dx - target_dx).abs() * weight_x).sum() / (
            weight_x.sum().clamp_min(1.0) * channels
        )
        loss_y = ((pred_dy - target_dy).abs() * weight_y).sum() / (
            weight_y.sum().clamp_min(1.0) * channels
        )
        losses.append((loss_x + loss_y) * 0.5)
    return torch.stack(losses).mean()


def gradient_error_map(
    rendered: torch.Tensor, photo: torch.Tensor, scale: int = 1
) -> torch.Tensor:
    """Return a padded HxW RGB-gradient consistency map for diagnostics."""
    if rendered.shape != photo.shape or rendered.ndim != 3:
        raise ValueError("Expected matching CxHxW RGB tensors")
    if scale > 1:
        size = (rendered.shape[-2] // scale, rendered.shape[-1] // scale)
        rendered = F.interpolate(rendered[None], size=size, mode="area")[0]
        photo = F.interpolate(photo[None], size=size, mode="area")[0]
    dx = (
        ((rendered[..., 1:] - rendered[..., :-1]) - (photo[..., 1:] - photo[..., :-1]))
        .abs()
        .mean(0)
    )
    dy = (
        (
            (rendered[..., 1:, :] - rendered[..., :-1, :])
            - (photo[..., 1:, :] - photo[..., :-1, :])
        )
        .abs()
        .mean(0)
    )
    result = F.pad(dx, (0, 1)) + F.pad(dy, (0, 0, 0, 1))
    result *= 0.5
    if scale > 1:
        result = F.interpolate(
            result[None, None], size=(989, 1320), mode="bilinear", align_corners=False
        )[0, 0]
    return result


class AuxLossController:
    """Apply one registered auxiliary target without changing camera sampling."""

    def __init__(self, config: dict, output: Path, attempt: str = "initial"):
        auxiliary = config["auxiliary_supervision"]
        self.mode = auxiliary["mode"]
        self.coefficient = float(auxiliary["coefficient"])
        self.dilation_px = int(auxiliary["region"]["dilation_px"])
        self.scales = tuple(auxiliary["gradient"]["scales"])
        self.labels = Path(auxiliary["label_root"])
        self.output = output
        self.cache: dict[str, torch.Tensor] = {}
        self.calls = 0
        self.labeled_calls = 0
        self.parameter_probe = None
        suffix = "" if attempt == "initial" else f"-{attempt}"
        self.log_path = output / f"aux_loss{suffix}.jsonl"
        self.summary_path = output / f"aux_summary{suffix}.json"

    def apply(self, iteration, camera, rendered, photo, base_loss, gaussians):
        if self.mode == "control":
            return base_loss
        name = Path(camera.image_name).name
        if not name.endswith(".png"):
            name += ".png"
        label_path = self.labels / name
        self.calls += 1
        if not label_path.exists():
            if iteration % 100 == 0:
                self._log(iteration, name, base_loss, None, len(gaussians.get_xyz))
            return base_loss

        mask = self.cache.get(name)
        if mask is None:
            mask = load_structure_mask(label_path, self.dilation_px).pin_memory()
            self.cache[name] = mask
        mask = mask.to(device=rendered.device, non_blocking=True)
        auxiliary = (
            masked_rgb_l1(rendered, photo, mask)
            if self.mode == "region_rgb"
            else multiscale_gradient_l1(rendered, photo, mask, self.scales)
        )
        if not torch.isfinite(auxiliary):
            raise FloatingPointError(
                f"Non-finite {self.mode} loss at iteration {iteration}"
            )
        self.labeled_calls += 1
        if self.parameter_probe is None:
            params = {
                "xyz": gaussians._xyz,
                "features_dc": gaussians._features_dc,
                "features_rest": gaussians._features_rest,
                "scaling": gaussians._scaling,
                "rotation": gaussians._rotation,
                "opacity": gaussians._opacity,
            }
            gradients = torch.autograd.grad(
                auxiliary, tuple(params.values()), retain_graph=True, allow_unused=True
            )
            self.parameter_probe = {
                key: {
                    "finite": bool(grad is not None and torch.isfinite(grad).all()),
                    "norm": 0.0 if grad is None else float(grad.norm().item()),
                }
                for key, grad in zip(params, gradients)
            }
        if iteration <= 10 or iteration % 100 == 0:
            self._log(
                iteration,
                name,
                base_loss,
                auxiliary,
                len(gaussians.get_xyz),
                int(mask.sum().item()),
            )
        return base_loss + self.coefficient * auxiliary

    def _log(self, iteration, name, base_loss, auxiliary, points, pixels=None):
        row = {
            "iteration": iteration,
            "view": name,
            "mode": self.mode,
            "base_loss": float(base_loss.detach()),
            "points": points,
            "labeled": auxiliary is not None,
        }
        if auxiliary is not None:
            row.update(
                aux_loss=float(auxiliary.detach()),
                weighted_aux_loss=float(self.coefficient * auxiliary.detach()),
                region_pixels=pixels,
            )
        with self.log_path.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    def finalize(self):
        record = {
            "mode": self.mode,
            "coefficient": self.coefficient,
            "calls": self.calls,
            "labeled_calls": self.labeled_calls,
            "unlabeled_calls": self.calls - self.labeled_calls,
            "parameter_gradient_probe": self.parameter_probe,
        }
        with self.summary_path.open("x") as stream:
            stream.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
        return record


def _validate_inputs(
    rendered: torch.Tensor, photo: torch.Tensor, mask: torch.Tensor
) -> None:
    if rendered.shape != photo.shape or rendered.ndim != 3 or rendered.shape[0] != 3:
        raise ValueError(
            f"Expected matching 3xHxW tensors, got {rendered.shape} and {photo.shape}"
        )
    if mask.shape != (1, rendered.shape[1], rendered.shape[2]):
        raise ValueError(
            f"Mask shape {mask.shape} does not match image {rendered.shape}"
        )
    if (
        not torch.isfinite(rendered).all()
        or not torch.isfinite(photo).all()
        or not torch.isfinite(mask).all()
    ):
        raise FloatingPointError("Non-finite supervision input")
