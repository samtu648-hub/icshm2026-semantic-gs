"""Closed-set five-class semantics on a frozen Standard 3DGS field.

Each Gaussian owns exactly five trainable logits.  The renderer converts them
to class probabilities before alpha compositing.  Empty transmittance is
assigned to the official background class, matching the semantic convention
used by the earlier frozen Scaffold SEM-001 baseline.
"""

from __future__ import annotations

import hashlib
import importlib


CLASS_COUNT = 5


def build_direct_logit_field(point_count: int, *, device="cuda", dtype=None):
    """Create the only trainable tensor in the minimal direct-semantic model."""
    torch = importlib.import_module("torch")
    if not isinstance(point_count, int) or point_count <= 0:
        raise ValueError("point_count must be a positive integer")
    dtype = dtype or torch.float32
    return torch.nn.Parameter(
        torch.zeros((point_count, CLASS_COUNT), device=device, dtype=dtype)
    )


def direct_logit_sha256(logits) -> str:
    """Hash shape, dtype and bytes for exact checkpoint/replay evidence."""
    tensor = logits.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.reshape(-1).view(importlib.import_module("torch").uint8).numpy().tobytes())
    return digest.hexdigest()


def render_direct_probabilities(native_render, view, model, pipe, logits):
    """Render five probabilities through the frozen native Gaussian rasterizer.

    Classes 0--2 and 3--4 are rendered in two native RGB passes.  The first
    pass uses an all-background canvas, so residual transmittance contributes
    to class 0 instead of producing a uniform softmax in empty pixels.
    """
    torch = importlib.import_module("torch")
    functional = importlib.import_module("torch.nn.functional")
    if logits.ndim != 2 or logits.shape != (model.get_xyz.shape[0], CLASS_COUNT):
        raise ValueError("Direct semantic logits do not match frozen Gaussian topology")
    if logits.device != model.get_xyz.device or logits.dtype != model.get_xyz.dtype:
        raise ValueError("Direct semantic logits must match frozen geometry device and dtype")
    probabilities = functional.softmax(logits, dim=1)
    background_first = torch.tensor(
        [1.0, 0.0, 0.0], device=logits.device, dtype=logits.dtype
    )
    background_zero = torch.zeros(3, device=logits.device, dtype=logits.dtype)
    first = native_render(
        view,
        model,
        pipe,
        background_first,
        override_color=probabilities[:, :3].contiguous(),
    )["render"]
    tail_colors = torch.cat(
        (probabilities[:, 3:5], torch.zeros_like(probabilities[:, :1])), dim=1
    )
    tail = native_render(
        view,
        model,
        pipe,
        background_zero,
        override_color=tail_colors.contiguous(),
    )["render"]
    output = torch.cat((first, tail[:2]), dim=0).clamp_min(0.0)
    output = output / output.sum(dim=0, keepdim=True).clamp_min(
        torch.finfo(output.dtype).tiny
    )
    if not torch.isfinite(output).all():
        raise FloatingPointError("Non-finite direct semantic probability")
    return output


__all__ = [
    "CLASS_COUNT",
    "build_direct_logit_field",
    "direct_logit_sha256",
    "render_direct_probabilities",
]
