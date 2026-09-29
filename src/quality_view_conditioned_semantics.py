"""First-order view-conditioned semantics on a frozen Gaussian field.

Each Gaussian owns five affine functions of its unit direction to the camera:
one bias plus x/y/z coefficients per class.  Geometry, opacity and RGB SH
coefficients remain frozen; only these semantic coefficients are trainable.
"""

from __future__ import annotations

import hashlib
import importlib


CLASS_COUNT = 5
BASIS_DIMENSION = 4


def build_directional_logit_field(point_count: int, *, device="cuda", dtype=None):
    """Return zero-initialized N x 5 x (bias,x,y,z) semantic coefficients."""
    torch = importlib.import_module("torch")
    if not isinstance(point_count, int) or point_count <= 0:
        raise ValueError("point_count must be a positive integer")
    dtype = dtype or torch.float32
    return torch.nn.Parameter(
        torch.zeros(
            (point_count, CLASS_COUNT, BASIS_DIMENSION),
            device=device,
            dtype=dtype,
        )
    )


def directional_logit_sha256(coefficients) -> str:
    """Hash tensor metadata and bytes for exact checkpoint evidence."""
    torch = importlib.import_module("torch")
    tensor = coefficients.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def direction_basis(view, xyz):
    """Build [1, unit(Gaussian-to-camera xyz)] for every frozen Gaussian."""
    torch = importlib.import_module("torch")
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("Frozen Gaussian positions must have shape N x 3")
    center = view.camera_center
    if center.shape != (3,) or center.device != xyz.device or center.dtype != xyz.dtype:
        raise ValueError("Camera center must match frozen geometry device and dtype")
    direction = torch.nn.functional.normalize(center[None] - xyz, dim=1, eps=1e-12)
    return torch.cat((torch.ones_like(direction[:, :1]), direction), dim=1)


def directional_logits(coefficients, view, model):
    """Evaluate per-Gaussian five-class logits for one registered camera."""
    if coefficients.ndim != 3 or coefficients.shape != (
        model.get_xyz.shape[0],
        CLASS_COUNT,
        BASIS_DIMENSION,
    ):
        raise ValueError("Directional semantic coefficients do not match frozen Gaussian topology")
    if coefficients.device != model.get_xyz.device or coefficients.dtype != model.get_xyz.dtype:
        raise ValueError("Directional coefficients must match frozen geometry device and dtype")
    basis = direction_basis(view, model.get_xyz)
    return (coefficients * basis[:, None, :]).sum(dim=2)


def render_directional_probabilities(native_render, view, model, pipe, coefficients):
    """Evaluate and alpha-composite five view-conditioned class probabilities."""
    torch = importlib.import_module("torch")
    logits = directional_logits(coefficients, view, model)
    probabilities = torch.nn.functional.softmax(logits, dim=1)
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
        raise FloatingPointError("Non-finite directional semantic probability")
    return output


__all__ = [
    "BASIS_DIMENSION",
    "CLASS_COUNT",
    "build_directional_logit_field",
    "direction_basis",
    "directional_logits",
    "directional_logit_sha256",
    "render_directional_probabilities",
]
