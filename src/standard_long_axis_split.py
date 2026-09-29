"""Long-axis split hook for the RGB151 Standard 3DGS candidate.

The hook changes how already-selected large Gaussians split. Candidate
selection, cloning, pruning, and every other training operation stay native.
"""

from __future__ import annotations

import math
from typing import Any


def select_split_mask(model: Any, grads: Any, threshold: float, extent: float) -> Any:
    """Reproduce the native Standard large-Gaussian selection rule."""
    import torch  # type: ignore[import-not-found]

    point_count = model.get_xyz.shape[0]
    padded = torch.zeros(point_count, dtype=grads.dtype, device=grads.device)
    padded[: grads.shape[0]] = grads.squeeze()
    return (padded >= threshold) & (
        model.get_scaling.max(dim=1).values > model.percent_dense * extent
    )


def long_axis_children(
    parent_xyz: Any,
    parent_scales: Any,
    rotation_matrices: Any,
    parent_opacity: Any,
    *,
    split_distance: float = 0.45,
    opacity_multiplier: float = 0.6,
) -> tuple[Any, Any, Any, Any]:
    """Return child XYZ, scales, opacity, and longest-axis indices.

    ``split_distance`` is d/L0 from ImprovedGS LAS, where L0 is represented
    by three standard deviations. Children are ordered positive-axis first,
    then negative-axis. Opacity is returned in activated space.
    """
    import torch  # type: ignore[import-not-found]

    if not 0.0 < split_distance <= 0.5:
        raise ValueError("split_distance must be in (0, 0.5]")
    if not 0.0 < opacity_multiplier <= 1.0:
        raise ValueError("opacity_multiplier must be in (0, 1]")
    if parent_xyz.ndim != 2 or parent_xyz.shape[1] != 3:
        raise ValueError("parent_xyz must have shape [P, 3]")
    if parent_scales.shape != parent_xyz.shape:
        raise ValueError("parent_scales must match parent_xyz")
    if rotation_matrices.shape != (parent_xyz.shape[0], 3, 3):
        raise ValueError("rotation_matrices must have shape [P, 3, 3]")
    if parent_opacity.shape != (parent_xyz.shape[0], 1):
        raise ValueError("parent_opacity must have shape [P, 1]")
    if not all(
        torch.isfinite(value).all()
        for value in (parent_xyz, parent_scales, rotation_matrices, parent_opacity)
    ):
        raise FloatingPointError("Nonfinite parent split parameters")
    if not torch.all(parent_scales > 0):
        raise ValueError("Parent scales must be positive")
    if not torch.all((parent_opacity > 0) & (parent_opacity < 1)):
        raise ValueError("Activated parent opacity must be in (0, 1)")

    max_values, max_indices = parent_scales.max(dim=1, keepdim=True)
    axis_mask = torch.zeros_like(parent_scales).scatter(1, max_indices, 1.0)
    local_offset = parent_scales * axis_mask * (3.0 * split_distance)
    world_offset = torch.bmm(rotation_matrices, local_offset.unsqueeze(-1)).squeeze(-1)
    child_xyz = torch.cat((parent_xyz + world_offset, parent_xyz - world_offset), dim=0)

    minor_rate = math.sqrt(1.0 - split_distance * split_distance)
    child_scales = parent_scales * minor_rate
    child_scales = child_scales.scatter(
        1, max_indices, max_values * (1.0 - split_distance)
    ).repeat(2, 1)
    child_opacity = (parent_opacity * opacity_multiplier).repeat(2, 1)

    if not all(
        torch.isfinite(value).all()
        for value in (child_xyz, child_scales, child_opacity)
    ):
        raise FloatingPointError("Nonfinite child split parameters")
    if not torch.all(child_scales > 0):
        raise FloatingPointError("Long-axis split produced non-positive scales")
    return child_xyz, child_scales, child_opacity, max_indices.squeeze(1)


def projected_covariance(scales: Any, rotation_matrices: Any, jacobian: Any) -> Any:
    """Project 3D covariance with a supplied local linear camera Jacobian."""
    covariance = (
        rotation_matrices
        @ scales.diag_embed().square()
        @ rotation_matrices.transpose(-1, -2)
    )
    if jacobian.ndim == 2:
        jacobian = jacobian.expand(scales.shape[0], -1, -1)
    return jacobian @ covariance @ jacobian.transpose(-1, -2)


def install(
    model_class: type,
    *,
    enabled: bool,
    split_distance: float = 0.45,
    opacity_multiplier: float = 0.6,
    preserve_native_rng_draw: bool = True,
) -> dict[str, Any]:
    """Install the process-local split hook and return its audit state."""
    original_split = model_class.densify_and_split
    state: dict[str, Any] = {
        "enabled": enabled,
        "method": "long_axis_split" if enabled else "native",
        "split_distance": split_distance,
        "opacity_multiplier": opacity_multiplier,
        "preserve_native_rng_draw": preserve_native_rng_draw,
        "calls": 0,
        "selected_parents": 0,
    }
    if not enabled:
        return state

    import torch  # type: ignore[import-not-found]
    from utils.general_utils import build_rotation  # type: ignore[import-not-found]

    def split(
        model: Any, grads: Any, threshold: float, extent: float, N: int = 2
    ) -> None:
        if N != 2:
            raise ValueError("RGB151 long-axis split requires native binary splitting")
        selected = select_split_mask(model, grads, threshold, extent)
        parent_count = int(selected.sum().item())
        parent_scales = model.get_scaling[selected]

        # Match the native draw shape. This keeps Torch RNG aligned whenever the
        # two arms reach a split event with the same selected topology.
        if preserve_native_rng_draw:
            stds = parent_scales.repeat(N, 1)
            torch.normal(mean=torch.zeros_like(stds), std=stds)

        rotations = build_rotation(model._rotation[selected])
        child_xyz, child_scales, child_opacity, _ = long_axis_children(
            model.get_xyz[selected],
            parent_scales,
            rotations,
            model.get_opacity[selected],
            split_distance=split_distance,
            opacity_multiplier=opacity_multiplier,
        )
        new_scaling = model.scaling_inverse_activation(child_scales)
        new_opacity = model.inverse_opacity_activation(child_opacity)
        new_rotation = model._rotation[selected].repeat(N, 1)
        new_features_dc = model._features_dc[selected].repeat(N, 1, 1)
        new_features_rest = model._features_rest[selected].repeat(N, 1, 1)
        new_tmp_radii = model.tmp_radii[selected].repeat(N)

        model.densification_postfix(
            child_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_tmp_radii,
        )
        prune_filter = torch.cat(
            (
                selected,
                torch.zeros(N * parent_count, dtype=torch.bool, device=selected.device),
            )
        )
        model.prune_points(prune_filter)
        state["calls"] += 1
        state["selected_parents"] += parent_count

    model_class.densify_and_split = split
    state["original_split"] = original_split
    return state
