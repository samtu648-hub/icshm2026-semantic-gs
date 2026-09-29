"""Contribution-aware semantic rendering for a frozen Scaffold-GS artifact.

This module is deliberately separate from the upstream RGB renderer and from
``scaffold_semantic_sidecar``.  It consumes the *frozen* Gaussian geometry and
opacity produced by Scaffold-GS, exposes the exact front-to-back per-Gaussian
contribution used by the pinned rasterizer,

``r(g, p) = T(g, p) * alpha(g, p) * G(g, p)``,

and uses a stop-gradient gate only to decide which semantic logits receive a
two-dimensional label gradient.  No RGB parameter is an input to the custom
autograd function, so a semantic loss cannot update RGB geometry, opacity or
appearance through this adapter.

The original rasterizer returns opaque workspace tensors.  The decoded fields
below mirror the pinned upstream ``GeometryState``, ``BinningState`` and
``ImageState`` layouts.  That dependency is intentional and recorded as a
small, isolated adapter rather than silently approximating visibility from
alpha compositing outputs.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from src.scaffold_semantic_sidecar import (
    AnchorSemanticSidecar,
    _raster_settings,
    frozen_generated_geometry,
)


GateName = Literal["cumulative", "relative_max"]
_GATE_MODES: dict[GateName, int] = {"cumulative": 1, "relative_max": 2}
_ALIGNMENT = 128
_BLOCK_WIDTH = 16
_BLOCK_HEIGHT = 16


@dataclass(frozen=True)
class VisibilityBuffers:
    """Decoded pinned-rasterizer workspaces, retained for autograd lifetime."""

    ranges: torch.Tensor
    point_list: torch.Tensor
    depths: torch.Tensor
    means2d: torch.Tensor
    conic_opacity: torch.Tensor
    geometry_buffer: torch.Tensor
    binning_buffer: torch.Tensor
    image_buffer: torch.Tensor
    num_rendered: int


def _source_digest() -> str:
    root = Path(__file__).resolve().parents[1] / "adapters" / "scaffold_gs" / "visibility_rasterizer"
    digest = hashlib.sha256()
    for path in (root / "visibility_rasterizer.cpp", root / "visibility_rasterizer.cu"):
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def load_visibility_extension(build_root: Path):
    """Build/load the semantic-only CUDA adapter outside the Git worktree."""
    import importlib.util
    import sys

    from torch.utils.cpp_extension import load

    source_root = Path(__file__).resolve().parents[1] / "adapters" / "scaffold_gs" / "visibility_rasterizer"
    build_root = build_root.resolve()
    build_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(build_root))
    # The suffix prevents a stale .so from being reused after a source change.
    name = f"icshm_visibility_rasterizer_{_source_digest()}"
    compiled = build_root / name / f"{name}.so"
    # Torch's in-process extension versioner asks Ninja to rebuild on the first
    # load of every new Python process, even when this exact source-digested
    # shared object is already present.  Directly importing the content-named
    # binary is equivalent to the final ``load`` step and lets read-only
    # inference/audit jobs reuse an already verified CUDA build.
    if compiled.is_file():
        spec = importlib.util.spec_from_file_location(name, compiled)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load cached visibility extension: {compiled}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    return load(
        name=name,
        sources=[str(source_root / "visibility_rasterizer.cpp"), str(source_root / "visibility_rasterizer.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


def _allocate_offset(cursor: int, base_pointer: int, count: int, element_size: int) -> tuple[int, int]:
    """Mirror ``CudaRasterizer::obtain`` for an opaque uint8 workspace."""
    start = cursor + ((-(base_pointer + cursor)) % _ALIGNMENT)
    return start, start + count * element_size


def _typed_workspace_view(buffer: torch.Tensor, offset: int, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    if buffer.dtype != torch.uint8 or buffer.dim() != 1 or not buffer.is_contiguous():
        raise TypeError("Pinned rasterizer workspace must be a contiguous uint8 vector")
    count = 1
    for dimension in shape:
        count *= int(dimension)
    element_size = torch.empty((), dtype=dtype).element_size()
    end = offset + count * element_size
    if offset < 0 or end > buffer.numel():
        raise RuntimeError("Pinned rasterizer workspace layout exceeded its allocated buffer")
    return buffer.narrow(0, offset, count * element_size).view(dtype).reshape(shape)


def _geometry_offsets(buffer: torch.Tensor, point_count: int) -> dict[str, int]:
    """Decode fields through conic_opacity from upstream GeometryState."""
    cursor = 0
    base = int(buffer.data_ptr())
    offsets: dict[str, int] = {}
    fields = (
        ("depths", point_count, torch.float32),
        ("clamped", point_count * 3, torch.bool),
        ("internal_radii", point_count, torch.int32),
        ("means2d", point_count * 2, torch.float32),
        ("cov3d", point_count * 6, torch.float32),
        ("conic_opacity", point_count * 4, torch.float32),
    )
    for name, count, dtype in fields:
        offset, cursor = _allocate_offset(cursor, base, count, torch.empty((), dtype=dtype).element_size())
        offsets[name] = offset
    return offsets


def _first_allocation_offset(buffer: torch.Tensor) -> int:
    return (-(int(buffer.data_ptr()))) % _ALIGNMENT


def rasterize_frozen_geometry_buffers(view: Any, geometry: tuple[Any, Any, Any, Any, Any]) -> VisibilityBuffers:
    """Run the pinned rasterizer once and decode its exact visibility inputs.

    This forward is under ``no_grad`` and uses dummy RGB colors.  It only
    requests the upsteam preprocessing/sort workspaces, not a replacement RGB
    render.  The actual RGB path remains the untouched upstream renderer.
    """
    from diff_gaussian_rasterization import _C

    xyz, opacity, scaling, rotation, _active_offsets = geometry
    point_count = int(xyz.shape[0])
    if point_count <= 0:
        raise RuntimeError("Scaffold produced no generated Gaussians for this camera")
    device = xyz.device
    background = torch.zeros(3, dtype=torch.float32, device=device)
    dummy_colors = torch.zeros((point_count, 3), dtype=torch.float32, device=device)
    empty = torch.empty(0, dtype=torch.float32, device=device)
    settings = _raster_settings(view, background)
    with torch.no_grad():
        result = _C.rasterize_gaussians(
            background,
            xyz.contiguous(),
            dummy_colors,
            opacity.contiguous(),
            scaling.contiguous(),
            rotation.contiguous(),
            1.0,
            empty,
            view.world_view_transform.contiguous(),
            view.full_proj_transform.contiguous(),
            float(settings.tanfovx),
            float(settings.tanfovy),
            int(view.image_height),
            int(view.image_width),
            empty,
            1,
            view.camera_center.contiguous(),
            False,
            False,
        )
    num_rendered, _color, _radii, geometry_buffer, binning_buffer, image_buffer = result
    if geometry_buffer.dtype != torch.uint8 or binning_buffer.dtype != torch.uint8 or image_buffer.dtype != torch.uint8:
        raise RuntimeError("Pinned rasterizer did not return opaque uint8 workspaces")
    geometry_offsets = _geometry_offsets(geometry_buffer, point_count)
    depths = _typed_workspace_view(geometry_buffer, geometry_offsets["depths"], (point_count,), torch.float32)
    means2d = _typed_workspace_view(geometry_buffer, geometry_offsets["means2d"], (point_count, 2), torch.float32)
    conic_opacity = _typed_workspace_view(geometry_buffer, geometry_offsets["conic_opacity"], (point_count, 4), torch.float32)
    rendered = int(num_rendered)
    if rendered:
        point_list = _typed_workspace_view(binning_buffer, _first_allocation_offset(binning_buffer), (rendered,), torch.int32)
    else:
        point_list = torch.empty(0, dtype=torch.int32, device=device)
    tiles_x = (int(view.image_width) + _BLOCK_WIDTH - 1) // _BLOCK_WIDTH
    tiles_y = (int(view.image_height) + _BLOCK_HEIGHT - 1) // _BLOCK_HEIGHT
    tile_count = tiles_x * tiles_y
    # ImageState starts with accum_alpha then n_contrib, then uint2 tile ranges.
    cursor = 0
    base = int(image_buffer.data_ptr())
    _alpha_offset, cursor = _allocate_offset(cursor, base, int(view.image_height) * int(view.image_width), 4)
    _ncontrib_offset, cursor = _allocate_offset(cursor, base, int(view.image_height) * int(view.image_width), 4)
    ranges_offset, _ = _allocate_offset(cursor, base, int(view.image_height) * int(view.image_width), 8)
    ranges = _typed_workspace_view(image_buffer, ranges_offset, (tile_count, 2), torch.int32)
    return VisibilityBuffers(
        ranges=ranges,
        point_list=point_list,
        depths=depths,
        means2d=means2d,
        conic_opacity=conic_opacity,
        geometry_buffer=geometry_buffer,
        binning_buffer=binning_buffer,
        image_buffer=image_buffer,
        num_rendered=rendered,
    )


class _VisibilityRasterize(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        probabilities: torch.Tensor,
        buffers: VisibilityBuffers,
        height: int,
        width: int,
        gate_mode: int,
        threshold: float,
        extension: Any,
    ):
        outputs = extension.forward(
            buffers.ranges,
            buffers.point_list,
            buffers.means2d,
            buffers.conic_opacity,
            probabilities.contiguous(),
            int(height),
            int(width),
            int(gate_mode),
            float(threshold),
        )
        numerator, foreground, maximum, contributor_count, retained_count, retained_mass = outputs
        ctx.save_for_backward(
            buffers.ranges,
            buffers.point_list,
            buffers.means2d,
            buffers.conic_opacity,
            foreground,
            maximum,
        )
        ctx.point_count = int(probabilities.shape[0])
        ctx.height = int(height)
        ctx.width = int(width)
        ctx.gate_mode = int(gate_mode)
        ctx.threshold = float(threshold)
        ctx.extension = extension
        ctx.mark_non_differentiable(foreground, maximum, contributor_count, retained_count, retained_mass)
        return numerator, foreground, maximum, contributor_count, retained_count, retained_mass

    @staticmethod
    def backward(ctx, grad_numerator, _grad_foreground, _grad_maximum, _grad_count, _grad_retained_count, _grad_retained_mass):
        ranges, point_list, means2d, conic_opacity, foreground, maximum = ctx.saved_tensors
        if grad_numerator is None:
            return None, None, None, None, None, None, None
        gradient = ctx.extension.backward(
            ranges,
            point_list,
            means2d,
            conic_opacity,
            grad_numerator.contiguous().to(dtype=torch.float32),
            foreground,
            maximum,
            ctx.height,
            ctx.width,
            ctx.point_count,
            ctx.gate_mode,
            ctx.threshold,
        )
        return gradient, None, None, None, None, None, None


def render_visibility_semantic_probabilities(
    view: Any,
    model: Any,
    sidecar: AnchorSemanticSidecar,
    pipe: Any,
    *,
    gate: GateName,
    threshold: float,
    extension: Any,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Render gated five-class semantics while keeping all RGB tensors frozen.

    Cumulative gating retains a front-to-back minimal prefix whose accumulated
    contribution reaches ``threshold * foreground_mass``.  Relative-max
    gating retains contributors with ``r >= threshold * max_g r(g,p)``.
    The binary selection is recomputed only from detached frozen geometry and
    opacity inside the CUDA adapter, hence it is a stop-gradient mask.
    """
    if gate not in _GATE_MODES:
        raise ValueError(f"Unsupported visibility gate: {gate}")
    if not 0.0 < float(threshold) <= 1.0:
        raise ValueError("Visibility threshold must be in (0, 1]")
    geometry, visible = frozen_generated_geometry(view, model, pipe)
    active_offsets = geometry[4]
    anchor_logits = sidecar.logits[visible]
    offset_logits = anchor_logits.repeat_interleave(model.n_offsets, dim=0)[active_offsets]
    offset_probabilities = torch.softmax(offset_logits, dim=1)
    visible_anchor_indices = torch.nonzero(visible, as_tuple=False).squeeze(1)
    offset_anchor_indices = visible_anchor_indices.repeat_interleave(model.n_offsets)[active_offsets]
    buffers = rasterize_frozen_geometry_buffers(view, geometry)
    numerator, foreground, maximum, contributor_count, retained_count, retained_mass = _VisibilityRasterize.apply(
        offset_probabilities,
        buffers,
        int(view.image_height),
        int(view.image_width),
        _GATE_MODES[gate],
        float(threshold),
        extension,
    )
    semantic_probabilities = numerator / numerator.sum(dim=0, keepdim=True).clamp_min(torch.finfo(numerator.dtype).tiny)
    stats = {
        "foreground_mass": foreground,
        "max_contribution": maximum,
        "contributor_count": contributor_count,
        "retained_contributor_count": retained_count,
        "retained_contribution_mass": retained_mass,
        "visible_anchor_mask": visible,
        "active_offset_mask": active_offsets,
        # These are returned only for an immediate diagnostic/autograd query;
        # callers must never serialize them.  They make the per-Gaussian
        # semantic-gradient support calculation auditable rather than inferred
        # from alpha compositing summaries.
        "offset_probabilities": offset_probabilities,
        "semantic_numerator": numerator,
        "offset_anchor_indices": offset_anchor_indices,
    }
    return semantic_probabilities, stats


def render_frozen_contribution_diagnostics(
    view: Any,
    model: Any,
    pipe: Any,
    *,
    extension: Any,
) -> dict[str, torch.Tensor]:
    """Return exact frozen-RGB contribution summaries without semantic gradients.

    The returned ``dominant_offset_index`` identifies the generated Gaussian
    with the largest marginal contribution at each pixel, or ``-1`` for a
    pixel with no generated-Gaussian contribution.  It is deliberately a
    diagnostic-only output: this function does not accept semantic logits,
    build a gate, or attach an autograd path to RGB state.
    """
    geometry, visible = frozen_generated_geometry(view, model, pipe)
    buffers = rasterize_frozen_geometry_buffers(view, geometry)
    foreground, maximum, count, dominant = extension.diagnostic_stats(
        buffers.ranges,
        buffers.point_list,
        buffers.means2d,
        buffers.conic_opacity,
        int(view.image_height),
        int(view.image_width),
    )
    active_offsets = geometry[4]
    visible_anchor_indices = torch.nonzero(visible, as_tuple=False).squeeze(1)
    offset_anchor_indices = visible_anchor_indices.repeat_interleave(model.n_offsets)[active_offsets]
    return {
        "foreground_mass": foreground,
        "max_contribution": maximum,
        "contributor_count": count,
        "dominant_offset_index": dominant,
        "offset_anchor_indices": offset_anchor_indices,
        # Diagnostic-only generated properties.  They remain detached because
        # ``frozen_generated_geometry`` executes under ``no_grad``.
        "generated_opacity": geometry[1],
        "generated_scaling": geometry[2],
    }


def render_frozen_depth_diagnostics(
    view: Any,
    model: Any,
    pipe: Any,
    *,
    extension: Any,
) -> dict[str, torch.Tensor]:
    """Render a diagnostic camera-space depth map from frozen RGB geometry.

    Depth is taken from the generated Gaussian with the largest exact
    front-to-back contribution at each pixel.  Invalid/background pixels are
    NaN.  The helper is read-only and reuses the same pinned-rasterizer
    workspaces as the semantic visibility adapter, so it does not introduce a
    second projection convention or an RGB gradient path.
    """
    geometry, _visible = frozen_generated_geometry(view, model, pipe)
    buffers = rasterize_frozen_geometry_buffers(view, geometry)
    foreground, maximum, count, dominant = extension.diagnostic_stats(
        buffers.ranges,
        buffers.point_list,
        buffers.means2d,
        buffers.conic_opacity,
        int(view.image_height),
        int(view.image_width),
    )
    depth = torch.full_like(foreground, float("nan"))
    valid = (dominant >= 0) & (dominant < buffers.depths.numel())
    depth[valid] = buffers.depths[dominant[valid].to(torch.long)]
    valid &= torch.isfinite(depth) & (depth > 0.0)
    depth[~valid] = float("nan")
    return {
        "depth": depth,
        "valid": valid,
        "foreground_mass": foreground,
        "max_contribution": maximum,
        "contributor_count": count,
        "dominant_offset_index": dominant,
    }


def per_pixel_retention_summary(stats: dict[str, torch.Tensor]) -> dict[str, float]:
    """Small, JSON-ready contribution diagnostics without retaining images."""
    foreground = stats["foreground_mass"]
    retained = stats["retained_contribution_mass"]
    contributors = stats["contributor_count"].to(torch.float32)
    retained_contributors = stats["retained_contributor_count"].to(torch.float32)
    valid = foreground > 0.0
    if not bool(valid.any()):
        return {
            "pixels_with_foreground": 0.0,
            "retained_contribution_mass_fraction": 0.0,
            "retained_contributor_fraction": 0.0,
            "max_contribution_mean": 0.0,
        }
    return {
        "pixels_with_foreground": float(valid.sum().item()),
        "retained_contribution_mass_fraction": float((retained[valid].sum() / foreground[valid].sum().clamp_min(torch.finfo(foreground.dtype).tiny)).item()),
        "retained_contributor_fraction": float((retained_contributors[valid].sum() / contributors[valid].sum().clamp_min(1.0)).item()),
        "max_contribution_mean": float(stats["max_contribution"][valid].mean().item()),
    }
