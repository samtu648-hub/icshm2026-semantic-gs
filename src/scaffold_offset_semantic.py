"""Offset-granular semantic sidecars for SEM-230.

Scaffold-GS generates several Gaussians from every saved anchor.  SEM-001
assigned the same five semantic logits to all of them.  This module keeps the
Scaffold model entirely outside the semantic parameter set and introduces one
*independent* background/deck residual for every generated offset.

The residual is deliberately restricted to classes 0 and 1.  Classes 2--4 are
not represented by this learning lane and are protected again at fusion time.
The five-class adapter exists solely to prove that an all-zero residual is
exactly equivalent to the historical SEM-001 inheritance rule.
"""

from __future__ import annotations

from typing import Any

import torch

from src.scaffold_semantic_sidecar import (
    CLASS_COUNT,
    _render_color_channels,
    frozen_generated_geometry,
)


BACKGROUND = 0
DECK = 1
MUTABLE_CLASS_IDS = (BACKGROUND, DECK)


def active_offset_indices(
    visible: torch.Tensor,
    active_offsets: torch.Tensor,
    *,
    n_offsets: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return global ``(anchor, offset)`` indices for active generated Gaussians.

    ``active_offsets`` is the upstream flattened index into
    ``visible_anchors × n_offsets``.  Keeping this mapping explicit is the
    key SEM-230 invariant: no generated offset may silently inherit another
    offset's residual.
    """
    if visible.dtype != torch.bool or visible.ndim != 1:
        raise ValueError("visible must be a 1-D boolean anchor mask")
    if int(n_offsets) <= 0:
        raise ValueError("n_offsets must be positive")
    visible_indices = torch.nonzero(visible, as_tuple=False).squeeze(1)
    flat_size = int(visible_indices.numel()) * int(n_offsets)
    if active_offsets.dtype == torch.bool:
        if active_offsets.ndim != 1 or int(active_offsets.numel()) != flat_size:
            raise ValueError("boolean active_offsets must match the visible-anchor offset table")
        flat = torch.nonzero(active_offsets, as_tuple=False).squeeze(1)
    else:
        flat = active_offsets.to(dtype=torch.long)
        if flat.ndim != 1 or (flat.numel() and (int(flat.min()) < 0 or int(flat.max()) >= flat_size)):
            raise ValueError("active_offsets is outside the visible-anchor offset table")
    anchor_indices = visible_indices.repeat_interleave(int(n_offsets))[flat]
    offset_indices = torch.arange(int(n_offsets), device=visible.device).repeat(visible_indices.numel())[flat]
    return anchor_indices, offset_indices


class OffsetSemanticResidualSidecar:
    """Frozen five-class anchor logits plus per-offset background/deck residuals.

    This is used for the mandatory zero-residual SEM-001 equivalence test.  It
    intentionally owns only ``offset_background_deck_residual`` as a
    trainable tensor; ``base_anchor_logits`` is a detached copy of the supplied
    historical anchor logits.
    """

    def __init__(
        self,
        base_anchor_logits: torch.Tensor,
        *,
        n_offsets: int,
        device: torch.device | str | None = None,
    ) -> None:
        if base_anchor_logits.ndim != 2 or base_anchor_logits.shape[1] != CLASS_COUNT:
            raise ValueError("base_anchor_logits must have shape [anchors, 5]")
        if int(n_offsets) <= 0:
            raise ValueError("n_offsets must be positive")
        resolved = base_anchor_logits.device if device is None else torch.device(device)
        self.base_anchor_logits = base_anchor_logits.detach().to(resolved, dtype=torch.float32).clone()
        self.n_offsets = int(n_offsets)
        self.offset_background_deck_residual = torch.nn.Parameter(
            torch.zeros(
                (self.base_anchor_logits.shape[0], self.n_offsets, 2),
                dtype=torch.float32,
                device=resolved,
            )
        )

    @property
    def anchor_count(self) -> int:
        return int(self.base_anchor_logits.shape[0])

    def offset_logits(self, visible: torch.Tensor, active_offsets: torch.Tensor) -> torch.Tensor:
        anchors, offsets = active_offset_indices(visible, active_offsets, n_offsets=self.n_offsets)
        inherited = self.base_anchor_logits[anchors]
        residual = self.offset_background_deck_residual[anchors, offsets]
        result = inherited.clone()
        result[:, BACKGROUND] = result[:, BACKGROUND] + residual[:, 0]
        result[:, DECK] = result[:, DECK] + residual[:, 1]
        return result

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "base_anchor_logits": self.base_anchor_logits.detach().cpu(),
            "offset_background_deck_residual": self.offset_background_deck_residual.detach().cpu(),
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        stored_base = payload.get("base_anchor_logits")
        stored_residual = payload.get("offset_background_deck_residual")
        if not isinstance(stored_base, torch.Tensor) or not torch.equal(
            stored_base.to(dtype=torch.float32, device="cpu"), self.base_anchor_logits.detach().cpu()
        ):
            raise ValueError("Offset residual checkpoint base anchor logits differ")
        if not isinstance(stored_residual, torch.Tensor) or tuple(stored_residual.shape) != tuple(self.offset_background_deck_residual.shape):
            raise ValueError("Offset residual checkpoint shape differs")
        with torch.no_grad():
            self.offset_background_deck_residual.copy_(
                stored_residual.to(self.offset_background_deck_residual.device, dtype=torch.float32)
            )


def render_offset_residual_semantic_probabilities(
    view: Any,
    model: Any,
    sidecar: OffsetSemanticResidualSidecar,
    pipe: Any,
) -> torch.Tensor:
    """Render five classes with offset residuals using SEM-001's rasterizer."""
    geometry, visible = frozen_generated_geometry(view, model, pipe)
    offset_logits = sidecar.offset_logits(visible, geometry[4])
    probabilities = torch.softmax(offset_logits, dim=1)
    rgb_like = _render_color_channels(
        geometry,
        probabilities[:, :3],
        view,
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=probabilities.device),
    )
    tail = _render_color_channels(
        geometry,
        torch.cat((probabilities[:, 3:5], torch.zeros_like(probabilities[:, :1])), dim=1),
        view,
        torch.zeros(3, dtype=torch.float32, device=probabilities.device),
    )
    output = torch.cat((rgb_like, tail[:2]), dim=0).clamp_min(0.0)
    return output / output.sum(dim=0, keepdim=True).clamp_min(torch.finfo(output.dtype).tiny)


def render_offset_binary_probabilities(
    view: Any,
    model: Any,
    sidecar: OffsetSemanticResidualSidecar,
    pipe: Any,
    *,
    return_contributions: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    """Render only the protected SEM-230 background/deck competition lane.

    The binary normalisation deliberately happens before rasterization, so
    classes 2--4 cannot steal probability mass from the only mutable lane.
    """
    geometry, visible = frozen_generated_geometry(view, model, pipe)
    offset_logits = sidecar.offset_logits(visible, geometry[4])[:, :2]
    probabilities = torch.softmax(offset_logits, dim=1)
    rendered = _render_color_channels(
        geometry,
        torch.cat((probabilities, torch.zeros_like(probabilities[:, :1])), dim=1),
        view,
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=probabilities.device),
    )[:2].clamp_min(0.0)
    output = rendered / rendered.sum(dim=0, keepdim=True).clamp_min(torch.finfo(rendered.dtype).tiny)
    if not return_contributions:
        return output
    anchors, offsets = active_offset_indices(visible, geometry[4], n_offsets=sidecar.n_offsets)
    residual = sidecar.offset_background_deck_residual[anchors, offsets].detach()
    opacity = geometry[1].detach().reshape(-1).clamp_min(0.0)
    weights = opacity / opacity.sum().clamp_min(torch.finfo(opacity.dtype).tiny)
    stats = {
        "active_generated_offsets": int(opacity.numel()),
        "opacity_sum": float(opacity.sum().cpu()),
        "opacity_weighted_residual_l1": float((weights * residual.abs().sum(dim=1)).sum().cpu()),
    }
    return output, stats
