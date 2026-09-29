"""Restricted first-order observation-direction residual for SEM-004 / S3.

The frozen SEM-001 five-class anchor logits remain ordinary tensors rather
than parameters.  The only trainable object is a three-vector for each anchor
listed by the frozen SEM-004-DIAG artifact.  The residual transfers logit mass
only between ``background`` and ``stay_cable``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch

from src.scaffold_semantic_sidecar import _render_color_channels, frozen_generated_geometry


CLASS_COUNT = 5
BACKGROUND = 0
STAY_CABLE = 2


def load_direction_anchor_indices(path: Path, *, expected_count: int, anchor_count: int) -> list[int]:
    """Load and strictly validate the immutable SEM-004-DIAG anchor list."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing SEM-004-DIAG direction anchor list: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "anchor_index" not in reader.fieldnames:
            raise ValueError("Direction anchor CSV has no anchor_index column")
        indices = [int(row["anchor_index"]) for row in reader]
    if len(indices) != expected_count:
        raise ValueError(f"Expected {expected_count} direction anchors, found {len(indices)}")
    if len(set(indices)) != len(indices):
        raise ValueError("Direction anchor CSV contains duplicate anchor indices")
    if indices != sorted(indices):
        raise ValueError("Direction anchor CSV must be deterministically sorted")
    if indices and (indices[0] < 0 or indices[-1] >= anchor_count):
        raise ValueError("Direction anchor CSV contains an index outside the frozen RGB anchor range")
    return indices


class DirectionalResidualSidecar:
    """Frozen SEM-001 logits plus a sparse, anchor-local first-order residual."""

    def __init__(
        self,
        sem001_logits: torch.Tensor,
        selected_anchor_indices: list[int] | torch.Tensor,
        *,
        cap: float,
        device: torch.device | str | None = None,
    ) -> None:
        if sem001_logits.ndim != 2 or sem001_logits.shape[1] != CLASS_COUNT:
            raise ValueError("SEM-001 logits must have shape [anchors, 5]")
        if not 0.0 < float(cap):
            raise ValueError("Residual cap must be positive")
        resolved = sem001_logits.device if device is None else torch.device(device)
        indices = torch.as_tensor(selected_anchor_indices, dtype=torch.long, device=resolved)
        if indices.ndim != 1 or indices.numel() == 0:
            raise ValueError("Selected direction anchors must be a non-empty 1-D list")
        if int(indices.min()) < 0 or int(indices.max()) >= int(sem001_logits.shape[0]):
            raise ValueError("Selected direction anchor is outside the SEM-001 anchor range")
        if int(torch.unique(indices).numel()) != int(indices.numel()):
            raise ValueError("Selected direction anchors contain duplicates")

        # These tensors are immutable inputs, deliberately not Parameters.
        self.base_logits = sem001_logits.detach().to(device=resolved, dtype=torch.float32).clone()
        self.selected_anchor_indices = indices.clone()
        self.cap = float(cap)
        dense_to_selected = torch.full((sem001_logits.shape[0],), -1, dtype=torch.long, device=resolved)
        dense_to_selected[self.selected_anchor_indices] = torch.arange(indices.numel(), device=resolved)
        self.dense_to_selected = dense_to_selected

        # Exactly three trainable coefficients per enabled anchor; zero is the
        # required SEM-001-equivalent initialization.
        self.direction_coefficients = torch.nn.Parameter(
            torch.zeros((indices.numel(), 3), dtype=torch.float32, device=resolved)
        )

    @property
    def anchor_count(self) -> int:
        return int(self.base_logits.shape[0])

    @property
    def selected_count(self) -> int:
        return int(self.selected_anchor_indices.numel())

    def adjusted_visible_logits(
        self,
        anchor_positions: torch.Tensor,
        visible: torch.Tensor,
        camera_center: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return view-conditioned logits for the currently visible anchors.

        The official-frame observation direction follows the pre-registration:
        ``normalize(anchor_position - camera_center)``.
        """
        if visible.dtype != torch.bool or visible.ndim != 1 or visible.shape[0] != self.anchor_count:
            raise ValueError("visible must be a bool mask matching the frozen anchor count")
        if anchor_positions.ndim != 2 or tuple(anchor_positions.shape) != (self.anchor_count, 3):
            raise ValueError("anchor_positions must have shape [anchors, 3]")
        if camera_center.numel() != 3:
            raise ValueError("camera_center must contain three coordinates")

        global_indices = torch.nonzero(visible, as_tuple=False).squeeze(1)
        base = self.base_logits[global_indices]
        directions = torch.nn.functional.normalize(
            anchor_positions[global_indices] - camera_center.reshape(1, 3), dim=1, eps=1e-12
        )
        selected_rows = self.dense_to_selected[global_indices]
        enabled = selected_rows >= 0
        gathered = self.direction_coefficients[selected_rows.clamp_min(0)]
        raw = (gathered * directions).sum(dim=1)
        delta = self.cap * torch.tanh(raw)
        delta = delta * enabled.to(dtype=delta.dtype)

        adjusted = torch.stack(
            (
                base[:, BACKGROUND] - delta,
                base[:, 1],
                base[:, STAY_CABLE] + delta,
                base[:, 3],
                base[:, 4],
            ),
            dim=1,
        )
        return adjusted, {
            "global_anchor_indices": global_indices,
            "enabled": enabled,
            "directions": directions,
            "delta": delta,
        }

    def checkpoint_state(self) -> dict[str, torch.Tensor]:
        return {
            "selected_anchor_indices": self.selected_anchor_indices.detach().cpu(),
            "direction_coefficients": self.direction_coefficients.detach().cpu(),
        }

    def load_checkpoint_state(self, payload: dict[str, Any]) -> None:
        stored_indices = payload.get("selected_anchor_indices")
        stored_coefficients = payload.get("direction_coefficients")
        if not isinstance(stored_indices, torch.Tensor) or not torch.equal(
            stored_indices.to(dtype=torch.long, device="cpu"), self.selected_anchor_indices.detach().cpu()
        ):
            raise ValueError("Residual checkpoint anchor mask differs from SEM-004-DIAG")
        if not isinstance(stored_coefficients, torch.Tensor) or tuple(stored_coefficients.shape) != tuple(
            self.direction_coefficients.shape
        ):
            raise ValueError("Residual checkpoint coefficient shape mismatch")
        with torch.no_grad():
            self.direction_coefficients.copy_(
                stored_coefficients.to(device=self.direction_coefficients.device, dtype=torch.float32)
            )


def render_directional_semantic_probabilities(
    view: Any,
    model: Any,
    sidecar: DirectionalResidualSidecar,
    pipe: Any,
    *,
    return_stats: bool = False,
):
    """Render five probabilities through the unchanged SEM-001 rasterization."""
    geometry, visible = frozen_generated_geometry(view, model, pipe)
    active_offsets = geometry[4]
    anchor_logits, stats = sidecar.adjusted_visible_logits(model.get_anchor.detach(), visible, view.camera_center)
    offset_logits = anchor_logits.repeat_interleave(model.n_offsets, dim=0)[active_offsets]
    probabilities = torch.softmax(offset_logits, dim=1)
    rgb_like = _render_color_channels(
        geometry,
        probabilities[:, :3],
        view,
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=probabilities.device),
    )
    last_two_as_rgb = torch.cat((probabilities[:, 3:5], torch.zeros_like(probabilities[:, :1])), dim=1)
    tail = _render_color_channels(
        geometry, last_two_as_rgb, view, torch.zeros(3, dtype=torch.float32, device=probabilities.device)
    )
    output = torch.cat((rgb_like, tail[:2]), dim=0).clamp_min(0.0)
    output = output / output.sum(dim=0, keepdim=True).clamp_min(torch.finfo(output.dtype).tiny)
    return (output, stats) if return_stats else output
