"""Fixed deck-interior loss emphasis for SEM-172."""

from __future__ import annotations

from typing import Any


def deck_interior_mask(targets: Any, *, radius: int) -> Any:
    """Return target deck pixels at least ``radius`` pixels from non-deck."""
    import torch
    import torch.nn.functional as functional

    if not isinstance(targets, torch.Tensor) or targets.ndim != 3:
        raise ValueError("Deck interior targets must be a BxHxW tensor")
    if int(radius) != radius or int(radius) < 1:
        raise ValueError("Deck interior erosion radius must be a positive integer")
    if torch.any((targets < 0) | (targets > 1)):
        raise ValueError("Deck interior targets must contain binary IDs")
    radius = int(radius)
    deck = (targets == 1).to(dtype=torch.float32).unsqueeze(1)
    non_deck = 1.0 - deck
    dilated_non_deck = functional.max_pool2d(
        non_deck,
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
    )
    return (deck * (1.0 - dilated_non_deck)).squeeze(1)


def interior_weighted_cross_entropy(
    logits: Any,
    targets: Any,
    *,
    class_weights: Any,
    radius: int,
    multiplier: float,
) -> tuple[Any, Any]:
    """Weighted CE with one fixed extra multiplier on eroded deck interior."""
    import torch
    import torch.nn.functional as functional

    if not isinstance(logits, torch.Tensor) or logits.ndim != 4:
        raise ValueError("Deck interior logits must be a Bx2xHxW tensor")
    if logits.shape[1] != 2 or tuple(logits.shape[0:1] + logits.shape[2:]) != tuple(
        targets.shape
    ):
        raise ValueError("Deck interior logits and targets do not align")
    if not isinstance(class_weights, torch.Tensor) or class_weights.shape != (2,):
        raise ValueError("Deck interior class weights must contain two values")
    if float(multiplier) < 1.0:
        raise ValueError("Deck interior multiplier must be at least one")
    interior = deck_interior_mask(targets, radius=radius).to(logits.device)
    spatial_weight = 1.0 + (float(multiplier) - 1.0) * interior
    per_pixel = functional.cross_entropy(
        logits.float(), targets, weight=class_weights, reduction="none"
    )
    effective_weight = class_weights[targets] * spatial_weight
    denominator = effective_weight.sum()
    if not torch.isfinite(denominator) or float(denominator.detach().cpu()) <= 0.0:
        raise FloatingPointError("Deck interior loss has an invalid denominator")
    loss = (per_pixel * spatial_weight).sum() / denominator
    return loss, interior


__all__ = ["deck_interior_mask", "interior_weighted_cross_entropy"]
