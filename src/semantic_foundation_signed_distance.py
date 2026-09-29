"""SEM-300: signed-distance supervision for a five-class foundation branch."""

from __future__ import annotations

from typing import Any

import numpy as np


ARMS = ("five_control", "foundation_signed_distance_candidate")


def foundation_signed_distance(label: np.ndarray, *, truncation_pixels: int) -> np.ndarray:
    """Return a clipped foundation signed-distance target in [-1, 1].

    Positive values are inside foundation, negative values outside. Both sides
    of the class boundary are supervised; the target does not use predictions
    or any inference-time morphology.
    """
    import cv2

    if label.dtype != np.uint8 or label.ndim != 2 or not label.size or label.max() > 4:
        raise ValueError("Invalid official semantic label grid")
    if int(truncation_pixels) != truncation_pixels or int(truncation_pixels) < 1:
        raise ValueError("Signed-distance truncation must be a positive integer")
    foundation = (label == 4).astype(np.uint8)
    inside = cv2.distanceTransform(foundation, cv2.DIST_C, 3)
    outside = cv2.distanceTransform(1 - foundation, cv2.DIST_C, 3)
    signed = inside - outside
    return np.clip(signed / float(truncation_pixels), -1.0, 1.0).astype(np.float32)


def five_class_signed_distance_loss(
    logits: Any,
    signed_distance: Any,
    targets: Any,
    distance_targets: Any,
    *,
    class_weights: Any,
    auxiliary_weight: float,
    smooth_l1_beta: float,
) -> tuple[Any, dict[str, Any]]:
    """Return weighted five-class CE plus an optional two-sided SDF objective."""
    import torch
    import torch.nn.functional as functional

    if not isinstance(logits, torch.Tensor) or logits.ndim != 4 or logits.shape[1] != 5:
        raise ValueError("Five-class logits must be Bx5xHxW")
    if not isinstance(signed_distance, torch.Tensor) or signed_distance.ndim != 4:
        raise ValueError("Signed-distance prediction must be Bx1xHxW")
    if signed_distance.shape[1] != 1 or signed_distance.shape[0] != logits.shape[0]:
        raise ValueError("Signed-distance prediction does not align with logits")
    if not isinstance(targets, torch.Tensor) or targets.shape != logits.shape[:1] + logits.shape[2:]:
        raise ValueError("Class targets do not align with logits")
    if not isinstance(distance_targets, torch.Tensor) or distance_targets.shape != signed_distance.shape:
        raise ValueError("Distance targets do not align with prediction")
    if not isinstance(class_weights, torch.Tensor) or class_weights.shape != (5,):
        raise ValueError("Class weights must contain five entries")
    if torch.any((targets < 0) | (targets > 4)):
        raise ValueError("Targets must contain official class IDs")
    if not torch.isfinite(distance_targets).all() or torch.any(torch.abs(distance_targets) > 1.0):
        raise ValueError("Distance targets must be finite and clipped to [-1, 1]")
    if float(auxiliary_weight) < 0.0 or float(smooth_l1_beta) <= 0.0:
        raise ValueError("Invalid signed-distance loss settings")

    ce = functional.cross_entropy(logits.float(), targets, weight=class_weights)
    if float(auxiliary_weight) == 0.0:
        zero = signed_distance.sum() * 0.0
        return ce, {"cross_entropy": ce.detach(), "signed_distance": zero.detach()}
    distance_loss = functional.smooth_l1_loss(
        signed_distance.float(), distance_targets.float(), beta=float(smooth_l1_beta)
    )
    return ce + float(auxiliary_weight) * distance_loss, {
        "cross_entropy": ce.detach(),
        "signed_distance": distance_loss.detach(),
    }


__all__ = ["ARMS", "foundation_signed_distance", "five_class_signed_distance_loss"]
