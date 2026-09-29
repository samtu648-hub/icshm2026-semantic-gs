"""SEM-280: five-class foundation boundary-completeness training objective."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.semantic_foundation_binary import fuse_foundation, training_plan, validate_fold


ARMS = ("five_control", "foundation_boundary_candidate")


def foundation_inner_boundary_mask(label: np.ndarray, *, radius: int) -> np.ndarray:
    """Return true-foundation pixels no farther than ``radius`` px inside its edge."""
    import cv2

    if label.dtype != np.uint8 or label.ndim != 2 or not label.size or label.max() > 4:
        raise ValueError("Invalid official semantic label grid")
    if int(radius) != radius or int(radius) < 1:
        raise ValueError("Foundation boundary radius must be a positive integer")
    foundation = (label == 4).astype(np.uint8)
    if not foundation.any():
        return np.zeros(label.shape, dtype=bool)
    distance_inside = cv2.distanceTransform(foundation, cv2.DIST_C, 3)
    return (foundation.astype(bool) & (distance_inside <= int(radius)))


def foundation_boundary_cross_entropy(
    logits: Any,
    targets: Any,
    boundary_mask: Any,
    *,
    class_weights: Any,
    multiplier: float,
) -> tuple[Any, Any]:
    """Five-class CE with one fixed multiplier on true foundation boundary pixels.

    Multiplier 1 deliberately dispatches to the unmodified weighted CE reduction so
    the control arm reproduces the SEM-270 five-class training path exactly.
    """
    import torch
    import torch.nn.functional as functional

    if not isinstance(logits, torch.Tensor) or logits.ndim != 4 or logits.shape[1] != 5:
        raise ValueError("Foundation boundary logits must be Bx5xHxW")
    if not isinstance(targets, torch.Tensor) or targets.ndim != 3:
        raise ValueError("Foundation boundary targets must be BxHxW")
    if tuple(logits.shape[0:1] + logits.shape[2:]) != tuple(targets.shape):
        raise ValueError("Foundation boundary logits and targets do not align")
    if not isinstance(boundary_mask, torch.Tensor) or boundary_mask.shape != targets.shape:
        raise ValueError("Foundation boundary mask and targets do not align")
    if not isinstance(class_weights, torch.Tensor) or class_weights.shape != (5,):
        raise ValueError("Foundation boundary class weights must contain five values")
    if torch.any((targets < 0) | (targets > 4)):
        raise ValueError("Foundation boundary targets must contain official IDs")
    if torch.any(boundary_mask & (targets != 4)):
        raise ValueError("Boundary emphasis may only select true foundation pixels")
    if float(multiplier) < 1.0:
        raise ValueError("Foundation boundary multiplier must be at least one")

    if float(multiplier) == 1.0:
        loss = functional.cross_entropy(logits.float(), targets, weight=class_weights)
        return loss, boundary_mask.to(dtype=torch.float32)

    spatial_weight = 1.0 + (float(multiplier) - 1.0) * boundary_mask.to(
        device=logits.device, dtype=torch.float32
    )
    per_pixel = functional.cross_entropy(
        logits.float(), targets, weight=class_weights, reduction="none"
    )
    effective_weight = class_weights[targets] * spatial_weight
    denominator = effective_weight.sum()
    if not torch.isfinite(denominator) or float(denominator.detach().cpu()) <= 0.0:
        raise FloatingPointError("Foundation boundary loss has an invalid denominator")
    return (per_pixel * spatial_weight).sum() / denominator, boundary_mask.to(
        dtype=torch.float32
    )


def quality_gates(baseline: dict, candidate: dict, control: dict, gates: dict) -> dict[str, bool]:
    base = baseline["per_class"]["foundation"]
    cand = candidate["per_class"]["foundation"]
    ctrl = control["per_class"]["foundation"]
    return {
        "foundation_iou_gain": cand["iou"] >= base["iou"] + gates["foundation_iou_gain"],
        "foundation_precision_gain": cand["precision"]
        >= base["precision"] + gates["foundation_precision_gain"],
        "foundation_recall_retention": cand["recall"]
        >= base["recall"] - gates["maximum_recall_drop"],
        "foundation_recall_floor": cand["recall"] >= gates["recall_floor"],
        "miou_nondecreasing_vs_baseline": candidate["mIoU_with_background"]
        >= baseline["mIoU_with_background"],
        "candidate_recall_improves_control": cand["recall"]
        >= ctrl["recall"] + gates["minimum_recall_gain_vs_control"],
        "candidate_retains_control_iou": cand["iou"]
        >= ctrl["iou"] - gates["maximum_iou_drop_vs_control"],
        "candidate_retains_control_miou": candidate["mIoU_with_background"]
        >= control["mIoU_with_background"] - gates["maximum_miou_drop_vs_control"],
    }


__all__ = [
    "ARMS",
    "foundation_inner_boundary_mask",
    "foundation_boundary_cross_entropy",
    "fuse_foundation",
    "training_plan",
    "validate_fold",
    "quality_gates",
]
