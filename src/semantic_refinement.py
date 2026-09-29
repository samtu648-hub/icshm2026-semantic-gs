"""Development-only error budgets and a frozen-backbone point residual head.

This is a PointRend-inspired adaptation, not an implementation of its adaptive
subdivision algorithm. No target label is an inference input.
"""
from __future__ import annotations

import numpy as np

from src.semantic_protected_fusion import validate_prediction


def mutable_lane(base: np.ndarray, static: np.ndarray) -> np.ndarray:
    """Allow BOTH background/deck directions, without dilating protection.

    Original foundation pixels rejected to background remain protected. Class
    2/3/4 prediction masks are exact; a pre-existing deck pixel is NOT protected.
    """
    validate_prediction(base, context="refinement base")
    validate_prediction(static, context="refinement static")
    if base.shape != static.shape:
        raise ValueError("Refinement grids differ")
    return (base <= 1) & ~((base == 0) & (static == 4))


def fuse_residual(base: np.ndarray, static: np.ndarray, residual: np.ndarray,
                  margin: float = 2.0) -> np.ndarray:
    if residual.shape != base.shape or not np.isfinite(residual).all():
        raise ValueError("Invalid point residual")
    if margin <= 0:
        raise ValueError("Baseline margin must be positive")
    lane = mutable_lane(base, static)
    signed = np.where(base == 1, margin, -margin) + residual
    out = base.copy()
    out[lane] = (signed[lane] > 0).astype(np.uint8)
    for class_id in (2, 3, 4):
        if not np.array_equal(out == class_id, base == class_id):
            raise RuntimeError("Refinement changed a protected class")
    return out


def class_boundary_distance(target: np.ndarray, class_id: int) -> np.ndarray:
    """Chessboard pixel distance to either side of a true class boundary."""
    import cv2
    mask = (target == class_id).astype(np.uint8)
    if not mask.any() or mask.all():
        return np.full(mask.shape, np.inf, np.float32)
    return np.maximum(cv2.distanceTransform(mask, cv2.DIST_C, 3),
                      cv2.distanceTransform(1 - mask, cv2.DIST_C, 3))


def point_head(input_channels: int = 75, hidden: int = 64):
    """Fresh, zero-output point MLP. Only these parameters are trainable."""
    import torch
    model = torch.nn.Sequential(torch.nn.Linear(input_channels, hidden),
                                torch.nn.ReLU(),
                                torch.nn.Linear(hidden, hidden),
                                torch.nn.ReLU(), torch.nn.Linear(hidden, 1))
    torch.nn.init.zeros_(model[-1].weight)
    torch.nn.init.zeros_(model[-1].bias)
    return model


def mask_fine_features(features, arm: str):
    """Equal-size control removes 64 fine channels and 3 native RGB channels."""
    if features.shape[-1] != 75 or arm not in ("coarse_control", "fine_candidate"):
        raise ValueError("Unknown refinement input/arm")
    if arm == "fine_candidate":
        return features
    copy = features.clone()
    copy[..., :67] = 0
    return copy
