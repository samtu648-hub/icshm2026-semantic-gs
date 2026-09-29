"""Strict spatial-OOF cross-view evidence for selective cable recovery.

This module deliberately cannot remove a ``stay_cable`` pixel already present
in the hard baseline.  It only decides which of a separately proposed set of
low-confidence recovery pixels may be restored.  Evidence is attached to the
same frozen Scaffold anchor that is dominant at a pixel, and is counted only
from other cameras.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.semantic_protected_fusion import CLASS_COUNT, validate_prediction


def dominant_anchor_map(
    dominant_offset_index: np.ndarray,
    offset_anchor_indices: np.ndarray,
    *,
    anchor_count: int,
) -> np.ndarray:
    """Map each diagnostic pixel to its dominant frozen anchor, or ``-1``.

    The rasterizer reports a generated-offset index, whereas semantic evidence
    is stored per anchor.  This conversion is pure indexing; it does not use
    labels, semantic logits, or RGB values.
    """
    dominant = np.asarray(dominant_offset_index)
    offsets = np.asarray(offset_anchor_indices)
    if int(anchor_count) <= 0:
        raise ValueError("anchor_count must be positive")
    if dominant.ndim != 2 or not np.issubdtype(dominant.dtype, np.integer):
        raise ValueError("dominant offset indices must be a two-dimensional integer map")
    if offsets.ndim != 1 or not np.issubdtype(offsets.dtype, np.integer):
        raise ValueError("offset anchor indices must be a one-dimensional integer vector")
    if offsets.size and (int(offsets.min()) < 0 or int(offsets.max()) >= int(anchor_count)):
        raise ValueError("offset anchor indices fall outside the frozen anchor range")
    result = np.full(dominant.shape, -1, dtype=np.int64)
    valid = (dominant >= 0) & (dominant < offsets.size)
    result[valid] = offsets[dominant[valid]].astype(np.int64, copy=False)
    return result


def view_anchor_cable_evidence(
    anchor_map: np.ndarray,
    cable_probability: np.ndarray,
    *,
    anchor_count: int,
    minimum_dominant_pixels: int,
    minimum_mean_cable_probability: float,
) -> np.ndarray:
    """Return sorted anchors supported as cable by one source camera.

    A source camera supports an anchor only when that anchor owns enough
    dominant pixels in the frozen rasterization and its *mean* OOF class-2
    probability on those pixels reaches the hard-0.30 evidence level.  Mean
    support prevents a few isolated high-confidence pixels from validating an
    otherwise background-like anchor observation.
    """
    anchors = np.asarray(anchor_map)
    probabilities = np.asarray(cable_probability)
    if anchors.ndim != 2 or not np.issubdtype(anchors.dtype, np.integer):
        raise ValueError("anchor_map must be a two-dimensional integer map")
    if probabilities.shape != anchors.shape or probabilities.ndim != 2:
        raise ValueError("cable_probability must match anchor_map")
    if not np.issubdtype(probabilities.dtype, np.floating) or not np.isfinite(probabilities).all():
        raise ValueError("cable_probability must be finite floating point")
    if np.any(probabilities < -1e-6) or np.any(probabilities > 1.0 + 1e-6):
        raise ValueError("cable_probability must lie in [0, 1]")
    if int(anchor_count) <= 0:
        raise ValueError("anchor_count must be positive")
    if int(minimum_dominant_pixels) < 1:
        raise ValueError("minimum_dominant_pixels must be at least one")
    if not 0.0 < float(minimum_mean_cable_probability) < 1.0:
        raise ValueError("minimum_mean_cable_probability must lie strictly between zero and one")
    if np.any(anchors < -1) or np.any(anchors >= int(anchor_count)):
        raise ValueError("anchor_map contains an invalid frozen anchor ID")

    valid = anchors >= 0
    if not np.any(valid):
        return np.empty(0, dtype=np.int64)
    ids, inverse, counts = np.unique(
        anchors[valid].astype(np.int64, copy=False),
        return_inverse=True,
        return_counts=True,
    )
    sums = np.bincount(
        inverse,
        weights=probabilities[valid].astype(np.float64, copy=False),
        minlength=ids.size,
    )
    means = sums / counts
    supported = (counts >= int(minimum_dominant_pixels)) & (
        means >= float(minimum_mean_cable_probability)
    )
    return ids[supported].astype(np.int64, copy=False)


def add_view_support(
    support_counts: np.ndarray,
    *,
    fold: int,
    supported_anchor_ids: np.ndarray,
) -> None:
    """Add one camera's boolean anchor evidence to a fold-local count table."""
    counts = np.asarray(support_counts)
    ids = np.asarray(supported_anchor_ids)
    if counts.ndim != 2 or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("support_counts must be a two-dimensional integer table")
    if not 0 <= int(fold) < counts.shape[0]:
        raise ValueError("fold index falls outside support_counts")
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("supported_anchor_ids must be an integer vector")
    if ids.size == 0:
        return
    if int(ids.min()) < 0 or int(ids.max()) >= counts.shape[1]:
        raise ValueError("supported anchor ID falls outside support_counts")
    if np.unique(ids).size != ids.size:
        raise ValueError("one source view may contribute each anchor at most once")
    maximum = np.iinfo(counts.dtype).max
    if np.any(counts[int(fold), ids] >= maximum):
        raise OverflowError("support count dtype cannot record another source view")
    counts[int(fold), ids] += 1


def other_view_support_map(
    anchor_map: np.ndarray,
    fold_support_counts: np.ndarray,
    current_view_supported_anchor_ids: np.ndarray,
) -> np.ndarray:
    """Return source-view counts for each pixel, excluding the current view."""
    anchors = np.asarray(anchor_map)
    counts = np.asarray(fold_support_counts)
    current = np.asarray(current_view_supported_anchor_ids)
    if anchors.ndim != 2 or not np.issubdtype(anchors.dtype, np.integer):
        raise ValueError("anchor_map must be a two-dimensional integer map")
    if counts.ndim != 1 or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("fold_support_counts must be an integer vector")
    if current.ndim != 1 or not np.issubdtype(current.dtype, np.integer):
        raise ValueError("current_view_supported_anchor_ids must be an integer vector")
    if np.any(anchors < -1) or np.any(anchors >= counts.size):
        raise ValueError("anchor_map contains an ID outside the support table")
    if current.size and (int(current.min()) < 0 or int(current.max()) >= counts.size):
        raise ValueError("current-view support ID falls outside the support table")
    if np.unique(current).size != current.size:
        raise ValueError("current-view support IDs must be unique")

    current_lookup = np.zeros(counts.size, dtype=np.bool_)
    current_lookup[current] = True
    result = np.zeros(anchors.shape, dtype=np.int32)
    valid = anchors >= 0
    if not np.any(valid):
        return result
    values = counts[anchors[valid]].astype(np.int32, copy=False)
    values -= current_lookup[anchors[valid]].astype(np.int32, copy=False)
    if np.any(values < 0):
        raise RuntimeError("current-view evidence is absent from the fold support table")
    result[valid] = values
    return result


def retain_recovery_by_crossview_support(
    recovery_additions: np.ndarray,
    source_view_support: np.ndarray,
    *,
    minimum_source_views: int,
) -> np.ndarray:
    """Keep only proposed recovery pixels with independent source support."""
    additions = np.asarray(recovery_additions)
    support = np.asarray(source_view_support)
    if additions.dtype != np.bool_ or additions.ndim != 2:
        raise ValueError("recovery_additions must be a two-dimensional boolean mask")
    if support.shape != additions.shape or not np.issubdtype(support.dtype, np.integer):
        raise ValueError("source_view_support must be an integer map matching recovery_additions")
    if int(minimum_source_views) < 1:
        raise ValueError("minimum_source_views must be at least one")
    if np.any(support < 0):
        raise ValueError("source_view_support cannot be negative")
    return additions & (support >= int(minimum_source_views))


def apply_retained_recovery(
    hard_prediction: np.ndarray,
    recovery_additions: np.ndarray,
    retained_recovery: np.ndarray,
) -> np.ndarray:
    """Add retained cable pixels to a hard map without ever removing a cable."""
    hard = validate_prediction(hard_prediction, context="SEM-130 hard prediction")
    additions = np.asarray(recovery_additions)
    retained = np.asarray(retained_recovery)
    if additions.dtype != np.bool_ or retained.dtype != np.bool_:
        raise ValueError("SEM-130 recovery masks must be boolean")
    if additions.shape != hard.shape or retained.shape != hard.shape:
        raise ValueError("SEM-130 recovery masks must match hard prediction")
    if np.any(retained & ~additions):
        raise ValueError("SEM-130 retained recovery must be a subset of proposed additions")
    if np.any(hard[additions] != 0):
        raise ValueError("SEM-130 additions must start from hard-background pixels")
    result = hard.copy()
    result[retained] = np.uint8(2)
    if int(result.max()) >= CLASS_COUNT:
        raise RuntimeError("SEM-130 produced a non-official class ID")
    if np.any((hard == 2) & (result != 2)):
        raise RuntimeError("SEM-130 removed a hard-baseline stay_cable pixel")
    return result


def oof_fold_by_name(
    oof_split: dict[str, Any], names: list[str]
) -> tuple[dict[str, int], dict[str, Any]]:
    """Map each training view to a strict spatial-OOF source fold.

    Sources for a target are limited to the other OOF views in its own fold.
    That construction is important: the frozen teacher that predicted every
    such source was trained without all views in the fold, including target.
    """
    expected = set(names)
    mapping: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for record in oof_split.get("folds", []):
        fold = int(record["fold"])
        oof = list(record["oof"])
        train = set(record["train"])
        if not set(oof).isdisjoint(train):
            raise ValueError(f"SEM-130 OOF fold {fold} train/OOF overlap")
        for name in oof:
            if name not in expected:
                raise ValueError(f"SEM-130 OOF source includes a non-training view: {name}")
            if name in mapping:
                raise ValueError(f"SEM-130 OOF view appears in multiple folds: {name}")
            if name in train:
                raise ValueError(f"SEM-130 target appears in its source model training set: {name}")
            mapping[name] = fold
            source_counts[str(fold)] = len(oof) - 1
    if set(mapping) != expected:
        raise ValueError("SEM-130 OOF folds do not cover exactly the authorised training views")
    return mapping, {
        "source_model_rule": "For target view v, sources are all other OOF views in v's spatial fold. The one frozen teacher model for that fold was trained without every fold OOF view, including v.",
        "source_views_exclude_current_camera": True,
        "target_absent_from_its_source_model_training": True,
        "other_source_views_per_fold": source_counts,
    }
