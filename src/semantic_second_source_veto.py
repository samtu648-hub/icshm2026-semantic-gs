"""A second pose-selected source can veto an existing dense foundation deletion."""
from __future__ import annotations
import numpy as np
from src.semantic_dense_review import native_support
from src.semantic_dense_final import inverse_radial


def second_source(target):
    sources, first = target["accessible_sources"], target["selected_source"]
    if len(set(sources)) != len(sources) or (sources[0] if sources else None) != first:
        raise ValueError("Expected sealed first-passing pose order")
    return sources[1] if len(sources) > 1 else None


def review_second_source(baseline, dense, source_labels=None, maps=None, target_pose=None,
                         source_pose=None, target_camera=None, source_camera=None, filters=None,
                         independent=False, target_is_distorted=False, enabled=True):
    """Only previous4-to0 pixels may be restored; no target GT or image IDs accepted."""
    if baseline.shape != dense.shape or baseline.ndim != 2:
        raise ValueError("Expected matching 2D predictions")
    changed = baseline != dense
    if np.any(changed & ((baseline != 4) | (dense != 0))):
        raise ValueError("Only old4-to0 changes can be reviewed")
    out = dense.copy()
    stats = {"old_deletions": int(changed.sum()), "reliable_old_deletions": 0,
             "foundation_veto_pixels": 0, "source_class_counts_reliable": [0] * 5}
    if not enabled or source_labels is None or not np.any(changed):
        return out, stats
    if baseline.shape != (target_camera.height, target_camera.width):
        raise ValueError("Target camera size differs")
    if source_labels.shape != (source_camera.height, source_camera.width):
        raise ValueError("Source label camera size differs")
    yy, xx = np.where(changed)
    points = np.stack((xx + .5, yy + .5), axis=-1)[None]
    if target_is_distorted:
        points = inverse_radial(points, target_camera, independent)
    sp, support = native_support(points, maps, target_pose, source_pose, target_camera,
                                 source_camera, filters, independent)
    valid = (np.isfinite(sp).all(-1) & (sp[..., 0] >= 0) & (sp[..., 0] < source_camera.width) &
             (sp[..., 1] >= 0) & (sp[..., 1] < source_camera.height))
    ij = np.floor(np.nan_to_num(sp, nan=0, posinf=0, neginf=0)).astype(np.int64)
    x = np.clip(ij[..., 0], 0, source_camera.width - 1)
    y = np.clip(ij[..., 1], 0, source_camera.height - 1)
    sampled = np.take(source_labels, y * source_camera.width + x) if independent else source_labels[y, x]
    reliable = (valid & support).ravel(); sampled = sampled.ravel()
    restore = reliable & (sampled == 4)
    if independent:
        restore_map = np.zeros(baseline.shape, bool); restore_map[yy, xx] = restore
        out = np.where(restore_map, baseline, dense).astype(dense.dtype)
    else:
        out[yy[restore], xx[restore]] = 4
    stats["reliable_old_deletions"] = int(reliable.sum())
    stats["foundation_veto_pixels"] = int(restore.sum())
    stats["source_class_counts_reliable"] = np.bincount(sampled[reliable], minlength=5).tolist()
    return out, stats
