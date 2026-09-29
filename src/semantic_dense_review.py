"""Native-pixel, source-only review of frozen foundation predictions."""
from __future__ import annotations
import numpy as np
import cv2
from src.semantic_dense_geometry import (bilinear_sample, fundamental_from_w2c,
                                         symmetric_epipolar_pixels)


def review_foundation(baseline, source_labels, reliable):
    if baseline.shape != source_labels.shape or baseline.shape != reliable.shape:
        raise ValueError("Review shape mismatch")
    output = baseline.copy()
    output[(baseline == 4) & (source_labels == 0) & reliable] = 0
    return output


def native_support(points, maps, target_pose, source_pose, target_camera, source_camera,
                   filters, independent=False):
    """Return source PINHOLE coordinates and support; accepts poses, never image IDs/GT.

    points are target PINHOLE pixel centers in edge-origin coordinates. Forward
    maps still have RoMa's 512 grid; interpolation is evaluated BEFORE filtering.
    """
    a, b, ca, cb = target_pose, source_pose, target_camera, source_camera
    if ca.model != "SIMPLE_RADIAL" or cb.model != "SIMPLE_RADIAL":
        raise ValueError("Expected official SIMPLE_RADIAL metadata")
    (ab, oa), (ba, ob) = maps
    q = points / [ca.width / 2, ca.height / 2] - 1
    if independent:
        import torch
        import torch.nn.functional as functional
        def sample(values, grid):
            scalar = values.ndim == 2
            values = values[..., None] if scalar else values
            height, width = values.shape[:2]
            valid = (np.isfinite(grid).all(-1) &
                     (np.abs(grid[..., 0]) <= 1 - 1 / width) &
                     (np.abs(grid[..., 1]) <= 1 - 1 / height))
            safe = np.nan_to_num(grid)
            result = functional.grid_sample(
                torch.from_numpy(np.ascontiguousarray(values)).permute(2, 0, 1)[None],
                torch.from_numpy(np.ascontiguousarray(safe))[None], mode="bilinear",
                padding_mode="border", align_corners=False)[0].permute(1, 2, 0).numpy()
            return (result[..., 0] if scalar else result), valid
    else:
        sample = bilinear_sample
    forward, valid_f = sample(ab, q)
    overlap, valid_o = sample(oa, q)
    backward, valid_b = sample(ba, forward)
    overlap_b, valid_ob = sample(ob, forward)
    cycle = np.linalg.norm((backward - q) * [ca.width / 2, ca.height / 2], axis=-1)
    valid = valid_f & valid_o & valid_b & valid_ob & np.isfinite(cycle)
    pre = (valid & (overlap >= filters["overlap_each_direction_minimum"]) &
           (overlap_b >= filters["overlap_each_direction_minimum"]) &
           (cycle <= filters["cycle_maximum_native_pixels"]))
    raw = (forward + 1) * [cb.width / 2, cb.height / 2]
    if independent:
        f, cx, cy, k = cb.params
        distorted = (raw - [cx, cy]) / f
        undistorted = distorted.copy()
        for _ in range(15):
            undistorted = distorted / (1 + k * (undistorted ** 2).sum(-1, keepdims=True))
        source_points = undistorted * f + [cx, cy]
        ta, tb = np.eye(4), np.eye(4)
        for transform, pose in ((ta, a), (tb, b)):
            transform[:3, :3], transform[:3, 3] = pose.rotation_world_to_camera, pose.tvec
        relative = tb @ np.linalg.inv(ta)
        x, y, z = relative[:3, 3]
        if np.linalg.norm(relative[:3, 3]) < 1e-10:
            raise ValueError("Degenerate baseline")
        essential = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]]) @ relative[:3, :3]
        def homogeneous_normalized(p, camera):
            focal, ox, oy, _ = camera.params
            xy = (p - [ox, oy]) / focal
            return np.concatenate((xy, np.ones((*xy.shape[:-1], 1))), axis=-1)
        aa, bb = homogeneous_normalized(points, ca), homogeneous_normalized(source_points, cb)
        lb, la = aa @ essential.T, bb @ essential
        residual = np.abs((bb * lb).sum(-1))
        epi = np.maximum(ca.params[0] * residual / np.maximum(np.linalg.norm(la[..., :2], axis=-1), 1e-15),
                         cb.params[0] * residual / np.maximum(np.linalg.norm(lb[..., :2], axis=-1), 1e-15))
    else:
        def matrix(c):
            focal, cx, cy, k = c.params
            return np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1.]]), np.array([k, 0., 0., 0., 0.])
        ka, _ = matrix(ca); kb, db = matrix(cb)
        source_points = cv2.undistortPoints(raw.reshape(-1, 1, 2), kb, db, P=kb).reshape(raw.shape)
        fundamental = fundamental_from_w2c(a.rotation_world_to_camera, a.tvec,
                                           b.rotation_world_to_camera, b.tvec, ka, kb)
        epi = symmetric_epipolar_pixels(points, source_points, fundamental)
    return source_points, pre & (epi <= filters["epipolar_maximum_native_pixels"])


def native_prediction(baseline, source_labels, maps, target_pose, source_pose,
                      target_camera, source_camera, filters, independent=False, rows_per_chunk=64):
    """No target labels are accepted by the inference interface."""
    if baseline.shape != (target_camera.height, target_camera.width):
        raise ValueError("Native baseline/camera dimensions differ")
    if source_labels.shape != (source_camera.height, source_camera.width):
        raise ValueError("Source PINHOLE labels/camera dimensions differ")
    result = baseline.copy()
    stats = {"reliable_pixels": 0, "outside_source_labels": 0, "changed_pixels": 0}
    for start in range(0, target_camera.height, rows_per_chunk):
        end = min(start + rows_per_chunk, target_camera.height)
        y, x = np.mgrid[start:end, :target_camera.width]
        points = np.stack((x + .5, y + .5), axis=-1)
        source_points, support = native_support(points, maps, target_pose, source_pose,
                                                target_camera, source_camera, filters, independent)
        valid = (np.isfinite(source_points).all(-1) & (source_points[..., 0] >= 0) &
                 (source_points[..., 0] < source_camera.width) & (source_points[..., 1] >= 0) &
                 (source_points[..., 1] < source_camera.height))
        safe = np.floor(np.nan_to_num(source_points)).astype(np.int64)
        if independent:
            # Flat indexed gathering independently implements edge-origin nearest-center sampling.
            index = np.clip(safe[..., 1], 0, source_camera.height - 1) * source_camera.width + np.clip(safe[..., 0], 0, source_camera.width - 1)
            sampled = np.take(source_labels, index)
            changed = (baseline[start:end] == 4) & valid & support & (sampled == 0)
            candidate = np.where(changed, np.uint8(0), baseline[start:end])
        else:
            sampled = source_labels[np.clip(safe[..., 1], 0, source_camera.height - 1),
                                    np.clip(safe[..., 0], 0, source_camera.width - 1)]
            candidate = review_foundation(baseline[start:end], sampled, support & valid)
        result[start:end] = candidate
        stats["reliable_pixels"] += int(support.sum())
        stats["outside_source_labels"] += int((support & ~valid).sum())
        stats["changed_pixels"] += int((candidate != baseline[start:end]).sum())
    return result, stats


def quality_gates(summary, folds, gates):
    return {
        "miou_gain": summary["delta_miou_pp"] >= gates["minimum_miou_delta_pp"],
        "foundation_iou_gain": summary["delta_foundation_iou_pp"] >= gates["minimum_foundation_iou_delta_pp"],
        "foundation_recall_loss": summary["delta_foundation_recall_pp"] >= -gates["maximum_foundation_recall_loss_pp"],
        "foundation_recall_floor": summary["candidate"]["per_class"]["foundation"]["recall"] >= gates["minimum_foundation_recall"],
        "improving_folds": sum(f["delta_foundation_iou_pp"] > 0 for f in folds) >= gates["minimum_improving_foundation_folds"],
        "fold_foundation_floor": all(f["delta_foundation_iou_pp"] >= gates["minimum_fold_foundation_iou_delta_pp"] for f in folds),
        "fold_miou_floor": all(f["delta_miou_pp"] >= gates["minimum_fold_miou_delta_pp"] for f in folds),
    }
