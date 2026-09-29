"""Unlearned dense geometry diagnostics in explicit native pixel coordinates."""
from __future__ import annotations
import numpy as np


def normalized_grid(height, width):
    y, x = np.meshgrid((np.arange(height) + .5) * 2 / height - 1,
                       (np.arange(width) + .5) * 2 / width - 1, indexing="ij")
    return np.stack((x, y), axis=-1)


def bilinear_sample(array, normalized):
    """NumPy align_corners=False sampler; invalid positions remain NaN."""
    h, w = array.shape[:2]
    x = (normalized[..., 0] + 1) * w / 2 - .5
    y = (normalized[..., 1] + 1) * h / 2 - .5
    valid = np.isfinite(normalized).all(-1) & (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    safe_x = np.clip(np.nan_to_num(x), 0, w - 1)
    safe_y = np.clip(np.nan_to_num(y), 0, h - 1)
    x0, y0 = np.floor(safe_x).astype(int), np.floor(safe_y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    dx, dy = safe_x - x0, safe_y - y0
    trailing = (None,) * (array.ndim - 2)
    expand = lambda a: a[(...,) + trailing]
    sampled = (array[y0, x0] * expand((1 - dx) * (1 - dy)) +
               array[y0, x1] * expand(dx * (1 - dy)) +
               array[y1, x0] * expand((1 - dx) * dy) +
               array[y1, x1] * expand(dx * dy))
    return np.where(expand(valid), sampled, np.nan), valid


def cycle_support(warp_ab, overlap_ab, warp_ba, overlap_ba, native_size, overlap_min, cycle_max):
    backward, valid = bilinear_sample(warp_ba, warp_ab)
    overlap_back, valid_overlap = bilinear_sample(overlap_ba, warp_ab)
    scale = np.asarray(native_size, dtype=np.float64) / 2  # width,height
    grid = normalized_grid(*warp_ab.shape[:2])
    distance = np.linalg.norm((backward - grid) * scale, axis=-1)
    valid &= valid_overlap & np.isfinite(backward).all(-1)
    confident = valid & (overlap_ab >= overlap_min) & (overlap_back >= overlap_min)
    return {"cycle": distance, "valid": valid, "confident": confident,
            "cycle_kept": confident & (distance <= cycle_max)}


def fundamental_from_w2c(rotation_a, translation_a, rotation_b, translation_b, intrinsics_a, intrinsics_b):
    rotation = rotation_b @ rotation_a.T
    t = translation_b - rotation @ translation_a
    if np.linalg.norm(t) < 1e-10:
        raise ValueError("Degenerate epipolar baseline")
    cross = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    fundamental = np.linalg.inv(intrinsics_b).T @ cross @ rotation @ np.linalg.inv(intrinsics_a)
    return fundamental / np.linalg.norm(fundamental)


def symmetric_epipolar_pixels(points_a, points_b, fundamental):
    a = np.concatenate((points_a, np.ones((*points_a.shape[:-1], 1))), axis=-1)
    b = np.concatenate((points_b, np.ones((*points_b.shape[:-1], 1))), axis=-1)
    line_b, line_a = a @ fundamental.T, b @ fundamental
    numerator = np.abs((b * line_b).sum(-1))
    distance_a = numerator / np.maximum(np.linalg.norm(line_a[..., :2], axis=-1), 1e-15)
    distance_b = numerator / np.maximum(np.linalg.norm(line_b[..., :2], axis=-1), 1e-15)
    return np.maximum(distance_a, distance_b)


def screen_pair(reliable_fraction, epipolar_inlier_fraction, matched_median, identity_median, gates):
    return {"coverage": reliable_fraction >= gates["minimum_reliable_grid_fraction_each_pair"],
            "epipolar_retention": epipolar_inlier_fraction is not None and epipolar_inlier_fraction >= gates["minimum_epipolar_inlier_fraction_after_cycle_each_pair"],
            "beats_identity_factor2": matched_median is not None and identity_median is not None and identity_median > 0 and matched_median <= .5 * identity_median}
