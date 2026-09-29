"""Source-label boundary evidence; no target GT is accepted by these helpers."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree


def foundation_distances(labels):
    """Distance between native PINHOLE label pixel centers, not polygon edges."""
    if labels.ndim != 2:
        raise ValueError("Expected two-dimensional source labels")
    if not np.any(labels == 4):
        return np.full(labels.shape, np.inf, dtype=np.float64)
    return distance_transform_edt(labels != 4)


def sample_boundary_distance(source_points, labels, distances, independent=False):
    if source_points.shape[-1] != 2 or labels.shape != distances.shape:
        raise ValueError("Source boundary shape mismatch")
    height, width = labels.shape
    valid = (np.isfinite(source_points).all(-1) & (source_points[..., 0] >= 0) &
             (source_points[..., 0] < width) & (source_points[..., 1] >= 0) &
             (source_points[..., 1] < height))
    indices = np.floor(np.nan_to_num(source_points, nan=0, posinf=0, neginf=0)).astype(np.int64)
    x = np.clip(indices[..., 0], 0, width - 1)
    y = np.clip(indices[..., 1], 0, height - 1)
    sampled = labels[y, x]
    values = distances[y, x]
    if independent:
        fy, fx = np.where(labels == 4)
        if fx.size:
            tree = cKDTree(np.column_stack((fx, fy)))
            values = tree.query(np.stack((x, y), axis=-1).reshape(-1, 2), workers=1)[0].reshape(x.shape)
        else:
            values = np.full(x.shape, np.inf)
    return sampled, np.where(valid, values, np.inf), valid


def boundary_histogram(distances, gt, upper_bounds):
    """Descriptive disjoint bins (previous bound, upper bound], with overflow."""
    limits = np.asarray(upper_bounds, dtype=float)
    if np.any(~np.isfinite(limits)) or np.any(np.diff(limits) <= 0) or limits[0] <= 0:
        raise ValueError("Expected positive increasing bin limits")
    d, classes = np.asarray(distances).ravel(), np.asarray(gt).ravel()
    if d.shape != classes.shape or np.any(d < 0) or np.any(np.isnan(d)):
        raise ValueError("Invalid distance samples")
    if np.any((classes < 0) | (classes > 4)):
        raise ValueError("Invalid GT class")
    ids = np.searchsorted(limits, d, side="left")
    out = np.zeros((len(limits) + 1, 5), np.int64)
    np.add.at(out, (ids, classes), 1)
    return out


def protect_boundary(baseline, dense, distances, radius):
    """A diagnostic or inference caller supplies distances only at old4-to0 pixels."""
    if radius < 0 or not np.isfinite(radius) or baseline.shape != dense.shape:
        raise ValueError("Invalid protection contract")
    changed = baseline != dense
    if np.any(changed & ((baseline != 4) | (dense != 0))) or distances.size != int(changed.sum()):
        raise ValueError("Expected only dense4-to0 changes and one distance per change")
    if np.any(np.isnan(distances)) or np.any(distances < 0):
        raise ValueError("Invalid distance values")
    output = dense.copy()
    restore = np.zeros(baseline.shape, bool)
    restore[changed] = np.asarray(distances).ravel() <= radius
    output[restore] = 4
    return output, restore
