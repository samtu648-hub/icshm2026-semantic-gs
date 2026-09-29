"""SEM344 hard-label camera contract; never interpolate or recalibrate decisions."""
from dataclasses import replace
import numpy as np
from src.camera_adapter import redistort_semantic, official_to_model_camera
from src.camera_adapter import SimpleRadialCamera


def pinhole(camera, transform):
    return replace(camera, pose=official_to_model_camera(camera.pose, transform), model='PINHOLE', radial_k=0.)


def official_labels(labels, camera):
    if labels.dtype != np.uint8 or labels.shape != (camera.height, camera.width) or labels.max() > 4:
        raise ValueError('Expected native five-class uint8 labels')
    if camera.model == 'PINHOLE':
        return labels.copy()
    if camera.model != 'SIMPLE_RADIAL':
        raise ValueError('Unregistered camera model')
    k = camera.matrix
    c = SimpleRadialCamera(camera.width, camera.height, k[0, 0], k[0, 2], k[1, 2], camera.radial_k)
    return redistort_semantic(labels, c)


def check_protected(raw, selected):
    changed = raw != selected
    if np.any(changed & ((raw != 3) | (selected != 4))):
        raise ValueError('Selection changed protected pixels')
    return int(changed.sum())


def projection_error(camera, transform, points):
    """Compare independent official/model pinhole projections of same 3D points."""
    native = pinhole(camera, transform)
    model_points = transform.scale * (points @ transform.rotation.T) + transform.translation
    official_xyz = points @ camera.pose.rotation.T + camera.pose.translation
    model_xyz = model_points @ native.pose.rotation.T + native.pose.translation
    a, b = official_xyz @ camera.matrix.T, model_xyz @ native.matrix.T
    depth_ok = (a[:, 2] > 1e-3) & (b[:, 2] > 1e-3)
    uv = a[:, :2] / np.where(depth_ok, a[:, 2], 1.)[:, None]
    keep = depth_ok & (uv[:, 0] >= 0) & (uv[:, 0] < camera.width) & (uv[:, 1] >= 0) & (uv[:, 1] < camera.height)
    if not keep.any():
        raise ValueError('No valid projection witnesses')
    return float(np.max(np.abs(a[keep, :2] / a[keep, 2:] - b[keep, :2] / b[keep, 2:])))
