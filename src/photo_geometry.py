"""Frozen photo-correspondence geometry. No Gaussian depths or semantic inputs."""
from itertools import combinations
import numpy as np


def resized_to_colmap(xy, original_wh=(1320, 989), resized_wh=(1024, 768)):
    # LoFTR coordinates index pixel centres at zero; COLMAP has edge origin.
    return (np.asarray(xy, dtype=np.float64) + .5) * (np.array(original_wh) / resized_wh)


def projection(cam):
    return cam.matrix @ np.column_stack((cam.pose.rotation, cam.pose.translation))


def center(cam):
    return -cam.pose.rotation.T @ cam.pose.translation


def project(cam, xyz):
    q = np.asarray(xyz) @ cam.pose.rotation.T + cam.pose.translation
    h = q @ cam.matrix.T
    return h[..., :2] / h[..., 2:3], q[..., 2]


def epipolar_error(a, b, ca, cb):
    r = cb.pose.rotation @ ca.pose.rotation.T
    t = cb.pose.translation - r @ ca.pose.translation
    x, y, z = t
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    f = np.linalg.inv(cb.matrix).T @ skew @ r @ np.linalg.inv(ca.matrix)
    ah = np.column_stack((a, np.ones(len(a))))
    bh = np.column_stack((b, np.ones(len(b))))
    la, lb = bh @ f, ah @ f.T
    v = np.abs(np.sum(bh * lb, axis=1))
    # Both directional point-to-line errors must satisfy the bound.
    return np.maximum(v / np.maximum(np.linalg.norm(la[:, :2], axis=1), 1e-15),
                      v / np.maximum(np.linalg.norm(lb[:, :2], axis=1), 1e-15))


def dlt(cams, xy):
    rows = []
    for cam, (x, y) in zip(cams, xy):
        p = projection(cam)
        rows.extend((x*p[2]-p[0], y*p[2]-p[1]))
    _, _, v = np.linalg.svd(rows, full_matrices=False)
    if abs(v[-1, 3]) < 1e-12:
        raise ValueError('degenerate_dlt')
    return v[-1, :3] / v[-1, 3]


def triangulate_track(observations, cameras):
    """Maximum sum-of-ray-angle triple builds; all remaining cameras check."""
    names = [o['camera'] for o in observations]
    result = {'observations': observations, 'accepted': False}
    if len(set(names)) != len(names):
        return result | {'reason': 'duplicate_camera_conflict'}
    if len(names) < 4:
        return result | {'reason': 'fewer_than_four_cameras'}
    cams = [cameras[n] for n in names]
    xy = np.array([o['xy'] for o in observations])
    rays = []
    for c, uv in zip(cams, xy):
        ray = c.pose.rotation.T @ np.linalg.solve(c.matrix, np.r_[uv, 1.])
        rays.append(ray / np.linalg.norm(ray))
    angles = np.degrees(np.arccos(np.clip(np.array(rays) @ np.array(rays).T, -1, 1)))
    triples = list(combinations(range(len(cams)), 3))
    chosen = min(triples, key=lambda q: (-sum(angles[i, j] for i, j in combinations(q, 2)), q))
    result['build_indices'] = list(chosen)
    result['check_indices'] = [i for i in range(len(cams)) if i not in chosen]
    build = [cams[i] for i in chosen]
    pixels = xy[list(chosen)]
    try:
        xyz = dlt(build, pixels)
        projected = [project(c, xyz) for c in cams]
        depths = np.array([p[1] for p in projected])
        errors = np.linalg.norm(np.array([p[0] for p in projected])-xy, axis=1)
        directions = xyz - np.array([center(c) for c in cams])
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        angle = float(np.degrees(np.arccos(np.clip(directions @ directions.T, -1, 1))).max())
        # Independent 1px standard-deviation perturbations in the six build coordinates.
        jac = []
        for k in range(6):
            plus, minus = pixels.copy(), pixels.copy()
            plus.flat[k] += 1.; minus.flat[k] -= 1.
            jac.append((dlt(build, plus)-dlt(build, minus))/2)
        jac = np.array(jac).T
        cov = jac @ jac.T
        # Depth along each observing ray, relative to range (not renderer depth).
        sigma = np.sqrt(np.maximum(np.einsum('ni,ij,nj->n', directions, cov, directions), 0))
        ranges = np.linalg.norm(xyz-np.array([center(c) for c in cams]), axis=1)
        rel = sigma / ranges
        result.update(xyz=xyz.tolist(), reprojection_error_px=errors.tolist(),
                      max_angle_deg=angle, covariance=cov.tolist(), relative_depth_std=rel.tolist())
        if not np.isfinite(np.r_[xyz, errors, rel]).all(): reason = 'nonfinite'
        elif np.any(depths <= 0): reason = 'nonpositive_depth'
        elif angle < 1: reason = 'low_parallax'
        elif np.any(errors[result['check_indices']] > 2): reason = 'extra_view_reprojection'
        elif np.any(errors[list(chosen)] > 2): reason = 'build_reprojection'
        elif np.any(rel > .02): reason = 'depth_uncertainty'
        else: reason = 'accepted'
    except (ValueError, np.linalg.LinAlgError) as exc:
        reason = str(exc)
    return result | {'accepted': reason == 'accepted', 'reason': reason}
