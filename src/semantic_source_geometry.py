"""Fixed SEM321 geometry formula and an independent recomputation for SEM322."""
from __future__ import annotations
import numpy as np
import cv2
from src.semantic_dense_geometry import (normalized_grid, cycle_support, fundamental_from_w2c,
                                         symmetric_epipolar_pixels, screen_pair)


def statistics(values):
    values = values[np.isfinite(values)]
    return {"count": int(values.size), "median": float(np.median(values)) if values.size else None,
            "p90": float(np.quantile(values, .9)) if values.size else None}


def geometry_row(pair, maps, images, cameras, cfg, independent=False):
    (ab, oa), (ba, ob) = maps
    a, b = images[pair["target"]], images[pair["source"]]
    ca, cb = cameras[a.camera_id], cameras[b.camera_id]
    if ca.model != "SIMPLE_RADIAL" or cb.model != "SIMPLE_RADIAL":
        raise ValueError("Unexpected camera model")
    filters = cfg["filters"]
    if independent:
        # This branch uses neither primary interpolation nor its F/undistort helpers.
        import torch
        import torch.nn.functional as functional
        h, w = oa.shape
        grid_tensor = torch.from_numpy(ab[None])
        def sample(values):
            return functional.grid_sample(torch.from_numpy(values).permute(2, 0, 1)[None], grid_tensor,
                mode="bilinear", padding_mode="border", align_corners=False)[0].permute(1, 2, 0).numpy()
        back, overlap_back = sample(ba), sample(ob[..., None])[..., 0]
        valid = (np.abs(ab[..., 0]) <= 1 - 1 / w) & (np.abs(ab[..., 1]) <= 1 - 1 / h)
        y, x = np.mgrid[:h, :w]
        grid = np.stack(((x + .5) * 2 / w - 1, (y + .5) * 2 / h - 1), -1)
        cycle = np.sqrt(np.sum(((back - grid) * [ca.width / 2, ca.height / 2]) ** 2, axis=-1))
        confident = valid & (oa >= filters["overlap_each_direction_minimum"]) & (overlap_back >= filters["overlap_each_direction_minimum"])
        pre = confident & (cycle <= filters["cycle_maximum_native_pixels"])
        ta, tb = np.eye(4), np.eye(4)
        for matrix, image in ((ta, a), (tb, b)):
            matrix[:3, :3], matrix[:3, 3] = image.rotation_world_to_camera, image.tvec
        relative = tb @ np.linalg.inv(ta)
        tx, ty, tz = relative[:3, 3]
        if np.linalg.norm(relative[:3, 3]) < 1e-10:
            raise ValueError("Degenerate baseline")
        essential = np.array([[0, -tz, ty], [tz, 0, -tx], [-ty, tx, 0]]) @ relative[:3, :3]
        def normalized(points, camera, distorted):
            f, cx, cy, k = camera.params
            xd = (points - [cx, cy]) / f
            out = xd.copy()
            if distorted:
                for _ in range(15): out = xd / (1 + k * (out * out).sum(-1, keepdims=True))
            return np.concatenate((out, np.ones((*out.shape[:-1], 1))), -1)
        aa = normalized((grid + 1) * [ca.width / 2, ca.height / 2], ca, False)
        def epi_for(source_grid):
            bb = normalized((source_grid + 1) * [cb.width / 2, cb.height / 2], cb, True)
            lb, la = aa @ essential.T, bb @ essential
            residual = np.abs((bb * lb).sum(-1))
            return np.maximum(ca.params[0] * residual / np.maximum(np.linalg.norm(la[..., :2], axis=-1), 1e-15),
                              cb.params[0] * residual / np.maximum(np.linalg.norm(lb[..., :2], axis=-1), 1e-15))
        epi, identity_epi = epi_for(ab), epi_for(grid)
    else:
        def matrix(camera):
            f, cx, cy, k = camera.params
            return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.]]), np.array([k, 0., 0., 0., 0.])
        ka, _ = matrix(ca); kb, db = matrix(cb)
        f = fundamental_from_w2c(a.rotation_world_to_camera, a.tvec, b.rotation_world_to_camera, b.tvec, ka, kb)
        support = cycle_support(ab, oa, ba, ob, (ca.width, ca.height),
                                filters["overlap_each_direction_minimum"], filters["cycle_maximum_native_pixels"])
        valid, confident, pre, cycle = (support[k] for k in ("valid", "confident", "cycle_kept", "cycle"))
        grid = normalized_grid(*oa.shape)
        pa = (grid + 1) * [ca.width / 2, ca.height / 2]
        def undistort(source_grid):
            p = (source_grid + 1) * [cb.width / 2, cb.height / 2]
            return cv2.undistortPoints(p.reshape(-1, 1, 2), kb, db, P=kb).reshape(p.shape)
        epi = symmetric_epipolar_pixels(pa, undistort(ab), f)
        identity_epi = symmetric_epipolar_pixels(pa, undistort(grid), f)
    reliable = pre & (epi <= filters["epipolar_maximum_native_pixels"])
    coverage = float(reliable.mean())
    retention = float(reliable.sum() / pre.sum()) if pre.any() else None
    q, qi = statistics(epi[pre]), statistics(identity_epi[pre])
    if independent:
        gates = cfg["geometry_screen"]
        checks = {"coverage": coverage >= gates["minimum_reliable_grid_fraction_each_pair"],
                  "epipolar_retention": retention is not None and retention >= gates["minimum_epipolar_inlier_fraction_after_cycle_each_pair"],
                  "beats_identity_factor2": q["median"] is not None and qi["median"] is not None and qi["median"] > 0 and q["median"] <= qi["median"] / 2}
    else:
        checks = screen_pair(coverage, retention, q["median"], qi["median"], cfg["geometry_screen"])
    return {k: pair[k] for k in ("fold", "target", "source", "rank")} | {
        "grid_points": int(oa.size), "native_size_wh": [ca.width, ca.height],
        "in_bounds": int(valid.sum()), "both_overlap_half": int(confident.sum()),
        "cycle_kept": int(pre.sum()), "reliable_points": int(reliable.sum()),
        "reliable_grid_fraction": coverage, "epipolar_inliers_after_cycle": retention,
        "cycle_native_pixels_on_confident": statistics(cycle[confident]),
        "epipolar_native_pixels_after_cycle": q, "identity_epipolar_on_same_support": qi,
        "coverage_4x4": [[float(tile.mean()) for tile in np.array_split(band, 4, axis=1)] for band in np.array_split(reliable, 4)],
        "checks": checks}


def compare_rows(expected, actual):
    quantile_keys = {"cycle_native_pixels_on_confident", "epipolar_native_pixels_after_cycle", "identity_epipolar_on_same_support"}
    if expected.keys() != actual.keys(): raise ValueError("Geometry row schema differs")
    for key in expected:
        if key in quantile_keys:
            if expected[key]["count"] != actual[key]["count"]: raise ValueError("Quantile counts differ")
            for q in ("median", "p90"):
                e, a = expected[key][q], actual[key][q]
                if (e is None) != (a is None) or (e is not None and not np.isclose(e, a, rtol=0, atol=2e-5)):
                    raise ValueError("Independent quantile differs: " + key)
        elif expected[key] != actual[key]:
            raise ValueError("Independent geometry differs: " + key)
