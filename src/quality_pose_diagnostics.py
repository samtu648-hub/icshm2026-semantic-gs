"""Train-only, camera-unfiltered correspondence diagnostics; no pose updates."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from pathlib import Path

import numpy as np


def stable_number(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


def training_access(split):
    fold = split["primary_fold"]
    train = fold["train_rgb_400"]
    excluded = set(fold["validation_labeled"]) | set(fold["buffer_excluded_all_cameras"]) | set(fold["withheld_unlabeled"])
    if len(train) != len(set(train)) or set(train) & excluded:
        raise ValueError("Duplicate or forbidden training views")
    if any(Path(n).name != n or not n.endswith(".png") for n in train):
        raise ValueError("Invalid training filename")
    return sorted(train), sorted(excluded)


def load_training_records(path, allowed):
    """Hash/scan of a mixed metadata file is not decoding excluded point tracks."""
    from scripts.prepare_rgb_colmap_scene import pose_from_fields, parse_observations
    allowed = set(allowed)
    records, pending = {}, None
    with Path(path).open() as stream:
        for raw in stream:
            line = raw.strip()
            if line.startswith("#"):
                continue
            if pending is None:
                if not line:
                    continue
                record = pose_from_fields(line.split())
                if record.name in records:
                    raise ValueError("Duplicate camera name")
                records[record.name] = record
                pending = record
            else:
                if pending.name in allowed:
                    pending.observations = parse_observations(line)
                pending = None
    if not allowed <= records.keys():
        raise ValueError("Missing training cameras")
    return {n: records[n] for n in sorted(allowed)}


def summary(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    good = values[np.isfinite(values)]
    result = {"count": int(values.size), "finite_count": int(good.size), "nonfinite_count": int(values.size - good.size)}
    if good.size:
        result.update(mean=float(good.mean()), median=float(np.median(good)), p90=float(np.quantile(good, .9)),
                      p99=float(np.quantile(good, .99)), maximum=float(good.max()),
                      above_1px=int((good > 1).sum()), above_3px=int((good > 3).sum()), above_5px=int((good > 5).sum()))
    return result


def fundamental_from_poses(r1, t1, r2, t2, k):
    relative_r = r2 @ r1.T
    relative_t = t2 - relative_r @ t1
    if np.linalg.norm(relative_t) < 1e-12:
        raise ValueError("No translational epipolar constraint")
    x, y, z = relative_t
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    inv = np.linalg.inv(k)
    f = inv.T @ skew @ relative_r @ inv
    return f / np.linalg.norm(f)


def sampson_distance(f, a, b):
    """Square root of Sampson error, in native undistorted pixels."""
    a = np.c_[a, np.ones(len(a))]
    b = np.c_[b, np.ones(len(b))]
    line_b, line_a = a @ f.T, b @ f
    numerator = np.einsum("ij,ij->i", b, line_b)
    denom = np.sum(line_b[:, :2] ** 2 + line_a[:, :2] ** 2, axis=1)
    out = np.full(len(a), np.inf)
    np.divide(np.abs(numerator), np.sqrt(denom), out=out, where=denom > 1e-30)
    return out


def symmetric_transfer(h, a, b):
    def project(matrix, xy):
        xyz = np.c_[xy, np.ones(len(xy))] @ matrix.T
        result = np.full((len(xy), 2), np.inf)
        np.divide(xyz[:, :2], xyz[:, 2:], out=result, where=np.abs(xyz[:, 2:]) > 1e-12)
        return result
    try:
        inverse = np.linalg.inv(h)
    except np.linalg.LinAlgError:
        return np.full(len(a), np.inf)
    return np.sqrt((np.sum((project(h, a) - b) ** 2, axis=1) + np.sum((project(inverse, b) - a) ** 2, axis=1)) / 2)


def triangulate(poses, xy, k):
    inv = np.linalg.inv(k)
    normalized = np.c_[xy, np.ones(len(xy))] @ inv.T
    rows = []
    for (r, t), point in zip(poses, normalized):
        p = np.c_[r, t]
        rows.extend((point[0] * p[2] - p[0], point[1] * p[2] - p[1]))
    if len(rows) < 4:
        raise ValueError("Triangulation requires at least two observations")
    _, _, vt = np.linalg.svd(np.asarray(rows), full_matrices=False)
    if abs(vt[-1, 3]) < 1e-12:
        return None
    point = vt[-1, :3] / vt[-1, 3]
    return point if np.isfinite(point).all() else None


def reprojection(point, poses, xy, k):
    projected = np.array([r @ point + t for r, t in poses])
    if not np.isfinite(projected).all() or (projected[:, 2] <= 1e-8).any():
        return None
    pixels = projected @ k.T
    return pixels[:, :2] / pixels[:, 2:] - xy


def audit_released_tracks(records, k, seed):
    tracks = defaultdict(list)
    names = sorted(records)
    for i, n in enumerate(names):
        for obs in records[n].observations:
            if obs.point_id >= 0:
                tracks[obs.point_id].append((i, obs.undistorted_xy))
    all_rows, check_rows, rejected = [], [], []
    lengths = defaultdict(int)
    for point_id in sorted(tracks):
        track = tracks[point_id]
        lengths[len(track)] += 1
        if len(track) < 2:
            continue
        indices = [i for i, _ in track]
        if len(indices) != len(set(indices)):
            rejected.append((point_id, 0))
            continue
        xy = np.array([p for _, p in track])
        poses = [(records[names[i]].rotation, records[names[i]].translation) for i in indices]
        point = triangulate(poses, xy, k)
        delta = None if point is None else reprojection(point, poses, xy, k)
        if delta is None:
            rejected.append((point_id, 1))
        else:
            all_rows.extend((point_id, i, *p, *d) for i, p, d in zip(indices, xy, delta))
        # Do not condition held-out evaluation on the full-track residual or fit success.
        if len(track) >= 3:
            held = stable_number(seed, point_id) % len(track)
            fit = [j for j in range(len(track)) if j != held]
            candidate = triangulate([poses[j] for j in fit], xy[fit], k)
            residual = None if candidate is None else reprojection(candidate, poses, xy, k)
            if residual is None:
                rejected.append((point_id, 2))
            else:
                check_rows.append((point_id, indices[held], *xy[held], *residual[held]))
    arrays = {"all_fit": np.asarray(all_rows, np.float64).reshape(-1, 6),
              "cross_prediction": np.asarray(check_rows, np.float64).reshape(-1, 6),
              "rejected": np.asarray(rejected, np.int64).reshape(-1, 2)}
    report = {"track_length_counts": dict(sorted(lengths.items())), "rejection_codes": {"0": "duplicate image in track", "1": "full triangulation invalid/behind camera", "2": "cross triangulation invalid/behind camera"},
              "rejection_counts": {str(code): int((arrays["rejected"][:, 1] == code).sum()) for code in range(3)}, "per_view": {}}
    for role in ("all_fit", "cross_prediction"):
        data = arrays[role]
        report[role] = summary(np.linalg.norm(data[:, 4:], axis=1))
    for i, n in enumerate(names):
        views = {}
        for role in ("all_fit", "cross_prediction"):
            rows = arrays[role][arrays[role][:, 1] == i]
            views[role] = summary(np.linalg.norm(rows[:, 4:], axis=1))
            if len(rows):
                views[role]["mean_signed_residual_xy_px"] = rows[:, 4:].mean(axis=0).tolist()
        report["per_view"][n] = views
    return report, arrays


def select_pairs(records, ranks, maximum_angle, minimum_baseline):
    names = sorted(records)
    centres = {n: -records[n].rotation.T @ records[n].translation for n in names}
    axes = {n: records[n].rotation[2] for n in names}
    pairs, missing = set(), []
    for a in names:
        eligible = []
        for b in names:
            if a == b:
                continue
            distance = float(np.linalg.norm(centres[a] - centres[b]))
            angle = math.degrees(math.acos(float(np.clip(axes[a] @ axes[b], -1, 1))))
            if distance > minimum_baseline and angle <= maximum_angle:
                eligible.append((distance, b))
        eligible.sort()
        for rank in ranks:
            if rank <= 0:
                raise ValueError("Neighbor ranks are one-based")
            if len(eligible) < rank:
                missing.append({"view": a, "rank": rank})
            else:
                pairs.add(tuple(sorted((a, eligible[rank - 1][1]))))
    return sorted(pairs), missing


def crossfit_matches(a, b, pair, seed):
    """Group near-identical location pairs (multiple SIFT orientations) before split."""
    seen, indices, held = set(), [], []
    for i, (pa, pb) in enumerate(zip(a, b)):
        key = tuple(np.rint(np.r_[pa, pb] * 4).astype(np.int64))
        if key in seen:
            continue
        seen.add(key)
        indices.append(i)
        held.append(stable_number(seed, *pair, *key) % 5 == 0)
    return np.asarray(indices, dtype=int), np.asarray(held, dtype=bool)


def occupied_cells(xy, width, height, grid):
    xy = np.asarray(xy)
    good = (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    cells = np.floor(xy[good] / [width, height] * grid).astype(int)
    return len(set(map(tuple, cells)))


def mutual_matches(desc_a, desc_b, config, seed):
    import cv2
    if desc_a is None or desc_b is None or min(len(desc_a), len(desc_b)) < 2:
        return np.empty((0, 2), dtype=int)
    cv2.setRNGSeed(int(seed % (2**31 - 1)))
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=config["flann_trees"]), dict(checks=config["flann_checks"]))
    def passing(first, second):
        matches = matcher.knnMatch(first, second, k=2)
        return {m[0].queryIdx: m[0].trainIdx for m in matches if len(m) == 2 and m[0].distance < config["ratio"] * m[1].distance}
    ab, ba = passing(desc_a, desc_b), passing(desc_b, desc_a)
    return np.array([(a, b) for a, b in sorted(ab.items()) if ba.get(b) == a], dtype=int).reshape(-1, 2)


def evaluate_pair(a, b, official_f, pair, config, seed, size):
    import cv2
    indices, held = crossfit_matches(a, b, pair, seed)
    a, b = a[indices], b[indices]
    report = {"pair": list(pair), "descriptor_matches": len(indices), "fit_matches": int((~held).sum()), "check_matches": int(held.sum()),
              "status": "insufficient_matches", "pose_inconsistency_candidate": False}
    arrays = {"a": a, "b": b, "check": held, "official_f": official_f}
    if (~held).sum() < config["minimum_fit_matches"] or held.sum() < config["minimum_check_matches"]:
        return report, arrays
    cv2.setRNGSeed(int(stable_number(seed, *pair) % (2**31 - 1)))
    f, mask = cv2.findFundamentalMat(a[~held], b[~held], cv2.USAC_MAGSAC, config["ransac_threshold_px"], config["confidence"], config["max_iterations"])
    if f is None or f.shape != (3, 3) or not np.isfinite(f).all() or np.linalg.norm(f) < 1e-15:
        report["status"] = "image_f_estimation_failed"
        return report, arrays
    h, _ = cv2.findHomography(a[~held], b[~held], cv2.USAC_MAGSAC, config["ransac_threshold_px"],
                             maxIters=config["max_iterations"], confidence=config["confidence"])
    official_error = sampson_distance(official_f, a[held], b[held])
    image_error = sampson_distance(f, a[held], b[held])
    h_error = np.full(held.sum(), np.inf) if h is None else symmetric_transfer(h, a[held], b[held])
    threshold = config["support_threshold_px"]
    f_support, h_support = float((image_error <= threshold).mean()), float((h_error <= threshold).mean())
    a_cells = occupied_cells(a[held], *size, config["grid"])
    b_cells = occupied_cells(b[held], *size, config["grid"])
    planar = h_support >= .8 * f_support
    report.update(status="completed_correspondence_crossfit_not_pose_truth", official_sampson_px=summary(official_error),
                  image_f_sampson_px=summary(image_error), image_h_symmetric_transfer_px=summary(h_error),
                  official_support=float((official_error <= threshold).mean()), image_f_support=f_support, image_h_support=h_support,
                  possible_planar_or_low_parallax=planar, occupied_check_cells=[a_cells, b_cells], fit_f_inliers=int(mask.sum()))
    supported = f_support >= .5 and min(a_cells, b_cells) >= config["minimum_check_occupied_cells"]
    report["pose_inconsistency_candidate"] = bool(supported and not planar and np.median(official_error) > 1.5 and np.median(official_error) >= 3 * np.median(image_error))
    arrays.update(image_f=f, official_check_error=official_error, image_f_check_error=image_error, image_h_check_error=h_error)
    if h is not None:
        arrays["image_h"] = h
    return report, arrays
