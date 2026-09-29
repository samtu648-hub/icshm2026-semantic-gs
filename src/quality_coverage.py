"""Opacity diagnostics: descriptive evidence, never a probability correction."""
import numpy as np


def coverage_counts(alpha, probabilities, target, bins=(0., .1, .25, .5, .75, .9, 1.), tolerance=2e-5):
    alpha, probabilities, target = np.asarray(alpha), np.asarray(probabilities), np.asarray(target)
    if (alpha.shape != target.shape or probabilities.shape != (5, *target.shape)
            or not np.isfinite(alpha).all() or not np.isfinite(probabilities).all()
            or alpha.min() < -tolerance or alpha.max() > 1 + tolerance
            or probabilities.min() < 0 or probabilities.max() > 1
            or not np.allclose(probabilities.sum(0), 1, atol=tolerance, rtol=0)
            or target.dtype != np.uint8 or target.min() < 0 or target.max() > 4):
        raise ValueError("Invalid native coverage, probability or target grid")
    if tuple(bins) != tuple(sorted(set(bins))) or bins[0] != 0 or bins[-1] != 1:
        raise ValueError("Coverage bins must partition [0,1]")
    # For raw compositing: foreground <= alpha, background >= 1-alpha.
    if ((probabilities[1:].max(0) > alpha + tolerance).any()
            or (probabilities[0] + tolerance < 1 - alpha).any()):
        raise ValueError("Static probabilities do not obey residual-background model")
    pred = probabilities.argmax(0)
    low = alpha < .5
    result = {"pixels": int(alpha.size), "low_coverage_pixels": int(low.sum()), "classes": {}}
    for c in range(5):
        gt = target == c
        missed = gt & (pred != c)
        result["classes"][str(c)] = {"gt": int(gt.sum()), "errors": int(missed.sum()),
            "low_coverage_gt": int((gt & low).sum()), "low_coverage_errors": int((missed & low).sum()),
            "high_coverage_errors": int((missed & ~low).sum()),
            "to_background": int((gt & (pred == 0)).sum()),
            "low_coverage_to_background": int((gt & low & (pred == 0)).sum())}
    a = np.clip(alpha, 0, 1)
    result["bins"] = []
    for i, (left, right) in enumerate(zip(bins[:-1], bins[1:])):
        selected = (a >= left) & ((a <= right) if i == len(bins)-2 else (a < right))
        result["bins"].append({"bounds": [left, right], "pixels": int(selected.sum()),
            "gt_counts": np.bincount(target[selected], minlength=5).tolist(),
            "error_counts": np.bincount(target[selected & (target != pred)], minlength=5).tolist()})
    return result


def retained_conditioning(ids, conditioning, expected_points):
    ids = np.asarray(ids, np.int64)
    pids = np.asarray(conditioning["point_id"], np.int64)
    if len(ids) != expected_points or len(set(ids.tolist())) != len(ids) or len(set(pids.tolist())) != len(pids):
        raise ValueError("Initialization or cross-prediction point IDs are not unique/complete")
    errors, angles = conditioning["error_px"], conditioning["source_max_ray_angle_deg"]
    if len(errors) != len(pids) or len(angles) != len(pids) or not np.isfinite(errors).all() or not np.isfinite(angles).all():
        raise ValueError("Invalid conditioning arrays")
    retained = np.isin(pids, ids)
    large, weak = errors > 5, angles < 1
    return {"initial_points": len(ids), "cross_predicted_tracks": len(pids),
        "cross_predicted_retained": int(retained.sum()), "initial_without_successful_cross_prediction": int(len(ids)-retained.sum()),
        "large_error_all": int(large.sum()), "large_error_retained": int((retained & large).sum()),
        "weak_all": int(weak.sum()), "weak_retained": int((retained & weak).sum()),
        "weak_large_retained": int((retained & weak & large).sum()),
        "well_conditioned_large_retained": int((retained & ~weak & large).sum())}, retained
