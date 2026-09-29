"""SEM332 one-way additive evidence; calibration/evaluation remain separate."""
import numpy as np


def add_deck_evidence(current, pre_foundation, post_foundation, five_deck, binary_deck, threshold):
    arrays = [np.asarray(a) for a in (current, pre_foundation, post_foundation, five_deck, binary_deck)]
    if any(a.shape != arrays[0].shape for a in arrays) or arrays[0].ndim != 2:
        raise ValueError("All evidence must share one native pixel grid")
    for a in arrays[:3]:
        if a.dtype != np.uint8 or not a.size or a.max() > 4:
            raise ValueError("Expected official class IDs 0..4")
    for a in arrays[3:]:
        if not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
            raise ValueError("Invalid expert probability")
    if threshold is not None and (not np.isfinite(threshold) or threshold < .5 or threshold > 1):
        raise ValueError("Unregistered confidence threshold")
    current, pre, post, five, binary = arrays
    eligible = (current == 0) & (pre == 0) & (post == 0)
    changed = eligible & (np.minimum(five, binary) >= threshold) if threshold is not None else np.zeros_like(eligible)
    out = current.copy()
    out[changed] = 1
    return out, {"changed": changed, "eligible": eligible, "history_excluded": (current == 0) & ~eligible}


def change_counts(before, after, target):
    if before.shape != after.shape or before.shape != target.shape:
        raise ValueError("Different modification evidence grids")
    changed = before != after
    return {"changed": int(changed.sum()),
        "corrected": int((changed & (before != target) & (after == target)).sum()),
        "harmed": int((changed & (before == target) & (after != target)).sum()),
        "wrong_to_wrong": int((changed & (before != target) & (after != target)).sum())}
