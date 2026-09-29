"""Matched old-role fusion; no geometry, labels or threshold selection in inference."""
import numpy as np


def validate_inputs(static, five, cable, deck):
    if static.dtype != np.uint8 or static.ndim != 2 or not static.size or static.max() > 4:
        raise ValueError("Expected official static class IDs")
    if five.shape != (5, *static.shape) or cable.shape != static.shape or deck.shape != static.shape:
        raise ValueError("Fusion inputs are not on the same PINHOLE grid")
    for p in (five, cable, deck):
        if not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
            raise ValueError("Invalid probability evidence")
    if not np.allclose(five.sum(0), 1, atol=2e-5, rtol=0):
        raise ValueError("Five-class probabilities do not sum to one")


def apply_stage(name, base, five, cable, deck, threshold, pre_foundation=None):
    validate_inputs(base, five, cable, deck)
    out = base.copy()
    if name not in ("fiveclass_lane", "cable", "foundation", "deck"):
        raise ValueError("Unknown registered fusion stage")
    if threshold is None:
        return out
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Invalid fusion threshold")
    if name in ("fiveclass_lane", "cable"):
        lane = (base == 0) | (base == 2)
        p = five[2] if name == "fiveclass_lane" else cable
        out[lane] = np.where(p[lane] >= threshold, 2, 0)
    elif name == "foundation":
        out[(base == 4) & (five[4] < threshold)] = 0
    else:
        if pre_foundation is None or pre_foundation.shape != base.shape or pre_foundation.dtype != np.uint8 or pre_foundation.max() > 4:
            raise ValueError("Deck recovery requires aligned pre-foundation class provenance")
        out[(base == 0) & (pre_foundation == 0) & (deck >= threshold)] = 1
    return out


def pipeline(static, five, cable, deck, thresholds):
    validate_inputs(static, five, cable, deck)
    result = {"static": static.copy()}
    current = static
    for name in ("fiveclass_lane", "cable", "foundation", "deck"):
        current = apply_stage(name, current, five, cable, deck, thresholds.get(name), result.get("cable"))
        result[name] = current
    return result


def confusion(target, prediction):
    if target.shape != prediction.shape or target.dtype != np.uint8 or prediction.dtype != np.uint8 or target.max() > 4 or prediction.max() > 4:
        raise ValueError("Invalid five-class scoring arrays")
    return np.bincount(target.ravel().astype(np.int64)*5+prediction.ravel(), minlength=25).reshape(5, 5)


def choose_threshold(records, target_class, *, deck_primary=False):
    """No weighted score: registered lexicographic metrics, off preferred on ties."""
    from src.quality_protocol import CLASSES
    def rank(row):
        m = row["metrics"]
        iou = m["per_class"][CLASSES[target_class]]["IoU"]
        mean = m["mIoU_with_background"]
        if iou is None or mean is None:
            raise ValueError("Calibration metric unavailable")
        primary = (iou, mean) if deck_primary else (mean, iou)
        return (*primary, row["threshold"] is None, row["threshold"] or 0.)
    if not records or not any(r["threshold"] is None for r in records):
        raise ValueError("Calibration must include an explicit off candidate")
    return max(records, key=rank)
