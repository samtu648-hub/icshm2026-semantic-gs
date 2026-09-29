"""SEM343: one frozen proposal direction, no averaging or learned weights."""
import numpy as np


def apply_gate(raw, fused, five, maximum_margin):
    """Raw is default; only tower -> foundation proposals may be adopted."""
    if raw.shape != fused.shape or five.shape != (5, *raw.shape):
        raise ValueError("Inputs must share the native grid")
    if maximum_margin is not None and (not np.isfinite(maximum_margin) or not 0 <= maximum_margin <= 1):
        raise ValueError("Invalid registered margin")
    result = raw.copy()
    if maximum_margin is not None:
        evidence = (raw == 3) & (fused == 4) & ((five[3]-five[4]) <= maximum_margin)
        result[evidence] = 4
    return result
