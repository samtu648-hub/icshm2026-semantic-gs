"""Fixed auxiliary review, not unrestricted foundation replacement."""
import numpy as np


def review_foundation(base, probability, *, enabled=True):
    if base.ndim != 2 or base.dtype != np.uint8 or not base.size or base.max()>4:
        raise ValueError("Invalid native baseline labels")
    p = np.asarray(probability)
    if p.shape != base.shape or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError("Invalid native foundation probability")
    if not isinstance(enabled, bool):
        raise ValueError("Reviewer is either enabled or bypassed")
    out = base.copy()
    if enabled:
        out[(base==4)&(p<0.5)] = 0
    return out
