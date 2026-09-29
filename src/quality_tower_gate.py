"""Single original-raw stay_cable -> tower proposal; no cascading foundation gate."""
import numpy as np


def apply_tower_gate(raw, base, static, five, margin):
    if raw.shape!=base.shape or raw.shape!=static.shape or five.shape!=(5,*raw.shape):
        raise ValueError('All inputs must share native five-class grid')
    if np.any((base!=raw)&~((raw==3)&(base==4))):
        raise ValueError('Baseline must be raw or sealed original tower->foundation only')
    if margin is not None and (not np.isfinite(margin) or not 0<=margin<=1):
        raise ValueError('Invalid probability margin')
    out=base.copy()
    if margin is not None:
        out[(raw==2)&(static==3)&((five[2]-five[3])<=margin)]=3
    return out
