"""SEM365 proposed-class coefficient blocks; preserve global head weights."""
import numpy as np
from src.quality_learned_selector import candidates, feature_rows, choose


def solve_blocks(gram_sum, rhs_sum, total_weight, row_counts, penalty):
    if total_weight <= 0: raise ValueError('Positive global head weight required')
    gram, rhs = gram_sum/total_weight, rhs_sum/total_weight
    active = np.asarray(row_counts) > 0
    coef = np.zeros_like(rhs)
    reg = np.eye(rhs.shape[1])*penalty; reg[-1,-1]=0
    for cid in np.where(active)[0]:
        coef[cid] = np.linalg.solve(gram[cid]+reg, rhs[cid])
    if not np.isfinite(coef).all():raise FloatingPointError('Nonfinite conditional coefficients')
    return dict(coef=coef, gram=gram, rhs=rhs, active=active)


def predict(data, model, chunk=32768):
    raw=data[0]
    if bool(model['fallback']):return raw.copy()
    allowed=candidates(raw,data[1]).reshape(5,-1)
    scores=np.full(allowed.shape,-np.inf)
    for cid in range(5):
        pixels=np.flatnonzero(allowed[cid])
        for start in range(0,len(pixels),chunk):
            ix=pixels[start:start+chunk]
            if not model['active'][cid]:scores[cid,ix]=0;continue
            x=feature_rows(data,ix,np.full(len(ix),cid))
            scores[cid,ix]=((x-model['mean'])/model['scale'])@model['coef'][cid]
    return choose(raw.ravel(),allowed,scores).reshape(raw.shape)
