"""SEM364 diagnostic counts; oracle information must never drive predictions."""
import numpy as np

BUCKETS = ['recovered', 'recoverable_kept_raw', 'recoverable_wrong_alternative',
           'unrecoverable_kept_raw', 'unrecoverable_changed_wrong', 'correct_kept', 'correct_harmed']


def source_counts(gt, raw, maps, reference=False):
    gt, raw = gt.ravel(), raw.ravel()
    bits = np.zeros(gt.shape, np.int64)
    for j, m in enumerate(maps): bits |= (m.ravel() == gt).astype(np.int64) << j
    errors = raw != gt
    recover = errors & (bits != 0)
    oracle = np.where(recover, gt, raw)
    if reference:
        hist = np.zeros((5,32),np.int64);conf=np.zeros((5,5),np.int64)
        for t in range(5):
            vals, counts = np.unique(bits[errors & (gt==t)],return_counts=True)
            hist[t,vals]=counts
            for p in range(5):conf[t,p]=np.count_nonzero((gt==t)&(oracle==p))
    else:
        hist=np.bincount(gt[errors].astype(np.int64)*32+bits[errors],minlength=160).reshape(5,32)
        conf=np.bincount(gt.astype(np.int64)*5+oracle,minlength=25).reshape(5,5)
    return hist, conf, recover


def decision_counts(gt, raw, pred, recover, reference=False):
    gt,raw,pred=gt.ravel(),raw.ravel(),pred.ravel()
    if np.any((raw!=gt)&(pred==gt)&~recover):raise ValueError('Saved prediction recovered a class outside candidate sources')
    if reference:
        rows=[recover&(pred==gt),recover&(pred==raw),recover&(pred!=raw)&(pred!=gt),
              (raw!=gt)&~recover&(pred==raw),(raw!=gt)&~recover&(pred!=raw),
              (raw==gt)&(pred==gt),(raw==gt)&(pred!=gt)]
        out=np.array([[np.count_nonzero((gt==t)&keep) for keep in rows] for t in range(5)],np.int64)
    else:
        category=np.where(raw==gt,np.where(pred==gt,5,6),
                          np.where(recover,np.where(pred==gt,0,np.where(pred==raw,1,2)),np.where(pred==raw,3,4)))
        out=np.bincount(gt.astype(np.int64)*7+category,minlength=35).reshape(5,7)
    if out.sum()!=gt.size:raise ValueError('Decision partition does not conserve pixels')
    return out
