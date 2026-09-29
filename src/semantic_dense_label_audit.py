"""Count class evidence on sealed geometric support; never produce predictions."""
from __future__ import annotations
import numpy as np

CLASS_NAMES = ("background", "deck", "stay_cable", "tower", "foundation")


def sample_labels(labels, points, independent=False):
    """Nearest edge-origin pixel center. Out-of-frame samples have no support."""
    h, w = labels.shape
    valid = np.isfinite(points).all(-1) & (points[..., 0] >= 0) & (points[..., 0] < w) & (points[..., 1] >= 0) & (points[..., 1] < h)
    safe = np.nan_to_num(points, nan=0, posinf=0, neginf=0)
    if independent:
        import torch
        import torch.nn.functional as functional
        norm = safe / np.array([w/2, h/2]) - 1
        values = functional.grid_sample(torch.tensor(labels, dtype=torch.float64)[None,None],
            torch.tensor(norm, dtype=torch.float64)[None], mode="nearest",padding_mode="border",align_corners=False)[0,0].numpy().astype(np.uint8)
    else:
        x = np.clip(np.floor(safe[...,0]).astype(int),0,w-1)
        y = np.clip(np.floor(safe[...,1]).astype(int),0,h-1)
        values = labels[y,x]
    return values, valid


def count_evidence(target, baseline, source, support, independent=False):
    if not (target.shape == baseline.shape == source.shape == support.shape): raise ValueError("Scope mismatch")
    if any(np.any((a<0)|(a>4)) for a in (target,baseline,source)): raise ValueError("Unexpected class ID")
    gt, old, new = target[support].astype(int),baseline[support].astype(int),source[support].astype(int)
    if independent:
        matrix = np.histogram2d(gt,new,bins=(np.arange(6)-.5,np.arange(6)-.5))[0].astype(np.int64)
        cube = np.zeros((5,5,5),np.int64)
        for a in range(5):
            for b in range(5):
                for c in range(5): cube[a,b,c] = np.count_nonzero((gt==a)&(old==b)&(new==c))
        totals = np.array([np.count_nonzero(target==c) for c in range(5)],np.int64)
    else:
        matrix = np.bincount(gt*5+new,minlength=25).reshape(5,5)
        cube = np.bincount(gt*25+old*5+new,minlength=125).reshape(5,5,5)
        totals = np.bincount(target.ravel().astype(int),minlength=5)
    return {"target_class_grid_counts":totals.tolist(),"label_confusion":matrix.tolist(),"transition_counts":cube.tolist(),
        "total_grid_points":int(target.size),"usable_semantic_points":int(support.sum()),
        "baseline_errors_total":int((target!=baseline).sum()),"baseline_errors_on_support":int((gt!=old).sum()),
        "corrected":int(((gt!=old)&(gt==new)).sum()),"harmed":int(((gt==old)&(gt!=new)).sum()),
        "both_wrong":int(((gt!=old)&(gt!=new)).sum()),"both_correct":int(((gt==old)&(gt==new)).sum())}


def aggregate(rows):
    scalar = ("total_grid_points","usable_semantic_points","baseline_errors_total","baseline_errors_on_support","corrected","harmed","both_wrong","both_correct")
    result = {k:sum(r[k] for r in rows) for k in scalar}
    result["target_count"] = len(rows)
    result["net_corrected"] = result["corrected"]-result["harmed"]
    cm = sum((np.array(r["label_confusion"],np.int64) for r in rows), np.zeros((5,5),np.int64))
    cube = sum((np.array(r["transition_counts"],np.int64) for r in rows), np.zeros((5,5,5),np.int64))
    totals = sum((np.array(r["target_class_grid_counts"],np.int64) for r in rows),np.zeros(5,np.int64))
    ratio = lambda a,b: float(a/b) if b else None
    result["source_label_agreement"] = ratio(np.trace(cm),cm.sum())
    result["baseline_error_coverage"] = ratio(result["baseline_errors_on_support"],result["baseline_errors_total"])
    result["correctable_error_fraction"] = ratio(result["corrected"],result["baseline_errors_total"])
    result["class_metrics"] = {}
    for c,name in enumerate(CLASS_NAMES):
        result["class_metrics"][name] = {"target_grid_points":int(totals[c]),"supported_target_points":int(cm[c].sum()),
            "target_class_coverage":ratio(cm[c].sum(),totals[c]),"source_class_points":int(cm[:,c].sum()),
            "source_class_precision":ratio(cm[c,c],cm[:,c].sum()),"agreement_on_supported_target_class":ratio(cm[c,c],cm[c].sum()),
            "corrected_true_class":int(cube[c,:,c].sum()-cube[c,c,c]),
            "harmed_true_class":int(cube[c,c,:].sum()-cube[c,c,c])}
    result["label_confusion"] = cm.tolist()
    return result
