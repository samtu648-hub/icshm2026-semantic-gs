"""Descriptive edit cohorts and tie-aware ranks; never an inference selector."""
from __future__ import annotations

import numpy as np


def ranking(scores, useful):
    """Exact AUROC and stepwise AP for useful vs ALL non-useful edits.

    Ties receive half credit in AUROC and are grouped in AP. Pixel ranks are
    descriptive, not independent observations or confidence intervals.
    """
    scores = np.asarray(scores).reshape(-1)
    useful = np.asarray(useful, dtype=bool).reshape(-1)
    if scores.shape != useful.shape or np.isnan(scores).any():
        raise ValueError("Invalid ranking inputs")
    n = len(scores); positives = int(useful.sum()); negatives = n - positives
    result = {"pixels": n, "useful": positives,
              "prevalence": positives/n if n else None, "auroc": None, "ap": None}
    if not positives or not negatives:
        return result
    _, index = np.unique(scores, return_inverse=True)
    pos = np.bincount(index, weights=useful).astype(np.int64)
    neg = np.bincount(index, weights=~useful).astype(np.int64)
    result["auroc"] = float(np.sum(pos * (np.cumsum(neg)-neg+0.5*neg)) / (positives*negatives))
    tp = np.cumsum(pos[::-1]); total = np.cumsum((pos+neg)[::-1])
    result["ap"] = float(np.sum((pos[::-1]/positives)*(tp/total)))
    return result


def edit_vectors(gt, base, new, probability, other_probability, gt_distance, base_distance):
    shape = base.shape
    arrays = (gt, new, probability, other_probability, gt_distance, base_distance)
    if base.ndim != 2 or any(a.shape != shape for a in arrays):
        raise ValueError("Mismatched native grids")
    if any(not np.isin(a, (0, 1, 2, 3, 4)).all() for a in (gt, base, new)):
        raise ValueError("Invalid label IDs")
    if any(not np.isfinite(p).all() or p.min()<0 or p.max()>1 for p in (probability, other_probability)):
        raise ValueError("Invalid probability")
    if any(not np.array_equal(base==i, new==i) for i in (1, 2, 3)):
        raise ValueError("Protected class changed")
    expected = base.copy(); lane = (base==0) | (base==4)
    expected[lane] = np.where(probability[lane]>=0.5, 4, 0)
    if not np.array_equal(expected, new):
        raise ValueError("Sealed probability/fusion mismatch")
    result = {}
    for direction, old_id, new_id in (("add", 0, 4), ("remove", 4, 0)):
        mask = (base==old_id) & (new==new_id)
        confidence = probability[mask] if new_id==4 else 1-probability[mask]
        other = other_probability[mask] if new_id==4 else 1-other_probability[mask]
        agrees = other_probability[mask]>=0.5 if new_id==4 else other_probability[mask]<0.5
        result[direction] = {"gt": gt[mask], "useful": gt[mask]==new_id,
                             "harmful": gt[mask]==old_id, "confidence": confidence,
                             "other_confidence": other, "other_agrees": agrees,
                             "gt_distance": gt_distance[mask], "base_distance": base_distance[mask]}
    return result


def describe(vectors, confidence_edges, distances):
    useful = vectors["useful"]; harmful = vectors["harmful"]
    neutral = ~useful & ~harmful
    n = len(useful)
    counts = {"pixels": n, "corrected": int(useful.sum()), "harmed": int(harmful.sum()),
              "neutral": int(neutral.sum()), "gt_counts": np.bincount(vectors["gt"], minlength=5).tolist()}
    counts["net_corrected"] = counts["corrected"]-counts["harmed"]
    counts["corrected_fraction"] = counts["corrected"]/n if n else None
    def cohort(mask):
        return {"pixels": int(mask.sum()), "corrected": int((mask&useful).sum()),
                "harmed": int((mask&harmful).sum()), "neutral": int((mask&neutral).sum())}
    bins = []
    for lo, hi in zip(confidence_edges[:-1], confidence_edges[1:]):
        bins.append({"lower_inclusive": lo, "upper_exclusive": hi,
                     **cohort((vectors["confidence"]>=lo)&(vectors["confidence"]<hi))})
    if sum(r["pixels"] for r in bins) != n:
        raise ValueError("Confidence bins do not partition edits")
    return {**counts, "confidence_bins": bins,
            "agreement": {"other_agrees": cohort(vectors["other_agrees"]),
                          "other_disagrees": cohort(~vectors["other_agrees"])},
            "boundary": {kind: {str(d): cohort(vectors[kind]<=d) for d in distances}
                         for kind in ("gt_distance", "base_distance")},
            "ranking": {"confidence": ranking(vectors["confidence"], useful),
                        "other_confidence_on_same_edits": ranking(vectors["other_confidence"], useful),
                        "negative_baseline_distance": ranking(-vectors["base_distance"], useful)}}


def direction_screen(fold_counts):
    """Prospective research screen only, never a production quality gate."""
    if set(fold_counts) != {"0", "1", "2"}:
        return False
    return (sum(r["corrected"] for r in fold_counts.values()) >= 1000
            and all(r["corrected"] > r["harmed"] for r in fold_counts.values()))
