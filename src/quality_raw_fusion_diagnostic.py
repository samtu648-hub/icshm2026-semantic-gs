"""Exact raw/fusion complementarity accounting; no selection or prediction policy."""
import numpy as np

from src.quality_protocol import CLASSES
from src.quality_semantic_audit import metrics


def transition_cube(raw, fused, target):
    """Include unchanged pixels. Axes: raw class, fused class, target class."""
    arrays = (raw, fused, target)
    if any(a.dtype != np.uint8 or a.shape != raw.shape or a.ndim != 2 or
           a.size == 0 or a.max() > 4 for a in arrays):
        raise ValueError("Expected aligned nonempty uint8 official class images")
    ids = raw.astype(np.int64)*25 + fused.astype(np.int64)*5 + target
    cube = np.bincount(ids.ravel(), minlength=125).reshape(5, 5, 5)
    # Independent Boolean accounting for every changed direction and GT class.
    for old in range(5):
        for new in range(5):
            if old != new:
                mask = (raw == old) & (fused == new)
                direct = [np.count_nonzero(mask & (target == t)) for t in range(5)]
                if not np.array_equal(cube[old, new], direct):
                    raise ValueError("Independent transition accounting differs")
    return cube


def describe(cube):
    cube = np.asarray(cube, dtype=np.int64)
    raw_cm, fusion_cm = cube.sum(axis=1).T, cube.sum(axis=0).T
    raw, fused = metrics(raw_cm), metrics(fusion_cm)
    corrected = harmed = wrong_to_wrong = both_correct = same_wrong = 0
    directions = []
    for old in range(5):
        both_correct += int(cube[old, old, old])
        same_wrong += int(cube[old, old].sum()-cube[old, old, old])
        for new in range(5):
            if old == new:
                continue
            counts = cube[old, new]
            c, h = int(counts[new]), int(counts[old])
            w = int(counts.sum())-c-h
            corrected += c; harmed += h; wrong_to_wrong += w
            directions.append({"raw": CLASSES[old], "fusion": CLASSES[new],
                               "truth_counts": counts.tolist(), "corrected": c,
                               "harmed": h, "wrong_to_wrong": w})
    per_class = {}
    for i, name in enumerate(CLASSES):
        gained = int(cube[:, i, i].sum()-cube[i, i, i])
        lost = int(cube[i, :, i].sum()-cube[i, i, i])
        if fused["per_class"][name]["tp"]-raw["per_class"][name]["tp"] != gained-lost:
            raise ValueError("Class recall accounting does not reconcile")
        per_class[name] = {"gained_tp": gained, "lost_tp": lost, "net_tp": gained-lost,
                          "delta_pp": {k: None if raw["per_class"][name][k] is None or
                          fused["per_class"][name][k] is None else
                          100*(fused["per_class"][name][k]-raw["per_class"][name][k])
                          for k in ("IoU", "precision", "recall")}}
    return {"raw": raw, "calibrated_fusion": fused,
            "delta_miou_pp": None if raw["mIoU_with_background"] is None else
            100*(fused["mIoU_with_background"]-raw["mIoU_with_background"]),
            "pixels": int(cube.sum()), "both_correct": both_correct,
            "raw_only_correct": harmed, "fusion_only_correct": corrected,
            "both_wrong_same_class": same_wrong, "both_wrong_different_class": wrong_to_wrong,
            "changed": corrected+harmed+wrong_to_wrong, "corrected": corrected,
            "harmed": harmed, "wrong_to_wrong": wrong_to_wrong,
            "per_class": per_class, "directions": directions,
            "cube_raw_fusion_truth": cube.tolist()}


def provenance_regions(static, fused):
    """Last responsible frozen stage, not an assertion of independent causality."""
    masks = {"static_deck": static == 1, "static_tower": static == 3,
             "foundation_retained": (static == 4) & (fused == 4),
             "foundation_rejected": (static == 4) & (fused == 0)}
    lane = (static == 0) | (static == 2)
    masks.update(fiveclass_stay_cable=lane & (fused == 2),
                 deck_recovery=lane & (fused == 1), residual_background=lane & (fused == 0))
    if not np.all(sum(m.astype(np.uint8) for m in masks.values()) == 1):
        raise ValueError("Frozen fusion provenance is not an exhaustive partition")
    return masks
