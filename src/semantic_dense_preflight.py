"""Small, model-independent contracts for the SEM320 correspondence preflight."""
from __future__ import annotations


def choose_pairs(folds, calibration, cameras):
    """Choose by released camera metadata only, never pixels or semantic labels."""
    calibration = set(calibration)
    pairs = []
    for fold in folds:
        train, oof, guard = (set(fold[k]) for k in ("train", "oof", "guard"))
        if train & oof or train & guard or oof & guard:
            raise ValueError("Spatial split overlap")
        targets = sorted(oof & calibration)
        if not targets or not train:
            raise ValueError("Empty preflight fold")
        target = targets[0]
        a = cameras[target]

        def rank(name):
            b = cameras[name]
            shared = len(set(a["tracks"]) & set(b["tracks"]))
            distance2 = sum((x - y) ** 2 for x, y in zip(a["center"], b["center"]))
            return -shared, distance2, name

        source = min(train, key=rank)
        negative_shared, distance2, _ = rank(source)
        pairs.append({"fold": fold["fold"], "target": target, "source": source,
                      "shared_tracks": -negative_shared, "center_distance": distance2 ** .5,
                      "source_in_same_fold_train": True})
    return pairs


def require_pair_scope(pairs, folds, calibration, developer, guard):
    by_fold = {fold["fold"]: fold for fold in folds}
    forbidden = set(developer) | set(guard)
    if len(pairs) != len(folds) or len({p["fold"] for p in pairs}) != len(folds):
        raise ValueError("Expected one pair per spatial fold")
    for pair in pairs:
        fold = by_fold[pair["fold"]]
        target, source = pair["target"], pair["source"]
        if (target not in calibration or target not in fold["oof"] or
                source not in fold["train"] or source in fold["oof"] or
                source in fold["guard"] or {target, source} & forbidden):
            raise ValueError("Forbidden pair input")


def pixel_center(normalized, size):
    """RoMa/COLMAP edge-origin pixel centers: first pixel=.5, not index0.

    For indexing a NumPy image, subtract .5 first. Keeping these two conventions
    distinct is required for later grid sampling and camera projection checks.
    """
    return (normalized + 1) * size / 2
