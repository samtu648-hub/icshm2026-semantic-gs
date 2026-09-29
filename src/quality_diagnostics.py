"""Pixel-exact stage provenance, with no automatic claim of geometric ground truth."""
from collections.abc import Mapping
import numpy as np

from src.quality_backbone import validate_probabilities
from src.semantic2d import confusion_metrics, update_confusion


def labels(array, shape=None):
    value = np.asarray(array)
    if value.ndim != 2 or value.dtype.kind not in "ui" or value.size == 0 or np.any(value < 0) or np.any(value > 4):
        raise ValueError("Expected nonempty single-channel integer labels 0..4")
    if shape is not None and value.shape != shape:
        raise ValueError("Labels must share one native pixel grid")
    return value


def stage_changes(before, after, target=None):
    before = labels(before)
    after = labels(after, before.shape)
    changed = before != after
    transitions = np.bincount((before.astype(np.int64) * 5 + after).ravel(), minlength=25).reshape(5, 5)
    result = {"changed_pixels": int(changed.sum()), "transitions_old_rows_new_columns": transitions.tolist(),
              "background_to_deck": int(((before == 0) & (after == 1)).sum()),
              "deck_deleted": int(((before == 1) & (after != 1)).sum()),
              "correct_repairs": None, "harmful_changes": None, "wrong_to_different_wrong": None}
    if target is not None:
        target = labels(target, before.shape)
        result.update(correct_repairs=int((changed & (before != target) & (after == target)).sum()),
                      harmful_changes=int((changed & (before == target) & (after != target)).sum()),
                      wrong_to_different_wrong=int((changed & (before != target) & (after != target)).sum()))
    return result, changed


def deck_gap_provenance(stages: Mapping, target):
    """Exact categorical provenance, never an inferred geometric cause.

    Only sequential deployed stages belong here. Parallel raw expert argmax
    must NOT be inserted into the sequence as if it were a deployed stage.
    """
    if not stages:
        raise ValueError("No sequential stages")
    target = labels(target)
    sequence = [(name, labels(value, target.shape)) for name, value in stages.items()]
    final_gap = (target == 1) & (sequence[-1][1] == 0)
    ever_deck = np.zeros(target.shape, bool)
    last_loss = np.full(target.shape, -1, np.int16)
    for index, (_, value) in enumerate(sequence):
        ever_deck |= value == 1
        if index:
            last_loss[(sequence[index - 1][1] == 1) & (value != 1)] = index
    earlier_correct = final_gap & ever_deck
    upstream_unresolved = final_gap & ~ever_deck
    counts = {sequence[i][0]: int((earlier_correct & (last_loss == i)).sum()) for i in range(1, len(sequence))}
    if sum(counts.values()) != int(earlier_correct.sum()):
        raise AssertionError("Lost-deck stage counts do not partition final gaps")
    return {"final_true_deck_as_background": int(final_gap.sum()),
            "earlier_deck_lost_later": int(earlier_correct.sum()),
            "never_deck_in_deployed_sequence": int(upstream_unresolved.sum()),
            "last_deck_loss_stage": counts,
            "geometry_cause": "unresolved; image error alone is not geometric ground truth"}, {
                "final_deck_gap": final_gap, "earlier_deck_lost_later": earlier_correct,
                "upstream_unresolved": upstream_unresolved}


def diagnose_stages(stages: Mapping, *, target=None, roi=None, probabilities=None,
                    has_same_camera_photo=False, interpolation_view=False):
    if not stages:
        raise ValueError("At least one named semantic stage is required")
    if interpolation_view and (target is not None or has_same_camera_photo):
        raise ValueError("Interpolated view cannot be asserted to have paired photographic/label truth")
    first = labels(next(iter(stages.values())))
    shape = first.shape
    if target is not None:
        target = labels(target, shape)
    roi = np.ones(shape, bool) if roi is None else np.asarray(roi)
    if roi.shape != shape or roi.dtype != bool or not roi.any():
        raise ValueError("ROI must be a nonempty boolean mask on the common grid")
    result = {"scope": "interpolation_continuity_only" if interpolation_view else "paired_view_diagnostics",
              "has_same_camera_photo": bool(has_same_camera_photo), "has_labels": target is not None,
              "geometry_attribution": "unresolved; paired RGB and manual or independent structural evidence required",
              "stages": {}, "changes": {}}
    masks = {}
    previous_name, previous = None, None
    for name, prediction in stages.items():
        prediction = labels(prediction, shape)
        if not isinstance(name, str) or not name:
            raise ValueError("Every stage must have an explicit name")
        metrics = None
        if target is not None:
            confusion = np.zeros((5, 5), np.int64)
            update_confusion(confusion, target[roi], prediction[roi])
            metrics = confusion_metrics(confusion)
        result["stages"][name] = {"metrics": metrics, "roi_pixels": int(roi.sum())}
        if previous is not None:
            change, mask = stage_changes(previous[roi][None], prediction[roi][None], None if target is None else target[roi][None])
            key = previous_name + " -> " + name
            result["changes"][key] = change
            masks[key] = (previous != prediction) & roi
        previous_name, previous = name, prediction
    for name, probability in (probabilities or {}).items():
        validate_probabilities(probability, *shape)
        result.setdefault("probability_diagnostics", {})[name] = {
            "mean_roi_max_probability": float(np.asarray(probability)[:, roi].max(axis=0).mean())}
    return result, masks
