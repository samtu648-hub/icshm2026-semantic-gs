"""Strict source-only semantic evidence on re-triangulated COLMAP tracks.

The contest release contains image-space ``POINT3D_ID`` correspondences but
not an official ``points3D.txt``.  The guarded RGB preparation re-triangulates
those released tracks without changing their observations.  This module uses
only those discrete observations: it does not use Scaffold depth, rendered
semantic probabilities, a target label, or a learned correspondence network.

It intentionally provides *diagnostic* coverage rather than a deployable
semantic prediction.  A track can only receive a source posterior from other
cameras in the current OOF fold.  The target view's label is read solely by
the caller after the evidence map is sealed, to measure whether this sparse
and independent source could possibly support a later dense method.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


TrackObservation = tuple[str, int, int]


def rounded_pixel(x: float, y: float, *, width: int, height: int) -> tuple[int, int] | None:
    """Return a valid nearest pixel for one COLMAP image observation."""
    px, py = int(np.rint(x)), int(np.rint(y))
    if 0 <= px < int(width) and 0 <= py < int(height):
        return px, py
    return None


def observation_index(
    images_by_name: Mapping[str, object],
    names: Iterable[str],
    *,
    width: int,
    height: int,
    permitted_track_ids: set[int] | None = None,
) -> dict[int, tuple[TrackObservation, ...]]:
    """Index released 2D track observations for the authorised view set."""
    output: dict[int, list[TrackObservation]] = defaultdict(list)
    for name in sorted(set(names)):
        image = images_by_name.get(name)
        if image is None:
            raise ValueError(f"prepared scene is missing authorised image {name}")
        for x, y, track_id in image.points2d:
            track_id = int(track_id)
            if track_id < 0 or (permitted_track_ids is not None and track_id not in permitted_track_ids):
                continue
            pixel = rounded_pixel(float(x), float(y), width=width, height=height)
            if pixel is not None:
                output[track_id].append((name, *pixel))
    return {track_id: tuple(entries) for track_id, entries in output.items()}


def binary_source_posteriors(
    observations: Mapping[int, Sequence[TrackObservation]],
    source_names: set[str],
    label_for_name: Callable[[str], np.ndarray],
    *,
    minimum_source_observations: int,
) -> tuple[dict[int, float], dict[int, int]]:
    """Estimate deck posterior from source labels only with a Beta(1,1) prior.

    ``label_for_name`` is never called for a name outside ``source_names``;
    callers can use that property to mechanically prove target-label
    isolation.  Class 1 is deck and every other official class is non-deck.
    """
    if int(minimum_source_observations) < 1:
        raise ValueError("minimum_source_observations must be positive")
    posterior: dict[int, float] = {}
    support: dict[int, int] = {}
    for track_id, track_observations in observations.items():
        votes: list[int] = []
        for name, x, y in track_observations:
            if name not in source_names:
                continue
            target = np.asarray(label_for_name(name))
            if target.ndim != 2 or not (0 <= y < target.shape[0] and 0 <= x < target.shape[1]):
                raise ValueError(f"invalid source label pixel for {name}")
            votes.append(int(target[y, x] == 1))
        if len(votes) >= int(minimum_source_observations):
            # The posterior is a continuous soft correspondence signal.  The
            # p>0.5 map used by the diagnostic is only the Bayes majority view,
            # never a tuned candidate threshold.
            posterior[int(track_id)] = float((sum(votes) + 1) / (len(votes) + 2))
            support[int(track_id)] = len(votes)
    return posterior, support


def target_majority_map(
    shape: tuple[int, int],
    observations: Mapping[int, Sequence[TrackObservation]],
    posterior: Mapping[int, float],
    *,
    target_name: str,
) -> np.ndarray:
    """Place source-only deck-majority evidence at target track observations."""
    height, width = map(int, shape)
    result = np.zeros((height, width), dtype=bool)
    for track_id, probability in posterior.items():
        if not 0.0 <= float(probability) <= 1.0:
            raise ValueError("track posterior must be a finite probability")
        if float(probability) <= 0.5:
            continue
        for name, x, y in observations.get(int(track_id), ()):
            if name == target_name:
                if not (0 <= x < width and 0 <= y < height):
                    raise ValueError("target observation lies outside target shape")
                result[y, x] = True
    return result


def deck_confusion(prediction: np.ndarray, target: np.ndarray) -> dict[str, int | float]:
    """Return binary deck confusion without accepting nonmatching maps."""
    pred, truth = np.asarray(prediction), np.asarray(target)
    if pred.shape != truth.shape or pred.ndim != 2:
        raise ValueError("prediction and target must be matching 2-D maps")
    predicted_deck, true_deck = pred == 1, truth == 1
    tp = int(np.logical_and(predicted_deck, true_deck).sum())
    fp = int(np.logical_and(predicted_deck, ~true_deck).sum())
    fn = int(np.logical_and(~predicted_deck, true_deck).sum())
    denominator = tp + fp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "iou": float(tp / denominator) if denominator else 1.0,
        "precision": float(tp / (tp + fp)) if tp + fp else 1.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 1.0,
    }


def oracle_deck_iou_gain(baseline: Mapping[str, int | float], covered_false_negatives: int) -> float:
    """Upper-bound deck-IoU gain if every covered FN were corrected perfectly."""
    covered = int(covered_false_negatives)
    fn = int(baseline["fn"])
    if not 0 <= covered <= fn:
        raise ValueError("covered false negatives must be within baseline false negatives")
    tp, fp = int(baseline["tp"]), int(baseline["fp"])
    old = float(baseline["iou"])
    new = (tp + covered) / (tp + fp + fn)
    return float(new - old)


def exact_oof_source_contract(target_name: str, source_names: set[str]) -> None:
    """Reject the only direct target-label leakage route in this diagnostic."""
    if target_name in source_names:
        raise ValueError("target camera must not supply semantic evidence to itself")
