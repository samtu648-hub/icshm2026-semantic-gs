"""Pure helpers for SEM-398's matched semantic-boundary split experiment.

The candidate and control share one eligibility mask and one split budget.  The
only experimental variable is the score used to rank eligible parent Gaussians.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def robust_signed_evidence(gradient: np.ndarray) -> np.ndarray:
    """Bound a view's signed evidence using its median non-zero magnitude."""
    values = np.asarray(gradient, dtype=np.float64)
    magnitude = np.abs(values)
    active = magnitude[magnitude > 0]
    scale = float(np.median(active)) if active.size else 1.0
    scale = max(scale, np.finfo(np.float64).tiny)
    return values / (magnitude + scale)


def add_signed_support(
    positive: np.ndarray, negative: np.ndarray, normalized: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate opposing-view support without cancelling it prematurely."""
    evidence = np.asarray(normalized, dtype=np.float64)
    return positive + np.maximum(evidence, 0.0), negative + np.maximum(-evidence, 0.0)


def conflict_score(positive: np.ndarray, negative: np.ndarray, prior: float = 4.0) -> np.ndarray:
    """Return shrunk multi-view conflict in [0, 1].

    A Gaussian scores highly only when different views provide material support
    for both signs.  One-sided evidence and unseen points score zero.
    """
    if prior < 0:
        raise ValueError("prior must be non-negative")
    pos = np.asarray(positive, dtype=np.float64)
    neg = np.asarray(negative, dtype=np.float64)
    if pos.shape != neg.shape:
        raise ValueError("positive and negative support must have the same shape")
    return 2.0 * np.minimum(pos, neg) / (prior + pos + neg)


def stable_topk(scores: np.ndarray, eligible: np.ndarray, budget: int) -> np.ndarray:
    """Rank by descending score and ascending stable Gaussian id."""
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    mask = np.asarray(eligible, dtype=bool).reshape(-1)
    if score.shape != mask.shape:
        raise ValueError("scores and eligibility mask must have equal length")
    if budget < 0:
        raise ValueError("budget must be non-negative")
    ids = np.flatnonzero(mask)
    if budget > ids.size:
        raise ValueError(f"split budget {budget} exceeds {ids.size} eligible parents")
    # lexsort's last key is primary: descending score, then ascending id.
    order = np.lexsort((ids, -score[ids]))
    return ids[order[:budget]]


def inherited_child_values(values: np.ndarray, selected: np.ndarray, children: int = 2) -> np.ndarray:
    """Mirror Standard 3DGS native split/prune ordering for sidecar values."""
    source = np.asarray(values)
    chosen = np.asarray(selected, dtype=np.int64).reshape(-1)
    if children < 1:
        raise ValueError("children must be positive")
    if chosen.size and (chosen.min() < 0 or chosen.max() >= len(source)):
        raise IndexError("selected parent index out of range")
    if np.unique(chosen).size != chosen.size:
        raise ValueError("selected parent ids must be unique")
    keep = np.ones(len(source), dtype=bool)
    keep[chosen] = False
    repeated = np.concatenate([source[chosen]] * children, axis=0)
    return np.concatenate([source[keep], repeated], axis=0)


@dataclass(frozen=True)
class PairSelection:
    control: np.ndarray
    candidate: np.ndarray
    overlap: int


def select_matched_pair(
    rgb_score: np.ndarray,
    semantic_score: np.ndarray,
    eligible: np.ndarray,
    budget: int,
) -> PairSelection:
    """Select equal-budget control and candidate parent ids."""
    control = stable_topk(rgb_score, eligible, budget)
    candidate = stable_topk(semantic_score, eligible, budget)
    overlap = int(np.intersect1d(control, candidate, assume_unique=True).size)
    return PairSelection(control=control, candidate=candidate, overlap=overlap)
