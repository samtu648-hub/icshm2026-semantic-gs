"""Pure helpers for the SEM-402 within-view mask-conflict diagnostic."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def mask_signal(foreground: np.ndarray, background: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return the COB-style foreground/background dominance signal in [0, 1]."""
    fg = np.asarray(foreground, dtype=np.float64)
    bg = np.asarray(background, dtype=np.float64)
    if fg.shape != bg.shape:
        raise ValueError("foreground and background arrays must have equal shape")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if np.any(fg < 0) or np.any(bg < 0):
        raise ValueError("contribution masses must be non-negative")
    return np.abs(fg - bg) / (fg + bg + eps)


@dataclass(frozen=True)
class ConflictSelection:
    mean_signal: np.ndarray
    qualified: np.ndarray
    ranked_eligible: np.ndarray


def select_conflicted_parents(
    signal_sum: np.ndarray,
    visible_views: np.ndarray,
    ambiguous_views: np.ndarray,
    eligible: np.ndarray,
    *,
    signal_threshold: float,
    minimum_visible_views: int,
    minimum_ambiguous_views: int,
) -> ConflictSelection:
    """Apply fixed coverage/conflict gates and rank by low mean mask signal."""
    sums = np.asarray(signal_sum, dtype=np.float64).reshape(-1)
    visible = np.asarray(visible_views, dtype=np.int64).reshape(-1)
    ambiguous = np.asarray(ambiguous_views, dtype=np.int64).reshape(-1)
    mask = np.asarray(eligible, dtype=bool).reshape(-1)
    if not (sums.shape == visible.shape == ambiguous.shape == mask.shape):
        raise ValueError("all arrays must have equal length")
    if minimum_visible_views < 1 or minimum_ambiguous_views < 1:
        raise ValueError("minimum view counts must be positive")
    if not 0.0 <= signal_threshold <= 1.0:
        raise ValueError("signal threshold must be in [0, 1]")
    mean = np.full(len(sums), np.inf, dtype=np.float64)
    covered = visible > 0
    mean[covered] = sums[covered] / visible[covered]
    qualified = (
        mask
        & (visible >= minimum_visible_views)
        & (ambiguous >= minimum_ambiguous_views)
        & (mean < signal_threshold)
    )
    ids = np.flatnonzero(mask)
    order = np.lexsort((ids, mean[ids]))
    return ConflictSelection(mean_signal=mean, qualified=qualified, ranked_eligible=ids[order])
