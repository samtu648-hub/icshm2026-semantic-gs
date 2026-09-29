"""Isolated two-source deck recovery, never called by the frozen submission.

Reliability masks must already come from native cycle/epipolar/coverage tests.
This module does not pretend to compute RoMa matches from class agreement.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SourceEvidence:
    source_name: str
    labels: np.ndarray
    reliable: np.ndarray
    confidence: np.ndarray


def recover_deck(base, deck_probability, sources, *, threshold, target_name, allowed_sources):
    sources = list(sources)
    base = np.asarray(base)
    p = np.asarray(deck_probability)
    if base.ndim != 2 or not np.issubdtype(base.dtype, np.integer) or np.any((base < 0) | (base > 4)):
        raise ValueError("Invalid base class IDs")
    if p.shape != base.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid deck probabilities")
    if threshold is not None and (not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("Invalid calibrated threshold")
    seen, support = set(), np.zeros(base.shape, np.uint16)
    conflicts = np.zeros(base.shape, bool)
    for source in sources:
        if source.source_name == target_name or source.source_name not in allowed_sources or source.source_name in seen:
            raise ValueError("Duplicate, self, or out-of-fold reference source")
        seen.add(source.source_name)
        if source.labels.shape != base.shape or source.reliable.shape != base.shape or source.confidence.shape != base.shape or source.reliable.dtype != bool:
            raise ValueError("Evidence is not on the native target pixel grid")
        valid = source.reliable
        if not np.issubdtype(source.labels.dtype, np.integer) or np.any((source.labels[valid] < 0) | (source.labels[valid] > 4)):
            raise ValueError("Invalid reliable source class")
        if not np.isfinite(source.confidence[valid]).all() or np.any((source.confidence[valid] < 0) | (source.confidence[valid] > 1)):
            raise ValueError("Invalid reliable source confidence")
        support += valid & (source.labels == 1)
        conflicts |= valid & (source.labels != 1)
    change = np.zeros(base.shape, bool) if threshold is None else (base == 0) & (p >= threshold) & (support >= 2) & ~conflicts
    result = base.copy()
    result[change] = 1
    y, x = np.nonzero(change)
    # Compact per-change evidence, sufficient to reconstruct the exact decision.
    trace = {"x": x, "y": y, "old_class": base[change], "new_class": result[change],
             "deck_probability": p[change], "support_count": support[change],
             "reason": "two_distinct_reliable_training_sources_and_calibrated_deck_expert",
             "sources": [{"name": s.source_name, "label": s.labels[change],
                          "reliable": s.reliable[change], "confidence": s.confidence[change]} for s in sources]}
    return result, trace
