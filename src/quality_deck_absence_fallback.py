"""Deterministic deck-only fallback for native predictions with no deck pixels."""

from __future__ import annotations

import numpy as np


def apply_deck_absence_fallback(
    native: np.ndarray,
    fallback: np.ndarray,
    *,
    minimum_fallback_deck_pixels: int,
    allowed_native_classes: tuple[int, ...] = (0, 2),
) -> tuple[np.ndarray, dict[str, int | bool | float]]:
    if (
        native.dtype != np.uint8
        or fallback.dtype != np.uint8
        or native.ndim != 2
        or native.shape != fallback.shape
        or native.max(initial=0) > 4
        or fallback.max(initial=0) > 4
    ):
        raise ValueError("Expected matched uint8 five-class label images")
    native_deck = int(np.count_nonzero(native == 1))
    fallback_deck = int(np.count_nonzero(fallback == 1))
    triggered = native_deck == 0 and fallback_deck >= minimum_fallback_deck_pixels
    output = native.copy()
    if triggered:
        allowed = np.isin(native, np.asarray(allowed_native_classes, dtype=np.uint8))
        output[(fallback == 1) & allowed] = 1
    changed = int(np.count_nonzero(output != native))
    return output, {
        "triggered": triggered,
        "native_deck_pixels": native_deck,
        "fallback_deck_pixels": fallback_deck,
        "added_deck_pixels": changed,
        "changed_pixels": changed,
        "total_pixels": int(native.size),
        "changed_fraction": changed / native.size,
    }
