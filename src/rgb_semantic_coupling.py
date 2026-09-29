"""Pure helpers for the RGB/semantic error coupling diagnostic."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


CLASS_COUNT = 5


def pearson_correlation(first: Iterable[float], second: Iterable[float]) -> float | None:
    """Return Pearson's r, or None when either input is constant."""

    x = np.asarray(list(first), dtype=np.float64)
    y = np.asarray(list(second), dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size < 2:
        raise ValueError("correlation inputs must be equal-length 1D arrays with at least two values")
    x = x - x.mean()
    y = y - y.mean()
    denominator = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    if denominator == 0.0:
        return None
    return float(np.dot(x, y) / denominator)


def average_ranks(values: Iterable[float]) -> np.ndarray:
    """Return one-based average ranks, matching the standard tie convention."""

    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("rank input must be one-dimensional")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    return ranks


def spearman_correlation(first: Iterable[float], second: Iterable[float]) -> float | None:
    """Return Spearman's rho using average ranks for ties."""

    return pearson_correlation(average_ranks(first), average_ranks(second))


def confusion_matrix(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Build the fixed five-class target-row/prediction-column confusion matrix."""

    target = np.asarray(target)
    prediction = np.asarray(prediction)
    if target.shape != prediction.shape or target.ndim != 2:
        raise ValueError("semantic target and prediction must be same-shape 2D arrays")
    if target.size and (target.min() < 0 or target.max() >= CLASS_COUNT):
        raise ValueError("semantic target contains an invalid class id")
    if prediction.size and (prediction.min() < 0 or prediction.max() >= CLASS_COUNT):
        raise ValueError("semantic prediction contains an invalid class id")
    encoded = target.astype(np.int64) * CLASS_COUNT + prediction.astype(np.int64)
    return np.bincount(encoded.ravel(), minlength=CLASS_COUNT**2).reshape(
        CLASS_COUNT, CLASS_COUNT
    )


def semantic_metrics(confusion: np.ndarray) -> dict[str, float]:
    """Return the per-view measures used by the coupling audit."""

    confusion = np.asarray(confusion, dtype=np.int64)
    if confusion.shape != (CLASS_COUNT, CLASS_COUNT):
        raise ValueError("confusion must be 5x5")
    ious: list[float] = []
    for class_id in range(CLASS_COUNT):
        tp = int(confusion[class_id, class_id])
        union = int(confusion[class_id, :].sum() + confusion[:, class_id].sum() - tp)
        ious.append(float(tp / union) if union else 0.0)
    return {
        "mIoU_with_background": float(np.mean(ious)),
        "mIoU_without_background": float(np.mean(ious[1:])),
        "stay_cable_IoU": ious[2],
    }


def reference_edge_strength(rgb: np.ndarray) -> np.ndarray:
    """Compute deterministic forward-difference luminance edge strength.

    The last row/column copy their nearest valid forward difference so image
    borders do not receive artificial zero gradients.
    """

    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB image must have HxWx3 shape")
    luminance = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    dx = np.empty_like(luminance)
    dy = np.empty_like(luminance)
    dx[:, :-1] = np.abs(luminance[:, 1:] - luminance[:, :-1])
    dx[:, -1] = dx[:, -2] if luminance.shape[1] > 1 else 0.0
    dy[:-1, :] = np.abs(luminance[1:, :] - luminance[:-1, :])
    dy[-1, :] = dy[-2, :] if luminance.shape[0] > 1 else 0.0
    return 0.5 * (dx + dy)

