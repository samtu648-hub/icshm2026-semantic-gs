"""Strict probability replay and independently counted segmentation metrics."""
import numpy as np
from src.quality_protocol import CLASSES


def check_replay(actual, stored, png):
    if actual.dtype != np.float32 or stored.dtype != np.float32 or actual.shape != stored.shape or actual.ndim != 3 or actual.shape[0] != 5:
        raise ValueError("Expected matching native float32 five-class probabilities")
    if not np.isfinite(actual).all() or np.any(actual < 0) or not np.allclose(actual.sum(0), 1, rtol=0, atol=2e-5):
        raise ValueError("Invalid replayed probabilities")
    if not np.array_equal(actual, stored) or png.dtype != np.uint8 or not np.array_equal(png, actual.argmax(0)):
        raise ValueError("Independent semantic probabilities or PNG differ from saved training export")


def independent_confusion(target, prediction):
    if target.shape != prediction.shape or target.ndim != 2 or target.dtype != np.uint8 or prediction.dtype != np.uint8:
        raise ValueError("Expected same-grid uint8 semantic class images")
    if target.max() > 4 or prediction.max() > 4:
        raise ValueError("Class ID outside official 0..4")
    fast = np.bincount(target.ravel().astype(np.int64) * 5 + prediction.ravel(), minlength=25).reshape(5, 5)
    independent = np.array([[np.count_nonzero((target == t) & (prediction == p)) for p in range(5)] for t in range(5)], dtype=np.int64)
    if not np.array_equal(fast, independent) or fast.sum() != target.size:
        raise ValueError("Independent confusion arithmetic disagrees")
    return fast


def metrics(matrix):
    matrix = np.asarray(matrix)
    if matrix.shape != (5, 5) or not np.issubdtype(matrix.dtype, np.integer) or (matrix < 0).any():
        raise ValueError("Expected nonnegative integer 5x5 confusion")
    result = {"per_class": {}, "confusion": matrix.tolist()}
    for i, name in enumerate(CLASSES):
        tp = int(matrix[i, i])
        fp, fn = int(matrix[:, i].sum() - tp), int(matrix[i].sum() - tp)
        result["per_class"][name] = {"tp": tp, "fp": fp, "fn": fn,
            "IoU": tp / (tp + fp + fn) if tp + fp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None}
    for key, names in (("mIoU_with_background", CLASSES), ("mIoU_without_background", CLASSES[1:])):
        values = [result["per_class"][n]["IoU"] for n in names if result["per_class"][n]["IoU"] is not None]
        result[key] = float(np.mean(values)) if values else None
    return result
