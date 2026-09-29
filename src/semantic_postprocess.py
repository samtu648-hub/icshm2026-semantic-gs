"""Pure helpers for the SEM-070 conservative semantic post-process.

The post-process is intentionally an inference-only operation.  It never
touches Scaffold RGB tensors or the SEM-001 checkpoint; it only transfers the
probability mass of selected per-anchor ``stay_cable`` predictions to
``background`` immediately before semantic rasterization.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

CLASS_NAMES = ("background", "deck", "stay_cable", "tower", "foundation")
CLASS_COUNT = len(CLASS_NAMES)
BACKGROUND = 0
STAY_CABLE = 2
PROTECTED_CLASSES = (1, 3, 4)


def training_half(image_name: str) -> int:
    """Return the preregistered deterministic hash-parity training half."""

    return int(hashlib.sha256(image_name.encode("utf-8")).hexdigest()[0], 16) & 1


def split_training_views(names: list[str], *, calibration_half: int = 0, check_half: int = 1) -> tuple[list[str], list[str]]:
    """Split only the supplied semantic-training names by stable hash parity."""

    if calibration_half == check_half or {calibration_half, check_half} != {0, 1}:
        raise ValueError("calibration_half and check_half must be the distinct parity values 0 and 1")
    normalized = [str(name) for name in names]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Training view names must be unique")
    calibration = [name for name in normalized if training_half(name) == calibration_half]
    check = [name for name in normalized if training_half(name) == check_half]
    if not calibration or not check:
        raise ValueError("Hash-parity split must produce non-empty calibration and check halves")
    return calibration, check


def validate_suppression_mask(mask: Any, anchor_count: int) -> np.ndarray:
    """Validate and return the only supported mask representation.

    The on-disk NPZ deliberately contains no anchor ID array: position ``i``
    is the frozen Scaffold anchor ID ``i``.  Metadata records the count and
    hash separately so the file remains a single ``[anchor_count]`` bool mask.
    """

    if int(anchor_count) <= 0:
        raise ValueError("anchor_count must be positive")
    array = np.asarray(mask)
    if array.ndim != 1 or array.shape[0] != int(anchor_count):
        raise ValueError(f"suppression mask must have shape [{int(anchor_count)}]")
    if array.dtype != np.bool_:
        raise TypeError("suppression mask must use boolean dtype")
    return np.ascontiguousarray(array, dtype=np.bool_)


def load_suppression_mask(path: Any, anchor_count: int) -> np.ndarray:
    """Load a strict one-key ``postprocess_mask.npz`` artifact."""

    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"mask"}:
            raise ValueError("postprocess_mask.npz must contain exactly one key: mask")
        return validate_suppression_mask(payload["mask"], anchor_count)


def save_suppression_mask(path: Any, mask: Any, anchor_count: int) -> str:
    """Write a compressed one-key mask and return its SHA-256 digest."""

    from pathlib import Path

    validated = validate_suppression_mask(mask, anchor_count)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, mask=validated)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return digest


def _validate_numpy_simplex(probabilities: np.ndarray) -> None:
    if probabilities.ndim < 2 or probabilities.shape[-1] != CLASS_COUNT:
        raise ValueError("probabilities must have a final dimension of five classes")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    sums = probabilities.sum(axis=-1)
    if not np.allclose(sums, 1.0, rtol=1e-5, atol=1e-5):
        raise ValueError("probabilities must lie on the five-class simplex")


def apply_anchor_probability_transfer(probabilities: np.ndarray, suppression_mask: Any) -> np.ndarray:
    """Apply ``p_background += p_stay_cable; p_stay_cable = 0`` to anchors."""

    values = np.asarray(probabilities)
    _validate_numpy_simplex(values)
    mask = validate_suppression_mask(suppression_mask, values.shape[-2])
    output = values.copy()
    if not bool(mask.any()):
        return output
    selected = output[..., mask, :]
    selected = selected.copy()
    selected[..., BACKGROUND] += selected[..., STAY_CABLE]
    selected[..., STAY_CABLE] = 0.0
    output[..., mask, :] = selected
    return output


def apply_offset_probability_transfer(
    probabilities: Any,
    offset_anchor_indices: Any,
    suppression_mask: Any,
) -> Any:
    """Torch implementation used by the frozen semantic renderer.

    ``probabilities`` is ``[generated_gaussians, 5]`` and the offset-to-anchor
    mapping is produced by the exact frozen Scaffold generator.  This function
    deliberately avoids an in-place write to the input tensor so a zero mask
    is bit-identical and no training graph can be attached accidentally.
    """

    import torch

    if not isinstance(probabilities, torch.Tensor):
        raise TypeError("probabilities must be a torch.Tensor")
    if probabilities.ndim != 2 or probabilities.shape[1] != CLASS_COUNT:
        raise ValueError("probabilities must have shape [generated_gaussians, 5]")
    if not torch.is_floating_point(probabilities):
        raise TypeError("probabilities must be floating point")
    if not bool(torch.isfinite(probabilities).all()) or bool((probabilities < 0).any()):
        raise ValueError("probabilities must be finite and non-negative")
    if not bool(torch.allclose(probabilities.sum(dim=1), torch.ones(probabilities.shape[0], device=probabilities.device, dtype=probabilities.dtype), rtol=1e-5, atol=1e-5)):
        raise ValueError("probabilities must lie on the five-class simplex")
    indices = torch.as_tensor(offset_anchor_indices, dtype=torch.long, device=probabilities.device)
    if indices.ndim != 1 or indices.shape[0] != probabilities.shape[0]:
        raise ValueError("offset_anchor_indices must match generated_gaussians")
    mask = torch.as_tensor(suppression_mask, dtype=torch.bool, device=probabilities.device)
    if mask.ndim != 1:
        raise ValueError("suppression mask must be one-dimensional")
    if indices.numel() and (bool((indices < 0).any()) or bool((indices >= mask.shape[0]).any())):
        raise ValueError("offset_anchor_indices contains an invalid frozen anchor ID")
    selected = mask[indices]
    output = probabilities.clone()
    if not bool(selected.any()):
        return output
    output[selected, BACKGROUND] = output[selected, BACKGROUND] + output[selected, STAY_CABLE]
    output[selected, STAY_CABLE] = 0.0
    return output


def new_anchor_statistics(anchor_count: int) -> dict[str, np.ndarray]:
    """Allocate per-half training-only dominant-anchor counters."""

    if int(anchor_count) <= 0:
        raise ValueError("anchor_count must be positive")
    shape = (2, int(anchor_count))
    return {
        "support_views": np.zeros(shape, dtype=np.int32),
        "dominant_pixels": np.zeros(shape, dtype=np.int64),
        "gt_background_pixels": np.zeros(shape, dtype=np.int64),
        "gt_stay_cable_pixels": np.zeros(shape, dtype=np.int64),
        "background_to_stay_cable_pixels": np.zeros(shape, dtype=np.int64),
        "stay_cable_true_positive_pixels": np.zeros(shape, dtype=np.int64),
    }


def update_anchor_statistics(
    statistics: dict[str, np.ndarray],
    *,
    train_half: int,
    anchor_indices: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
) -> None:
    """Update counters from exact dominant-anchor pixels for one labeled view."""

    if int(train_half) not in (0, 1):
        raise ValueError("train_half must be 0 or 1")
    if anchor_indices.shape != target.shape or target.shape != prediction.shape or target.ndim != 2:
        raise ValueError("anchor_indices, target and prediction must be matching 2-D arrays")
    anchor_count = statistics["support_views"].shape[1]
    flat_anchor = anchor_indices.reshape(-1).astype(np.int64, copy=False)
    flat_target = target.reshape(-1).astype(np.int64, copy=False)
    flat_prediction = prediction.reshape(-1).astype(np.int64, copy=False)
    valid = (flat_anchor >= 0) & (flat_anchor < anchor_count)
    if not bool(valid.any()):
        return
    ids = flat_anchor[valid]
    labels = flat_target[valid]
    predictions = flat_prediction[valid]
    unique_ids = np.unique(ids)
    statistics["support_views"][train_half, unique_ids] += 1
    statistics["dominant_pixels"][train_half] += np.bincount(ids, minlength=anchor_count).astype(np.int64)
    bg = labels == BACKGROUND
    cable = labels == STAY_CABLE
    fp = bg & (predictions == STAY_CABLE)
    tp = cable & (predictions == STAY_CABLE)
    statistics["gt_background_pixels"][train_half] += np.bincount(ids[bg], minlength=anchor_count).astype(np.int64)
    statistics["gt_stay_cable_pixels"][train_half] += np.bincount(ids[cable], minlength=anchor_count).astype(np.int64)
    statistics["background_to_stay_cable_pixels"][train_half] += np.bincount(ids[fp], minlength=anchor_count).astype(np.int64)
    statistics["stay_cable_true_positive_pixels"][train_half] += np.bincount(ids[tp], minlength=anchor_count).astype(np.int64)


def finalize_anchor_rows(statistics: dict[str, np.ndarray], anchor_logits: np.ndarray, *, calibration_half: int = 0) -> list[dict[str, Any]]:
    """Create deterministic, JSON/CSV-ready rows from calibration statistics."""

    logits = np.asarray(anchor_logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != CLASS_COUNT or logits.shape[0] != statistics["support_views"].shape[1]:
        raise ValueError("anchor_logits shape does not match anchor statistics")
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    base_class = probabilities.argmax(axis=1)
    rows: list[dict[str, Any]] = []
    for anchor_id in range(logits.shape[0]):
        bg = int(statistics["gt_background_pixels"][calibration_half, anchor_id])
        cable = int(statistics["gt_stay_cable_pixels"][calibration_half, anchor_id])
        fp = int(statistics["background_to_stay_cable_pixels"][calibration_half, anchor_id])
        tp = int(statistics["stay_cable_true_positive_pixels"][calibration_half, anchor_id])
        rows.append(
            {
                "anchor_index": anchor_id,
                "base_class_id": int(base_class[anchor_id]),
                "base_class_name": CLASS_NAMES[int(base_class[anchor_id])],
                "base_stay_cable_probability": float(probabilities[anchor_id, STAY_CABLE]),
                "stay_cable_leaning": bool(int(base_class[anchor_id]) == STAY_CABLE),
                "calibration_support_views": int(statistics["support_views"][calibration_half, anchor_id]),
                "calibration_dominant_pixels": int(statistics["dominant_pixels"][calibration_half, anchor_id]),
                "calibration_gt_background_support": bg,
                "calibration_gt_stay_cable_support": cable,
                "calibration_background_to_stay_cable_pixels": fp,
                "calibration_true_stay_cable_pixels": tp,
                "calibration_false_positive_rate": float(fp / bg) if bg else 0.0,
                "calibration_true_cable_retention": float(tp / cable) if cable else None,
                "check_support_views": int(statistics["support_views"][1 - calibration_half, anchor_id]),
                "check_dominant_pixels": int(statistics["dominant_pixels"][1 - calibration_half, anchor_id]),
                "check_gt_background_support": int(statistics["gt_background_pixels"][1 - calibration_half, anchor_id]),
                "check_gt_stay_cable_support": int(statistics["gt_stay_cable_pixels"][1 - calibration_half, anchor_id]),
                "check_background_to_stay_cable_pixels": int(statistics["background_to_stay_cable_pixels"][1 - calibration_half, anchor_id]),
                "check_true_stay_cable_pixels": int(statistics["stay_cable_true_positive_pixels"][1 - calibration_half, anchor_id]),
            }
        )
    return rows


def mask_for_threshold(rows: list[dict[str, Any]], threshold: float, *, minimum_support_views: int, minimum_dominant_pixels: int) -> np.ndarray:
    """Build a candidate mask using calibration-only anchor rows."""

    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be in [0,1]")
    mask = np.zeros(len(rows), dtype=np.bool_)
    for row in rows:
        mask[int(row["anchor_index"])] = bool(
            row["stay_cable_leaning"]
            and int(row["calibration_support_views"]) >= int(minimum_support_views)
            and int(row["calibration_dominant_pixels"]) >= int(minimum_dominant_pixels)
            and float(row["calibration_false_positive_rate"]) >= float(threshold)
        )
    return mask


def semantic_error_image(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Return the fixed four-way semantic error palette used by SEM-070."""

    if target.shape != prediction.shape or target.ndim != 2:
        raise ValueError("target and prediction must be matching 2-D maps")
    error = np.zeros((*target.shape, 3), dtype=np.uint8)
    equal = target == prediction
    error[equal] = (28, 28, 28)
    error[(target == BACKGROUND) & (prediction == STAY_CABLE)] = (235, 65, 54)
    error[(target == STAY_CABLE) & (prediction == BACKGROUND)] = (66, 133, 244)
    other = ~equal & ~((target == BACKGROUND) & (prediction == STAY_CABLE)) & ~((target == STAY_CABLE) & (prediction == BACKGROUND))
    error[other] = (178, 102, 255)
    return error


def semantic_quad(target: np.ndarray, raw: np.ndarray, cleaned: np.ndarray) -> np.ndarray:
    """Return GT / raw / cleaned / cleaned-error panels with fixed class colors."""

    if target.shape != raw.shape or raw.shape != cleaned.shape:
        raise ValueError("semantic maps must have matching shapes")
    palette = np.asarray(
        [[0, 0, 0], [230, 86, 37], [255, 210, 0], [38, 139, 210], [133, 153, 0]], dtype=np.uint8
    )
    return np.concatenate((palette[target], palette[raw], palette[cleaned], semantic_error_image(target, cleaned)), axis=1)
