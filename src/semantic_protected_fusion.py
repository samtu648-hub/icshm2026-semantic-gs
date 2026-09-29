"""Training-only protected fusion contracts for SEM-110.

SEM-110 combines a frozen SEM-001 class map with a spatially out-of-fold
rendered-RGB teacher.  The fused decision may only switch background and
stay_cable.  Pixels whose SEM-001 prediction is deck, tower or foundation are
bit-identical in the fused prediction, which makes protected-class prediction
masks invariant by construction.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any

import numpy as np

from src.semantic2d import CLASS_COUNT, confusion_metrics, update_confusion

CLASS_NAMES = ("background", "deck", "stay_cable", "tower", "foundation")
PROTECTED_CLASS_IDS = (1, 3, 4)


def validate_prediction(prediction: np.ndarray, *, context: str) -> np.ndarray:
    """Validate a full-frame official five-class ID map."""
    if prediction.dtype != np.uint8 or prediction.ndim != 2:
        raise ValueError(f"{context} must be a uint8 HxW class map")
    if (
        prediction.size == 0
        or int(prediction.min()) < 0
        or int(prediction.max()) >= CLASS_COUNT
    ):
        raise ValueError(f"{context} contains an invalid official class ID")
    return prediction


def validate_probabilities(
    probabilities: np.ndarray, *, shape: tuple[int, int]
) -> np.ndarray:
    """Validate the frozen teacher's five-class probability simplex."""
    if probabilities.dtype not in (np.float16, np.float32, np.float64):
        raise ValueError("SEM-110 teacher probabilities must be floating point")
    if probabilities.shape != (CLASS_COUNT, *shape):
        raise ValueError(
            "SEM-110 teacher probabilities must have shape "
            f"({CLASS_COUNT}, {shape[0]}, {shape[1]}), got {probabilities.shape}"
        )
    if not np.isfinite(probabilities).all() or np.any(probabilities < -1e-5):
        raise ValueError(
            "SEM-110 teacher probabilities are not finite non-negative values"
        )
    if not np.allclose(probabilities.sum(axis=0), 1.0, rtol=0.0, atol=2e-3):
        raise ValueError("SEM-110 teacher probabilities are not a five-class simplex")
    return probabilities.astype(np.float32, copy=False)


def protected_mask(base_prediction: np.ndarray) -> np.ndarray:
    """Return exactly the SEM-001 pixels whose class may never be changed."""
    validate_prediction(base_prediction, context="SEM-110 base prediction")
    return np.isin(base_prediction, PROTECTED_CLASS_IDS)


def fuse_background_stay_cable(
    base_prediction: np.ndarray,
    teacher_probabilities: np.ndarray,
    *,
    stay_cable_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse only the mutable background/stay_cable lane.

    ``stay_cable_threshold`` is applied to the OOF teacher's official class-2
    probability.  On protected SEM-001 predictions, no channel or class from
    the teacher is consulted.  On the remaining pixels, the output is exactly
    background or stay_cable; this intentionally prevents the 2D stream from
    introducing a protected-class error.
    """
    if not 0.0 < float(stay_cable_threshold) < 1.0:
        raise ValueError(
            "SEM-110 stay_cable threshold must lie strictly between zero and one"
        )
    base = validate_prediction(base_prediction, context="SEM-110 base prediction")
    teacher = validate_probabilities(teacher_probabilities, shape=base.shape)
    fixed = protected_mask(base)
    fused = base.copy()
    mutable = ~fixed
    fused[mutable] = np.where(
        teacher[2, mutable] >= float(stay_cable_threshold), 2, 0
    ).astype(np.uint8)
    if not np.array_equal(fused[fixed], base[fixed]):
        raise RuntimeError("SEM-110 modified a protected SEM-001 prediction")
    return fused, fixed


def deterministic_half(name: str) -> str:
    """Return a label-independent stable calibration/check partition name."""
    if not isinstance(name, str) or not name.endswith(".png"):
        raise ValueError("SEM-110 view names must be PNG filenames")
    digest = hashlib.blake2b(
        name.encode("utf-8"), digest_size=8, person=b"sem110-v1"
    ).digest()
    return "calibration" if int.from_bytes(digest, "big") % 2 == 0 else "check"


def partition_training_names(
    names: Iterable[str], *, expected_count: int
) -> dict[str, list[str]]:
    """Partition exactly the authorised training names without label access."""
    ordered = list(names)
    if len(ordered) != expected_count or len(set(ordered)) != expected_count:
        raise ValueError("SEM-110 requires its exact unique training-view list")
    result = {"calibration": [], "check": []}
    for name in ordered:
        result[deterministic_half(name)].append(name)
    if not result["calibration"] or not result["check"]:
        raise RuntimeError("SEM-110 deterministic partition produced an empty half")
    return result


def merge_confusions(confusions: Iterable[np.ndarray]) -> np.ndarray:
    """Combine validated five-class confusions."""
    combined = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    for confusion in confusions:
        if confusion.shape != combined.shape or not np.issubdtype(
            confusion.dtype, np.integer
        ):
            raise ValueError("SEM-110 received an invalid five-class confusion")
        combined += confusion.astype(np.int64, copy=False)
    return combined


def local_confusion(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Build one official target-row/prediction-column confusion matrix."""
    target = validate_prediction(target, context="SEM-110 target")
    prediction = validate_prediction(prediction, context="SEM-110 prediction")
    if target.shape != prediction.shape:
        raise ValueError("SEM-110 target and prediction shapes differ")
    confusion = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    update_confusion(confusion, target, prediction)
    return confusion


def protected_fusion_gate(
    *,
    baseline_metrics: dict[str, Any],
    fused_metrics: dict[str, Any],
    protected_predictions_exact: bool,
    gates: dict[str, Any],
) -> dict[str, Any]:
    """Apply SEM-110's training-only qualification gate."""
    base_fp = int(baseline_metrics["background_to_stay_cable_pixels"])
    fused_fp = int(fused_metrics["background_to_stay_cable_pixels"])
    cable = fused_metrics["per_class"]["stay_cable"]
    protected = fused_metrics["per_class"]
    checks = {
        "protected_prediction_pixels_exact": bool(protected_predictions_exact),
        "mIoU_with_background": float(fused_metrics["mIoU_with_background"])
        >= float(baseline_metrics["mIoU_with_background"]),
        "stay_cable_iou": float(cable["iou"]) >= float(gates["stay_cable_iou_min"]),
        "stay_cable_recall": float(cable["recall"])
        >= float(gates["stay_cable_recall_min"]),
        "background_to_stay_cable_reduction": fused_fp
        <= base_fp
        * (1.0 - float(gates["background_to_stay_cable_reduction_min_fraction"])),
    }
    for name in ("deck", "tower", "foundation"):
        checks[f"{name}_prediction_mask_count_exact"] = int(
            protected[name]["tp"]
        ) + int(protected[name]["fp"]) == int(
            baseline_metrics["per_class"][name]["tp"]
        ) + int(baseline_metrics["per_class"][name]["fp"])
    return {
        "baseline_background_to_stay_cable_pixels": base_fp,
        "fused_background_to_stay_cable_pixels": fused_fp,
        "background_to_stay_cable_reduction_fraction": float(1.0 - fused_fp / base_fp)
        if base_fp
        else 0.0,
        "gates": {
            "stay_cable_iou_min": float(gates["stay_cable_iou_min"]),
            "stay_cable_recall_min": float(gates["stay_cable_recall_min"]),
            "background_to_stay_cable_reduction_min_fraction": float(
                gates["background_to_stay_cable_reduction_min_fraction"]
            ),
            "protected_prediction_pixels_exact": True,
            "mIoU_with_background_drop_max": 0.0,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def selection_key(record: dict[str, Any]) -> tuple[float, float]:
    """Maximise false-positive reduction; break ties toward the higher threshold."""
    return (
        float(record["gate"]["background_to_stay_cable_reduction_fraction"]),
        float(record["stay_cable_threshold"]),
    )


def select_calibration_candidate(
    records: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    """Select only among passing calibration candidates; never inspect check data."""
    passing = [
        record for record in records if record.get("gate", {}).get("passed") is True
    ]
    return max(passing, key=selection_key) if passing else None


def metrics_from_confusion(confusion: np.ndarray) -> dict[str, Any]:
    """Expose the common five-class metric schema with SEM-110-safe typing."""
    metrics = confusion_metrics(confusion)
    if any(metrics["per_class"][name]["iou"] is None for name in CLASS_NAMES):
        raise RuntimeError(
            "SEM-110 expected every official class to occur in its aggregate"
        )
    return metrics
