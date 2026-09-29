"""Pure helpers for the SEM-100 rendered-RGB semantic-domain preflight.

The preflight deliberately compares an already-trained OOF 2D teacher on a
real training image with that same teacher on the frozen Scaffold RGB rendering
at the exact same camera.  It does not train a model, alter a checkpoint, or
select a validation-time configuration.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np

from src.semantic2d import CLASS_COUNT, CLASS_NAMES

SEMANTIC_PALETTE = np.asarray(
    [
        [30, 30, 30],
        [80, 160, 245],
        [245, 210, 65],
        [220, 80, 80],
        [170, 100, 190],
    ],
    dtype=np.uint8,
)


def select_preflight_oof_views(
    oof_split: dict[str, Any],
    requested_by_fold: dict[str, Iterable[str]],
    *,
    expected_count: int,
) -> list[dict[str, Any]]:
    """Validate and materialize a fixed, spatially OOF training-only plan."""
    validation = set(oof_split.get("fixed_validation_read_only", []))
    fold_lookup = {int(fold["fold"]): set(fold["oof"]) for fold in oof_split["folds"]}
    if set(requested_by_fold) != {str(index) for index in fold_lookup}:
        raise ValueError("SEM-100 requested views must name every and only OOF fold")
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fold in sorted(fold_lookup):
        names = list(requested_by_fold[str(fold)])
        if not names:
            raise ValueError(f"SEM-100 fold {fold} has no requested OOF view")
        for name in names:
            if not isinstance(name, str) or not name.endswith(".png"):
                raise ValueError(f"SEM-100 has an invalid image name: {name!r}")
            if name in seen:
                raise ValueError(f"SEM-100 requested an OOF image twice: {name}")
            if name in validation:
                raise ValueError(f"SEM-100 must not read fixed validation view {name}")
            if name not in fold_lookup[fold]:
                raise ValueError(f"SEM-100 {name} is not an OOF view for fold {fold}")
            seen.add(name)
            selected.append({"name": name, "fold": fold})
    if len(selected) != expected_count:
        raise ValueError(
            f"SEM-100 expected {expected_count} fixed preflight views, got {len(selected)}"
        )
    return selected


def validate_probability_simplex(probabilities: np.ndarray) -> None:
    """Check the five-class tensor contract without assuming a fixed image size."""
    if probabilities.ndim != 3 or probabilities.shape[0] != CLASS_COUNT:
        raise ValueError(
            f"Expected [{CLASS_COUNT}, H, W] probabilities, got {probabilities.shape}"
        )
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
        raise ValueError("Probabilities must be finite and non-negative")
    if not np.allclose(probabilities.sum(axis=0), 1.0, atol=3e-3, rtol=0.0):
        raise ValueError("Five-class probabilities do not form a simplex")


def rgb_psnr_db(reference: np.ndarray, candidate: np.ndarray) -> float | None:
    """Return finite RGB PSNR; identical images use ``None`` rather than Infinity."""
    if (
        reference.shape != candidate.shape
        or reference.ndim != 3
        or reference.shape[2] != 3
    ):
        raise ValueError("RGB PSNR expects same-shaped [H, W, 3] arrays")
    difference = reference.astype(np.float64) - candidate.astype(np.float64)
    mse = float(np.mean(difference * difference))
    return None if mse == 0.0 else float(20.0 * math.log10(255.0 / math.sqrt(mse)))


def semantic_rgb(labels: np.ndarray) -> np.ndarray:
    if labels.ndim != 2 or labels.dtype != np.uint8:
        raise ValueError("Semantic labels must be a uint8 [H, W] array")
    if labels.size and int(labels.max()) >= CLASS_COUNT:
        raise ValueError(f"Semantic labels must be in 0..{CLASS_COUNT - 1}")
    return SEMANTIC_PALETTE[labels]


def render_domain_panel(
    real_rgb: np.ndarray,
    rendered_rgb: np.ndarray,
    target: np.ndarray,
    real_prediction: np.ndarray,
    rendered_prediction: np.ndarray,
) -> np.ndarray:
    """Return real / rendered RGB and GT / real / rendered semantic panels."""
    if (
        real_rgb.shape != rendered_rgb.shape
        or real_rgb.ndim != 3
        or real_rgb.shape[2] != 3
    ):
        raise ValueError("Real and rendered RGB panels must share [H, W, 3]")
    height, width = real_rgb.shape[:2]
    for name, labels in {
        "target": target,
        "real_prediction": real_prediction,
        "rendered_prediction": rendered_prediction,
    }.items():
        if labels.shape != (height, width):
            raise ValueError(
                f"{name} shape {labels.shape} does not match RGB {height}x{width}"
            )
    return np.concatenate(
        [
            real_rgb,
            rendered_rgb,
            semantic_rgb(target),
            semantic_rgb(real_prediction),
            semantic_rgb(rendered_prediction),
        ],
        axis=1,
    )


def domain_metric_delta(
    real_metrics: dict[str, Any], rendered_metrics: dict[str, Any]
) -> dict[str, Any]:
    """Pair the score change caused by replacing real RGB with rendered RGB."""
    real_stay = real_metrics["per_class"][CLASS_NAMES[2]]
    rendered_stay = rendered_metrics["per_class"][CLASS_NAMES[2]]
    real_fp = int(real_metrics["background_to_stay_cable_pixels"])
    rendered_fp = int(rendered_metrics["background_to_stay_cable_pixels"])
    return {
        "mIoU_with_background_delta": float(
            rendered_metrics["mIoU_with_background"]
            - real_metrics["mIoU_with_background"]
        ),
        "mIoU_without_background_delta": float(
            rendered_metrics["mIoU_without_background"]
            - real_metrics["mIoU_without_background"]
        ),
        "stay_cable_IoU_delta": float(rendered_stay["iou"] - real_stay["iou"]),
        "stay_cable_precision_delta": float(
            (rendered_stay["precision"] or 0.0) - (real_stay["precision"] or 0.0)
        ),
        "stay_cable_recall_delta": float(
            (rendered_stay["recall"] or 0.0) - (real_stay["recall"] or 0.0)
        ),
        "background_to_stay_cable_pixels_delta": rendered_fp - real_fp,
        "background_to_stay_cable_fraction_delta": float(
            rendered_metrics["background_to_stay_cable_fraction"]
            - real_metrics["background_to_stay_cable_fraction"]
        ),
    }
