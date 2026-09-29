"""Training-only contracts for the SEM-101 rendered-RGB 2D OOF study.

SEM-101 changes exactly one SEM-051 variable: SegFormer sees a frozen
Scaffold-GS rendering rather than a released photograph.  These helpers are
intentionally NumPy-only so that the access, OOF-completeness, and fixed-gate
contracts can be tested without CUDA or model weights.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from src.semantic2d import CLASS_COUNT
from src.semantic_render_domain import semantic_rgb


def select_rendered_training_views(
    split: dict[str, Any], *, expected_count: int = 219
) -> list[str]:
    """Return the frozen labelled training list after strict leakage checks."""
    names = list(split.get("semantic_train_labeled", []))
    validation = set(split.get("semantic_validation_labeled", []))
    unlabeled = set(split.get("excluded_unlabeled", []))
    if len(names) != expected_count or len(set(names)) != expected_count:
        raise ValueError(
            f"SEM-101 requires {expected_count} unique labelled training views, got "
            f"{len(names)} / {len(set(names))}"
        )
    if any(not isinstance(name, str) or not name.endswith(".png") for name in names):
        raise ValueError("SEM-101 training view names must be PNG filenames")
    names_set = set(names)
    if names_set & validation:
        raise ValueError("SEM-101 training render plan leaks fixed validation")
    if names_set & unlabeled:
        raise ValueError("SEM-101 training render plan leaks unlabeled RGB")
    return names


def validate_rendered_dataset_manifest(
    manifest: dict[str, Any], training_names: Iterable[str]
) -> None:
    """Ensure a rendered dataset is complete and was created training-only."""
    expected = list(training_names)
    access = manifest.get("training_only_access", {})
    if manifest.get("experiment_id") != "SEM-101-RENDERED-SEG-OOF":
        raise ValueError("Rendered RGB manifest belongs to another experiment")
    if manifest.get("status") != "completed_training_rgb_render":
        raise ValueError(
            "Rendered RGB dataset is not a completed SEM-101 training render"
        )
    if access.get("fixed_validation_views_opened") is not False:
        raise ValueError(
            "Rendered RGB manifest does not prove fixed validation stayed closed"
        )
    if access.get("unlabeled_images_opened") is not False:
        raise ValueError(
            "Rendered RGB manifest does not prove unlabeled RGB stayed closed"
        )
    if access.get("labels_opened") is not False:
        raise ValueError("Rendered RGB creation must not open labels")
    if access.get("external_camera_mode_present") is not False:
        raise ValueError("Rendered RGB creation must not expose external-camera mode")
    rendered = manifest.get("rendered_images", [])
    names = [record.get("name") for record in rendered]
    if names != expected or len(set(names)) != len(expected):
        raise ValueError(
            "Rendered RGB manifest names do not exactly match the frozen training plan"
        )
    if any(
        not isinstance(record.get("sha256"), str) or len(record["sha256"]) != 64
        for record in rendered
    ):
        raise ValueError("Rendered RGB manifest has an invalid image hash")


def validate_oof_prediction_plan(
    *,
    training_names: Iterable[str],
    oof_split: dict[str, Any],
    fold_records: Iterable[dict[str, Any]],
) -> None:
    """Require exactly one prediction for every training view from its OOF fold."""
    expected_training = set(training_names)
    validation = set(oof_split.get("fixed_validation_read_only", []))
    expected_by_fold = {
        int(fold["fold"]): set(fold["oof"]) for fold in oof_split.get("folds", [])
    }
    if set(expected_by_fold) != {0, 1, 2}:
        raise ValueError("SEM-101 requires the frozen three-fold SEM-005 OOF split")
    observed: set[str] = set()
    for record in fold_records:
        fold = int(record["fold"])
        names = list(record["names"])
        if fold not in expected_by_fold:
            raise ValueError(f"SEM-101 has an unknown OOF fold {fold}")
        if set(names) != expected_by_fold[fold] or len(names) != len(set(names)):
            raise ValueError(
                f"SEM-101 fold {fold} predictions do not exactly match its OOF views"
            )
        if set(names) & validation:
            raise ValueError(f"SEM-101 fold {fold} leaks fixed validation")
        if observed & set(names):
            raise ValueError("SEM-101 records an OOF view more than once")
        observed.update(names)
    if observed != expected_training:
        raise ValueError(
            "SEM-101 OOF predictions do not cover exactly the 219 training views"
        )


def oof_qualification_gate(
    metrics: dict[str, Any], gates: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate the pre-registered rendered-domain OOF qualification gate."""
    cable = metrics.get("per_class", {}).get("stay_cable", {})
    checks = {
        "mIoU_with_background": float(metrics["mIoU_with_background"])
        >= float(gates["mIoU_with_background_min"]),
        "stay_cable_iou": float(cable["iou"]) >= float(gates["stay_cable_iou_min"]),
        "stay_cable_precision": float(cable["precision"])
        >= float(gates["stay_cable_precision_min"]),
        "stay_cable_recall": float(cable["recall"])
        >= float(gates["stay_cable_recall_min"]),
        "background_to_stay_cable_fraction": float(
            metrics["background_to_stay_cable_fraction"]
        )
        <= float(gates["background_to_stay_cable_fraction_max"]),
    }
    return {
        "gates": {
            "mIoU_with_background_min": float(gates["mIoU_with_background_min"]),
            "stay_cable_iou_min": float(gates["stay_cable_iou_min"]),
            "stay_cable_precision_min": float(gates["stay_cable_precision_min"]),
            "stay_cable_recall_min": float(gates["stay_cable_recall_min"]),
            "background_to_stay_cable_fraction_max": float(
                gates["background_to_stay_cable_fraction_max"]
            ),
        },
        "selection_rule": gates.get("selection_rule"),
        "checks": checks,
        "passed": all(checks.values()),
    }


def rendered_segmentation_panel(
    rendered_rgb: np.ndarray, target: np.ndarray, prediction: np.ndarray
) -> np.ndarray:
    """Build a fixed rendered-RGB / GT / prediction / error training panel."""
    if (
        rendered_rgb.dtype != np.uint8
        or rendered_rgb.ndim != 3
        or rendered_rgb.shape[2] != 3
    ):
        raise ValueError("Rendered RGB panel input must be uint8 HxWx3")
    if target.shape != prediction.shape or target.shape != rendered_rgb.shape[:2]:
        raise ValueError("Rendered RGB panel shapes differ")
    if target.ndim != 2 or target.min() < 0 or target.max() >= CLASS_COUNT:
        raise ValueError("Rendered RGB panel target has an invalid class ID")
    if prediction.min() < 0 or prediction.max() >= CLASS_COUNT:
        raise ValueError("Rendered RGB panel prediction has an invalid class ID")
    error = np.zeros_like(rendered_rgb)
    error[target != prediction] = np.array([255, 48, 48], dtype=np.uint8)
    return np.concatenate(
        [rendered_rgb, semantic_rgb(target), semantic_rgb(prediction), error], axis=1
    )
