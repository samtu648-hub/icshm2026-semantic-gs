"""Pure protocol checks for the SEM-102 rendered-domain final model."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

CLASS_NAMES = ("background", "deck", "stay_cable", "tower", "foundation")


def require_exact_view_set(
    names: Iterable[str], expected: Iterable[str], *, context: str
) -> list[str]:
    """Return an ordered view plan after rejecting duplicates or mismatches."""
    observed = list(names)
    required = list(expected)
    if observed != required or len(observed) != len(set(observed)):
        raise ValueError(f"{context} does not exactly match its frozen view plan")
    return observed


def validate_sem101_parent_summary(
    summary: dict[str, Any], *, expected_rendered_manifest_sha256: str
) -> None:
    """Require the completed, passing SEM-101 OOF record before SEM-102 starts."""
    if summary.get("experiment_id") != "SEM-101-RENDERED-SEG-OOF":
        raise ValueError("SEM-102 requires the SEM-101 rendered-domain OOF summary")
    if summary.get("status") != "completed_oof_gate_passed":
        raise ValueError("SEM-102 requires SEM-101's completed passing OOF gate")
    gate = summary.get("oof_qualification_gate", {})
    if gate.get("passed") is not True or not all(gate.get("checks", {}).values()):
        raise ValueError("SEM-101 parent OOF qualification gate did not pass")
    for record in summary.get("fold_results", []):
        if not isinstance(record.get("checkpoint_sha256"), str):
            raise TypeError("SEM-101 parent summary has an invalid fold checkpoint")
    if len(summary.get("fold_results", [])) != 3:
        raise ValueError("SEM-101 parent summary does not contain exactly three folds")
    # The summary itself does not duplicate the rendered dataset hash.  Its
    # fold contracts did, and SEM-102 records that immutable manifest hash in
    # configuration and validates it again from the dataset manifest.
    if len(expected_rendered_manifest_sha256) != 64:
        raise ValueError("SEM-102 rendered dataset manifest hash is invalid")


def validate_final_training_result(
    result: dict[str, Any], *, checkpoint_sha256: str, config_sha256: str
) -> None:
    """Require a fully frozen SEM-102 checkpoint before validation is opened."""
    if result.get("experiment_id") != "SEM-102-RENDERED-SEG-FINAL":
        raise ValueError("Final checkpoint result belongs to another experiment")
    if result.get("status") != "completed_final_checkpoint_frozen":
        raise ValueError("SEM-102 final checkpoint is not frozen")
    if result.get("checkpoint_reload") != "passed":
        raise ValueError("SEM-102 final checkpoint did not pass reload")
    if result.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("SEM-102 final checkpoint hash differs from its result")
    if result.get("config_sha256") != config_sha256:
        raise ValueError("SEM-102 final checkpoint used another config")
    if result.get("fixed_validation_opened") is not False:
        raise ValueError("SEM-102 final training opened fixed validation")
    if result.get("unlabeled_rgb_opened") is not False:
        raise ValueError("SEM-102 final training opened unlabelled RGB")
    if result.get("real_rgb_input_opened") is not False:
        raise ValueError("SEM-102 final training opened real RGB")


def final_retention_gate(
    metrics: dict[str, Any], baseline: dict[str, Any], gates: dict[str, Any]
) -> dict[str, Any]:
    """Apply the fixed SEM-102 replacement gate against recorded SEM-001 values."""
    per_class = metrics.get("per_class", {})
    baseline_class = baseline.get("per_class", {})
    protected = tuple(gates["protected_classes"])
    checks = {
        "mIoU_with_background": float(metrics["mIoU_with_background"])
        >= float(baseline["mIoU_with_background"])
        - float(gates["mIoU_with_background_drop_max"]),
        "stay_cable_iou": float(per_class["stay_cable"]["iou"])
        >= float(baseline_class["stay_cable"]["iou"])
        - float(gates["stay_cable_iou_drop_max"]),
        "stay_cable_recall": float(per_class["stay_cable"]["recall"])
        >= float(baseline_class["stay_cable"]["recall"])
        - float(gates["stay_cable_recall_drop_max"]),
        "background_to_stay_cable_reduction": int(
            metrics["background_to_stay_cable_pixels"]
        )
        <= int(baseline["background_to_stay_cable_pixels"])
        * (1.0 - float(gates["background_to_stay_cable_reduction_min_fraction"])),
        "worst_10_percent_mIoU_without_background": float(
            metrics["worst_10_percent"]["mean_mIoU_without_background"]
        )
        >= float(baseline["worst_10_percent_mIoU_without_background"])
        - float(gates["worst_10_percent_mIoU_without_background_drop_max"]),
        "challenge_140_146_mIoU_without_background": float(
            metrics["challenge_140_146"]["mean_mIoU_without_background"]
        )
        >= float(baseline["challenge_140_146_mIoU_without_background"])
        - float(gates["challenge_140_146_mIoU_without_background_drop_max"]),
    }
    for name in protected:
        checks[f"{name}_iou"] = float(per_class[name]["iou"]) >= float(
            baseline_class[name]["iou"]
        ) - float(gates["protected_class_iou_drop_max"])
    return {
        "baseline": baseline,
        "gates": {
            "mIoU_with_background_drop_max": float(
                gates["mIoU_with_background_drop_max"]
            ),
            "stay_cable_iou_drop_max": float(gates["stay_cable_iou_drop_max"]),
            "stay_cable_recall_drop_max": float(gates["stay_cable_recall_drop_max"]),
            "protected_classes": list(protected),
            "protected_class_iou_drop_max": float(
                gates["protected_class_iou_drop_max"]
            ),
            "background_to_stay_cable_reduction_min_fraction": float(
                gates["background_to_stay_cable_reduction_min_fraction"]
            ),
            "worst_10_percent_mIoU_without_background_drop_max": float(
                gates["worst_10_percent_mIoU_without_background_drop_max"]
            ),
            "challenge_140_146_mIoU_without_background_drop_max": float(
                gates["challenge_140_146_mIoU_without_background_drop_max"]
            ),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
