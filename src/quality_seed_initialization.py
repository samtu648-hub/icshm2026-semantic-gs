"""Seed-aware Mask2Former initialization contracts."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np


FRESH_CLASSIFIER_KEYS = (
    "sem_seg_head.predictor.class_embed.weight",
    "sem_seg_head.predictor.class_embed.bias",
)
FRESH_CLASSIFIER_SHAPES = {
    FRESH_CLASSIFIER_KEYS[0]: [6, 256],
    FRESH_CLASSIFIER_KEYS[1]: [6],
}


def tensor_manifest(state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Record exact tensor identities without serializing tensor values."""
    manifest: dict[str, dict[str, Any]] = {}
    for name, value in sorted(state.items()):
        if hasattr(value, "detach"):
            array = value.detach().cpu().contiguous().numpy()
        else:
            array = np.ascontiguousarray(np.asarray(value))
        finite = bool(np.isfinite(array).all()) if array.dtype.kind in "fc" else True
        manifest[name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "finite": finite,
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    return manifest


def validate_same_seed_pair(
    control: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    control_load: Mapping[str, Any],
    candidate_load: Mapping[str, Any],
) -> dict[str, Any]:
    """Require exact full-state and pretrained-load equality for paired arms."""
    if control_load != candidate_load:
        raise ValueError("Paired pretrained load reports differ")
    if control.keys() != candidate.keys():
        raise ValueError("Paired model-state tensor names differ")
    changed = [name for name in control if control[name] != candidate[name]]
    if changed:
        raise ValueError(f"Same-seed paired model states differ: {changed}")
    return {
        "tensor_count": len(control),
        "all_tensors_exact": True,
        "pretrained_load_reports_exact": True,
        "classifier_head_exact": all(
            control[name] == candidate[name] for name in FRESH_CLASSIFIER_KEYS
        ),
    }


def validate_cross_seed_reference(
    current: Mapping[str, Mapping[str, Any]],
    reference: Mapping[str, Mapping[str, Any]],
    current_load: Mapping[str, Any],
    reference_load: Mapping[str, Any],
) -> dict[str, Any]:
    """Allow only the two fresh five-class head tensors to vary by seed."""
    if current_load != reference_load:
        raise ValueError("Cross-seed pretrained load reports differ")
    if current.keys() != reference.keys():
        raise ValueError("Cross-seed model-state tensor names differ")
    if len(current) != 650:
        raise ValueError(f"Expected 650 model-state tensors, found {len(current)}")

    for name, shape in FRESH_CLASSIFIER_SHAPES.items():
        for role, record in (
            ("current", current[name]),
            ("reference", reference[name]),
        ):
            if record["shape"] != shape:
                raise ValueError(
                    f"Fresh classifier tensor has wrong shape: {role} {name}"
                )
            if not record["finite"]:
                raise ValueError(f"Fresh classifier tensor is nonfinite: {role} {name}")

    changed = [name for name in current if current[name] != reference[name]]
    unexpected = sorted(set(changed) - set(FRESH_CLASSIFIER_KEYS))
    if unexpected:
        raise ValueError(f"Non-head tensors differ across seeds: {unexpected}")
    missing_expected = sorted(set(FRESH_CLASSIFIER_KEYS) - set(changed))
    if missing_expected:
        raise ValueError(
            f"Expected fresh classifier tensors to differ across seeds: {missing_expected}"
        )
    return {
        "tensor_count": len(current),
        "non_head_tensor_count": len(current) - len(FRESH_CLASSIFIER_KEYS),
        "non_head_tensors_exact": True,
        "allowed_changed_tensors": list(FRESH_CLASSIFIER_KEYS),
        "pretrained_load_reports_exact": True,
        "current_head_sha256": {
            name: current[name]["sha256"] for name in FRESH_CLASSIFIER_KEYS
        },
        "reference_head_sha256": {
            name: reference[name]["sha256"] for name in FRESH_CLASSIFIER_KEYS
        },
        "current_head_shapes": {
            name: current[name]["shape"] for name in FRESH_CLASSIFIER_KEYS
        },
        "current_heads_finite": True,
        "cross_seed_head_difference_expected": True,
    }
