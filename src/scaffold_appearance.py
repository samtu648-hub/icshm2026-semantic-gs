"""Deterministic appearance-camera selection for frozen Scaffold-GS.

Scaffold-GS learns one appearance embedding per training camera.  Its upstream
test renderer assigns test-camera ordinals to that training-only table, while
the submission adapter historically used the fixed training index zero.  This
module provides a metadata-only alternative: use the embedding belonging to
the nearest training camera center in the official COLMAP frame.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def _camera_name(record: dict[str, Any]) -> str:
    name = str(record.get("img_name", ""))
    if not name:
        raise ValueError("camera record is missing img_name")
    return name.removesuffix(".png")


def _camera_center(record: dict[str, Any]) -> np.ndarray:
    center = np.asarray(record.get("position"), dtype=np.float64)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("camera position must contain three finite values")
    return center


def partition_scaffold_camera_records(
    records: Iterable[dict[str, Any]], validation_names: Iterable[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Recover the exact upstream test/train ordering stored in cameras.json.

    The pinned Scene writes the sorted test-camera block first and the sorted
    training-camera block second.  Training used ``shuffle=False``, so the
    index inside the second block is also the learned appearance embedding ID.
    """

    items = list(records)
    expected_validation = sorted(str(name).removesuffix(".png") for name in validation_names)
    if len(expected_validation) != len(set(expected_validation)):
        raise ValueError("validation camera names must be unique")
    if len(items) <= len(expected_validation):
        raise ValueError("cameras.json contains no training-camera block")
    test_records = items[: len(expected_validation)]
    train_records = items[len(expected_validation) :]
    test_names = [_camera_name(record) for record in test_records]
    if test_names != expected_validation:
        raise ValueError("cameras.json test prefix differs from the frozen validation split")
    train_names = [_camera_name(record) for record in train_records]
    if train_names != sorted(train_names) or set(train_names) & set(test_names):
        raise ValueError("cameras.json training block is not sorted and disjoint")
    for record in items:
        _camera_center(record)
    return test_records, train_records


def nearest_training_appearance(
    target: dict[str, Any], training_records: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    """Select one training embedding by Euclidean camera-center distance.

    Ties are resolved by the learned embedding index, which makes the rule
    deterministic without inspecting RGB pixels or semantic labels.
    """

    training = list(training_records)
    if not training:
        raise ValueError("at least one training camera is required")
    target_center = _camera_center(target)
    centers = np.stack([_camera_center(record) for record in training])
    distances = np.linalg.norm(centers - target_center[None, :], axis=1)
    appearance_id = int(np.argmin(distances))
    return {
        "target_name": _camera_name(target) + ".png",
        "appearance_id": appearance_id,
        "source_name": _camera_name(training[appearance_id]) + ".png",
        "center_distance": float(distances[appearance_id]),
    }
