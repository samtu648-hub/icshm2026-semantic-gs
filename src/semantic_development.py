"""Explicitly non-blind development ranking for protected semantic fusion."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def development_score(metrics: dict[str, Any]) -> float:
    """Balance global quality, cable overlap and cable completeness for dev ranking."""
    cable = metrics["per_class"]["stay_cable"]
    values = (
        metrics["mIoU_with_background"],
        cable["iou"],
        cable["recall"],
    )
    if any(value is None for value in values):
        raise ValueError("Development ranking requires defined global and cable metrics")
    return float(0.55 * values[0] + 0.30 * values[1] + 0.15 * values[2])


def select_development_candidate(
    records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Choose a displayed developer leader; this is not a blind-test selector."""
    candidates = list(records)
    if not candidates:
        raise ValueError("Development ranking requires at least one threshold record")
    return max(
        candidates,
        key=lambda record: (
            float(record["developer_score"]),
            float(record["metrics"]["per_class"]["stay_cable"]["iou"]),
            -int(record["metrics"]["background_to_stay_cable_pixels"]),
            float(record["stay_cable_threshold"]),
        ),
    )
