"""Mutually exclusive bridge-support defect classification helpers."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import numpy as np


DEFECT_UNSUPPORTED = "unsupported_projection"
DEFECT_MIXED = "existing_mixed_or_wrong_geometry"
DEFECT_CLEAN = "supported_clean"


def classify_view_structure(
    row: dict[str, Any],
    *,
    minimum_projected_coverage: float = 0.5,
    maximum_nonclean_fraction: float = 0.25,
) -> str:
    """Classify one registered view/class without overlapping defect labels.

    ``top_ellipse_coverage_fraction`` measures whether any leading Gaussian
    contributors project over the localized structure.  Once that support is
    present, the gap to ``true_like_ellipse_coverage_fraction`` measures how
    much of it is supplied only by mixed/background contributors.
    """
    total = float(row["top_ellipse_coverage_fraction"])
    clean = float(row["true_like_ellipse_coverage_fraction"])
    if total < minimum_projected_coverage:
        return DEFECT_UNSUPPORTED
    nonclean_fraction = max(total - clean, 0.0) / max(total, 1e-12)
    if nonclean_fraction > maximum_nonclean_fraction:
        return DEFECT_MIXED
    return DEFECT_CLEAN


def aggregate_gaussian_records(
    records: Iterable[dict[str, Any]],
    *,
    minimum_views: int = 3,
    minimum_mixed_fraction: float = 0.5,
    minimum_background_fraction: float = 0.35,
) -> list[dict[str, Any]]:
    """Aggregate final-model contributors into a reviewable wrong-support list."""
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in records:
        if not row.get("top_structure") or row.get("class") not in {"deck", "stay_cable", "tower"}:
            continue
        grouped.setdefault((str(row["class"]), int(row["gaussian_id"])), []).append(row)

    output: list[dict[str, Any]] = []
    for (class_name, gaussian_id), rows in sorted(grouped.items()):
        unique_views = sorted({str(row["view"]) for row in rows})
        if len(unique_views) < minimum_views:
            continue
        kinds = Counter(str(row["kind"]) for row in rows)
        mixed_count = kinds["artifact_like"] + kinds["ambiguous"]
        mixed_fraction = mixed_count / max(len(rows), 1)
        background = np.asarray([float(row["near_background_fraction"]) for row in rows], dtype=np.float64)
        orientation = np.asarray(
            [float(row["orientation_error_deg"]) for row in rows if row.get("orientation_error_deg") is not None],
            dtype=np.float64,
        )
        if mixed_fraction < minimum_mixed_fraction or float(np.median(background)) < minimum_background_fraction:
            continue
        output.append({
            "class": class_name,
            "gaussian_id": gaussian_id,
            "defect": DEFECT_MIXED,
            "views": unique_views,
            "observation_count": len(rows),
            "mixed_fraction": float(mixed_fraction),
            "near_background_fraction_p50": float(np.median(background)),
            "orientation_error_deg_p50": float(np.median(orientation)) if len(orientation) else None,
            "major_radius_px_p50": float(np.median([row["major_radius_px"] for row in rows])),
            "minor_radius_px_p50": float(np.median([row["minor_radius_px"] for row in rows])),
            "opacity_p50": float(np.median([row["opacity"] for row in rows])),
        })
    return output

