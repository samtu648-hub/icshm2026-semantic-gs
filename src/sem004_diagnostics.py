"""Pure, read-only diagnostics used by SEM-004-DIAG.

The helpers in this module intentionally have no trainable tensors, optimizer,
or checkpoint writer.  They turn a frozen semantic ID map and an official
LabelMe-derived ID map into deterministic error-region and geometry summaries.
All thresholds are supplied by the experiment configuration so validation is
only ever an application of training-side-frozen rules.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import cv2
import numpy as np


BACKGROUND = 0
STAY_CABLE = 2
CLASS_COUNT = 5


def confusion(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Return the fixed SEM-001 target-row/prediction-column confusion matrix."""
    if target.shape != prediction.shape or target.ndim != 2:
        raise ValueError("target and prediction must be matching 2-D label maps")
    return np.bincount(
        CLASS_COUNT * target.reshape(-1).astype(np.int64) + prediction.reshape(-1).astype(np.int64),
        minlength=CLASS_COUNT**2,
    ).reshape(CLASS_COUNT, CLASS_COUNT)


def cable_distance(target: np.ndarray) -> np.ndarray:
    """Euclidean pixel distance to the true stay_cable region."""
    cable = target == STAY_CABLE
    # OpenCV distanceTransform assigns zero to zero-valued pixels.  Inverting
    # the cable mask therefore yields distance-to-cable for every non-cable
    # pixel and zero on cable itself.
    return cv2.distanceTransform((~cable).astype(np.uint8), cv2.DIST_L2, 3)


def _component_rows(mask: np.ndarray, distance: np.ndarray | None = None) -> list[dict[str, float | int]]:
    """Connected-component rows with area, elongation and compactness."""
    labels, _component_ids, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    rows: list[dict[str, float | int]] = []
    for component_id in range(1, labels):
        x, y, width, height, area = (int(value) for value in stats[component_id])
        component = ( _component_ids == component_id ).astype(np.uint8)
        contours, _hierarchy = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
        compactness = float((4.0 * math.pi * area) / (perimeter * perimeter)) if perimeter > 0.0 else 0.0
        row: dict[str, float | int] = {
            "component_id": component_id,
            "area_px": area,
            "bbox_x": x,
            "bbox_y": y,
            "bbox_width": width,
            "bbox_height": height,
            "bbox_aspect_ratio": float(max(width, height) / max(min(width, height), 1)),
            "perimeter_px": perimeter,
            "compactness": compactness,
            "thinness": float(1.0 - compactness),
        }
        if distance is not None:
            values = distance[_component_ids == component_id]
            row["distance_to_gt_cable_min_px"] = float(values.min()) if values.size else None
            row["distance_to_gt_cable_mean_px"] = float(values.mean()) if values.size else None
        rows.append(row)
    return rows


def component_distribution(mask: np.ndarray, distance: np.ndarray | None = None) -> dict[str, Any]:
    rows = _component_rows(mask, distance)
    areas = np.asarray([row["area_px"] for row in rows], dtype=np.float64)
    elongations = np.asarray([row["bbox_aspect_ratio"] for row in rows], dtype=np.float64)
    thinness = np.asarray([row["thinness"] for row in rows], dtype=np.float64)
    return {
        "count": len(rows),
        "area_px": _distribution(areas),
        "bbox_aspect_ratio": _distribution(elongations),
        "thinness": _distribution(thinness),
        "rows": rows,
    }


def _distribution(values: np.ndarray) -> dict[str, float | int | None]:
    if values.size == 0:
        return {"count": 0, "sum": 0.0, "mean": None, "median": None, "p90": None, "max": None}
    return {
        "count": int(values.size),
        "sum": float(values.sum()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "max": float(values.max()),
    }


def error_structure(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    near_boundary_px: float,
    small_component_area_px: int,
    large_component_area_px: int,
) -> dict[str, Any]:
    """Summarize the frozen background/stay_cable error geometry for one view."""
    if near_boundary_px < 0 or small_component_area_px < 1 or large_component_area_px <= small_component_area_px:
        raise ValueError("SEM-004 error-region rules are invalid")
    distance = cable_distance(target)
    false_positive = (target == BACKGROUND) & (prediction == STAY_CABLE)
    false_negative = (target == STAY_CABLE) & (prediction == BACKGROUND)
    predicted_cable = prediction == STAY_CABLE
    true_cable = target == STAY_CABLE
    fp_distance = distance[false_positive]
    fp_components = _component_rows(false_positive, distance)
    small = [row for row in fp_components if int(row["area_px"]) <= small_component_area_px]
    large = [row for row in fp_components if int(row["area_px"]) >= large_component_area_px]
    near_pixels = int((false_positive & (distance <= near_boundary_px)).sum())
    remote_pixels = int((false_positive & (distance > near_boundary_px)).sum())
    return {
        "background_to_stay_cable_pixels": int(false_positive.sum()),
        "stay_cable_to_background_pixels": int(false_negative.sum()),
        "false_positive_distance_to_gt_stay_cable_px": _distribution(fp_distance.astype(np.float64, copy=False)),
        "false_positive_categories": {
            "near_gt_stay_cable_boundary_expansion": {
                "definition": f"GT background predicted stay_cable at distance <= {near_boundary_px:g} px from GT stay_cable",
                "pixels": near_pixels,
            },
            "spatially_remote_from_gt_stay_cable": {
                "definition": f"GT background predicted stay_cable at distance > {near_boundary_px:g} px from GT stay_cable",
                "pixels": remote_pixels,
            },
            "small_fragments": {
                "definition": f"8-connected false-positive component area <= {small_component_area_px} px; may overlap the distance categories",
                "components": len(small),
                "pixels": int(sum(int(row["area_px"]) for row in small)),
            },
            "large_area_connected_expansions": {
                "definition": f"8-connected false-positive component area >= {large_component_area_px} px; may overlap the distance categories",
                "components": len(large),
                "pixels": int(sum(int(row["area_px"]) for row in large)),
            },
        },
        "prediction_stay_cable_components": component_distribution(predicted_cable, distance),
        "ground_truth_stay_cable_components": component_distribution(true_cable, distance),
        "false_positive_components": {"count": len(fp_components), "rows": fp_components},
    }


def semantic_triptych(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Return a GT / prediction / error RGB visualization without source RGB."""
    palette = np.asarray(
        [[0, 0, 0], [230, 86, 37], [255, 210, 0], [38, 139, 210], [133, 153, 0]], dtype=np.uint8
    )
    gt = palette[target]
    pred = palette[prediction]
    error = np.zeros_like(gt)
    correct = target == prediction
    error[correct] = np.asarray([28, 28, 28], dtype=np.uint8)
    error[(target == BACKGROUND) & (prediction == STAY_CABLE)] = np.asarray([235, 65, 54], dtype=np.uint8)
    error[(target == STAY_CABLE) & (prediction == BACKGROUND)] = np.asarray([66, 133, 244], dtype=np.uint8)
    other = ~correct & ~((target == BACKGROUND) & (prediction == STAY_CABLE)) & ~((target == STAY_CABLE) & (prediction == BACKGROUND))
    error[other] = np.asarray([178, 102, 255], dtype=np.uint8)
    return np.concatenate((gt, pred, error), axis=1)


def add_counter(target: Counter[str], value: Counter[str]) -> None:
    """Update a Counter while retaining only JSON-compatible string keys."""
    target.update({str(key): int(count) for key, count in value.items()})
