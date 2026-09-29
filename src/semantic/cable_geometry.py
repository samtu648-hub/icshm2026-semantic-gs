"""Deterministic stay_cable mask centerlines, 3D segments, and check metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage


@dataclass(frozen=True)
class CenterlineObservation:
    mask: np.ndarray
    skeleton: np.ndarray
    sample_pixels_xy: np.ndarray
    sample_widths_px: np.ndarray
    endpoints_xy: np.ndarray
    branches_xy: np.ndarray
    component_count: int
    skeleton_component_count: int


@dataclass(frozen=True)
class CableCurve:
    start: np.ndarray
    end: np.ndarray
    tangent: np.ndarray
    length: float
    inlier_indices: np.ndarray
    confidence: float
    support_views: tuple[str, ...]
    local_width_px: float

    def sample(self, spacing: float) -> np.ndarray:
        count = max(2, int(np.ceil(self.length / spacing)) + 1)
        return np.linspace(self.start, self.end, count, dtype=np.float64)


def _remove_small_components(binary: np.ndarray, minimum_area: int, connectivity: int) -> tuple[np.ndarray, int]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=connectivity)
    output = np.zeros_like(binary, dtype=bool)
    retained = 0
    for component in range(1, count):
        if int(stats[component, cv2.CC_STAT_AREA]) >= minimum_area:
            output[labels == component] = True
            retained += 1
    return output, retained


def _sample_skeleton(skeleton: np.ndarray, widths: np.ndarray, spacing: int, maximum: int) -> tuple[np.ndarray, np.ndarray]:
    yx = np.argwhere(skeleton)
    if not len(yx):
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)
    cells: dict[tuple[int, int], tuple[float, int, int]] = {}
    half = (spacing - 1) / 2.0
    for y, x in yx:
        cell = (int(y) // spacing, int(x) // spacing)
        cy, cx = cell[0] * spacing + half, cell[1] * spacing + half
        score = float((y - cy) ** 2 + (x - cx) ** 2)
        candidate = (score, int(y), int(x))
        if cell not in cells or candidate < cells[cell]:
            cells[cell] = candidate
    selected = sorted((value[1], value[2]) for value in cells.values())
    if len(selected) > maximum:
        indices = np.linspace(0, len(selected) - 1, maximum, dtype=np.int64)
        selected = [selected[int(index)] for index in indices]
    pixels = np.asarray([(x, y) for y, x in selected], dtype=np.float64)
    local_widths = np.asarray([widths[y, x] for y, x in selected], dtype=np.float64)
    return pixels, local_widths


def extract_centerline(label_path: Path, class_id: int, config: dict) -> CenterlineObservation:
    label = np.asarray(Image.open(label_path).convert("L"), dtype=np.uint8)
    mask = label == int(class_id)
    close_size = int(config["close_kernel_px"])
    if close_size:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=int(config["close_iterations"])) > 0
    open_size = int(config["open_kernel_px"])
    if open_size:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=int(config["open_iterations"])) > 0
    mask, component_count = _remove_small_components(
        mask, int(config["minimum_component_area_px"]), int(config["connectivity"])
    )
    skeleton_u8 = cv2.ximgproc.thinning(mask.astype(np.uint8) * 255, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    skeleton, skeleton_component_count = _remove_small_components(
        skeleton_u8 > 0, int(config["minimum_skeleton_component_px"]), int(config["connectivity"])
    )
    distance_inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    widths = 2.0 * distance_inside
    neighbor_kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_kernel[1, 1] = 0
    neighbors = cv2.filter2D(skeleton.astype(np.uint8), cv2.CV_16U, neighbor_kernel, borderType=cv2.BORDER_CONSTANT)
    endpoints_yx = np.argwhere(skeleton & (neighbors == int(config["endpoint_neighbor_count"])))
    branches_yx = np.argwhere(skeleton & (neighbors >= int(config["branch_neighbor_count_minimum"])))
    samples, sample_widths = _sample_skeleton(
        skeleton,
        widths,
        int(config["sample_grid_spacing_px"]),
        int(config["maximum_samples_per_image"]),
    )
    return CenterlineObservation(
        mask=mask,
        skeleton=skeleton,
        sample_pixels_xy=samples,
        sample_widths_px=sample_widths,
        endpoints_xy=np.asarray([(x, y) for y, x in endpoints_yx], dtype=np.float64).reshape(-1, 2),
        branches_xy=np.asarray([(x, y) for y, x in branches_yx], dtype=np.float64).reshape(-1, 2),
        component_count=component_count,
        skeleton_component_count=skeleton_component_count,
    )


def nearest_skeleton(skeleton: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return distance and nearest-skeleton coordinates for every image pixel."""
    if not np.any(skeleton):
        shape = skeleton.shape
        return np.full(shape, np.inf, dtype=np.float32), np.full((2, *shape), -1, dtype=np.int32)
    distance, indices = ndimage.distance_transform_edt(~skeleton, return_indices=True)
    return distance.astype(np.float32), indices.astype(np.int32)


def voxel_representatives(points: np.ndarray, scores: np.ndarray, voxel_size: float, maximum: int) -> np.ndarray:
    if not len(points):
        return np.empty((0,), dtype=np.int64)
    cells = np.floor(np.asarray(points, dtype=np.float64) / voxel_size).astype(np.int64)
    order = np.lexsort((np.arange(len(points)), -np.asarray(scores, dtype=np.float64)))
    retained: dict[tuple[int, int, int], int] = {}
    for index in order:
        key = tuple(int(value) for value in cells[index])
        retained.setdefault(key, int(index))
    selected = sorted(retained.values(), key=lambda index: (-float(scores[index]), index))
    return np.asarray(selected[:maximum], dtype=np.int64)


def _line_distances(points: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    relative = points - origin
    along = relative @ direction
    perpendicular = relative - along[:, None] * direction[None]
    return np.linalg.norm(perpendicular, axis=1), along


def fit_piecewise_lines(
    points: np.ndarray,
    confidences: np.ndarray,
    support_views: list[tuple[str, ...]],
    widths_px: np.ndarray,
    config: dict,
    seed: int,
) -> list[CableCurve]:
    points = np.asarray(points, dtype=np.float64)
    active = np.arange(len(points), dtype=np.int64)
    rng = np.random.default_rng(seed)
    curves: list[CableCurve] = []
    radius = float(config["line_inlier_radius_official"])
    minimum = int(config["minimum_inlier_candidates"])
    for _ in range(int(config["maximum_curves"])):
        if len(active) < minimum:
            break
        best: tuple[float, np.ndarray] | None = None
        iterations = min(int(config["ransac_iterations_per_curve"]), max(1, len(active) * (len(active) - 1) // 2))
        for _attempt in range(iterations):
            pair = rng.choice(active, size=2, replace=False)
            delta = points[pair[1]] - points[pair[0]]
            norm = float(np.linalg.norm(delta))
            if norm < float(config["minimum_curve_length_official"]) * 0.25:
                continue
            distances, _ = _line_distances(points[active], points[pair[0]], delta / norm)
            inliers = active[distances <= radius]
            if len(inliers) < minimum:
                continue
            score = float(np.sum(confidences[inliers])) + 0.01 * len(inliers)
            if best is None or score > best[0]:
                best = (score, inliers)
        if best is None:
            break
        inliers = best[1]
        center = np.average(points[inliers], axis=0, weights=np.maximum(confidences[inliers], 1e-6))
        centered = points[inliers] - center
        _, _, vh = np.linalg.svd(centered * np.sqrt(np.maximum(confidences[inliers], 1e-6))[:, None], full_matrices=False)
        direction = vh[0]
        distances, along = _line_distances(points[inliers], center, direction)
        inliers = inliers[distances <= radius]
        along = along[distances <= radius]
        order = np.argsort(along)
        ordered_indices, ordered_along = inliers[order], along[order]
        gaps = np.diff(ordered_along)
        boundaries = np.flatnonzero(gaps > float(config["maximum_along_curve_gap_official"])) + 1
        runs = np.split(np.arange(len(ordered_indices)), boundaries)
        run = max(runs, key=lambda values: (len(values), float(np.sum(confidences[ordered_indices[values]]))))
        inliers = ordered_indices[run]
        if len(inliers) < minimum:
            active = active[~np.isin(active, best[1])]
            continue
        center = np.average(points[inliers], axis=0, weights=np.maximum(confidences[inliers], 1e-6))
        centered = points[inliers] - center
        _, _, vh = np.linalg.svd(centered * np.sqrt(np.maximum(confidences[inliers], 1e-6))[:, None], full_matrices=False)
        direction = vh[0]
        _, along = _line_distances(points[inliers], center, direction)
        low, high = np.quantile(along, np.asarray(config["endpoint_quantiles"], dtype=np.float64))
        length = float(high - low)
        if length < float(config["minimum_curve_length_official"]):
            active = active[~np.isin(active, best[1])]
            continue
        bin_width = max(2.0 * radius, np.finfo(float).eps)
        bin_count = max(1, int(np.ceil(length / bin_width)))
        occupied = len(np.unique(np.clip(((along - low) / bin_width).astype(int), 0, bin_count - 1)))
        if occupied / bin_count < float(config["minimum_occupied_bin_fraction"]):
            active = active[~np.isin(active, best[1])]
            continue
        start, end = center + low * direction, center + high * direction
        if tuple(start.tolist()) > tuple(end.tolist()):
            start, end, direction = end, start, -direction
        views = tuple(sorted(set().union(*(set(support_views[index]) for index in inliers))))
        curves.append(
            CableCurve(
                start=start,
                end=end,
                tangent=direction / np.linalg.norm(direction),
                length=length,
                inlier_indices=np.sort(inliers),
                confidence=float(np.mean(confidences[inliers])),
                support_views=views,
                local_width_px=float(np.median(widths_px[inliers])),
            )
        )
        active = active[~np.isin(active, inliers)]
    return curves


def rasterize_projected_curves(
    curves: Iterable[CableCurve],
    project,
    width: int,
    height: int,
    spacing: float,
) -> np.ndarray:
    canvas = np.zeros((height, width), dtype=np.uint8)
    for curve in curves:
        samples = curve.sample(spacing)
        pixels, depth = project(samples)
        valid = (
            (depth > 0)
            & np.isfinite(pixels).all(axis=1)
            & (pixels[:, 0] >= 0)
            & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < height)
        )
        rounded = np.rint(pixels).astype(np.int32)
        for index in range(len(samples) - 1):
            if valid[index] and valid[index + 1]:
                cv2.line(canvas, tuple(rounded[index]), tuple(rounded[index + 1]), 255, 1, cv2.LINE_8)
    return canvas > 0


def centerline_metrics(prediction: np.ndarray, target: np.ndarray, tolerances: Iterable[int], coverage_recall: float) -> dict:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    tp = int(np.count_nonzero(prediction & target))
    pred_count, target_count = int(np.count_nonzero(prediction)), int(np.count_nonzero(target))
    exact_precision = tp / pred_count if pred_count else 0.0
    exact_recall = tp / target_count if target_count else 0.0
    union = int(np.count_nonzero(prediction | target))
    result: dict[str, object] = {
        "predicted_pixels": pred_count,
        "target_pixels": target_count,
        "exact_precision": exact_precision,
        "exact_recall": exact_recall,
        "exact_iou": tp / union if union else 0.0,
    }
    pred_distance = ndimage.distance_transform_edt(~prediction) if pred_count else np.full(prediction.shape, np.inf)
    target_distance = ndimage.distance_transform_edt(~target) if target_count else np.full(target.shape, np.inf)
    for tolerance in tolerances:
        precision = float(np.count_nonzero(prediction & (target_distance <= tolerance)) / pred_count) if pred_count else 0.0
        recall = float(np.count_nonzero(target & (pred_distance <= tolerance)) / target_count) if target_count else 0.0
        result[f"precision_at_{tolerance}px"] = precision
        result[f"recall_at_{tolerance}px"] = recall
    prediction_errors = target_distance[prediction]
    result["prediction_error_median_px"] = float(np.median(prediction_errors)) if len(prediction_errors) else float("inf")
    result["prediction_error_p90_px"] = float(np.quantile(prediction_errors, 0.9)) if len(prediction_errors) else float("inf")
    result["covered_at_2px"] = bool(result.get("recall_at_2px", 0.0) >= coverage_recall)
    return result


def write_case_overlay(target: np.ndarray, prediction: np.ndarray, path: Path) -> None:
    image = np.zeros((*target.shape, 3), dtype=np.uint8)
    image[target] = (0, 180, 0)
    image[prediction] = (220, 0, 0)
    image[target & prediction] = (255, 255, 255)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(path)
