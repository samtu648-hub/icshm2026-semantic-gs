"""Pinhole epipolar geometry and robust multi-view triangulation for Cable3D."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from geometry_prior.camera_io import ColmapImage


@dataclass(frozen=True)
class CableObservation:
    image_name: str
    pixel_xy: np.ndarray
    width_px: float
    image: ColmapImage
    source: str


@dataclass(frozen=True)
class TriangulatedCandidate:
    xyz: np.ndarray
    observations: tuple[CableObservation, ...]
    reprojection_errors_px: np.ndarray
    minimum_baseline: float
    maximum_baseline: float
    minimum_triangulation_angle_deg: float
    confidence: float


def pinhole_matrix(focal: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def projection_matrix(image: ColmapImage, intrinsic: np.ndarray) -> np.ndarray:
    extrinsic = np.column_stack([image.rotation_world_to_camera, image.tvec])
    return intrinsic @ extrinsic


def pixel_rays_world(pixels_xy: np.ndarray, intrinsic: np.ndarray, image: ColmapImage) -> tuple[np.ndarray, np.ndarray]:
    pixels = np.asarray(pixels_xy, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack([pixels, np.ones(len(pixels))])
    camera_rays = homogeneous @ np.linalg.inv(intrinsic).T
    world_rays = camera_rays @ image.rotation_camera_to_world.T
    world_rays /= np.linalg.norm(world_rays, axis=1, keepdims=True)
    origins = np.repeat(image.center[None], len(pixels), axis=0)
    return origins, world_rays


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def fundamental_matrix(left: ColmapImage, right: ColmapImage, intrinsic: np.ndarray) -> np.ndarray:
    rotation = right.rotation_world_to_camera @ left.rotation_camera_to_world
    translation = right.tvec - rotation @ left.tvec
    essential = _skew(translation) @ rotation
    inverse = np.linalg.inv(intrinsic)
    fundamental = inverse.T @ essential @ inverse
    return fundamental / max(float(np.linalg.norm(fundamental)), np.finfo(float).eps)


def symmetric_epipolar_distance(left_xy: np.ndarray, right_xy: np.ndarray, fundamental: np.ndarray) -> np.ndarray:
    left = np.column_stack([np.asarray(left_xy, dtype=np.float64).reshape(-1, 2), np.ones(len(np.asarray(left_xy).reshape(-1, 2)))])
    right = np.column_stack([np.asarray(right_xy, dtype=np.float64).reshape(-1, 2), np.ones(len(np.asarray(right_xy).reshape(-1, 2)))])
    if len(left) != len(right):
        raise ValueError("left_xy and right_xy need equal length")
    right_lines = left @ fundamental.T
    left_lines = right @ fundamental
    numerator = np.abs(np.sum(right * right_lines, axis=1))
    right_denominator = np.linalg.norm(right_lines[:, :2], axis=1)
    left_denominator = np.linalg.norm(left_lines[:, :2], axis=1)
    return 0.5 * numerator * (
        1.0 / np.maximum(right_denominator, np.finfo(float).eps)
        + 1.0 / np.maximum(left_denominator, np.finfo(float).eps)
    )


def triangulate_dlt(observations: list[CableObservation], intrinsic: np.ndarray) -> np.ndarray:
    if len(observations) < 2:
        raise ValueError("At least two observations are required")
    rows = []
    for observation in observations:
        projection = projection_matrix(observation.image, intrinsic)
        x, y = np.asarray(observation.pixel_xy, dtype=np.float64)
        rows.extend([x * projection[2] - projection[0], y * projection[2] - projection[1]])
    _, _, vh = np.linalg.svd(np.asarray(rows), full_matrices=False)
    homogeneous = vh[-1]
    if abs(float(homogeneous[3])) < 1e-12:
        raise ValueError("Degenerate homogeneous triangulation")
    point = homogeneous[:3] / homogeneous[3]
    if not np.isfinite(point).all():
        raise ValueError("Triangulation produced non-finite coordinates")
    return point


def reproject(point_xyz: np.ndarray, image: ColmapImage, intrinsic: np.ndarray) -> tuple[np.ndarray, float]:
    camera = image.rotation_world_to_camera @ np.asarray(point_xyz, dtype=np.float64) + image.tvec
    depth = float(camera[2])
    if abs(depth) < np.finfo(float).eps:
        return np.array([np.inf, np.inf]), depth
    pixel = intrinsic @ camera
    return pixel[:2] / pixel[2], depth


def triangulation_angle_deg(point_xyz: np.ndarray, left: ColmapImage, right: ColmapImage) -> float:
    a = np.asarray(point_xyz, dtype=np.float64) - left.center
    b = np.asarray(point_xyz, dtype=np.float64) - right.center
    a /= max(float(np.linalg.norm(a)), np.finfo(float).eps)
    b /= max(float(np.linalg.norm(b)), np.finfo(float).eps)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def _independent_indices(observations: list[CableObservation], errors: np.ndarray, minimum_baseline: float) -> list[int]:
    ordered = sorted(range(len(observations)), key=lambda index: (float(errors[index]), observations[index].image_name))
    selected: list[int] = []
    for index in ordered:
        center = observations[index].image.center
        if all(float(np.linalg.norm(center - observations[other].image.center)) >= minimum_baseline for other in selected):
            selected.append(index)
    return selected


def robust_triangulate(
    observations: list[CableObservation],
    intrinsic: np.ndarray,
    minimum_baseline: float,
    maximum_baseline: float,
    minimum_angle_deg: float,
    reprojection_threshold_px: float,
    minimum_support: int,
) -> TriangulatedCandidate | None:
    unique: dict[str, CableObservation] = {}
    for observation in observations:
        unique.setdefault(observation.image_name, observation)
    observations = [unique[name] for name in sorted(unique)]
    if len(observations) < minimum_support:
        return None
    best: tuple[int, float, float, np.ndarray, np.ndarray] | None = None
    for left_index, right_index in combinations(range(len(observations)), 2):
        left, right = observations[left_index], observations[right_index]
        baseline = float(np.linalg.norm(left.image.center - right.image.center))
        if baseline < minimum_baseline or baseline > maximum_baseline:
            continue
        try:
            point = triangulate_dlt([left, right], intrinsic)
        except ValueError:
            continue
        angle = triangulation_angle_deg(point, left.image, right.image)
        if angle < minimum_angle_deg:
            continue
        errors, positive = [], []
        for observation in observations:
            pixel, depth = reproject(point, observation.image, intrinsic)
            errors.append(float(np.linalg.norm(pixel - observation.pixel_xy)))
            positive.append(depth > 0)
        errors_array = np.asarray(errors)
        inliers = (errors_array <= reprojection_threshold_px) & np.asarray(positive)
        independent = _independent_indices(
            [observation for observation, keep in zip(observations, inliers) if keep],
            errors_array[inliers],
            minimum_baseline,
        )
        support = len(independent)
        score = (support, -float(np.median(errors_array[inliers])) if np.any(inliers) else -np.inf, angle)
        if support >= minimum_support and (best is None or score > best[:3]):
            best = (*score, point, inliers)
    if best is None:
        return None
    inlier_observations = [observation for observation, keep in zip(observations, best[4]) if keep]
    try:
        point = triangulate_dlt(inlier_observations, intrinsic)
    except ValueError:
        return None
    errors, positive = [], []
    for observation in observations:
        pixel, depth = reproject(point, observation.image, intrinsic)
        errors.append(float(np.linalg.norm(pixel - observation.pixel_xy)))
        positive.append(depth > 0)
    errors_array = np.asarray(errors)
    inliers = (errors_array <= reprojection_threshold_px) & np.asarray(positive)
    selected_local = _independent_indices(
        [observation for observation, keep in zip(observations, inliers) if keep],
        errors_array[inliers],
        minimum_baseline,
    )
    filtered = [observation for observation, keep in zip(observations, inliers) if keep]
    if len(selected_local) < minimum_support:
        return None
    retained = [filtered[index] for index in selected_local]
    retained_errors = np.asarray(
        [float(np.linalg.norm(reproject(point, observation.image, intrinsic)[0] - observation.pixel_xy)) for observation in retained]
    )
    baselines = [float(np.linalg.norm(a.image.center - b.image.center)) for a, b in combinations(retained, 2)]
    angles = [triangulation_angle_deg(point, a.image, b.image) for a, b in combinations(retained, 2)]
    confidence = float(len(retained) / (1.0 + np.median(retained_errors)))
    return TriangulatedCandidate(
        xyz=point,
        observations=tuple(retained),
        reprojection_errors_px=retained_errors,
        minimum_baseline=min(baselines),
        maximum_baseline=max(baselines),
        minimum_triangulation_angle_deg=min(angles),
        confidence=confidence,
    )
