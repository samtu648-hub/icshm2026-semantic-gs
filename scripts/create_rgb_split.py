#!/usr/bin/env python3
"""Create a deterministic, guarded spatial split from COLMAP camera centers."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class CameraRecord:
    """A COLMAP image record expressed in the official world coordinate system."""

    name: str
    image_id: int
    center: tuple[float, float, float]


def natural_key(name: str) -> tuple[int, int | str, str]:
    stem = Path(name).stem
    if stem.isdigit():
        return (0, int(stem), name)
    return (1, stem, name)


def quaternion_to_rotation(
    qw: float, qx: float, qy: float, qz: float
) -> tuple[tuple[float, float, float], ...]:
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm == 0:
        raise ValueError("COLMAP quaternion has zero norm")
    qw, qx, qy, qz = (value / norm for value in (qw, qx, qy, qz))
    return (
        (1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)),
        (2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)),
        (2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)),
    )


def camera_center_from_world_to_camera(
    rotation: tuple[tuple[float, float, float], ...], translation: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Return C = -R^T t for COLMAP's CAMERA_FROM_WORLD convention."""
    return tuple(
        -sum(rotation[row][column] * translation[row] for row in range(3))
        for column in range(3)
    )


def parse_colmap_images(path: Path) -> list[CameraRecord]:
    records: list[CameraRecord] = []
    names: set[str] = set()
    ids: set[int] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 10 or Path(fields[9]).suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            image_id = int(fields[0])
            quaternion = tuple(float(value) for value in fields[1:5])
            translation = tuple(float(value) for value in fields[5:8])
            int(fields[8])
        except ValueError as exc:
            raise ValueError(f"Malformed COLMAP image pose line: {line[:160]}") from exc
        name = Path(fields[9]).name
        if name in names or image_id in ids:
            raise ValueError(f"Duplicate COLMAP image record: {name}")
        rotation = quaternion_to_rotation(*quaternion)
        records.append(
            CameraRecord(
                name=name,
                image_id=image_id,
                center=camera_center_from_world_to_camera(rotation, translation),
            )
        )
        names.add(name)
        ids.add(image_id)
    if not records:
        raise ValueError(f"No COLMAP image records found in {path}")
    return sorted(records, key=lambda record: natural_key(record.name))


def list_image_names(directory: Path) -> set[str]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    return {
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def squared_distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right))


def kmeans_labels(records: list[CameraRecord], clusters: int, seed: int) -> list[int]:
    if clusters < 2 or clusters > len(records):
        raise ValueError(f"clusters must be in [2, {len(records)}], got {clusters}")

    raw_points = [record.center for record in records]
    means = tuple(statistics.fmean(point[axis] for point in raw_points) for axis in range(3))
    scales = tuple(
        statistics.pstdev(point[axis] for point in raw_points) or 1.0 for axis in range(3)
    )
    points = [tuple((point[axis] - means[axis]) / scales[axis] for axis in range(3)) for point in raw_points]
    rng = random.Random(seed)
    centroids = [points[index] for index in rng.sample(range(len(points)), clusters)]
    labels = [-1] * len(points)

    for _ in range(200):
        next_labels = [
            min(range(clusters), key=lambda group: (squared_distance(point, centroids[group]), group))
            for point in points
        ]
        groups = [[index for index, label in enumerate(next_labels) if label == group] for group in range(clusters)]
        if any(not group for group in groups):
            assigned_distances = [
                squared_distance(points[index], centroids[next_labels[index]])
                for index in range(len(points))
            ]
            used: set[int] = set()
            for group, members in enumerate(groups):
                if members:
                    continue
                replacement = max(
                    (index for index in range(len(points)) if index not in used),
                    key=lambda index: (assigned_distances[index], records[index].name),
                )
                centroids[group] = points[replacement]
                used.add(replacement)
            continue
        next_centroids = [
            tuple(statistics.fmean(points[index][axis] for index in members) for axis in range(3))
            for members in groups
        ]
        if next_labels == labels and next_centroids == centroids:
            return next_labels
        labels = next_labels
        centroids = next_centroids
    raise RuntimeError("k-means did not converge within 200 iterations")


def nearest_neighbor_distances(records: list[CameraRecord]) -> list[float]:
    if len(records) < 2:
        raise ValueError("At least two camera records are required")
    return [
        min(math.dist(record.center, other.center) for other in records if other != record)
        for record in records
    ]


def guarded_fold(
    records: list[CameraRecord],
    holdout_names: set[str],
    labeled_names: set[str],
    unlabeled_names: set[str],
    guard_radius: float,
) -> dict[str, object]:
    holdout = [record for record in records if record.name in holdout_names]
    remaining = [record for record in records if record.name not in holdout_names]
    buffer_names = {
        record.name
        for record in remaining
        if min(math.dist(record.center, held_out.center) for held_out in holdout) <= guard_radius
    }
    train_names = {record.name for record in remaining if record.name not in buffer_names}
    validation_labeled = holdout_names & labeled_names
    buffer_labeled = buffer_names & labeled_names
    buffer_unlabeled = buffer_names & unlabeled_names
    minimum_train_distance = (
        min(
            math.dist(record.center, held_out.center)
            for record in records
            if record.name in train_names
            for held_out in holdout
        )
        if train_names
        else None
    )
    overlap = (train_names & holdout_names) | (train_names & buffer_names) | (holdout_names & buffer_names)
    if overlap or not validation_labeled or not train_names:
        raise ValueError("Spatial split is invalid: overlapping or empty train/validation partition")
    return {
        "holdout_all_cameras": sorted(holdout_names, key=natural_key),
        "validation_labeled": sorted(validation_labeled, key=natural_key),
        "withheld_unlabeled": sorted(holdout_names & unlabeled_names, key=natural_key),
        "buffer_excluded_all_cameras": sorted(buffer_names, key=natural_key),
        "buffer_excluded_labeled": sorted(buffer_labeled, key=natural_key),
        "buffer_excluded_unlabeled": sorted(buffer_unlabeled, key=natural_key),
        "train_rgb_400": sorted(train_names, key=natural_key),
        "train_rgb_300": sorted(train_names & labeled_names, key=natural_key),
        "checks": {
            "no_filename_overlap": not overlap,
            "no_train_camera_within_guard_radius": (
                minimum_train_distance is None or minimum_train_distance > guard_radius
            ),
            "minimum_train_to_holdout_distance": minimum_train_distance,
        },
    }


def create_split(
    records: list[CameraRecord],
    labeled_names: set[str],
    unlabeled_names: set[str],
    clusters: int,
    seed: int,
    guard_multiplier: float,
    challenge_views: set[str],
) -> dict[str, object]:
    camera_names = {record.name for record in records}
    released_names = labeled_names | unlabeled_names
    if labeled_names & unlabeled_names:
        raise ValueError("A released RGB image cannot be both labeled and unlabeled")
    if camera_names != released_names:
        missing_camera = sorted(released_names - camera_names, key=natural_key)
        missing_image = sorted(camera_names - released_names, key=natural_key)
        raise ValueError(
            "COLMAP/image filename mismatch: "
            f"missing_camera_records={missing_camera[:5]}, missing_image_files={missing_image[:5]}"
        )
    if not challenge_views <= labeled_names:
        raise ValueError(f"Challenge views are not all labeled images: {sorted(challenge_views - labeled_names)}")
    if guard_multiplier <= 0:
        raise ValueError("guard_multiplier must be positive")

    labels = kmeans_labels(records, clusters=clusters, seed=seed)
    raw_groups = {
        group: {record.name for record, label in zip(records, labels) if label == group}
        for group in range(clusters)
    }
    ordered_groups = sorted(
        raw_groups.values(),
        key=lambda names: tuple(
            statistics.fmean(record.center[axis] for record in records if record.name in names)
            for axis in range(3)
        ),
    )
    nearest = nearest_neighbor_distances(records)
    guard_radius = statistics.median(nearest) * guard_multiplier
    folds: list[dict[str, object]] = []
    for index, group_names in enumerate(ordered_groups, start=1):
        fold = guarded_fold(
            records,
            holdout_names=group_names,
            labeled_names=labeled_names,
            unlabeled_names=unlabeled_names,
            guard_radius=guard_radius,
        )
        fold["fold_id"] = f"spatial-{index:02d}"
        fold["challenge_views"] = sorted(group_names & challenge_views, key=natural_key)
        folds.append(fold)

    challenge_counts = [len(fold["challenge_views"]) for fold in folds]
    primary_index = max(
        range(len(folds)),
        key=lambda index: (challenge_counts[index], len(folds[index]["validation_labeled"]), -index),
    )
    primary = folds[primary_index]
    if not all(fold["checks"]["no_filename_overlap"] for fold in folds):
        raise ValueError("One or more folds has a filename overlap")
    if not all(fold["checks"]["no_train_camera_within_guard_radius"] for fold in folds):
        raise ValueError("One or more folds violates the guard radius")

    return {
        "schema_version": 1,
        "split_id": "rgb_r1_spatial_v1",
        "purpose": "R1 RGB backbone comparison with guarded spatial validation",
        "source_coordinate_frame": "official_colmap_reference",
        "pose_convention": "CAMERA_FROM_WORLD; centers computed as C = -R^T t",
        "source_counts": {
            "all_released_rgb": len(released_names),
            "labeled_rgb": len(labeled_names),
            "unlabeled_rgb": len(unlabeled_names),
            "camera_records": len(records),
        },
        "grouping": {
            "method": "deterministic k-means over standardized official COLMAP camera centers",
            "clusters": clusters,
            "seed": seed,
            "guard_rule": "Exclude every non-holdout camera at or below guard_radius from training.",
            "guard_radius_colmap_units": guard_radius,
            "guard_multiplier_of_median_global_nearest_neighbor_distance": guard_multiplier,
            "median_global_nearest_neighbor_distance_colmap_units": statistics.median(nearest),
        },
        "challenge_views": sorted(challenge_views, key=natural_key),
        "primary_fold_id": primary["fold_id"],
        "primary_fold": primary,
        "folds": folds,
        "global_checks": {
            "camera_and_released_image_names_match": True,
            "every_camera_is_assigned_to_exactly_one_holdout_fold": sum(
                len(fold["holdout_all_cameras"]) for fold in folds
            )
            == len(records),
            "all_folds_pass_filename_and_guard_checks": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--write", type=Path, required=True, help="Tracked, repository-relative split output.")
    parser.add_argument("--split-id", default="rgb_r1_spatial_v1")
    parser.add_argument("--clusters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--guard-multiplier", type=float, default=3.0)
    parser.add_argument(
        "--challenge-views",
        default="140.png,141.png,142.png,143.png,144.png,145.png,146.png",
        help="Comma-separated labeled views reported as historically difficult.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.dataset_root.resolve()
    records = parse_colmap_images(root / "camera_parameters" / "images.txt")
    payload = create_split(
        records=records,
        labeled_names=list_image_names(root / "images"),
        unlabeled_names=list_image_names(root / "unlabeled_Images"),
        clusters=args.clusters,
        seed=args.seed,
        guard_multiplier=args.guard_multiplier,
        challenge_views={name.strip() for name in args.challenge_views.split(",") if name.strip()},
    )
    payload["split_id"] = args.split_id
    args.write.parent.mkdir(parents=True, exist_ok=True)
    args.write.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    primary = payload["primary_fold"]
    print(f"Wrote spatial split: {args.write}")
    print(
        "Primary fold "
        f"{payload['primary_fold_id']}: "
        f"train_400={len(primary['train_rgb_400'])}, "
        f"train_300={len(primary['train_rgb_300'])}, "
        f"validation_labeled={len(primary['validation_labeled'])}, "
        f"buffer={len(primary['buffer_excluded_all_cameras'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
