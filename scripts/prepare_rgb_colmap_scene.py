#!/usr/bin/env python3
"""Prepare one guarded RGB split as a PINHOLE COLMAP scene without mutating raw data.

The released camera is SIMPLE_RADIAL and deliberately has no points3D.txt. This tool applies the
same full-frame OpenCV undistortion to every selected image, preserves the official world frame, and
retriangulates the existing point tracks in images.txt for 3DGS initialization.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


@dataclass
class Observation:
    x: float
    y: float
    point_id: int
    point_index: int
    undistorted_xy: tuple[float, float] | None = None


@dataclass
class ImageRecord:
    image_id: int
    qw: float
    qx: float
    qy: float
    qz: float
    tx: float
    ty: float
    tz: float
    camera_id: int
    name: str
    observations: list[Observation]

    @property
    def rotation(self) -> np.ndarray:
        qw, qx, qy, qz = self.qw, self.qx, self.qy, self.qz
        norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        if norm == 0:
            raise ValueError(f"Image {self.name} has a zero-norm quaternion")
        qw, qx, qy, qz = (value / norm for value in (qw, qx, qy, qz))
        return np.array(
            [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ],
            dtype=np.float64,
        )

    @property
    def translation(self) -> np.ndarray:
        return np.array([self.tx, self.ty, self.tz], dtype=np.float64)


def parse_single_simple_radial_camera(path: Path) -> tuple[int, int, int, float, float, float, float]:
    entries = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    if len(entries) != 1 or len(entries[0]) != 8 or entries[0][1] != "SIMPLE_RADIAL":
        raise ValueError("Expected exactly one SIMPLE_RADIAL camera in cameras.txt")
    camera_id, _, width, height, focal, cx, cy, radial = entries[0]
    return int(camera_id), int(width), int(height), float(focal), float(cx), float(cy), float(radial)


def pose_from_fields(fields: list[str]) -> ImageRecord:
    if len(fields) < 10 or Path(fields[9]).suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"Malformed COLMAP pose line: {' '.join(fields[:10])}")
    return ImageRecord(
        image_id=int(fields[0]),
        qw=float(fields[1]),
        qx=float(fields[2]),
        qy=float(fields[3]),
        qz=float(fields[4]),
        tx=float(fields[5]),
        ty=float(fields[6]),
        tz=float(fields[7]),
        camera_id=int(fields[8]),
        name=Path(fields[9]).name,
        observations=[],
    )


def parse_observations(line: str) -> list[Observation]:
    fields = line.split()
    if len(fields) % 3 != 0:
        raise ValueError("COLMAP point2D line does not contain x/y/point-id triples")
    return [
        Observation(float(fields[index]), float(fields[index + 1]), int(fields[index + 2]), index // 3)
        for index in range(0, len(fields), 3)
    ]


def parse_colmap_images(path: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    pending: ImageRecord | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        is_pose = len(fields) >= 10 and Path(fields[9]).suffix.lower() in IMAGE_SUFFIXES
        if is_pose:
            if pending is not None:
                records.append(pending)
            pending = pose_from_fields(fields)
            continue
        if pending is None:
            raise ValueError("Found point2D line before the first image pose")
        pending.observations = parse_observations(line)
        records.append(pending)
        pending = None
    if pending is not None:
        records.append(pending)
    names = [record.name for record in records]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate image names in COLMAP images.txt")
    return records


def undistort_observations(records: list[ImageRecord], camera_matrix: np.ndarray, radial: float) -> None:
    coefficients = np.array([radial, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    for record in records:
        if not record.observations:
            continue
        raw = np.array([[[observation.x, observation.y]] for observation in record.observations], dtype=np.float64)
        converted = cv2.undistortPoints(raw, camera_matrix, coefficients, P=camera_matrix).reshape(-1, 2)
        for observation, point in zip(record.observations, converted):
            observation.undistorted_xy = (float(point[0]), float(point[1]))


def projection_matrix(record: ImageRecord, camera_matrix: np.ndarray) -> np.ndarray:
    return camera_matrix @ np.concatenate((record.rotation, record.translation.reshape(3, 1)), axis=1)


def triangulate_tracks(
    records: list[ImageRecord], camera_matrix: np.ndarray, max_reprojection_error: float
) -> tuple[list[dict[str, object]], dict[str, list[tuple[int, int, int, int]]]]:
    tracks: dict[int, list[tuple[ImageRecord, Observation]]] = defaultdict(list)
    for record in records:
        for observation in record.observations:
            if observation.point_id >= 0 and observation.undistorted_xy is not None:
                tracks[observation.point_id].append((record, observation))

    points: list[dict[str, object]] = []
    color_jobs: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
    for point_id in sorted(tracks):
        track = tracks[point_id]
        if len(track) < 2:
            continue
        rows: list[np.ndarray] = []
        matrices: list[np.ndarray] = []
        for record, observation in track:
            matrix = projection_matrix(record, camera_matrix)
            u, v = observation.undistorted_xy or (None, None)
            if u is None or v is None:
                continue
            rows.extend((u * matrix[2] - matrix[0], v * matrix[2] - matrix[1]))
            matrices.append(matrix)
        if len(rows) < 4:
            continue
        _, _, right_vectors = np.linalg.svd(np.stack(rows), full_matrices=False)
        homogeneous = right_vectors[-1]
        if abs(homogeneous[3]) < 1e-10:
            continue
        xyz = homogeneous[:3] / homogeneous[3]
        errors: list[float] = []
        valid_depth = True
        for matrix, (_, observation) in zip(matrices, track):
            projected = matrix @ np.append(xyz, 1.0)
            if projected[2] <= 1e-8:
                valid_depth = False
                break
            estimated = projected[:2] / projected[2]
            observed = np.array(observation.undistorted_xy)
            errors.append(float(np.linalg.norm(estimated - observed)))
        mean_error = float(np.mean(errors)) if errors else float("inf")
        if not valid_depth or not np.isfinite(mean_error) or mean_error > max_reprojection_error:
            continue
        point_index = len(points)
        points.append(
            {
                "point_id": point_id,
                "xyz": xyz,
                "rgb": (128, 128, 128),
                "mean_reprojection_error_px": mean_error,
                "track_length": len(track),
            }
        )
        color_record, color_observation = track[0]
        u, v = color_observation.undistorted_xy or (0.0, 0.0)
        color_jobs[color_record.name].append((point_index, round(u), round(v), point_id))
    if not points:
        raise ValueError("No valid points could be triangulated from the selected image tracks")
    return points, color_jobs


def find_image_paths(root: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for directory_name in ("images", "unlabeled_Images"):
        directory = root / directory_name
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                if path.name in images:
                    raise ValueError(f"Duplicate RGB filename across released directories: {path.name}")
                images[path.name] = path
    return images


def write_colmap_text(
    sparse_dir: Path,
    records: list[ImageRecord],
    width: int,
    height: int,
    focal: float,
    cx: float,
    cy: float,
    points: list[dict[str, object]],
) -> None:
    sparse_dir.mkdir(parents=True, exist_ok=True)
    (sparse_dir / "cameras.txt").write_text(
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        "# Number of cameras: 1\n"
        f"1 PINHOLE {width} {height} {focal:.17g} {focal:.17g} {cx:.17g} {cy:.17g}\n",
        encoding="utf-8",
    )
    with (sparse_dir / "images.txt").open("w", encoding="utf-8") as handle:
        handle.write("# Image list with two lines of data per image:\n")
        handle.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        handle.write(f"# Number of images: {len(records)}\n")
        for record in records:
            handle.write(
                f"{record.image_id} {record.qw:.17g} {record.qx:.17g} {record.qy:.17g} {record.qz:.17g} "
                f"{record.tx:.17g} {record.ty:.17g} {record.tz:.17g} 1 {record.name}\n"
            )
            handle.write(
                " ".join(
                    f"{observation.undistorted_xy[0]:.17g} {observation.undistorted_xy[1]:.17g} {observation.point_id}"
                    for observation in record.observations
                    if observation.undistorted_xy is not None
                )
                + "\n"
            )
    with (sparse_dir / "points3D.txt").open("w", encoding="utf-8") as handle:
        handle.write("# Re-triangulated from released images.txt point tracks.\n")
        for point in points:
            x, y, z = point["xyz"]
            red, green, blue = point["rgb"]
            handle.write(
                f"{point['point_id']} {x:.17g} {y:.17g} {z:.17g} {red} {green} {blue} "
                f"{point['mean_reprojection_error_px']:.17g}\n"
            )


def prepare_scene(
    dataset_root: Path,
    split_path: Path,
    output_root: Path,
    training_key: str,
    max_reprojection_error: float,
    evaluation_key: str | None = None,
) -> dict[str, object]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output already exists and is not empty: {output_root}")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    training_names = set(split["primary_fold"][training_key])
    evaluation_names = set(split["primary_fold"][evaluation_key]) if evaluation_key is not None else set()
    if training_names & evaluation_names:
        raise ValueError("Training and evaluation image lists must be disjoint")
    selected_names = training_names | evaluation_names
    root = dataset_root.resolve()
    camera_id, width, height, focal, cx, cy, radial = parse_single_simple_radial_camera(
        root / "camera_parameters" / "cameras.txt"
    )
    records_by_name = {record.name: record for record in parse_colmap_images(root / "camera_parameters" / "images.txt")}
    if camera_id != 1 or any(records_by_name[name].camera_id != camera_id for name in selected_names):
        raise ValueError("Selected records must use the released SIMPLE_RADIAL camera")
    if selected_names - records_by_name.keys():
        raise ValueError("Split references images missing from COLMAP images.txt")
    paths_by_name = find_image_paths(root)
    if selected_names - paths_by_name.keys():
        raise ValueError("Split references images missing from the released dataset")
    records = sorted((records_by_name[name] for name in selected_names), key=lambda record: record.image_id)
    training_records = sorted((records_by_name[name] for name in training_names), key=lambda record: record.image_id)
    camera_matrix = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    undistort_observations(records, camera_matrix, radial)
    # Held-out images supply only validation cameras and reference RGB.  They must never
    # influence the initialization point cloud.
    points, color_jobs = triangulate_tracks(training_records, camera_matrix, max_reprojection_error)

    output_root.mkdir(parents=True)
    image_dir = output_root / "images"
    image_dir.mkdir()
    distortion = np.array([radial, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    for record in records:
        image = cv2.imread(str(paths_by_name[record.name]), cv2.IMREAD_COLOR)
        if image is None or image.shape[1] != width or image.shape[0] != height:
            raise ValueError(f"Unexpected released image dimensions: {record.name}")
        undistorted = cv2.undistort(image, camera_matrix, distortion, None, camera_matrix)
        if not cv2.imwrite(str(image_dir / record.name), undistorted):
            raise OSError(f"Could not write undistorted image: {record.name}")
        for point_index, u, v, _ in color_jobs.get(record.name, []):
            if 0 <= u < width and 0 <= v < height:
                blue, green, red = undistorted[v, u]
                points[point_index]["rgb"] = (int(red), int(green), int(blue))

    write_colmap_text(output_root / "sparse" / "0", records, width, height, focal, cx, cy, points)
    if evaluation_names:
        (output_root / "sparse" / "0" / "test.txt").write_text(
            "".join(f"{name}\n" for name in sorted(evaluation_names)), encoding="utf-8"
        )
    metadata = {
        "schema_version": 1,
        "split_id": split["split_id"],
        "training_key": training_key,
        "source_coordinate_frame": "official_colmap_reference",
        "distortion_policy": {
            "input_model": "SIMPLE_RADIAL",
            "output_model": "PINHOLE",
            "method": "OpenCV cv2.undistort with newCameraMatrix equal to released K; full resolution and no crop",
            "resolution": [width, height],
            "focal": focal,
            "principal_point": [cx, cy],
            "radial_k": radial,
        },
        "initialization": {
            "method": "DLT re-triangulation of shared released images.txt POINT3D_ID tracks after undistortion",
            "max_mean_reprojection_error_px": max_reprojection_error,
            "accepted_points": len(points),
            "points3D_source": "generated locally; official points3D.txt is intentionally absent",
        },
        "counts": {
            "training_images": len(training_records),
            "evaluation_images": len(evaluation_names),
            "scene_images": len(records),
            "retriangulated_points": len(points),
        },
        "evaluation": {
            "key": evaluation_key,
            "selection": "explicit sparse/0/test.txt" if evaluation_names else "none",
            "initialization_uses_evaluation_images": False,
        },
        "path_policy": "All paths are relative to this prepared scene; no source absolute path is recorded.",
    }
    (output_root / "preparation_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--training-key", choices=("train_rgb_400", "train_rgb_300"), default="train_rgb_400")
    parser.add_argument(
        "--evaluation-key",
        choices=("validation_labeled",),
        default=None,
        help="Optional held-out list written to sparse/0/test.txt; it is excluded from initialization.",
    )
    parser.add_argument("--max-mean-reprojection-error-px", type=float, default=3.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = prepare_scene(
        dataset_root=args.dataset_root,
        split_path=args.split,
        output_root=args.output_root,
        training_key=args.training_key,
        max_reprojection_error=args.max_mean_reprojection_error_px,
        evaluation_key=args.evaluation_key,
    )
    print(json.dumps(metadata["counts"], ensure_ascii=False))
    print(f"Prepared PINHOLE scene: {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
