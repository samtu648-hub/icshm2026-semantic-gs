"""Read the released COLMAP text model without depending on pycolmap.

The released ``images.txt`` poses are COLMAP/OpenCV world-to-camera poses.  This
module keeps that convention explicit and exposes the corresponding camera-to-
world rotation and center for geometry-prior preparation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import cv2


@dataclass(frozen=True)
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]


@dataclass(frozen=True)
class ColmapImage:
    image_id: int
    name: str
    camera_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    points2d: tuple[tuple[float, float, int], ...]

    @property
    def rotation_world_to_camera(self) -> np.ndarray:
        return qvec_to_rotation(self.qvec)

    @property
    def rotation_camera_to_world(self) -> np.ndarray:
        return self.rotation_world_to_camera.T

    @property
    def center(self) -> np.ndarray:
        return -self.rotation_camera_to_world @ self.tvec

    @property
    def optical_axis(self) -> np.ndarray:
        """COLMAP/OpenCV camera +Z expressed in world coordinates."""
        axis = self.rotation_camera_to_world[:, 2]
        return axis / max(float(np.linalg.norm(axis)), np.finfo(float).eps)

    @property
    def point3d_ids(self) -> frozenset[int]:
        return frozenset(point_id for _, _, point_id in self.points2d if point_id >= 0)


@dataclass(frozen=True)
class ColmapRig:
    rig_id: int
    num_sensors: int
    ref_sensor_type: str
    ref_sensor_id: int
    sensors: tuple[tuple[str, int, bool, tuple[float, ...] | None], ...]


@dataclass(frozen=True)
class ColmapFrame:
    frame_id: int
    rig_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    data_ids: tuple[tuple[str, int, int], ...]


def _data_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line and not line.startswith("#"):
                yield line


def read_cameras_txt(path: str | Path) -> dict[int, ColmapCamera]:
    cameras: dict[int, ColmapCamera] = {}
    for line in _data_lines(Path(path)):
        fields = line.split()
        if len(fields) < 5:
            raise ValueError(f"Invalid COLMAP camera line: {line[:120]}")
        camera_id, model, width, height = int(fields[0]), fields[1], int(fields[2]), int(fields[3])
        cameras[camera_id] = ColmapCamera(camera_id, model, width, height, tuple(map(float, fields[4:])))
    return cameras


def _parse_points2d(line: str) -> tuple[tuple[float, float, int], ...]:
    fields = line.split()
    if len(fields) % 3:
        raise ValueError("COLMAP image observation line must contain x y POINT3D_ID triples")
    return tuple((float(fields[i]), float(fields[i + 1]), int(fields[i + 2])) for i in range(0, len(fields), 3))


def read_images_txt(path: str | Path) -> dict[int, ColmapImage]:
    lines = list(_data_lines(Path(path)))
    images: dict[int, ColmapImage] = {}
    if len(lines) % 2:
        raise ValueError("COLMAP images.txt must contain pose/observation pairs")
    for index in range(0, len(lines), 2):
        fields = lines[index].split()
        if len(fields) < 10:
            raise ValueError(f"Invalid COLMAP image pose line: {lines[index][:160]}")
        image_id = int(fields[0])
        qvec = np.asarray(list(map(float, fields[1:5])), dtype=np.float64)
        tvec = np.asarray(list(map(float, fields[5:8])), dtype=np.float64)
        camera_id = int(fields[8])
        name = " ".join(fields[9:])
        if not np.isclose(np.linalg.norm(qvec), 1.0, atol=1e-4):
            raise ValueError(f"Non-unit COLMAP quaternion for image {name}")
        images[image_id] = ColmapImage(image_id, name, camera_id, qvec, tvec, _parse_points2d(lines[index + 1]))
    return images


def read_rigs_txt(path: str | Path) -> dict[int, ColmapRig]:
    rigs: dict[int, ColmapRig] = {}
    for line in _data_lines(Path(path)):
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"Invalid COLMAP rig line: {line[:160]}")
        rig_id, num_sensors = int(fields[0]), int(fields[1])
        ref_type, ref_id = fields[2], int(fields[3])
        cursor = 4
        sensors = []
        if cursor == len(fields) and num_sensors == 1:
            sensors.append((ref_type, ref_id, False, None))
            rigs[rig_id] = ColmapRig(rig_id, num_sensors, ref_type, ref_id, tuple(sensors))
            continue
        for _ in range(num_sensors):
            if cursor + 2 > len(fields):
                raise ValueError(f"Truncated COLMAP rig sensor list: {line[:160]}")
            sensor_type, sensor_id, has_pose = fields[cursor], int(fields[cursor + 1]), bool(int(fields[cursor + 2]))
            cursor += 3
            pose = None
            if has_pose:
                if cursor + 7 > len(fields):
                    raise ValueError(f"Truncated COLMAP rig sensor pose: {line[:160]}")
                pose = tuple(map(float, fields[cursor : cursor + 7]))
                cursor += 7
            sensors.append((sensor_type, sensor_id, has_pose, pose))
        rigs[rig_id] = ColmapRig(rig_id, num_sensors, ref_type, ref_id, tuple(sensors))
    return rigs


def read_frames_txt(path: str | Path) -> dict[int, ColmapFrame]:
    frames: dict[int, ColmapFrame] = {}
    for line in _data_lines(Path(path)):
        fields = line.split()
        if len(fields) < 10:
            raise ValueError(f"Invalid COLMAP frame line: {line[:160]}")
        frame_id, rig_id = int(fields[0]), int(fields[1])
        qvec = np.asarray(list(map(float, fields[2:6])), dtype=np.float64)
        tvec = np.asarray(list(map(float, fields[6:9])), dtype=np.float64)
        count = int(fields[9])
        cursor = 10
        data_ids = []
        for _ in range(count):
            if cursor + 3 > len(fields):
                raise ValueError(f"Truncated COLMAP frame data IDs: {line[:160]}")
            data_ids.append((fields[cursor], int(fields[cursor + 1]), int(fields[cursor + 2])))
            cursor += 3
        frames[frame_id] = ColmapFrame(frame_id, rig_id, qvec, tvec, tuple(data_ids))
    return frames


def qvec_to_rotation(qvec: np.ndarray) -> np.ndarray:
    """Convert COLMAP Hamilton quaternion ``[qw, qx, qy, qz]`` to R."""
    q = np.asarray(qvec, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError("COLMAP quaternion must have shape (4,)")
    q = q / np.linalg.norm(q)
    qw, qx, qy, qz = q
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def rotation_to_qvec(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix to COLMAP ``[qw, qx, qy, qz]``."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6):
        raise ValueError("rotation must be an orthonormal 3x3 matrix")
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        q = np.array([0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale, (matrix[0, 2] - matrix[2, 0]) / scale, (matrix[1, 0] - matrix[0, 1]) / scale])
    else:
        diagonal = np.diag(matrix)
        pivot = int(np.argmax(diagonal))
        next_axis = (pivot + 1) % 3
        last_axis = (pivot + 2) % 3
        scale = 2.0 * np.sqrt(max(1.0 + diagonal[pivot] - diagonal[next_axis] - diagonal[last_axis], 1e-12))
        q = np.zeros(4, dtype=np.float64)
        q[pivot + 1] = 0.25 * scale
        q[0] = (matrix[last_axis, next_axis] - matrix[next_axis, last_axis]) / scale
        q[next_axis + 1] = (matrix[next_axis, pivot] + matrix[pivot, next_axis]) / scale
        q[last_axis + 1] = (matrix[last_axis, pivot] + matrix[pivot, last_axis]) / scale
    if q[0] < 0:
        q *= -1
    return q / np.linalg.norm(q)


def pose_roundtrip(rotation: np.ndarray, translation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return w2c after w2c -> c2w -> w2c for unit tests and audits."""
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    center = -rotation.T @ translation
    recovered_translation = -rotation @ center
    return rotation, recovered_translation


def world_to_camera(points_world: np.ndarray, image: ColmapImage) -> np.ndarray:
    points = np.asarray(points_world, dtype=np.float64)
    return points @ image.rotation_world_to_camera.T + image.tvec


def project_world_points(points_world: np.ndarray, image: ColmapImage, camera: ColmapCamera) -> tuple[np.ndarray, np.ndarray]:
    """Project world points using the official COLMAP camera model."""
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    rotation = image.rotation_world_to_camera
    rvec, _ = cv2.Rodrigues(rotation)
    if camera.model == "SIMPLE_RADIAL":
        if len(camera.params) != 4:
            raise ValueError("SIMPLE_RADIAL expects focal, cx, cy, k")
        focal, cx, cy, radial = camera.params
        matrix = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]])
        distortion = np.array([radial, 0.0, 0.0, 0.0, 0.0])
    elif camera.model == "PINHOLE":
        if len(camera.params) != 4:
            raise ValueError("PINHOLE expects fx, fy, cx, cy")
        fx, fy, cx, cy = camera.params
        matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        distortion = np.zeros(5)
    else:
        raise ValueError(f"Unsupported camera model for projection: {camera.model}")
    pixels, _ = cv2.projectPoints(points, rvec, image.tvec, matrix, distortion)
    return pixels.reshape(-1, 2), world_to_camera(points, image)[:, 2]


def backproject_pixels(pixels: np.ndarray, depth_camera: np.ndarray, camera: ColmapCamera, image: ColmapImage) -> np.ndarray:
    """Backproject official pixels with camera-space depth into world coordinates."""
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
    depth = np.asarray(depth_camera, dtype=np.float64).reshape(-1)
    if pixels.shape[0] != depth.shape[0]:
        raise ValueError("pixels and depth_camera must have the same count")
    if camera.model == "SIMPLE_RADIAL":
        focal, cx, cy, radial = camera.params
        matrix = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]])
        distortion = np.array([radial, 0.0, 0.0, 0.0, 0.0])
    elif camera.model == "PINHOLE":
        fx, fy, cx, cy = camera.params
        matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        distortion = np.zeros(5)
    else:
        raise ValueError(f"Unsupported camera model for backprojection: {camera.model}")
    normalized = cv2.undistortPoints(pixels, matrix, distortion).reshape(-1, 2)
    points_camera = np.column_stack([normalized * depth[:, None], depth])
    return (points_camera - image.tvec) @ image.rotation_world_to_camera
