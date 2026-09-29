"""Official-P0 MapAnything adapter for the GeometryPrior v2 protocol.

The optional MapAnything dependency is imported only inside the real execution
functions.  This keeps camera/split validation and unit tests independent of
the isolated forward-model environment and, crucially, prevents a dry run from
silently downloading a model.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from .camera_io import ColmapCamera, ColmapImage, read_cameras_txt, read_images_txt, world_to_camera
from .windowing import covisibility_windows


OFFICIAL_MAPANYTHING_URL = "https://github.com/facebookresearch/map-anything"
OFFICIAL_MAPANYTHING_MODEL = "facebook/map-anything-apache"
GEOMETRY_SCHEMA_ID = "geometry_prior_npz_v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_commit(source_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def camera_intrinsics(camera: ColmapCamera) -> np.ndarray:
    if camera.model != "PINHOLE" or len(camera.params) != 4:
        raise ValueError("MapAnything P0 adapter expects the prepared full-frame PINHOLE scene")
    fx, fy, cx, cy = camera.params
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def camera_to_world(image: ColmapImage) -> np.ndarray:
    """Return the official OpenCV/RDF cam2world matrix from COLMAP w2c."""
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = image.rotation_camera_to_world
    matrix[:3, 3] = image.center
    return matrix


def read_sparse_points(path: Path) -> dict[int, np.ndarray]:
    """Read only point ID and XYZ from a COLMAP text sparse model."""
    points: dict[int, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 4:
                raise ValueError(f"Malformed COLMAP points3D line: {line[:120]}")
            points[int(fields[0])] = np.asarray([float(fields[1]), float(fields[2]), float(fields[3])], dtype=np.float64)
    if not points:
        raise ValueError(f"No sparse points found in {path}")
    return points


def training_names_from_split(split_path: Path) -> list[str]:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    primary = split.get("primary_fold")
    if not isinstance(primary, dict):
        raise ValueError("split is missing primary_fold")
    names = primary.get("train_rgb_400")
    holdout = set(primary.get("holdout_all_cameras", []))
    buffer = set(primary.get("buffer_excluded_all_cameras", []))
    if not isinstance(names, list) or not names:
        raise ValueError("primary fold is missing train_rgb_400")
    if set(names) & (holdout | buffer):
        raise ValueError("training list overlaps validation or guard-band cameras")
    return sorted(names)


def build_window_manifest(
    *,
    graph_path: Path,
    split_path: Path,
    window_size: int,
    overlap: int,
    max_windows: int | None,
    experiment_id: str,
    source_commit_value: str,
    checkpoint_sha256: str,
    checkpoint_revision: str,
) -> dict[str, Any]:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    training_names = training_names_from_split(split_path)
    windows = covisibility_windows(
        graph,
        training_names,
        window_size=window_size,
        overlap=overlap,
        max_windows=max_windows,
    )
    return {
        "schema_version": 2,
        "geometry_output_schema": {"schema_id": GEOMETRY_SCHEMA_ID},
        "experiment_id": experiment_id,
        "model": "MapAnything",
        "model_source_url": OFFICIAL_MAPANYTHING_URL,
        "model_source_commit": source_commit_value,
        "model_checkpoint": OFFICIAL_MAPANYTHING_MODEL,
        "model_checkpoint_revision": checkpoint_revision,
        "checkpoint_sha256": checkpoint_sha256,
        "coordinate_frame": "official_colmap_reference",
        "pose_convention": "official_COLMAP_CAMERA_FROM_WORLD_to_OpenCV_cam2world",
        "input_images_scope": "rgb_r1_spatial_v1 primary train_rgb_400 only",
        "training_image_count": len(training_names),
        "window_size": window_size,
        "overlap": overlap,
        "window_count": len(windows),
        "windows": windows,
        "model_execution": "not_run",
    }


def _tensor_numpy(value: Any) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def _read_rgb(path: Path) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read RGB image {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _scale_depth_to_sparse(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    image: ColmapImage,
    sparse_points: dict[int, np.ndarray],
    *,
    minimum_correspondences: int,
    minimum_depth: float,
) -> tuple[np.ndarray, float, int]:
    ids = sorted(image.point3d_ids.intersection(sparse_points))
    if not ids:
        raise ValueError(f"No official sparse tracks for {image.name}")
    world = np.stack([sparse_points[point_id] for point_id in ids])
    camera = world_to_camera(world, image)
    z = camera[:, 2]
    positive = z > minimum_depth
    u = intrinsics[0, 0] * (camera[:, 0] / np.maximum(z, minimum_depth)) + intrinsics[0, 2]
    v = intrinsics[1, 1] * (camera[:, 1] / np.maximum(z, minimum_depth)) + intrinsics[1, 2]
    x = np.rint(u).astype(np.int64)
    y = np.rint(v).astype(np.int64)
    inside = (x >= 0) & (x < depth.shape[1]) & (y >= 0) & (y < depth.shape[0])
    predicted = np.zeros_like(z)
    predicted[inside] = depth[y[inside], x[inside]]
    usable = positive & inside & np.isfinite(predicted) & (predicted > minimum_depth)
    count = int(usable.sum())
    if count < minimum_correspondences:
        raise ValueError(
            f"{image.name}: only {count} usable sparse depth correspondences; need {minimum_correspondences}"
        )
    ratio = float(np.median(z[usable] / predicted[usable]))
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError(f"{image.name}: invalid MapAnything-to-P0 depth scale {ratio}")
    return depth * ratio, ratio, count


def _world_points_from_p0(depth: np.ndarray, intrinsics: np.ndarray, cam2world: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32), indexing="xy")
    points_camera = np.stack(
        [
            (x - intrinsics[0, 2]) * depth / intrinsics[0, 0],
            (y - intrinsics[1, 2]) * depth / intrinsics[1, 1],
            depth,
        ],
        axis=-1,
    )
    return (points_camera @ cam2world[:3, :3].T + cam2world[:3, 3]).astype(np.float32)


def _write_view_npz(
    output_path: Path,
    *,
    depth: np.ndarray,
    confidence: np.ndarray,
    valid_mask: np.ndarray,
    world_points: np.ndarray,
    intrinsics: np.ndarray,
    cam2world: np.ndarray,
    pred_cam2world: np.ndarray | None,
    depth_scale_ratio: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "depth": depth.astype(np.float32),
        "confidence": confidence.astype(np.float32),
        "valid_mask": valid_mask.astype(np.uint8),
        "world_points": world_points.astype(np.float32),
        "pinhole_intrinsics": intrinsics.astype(np.float64),
        "official_cam2world": cam2world.astype(np.float64),
        "depth_scale_ratio": np.asarray(depth_scale_ratio, dtype=np.float64),
    }
    if pred_cam2world is not None:
        arrays["pred_cam2world"] = pred_cam2world.astype(np.float64)
    np.savez_compressed(output_path, **arrays)


def run_mapanything_windows(
    *,
    scene_root: Path,
    output_dir: Path,
    window_manifest: dict[str, Any],
    checkpoint_root: Path,
    checkpoint_file: Path,
    amp_dtype: str,
    minimum_correspondences: int,
    minimum_depth: float,
) -> dict[str, Any]:
    """Run MapAnything only after the caller has completed source/hash gates."""
    import torch
    from mapanything.models import MapAnything
    from mapanything.utils.image import preprocess_inputs

    sparse = scene_root / "sparse" / "0"
    cameras = read_cameras_txt(sparse / "cameras.txt")
    images = read_images_txt(sparse / "images.txt")
    image_by_name = {image.name: image for image in images.values()}
    sparse_points = read_sparse_points(sparse / "points3D.txt")
    training_names = set(training_names_from_split(Path(window_manifest["split_path"])))
    for window in window_manifest["windows"]:
        unknown = set(window["images"]) - training_names
        if unknown:
            raise ValueError(f"Window includes non-training image(s): {sorted(unknown)}")

    if not torch.cuda.is_available():
        raise RuntimeError("MapAnything real execution requires CUDA")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    start = time.monotonic()
    # Loading from an exact local snapshot prevents a network fetch or a mutable
    # Hub revision during a registered experiment.
    model = MapAnything.from_pretrained(str(checkpoint_root), local_files_only=True).to(device).eval()
    model_load_seconds = time.monotonic() - start
    records: list[dict[str, Any]] = []
    windows_root = output_dir / "windows"
    try:
        for window in window_manifest["windows"]:
            raw_views = []
            p0_by_name: dict[str, tuple[ColmapImage, np.ndarray, np.ndarray]] = {}
            for name in window["images"]:
                image = image_by_name.get(name)
                if image is None:
                    raise ValueError(f"Prepared scene lacks camera for {name}")
                camera = cameras.get(image.camera_id)
                if camera is None:
                    raise ValueError(f"Prepared scene lacks camera ID {image.camera_id}")
                intrinsic = camera_intrinsics(camera)
                cam2world = camera_to_world(image)
                p0_by_name[name] = (image, intrinsic, cam2world)
                raw_views.append(
                    {
                        "img": _read_rgb(scene_root / "images" / name),
                        "intrinsics": intrinsic.astype(np.float32),
                        "camera_poses": cam2world.astype(np.float32),
                        "is_metric_scale": torch.tensor([False], dtype=torch.bool),
                        "instance": [name],
                        "idx": [len(raw_views)],
                    }
                )
            processed = preprocess_inputs(raw_views, resize_mode="fixed_mapping", resolution_set=518)
            processed_intrinsics = {
                name: _tensor_numpy(view["intrinsics"][0]) for name, view in zip(window["images"], processed)
            }
            processed_poses = {
                name: _tensor_numpy(view["camera_poses"][0]) for name, view in zip(window["images"], processed)
            }
            predictions = model.infer(
                processed,
                memory_efficient_inference=True,
                minibatch_size=1,
                use_amp=True,
                amp_dtype=amp_dtype,
                apply_mask=True,
                mask_edges=True,
                apply_confidence_mask=False,
                use_multiview_confidence=False,
                ignore_calibration_inputs=False,
                ignore_depth_inputs=True,
                ignore_pose_inputs=False,
                ignore_depth_scale_inputs=True,
                ignore_pose_scale_inputs=True,
            )
            for name, prediction in zip(window["images"], predictions):
                image, _, cam2world = p0_by_name[name]
                depth = _tensor_numpy(prediction["depth_z"][0, ..., 0])
                confidence = _tensor_numpy(prediction["conf"][0])
                mask = _tensor_numpy(prediction["mask"][0, ..., 0])
                intrinsic = processed_intrinsics[name]
                scaled_depth, scale_ratio, correspondence_count = _scale_depth_to_sparse(
                    depth,
                    intrinsic,
                    image,
                    sparse_points,
                    minimum_correspondences=minimum_correspondences,
                    minimum_depth=minimum_depth,
                )
                valid = (
                    (mask > 0)
                    & np.isfinite(scaled_depth)
                    & np.isfinite(confidence)
                    & (scaled_depth > minimum_depth)
                )
                world_points = _world_points_from_p0(scaled_depth, intrinsic, cam2world)
                world_points[~valid] = np.nan
                predicted_pose = None
                if "camera_poses" in prediction:
                    predicted_pose = _tensor_numpy(prediction["camera_poses"][0])
                output_path = windows_root / window["window_id"] / f"{Path(name).stem}.npz"
                _write_view_npz(
                    output_path,
                    depth=scaled_depth,
                    confidence=confidence,
                    valid_mask=valid,
                    world_points=world_points,
                    intrinsics=intrinsic,
                    cam2world=cam2world,
                    pred_cam2world=predicted_pose,
                    depth_scale_ratio=scale_ratio,
                )
                records.append(
                    {
                        "window_manifest_id": window["window_id"],
                        "image_name": name,
                        "official_camera_id": image.camera_id,
                        "input_resolution": [int(depth.shape[1]), int(depth.shape[0])],
                        "coordinate_frame": "official_colmap_reference",
                        "depth_scale": {
                            "method": "per_view_median_official_sparse_depth_ratio",
                            "ratio": scale_ratio,
                            "correspondence_count": correspondence_count,
                        },
                        "source_model": "MapAnything",
                        "source_commit": window_manifest["model_source_commit"],
                        "checkpoint_sha256": sha256_file(checkpoint_file),
                        "output_npz": str(output_path.relative_to(output_dir)),
                        "finite_depth_fraction": float(np.isfinite(scaled_depth).mean()),
                        "valid_fraction": float(valid.mean()),
                        "p0_input_pose_max_abs_error": float(np.max(np.abs(cam2world - processed_poses[name]))),
                    }
                )
    finally:
        torch.cuda.synchronize()
        peak_memory_mib = int(torch.cuda.max_memory_allocated() / (1024 * 1024))
        del model
        torch.cuda.empty_cache()

    return {
        "model_execution": "completed",
        "model_load_seconds": model_load_seconds,
        "inference_seconds": time.monotonic() - start,
        "peak_torch_allocated_mib": peak_memory_mib,
        "amp_dtype": amp_dtype,
        "records": records,
    }
