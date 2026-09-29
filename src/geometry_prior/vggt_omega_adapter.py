"""Real VGGT-Omega adapter and CPU audits for the GEO-025-PF smoke.

The module has no top-level model or torch import. Model dependencies are
loaded only by the two explicitly real entry points.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .camera_io import (
    ColmapCamera,
    ColmapImage,
    read_cameras_txt,
    read_images_txt,
    world_to_camera,
)
from .vggt_omega import CHECKPOINT_SHA256, SOURCE_COMMIT, training_names_from_split


MODEL_NAME = "VGGT-Omega-1B-512"
PREPROCESS_MODE = "balanced"
IMAGE_RESOLUTION = 512
PATCH_SIZE = 16
MINIMUM_CORRESPONDENCES = 64
MINIMUM_DEPTH = 1e-6
REPEAT_COUNT = 2
REQUIRED_RECORD_METADATA = (
    "source_model",
    "source_commit",
    "checkpoint_sha256",
    "image_name",
    "official_camera_id",
    "input_resolution",
    "coordinate_frame",
    "depth_scale",
    "window_manifest_id",
)
REQUIRED_NPZ_ARRAYS = {
    "depth",
    "confidence",
    "valid_mask",
    "world_points",
    "pinhole_intrinsics",
    "official_cam2world",
    "depth_scale_ratio",
}


def balanced_preprocess_transform(
    width: int,
    height: int,
    *,
    image_resolution: int = IMAGE_RESOLUTION,
    patch_size: int = PATCH_SIZE,
) -> dict[str, Any]:
    """Mirror upstream balanced shape and extreme-aspect center-crop math."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if image_resolution <= 0 or patch_size <= 0 or image_resolution % patch_size:
        raise ValueError(
            "image_resolution must be positive and divisible by patch_size"
        )
    crop_left = 0
    crop_top = 0
    crop_width = width
    crop_height = height
    aspect_ratio = height / width
    if aspect_ratio < 0.5:
        crop_width = min(width, max(1, int(round(height / 0.5))))
        crop_left = max((width - crop_width) // 2, 0)
    elif aspect_ratio > 2.0:
        crop_height = min(height, max(1, int(round(width * 2.0))))
        crop_top = max((height - crop_height) // 2, 0)
    cropped_aspect = crop_height / crop_width
    token_number = (image_resolution // patch_size) ** 2
    width_patches_float = np.sqrt(token_number / cropped_aspect)
    height_patches_float = token_number / width_patches_float
    width_patches = max(1, int(np.round(width_patches_float)))
    height_patches = max(1, int(np.round(height_patches_float)))
    target_width = width_patches * patch_size
    target_height = height_patches * patch_size
    return {
        "mode": PREPROCESS_MODE,
        "image_resolution": image_resolution,
        "patch_size": patch_size,
        "source_size_wh": [width, height],
        "crop_xywh": [crop_left, crop_top, crop_width, crop_height],
        "crop_applied": [crop_left, crop_top, crop_width, crop_height]
        != [0, 0, width, height],
        "target_size_wh": [target_width, target_height],
        "scale_xy": [target_width / crop_width, target_height / crop_height],
        "interpolation": "PIL.Image.Resampling.BICUBIC",
        "padding": "none_for_uniform_prepared_scene",
    }


def scale_pinhole_intrinsics(
    intrinsics: np.ndarray, transform: dict[str, Any]
) -> np.ndarray:
    """Apply the exact crop and independent x/y resize to a PINHOLE matrix."""
    matrix = np.asarray(intrinsics, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("intrinsics must be a finite 3x3 matrix")
    left, top, _, _ = transform["crop_xywh"]
    scale_x, scale_y = transform["scale_xy"]
    output = matrix.copy()
    output[0, 0] *= scale_x
    output[1, 1] *= scale_y
    output[0, 2] = (output[0, 2] - left) * scale_x
    output[1, 2] = (output[1, 2] - top) * scale_y
    return output


def camera_intrinsics(camera: ColmapCamera) -> np.ndarray:
    if camera.model != "PINHOLE" or len(camera.params) != 4:
        raise ValueError(
            f"prepared VGGT-Omega scene must use PINHOLE, got {camera.model}"
        )
    fx, fy, cx, cy = camera.params
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def camera_to_world(image: ColmapImage) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = image.rotation_camera_to_world
    matrix[:3, 3] = image.center
    return matrix


def camera_from_world_to_cam2world(extrinsics: np.ndarray) -> np.ndarray:
    matrix = np.asarray(extrinsics, dtype=np.float64)
    if matrix.shape == (3, 4):
        matrix = np.vstack((matrix, np.asarray([0.0, 0.0, 0.0, 1.0])))
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(
            f"predicted camera-from-world must be finite 3x4 or 4x4, got {matrix.shape}"
        )
    return np.linalg.inv(matrix)


def read_sparse_points(path: Path) -> dict[int, np.ndarray]:
    points: dict[int, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 4:
                raise ValueError(f"Malformed COLMAP points3D line: {line[:120]}")
            points[int(fields[0])] = np.asarray(
                list(map(float, fields[1:4])), dtype=np.float64
            )
    if not points:
        raise ValueError(f"No sparse points found in {path}")
    return points


def scale_depth_to_sparse(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    image: ColmapImage,
    sparse_points: dict[int, np.ndarray],
    *,
    minimum_correspondences: int = MINIMUM_CORRESPONDENCES,
    minimum_depth: float = MINIMUM_DEPTH,
) -> tuple[np.ndarray, float, int]:
    """Scale camera +Z depth by median official sparse-z/predicted-z."""
    predicted_depth = np.asarray(depth)
    if predicted_depth.ndim != 2:
        raise ValueError("predicted depth must have shape HxW")
    ids = sorted(image.point3d_ids.intersection(sparse_points))
    if not ids:
        raise ValueError(f"No official sparse tracks for {image.name}")
    world = np.stack([sparse_points[point_id] for point_id in ids])
    camera = world_to_camera(world, image)
    official_z = camera[:, 2]
    matrix = np.asarray(intrinsics, dtype=np.float64)
    u = (
        matrix[0, 0] * camera[:, 0] / np.maximum(official_z, minimum_depth)
        + matrix[0, 2]
    )
    v = (
        matrix[1, 1] * camera[:, 1] / np.maximum(official_z, minimum_depth)
        + matrix[1, 2]
    )
    x = np.rint(u).astype(np.int64)
    y = np.rint(v).astype(np.int64)
    inside = (
        (x >= 0)
        & (x < predicted_depth.shape[1])
        & (y >= 0)
        & (y < predicted_depth.shape[0])
    )
    sampled = np.full(official_z.shape, np.nan, dtype=np.float64)
    sampled[inside] = predicted_depth[y[inside], x[inside]]
    usable = (
        (official_z > minimum_depth)
        & inside
        & np.isfinite(sampled)
        & (sampled > minimum_depth)
    )
    count = int(usable.sum())
    if count < minimum_correspondences:
        raise ValueError(
            f"{image.name}: only {count} usable sparse depth correspondences; need {minimum_correspondences}"
        )
    ratio = float(np.median(official_z[usable] / sampled[usable]))
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError(f"{image.name}: invalid VGGT-Omega-to-P0 depth scale {ratio}")
    return (predicted_depth * ratio).astype(np.float32), ratio, count


def world_points_from_official_p0(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    official_cam2world: np.ndarray,
) -> np.ndarray:
    depth_array = np.asarray(depth, dtype=np.float64)
    matrix = np.asarray(intrinsics, dtype=np.float64)
    cam2world = np.asarray(official_cam2world, dtype=np.float64)
    height, width = depth_array.shape
    x, y = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
        indexing="xy",
    )
    camera_points = np.stack(
        (
            (x - matrix[0, 2]) * depth_array / matrix[0, 0],
            (y - matrix[1, 2]) * depth_array / matrix[1, 1],
            depth_array,
        ),
        axis=-1,
    )
    return (camera_points @ cam2world[:3, :3].T + cam2world[:3, 3]).astype(np.float32)


def valid_depth_mask(depth: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    depth_array = np.asarray(depth)
    confidence_array = np.asarray(confidence)
    if depth_array.shape != confidence_array.shape or depth_array.ndim != 2:
        raise ValueError("depth and confidence must have the same HxW shape")
    return (
        np.isfinite(depth_array)
        & (depth_array > MINIMUM_DEPTH)
        & np.isfinite(confidence_array)
    )


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_geometry_npz(
    path: Path,
    *,
    depth: np.ndarray,
    confidence: np.ndarray,
    valid_mask: np.ndarray,
    world_points: np.ndarray,
    pinhole_intrinsics: np.ndarray,
    official_cam2world: np.ndarray,
    pred_cam2world: np.ndarray | None,
    depth_scale_ratio: float,
) -> None:
    arrays = {
        "depth": np.asarray(depth, dtype=np.float32),
        "confidence": np.asarray(confidence, dtype=np.float32),
        "valid_mask": np.asarray(valid_mask, dtype=np.uint8),
        "world_points": np.asarray(world_points, dtype=np.float32),
        "pinhole_intrinsics": np.asarray(pinhole_intrinsics, dtype=np.float64),
        "official_cam2world": np.asarray(official_cam2world, dtype=np.float64),
        "depth_scale_ratio": np.asarray(depth_scale_ratio, dtype=np.float64),
    }
    if pred_cam2world is not None:
        arrays["pred_cam2world"] = np.asarray(pred_cam2world, dtype=np.float64)
    _atomic_npz(path, arrays)


def audit_geometry_npz(
    path: Path, *, expected_resolution_wh: tuple[int, int] | None = None
) -> dict[str, Any]:
    """Strictly audit one actual geometry_prior_npz_v2 archive."""
    if not path.is_file():
        raise FileNotFoundError(f"Geometry NPZ does not exist: {path}")
    with np.load(path, allow_pickle=False) as archive:
        missing = REQUIRED_NPZ_ARRAYS - set(archive.files)
        if missing:
            raise ValueError(f"{path}: missing required arrays {sorted(missing)}")
        unexpected = set(archive.files) - (REQUIRED_NPZ_ARRAYS | {"pred_cam2world"})
        if unexpected:
            raise ValueError(f"{path}: unexpected arrays {sorted(unexpected)}")
        depth = archive["depth"]
        confidence = archive["confidence"]
        valid_mask = archive["valid_mask"]
        world_points = archive["world_points"]
        intrinsics = archive["pinhole_intrinsics"]
        cam2world = archive["official_cam2world"]
        scale_ratio = archive["depth_scale_ratio"]
        pred_cam2world = (
            archive["pred_cam2world"] if "pred_cam2world" in archive.files else None
        )

    expected_dtypes = {
        "depth": (depth.dtype, np.dtype(np.float32)),
        "confidence": (confidence.dtype, np.dtype(np.float32)),
        "valid_mask": (valid_mask.dtype, np.dtype(np.uint8)),
        "world_points": (world_points.dtype, np.dtype(np.float32)),
        "pinhole_intrinsics": (intrinsics.dtype, np.dtype(np.float64)),
        "official_cam2world": (cam2world.dtype, np.dtype(np.float64)),
        "depth_scale_ratio": (scale_ratio.dtype, np.dtype(np.float64)),
    }
    if pred_cam2world is not None:
        expected_dtypes["pred_cam2world"] = (pred_cam2world.dtype, np.dtype(np.float64))
    wrong = {
        name: str(actual)
        for name, (actual, expected) in expected_dtypes.items()
        if actual != expected
    }
    if wrong:
        raise ValueError(f"{path}: wrong array dtypes {wrong}")
    if (
        depth.ndim != 2
        or confidence.shape != depth.shape
        or valid_mask.shape != depth.shape
    ):
        raise ValueError(
            f"{path}: depth, confidence and valid_mask must share HxW shape"
        )
    if world_points.shape != depth.shape + (3,):
        raise ValueError(f"{path}: world_points must have shape HxWx3")
    if intrinsics.shape != (3, 3) or cam2world.shape != (4, 4):
        raise ValueError(f"{path}: invalid intrinsic or official pose shape")
    if pred_cam2world is not None and pred_cam2world.shape != (4, 4):
        raise ValueError(f"{path}: pred_cam2world must have shape 4x4")
    if scale_ratio.shape != ():
        raise ValueError(f"{path}: depth_scale_ratio must be a scalar")
    if expected_resolution_wh is not None and depth.shape != (
        expected_resolution_wh[1],
        expected_resolution_wh[0],
    ):
        raise ValueError(f"{path}: output resolution does not match manifest")
    if not np.isin(valid_mask, (0, 1)).all():
        raise ValueError(f"{path}: valid_mask must contain only 0 and 1")
    expected_valid = valid_depth_mask(depth, confidence)
    valid = valid_mask.astype(bool)
    if not np.array_equal(valid, expected_valid):
        raise ValueError(
            f"{path}: valid_mask does not exactly match finite positive depth and finite confidence"
        )
    if not valid.any():
        raise ValueError(f"{path}: valid_mask contains no valid output")
    finite_fractions = {
        "depth": float(np.isfinite(depth).mean()),
        "confidence": float(np.isfinite(confidence).mean()),
        "world_points": float(np.isfinite(world_points).mean()),
    }
    if any(value != 1.0 for value in finite_fractions.values()):
        raise ValueError(f"{path}: nonfinite full-grid outputs {finite_fractions}")
    if np.any(depth[valid] <= MINIMUM_DEPTH):
        raise ValueError(f"{path}: valid depth is nonpositive")
    if (
        not np.isfinite(intrinsics).all()
        or intrinsics[0, 0] <= 0
        or intrinsics[1, 1] <= 0
        or not np.allclose(intrinsics[2], [0.0, 0.0, 1.0])
        or not np.isclose(intrinsics[0, 1], 0.0)
        or not np.isclose(intrinsics[1, 0], 0.0)
    ):
        raise ValueError(f"{path}: invalid PINHOLE intrinsics")
    if not np.isfinite(cam2world).all() or not np.allclose(
        cam2world[3], [0.0, 0.0, 0.0, 1.0]
    ):
        raise ValueError(f"{path}: invalid official_cam2world")
    rotation = cam2world[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-6
    ):
        raise ValueError(
            f"{path}: official_cam2world rotation is not proper orthonormal"
        )
    if pred_cam2world is not None and not np.isfinite(pred_cam2world).all():
        raise ValueError(f"{path}: pred_cam2world is nonfinite")
    ratio = float(scale_ratio)
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError(f"{path}: depth_scale_ratio must be finite and positive")

    expected_world = world_points_from_official_p0(depth, intrinsics, cam2world)
    world_error = np.linalg.norm(
        world_points.astype(np.float64) - expected_world.astype(np.float64), axis=2
    )
    if not np.allclose(world_points, expected_world, rtol=2e-5, atol=2e-5):
        raise ValueError(f"{path}: world_points are not official-P0 backprojections")

    world = world_points[valid].astype(np.float64)
    camera = (world - cam2world[:3, 3]) @ rotation
    rows, columns = np.nonzero(valid)
    projected_x = intrinsics[0, 0] * camera[:, 0] / camera[:, 2] + intrinsics[0, 2]
    projected_y = intrinsics[1, 1] * camera[:, 1] / camera[:, 2] + intrinsics[1, 2]
    pixel_error = np.maximum(np.abs(projected_x - columns), np.abs(projected_y - rows))
    depth_error = np.abs(camera[:, 2] - depth[valid].astype(np.float64))
    if float(pixel_error.max(initial=0.0)) > 1e-3 or not np.allclose(
        camera[:, 2], depth[valid], rtol=2e-5, atol=2e-5
    ):
        raise ValueError(f"{path}: official P0 project/backproject roundtrip failed")
    return {
        "path": str(path),
        "shape_hw": list(depth.shape),
        "valid_count": int(valid.sum()),
        "valid_fraction": float(valid.mean()),
        "finite_fractions": finite_fractions,
        "max_world_backprojection_error": float(world_error.max(initial=0.0)),
        "max_project_roundtrip_pixel_error": float(pixel_error.max(initial=0.0)),
        "max_project_roundtrip_depth_error": float(depth_error.max(initial=0.0)),
    }


def audit_geometry_outputs(
    output_dir: Path,
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    split: dict,
) -> dict[str, Any]:
    training_names = set(training_names_from_split(split))
    expected_pairs = {
        (window["window_manifest_id"], name)
        for window in manifest["windows"]
        for name in window["images"]
    }
    actual_pairs: set[tuple[str, str]] = set()
    audits = []
    for record in records:
        missing_metadata = [
            key for key in REQUIRED_RECORD_METADATA if key not in record
        ]
        if missing_metadata:
            raise ValueError(
                f"record is missing geometry_prior_npz_v2 metadata: {missing_metadata}"
            )
        pair = (record["window_manifest_id"], record["image_name"])
        if pair in actual_pairs:
            raise ValueError(f"duplicate geometry record: {pair}")
        actual_pairs.add(pair)
        if record["image_name"] not in training_names:
            raise ValueError(
                f"geometry output leaks non-training image: {record['image_name']}"
            )
        output_npz = record.get("output_npz")
        if not isinstance(output_npz, str) or not output_npz:
            raise ValueError("record is missing a nonempty output_npz path")
        relative = Path(output_npz)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"invalid output_npz path: {relative}")
        width, height = record["input_resolution"]
        audits.append(
            audit_geometry_npz(
                output_dir / relative, expected_resolution_wh=(int(width), int(height))
            )
        )
    if actual_pairs != expected_pairs:
        missing = sorted(expected_pairs - actual_pairs)
        extra = sorted(actual_pairs - expected_pairs)
        raise ValueError(
            f"geometry record coverage mismatch; missing={missing}, extra={extra}"
        )
    selected_images = {name for _, name in actual_pairs}
    if manifest.get("scope") == "full_training" and selected_images != training_names:
        raise ValueError(
            "full-training output does not cover exactly primary_fold.train_rgb_400"
        )
    return {
        "record_count": len(records),
        "unique_image_count": len(selected_images),
        "split_isolation": "primary_fold.train_rgb_400_only",
        "npz_audits": audits,
        "all_valid_values_finite": True,
        "all_full_grid_values_finite": True,
        "all_valid_depth_positive": True,
        "all_p0_roundtrips_passed": True,
    }


def repeatability_metrics(
    first: dict[str, np.ndarray], second: dict[str, np.ndarray]
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key in ("depth", "confidence", "pred_cam2world"):
        left = first.get(key)
        right = second.get(key)
        if left is None or right is None:
            metrics[key] = {
                "present_in_both": left is None and right is None,
                "exact_equal": left is None and right is None,
                "max_abs_difference": None,
            }
            continue
        left_array = np.asarray(left)
        right_array = np.asarray(right)
        if left_array.shape != right_array.shape:
            metrics[key] = {
                "present_in_both": True,
                "exact_equal": False,
                "max_abs_difference": None,
                "shape_equal": False,
            }
            continue
        exact = bool(np.array_equal(left_array, right_array, equal_nan=True))
        finite = np.isfinite(left_array) & np.isfinite(right_array)
        finite_pattern_equal = bool(
            np.array_equal(np.isfinite(left_array), np.isfinite(right_array))
        )
        maximum = (
            float(np.max(np.abs(left_array[finite] - right_array[finite])))
            if finite.any()
            else 0.0
        )
        if not finite_pattern_equal:
            maximum = None
        metrics[key] = {
            "present_in_both": True,
            "exact_equal": exact,
            "max_abs_difference": maximum,
            "shape_equal": True,
            "finite_pattern_equal": finite_pattern_equal,
        }
    metrics["all_exact_equal"] = all(item["exact_equal"] for item in metrics.values())
    return metrics


def _import_upstream(source_root: Path) -> tuple[Any, Any, Any, Any]:
    source_root = source_root.resolve()
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    models = importlib.import_module("vggt_omega.models")
    load_module = importlib.import_module("vggt_omega.utils.load_fn")
    pose_module = importlib.import_module("vggt_omega.utils.pose_enc")
    models_file = getattr(models, "__file__", None)
    if not isinstance(models_file, str):
        raise RuntimeError("VGGT-Omega models module has no filesystem origin")
    imported_path = Path(models_file).resolve()
    if not imported_path.is_relative_to(source_root):
        raise RuntimeError(
            f"VGGT-Omega imported outside fixed source checkout: {imported_path}"
        )
    return (
        models.VGGTOmega,
        load_module.load_and_preprocess_images,
        pose_module.encoding_to_camera,
        imported_path,
    )


def _strict_cpu_load(
    source_root: Path, checkpoint: Path
) -> tuple[Any, Any, dict[str, Any]]:
    torch = importlib.import_module("torch")

    model_class, preprocess, encoding_to_camera, imported_path = _import_upstream(
        source_root
    )
    started = time.monotonic()
    # The adapter owns the explicit bf16/fp16 context so each fresh-process
    # retry follows the preregistered precision order instead of auto-selecting.
    model = model_class(autocast=False).eval()
    state_dict = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict):
        raise TypeError(
            "VGGT-Omega checkpoint must contain a direct state_dict mapping"
        )
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict state_dict load returned missing or unexpected keys")
    facts = {
        "strict_state_dict": True,
        "state_dict_key_count": len(state_dict),
        "model_load_seconds_cpu": time.monotonic() - started,
        "imported_model_path": str(imported_path),
    }
    del state_dict
    return model, (preprocess, encoding_to_camera), facts


def validate_checkpoint_load(source_root: Path, checkpoint: Path) -> dict[str, Any]:
    """Instantiate and strict-load the fixed checkpoint on CPU only."""
    model, _, facts = _strict_cpu_load(source_root, checkpoint)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    del model
    return {
        "status": "validated_cpu_strict_load",
        "model": MODEL_NAME,
        "source_commit": SOURCE_COMMIT,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "parameter_count": parameter_count,
        "environment": runtime_environment(),
        **facts,
    }


def runtime_environment() -> dict[str, Any]:
    torch = importlib.import_module("torch")
    packages = {}
    for distribution in (
        "torch",
        "torchvision",
        "numpy",
        "opencv-python",
        "Pillow",
        "einops",
        "safetensors",
        "scipy",
        "vggt-omega",
    ):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch_cuda_build": torch.version.cuda,
        "packages": packages,
    }


def _prediction_to_numpy(
    predictions: dict[str, Any], encoding_to_camera: Any
) -> dict[str, np.ndarray]:
    extrinsics, _ = encoding_to_camera(
        predictions["pose_enc"], predictions["images"].shape[-2:]
    )
    depth = predictions["depth"].detach().float().cpu().numpy()[0, ..., 0]
    confidence = predictions["depth_conf"].detach().float().cpu().numpy()[0]
    predicted_w2c = extrinsics.detach().float().cpu().numpy()[0]
    predicted_c2w = np.stack(
        [camera_from_world_to_cam2world(item) for item in predicted_w2c]
    )
    return {
        "depth": depth.astype(np.float32),
        "confidence": confidence.astype(np.float32),
        "pred_cam2world": predicted_c2w.astype(np.float64),
    }


def _scene_inputs(
    scene_root: Path,
    manifest: dict[str, Any],
    split: dict,
) -> tuple[
    dict[int, ColmapCamera],
    dict[str, ColmapImage],
    dict[int, np.ndarray],
    dict[str, Any],
]:
    sparse_root = scene_root / "sparse" / "0"
    required = (
        scene_root / "images",
        sparse_root / "cameras.txt",
        sparse_root / "images.txt",
        sparse_root / "points3D.txt",
        sparse_root / "test.txt",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"prepared scene is incomplete: {missing}")
    cameras = read_cameras_txt(sparse_root / "cameras.txt")
    images_by_name = {
        image.name: image
        for image in read_images_txt(sparse_root / "images.txt").values()
    }
    sparse_points = read_sparse_points(sparse_root / "points3D.txt")
    training_names = set(training_names_from_split(split))
    test_names = {
        line.strip()
        for line in (sparse_root / "test.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    selected_names = {
        name for window in manifest["windows"] for name in window["images"]
    }
    if selected_names - training_names:
        raise ValueError(
            f"manifest contains non-training images: {sorted(selected_names - training_names)}"
        )
    if selected_names & test_names:
        raise ValueError(
            f"manifest overlaps sparse/0/test.txt: {sorted(selected_names & test_names)}"
        )
    transforms: dict[str, Any] = {}
    for name in sorted(selected_names):
        image = images_by_name.get(name)
        if image is None:
            raise ValueError(f"prepared scene lacks official camera for {name}")
        camera = cameras.get(image.camera_id)
        if camera is None:
            raise ValueError(f"prepared scene lacks camera ID {image.camera_id}")
        if camera.width != 1320 or camera.height != 989:
            raise ValueError(f"{name}: prepared camera must be 1320x989")
        camera_intrinsics(camera)
        image_path = scene_root / "images" / name
        if not image_path.is_file():
            raise FileNotFoundError(f"prepared scene lacks RGB image {image_path}")
        transform = balanced_preprocess_transform(camera.width, camera.height)
        if transform["crop_applied"] or transform["target_size_wh"] != [592, 448]:
            raise ValueError(
                f"{name}: unexpected fixed preprocessing transform {transform}"
            )
        transforms[name] = transform
    return (
        cameras,
        images_by_name,
        sparse_points,
        {
            "test_image_count": len(test_names),
            "selected_training_image_count": len(selected_names),
            "transforms": transforms,
        },
    )


def run_vggt_omega_windows(
    *,
    source_root: Path,
    checkpoint: Path,
    scene_root: Path,
    output_dir: Path,
    manifest: dict[str, Any],
    split: dict,
    artifact_gates: dict[str, Any],
    repeat_count: int = REPEAT_COUNT,
    amp_dtype: str = "bf16",
    experiment_id: str = "GEO-025-PF",
    run_scope: str = "smoke",
) -> dict[str, Any]:
    """Run fixed VGGT-Omega windows and emit audited official-P0 geometry."""
    required_repeats = REPEAT_COUNT if run_scope == "smoke" else 1
    if run_scope not in {"smoke", "full_training"}:
        raise ValueError("run_scope must be smoke or full_training")
    if repeat_count != required_repeats:
        raise ValueError(
            f"{run_scope} requires repeat_count={required_repeats}, got {repeat_count}"
        )
    cameras, images_by_name, sparse_points, scene_facts = _scene_inputs(
        scene_root, manifest, split
    )

    torch = importlib.import_module("torch")

    if not torch.cuda.is_available():
        raise RuntimeError("VGGT-Omega real execution requires CUDA")
    if amp_dtype not in {"bf16", "fp16"}:
        raise ValueError("amp_dtype must be bf16 or fp16")
    if amp_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The fixed bf16 attempt is unsupported on this GPU")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    wall_started = time.monotonic()
    model, upstream, load_facts = _strict_cpu_load(source_root, checkpoint)
    preprocess, encoding_to_camera = upstream
    model = model.to(device).eval()
    torch.cuda.synchronize(device)
    model_load_seconds = time.monotonic() - wall_started
    records: list[dict[str, Any]] = []
    repeatability: list[dict[str, Any]] = []
    input_shapes: list[dict[str, Any]] = []
    forward_seconds = 0.0
    if amp_dtype == "bf16":
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = torch.float16
    try:
        for window in manifest["windows"]:
            names = window["images"]
            paths = [str(scene_root / "images" / name) for name in names]
            images = preprocess(
                paths,
                mode=PREPROCESS_MODE,
                image_resolution=IMAGE_RESOLUTION,
                patch_size=PATCH_SIZE,
            )
            expected_shape = (len(names), 3, 448, 592)
            if tuple(images.shape) != expected_shape:
                raise ValueError(
                    f"{window['window_manifest_id']}: upstream preprocessing shape "
                    f"{tuple(images.shape)} != {expected_shape}"
                )
            input_shapes.append(
                {
                    "window_manifest_id": window["window_manifest_id"],
                    "model_input_shape": list(images.shape),
                    "preprocessing_transform": scene_facts["transforms"][names[0]],
                }
            )
            images = images.to(device)
            repeats: list[dict[str, np.ndarray]] = []
            repeat_seconds = []
            for repeat_index in range(repeat_count):
                torch.cuda.synchronize(device)
                forward_started = time.monotonic()
                with (
                    torch.inference_mode(),
                    torch.autocast(device_type="cuda", dtype=autocast_dtype),
                ):
                    predictions = model(images)
                torch.cuda.synchronize(device)
                elapsed = time.monotonic() - forward_started
                forward_seconds += elapsed
                repeat_seconds.append(elapsed)
                converted = _prediction_to_numpy(predictions, encoding_to_camera)
                repeats.append(converted)
                del predictions

                if repeat_index == 0:
                    for index, name in enumerate(names):
                        image = images_by_name[name]
                        camera = cameras[image.camera_id]
                        transform = scene_facts["transforms"][name]
                        official_intrinsics = scale_pinhole_intrinsics(
                            camera_intrinsics(camera), transform
                        )
                        scaled_depth, ratio, correspondence_count = (
                            scale_depth_to_sparse(
                                converted["depth"][index],
                                official_intrinsics,
                                image,
                                sparse_points,
                            )
                        )
                        confidence = converted["confidence"][index]
                        valid = valid_depth_mask(scaled_depth, confidence)
                        official_cam2world = camera_to_world(image)
                        world_points = world_points_from_official_p0(
                            scaled_depth, official_intrinsics, official_cam2world
                        )
                        output_path = (
                            output_dir
                            / "windows"
                            / window["window_manifest_id"]
                            / f"{Path(name).stem}.npz"
                        )
                        write_geometry_npz(
                            output_path,
                            depth=scaled_depth,
                            confidence=confidence,
                            valid_mask=valid,
                            world_points=world_points,
                            pinhole_intrinsics=official_intrinsics,
                            official_cam2world=official_cam2world,
                            pred_cam2world=converted["pred_cam2world"][index],
                            depth_scale_ratio=ratio,
                        )
                        records.append(
                            {
                                "window_manifest_id": window["window_manifest_id"],
                                "image_name": name,
                                "official_camera_id": image.camera_id,
                                "input_resolution": transform["target_size_wh"],
                                "coordinate_frame": "official_colmap_reference",
                                "depth_scale": {
                                    "method": "per_view_median_official_sparse_depth_ratio",
                                    "ratio": ratio,
                                    "correspondence_count": correspondence_count,
                                    "minimum_correspondences": MINIMUM_CORRESPONDENCES,
                                    "minimum_depth": MINIMUM_DEPTH,
                                },
                                "source_model": MODEL_NAME,
                                "source_commit": SOURCE_COMMIT,
                                "checkpoint_sha256": CHECKPOINT_SHA256,
                                "output_npz": str(output_path.relative_to(output_dir)),
                                "valid_rule": "finite_positive_depth_and_finite_confidence_no_threshold",
                                "valid_fraction": float(valid.mean()),
                                "finite_depth_fraction": float(
                                    np.isfinite(scaled_depth).mean()
                                ),
                                "finite_confidence_fraction": float(
                                    np.isfinite(confidence).mean()
                                ),
                                "finite_world_points_fraction": float(
                                    np.isfinite(world_points).mean()
                                ),
                                "preprocessing_transform": transform,
                                "world_point_pose": "official_cam2world_P0_only",
                                "predicted_camera_role": "diagnostic_only",
                            }
                        )
            if repeat_count == REPEAT_COUNT:
                metrics = repeatability_metrics(repeats[0], repeats[1])
                repeatability.append(
                    {
                        "window_manifest_id": window["window_manifest_id"],
                        "repeat_count": repeat_count,
                        "repeat_inference_seconds": repeat_seconds,
                        "hard_gate": True,
                        "metrics": metrics,
                    }
                )
            del repeats, images
            completed_windows = len(input_shapes)
            if completed_windows % 10 == 0 or completed_windows == len(
                manifest["windows"]
            ):
                _atomic_json(
                    output_dir / "progress.json",
                    {
                        "status": "running",
                        "experiment_id": experiment_id,
                        "run_scope": run_scope,
                        "completed_windows": completed_windows,
                        "total_windows": len(manifest["windows"]),
                        "record_count": len(records),
                        "records": records,
                    },
                )
        torch.cuda.synchronize(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
    finally:
        del model
        torch.cuda.empty_cache()

    audit = audit_geometry_outputs(output_dir, manifest, records, split)
    repeatability_all_exact = (
        all(item["metrics"]["all_exact_equal"] for item in repeatability)
        if repeatability
        else None
    )
    return {
        "schema_version": 2,
        "geometry_output_schema": manifest["geometry_output_schema"],
        "experiment_id": experiment_id,
        "run_scope": run_scope,
        "status": (
            "completed_real_smoke"
            if run_scope == "smoke"
            else "completed_full_training_geometry"
        ),
        "model_execution": "completed",
        "model": MODEL_NAME,
        "artifact_gates": artifact_gates,
        "model_load_seconds": model_load_seconds,
        "cpu_strict_load_facts": load_facts,
        "environment": runtime_environment(),
        "inference_seconds": forward_seconds,
        "wall_seconds": time.monotonic() - wall_started,
        "torch_peak_allocated_bytes": peak_allocated,
        "torch_peak_reserved_bytes": peak_reserved,
        "cuda": {
            "device_index": int(device.index or 0),
            "device_name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "amp_dtype": amp_dtype,
        },
        "determinism": {
            "seed": 0,
            "model_eval": True,
            "inference_mode": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "repeat_count_per_window": repeat_count,
            "exact_equality_is_reported_not_silently_required": True,
        },
        "scene": scene_facts,
        "model_inputs": input_shapes,
        "co_visibility": [
            {
                "window_manifest_id": window["window_manifest_id"],
                "anchor_image": window["anchor_image"],
                "images": window["images"],
                "facts": window["co_visibility_facts"],
            }
            for window in manifest["windows"]
        ],
        "repeatability": repeatability,
        "repeatability_all_exact": repeatability_all_exact,
        "records": records,
        "audit": audit,
    }
