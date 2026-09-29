"""RGB plus frozen-surface geometry input for the SEM-220 deck branch."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np

from src.geometry_prior.camera_io import ColmapCamera
from src.semantic_cable_binary import IMAGE_SHAPE, normalize_rgb
from src.semantic_deck_binary import BINARY_CLASS_NAMES, load_binary_model
from src.semantic_multiview_deck import DepthObservation
from src.semantic2d import tile_starts


GEOMETRY_CHANNELS = ("robust_log_depth", "camera_normal_x", "camera_normal_y", "camera_normal_z")


def frozen_surface_features(
    observation: DepthObservation,
    camera: ColmapCamera,
    *,
    minimum_foreground_mass: float,
) -> np.ndarray:
    """Return four label-free camera-relative geometry channels in [-1, 1]."""
    if observation.shape != IMAGE_SHAPE:
        raise ValueError("SEM-220 depth must use the native 989x1320 grid")
    if camera.model != "PINHOLE" or len(camera.params) != 4:
        raise ValueError("SEM-220 currently requires the released PINHOLE camera")
    valid = observation.valid(minimum_foreground_mass)
    if not bool(valid.any()):
        raise ValueError("SEM-220 geometry has no renderer-supported pixels")
    depth = np.asarray(observation.depth, dtype=np.float32)
    log_depth = np.zeros_like(depth)
    valid_log = np.log(np.maximum(depth[valid], np.float32(1e-6)))
    center = float(np.median(valid_log))
    low, high = np.quantile(valid_log, [0.05, 0.95]).astype(float)
    scale = max(high - low, 1e-6)
    log_depth[valid] = np.clip((valid_log - center) / scale, -1.0, 1.0)

    fx, fy, cx, cy = map(float, camera.params)
    y, x = np.indices(depth.shape, dtype=np.float32)
    point = np.stack(
        (
            ((x - np.float32(cx)) / np.float32(fx)) * depth,
            ((y - np.float32(cy)) / np.float32(fy)) * depth,
            depth,
        ),
        axis=-1,
    )
    tangent_x = np.gradient(point, axis=1)
    tangent_y = np.gradient(point, axis=0)
    normal = np.cross(tangent_x, tangent_y)
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = normal / np.maximum(norm, np.float32(1e-6))
    # Use one deterministic camera-facing orientation so that identical
    # surfaces cannot flip their three input channels across views.
    orientation = np.where(normal[..., 2:3] > 0.0, -1.0, 1.0)
    normal *= orientation
    normal[~valid] = 0.0
    features = np.concatenate((log_depth[..., None], normal), axis=-1)
    if features.shape != (*IMAGE_SHAPE, 4) or not np.isfinite(features).all():
        raise RuntimeError("SEM-220 produced invalid geometry features")
    return np.clip(features, -1.0, 1.0).astype(np.float32)


def load_rgbd_model(model_dir: Path, seed: int):
    """Load SegFormer-B2 and zero-initialize four additional input channels."""
    import torch

    model = load_binary_model(model_dir, seed)
    old = model.segformer.encoder.patch_embeddings[0].proj
    if old.in_channels != 3:
        raise ValueError("SEM-220 expected the frozen three-channel patch embedding")
    replacement = torch.nn.Conv2d(
        7,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=old.bias is not None,
        padding_mode=old.padding_mode,
    )
    with torch.no_grad():
        replacement.weight.zero_()
        replacement.weight[:, :3].copy_(old.weight)
        if old.bias is not None:
            replacement.bias.copy_(old.bias)
    model.segformer.encoder.patch_embeddings[0].proj = replacement
    model.config.num_channels = 7
    return model


def normalize_rgbd(image: np.ndarray, geometry: np.ndarray):
    """Combine frozen ImageNet-normalized RGB and bounded geometry."""
    import torch

    if geometry.shape != (*image.shape[:2], 4):
        raise ValueError("SEM-220 RGB and geometry grids do not match")
    if geometry.dtype not in (np.float16, np.float32, np.float64) or not np.isfinite(geometry).all():
        raise ValueError("SEM-220 geometry must be finite floating point")
    rgb = normalize_rgb(image)
    extra = torch.from_numpy(np.ascontiguousarray(geometry.transpose(2, 0, 1))).float()
    return torch.cat((rgb, extra), dim=0)


def random_native_rgbd_pair(
    image: np.ndarray,
    geometry: np.ndarray,
    label: np.ndarray,
    crop: int,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample one aligned native crop and horizontal flip."""
    if image.shape[:2] != label.shape or geometry.shape != (*label.shape, 4) or label.shape != IMAGE_SHAPE:
        raise ValueError("SEM-220 RGB, geometry and label grids do not match")
    height, width = label.shape
    if crop <= 0 or crop > min(height, width):
        raise ValueError("SEM-220 crop does not fit the native image")
    top = rng.randint(0, height - crop)
    left = rng.randint(0, width - crop)
    image_crop = image[top : top + crop, left : left + crop]
    geometry_crop = geometry[top : top + crop, left : left + crop]
    label_crop = label[top : top + crop, left : left + crop]
    if rng.random() < 0.5:
        image_crop = image_crop[:, ::-1]
        geometry_crop = geometry_crop[:, ::-1]
        label_crop = label_crop[:, ::-1]
        # A horizontal reflection reverses the camera-space normal x axis.
        geometry_crop = geometry_crop.copy()
        geometry_crop[..., 1] *= -1.0
    return (
        np.ascontiguousarray(image_crop),
        np.ascontiguousarray(geometry_crop),
        np.ascontiguousarray(label_crop),
    )


def sliding_rgbd_deck_probability(
    model: Any,
    image: np.ndarray,
    geometry: np.ndarray,
    *,
    crop: int,
    stride: int,
    device: Any,
    amp_dtype: Any,
) -> np.ndarray:
    """Predict one full native deck probability map from aligned RGB+geometry."""
    import torch
    import torch.nn.functional as functional

    if geometry.shape != (*image.shape[:2], 4):
        raise ValueError("SEM-220 sliding inputs do not match")
    score_sum = np.zeros(image.shape[:2], dtype=np.float32)
    count = np.zeros(image.shape[:2], dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for top in tile_starts(image.shape[0], crop, stride):
            for left in tile_starts(image.shape[1], crop, stride):
                tile = normalize_rgbd(
                    image[top : top + crop, left : left + crop],
                    geometry[top : top + crop, left : left + crop],
                ).unsqueeze(0).to(device)
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(pixel_values=tile).logits
                    logits = functional.interpolate(
                        logits, size=(crop, crop), mode="bilinear", align_corners=False
                    )
                    probability = torch.softmax(logits.float(), dim=1)[0, 1]
                score_sum[top : top + crop, left : left + crop] += probability.cpu().numpy()
                count[top : top + crop, left : left + crop] += 1.0
    if np.any(count == 0):
        raise RuntimeError("SEM-220 sliding inference left uncovered pixels")
    return score_sum / count


__all__ = [
    "BINARY_CLASS_NAMES",
    "GEOMETRY_CHANNELS",
    "frozen_surface_features",
    "load_rgbd_model",
    "normalize_rgbd",
    "random_native_rgbd_pair",
    "sliding_rgbd_deck_probability",
]
