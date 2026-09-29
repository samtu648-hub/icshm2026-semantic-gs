"""Frozen Mask2Former multiscale and automatic bridge-strip inference helpers."""

from __future__ import annotations

import numpy as np
from PIL import Image


def normalize_probability(probability: np.ndarray) -> np.ndarray:
    value = np.asarray(probability, dtype=np.float32)
    if value.ndim != 3 or value.shape[0] != 5 or not value.size:
        raise ValueError("Expected CxHxW five-class probability")
    if not np.isfinite(value).all() or (value < 0).any():
        raise ValueError("Probability must be finite and nonnegative")
    total = value.sum(axis=0, keepdims=True)
    if (total <= 0).any() or not np.isfinite(total).all():
        raise ValueError("Probability has empty semantic mass")
    value = value / total
    if not np.allclose(value.sum(axis=0), 1.0, atol=2e-5):
        raise ValueError("Probability normalization failed")
    return value


def resize_rgb(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    value = np.asarray(rgb)
    if value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8:
        raise ValueError("Expected uint8 HxWx3 RGB")
    if height <= 0 or width <= 0:
        raise ValueError("Invalid resize dimensions")
    return np.asarray(
        Image.fromarray(value, mode="RGB").resize((width, height), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )


def resize_probability(probability: np.ndarray, height: int, width: int) -> np.ndarray:
    value = normalize_probability(probability)
    resized = np.stack(
        [
            np.asarray(
                Image.fromarray(channel, mode="F").resize((width, height), Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
            for channel in value
        ],
        axis=0,
    )
    return normalize_probability(resized)


def scaled_shape(height: int, width: int, scale: float) -> tuple[int, int]:
    if height <= 0 or width <= 0 or scale < 1.0:
        raise ValueError("Scale audit requires positive dimensions and scale >= 1")
    return int(round(height * scale)), int(round(width * scale))


def bridge_strip_bounds(
    native_prediction: np.ndarray,
    source_height: int,
    anchor_classes: tuple[int, ...] = (1, 2),
    percentile: float = 85.0,
    fallback_anchor_y: int = 494,
) -> dict[str, int | bool]:
    prediction = np.asarray(native_prediction)
    if prediction.ndim != 2 or prediction.size == 0 or prediction.min() < 0 or prediction.max() > 4:
        raise ValueError("Native prediction must contain class IDs 0..4")
    height = prediction.shape[0]
    if source_height <= 0 or source_height > height:
        raise ValueError("Invalid strip height")
    if not 0 <= percentile <= 100:
        raise ValueError("Invalid anchor percentile")
    support_y = np.where(np.isin(prediction, anchor_classes))[0]
    used_fallback = support_y.size == 0
    anchor = int(fallback_anchor_y) if used_fallback else int(
        np.percentile(support_y, percentile, method="nearest")
    )
    anchor = min(max(anchor, 0), height - 1)
    top = min(max(anchor - source_height // 2, 0), height - source_height)
    return {
        "anchor_y": anchor,
        "top": top,
        "bottom": top + source_height,
        "support_pixels": int(support_y.size),
        "used_fallback": used_fallback,
    }


def fixed_ensemble(probabilities: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    if not weights or set(probabilities) != set(weights):
        raise ValueError("Probability and weight arms must match exactly")
    if any(not np.isfinite(weight) or weight < 0 for weight in weights.values()):
        raise ValueError("Weights must be finite and nonnegative")
    total_weight = float(sum(weights.values()))
    if not np.isclose(total_weight, 1.0, atol=1e-8):
        raise ValueError("Fixed ensemble weights must sum to one")
    shapes = {np.asarray(value).shape for value in probabilities.values()}
    if len(shapes) != 1:
        raise ValueError("Probability arms must share a shape")
    result = sum(float(weights[name]) * normalize_probability(value) for name, value in probabilities.items())
    return normalize_probability(result)


def tiled_probability(model, rgb: np.ndarray, tile: int = 768, stride: int = 512) -> np.ndarray:
    """Run the frozen author model over an arbitrary >=tile RGB image."""
    import torch

    from src.quality_mask2former import semantic_probability
    from src.semantic2d import tile_starts

    value = np.asarray(rgb)
    if value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8:
        raise ValueError("Expected uint8 HxWx3 RGB")
    height, width = value.shape[:2]
    if height < tile or width < tile or tile <= 0 or stride <= 0:
        raise ValueError("Tiled input must be at least one registered tile")
    score = np.zeros((5, height, width), np.float32)
    count = np.zeros((height, width), np.float32)
    model.eval()
    with torch.inference_mode():
        for top in tile_starts(height, tile, stride):
            for left in tile_starts(width, tile, stride):
                crop = torch.from_numpy(
                    value[top : top + tile, left : left + tile].transpose(2, 0, 1).copy()
                ).float()
                model_input = [{"image": crop, "height": tile, "width": tile}]
                with torch.autocast("cuda", dtype=torch.float16):
                    scores = model(model_input)[0]["sem_seg"]
                if scores.device.type != "cuda":
                    raise RuntimeError("Frozen inference unexpectedly left CUDA")
                try:
                    probability = semantic_probability(scores.float().cpu().numpy())
                except ValueError as error:
                    if str(error) != "Invalid/empty semantic mass; do not fabricate background":
                        raise
                    with torch.autocast("cuda", enabled=False):
                        scores = model(model_input)[0]["sem_seg"]
                    probability = semantic_probability(scores.float().cpu().numpy())
                score[:, top : top + tile, left : left + tile] += probability
                count[top : top + tile, left : left + tile] += 1
    if (count == 0).any():
        raise ValueError("Uncovered sliding-inference pixel")
    return normalize_probability(score / count[None])


def scaled_probability(model, rgb: np.ndarray, scale: float, tile: int, stride: int) -> np.ndarray:
    height, width = rgb.shape[:2]
    scaled_height, scaled_width = scaled_shape(height, width, scale)
    scaled_rgb = resize_rgb(rgb, scaled_height, scaled_width)
    probability = tiled_probability(model, scaled_rgb, tile=tile, stride=stride)
    return resize_probability(probability, height, width)


def strip_probability(
    model,
    rgb: np.ndarray,
    native_probability: np.ndarray,
    *,
    source_height: int,
    scale: float,
    anchor_classes: tuple[int, ...],
    anchor_y_percentile: float,
    fallback_anchor_y: int,
    tile: int,
    stride: int,
) -> tuple[np.ndarray, dict[str, int | bool]]:
    height, width = rgb.shape[:2]
    native = normalize_probability(native_probability)
    if native.shape[1:] != (height, width):
        raise ValueError("Native probability and RGB shapes differ")
    bounds = bridge_strip_bounds(
        native.argmax(axis=0),
        source_height,
        anchor_classes,
        anchor_y_percentile,
        fallback_anchor_y,
    )
    top, bottom = int(bounds["top"]), int(bounds["bottom"])
    strip = rgb[top:bottom]
    scaled_height, scaled_width = scaled_shape(source_height, width, scale)
    strip_rgb = resize_rgb(strip, scaled_height, scaled_width)
    local = tiled_probability(model, strip_rgb, tile=tile, stride=stride)
    local = resize_probability(local, source_height, width)
    result = native.copy()
    result[:, top:bottom] = local
    return normalize_probability(result), bounds
