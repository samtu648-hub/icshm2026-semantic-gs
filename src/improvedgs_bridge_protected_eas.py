"""Bridge-protected reliability weighting for ImprovedGS EAS densification.

The photometric training loss is intentionally untouched.  This module only
multiplies the fixed per-image edge maps used by ImprovedGS to rank otherwise
eligible Gaussian split candidates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter


def canonical_image_name(camera: Any) -> str:
    """Return the registered PNG name for an ImprovedGS camera object."""
    name = Path(camera.image_name).name
    return name if name.endswith(".png") else name + ".png"


def build_reliability_map(
    photo_rgb: np.ndarray,
    parent_rgb: np.ndarray,
    semantic: np.ndarray | None,
    *,
    extra_protected_mask: np.ndarray | None = None,
    residual_sigma: float = 3.0,
    lower_quantile: float = 0.50,
    upper_quantile: float = 0.90,
    minimum_reliability: float = 0.25,
    bridge_dilation_pixels: int = 12,
) -> tuple[np.ndarray, dict[str, float]]:
    """Build a fixed reliability map from RGB170 residuals and bridge labels.

    Non-bridge pixels interpolate monotonically from reliability one at the
    median smoothed parent residual to ``minimum_reliability`` at the 90th
    percentile.  All label-localized bridge pixels, plus a fixed dilation for
    thin boundaries, are restored to one.  Labels therefore protect 2D bridge
    evidence but never create or move a 3D point.
    """
    if photo_rgb.shape != parent_rgb.shape or photo_rgb.ndim != 3 or photo_rgb.shape[2] != 3:
        raise ValueError("photo and parent RGB arrays must have equal HxWx3 shapes")
    if not 0.0 <= minimum_reliability <= 1.0:
        raise ValueError("minimum_reliability must lie in [0,1]")
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("residual quantiles must be ordered inside [0,1]")
    if semantic is not None and semantic.shape != photo_rgb.shape[:2]:
        raise ValueError("semantic mask shape does not match RGB")
    if extra_protected_mask is not None and extra_protected_mask.shape != photo_rgb.shape[:2]:
        raise ValueError("extra protected mask shape does not match RGB")

    photo = photo_rgb.astype(np.float32) / 255.0
    parent = parent_rgb.astype(np.float32) / 255.0
    residual = np.mean(np.abs(photo - parent), axis=2)
    smoothed = gaussian_filter(residual, sigma=float(residual_sigma), mode="reflect")

    if semantic is None:
        bridge = np.zeros(smoothed.shape, dtype=bool)
    else:
        bridge = np.asarray(semantic) > 0
    protected = binary_dilation(bridge, iterations=int(bridge_dilation_pixels)) if bridge.any() else bridge
    if extra_protected_mask is not None:
        protected = protected | np.asarray(extra_protected_mask, dtype=bool)
    reference = smoothed[~protected]
    if reference.size == 0:
        reference = smoothed.reshape(-1)
    low = float(np.quantile(reference, lower_quantile))
    high = float(np.quantile(reference, upper_quantile))
    span = max(high - low, 1e-8)
    confidence = np.clip((high - smoothed) / span, 0.0, 1.0)
    reliability = minimum_reliability + (1.0 - minimum_reliability) * confidence
    reliability[protected] = 1.0
    reliability = reliability.astype(np.float32, copy=False)
    stats = {
        "residual_q_low": low,
        "residual_q_high": high,
        "residual_mean": float(smoothed.mean()),
        "reliability_mean": float(reliability.mean()),
        "minimum_fraction": float(np.mean(reliability <= minimum_reliability + 1e-6)),
        "bridge_fraction": float(bridge.mean()),
        "protected_fraction": float(protected.mean()),
        "extra_protected_fraction": float(np.asarray(extra_protected_mask, dtype=bool).mean()) if extra_protected_mask is not None else 0.0,
        "protected_minimum_reliability": float(reliability[protected].min()) if protected.any() else 1.0,
    }
    return reliability, stats


def quantize_reliability(reliability: np.ndarray) -> np.ndarray:
    """Quantize a reliability map to an auditable lossless-size uint8 PNG."""
    return np.rint(np.clip(reliability, 0.0, 1.0) * 255.0).astype(np.uint8)


def install_bridge_protected_eas(
    *,
    training_context_module: Any,
    reliability_root: Path,
    expected_names: set[str],
) -> dict[str, Any]:
    """Patch only ImprovedGS edge-map preparation for the current process."""
    original_prepare = training_context_module.prepare_edge_maps
    state: dict[str, Any] = {
        "installed": True,
        "prepare_calls": 0,
        "prepared_views": 0,
        "minimum_weight": None,
        "maximum_weight": None,
        "mean_reliability": None,
        "original_prepare_module": getattr(original_prepare, "__module__", None),
    }

    def prepare_edge_maps(cameras: list[Any], opt: Any) -> list[Any]:
        import torch

        edges = original_prepare(cameras, opt)
        names = [canonical_image_name(camera) for camera in cameras]
        if len(names) != len(edges) or set(names) != expected_names:
            raise ValueError("reliability maps do not match the registered training cameras")
        weighted: list[Any] = []
        means: list[float] = []
        minima: list[float] = []
        maxima: list[float] = []
        for name, edge in zip(names, edges):
            path = reliability_root / name
            if not path.is_file():
                raise FileNotFoundError(path)
            reliability8 = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
            if reliability8.shape != tuple(edge.shape):
                raise ValueError(f"reliability shape mismatch for {name}: {reliability8.shape} != {tuple(edge.shape)}")
            reliability = torch.from_numpy(reliability8.copy()).to(device=edge.device, dtype=edge.dtype) / 255.0
            weighted.append(edge * reliability)
            means.append(float(reliability.mean()))
            minima.append(float(reliability.min()))
            maxima.append(float(reliability.max()))
        state.update(
            prepare_calls=int(state["prepare_calls"]) + 1,
            prepared_views=len(weighted),
            minimum_weight=min(minima),
            maximum_weight=max(maxima),
            mean_reliability=float(np.mean(means)),
        )
        return weighted

    training_context_module.prepare_edge_maps = prepare_edge_maps
    state["original_prepare"] = original_prepare
    return state
