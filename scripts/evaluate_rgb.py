#!/usr/bin/env python3
"""Compute the frozen local RGB metrics for an R1 candidate.

This intentionally does not claim to reproduce the organizers' still-unreleased
evaluation. It makes every local backbone comparison use one explicit decoder,
metric implementation and aggregation policy.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


LPIPS_PACKAGE = "lpips==0.1.4"
LPIPS_NET = "alex"
LPIPS_VERSION = "0.1"
SSIM_WINDOW_SIZE = 11
SSIM_SIGMA = 1.5


def load_rgb(path: Path) -> np.ndarray:
    """Return an image as an HWC float32 RGB array in [0, 1]."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def require_equal_shape(reference: np.ndarray, prediction: np.ndarray, name: str) -> None:
    if reference.shape != prediction.shape:
        raise ValueError(
            f"{name}: reference shape {reference.shape} does not match prediction shape {prediction.shape}"
        )


def psnr(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Per-view RGB PSNR with the fixed [0, 1] value range."""
    require_equal_shape(reference, prediction, "PSNR")
    mse = float(np.mean((reference - prediction) ** 2, dtype=np.float64))
    return math.inf if mse == 0.0 else -10.0 * math.log10(mse)


def ssim(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Per-view RGB SSIM with the documented Gaussian-window implementation."""
    require_equal_shape(reference, prediction, "SSIM")
    if reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError(f"SSIM expects HWC RGB input, got {reference.shape}")
    gaussian = cv2.getGaussianKernel(SSIM_WINDOW_SIZE, SSIM_SIGMA)
    kernel = gaussian @ gaussian.T
    border = cv2.BORDER_REFLECT_101
    mu_ref = cv2.filter2D(reference, -1, kernel, borderType=border)
    mu_pred = cv2.filter2D(prediction, -1, kernel, borderType=border)
    sigma_ref = cv2.filter2D(reference * reference, -1, kernel, borderType=border) - mu_ref * mu_ref
    sigma_pred = cv2.filter2D(prediction * prediction, -1, kernel, borderType=border) - mu_pred * mu_pred
    covariance = cv2.filter2D(reference * prediction, -1, kernel, borderType=border) - mu_ref * mu_pred
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * mu_ref * mu_pred + c1) * (2.0 * covariance + c2)
    denominator = (mu_ref * mu_ref + mu_pred * mu_pred + c1) * (sigma_ref + sigma_pred + c2)
    return float(np.mean(numerator / denominator, dtype=np.float64))


def build_lpips_model(device: str) -> tuple[Any, Any, str]:
    """Build the fixed LPIPS model only when a real LPIPS score is requested."""
    try:
        import lpips
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"LPIPS is required for a real R1 comparison. Install {LPIPS_PACKAGE} in the candidate environment."
        ) from exc
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--lpips-device cuda was requested but PyTorch cannot access a CUDA device")
    model = lpips.LPIPS(net=LPIPS_NET, version=LPIPS_VERSION).to(device).eval()
    return model, torch, device


def lpips_score(reference: np.ndarray, prediction: np.ndarray, model: Any, torch: Any, device: str) -> float:
    """Compute the fixed AlexNet LPIPS score for one image pair."""
    require_equal_shape(reference, prediction, "LPIPS")
    reference_tensor = torch.from_numpy(np.ascontiguousarray(reference.transpose(2, 0, 1))).unsqueeze(0)
    prediction_tensor = torch.from_numpy(np.ascontiguousarray(prediction.transpose(2, 0, 1))).unsqueeze(0)
    reference_tensor = (reference_tensor * 2.0 - 1.0).to(device)
    prediction_tensor = (prediction_tensor * 2.0 - 1.0).to(device)
    with torch.inference_mode():
        return float(model(reference_tensor, prediction_tensor).item())


def validation_view_names(split_path: Path) -> list[str]:
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    try:
        names = payload["primary_fold"]["validation_labeled"]
    except KeyError as exc:
        raise ValueError(f"{split_path} does not contain primary_fold.validation_labeled") from exc
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError(f"{split_path} has an invalid primary_fold.validation_labeled list")
    return names


def finite_or_string(value: float) -> float | str:
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if math.isnan(value):
        return "nan"
    return value


def aggregate(rows: list[dict[str, float | str]], key: str, lower_is_worse: bool) -> dict[str, Any]:
    values = [float(row[key]) for row in rows]
    worst_count = max(1, math.ceil(len(values) * 0.10))
    ordered = sorted(rows, key=lambda row: float(row[key]), reverse=not lower_is_worse)
    worst_rows = ordered[:worst_count]
    return {
        "mean": finite_or_string(float(np.mean(values))),
        "median": finite_or_string(float(np.median(values))),
        "worst_10_percent_mean": finite_or_string(float(np.mean([float(row[key]) for row in worst_rows]))),
        "worst_10_percent_views": [str(row["view"]) for row in worst_rows],
    }


def evaluate(
    reference_dir: Path,
    prediction_dir: Path,
    views: list[str],
    lpips_device: str,
    skip_lpips: bool,
) -> tuple[list[dict[str, float | str]], dict[str, Any]]:
    """Evaluate all requested views and return per-view rows plus reproducibility metadata."""
    model = torch = device = None
    if not skip_lpips:
        model, torch, device = build_lpips_model(lpips_device)
    rows: list[dict[str, float | str]] = []
    for name in views:
        reference_path = reference_dir / name
        prediction_path = prediction_dir / name
        if not reference_path.is_file():
            raise FileNotFoundError(f"Missing reference image: {reference_path}")
        if not prediction_path.is_file():
            raise FileNotFoundError(f"Missing prediction image: {prediction_path}")
        reference = load_rgb(reference_path)
        prediction = load_rgb(prediction_path)
        require_equal_shape(reference, prediction, name)
        row: dict[str, float | str] = {
            "view": name,
            "PSNR_dB": psnr(reference, prediction),
            "SSIM": ssim(reference, prediction),
        }
        if model is not None and torch is not None and device is not None:
            row["LPIPS_Alex"] = lpips_score(reference, prediction, model, torch, device)
        rows.append(row)
    metadata: dict[str, Any] = {
        "image_decode": "Pillow RGB float32 / 255; no resize, crop, or mask",
        "PSNR": "per-view full-image RGB MSE in [0,1]",
        "SSIM": {
            "window": SSIM_WINDOW_SIZE,
            "sigma": SSIM_SIGMA,
            "C1": 0.01**2,
            "C2": 0.03**2,
            "border": "BORDER_REFLECT_101",
        },
        "LPIPS": None
        if skip_lpips
        else {
            "package": LPIPS_PACKAGE,
            "installed_version": importlib.metadata.version("lpips"),
            "net": LPIPS_NET,
            "version": LPIPS_VERSION,
            "device": device,
            "input_range": "[-1,1]",
        },
    }
    return rows, metadata


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["view", "PSNR_dB", "SSIM", "LPIPS_Alex"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True, help="Frozen R1 split JSON.")
    parser.add_argument("--reference-dir", type=Path, required=True, help="Prepared PINHOLE validation RGB directory.")
    parser.add_argument("--prediction-dir", type=Path, required=True, help="Candidate renders named like the validation views.")
    parser.add_argument("--per-view-csv", type=Path, required=True, help="Output path for the per-view metric CSV.")
    parser.add_argument("--summary-json", type=Path, required=True, help="Output path for the aggregate metric JSON.")
    parser.add_argument("--lpips-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--skip-lpips", action="store_true", help="Self-test only; forbidden for an R1 ranking or RGB-060.")
    args = parser.parse_args()

    views = validation_view_names(args.split)
    rows, implementation = evaluate(
        args.reference_dir, args.prediction_dir, views, args.lpips_device, args.skip_lpips
    )
    write_csv(args.per_view_csv, rows)
    metrics: dict[str, Any] = {
        "PSNR_dB": aggregate(rows, "PSNR_dB", lower_is_worse=True),
        "SSIM": aggregate(rows, "SSIM", lower_is_worse=True),
    }
    if not args.skip_lpips:
        metrics["LPIPS_Alex"] = aggregate(rows, "LPIPS_Alex", lower_is_worse=False)
    summary = {
        "schema_version": 1,
        "split": str(args.split),
        "views": views,
        "view_count": len(views),
        "implementation": implementation,
        "metrics": metrics,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"view_count": len(views), "metrics": metrics}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
