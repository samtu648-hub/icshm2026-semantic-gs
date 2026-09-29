#!/usr/bin/env python3
"""Freeze RGB147-149 coefficient scales and build supervision-content panels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256
from src.standard_rgb_supervision import (
    gradient_error_map,
    load_structure_mask,
    masked_rgb_l1,
    multiscale_gradient_l1,
)


def tensor(path: Path, device="cuda") -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).to(device)


def gradient_magnitude(image: torch.Tensor) -> torch.Tensor:
    dx = (image[..., 1:] - image[..., :-1]).abs().mean(0)
    dy = (image[..., 1:, :] - image[..., :-1, :]).abs().mean(0)
    return 0.5 * (
        torch.nn.functional.pad(dx, (0, 1)) + torch.nn.functional.pad(dy, (0, 0, 0, 1))
    )


def heatmap(values: np.ndarray) -> Image.Image:
    scale = float(np.percentile(values, 99))
    normalized = np.clip(values / max(scale, 1e-8), 0, 1)
    rgb = np.stack((normalized, np.sqrt(normalized) * 0.65, 1.0 - normalized), axis=-1)
    return Image.fromarray(np.rint(rgb * 255).astype(np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/quality/rgb147/calibration-r2"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    source = ROOT / "outputs/quality/rgb140/source-r1"
    sys.path[:0] = [str(source), str(source / "submodules/diff-gaussian-rasterization")]
    from utils.loss_utils import l1_loss, ssim

    try:
        from fused_ssim import fused_ssim
    except ImportError:
        fused_ssim = None

    labels = ROOT / "outputs/quality/all_data/SEM-370-r1/labels"
    manifest_path = ROOT / "outputs/quality/all_data/SEM-370-r1/label_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["image_count"] != 300 or manifest["class_mapping"]["stay_cable"] != 2:
        raise ValueError("Unexpected label manifest")
    for record in manifest["records"]:
        path = ROOT / "outputs/quality/all_data/SEM-370-r1" / record["label_png"]
        if sha256(path) != record["label_png_sha256"]:
            raise ValueError(f"Label hash changed: {path}")

    config = json.loads(
        (
            ROOT / "experiments/runs/RGB-148-STANDARD-GT-REGION-RGB/config.yaml"
        ).read_text()
    )
    views = config["coefficient_calibration"]["views"]
    target_ratio = config["coefficient_calibration"]["target_image_gradient_ratio"]
    photos = ROOT / "outputs/quality/all_data/RGB-120-r2/scene/images"
    renders = ROOT / "outputs/quality/rgb140/full-r2/evaluation/images"
    rows = []
    for name in views:
        predicted = tensor(renders / name).requires_grad_(True)
        photo = tensor(photos / name)
        mask = load_structure_mask(labels / name, dilation_px=4).cuda()
        l1 = l1_loss(predicted, photo)
        similarity = (
            fused_ssim(predicted[None], photo[None])
            if fused_ssim
            else ssim(predicted, photo)
        )
        base = 0.8 * l1 + 0.2 * (1.0 - similarity)
        rgb = masked_rgb_l1(predicted, photo, mask)
        gradient = multiscale_gradient_l1(predicted, photo, mask, (1, 4))
        base_grad, rgb_grad, gradient_grad = [
            torch.autograd.grad(value, predicted, retain_graph=True)[0]
            for value in (base, rgb, gradient)
        ]
        row = {
            "view": name,
            "region_pixels": int(mask.sum()),
            "base_loss": float(base.detach()),
            "region_rgb_loss": float(rgb.detach()),
            "region_gradient_loss": float(gradient.detach()),
            "base_image_gradient_norm": float(base_grad.norm()),
            "region_rgb_image_gradient_norm": float(rgb_grad.norm()),
            "region_gradient_image_gradient_norm": float(gradient_grad.norm()),
        }
        row["base_to_region_rgb_gradient_ratio"] = (
            row["base_image_gradient_norm"] / row["region_rgb_image_gradient_norm"]
        )
        row["base_to_region_gradient_gradient_ratio"] = (
            row["base_image_gradient_norm"] / row["region_gradient_image_gradient_norm"]
        )
        rows.append(row)

    median_base = float(np.median([row["base_image_gradient_norm"] for row in rows]))
    median_rgb = float(
        np.median([row["region_rgb_image_gradient_norm"] for row in rows])
    )
    median_gradient = float(
        np.median([row["region_gradient_image_gradient_norm"] for row in rows])
    )
    lambda_rgb = target_ratio * median_base / median_rgb
    lambda_gradient = target_ratio * median_base / median_gradient

    # A constant offset has zero true image-gradient discrepancy. Masking first
    # creates a false contour and is retained here only as a negative control.
    synthetic_mask = torch.zeros(1, 32, 32, device="cuda")
    synthetic_mask[:, :, 16:] = 1
    synthetic_render = torch.full((3, 32, 32), 0.4, device="cuda")
    synthetic_photo = torch.full((3, 32, 32), 0.6, device="cuda")
    correct = multiscale_gradient_l1(
        synthetic_render, synthetic_photo, synthetic_mask, (1,)
    )
    wrong = multiscale_gradient_l1(
        synthetic_render * synthetic_mask,
        synthetic_photo * synthetic_mask,
        torch.ones_like(synthetic_mask),
        (1,),
    )

    fixed = json.loads(
        (ROOT / "experiments/runs/RGB-132-THREE-BACKBONE-DIAG/config.yaml").read_text()
    )["views"]
    panels = args.output / "panels"
    panels.mkdir()
    panel_records = []
    for name in fixed:
        photo_image = tensor(photos / name, device="cpu")
        render_image = tensor(renders / name, device="cpu")
        mask = load_structure_mask(labels / name, 4)
        photo_grad = gradient_magnitude(photo_image)
        render_grad = gradient_magnitude(render_image)
        rgb_error = (photo_image - render_image).abs().mean(0)
        grad_error = 0.5 * (
            gradient_error_map(render_image, photo_image, 1)
            + gradient_error_map(render_image, photo_image, 4)
        )
        overlay = np.asarray(Image.open(photos / name).convert("RGB")).copy()
        region = mask[0].numpy().astype(bool)
        overlay[region] = np.rint(
            0.55 * overlay[region] + 0.45 * np.array([255, 176, 0])
        ).astype(np.uint8)
        tiles = [
            ("Photo", Image.open(photos / name).convert("RGB")),
            ("RGB140 render", Image.open(renders / name).convert("RGB")),
            ("GT structure + 4px", Image.fromarray(overlay)),
            ("Photo RGB gradient", heatmap(photo_grad.numpy())),
            ("Render RGB gradient", heatmap(render_grad.numpy())),
            ("Region RGB error", heatmap((rgb_error * mask[0]).numpy())),
            (
                "Region multiscale gradient error",
                heatmap((grad_error * mask[0]).numpy()),
            ),
        ]
        width = 440
        height = round(width * 989 / 1320)
        canvas = Image.fromarray(
            np.full((height + 28, width * len(tiles), 3), 255, dtype=np.uint8)
        )
        draw = ImageDraw.Draw(canvas)
        for index, (title, image) in enumerate(tiles):
            draw.text((index * width + 5, 6), title, fill="black")
            canvas.paste(image.resize((width, height)), (index * width, 28))
        destination = panels / name
        canvas.save(destination)
        panel_records.append(
            {
                "view": name,
                "path": str(destination.relative_to(args.output)),
                "region_pixels": int(mask.sum()),
            }
        )

    result = {
        "schema_version": 1,
        "status": "completed_pretraining_calibration",
        "method": "fixed12 frozen RGB140 quantized renders; ratio of median image-space gradient norms; target auxiliary/base=0.2",
        "target_auxiliary_image_gradient_ratio": target_ratio,
        "lambda_region_rgb": lambda_rgb,
        "lambda_region_gradient": lambda_gradient,
        "gradient_norm_medians": {
            "base": median_base,
            "region_rgb": median_rgb,
            "region_gradient": median_gradient,
            "weighted_region_rgb_to_base": lambda_rgb * median_rgb / median_base,
            "weighted_region_gradient_to_base": lambda_gradient
            * median_gradient
            / median_base,
        },
        "views": views,
        "rows": rows,
        "label_manifest_sha256": sha256(manifest_path),
        "label_hashes_verified": len(manifest["records"]),
        "synthetic_mask_edge_check": {
            "correct_full_image_first": float(correct),
            "wrong_mask_first": float(wrong),
        },
        "diagnostic_panels": panel_records,
        "quality_metrics_read": False,
    }
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "lambda_region_rgb",
                    "lambda_region_gradient",
                    "synthetic_mask_edge_check",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
