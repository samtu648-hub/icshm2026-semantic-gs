#!/usr/bin/env python3
"""Evaluate the ImprovedGS pair on all released photos and bridge-focused crops."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_rgb import build_lpips_model, lpips_score, psnr, ssim
from src.quality_protocol import sha256, write_new_json

BRIDGE_ROIS = {
    "385.png": [0, 260, 880, 650],
    "386.png": [0, 260, 900, 650],
    "387.png": [0, 250, 950, 650],
    "388.png": [0, 250, 1000, 650],
    "389.png": [0, 250, 1080, 670],
    "390.png": [180, 250, 1320, 690],
    "391.png": [150, 230, 1320, 680],
    "392.png": [0, 220, 1180, 670],
}
SECONDARY_NAMES = [f"{index:03d}.png" for index in range(235, 246)]


def load8(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def metrics(reference8: np.ndarray, prediction8: np.ndarray) -> dict[str, float]:
    reference = reference8.astype(np.float32) / 255.0
    prediction = prediction8.astype(np.float32) / 255.0
    return {"PSNR": psnr(reference, prediction), "SSIM": ssim(reference, prediction)}


def montage(paths: list[tuple[str, Path]], crop: list[int] | None, output: Path) -> None:
    cells = []
    for label, path in paths:
        image = Image.open(path).convert("RGB")
        if crop is not None:
            image = image.crop(tuple(crop))
        image.thumbnail((420, 300))
        cells.append((label, image.copy()))
    width = 440 * len(cells)
    height = 335
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(cells):
        x = index * 440
        draw.text((x + 8, 7), label, fill="black")
        canvas.paste(image, (x + 8, 30))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=94)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--control-model", type=Path, required=True)
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--rgb140-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    sys.path[:0] = [
        str(args.source),
        str(args.source / "submodules/diff-gaussian-rasterization"),
        str(args.source / "submodules/simple-knn"),
        str(args.source / "submodules/fused-ssim"),
    ]
    import torch
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from src.quality_cameras import read_camera_requests
    from src.quality_native_gaussian import make_native_view

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cameras = read_camera_requests(args.scene / "sparse/0")
    image_names = sorted(path.name for path in (args.scene / "images").glob("*.png"))
    if len(image_names) != 400 or any(name not in cameras for name in image_names):
        raise ValueError("Expected exactly 400 official camera/image pairs")
    pipe = SimpleNamespace(debug=False, antialiasing=False, depth_ratio=0.0)
    background = torch.zeros(3, device="cuda")
    model_paths = {"RGB164": args.control_model, "RGB165": args.candidate_model}
    started = time.monotonic()

    for label, model_root in model_paths.items():
        for iteration, names in ((15000, list(BRIDGE_ROIS) + SECONDARY_NAMES), (30000, image_names)):
            ply = model_root / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
            if not ply.is_file():
                raise FileNotFoundError(ply)
            model = GaussianModel(3)
            model.load_ply(str(ply))
            output_dir = args.output / "images" / label / str(iteration)
            output_dir.mkdir(parents=True)
            with torch.inference_mode():
                for name in names:
                    rendered = render(
                        make_native_view(cameras[name]), model, pipe, background,
                        track_gradients=False, inference_only=True,
                    )["render"]
                    image8 = np.rint(rendered.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(image8).save(output_dir / name)
            del model
            torch.cuda.empty_cache()

    lpips_model, lpips_torch, lpips_device = build_lpips_model("cuda")
    per_view = []
    for name in image_names:
        reference8 = load8(args.scene / "images" / name)
        reference = reference8.astype(np.float32) / 255.0
        row: dict[str, object] = {"name": name}
        for label in model_paths:
            prediction8 = load8(args.output / "images" / label / "30000" / name)
            values = metrics(reference8, prediction8)
            values["LPIPS"] = lpips_score(
                reference,
                prediction8.astype(np.float32) / 255.0,
                lpips_model,
                lpips_torch,
                lpips_device,
            )
            row[label] = values
        rgb140_path = args.rgb140_images / name
        if not rgb140_path.is_file():
            raise FileNotFoundError(rgb140_path)
        rgb1408 = load8(rgb140_path)
        rgb140_values = metrics(reference8, rgb1408)
        rgb140_values["LPIPS"] = lpips_score(
            reference,
            rgb1408.astype(np.float32) / 255.0,
            lpips_model,
            lpips_torch,
            lpips_device,
        )
        row["RGB140"] = rgb140_values
        per_view.append(row)
    del lpips_model
    torch.cuda.empty_cache()

    aggregate = {}
    for label in ("RGB140", "RGB164", "RGB165"):
        aggregate[label] = {
            metric: float(np.mean([row[label][metric] for row in per_view]))
            for metric in ("PSNR", "SSIM", "LPIPS")
        }

    bridge_rows = []
    for name, roi in BRIDGE_ROIS.items():
        x0, y0, x1, y1 = roi
        reference8 = load8(args.scene / "images" / name)[y0:y1, x0:x1]
        row: dict[str, object] = {"name": name, "roi": roi}
        sources = {
            "RGB140": args.rgb140_images / name,
            "RGB164_15000": args.output / "images/RGB164/15000" / name,
            "RGB164_30000": args.output / "images/RGB164/30000" / name,
            "RGB165_15000": args.output / "images/RGB165/15000" / name,
            "RGB165_30000": args.output / "images/RGB165/30000" / name,
        }
        for label, path in sources.items():
            row[label] = metrics(reference8, load8(path)[y0:y1, x0:x1])
        bridge_rows.append(row)
        montage(
            [("photo", args.scene / "images" / name)] + list(sources.items()),
            roi,
            args.output / "montages/bridge" / f"{name[:-4]}.jpg",
        )

    bridge_aggregate = {}
    for label in ("RGB140", "RGB164_15000", "RGB164_30000", "RGB165_15000", "RGB165_30000"):
        bridge_aggregate[label] = {
            metric: float(np.mean([row[label][metric] for row in bridge_rows]))
            for metric in ("PSNR", "SSIM")
        }

    for name in SECONDARY_NAMES:
        montage(
            [
                ("photo", args.scene / "images" / name),
                ("RGB140", args.rgb140_images / name),
                ("RGB164 30k", args.output / "images/RGB164/30000" / name),
                ("RGB165 30k", args.output / "images/RGB165/30000" / name),
            ],
            None,
            args.output / "montages/secondary" / f"{name[:-4]}.jpg",
        )

    result = {
        "schema_version": 1,
        "status": "completed",
        "quality_scope": "released all400 training fit; bridge crops are manually fixed diagnostics",
        "models": {label: {"path": str(path), "ply_sha256": sha256(path / "point_cloud/iteration_30000/point_cloud.ply")}
                   for label, path in model_paths.items()},
        "rgb140_images": str(args.rgb140_images),
        "aggregate": aggregate,
        "bridge_aggregate": bridge_aggregate,
        "bridge_rows": bridge_rows,
        "per_view": per_view,
        "seconds": time.monotonic() - started,
    }
    write_new_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()
