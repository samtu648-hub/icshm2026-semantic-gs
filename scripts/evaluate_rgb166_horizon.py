#!/usr/bin/env python3
"""Evaluate RGB166 endpoints globally and on labeled bridge structure."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_rgb import build_lpips_model, lpips_score, psnr, ssim
from src.quality_protocol import sha256, write_new_json

ENDPOINTS = (30_000, 35_000, 40_000, 45_000, 50_000)
CLASSES = {1: "deck", 2: "stay_cable", 3: "tower", 4: "foundation"}
DIFFICULT_GROUPS = {
    "legacy_fixed": ["054.png", "061.png", "081.png", "088.png", "140.png", "145.png", "159.png", "160.png", "180.png", "183.png"],
    "bridge_159_160": ["159.png", "160.png"],
    "bridge_235_245": [f"{index:03d}.png" for index in range(235, 246)],
    "bridge_385_392": [f"{index:03d}.png" for index in range(385, 393)],
}
BRIDGE_ROIS = {
    "385.png": [0, 260, 880, 650], "386.png": [0, 260, 900, 650],
    "387.png": [0, 250, 950, 650], "388.png": [0, 250, 1000, 650],
    "389.png": [0, 250, 1080, 670], "390.png": [180, 250, 1320, 690],
    "391.png": [150, 230, 1320, 680], "392.png": [0, 220, 1180, 670],
}


def load8(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def gray01(image8: np.ndarray) -> np.ndarray:
    image = image8.astype(np.float32) / 255.0
    return 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]


def gradient(gray: np.ndarray) -> np.ndarray:
    dy, dx = np.gradient(gray)
    return np.sqrt(dx * dx + dy * dy)


def morphology(mask: np.ndarray, radius: int, mode: str) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    size = radius * 2 + 1
    filtered = image.filter(ImageFilter.MaxFilter(size) if mode == "dilate" else ImageFilter.MinFilter(size))
    return np.asarray(filtered) > 0


def boundary(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    return morphology(mask, radius, "dilate") & ~morphology(mask, radius, "erode")


def local_contrast(gray: np.ndarray) -> np.ndarray:
    blurred = np.asarray(Image.fromarray(np.rint(gray * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2)), dtype=np.float32) / 255.0
    return np.abs(gray - blurred)


def image_metrics(reference8: np.ndarray, prediction8: np.ndarray) -> dict[str, float]:
    reference = reference8.astype(np.float32) / 255.0
    prediction = prediction8.astype(np.float32) / 255.0
    return {"PSNR": psnr(reference, prediction), "SSIM": ssim(reference, prediction)}


def aggregate(values: list[dict[str, float]], keys: tuple[str, ...]) -> dict[str, float]:
    return {key: float(np.mean([value[key] for value in values])) for key in keys}


def structural_metrics(reference8: np.ndarray, prediction8: np.ndarray, semantic: np.ndarray) -> dict[str, dict[str, float]]:
    ref_gray, pred_gray = gray01(reference8), gray01(prediction8)
    ref_grad, pred_grad = gradient(ref_gray), gradient(pred_gray)
    ref_contrast, pred_contrast = local_contrast(ref_gray), local_contrast(pred_gray)
    result: dict[str, dict[str, float]] = {}
    for class_id, class_name in CLASSES.items():
        mask = semantic == class_id
        if not mask.any():
            continue
        band = boundary(mask)
        pixel_error = np.abs(prediction8.astype(np.float32) - reference8.astype(np.float32))[mask]
        item = {
            "pixels": float(mask.sum()),
            "region_mae_255": float(pixel_error.mean()),
            "region_psnr": psnr(reference8.astype(np.float32)[mask] / 255.0, prediction8.astype(np.float32)[mask] / 255.0),
            "gradient_l1": float(np.abs(ref_grad[mask] - pred_grad[mask]).mean()),
            "boundary_gradient_l1": float(np.abs(ref_grad[band] - pred_grad[band]).mean()),
            "boundary_gradient_ratio": float(pred_grad[band].mean() / max(ref_grad[band].mean(), 1e-8)),
        }
        if class_id == 2:
            visible = mask & (ref_contrast >= 3.0 / 255.0)
            item["visible_reference_pixels"] = float(visible.sum())
            item["contrast_retention_ratio"] = float(pred_contrast[visible].sum() / max(ref_contrast[visible].sum(), 1e-8))
            item["visible_support_fraction"] = float((pred_contrast[visible] >= 0.5 * ref_contrast[visible]).mean()) if visible.any() else 0.0
        result[class_name] = item
    return result


def weighted_structural_aggregate(rows: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for class_name in CLASSES.values():
        items = [row[class_name] for row in rows if class_name in row]
        if not items:
            continue
        weights = np.asarray([item["pixels"] for item in items], dtype=np.float64)
        keys = [key for key in items[0] if key != "pixels" and all(key in item for item in items)]
        output[class_name] = {"pixels": float(weights.sum())}
        for key in keys:
            if key == "visible_reference_pixels":
                output[class_name][key] = float(sum(item[key] for item in items))
            else:
                output[class_name][key] = float(np.average([item[key] for item in items], weights=weights))
    return output


def make_montage(name: str, paths: list[tuple[str, Path]], output: Path) -> None:
    crop = BRIDGE_ROIS.get(name)
    cells = []
    for label, path in paths:
        image = Image.open(path).convert("RGB")
        if crop:
            image = image.crop(tuple(crop))
        image.thumbnail((420, 300))
        cells.append((label, image.copy()))
    canvas = Image.new("RGB", (440 * len(cells), 335), "white")
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
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--parent-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    sys.path[:0] = [str(args.source), str(args.source / "submodules/diff-gaussian-rasterization"),
                    str(args.source / "submodules/simple-knn"), str(args.source / "submodules/fused-ssim")]
    import torch
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from src.quality_cameras import read_camera_requests
    from src.quality_native_gaussian import make_native_view

    cameras = read_camera_requests(args.scene / "sparse/0")
    names = sorted(path.name for path in (args.scene / "images").glob("*.png"))
    if len(names) != 400 or any(name not in cameras for name in names):
        raise ValueError("Expected 400 released official camera/image pairs")
    label_names = sorted(path.name for path in args.labels.glob("*.png"))
    if len(label_names) != 300:
        raise ValueError("Expected 300 semantic masks")
    pipe = SimpleNamespace(debug=False, antialiasing=False, depth_ratio=0.0)
    background = torch.zeros(3, device="cuda")
    started = time.monotonic()

    for iteration in ENDPOINTS:
        ply = args.model / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
        if not ply.is_file():
            raise FileNotFoundError(ply)
        model = GaussianModel(3)
        model.load_ply(str(ply))
        image_root = args.output / "images" / str(iteration)
        image_root.mkdir(parents=True)
        with torch.inference_mode():
            for index, name in enumerate(names):
                rendered = render(make_native_view(cameras[name]), model, pipe, background,
                                  track_gradients=False, inference_only=True)["render"]
                image8 = np.rint(rendered.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(image8).save(image_root / name)
                if index % 100 == 0:
                    print(json.dumps({"iteration": iteration, "rendered": index + 1}), flush=True)
        del model
        torch.cuda.empty_cache()

    labels = ["parent_30k", *[str(endpoint) for endpoint in ENDPOINTS]]
    roots = {"parent_30k": args.parent_images, **{str(endpoint): args.output / "images" / str(endpoint) for endpoint in ENDPOINTS}}
    lpips_model, lpips_torch, lpips_device = build_lpips_model("cuda")
    per_view: list[dict[str, object]] = []
    structure_rows: dict[str, list[dict[str, dict[str, float]]]] = {label: [] for label in labels}
    for name in names:
        reference8 = load8(args.scene / "images" / name)
        reference = reference8.astype(np.float32) / 255.0
        row: dict[str, object] = {"name": name}
        semantic_path = args.labels / name
        semantic = np.asarray(Image.open(semantic_path)) if semantic_path.is_file() else None
        for label in labels:
            prediction8 = load8(roots[label] / name)
            values = image_metrics(reference8, prediction8)
            values["LPIPS"] = lpips_score(reference, prediction8.astype(np.float32) / 255.0,
                                           lpips_model, lpips_torch, lpips_device)
            row[label] = values
            if semantic is not None:
                structure_rows[label].append(structural_metrics(reference8, prediction8, semantic))
        per_view.append(row)
    del lpips_model
    torch.cuda.empty_cache()

    all400 = {label: aggregate([row[label] for row in per_view], ("PSNR", "SSIM", "LPIPS")) for label in labels}
    structures = {label: weighted_structural_aggregate(rows) for label, rows in structure_rows.items()}
    difficult = {}
    for group, group_names in DIFFICULT_GROUPS.items():
        selected = [row for row in per_view if row["name"] in group_names]
        difficult[group] = {label: aggregate([row[label] for row in selected], ("PSNR", "SSIM", "LPIPS")) for label in labels}
    crop_385_392 = {}
    for label in labels:
        values = []
        for name, (x0, y0, x1, y1) in BRIDGE_ROIS.items():
            values.append(image_metrics(load8(args.scene / "images" / name)[y0:y1, x0:x1],
                                        load8(roots[label] / name)[y0:y1, x0:x1]))
        crop_385_392[label] = aggregate(values, ("PSNR", "SSIM"))

    montage_names = sorted(set(sum(DIFFICULT_GROUPS.values(), [])))
    for name in montage_names:
        make_montage(name, [("photo", args.scene / "images" / name)] + [(label, roots[label] / name) for label in labels],
                     args.output / "montages" / f"{name[:-4]}.jpg")

    result = {
        "schema_version": 1,
        "status": "completed",
        "scope": "all400 fit plus all300 labeled bridge structure and fixed difficult groups; parent comparison included",
        "endpoint_ply_sha256": {str(endpoint): sha256(args.model / "point_cloud" / f"iteration_{endpoint}" / "point_cloud.ply") for endpoint in ENDPOINTS},
        "all400": all400,
        "all300_labeled_bridge_structure": structures,
        "difficult_groups": difficult,
        "bridge_385_392_manual_crop": crop_385_392,
        "per_view": per_view,
        "seconds": time.monotonic() - started,
    }
    write_new_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()
