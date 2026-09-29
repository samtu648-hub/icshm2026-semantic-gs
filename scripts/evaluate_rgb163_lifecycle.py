#!/usr/bin/env python3
"""Render fixed bridge views from Standard checkpoints across densification."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_rgb import psnr, ssim
from src.quality_protocol import sha256, write_new_json


def laplacian_energy(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return float(np.mean(np.abs(cv2.Laplacian(gray, cv2.CV_32F))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())

    sys.path[:0] = [str(args.source), str(args.source / "submodules/diff-gaussian-rasterization"),
                    str(args.source / "submodules/simple-knn")]
    import torch
    from gaussian_renderer import render
    from scene import GaussianModel
    from src.quality_cameras import read_camera_requests
    from src.quality_native_gaussian import make_native_view

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cameras = read_camera_requests(args.scene / "sparse/0")
    groups = {key: [f"{value:03d}.png" for value in values]
              for key, values in config["diagnostic_view_groups"].items()}
    names = list(dict.fromkeys(name for values in groups.values() for name in values))
    if any(name not in cameras for name in names):
        raise ValueError("Diagnostic camera missing from all400 scene")
    pipeline = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False,
                               debug=False, antialiasing=False)
    background = torch.zeros(3, device="cuda")
    rows, summaries = [], []
    started = time.monotonic()
    reference_energy = {}
    for name in names:
        ref8 = np.asarray(Image.open(args.scene / "images" / name).convert("RGB"))
        reference_energy[name] = laplacian_energy(ref8)

    for iteration in config["save_iterations"]:
        ply = args.model / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
        if not ply.is_file():
            raise FileNotFoundError(ply)
        model = GaussianModel(3)
        model.load_ply(str(ply))
        image_dir = args.output / "images" / str(iteration)
        image_dir.mkdir(parents=True)
        with torch.inference_mode():
            for name in names:
                tensor = render(make_native_view(cameras[name]), model, pipeline, background)["render"]
                pred8 = np.rint(tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(pred8).save(image_dir / name)
                ref8 = np.asarray(Image.open(args.scene / "images" / name).convert("RGB"))
                ref = ref8.astype(np.float32) / 255.0
                pred = pred8.astype(np.float32) / 255.0
                energy = laplacian_energy(pred8)
                rows.append({"iteration": iteration, "name": name, "group": next(
                    key for key, values in groups.items() if name in values),
                    "PSNR": psnr(ref, pred), "SSIM": ssim(ref, pred),
                    "laplacian_energy": energy,
                    "laplacian_retention": energy / max(reference_energy[name], 1e-12)})
        active = [row for row in rows if row["iteration"] == iteration]
        summaries.append({"iteration": iteration, "gaussians": int(model.get_xyz.shape[0]),
                          "PSNR": float(np.mean([row["PSNR"] for row in active])),
                          "SSIM": float(np.mean([row["SSIM"] for row in active])),
                          "laplacian_retention": float(np.mean([row["laplacian_retention"] for row in active]))})
        del model
        torch.cuda.empty_cache()

    baseline_dir = ROOT / "outputs/quality/backbone_comparison/standard/export/images"
    replay = []
    for name in names:
        old = np.asarray(Image.open(baseline_dir / name).convert("RGB")).astype(np.int16)
        new = np.asarray(Image.open(args.output / "images/30000" / name).convert("RGB")).astype(np.int16)
        replay.append({"name": name, "max_abs_diff_255": int(np.abs(old - new).max()),
                       "mean_abs_diff_255": float(np.abs(old - new).mean())})

    montage_dir = args.output / "montages"
    montage_dir.mkdir()
    columns = [("照片", None)] + [(str(value), value) for value in config["save_iterations"]]
    for group, group_names in groups.items():
        thumb_w, thumb_h, label_h = 330, 247, 24
        canvas = Image.new("RGB", (thumb_w * len(columns), (thumb_h + label_h) * len(group_names)), "white")
        draw = ImageDraw.Draw(canvas)
        for row_index, name in enumerate(group_names):
            for col_index, (label, iteration) in enumerate(columns):
                y = row_index * (thumb_h + label_h)
                draw.text((col_index * thumb_w + 5, y + 4), f"{name} · {label}", fill="black")
                path = args.scene / "images" / name if iteration is None else args.output / "images" / str(iteration) / name
                image = Image.open(path).convert("RGB").resize((thumb_w, thumb_h))
                canvas.paste(image, (col_index * thumb_w, y + label_h))
        canvas.save(montage_dir / f"{group}.jpg", quality=92)

    result = {"schema_version": 1, "experiment_id": config["experiment_id"], "status": "completed",
              "config_sha256": sha256(args.config), "checkpoint_summaries": summaries,
              "per_view": rows, "groups": groups, "rgb130_final_replay": replay,
              "rgb130_final_exact": all(row["max_abs_diff_255"] == 0 for row in replay),
              "seconds": time.monotonic() - started}
    write_new_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()
