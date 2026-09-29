#!/usr/bin/env python3
"""Render RGB175/RGB176 on the fixed 144-pose no-GT trajectory."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.render_rgb171_trajectory import temporal_metrics
from src.quality_protocol import sha256, write_new_json

MODELS = {
    "RGB175_1M": ROOT / "outputs/quality/RGB-175/full-r1/model/point_cloud/iteration_50000/point_cloud.ply",
    "RGB176_15M": ROOT / "outputs/quality/RGB-176/full-r1/model/point_cloud/iteration_50000/point_cloud.ply",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/quality/source/improvedgs-full")
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--rgb170-images", type=Path, default=ROOT / "outputs/quality/RGB-170/full-r1/trajectory-r1/50000")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/quality/QUAL-026/r1/trajectory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(args.source), str(args.source / "submodules/diff-gaussian-rasterization"),
                    str(args.source / "submodules/simple-knn"), str(args.source / "submodules/fused-ssim")]
    import torch
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from src.quality_cameras import read_camera_requests
    from src.quality_native_gaussian import make_native_view

    cameras = read_camera_requests(args.trajectory)
    names = sorted(cameras)
    if len(names) != 144 or any(not (args.rgb170_images / name).is_file() for name in names):
        raise ValueError("Expected the frozen 144 cameras and complete RGB170 renders")
    pipe = SimpleNamespace(debug=False, antialiasing=False, depth_ratio=0.0)
    background = torch.zeros(3, device="cuda")
    roots = {"RGB170_full400": args.rgb170_images}
    for arm, ply in MODELS.items():
        root = args.output / arm
        root.mkdir()
        roots[arm] = root
        model = GaussianModel(3)
        model.load_ply(str(ply))
        with torch.inference_mode():
            for name in names:
                camera = replace(cameras[name], model="PINHOLE", radial_k=0.0)
                tensor = render(make_native_view(camera), model, pipe, background,
                                track_gradients=False, inference_only=True)["render"]
                array = np.rint(tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(array).save(root / name)
        del model
        torch.cuda.empty_cache()

    differences = []
    comparison_frames = args.output / "comparison_frames"
    comparison_frames.mkdir()
    labels = (("RGB170 · full400", roots["RGB170_full400"]),
              ("RGB175 · spatial 1.0M", roots["RGB175_1M"]),
              ("RGB176 · spatial 1.5M", roots["RGB176_15M"]))
    for index, name in enumerate(names):
        arrays = {key: np.asarray(Image.open(root / name).convert("RGB"), dtype=np.float32)
                  for key, root in roots.items()}
        differences.append({
            "name": name,
            "mae_15M_vs_1M_255": float(np.abs(arrays["RGB176_15M"] - arrays["RGB175_1M"]).mean()),
            "mae_15M_vs_rgb170_255": float(np.abs(arrays["RGB176_15M"] - arrays["RGB170_full400"]).mean()),
        })
        canvas = Image.new("RGB", (1920, 504), "black")
        draw = ImageDraw.Draw(canvas)
        for column, (label, root) in enumerate(labels):
            x = 640 * column
            draw.text((x + 8, 6), f"{name} · {label}", fill="white")
            canvas.paste(Image.open(root / name).convert("RGB").resize((640, 480)), (x, 24))
        canvas.save(comparison_frames / f"frame{index:04d}.png")
    video = args.output / "rgb170-vs-rgb175-vs-rgb176-trajectory.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-framerate", "12",
        "-i", str(comparison_frames / "frame%04d.png"), "-c:v", "libx264", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video),
    ], check=True)
    temporal = {key: temporal_metrics(root, names) for key, root in roots.items()}
    write_new_json(args.output / "result.json", {
        "schema_version": 1, "status": "completed",
        "scope": "fixed 144-pose trajectory without RGB ground truth; continuity and visual stability only",
        "view_count": 144,
        "trajectory_sha256": {name: sha256(args.trajectory / name) for name in ("images.txt", "cameras.txt")},
        "model_sha256": {key: sha256(path) for key, path in MODELS.items()},
        "code_sha256": sha256(ROOT / "scripts/render_rgb175_rgb176_trajectory.py"),
        "temporal": temporal,
        "RGB176_minus_RGB175_temporal": {key: temporal["RGB176_15M"][key] - temporal["RGB175_1M"][key]
                                          for key in temporal["RGB176_15M"]},
        "mean_15M_vs_1M_mae_255": float(np.mean([row["mae_15M_vs_1M_255"] for row in differences])),
        "mean_15M_vs_rgb170_mae_255": float(np.mean([row["mae_15M_vs_rgb170_255"] for row in differences])),
        "per_view_difference": differences,
        "video": str(video.relative_to(ROOT)), "video_sha256": sha256(video),
    })


if __name__ == "__main__":
    main()
