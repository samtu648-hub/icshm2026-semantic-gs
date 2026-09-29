#!/usr/bin/env python3
"""Render all five RGB166 endpoints on the frozen 144-camera path for RGB168 comparison."""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_rgb167_scaffold_growth20k import temporal_metrics
from src.quality_protocol import sha256, write_new_json

ENDPOINTS = (30_000, 35_000, 40_000, 45_000, 50_000)


def frame(name: str, roots: dict[str, Path]) -> Image.Image:
    cell_w, cell_h, label_h = 440, 330, 22
    canvas = Image.new("RGB", (cell_w * 3, (cell_h + label_h) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, root) in enumerate(roots.items()):
        x, y = (index % 3) * cell_w, (index // 3) * (cell_h + label_h)
        draw.text((x + 5, y + 4), f"{name} · {label}", fill="black")
        canvas.paste(Image.open(root / name).convert("RGB").resize((cell_w, cell_h)), (x, y + label_h))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
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

    cameras = read_camera_requests(args.trajectory)
    names = sorted(cameras)
    if len(names) != 144:
        raise ValueError(f"Expected 144 trajectory cameras, got {len(names)}")
    pipe = SimpleNamespace(debug=False, antialiasing=False, depth_ratio=0.0)
    background = torch.zeros(3, device="cuda")
    roots, render_stats, model_hashes = {}, {}, {}
    for endpoint in ENDPOINTS:
        ply = args.model / "point_cloud" / f"iteration_{endpoint}" / "point_cloud.ply"
        model_hashes[str(endpoint)] = sha256(ply)
        model = GaussianModel(3)
        model.load_ply(str(ply))
        directory = args.output / str(endpoint)
        directory.mkdir()
        roots[str(endpoint)] = directory
        timings = []
        with torch.inference_mode():
            for index, name in enumerate(names):
                camera = replace(cameras[name], model="PINHOLE", radial_k=0.0)
                torch.cuda.synchronize()
                mark = time.monotonic()
                tensor = render(make_native_view(camera), model, pipe, background,
                                track_gradients=False, inference_only=True)["render"]
                torch.cuda.synchronize()
                timings.append(time.monotonic() - mark)
                image8 = np.rint(tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(image8).save(directory / name)
                if index % 48 == 0:
                    print(json.dumps({"endpoint": endpoint, "rendered": index + 1}), flush=True)
        render_stats[str(endpoint)] = {
            "median_seconds_excluding_first": float(np.median(timings[1:])),
            "fps_from_median": float(1.0 / np.median(timings[1:])),
        }
        del model
        torch.cuda.empty_cache()

    temporal = {label: temporal_metrics(root, names) for label, root in roots.items()}
    contacts = args.output / "contacts"
    contacts.mkdir()
    for index in (0, 24, 48, 71, 72, 73, 96, 120, 143):
        frame(names[index], roots).save(contacts / f"{index:03d}.png")
    write_new_json(args.output / "result.json", {
        "schema_version": 1, "status": "completed_all_five_endpoint_trajectory",
        "quality_scope": "fixed 144-pose external trajectory without RGB ground truth; qualitative continuity only",
        "view_count": len(names), "trajectory_names": names,
        "trajectory_sha256": {name: sha256(args.trajectory / name) for name in ("images.txt", "cameras.txt")},
        "model_sha256": model_hashes, "temporal_descriptors": temporal, "render": render_stats,
    })


if __name__ == "__main__":
    main()
