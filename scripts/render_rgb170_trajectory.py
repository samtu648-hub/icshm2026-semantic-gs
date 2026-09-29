#!/usr/bin/env python3
"""Render RGB170 20k/50k on the fixed 144-pose trajectory."""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256, write_new_json


def temporal_metrics(root: Path, names: list[str]) -> dict[str, float]:
    frames = [np.asarray(Image.open(root / name).convert("RGB"), dtype=np.float32) / 255.0 for name in names]
    adjacent = [float(np.abs(frames[i] - frames[i - 1]).mean()) for i in range(1, len(frames))]
    second = [float(np.abs(frames[i] - 2 * frames[i - 1] + frames[i - 2]).mean())
              for i in range(2, len(frames))]
    return {"mean_adjacent_l1": float(np.mean(adjacent)), "mean_second_difference": float(np.mean(second))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline-images", type=Path, required=True)
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
    endpoints = (15_000, 50_000)
    model_paths = {value: args.model / f"point_cloud/iteration_{value}/point_cloud.ply" for value in endpoints}
    for iteration, ply in model_paths.items():
        model = GaussianModel(3)
        model.load_ply(str(ply))
        root = args.output / str(iteration)
        root.mkdir()
        with torch.inference_mode():
            for name in names:
                camera = replace(cameras[name], model="PINHOLE", radial_k=0.0)
                tensor = render(make_native_view(camera), model, pipe, background,
                                track_gradients=False, inference_only=True)["render"]
                image8 = np.rint(tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(image8).save(root / name)
        del model
        torch.cuda.empty_cache()

    differences = []
    for name in names:
        baseline = np.asarray(Image.open(args.baseline_images / name).convert("RGB"), dtype=np.float32)
        candidate = np.asarray(Image.open(args.output / "50000" / name).convert("RGB"), dtype=np.float32)
        differences.append({"name": name, "mae_255": float(np.abs(candidate - baseline).mean())})
    selected = list(dict.fromkeys(
        [names[index] for index in range(0, len(names), 12)]
        + [row["name"] for row in sorted(differences, key=lambda item: item["mae_255"], reverse=True)[:12]]
    ))
    sources = (("RGB166 50k", args.baseline_images), ("RGB170 15k", args.output / "15000"),
               ("RGB170 50k", args.output / "50000"))
    cell_w, cell_h, label_h = 330, 247, 24
    canvas = Image.new("RGB", (cell_w * len(sources), (cell_h + label_h) * len(selected)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, name in enumerate(selected):
        for col_index, (label, root) in enumerate(sources):
            y = row_index * (cell_h + label_h)
            draw.text((col_index * cell_w + 5, y + 4), f"{name} · {label}", fill="black")
            image = Image.open(root / name).convert("RGB").resize((cell_w, cell_h))
            canvas.paste(image, (col_index * cell_w, y + label_h))
    canvas.save(args.output / "review.jpg", quality=94)
    write_new_json(args.output / "result.json", {
        "schema_version": 1, "status": "completed",
        "scope": "fixed 144-pose trajectory without RGB ground truth; continuity and visual stability only",
        "view_count": len(names),
        "trajectory_sha256": {name: sha256(args.trajectory / name) for name in ("images.txt", "cameras.txt")},
        "model_sha256": {str(value): sha256(path) for value, path in model_paths.items()},
        "baseline_temporal": temporal_metrics(args.baseline_images, names),
        "candidate15_temporal": temporal_metrics(args.output / "15000", names),
        "candidate50_temporal": temporal_metrics(args.output / "50000", names),
        "mean_candidate50_vs_rgb16650_mae_255": float(np.mean([row["mae_255"] for row in differences])),
        "selected_review_frames": selected, "per_view_difference": differences,
    })


if __name__ == "__main__":
    main()
