#!/usr/bin/env python3
"""Render RGB166 30k/40k/50k on the fixed 144-pose novel trajectory."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
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

    cameras = read_camera_requests(args.trajectory)
    names = sorted(cameras)
    if len(names) != 144:
        raise ValueError(f"Expected 144 trajectory cameras, got {len(names)}")
    pipe = SimpleNamespace(debug=False, antialiasing=False, depth_ratio=0.0)
    background = torch.zeros(3, device="cuda")
    endpoint_paths = {
        label: args.model / "point_cloud" / f"iteration_{label}" / "point_cloud.ply"
        for label in (30_000, 40_000, 50_000)
    }
    for iteration, ply in endpoint_paths.items():
        model = GaussianModel(3)
        model.load_ply(str(ply))
        directory = args.output / str(iteration)
        directory.mkdir()
        with torch.inference_mode():
            for name in names:
                camera = replace(cameras[name], model="PINHOLE", radial_k=0.0)
                tensor = render(make_native_view(camera), model, pipe, background,
                                track_gradients=False, inference_only=True)["render"]
                image8 = np.rint(tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(image8).save(directory / name)
        del model
        torch.cuda.empty_cache()

    rows = []
    for name in names:
        images = {
            "parent": np.asarray(Image.open(args.parent_images / name).convert("RGB"), dtype=np.int16),
            **{str(endpoint): np.asarray(Image.open(args.output / str(endpoint) / name).convert("RGB"), dtype=np.int16)
               for endpoint in endpoint_paths},
        }
        rows.append({
            "name": name,
            "mae_30k_vs_parent_255": float(np.abs(images["30000"] - images["parent"]).mean()),
            "mae_40k_vs_30k_255": float(np.abs(images["40000"] - images["30000"]).mean()),
            "mae_50k_vs_40k_255": float(np.abs(images["50000"] - images["40000"]).mean()),
            "mae_50k_vs_parent_255": float(np.abs(images["50000"] - images["parent"]).mean()),
        })

    selected = list(dict.fromkeys(
        [names[index] for index in range(0, len(names), 12)]
        + [row["name"] for row in sorted(rows, key=lambda item: item["mae_50k_vs_40k_255"], reverse=True)[:12]]
    ))
    sources = (("RGB165 parent 30k", args.parent_images), ("RGB166 30k", args.output / "30000"),
               ("RGB166 40k", args.output / "40000"), ("RGB166 50k", args.output / "50000"))
    cell_w, cell_h, label_h = 330, 247, 24
    canvas = Image.new("RGB", (cell_w * len(sources), (cell_h + label_h) * len(selected)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, name in enumerate(selected):
        for col_index, (label, directory) in enumerate(sources):
            y = row_index * (cell_h + label_h)
            draw.text((col_index * cell_w + 5, y + 4), f"{name} · {label}", fill="black")
            image = Image.open(directory / name).convert("RGB").resize((cell_w, cell_h))
            canvas.paste(image, (col_index * cell_w, y + label_h))
    canvas.save(args.output / "review.jpg", quality=94)
    write_new_json(args.output / "result.json", {
        "schema_version": 1,
        "status": "completed",
        "quality_scope": "fixed 144-pose novel trajectory without RGB ground truth; qualitative stability only",
        "view_count": len(names),
        "trajectory_sha256": {name: sha256(args.trajectory / name) for name in ("images.txt", "cameras.txt")},
        "model_sha256": {str(endpoint): sha256(path) for endpoint, path in endpoint_paths.items()},
        "mean_mae_30k_vs_parent_255": float(np.mean([row["mae_30k_vs_parent_255"] for row in rows])),
        "mean_mae_40k_vs_30k_255": float(np.mean([row["mae_40k_vs_30k_255"] for row in rows])),
        "mean_mae_50k_vs_40k_255": float(np.mean([row["mae_50k_vs_40k_255"] for row in rows])),
        "mean_mae_50k_vs_parent_255": float(np.mean([row["mae_50k_vs_parent_255"] for row in rows])),
        "selected_review_frames": selected,
        "per_view": rows,
    })


if __name__ == "__main__":
    main()
