#!/usr/bin/env python3
"""Render a bridge-repair candidate on the fixed 144-pose no-GT trajectory."""
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
    adjacent = [float(np.abs(frames[index] - frames[index - 1]).mean()) for index in range(1, len(frames))]
    second = [float(np.abs(frames[index] - 2 * frames[index - 1] + frames[index - 2]).mean())
              for index in range(2, len(frames))]
    return {"mean_adjacent_l1": float(np.mean(adjacent)),
            "p95_adjacent_l1": float(np.percentile(adjacent, 95)),
            "mean_second_difference": float(np.mean(second)),
            "p95_second_difference": float(np.percentile(second, 95))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--rgb166-images", type=Path, required=True)
    parser.add_argument("--rgb170-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-key", default="RGB171_50000")
    parser.add_argument("--candidate-display", default="RGB171 50k")
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
    for root in (args.rgb166_images, args.rgb170_images):
        if any(not (root / name).is_file() for name in names):
            raise FileNotFoundError(f"Incomplete trajectory root: {root}")
    model = GaussianModel(3)
    model.load_ply(str(args.ply))
    pipe = SimpleNamespace(debug=False, antialiasing=False, depth_ratio=0.0)
    background = torch.zeros(3, device="cuda")
    candidate_root = args.output / "50000"
    candidate_root.mkdir()
    with torch.inference_mode():
        for name in names:
            camera = replace(cameras[name], model="PINHOLE", radial_k=0.0)
            tensor = render(make_native_view(camera), model, pipe, background,
                            track_gradients=False, inference_only=True)["render"]
            image8 = np.rint(tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(image8).save(candidate_root / name)
    differences = []
    for name in names:
        candidate = np.asarray(Image.open(candidate_root / name).convert("RGB"), dtype=np.float32)
        rgb166 = np.asarray(Image.open(args.rgb166_images / name).convert("RGB"), dtype=np.float32)
        rgb170 = np.asarray(Image.open(args.rgb170_images / name).convert("RGB"), dtype=np.float32)
        differences.append({"name": name, "mae_255_vs_rgb166": float(np.abs(candidate - rgb166).mean()),
                            "mae_255_vs_rgb170": float(np.abs(candidate - rgb170).mean())})
    selected = list(dict.fromkeys([names[index] for index in range(0, len(names), 12)] +
                                  [row["name"] for row in sorted(differences,
                                   key=lambda item: item["mae_255_vs_rgb170"], reverse=True)[:12]]))
    sources = (("RGB166 50k", args.rgb166_images), ("RGB170 50k", args.rgb170_images),
               (args.candidate_display, candidate_root))
    cell_w, cell_h, label_h = 330, 247, 24
    canvas = Image.new("RGB", (cell_w * len(sources), (cell_h + label_h) * len(selected)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, name in enumerate(selected):
        for col_index, (label, root) in enumerate(sources):
            y = row_index * (cell_h + label_h)
            draw.text((col_index * cell_w + 5, y + 4), f"{name} · {label}", fill="black")
            canvas.paste(Image.open(root / name).convert("RGB").resize((cell_w, cell_h)),
                         (col_index * cell_w, y + label_h))
    canvas.save(args.output / "review.jpg", quality=94)
    metrics = {"RGB166_50000": temporal_metrics(args.rgb166_images, names),
               "RGB170_50000": temporal_metrics(args.rgb170_images, names),
               args.candidate_key: temporal_metrics(candidate_root, names)}
    write_new_json(args.output / "result.json", {
        "schema_version": 1, "status": "completed", "candidate_key": args.candidate_key,
        "scope": "fixed 144-pose trajectory without RGB ground truth; continuity/stability proxies and visual review only",
        "view_count": len(names),
        "trajectory_sha256": {name: sha256(args.trajectory / name) for name in ("images.txt", "cameras.txt")},
        "model_sha256": sha256(args.ply), "temporal": metrics,
        "candidate_minus_rgb170_temporal": {key: float(metrics[args.candidate_key][key] - metrics["RGB170_50000"][key])
                                             for key in metrics[args.candidate_key]},
        "mean_candidate_vs_rgb170_mae_255": float(np.mean([row["mae_255_vs_rgb170"] for row in differences])),
        "mean_candidate_vs_rgb166_mae_255": float(np.mean([row["mae_255_vs_rgb166"] for row in differences])),
        "selected_review_frames": selected, "per_view_difference": differences,
    })


if __name__ == "__main__":
    main()
