#!/usr/bin/env python3
"""Summarize bridge-focused and global evidence for the ImprovedGS pair."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256, write_new_json
from src.standard_rgb_supervision import multiscale_gradient_l1


def tensor(path: Path, roi: list[int]) -> torch.Tensor:
    x0, y0, x1, y1 = roi
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)[y0:y1, x0:x1] / 255.0
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--scene-images", type=Path, required=True)
    parser.add_argument("--rgb140-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluation = json.loads(args.evaluation.read_text())
    trajectory = json.loads(args.trajectory.read_text())
    image_roots = {
        "RGB140": args.rgb140_images,
        "RGB164": args.evaluation.parent / "images/RGB164/30000",
        "RGB165": args.evaluation.parent / "images/RGB165/30000",
    }
    gradient_rows = []
    for row in evaluation["bridge_rows"]:
        name, roi = row["name"], row["roi"]
        reference = tensor(args.scene_images / name, roi)
        mask = torch.ones((1, reference.shape[1], reference.shape[2]), dtype=torch.float32)
        values = {label: float(multiscale_gradient_l1(tensor(root / name, roi), reference, mask, (1, 4)))
                  for label, root in image_roots.items()}
        gradient_rows.append({"name": name, **values})

    names_by_group = {
        "bridge_385_392": {f"{index:03d}.png" for index in range(385, 393)},
        "secondary_235_245": {f"{index:03d}.png" for index in range(235, 246)},
    }
    group_metrics = {}
    for group, names in names_by_group.items():
        rows = [row for row in evaluation["per_view"] if row["name"] in names]
        group_metrics[group] = {
            label: {metric: float(np.mean([row[label][metric] for row in rows]))
                    for metric in ("PSNR", "SSIM", "LPIPS")}
            for label in ("RGB140", "RGB164", "RGB165")
        }

    aggregate = evaluation["aggregate"]
    bridge = evaluation["bridge_aggregate"]
    summary = {
        "schema_version": 1,
        "status": "completed",
        "evaluation_sha256": sha256(args.evaluation),
        "trajectory_sha256": sha256(args.trajectory),
        "global": aggregate,
        "global_delta_RGB165_minus_RGB140": {
            "PSNR": aggregate["RGB165"]["PSNR"] - aggregate["RGB140"]["PSNR"],
            "SSIM": aggregate["RGB165"]["SSIM"] - aggregate["RGB140"]["SSIM"],
            "LPIPS": aggregate["RGB165"]["LPIPS"] - aggregate["RGB140"]["LPIPS"],
        },
        "global_delta_RGB165_minus_RGB164": {
            "PSNR": aggregate["RGB165"]["PSNR"] - aggregate["RGB164"]["PSNR"],
            "SSIM": aggregate["RGB165"]["SSIM"] - aggregate["RGB164"]["SSIM"],
            "LPIPS": aggregate["RGB165"]["LPIPS"] - aggregate["RGB164"]["LPIPS"],
        },
        "bridge_crop": bridge,
        "bridge_crop_delta_RGB165_minus_RGB140": {
            metric: bridge["RGB165_30000"][metric] - bridge["RGB140"][metric] for metric in ("PSNR", "SSIM")
        },
        "bridge_crop_delta_RGB165_minus_RGB164": {
            metric: bridge["RGB165_30000"][metric] - bridge["RGB164_30000"][metric] for metric in ("PSNR", "SSIM")
        },
        "bridge_gradient_l1": {
            label: float(np.mean([row[label] for row in gradient_rows])) for label in image_roots
        },
        "bridge_gradient_rows": gradient_rows,
        "groups": group_metrics,
        "trajectory": {
            key: trajectory[key] for key in (
                "view_count", "mean_mae_165_vs_164_255", "mean_mae_165_vs_140_255", "mean_mae_164_vs_140_255"
            )
        },
        "conclusion": "RGB165 is the strongest bridge-fit candidate, but its incremental gain over matched RGB164 is mixed and the visible bridge/water smear remains on both training and novel views.",
    }
    write_new_json(args.output, summary)


if __name__ == "__main__":
    main()
