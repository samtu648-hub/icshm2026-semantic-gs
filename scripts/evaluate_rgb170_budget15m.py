#!/usr/bin/env python3
"""Evaluate RGB170 endpoints against the frozen RGB166 50k reconstruction."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_rgb import build_lpips_model, lpips_score
from scripts.evaluate_rgb166_horizon import (
    BRIDGE_ROIS, CLASSES, DIFFICULT_GROUPS, aggregate, image_metrics, load8,
    make_montage, structural_metrics, weighted_structural_aggregate,
)
from src.quality_protocol import sha256, write_new_json

ENDPOINTS = (15_000, 20_000, 30_000, 40_000, 50_000)
BASELINE = "RGB166_50000"


def vertex_count(path: Path) -> int:
    with path.open("rb") as handle:
        while True:
            line = handle.readline().decode("ascii", errors="strict").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header" or not line:
                raise ValueError(f"PLY vertex declaration missing: {path}")


def delta(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {key: float(candidate[key] - baseline[key]) for key in candidate}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
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

    cameras = read_camera_requests(args.scene / "sparse/0")
    names = sorted(path.name for path in (args.scene / "images").glob("*.png"))
    label_names = sorted(path.name for path in args.labels.glob("*.png"))
    if len(names) != 400 or len(label_names) != 300 or not set(names) <= set(cameras):
        raise ValueError("Expected 400 official image/camera pairs and 300 labels")
    pipe = SimpleNamespace(debug=False, antialiasing=False, depth_ratio=0.0)
    background = torch.zeros(3, device="cuda")
    started = time.monotonic()
    endpoint_ply = {iteration: args.model / f"point_cloud/iteration_{iteration}/point_cloud.ply"
                    for iteration in ENDPOINTS}
    point_counts = {}
    for iteration, ply in endpoint_ply.items():
        if not ply.is_file():
            raise FileNotFoundError(ply)
        point_counts[str(iteration)] = vertex_count(ply)
        model = GaussianModel(3)
        model.load_ply(str(ply))
        image_root = args.output / "images" / str(iteration)
        image_root.mkdir(parents=True)
        with torch.inference_mode():
            for index, name in enumerate(names):
                tensor = render(make_native_view(cameras[name]), model, pipe, background,
                                track_gradients=False, inference_only=True)["render"]
                image8 = np.rint(tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(image8).save(image_root / name)
                if index % 100 == 0:
                    print(json.dumps({"iteration": iteration, "rendered": index + 1}), flush=True)
        del model
        torch.cuda.empty_cache()

    labels = [BASELINE, *[str(value) for value in ENDPOINTS]]
    roots = {BASELINE: args.baseline_images,
             **{str(value): args.output / "images" / str(value) for value in ENDPOINTS}}
    lpips_model, lpips_torch, lpips_device = build_lpips_model("cuda")
    per_view = []
    structure_rows = {label: [] for label in labels}
    for name in names:
        reference8 = load8(args.scene / "images" / name)
        reference = reference8.astype(np.float32) / 255.0
        semantic_path = args.labels / name
        semantic = np.asarray(Image.open(semantic_path)) if semantic_path.is_file() else None
        row = {"name": name}
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

    all400 = {label: aggregate([row[label] for row in per_view], ("PSNR", "SSIM", "LPIPS"))
              for label in labels}
    structures = {label: weighted_structural_aggregate(rows) for label, rows in structure_rows.items()}
    difficult = {}
    for group, group_names in DIFFICULT_GROUPS.items():
        selected = [row for row in per_view if row["name"] in group_names]
        difficult[group] = {label: aggregate([row[label] for row in selected], ("PSNR", "SSIM", "LPIPS"))
                            for label in labels}
    crop = {}
    for label in labels:
        values = []
        for name, (x0, y0, x1, y1) in BRIDGE_ROIS.items():
            values.append(image_metrics(load8(args.scene / "images" / name)[y0:y1, x0:x1],
                                        load8(roots[label] / name)[y0:y1, x0:x1]))
        crop[label] = aggregate(values, ("PSNR", "SSIM"))

    montage_names = sorted(set(sum(DIFFICULT_GROUPS.values(), [])))
    for name in montage_names:
        make_montage(name, [
            ("photo", args.scene / "images" / name),
            ("RGB166 50k", roots[BASELINE] / name),
            ("RGB170 15k", roots["15000"] / name),
            ("RGB170 20k", roots["20000"] / name),
            ("RGB170 50k", roots["50000"] / name),
        ], args.output / "montages" / f"{name[:-4]}.jpg")

    class_delta = {
        class_name: {
            key: float(structures["50000"][class_name][key] - structures[BASELINE][class_name][key])
            for key in structures["50000"][class_name]
            if key != "pixels" and key in structures[BASELINE][class_name]
        }
        for class_name in CLASSES.values()
    }
    difficult_delta = {
        group: delta(values["50000"], values[BASELINE]) for group, values in difficult.items()
    }
    result = {
        "schema_version": 1, "status": "completed",
        "scope": "all400 training fit, all300 labeled bridge structure and fixed difficult groups",
        "point_counts": point_counts,
        "endpoint_sha256": {str(key): sha256(value) for key, value in endpoint_ply.items()},
        "all400": all400, "all300_labeled_bridge_structure": structures,
        "difficult_groups": difficult, "bridge_385_392_manual_crop": crop,
        "candidate50_minus_rgb16650": {
            "all400": delta(all400["50000"], all400[BASELINE]),
            "bridge_classes": class_delta,
            "difficult_groups": difficult_delta,
            "bridge_385_392_crop": delta(crop["50000"], crop[BASELINE]),
            "per_view_psnr_better": sum(row["50000"]["PSNR"] > row[BASELINE]["PSNR"] for row in per_view),
        },
        "per_view": per_view, "seconds": time.monotonic() - started,
    }
    write_new_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()
