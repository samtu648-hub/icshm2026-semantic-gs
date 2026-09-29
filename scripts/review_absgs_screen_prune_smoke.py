#!/usr/bin/env python3
"""Render and compare the fixed bridge views for a QUAL021 smoke pair."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_rgb import aggregate, evaluate
from scripts.run_absgs_screen_prune_lifecycle import arm_registration
from scripts.run_rgb152_spatial import runtime_environment
from src.quality_protocol import sha256, write_new_json


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    suite = json.loads(
        (ROOT / "experiments/runs/QUAL-021-ABSGS-SCREEN-PRUNE-FIX/config.yaml").read_text()
    )
    output = ROOT / suite["output_path"] / f"{args.run_name}-bridge-review"
    output.mkdir(parents=True, exist_ok=False)
    _, first = arm_registration(suite["arms"][0])
    scene = ROOT / first["data"]["scene"]
    source = ROOT / first["output"]["source"]
    environment, _ = runtime_environment(args.artifact_root.resolve(), source)
    os.environ.update(environment)
    sys.path[:0] = [
        str(source),
        str(source / "submodules/diff-gaussian-rasterization"),
        str(source / "submodules/simple-knn"),
    ]
    import numpy as np
    import torch
    from PIL import Image
    from arguments import PipelineParams
    from gaussian_renderer import GaussianModel, render
    from src.quality_cameras import read_camera_requests
    from src.quality_native_gaussian import make_native_view

    cameras = read_camera_requests(scene / "sparse/0")
    names = [name for name in suite["bridge_review_views"] if name in cameras]
    if len(names) != len(suite["bridge_review_views"]):
        raise ValueError("A registered bridge review camera is unavailable")
    pipeline_parser = argparse.ArgumentParser()
    pipeline = PipelineParams(pipeline_parser).extract(pipeline_parser.parse_args([]))
    background = torch.zeros(3, device="cuda")
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "starting",
        "run_name": args.run_name,
        "views": names,
        "arms": {},
    }
    with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for experiment_id in suite["arms"]:
            _, config = arm_registration(experiment_id)
            run = ROOT / config["output"]["root"] / args.run_name
            training = json.loads((run / "result.json").read_text())
            iteration = int(training["iterations"])
            images = output / experiment_id
            images.mkdir()
            model = GaussianModel(3)
            model.load_ply(
                str(run / f"model/point_cloud/iteration_{iteration}/point_cloud.ply")
            )
            with torch.inference_mode():
                for name in names:
                    rgb = render(
                        make_native_view(cameras[name]), model, pipeline, background
                    )["render"]
                    array = np.rint(
                        rgb.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
                    ).astype(np.uint8)
                    Image.fromarray(array).save(images / name)
            rows, implementation = evaluate(
                scene / "images", images, names, lpips_device="cuda", skip_lpips=True
            )
            write_csv(output / f"{experiment_id}.csv", rows)
            result["arms"][experiment_id] = {
                "point_count": int(model.get_xyz.shape[0]),
                "metrics": {
                    key: aggregate(rows, key, lower_is_worse=True)
                    for key in ("PSNR_dB", "SSIM")
                },
                "per_view": rows,
                "image_hashes": {name: sha256(images / name) for name in names},
                "implementation": implementation,
            }
            del model
            torch.cuda.empty_cache()
    result["status"] = "completed_read_only"
    write_new_json(output / "result.json", result)


if __name__ == "__main__":
    main()
