#!/usr/bin/env python3
"""Paired all400 render benchmark for RGB166, RGB170 and a candidate."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256, write_new_json


def vertex_count(path: Path) -> int:
    with path.open("rb") as handle:
        while True:
            line = handle.readline().decode("ascii", errors="strict").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header" or not line:
                raise ValueError(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--rgb166", type=Path, required=True)
    parser.add_argument("--rgb170", type=Path, required=True)
    parser.add_argument("--rgb171", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-key", default="RGB171_50000")
    args = parser.parse_args()
    sys.path[:0] = [str(args.source), str(args.source / "submodules/diff-gaussian-rasterization"),
                    str(args.source / "submodules/simple-knn"), str(args.source / "submodules/fused-ssim")]
    import torch
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from src.quality_cameras import read_camera_requests
    from src.quality_native_gaussian import make_native_view

    cameras = read_camera_requests(args.scene / "sparse/0")
    names = sorted(path.name for path in (args.scene / "images").glob("*.png"))
    if len(names) != 400:
        raise ValueError(len(names))
    pipe = SimpleNamespace(debug=False, antialiasing=False, depth_ratio=0.0)
    background = torch.zeros(3, device="cuda")
    results = {}
    for label, ply in (("RGB166_50000", args.rgb166), ("RGB170_50000", args.rgb170),
                       (args.candidate_key, args.rgb171)):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = GaussianModel(3)
        model.load_ply(str(ply))
        samples = []
        checksum = 0.0
        with torch.inference_mode():
            for index, name in enumerate(names):
                torch.cuda.synchronize()
                started = time.perf_counter()
                tensor = render(make_native_view(cameras[name]), model, pipe, background,
                                track_gradients=False, inference_only=True)["render"]
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                if index >= 5:
                    samples.append(elapsed)
                checksum += float(tensor[0, 0, 0])
        results[label] = {
            "ply_sha256": sha256(ply), "ply_bytes": ply.stat().st_size,
            "gaussians": vertex_count(ply), "timed_views": len(samples),
            "median_seconds": float(np.median(samples)), "p90_seconds": float(np.percentile(samples, 90)),
            "fps_from_median": float(1.0 / np.median(samples)),
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "checksum": checksum,
        }
        del model
    write_new_json(args.output, {"schema_version": 1, "status": "completed", "candidate_key": args.candidate_key,
                                 "scope": "same process, all400 official cameras, first five warmup views excluded",
                                 "models": results})


if __name__ == "__main__":
    main()
