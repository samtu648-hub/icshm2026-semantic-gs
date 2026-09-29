#!/usr/bin/env python3
"""Fresh-process finite reload and deterministic render audit for RGB171 preflight."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quality_protocol import sha256, write_new_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path[:0] = [
        str(args.source), str(args.source / "submodules/diff-gaussian-rasterization"),
        str(args.source / "submodules/simple-knn"), str(args.source / "submodules/fused-ssim"),
    ]
    import numpy as np
    import torch
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from src.quality_cameras import read_camera_requests
    from src.quality_native_gaussian import make_native_view

    model = GaussianModel(3)
    model.load_ply(str(args.ply))
    names = ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation")
    finite = {name: bool(torch.isfinite(getattr(model, name)).all()) for name in names}
    camera = read_camera_requests(args.scene / "sparse/0")["159.png"]
    pipe = SimpleNamespace(debug=False, antialiasing=False, depth_ratio=0.0)
    background = torch.zeros(3, device="cuda")
    view = make_native_view(camera)
    with torch.inference_mode():
        first = render(view, model, pipe, background, track_gradients=False, inference_only=True)["render"]
        second = render(view, model, pipe, background, track_gradients=False, inference_only=True)["render"]
    result = {
        "schema_version": 1,
        "status": "passed" if all(finite.values()) and torch.equal(first, second) else "failed",
        "ply_sha256": sha256(args.ply),
        "gaussians": int(model.get_xyz.shape[0]),
        "finite": finite,
        "render_exact_repeat": bool(torch.equal(first, second)),
        "render_min": float(first.min()),
        "render_max": float(first.max()),
        "render_mean": float(first.mean()),
        "render_uint8_sha256": __import__("hashlib").sha256(
            np.rint(first.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).tobytes()
        ).hexdigest(),
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }
    write_new_json(args.output, result)
    if result["status"] != "passed":
        raise RuntimeError(result)


if __name__ == "__main__":
    main()
