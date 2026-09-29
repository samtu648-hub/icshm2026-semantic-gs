#!/usr/bin/env python3
"""Fresh RGB140 renderer reload after SEM379 training; require all 544 PNGs exact."""
from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quality_protocol import sha256, verify_hashes, write_new_json


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    args.artifact_root = args.artifact_root.resolve()
    started = time.monotonic()
    batch = ROOT / "outputs/quality/SEM-379/r1"
    run = ROOT / "outputs/quality/rgb140/full-r2"
    evaluation = run / "evaluation"
    parent = read(run / "result.json")
    source_record = read(evaluation / "result.json")
    export = read(batch / "export/result.json")
    assert read(batch / "mask2former/result.json")["status"] == "completed"
    assert read(batch / "inference/result.json")["status"] == "completed"
    verify_hashes(run, parent["artifacts"])
    verify_hashes(batch / "export/images", export["image_hashes"])
    with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            text=True,
        )
        if any(line.strip() and "ToDesk" not in line for line in apps.splitlines()):
            raise RuntimeError("Other GPU compute process active")
        source = ROOT / "outputs/quality/rgb140/source-r1"
        sys.path[:0] = [str(source), str(source / "submodules/diff-gaussian-rasterization"),
                        str(source / "submodules/simple-knn")]
        import numpy as np
        import torch
        from PIL import Image
        from arguments import PipelineParams
        from gaussian_renderer import GaussianModel, render
        from src.quality_cameras import read_camera_requests
        from src.quality_native_gaussian import make_native_view

        torch.set_num_threads(4)
        torch.manual_seed(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = GaussianModel(3)
        model.load_ply(str(run / "model/point_cloud/iteration_30000/point_cloud.ply"))
        pipeline = PipelineParams(argparse.ArgumentParser()).extract(
            argparse.Namespace(convert_SHs_python=False, compute_cov3D_python=False,
                               debug=False, antialiasing=False))
        background = torch.zeros(3, device="cuda")
        cameras = read_camera_requests(ROOT / "outputs/quality/all_data/RGB-120-r2/scene/sparse/0")
        trajectory_root = args.artifact_root / "results/frozen_scaffold_sem001/flythrough_v1/cameras"
        trajectory = read_camera_requests(trajectory_root)
        requests = {**cameras, **{name: replace(camera, model="PINHOLE", radial_k=0.)
                                  for name, camera in trajectory.items()}}
        assert set(requests) == set(export["image_hashes"])
        exact = 0
        with torch.inference_mode():
            for name, camera in sorted(requests.items()):
                rgb = render(make_native_view(camera), model, pipeline, background)["render"]
                actual = np.rint(rgb.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                expected = np.asarray(Image.open(batch / "export/images" / name))
                assert np.array_equal(actual, expected), name
                exact += 1
        verify_hashes(run, parent["artifacts"])
        verify_hashes(batch / "export/images", export["image_hashes"])
    write_new_json(batch / "fresh_rgb_audit.json", {
        "schema_version": 1, "status": "completed",
        "experiment_id": "SEM-379-CABLE-REGION-SAMPLING",
        "fresh_native_png_exact": exact,
        "source_evaluation_sha256": sha256(evaluation / "result.json"),
        "source_evaluation_matches_training_binding": sha256(evaluation / "result.json") == export["source_evaluation_sha256"],
        "trajectory_metadata_exact": source_record["trajectory_metadata_sha256"] == export["trajectory_metadata_sha256"],
        "rgb140_endpoint_unchanged": True,
        "seconds": time.monotonic() - started, "script_sha256": sha256(Path(__file__)),
    })


if __name__ == "__main__":
    main()
