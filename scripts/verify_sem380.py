#!/usr/bin/env python3
"""Verify SEM380 checkpoints and fresh RGB148 rendering after semantic training."""

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

EID = "SEM-380-RGB148-GT-REGION-RGB-ALL300"
BATCH = ROOT / "outputs/quality/SEM-380/r2"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    args.artifact_root = args.artifact_root.resolve()
    started = time.monotonic()
    config = read(ROOT / "experiments/runs" / EID / "config.yaml")
    binding = read(BATCH / "binding.json")
    export = read(BATCH / "export/result.json")
    train = read(BATCH / "mask2former/result.json")
    inference = read(BATCH / "inference/result.json")
    assert train["status"] == inference["status"] == "completed"
    assert train["trained_steps"] == 6000

    import torch

    torch.set_num_threads(4)
    checkpoints = []
    for name in ("model-3000.pt", "model-6000.pt"):
        path = BATCH / "mask2former" / name
        payload = torch.load(path, map_location="cpu", weights_only=True)
        state = payload["model"]
        finite = all(torch.isfinite(tensor).all().item() for tensor in state.values())
        checkpoints.append(
            {
                "name": name,
                "steps": payload["steps"],
                "tensors": len(state),
                "finite": finite,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
        del payload, state
    assert all(row["finite"] for row in checkpoints)
    assert [row["steps"] for row in checkpoints] == [3000, 6000]
    assert checkpoints[1]["sha256"] == inference["checkpoint_sha256"]

    run = ROOT / config["rgb_run"]
    evaluation = ROOT / config["rgb_evaluation"]
    parent = read(run / "result.json")
    source_record = read(evaluation / "result.json")
    verify_hashes(run, parent["artifacts"])
    verify_hashes(evaluation / "images", source_record["image_hashes"])
    verify_hashes(BATCH / "export/images", export["image_hashes"])
    assert (
        source_record["trajectory_metadata_sha256"]
        == export["trajectory_metadata_sha256"]
    )

    with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
        queued = time.monotonic()
        while True:
            fcntl.flock(lock, fcntl.LOCK_EX)
            apps = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name,used_memory",
                    "--format=csv,noheader",
                ],
                text=True,
            )
            if not any(
                line.strip() and "ToDesk" not in line for line in apps.splitlines()
            ):
                break
            fcntl.flock(lock, fcntl.LOCK_UN)
            time.sleep(30)

        source = ROOT / "outputs/quality/rgb140/source-r1"
        sys.path[:0] = [
            str(source),
            str(source / "submodules/diff-gaussian-rasterization"),
            str(source / "submodules/simple-knn"),
        ]
        import numpy as np
        from PIL import Image
        from arguments import PipelineParams
        from gaussian_renderer import GaussianModel, render
        from src.quality_cameras import read_camera_requests
        from src.quality_native_gaussian import make_native_view

        torch.manual_seed(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = GaussianModel(3)
        model.load_ply(str(run / "model/point_cloud/iteration_30000/point_cloud.ply"))
        pipeline = PipelineParams(argparse.ArgumentParser()).extract(
            argparse.Namespace(
                convert_SHs_python=False,
                compute_cov3D_python=False,
                debug=False,
                antialiasing=False,
            )
        )
        background = torch.zeros(3, device="cuda")
        cameras = read_camera_requests(
            ROOT / "outputs/quality/all_data/RGB-120-r2/scene/sparse/0"
        )
        trajectory = read_camera_requests(
            args.artifact_root / "results/frozen_scaffold_sem001/flythrough_v1/cameras"
        )
        requests = {
            **cameras,
            **{
                name: replace(camera, model="PINHOLE", radial_k=0.0)
                for name, camera in trajectory.items()
            },
        }
        assert set(requests) == set(export["image_hashes"])
        exact = 0
        with torch.inference_mode():
            for name, camera in sorted(requests.items()):
                rgb = render(make_native_view(camera), model, pipeline, background)[
                    "render"
                ]
                actual = np.rint(
                    rgb.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
                ).astype(np.uint8)
                expected = np.asarray(Image.open(BATCH / "export/images" / name))
                assert np.array_equal(actual, expected), name
                exact += 1
        queue_seconds = time.monotonic() - queued

    verify_hashes(run, parent["artifacts"])
    verify_hashes(evaluation / "images", source_record["image_hashes"])
    verify_hashes(BATCH / "export/images", export["image_hashes"])
    verify_hashes(args.artifact_root, binding["frozen18"])
    verify_hashes(ROOT / config["baseline_semantic"], binding["sem378_sha256"])
    write_new_json(
        BATCH / "verification.json",
        {
            "schema_version": 1,
            "status": "completed",
            "experiment_id": EID,
            "checkpoints": checkpoints,
            "fresh_native_png_exact": exact,
            "source_evaluation_sha256": sha256(evaluation / "result.json"),
            "source_evaluation_matches_training_binding": (
                sha256(evaluation / "result.json")
                == binding["rgb148_evaluation_sha256"]
            ),
            "trajectory_metadata_exact": True,
            "rgb148_endpoint_unchanged": True,
            "sem378_unchanged": True,
            "frozen18_exact": True,
            "queue_and_render_seconds": queue_seconds,
            "seconds": time.monotonic() - started,
            "script_sha256": sha256(Path(__file__)),
        },
    )


if __name__ == "__main__":
    main()
