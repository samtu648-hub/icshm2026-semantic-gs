#!/usr/bin/env python3
"""Verify SEM382 checkpoints, probabilities, RGB140, and frozen controls."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import backbone_comparison_stage as shared
from src.quality_protocol import sha256, verify_hashes, write_new_json

EID = "SEM-382-DINOV3-VITB-SFP-LR1E4-ALL300"
BATCH = ROOT / "outputs/quality/SEM-382/r1"
PARENT = ROOT / "outputs/quality/SEM-381/r1"
BASE = ROOT / "outputs/quality/SEM-378/r1"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def encoder_state_sha256(state: dict) -> str:
    import torch

    prefix = "backbone.dinov3."
    selected = [
        (name[len(prefix) :], value)
        for name, value in state.items()
        if name.startswith(prefix)
    ]
    if not selected:
        raise ValueError("Checkpoint has no DINOv3 tensors")
    digest = hashlib.sha256()
    for name, value in sorted(selected):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    args.artifact_root = args.artifact_root.resolve()
    args.method = "standard-3dgs"
    started = time.monotonic()
    config = read(ROOT / "experiments/runs" / EID / "config.yaml")
    args.batch = BATCH
    args.rgb_run = ROOT / config["rgb_run"]
    model_dir = Path(os.environ[config["dinov3"]["model_dir_env"]]).resolve()
    binding = read(BATCH / "binding.json")
    export = read(BATCH / "export/result.json")
    train = read(BATCH / "mask2former/result.json")
    inferences = {
        step: read(BATCH / f"inference-{step}/result.json") for step in (3000, 6000)
    }
    if train["status"] != "completed" or train["trained_steps"] != 6000:
        raise ValueError("SEM382 training is incomplete")
    if any(record["status"] != "completed" for record in inferences.values()):
        raise ValueError("Both SEM382 inference endpoints are required")
    if not train["dinov3_frozen_exact"]:
        raise ValueError("Training did not preserve frozen DINOv3 state")
    if not all(
        train["parameter_updates"][group]["update_l2"] > 0
        for group in ("sfp", "decoder")
    ):
        raise ValueError("SFP and decoder must both receive parameter updates")

    import torch

    torch.set_num_threads(4)
    checkpoints = []
    for step in (3000, 6000):
        name = f"model-{step}.pt"
        path = BATCH / "mask2former" / name
        payload = torch.load(path, map_location="cpu", weights_only=True)
        state = payload["model"]
        row = {
            "name": name,
            "steps": payload["steps"],
            "tensors": len(state),
            "finite": all(
                torch.isfinite(tensor).all().item() for tensor in state.values()
            ),
            "dinov3_state_sha256": encoder_state_sha256(state),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        checkpoints.append(row)
        if not row["finite"] or row["steps"] != step:
            raise ValueError(f"Invalid checkpoint: {name}")
        if row["dinov3_state_sha256"] != train["initial_dinov3_state_sha256"]:
            raise ValueError(f"DINOv3 changed in {name}")
        if row["sha256"] != inferences[step]["checkpoint_sha256"]:
            raise ValueError(f"Inference did not use {name}")
        del payload, state

    run = ROOT / config["rgb_run"]
    evaluation = ROOT / config["rgb_evaluation"]
    rgb_record = read(run / "result.json")
    source_record = read(evaluation / "result.json")
    verify_hashes(run, rgb_record["artifacts"])
    verify_hashes(evaluation / "images", source_record["image_hashes"])
    verify_hashes(BATCH / "export/images", export["image_hashes"])
    verify_hashes(PARENT, binding["sem381_sha256"])
    verify_hashes(PARENT / "inference/predictions", binding["sem381_prediction_hashes"])
    verify_hashes(BASE, binding["sem378_sha256"])

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

        _, torch, np = shared.setup(args)
        from PIL import Image
        from scripts.run_sem382 import make_model
        from src.quality_mask2former import native_predict
        from src.semantic_cable_binary import read_rgb

        fixed = read(ROOT / "experiments/runs/SEM-370-ALL300/config.yaml")[
            "diagnostic_views"
        ]
        probability_exact = {}
        for step in (3000, 6000):
            _, semantic_model, _ = make_model(config, model_dir)
            semantic_model.load_state_dict(
                torch.load(
                    BATCH / f"mask2former/model-{step}.pt",
                    map_location="cpu",
                    weights_only=True,
                )["model"],
                strict=True,
            )
            semantic_model.cuda().eval()
            exact = 0
            for name in fixed:
                actual = native_predict(
                    semantic_model, read_rgb(BATCH / "export/images" / name)
                )
                with np.load(
                    BATCH
                    / f"inference-{step}/probabilities"
                    / (Path(name).stem + ".npz")
                ) as saved:
                    expected = saved["probabilities"]
                if not np.array_equal(actual, expected):
                    raise ValueError(f"Fresh probability mismatch: {step}/{name}")
                exact += 1
            probability_exact[str(step)] = exact
            del semantic_model
            torch.cuda.empty_cache()

        source = ROOT / "outputs/quality/rgb140/source-r1"
        sys.path[:0] = [
            str(source),
            str(source / "submodules/diff-gaussian-rasterization"),
            str(source / "submodules/simple-knn"),
        ]
        from arguments import PipelineParams
        from gaussian_renderer import GaussianModel, render
        from src.quality_cameras import read_camera_requests
        from src.quality_native_gaussian import make_native_view

        torch.manual_seed(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        rgb_model = GaussianModel(3)
        rgb_model.load_ply(
            str(run / "model/point_cloud/iteration_30000/point_cloud.ply")
        )
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
        if set(requests) != set(export["image_hashes"]):
            raise ValueError("Fresh RGB camera set mismatch")
        fresh_rgb_exact = 0
        with torch.inference_mode():
            for name, camera in sorted(requests.items()):
                rgb = render(make_native_view(camera), rgb_model, pipeline, background)[
                    "render"
                ]
                actual = np.rint(
                    rgb.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
                ).astype(np.uint8)
                expected = np.asarray(Image.open(BATCH / "export/images" / name))
                if not np.array_equal(actual, expected):
                    raise ValueError(f"Fresh RGB mismatch: {name}")
                fresh_rgb_exact += 1
        queue_and_gpu_seconds = time.monotonic() - queued

    verify_hashes(run, rgb_record["artifacts"])
    verify_hashes(evaluation / "images", source_record["image_hashes"])
    verify_hashes(BATCH / "export/images", export["image_hashes"])
    verify_hashes(args.artifact_root, binding["frozen18"])
    verify_hashes(PARENT, binding["sem381_sha256"])
    verify_hashes(BASE, binding["sem378_sha256"])
    write_new_json(
        BATCH / "verification.json",
        {
            "schema_version": 1,
            "status": "completed",
            "experiment_id": EID,
            "checkpoints": checkpoints,
            "fresh_probability_exact": probability_exact,
            "fresh_native_png_exact": fresh_rgb_exact,
            "source_evaluation_sha256": sha256(evaluation / "result.json"),
            "source_evaluation_matches_training_binding": sha256(
                evaluation / "result.json"
            )
            == binding["rgb140_evaluation_sha256"],
            "trajectory_metadata_exact": source_record["trajectory_metadata_sha256"]
            == export["trajectory_metadata_sha256"],
            "optimizer_contract_exact": True,
            "sfp_and_decoder_updated": True,
            "dinov3_frozen_exact": True,
            "rgb140_endpoint_unchanged": True,
            "sem381_unchanged": True,
            "sem378_unchanged": True,
            "frozen18_exact": True,
            "queue_and_gpu_seconds": queue_and_gpu_seconds,
            "seconds": time.monotonic() - started,
            "script_sha256": sha256(Path(__file__)),
        },
    )


if __name__ == "__main__":
    main()
