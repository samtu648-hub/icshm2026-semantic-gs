#!/usr/bin/env python3
"""Validate RGB140's isolated AbsGS split-gradient adaptation on a frozen model."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def worker(source: Path, model_path: Path, scene_path: Path, output: Path) -> None:
    sys.path[:0] = [
        str(source),
        str(source / "submodules/diff-gaussian-rasterization"),
        str(source / "submodules/simple-knn"),
    ]
    import torch
    from arguments import ModelParams, PipelineParams
    from gaussian_renderer import render
    from scene import GaussianModel, Scene
    from utils.loss_utils import l1_loss
    from utils.image_utils import psnr

    parser = ArgumentParser()
    model_group = ModelParams(parser)
    pipe_group = PipelineParams(parser)
    args = parser.parse_args([
        "--source_path", str(scene_path),
        "--model_path", str(model_path),
        "--resolution", "1",
        "--data_device", "cpu",
    ])
    dataset = model_group.extract(args)
    pipe = pipe_group.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=30_000, shuffle=False)
    camera = sorted(scene.getTrainCameras(), key=lambda c: c.image_name)[0]
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    package = render(camera, gaussians, pipe, background, use_trained_exp=False)
    image = package["render"]
    ground_truth = camera.original_image.cuda()
    loss = l1_loss(image, ground_truth)
    loss.backward()
    screen_grad = package["viewspace_points"].grad.detach().cpu()
    payload = {
        "image_name": camera.image_name,
        "image": image.detach().cpu(),
        "screen_signed": screen_grad[:, :2],
        "xyz_grad": gaussians._xyz.grad.detach().cpu(),
        "opacity_grad": gaussians._opacity.grad.detach().cpu(),
        "loss": float(loss),
        "psnr": float(psnr(image, ground_truth).mean()),
    }
    if screen_grad.shape[1] == 4:
        payload["screen_absolute"] = screen_grad[:, 2:]
        # Independent oracle: sum native signed x/y gradients from eight
        # one-pixel backward passes, take abs before cross-pixel reduction,
        # and compare with the extra z/w channels from one combined pass.
        oracle_package = render(camera, gaussians, pipe, background, use_trained_exp=False)
        oracle_image = oracle_package["render"]
        oracle_means = oracle_package["viewspace_points"]
        pixels = [(80, 760), (100, 800), (200, 920), (1100, 790),
                  (700, 400), (550, 650), (80, 180), (900, 820)]
        oracle = torch.zeros((screen_grad.shape[0], 2), device="cuda")
        combined_seed = torch.zeros_like(oracle_image)
        for x, y in pixels:
            one = torch.zeros_like(oracle_image)
            one[:, y, x] = 1e-5
            combined_seed += one
            one_grad = torch.autograd.grad(oracle_image, oracle_means, one, retain_graph=True)[0]
            oracle += one_grad[:, :2].abs()
        combined_grad = torch.autograd.grad(oracle_image, oracle_means, combined_seed)[0]
        payload["absolute_eight_pixel_oracle_max_error"] = float((oracle - combined_grad[:, 2:]).abs().max())
    torch.save(payload, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--baseline-model", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()

    if args.worker:
        worker(args.candidate_source.resolve(), args.baseline_model.resolve(), args.scene.resolve(), args.worker_output.resolve())
        return

    args.output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    python = Path(sys.executable)
    records = {}
    for name, source in (("baseline", args.baseline_source), ("candidate", args.candidate_source)):
        artifact = args.output / f"{name}.pt"
        command = [
            str(python), str(Path(__file__).resolve()),
            "--baseline-source", str(args.baseline_source),
            "--candidate-source", str(source),
            "--baseline-model", str(args.baseline_model),
            "--scene", str(args.scene),
            "--output", str(args.output),
            "--worker", "--worker-output", str(artifact),
        ]
        completed = subprocess.run(command, env=env, text=True, capture_output=True)
        records[name] = {"command": command, "returncode": completed.returncode,
                         "stdout": completed.stdout, "stderr": completed.stderr}
        if completed.returncode:
            (args.output / "subprocesses.json").write_text(json.dumps(records, indent=2) + "\n")
            raise RuntimeError(f"{name} preflight worker failed")

    import torch
    baseline = torch.load(args.output / "baseline.pt", map_location="cpu", weights_only=False)
    candidate = torch.load(args.output / "candidate.pt", map_location="cpu", weights_only=False)
    absolute = candidate["screen_absolute"]
    signed = candidate["screen_signed"]
    finite = bool(torch.isfinite(absolute).all())
    triangle_ok = bool((absolute + 2e-5 >= signed.abs()).all())
    result = {
        "status": "passed",
        "image_name": baseline["image_name"],
        "point_count": int(signed.shape[0]),
        "forward_max_abs_diff": float((baseline["image"] - candidate["image"]).abs().max()),
        "signed_screen_grad_max_abs_diff": float((baseline["screen_signed"] - signed).abs().max()),
        "xyz_grad_max_abs_diff": float((baseline["xyz_grad"] - candidate["xyz_grad"]).abs().max()),
        "opacity_grad_max_abs_diff": float((baseline["opacity_grad"] - candidate["opacity_grad"]).abs().max()),
        "xyz_grad_relative_l2_diff": float(torch.linalg.vector_norm(baseline["xyz_grad"] - candidate["xyz_grad"]) / torch.linalg.vector_norm(baseline["xyz_grad"])),
        "xyz_grad_diff_p99_99": float(torch.quantile((baseline["xyz_grad"] - candidate["xyz_grad"]).abs().flatten(), 0.9999)),
        "absolute_gradient_finite": finite,
        "absolute_gradient_nonzero_count": int((absolute > 0).any(dim=1).sum()),
        "absolute_dominates_abs_signed_components": triangle_ok,
        "absolute_eight_pixel_oracle_max_error": candidate["absolute_eight_pixel_oracle_max_error"],
        "signed_norm_ge_0_0002": int((signed.norm(dim=1) >= 0.0002).sum()),
        "absolute_norm_ge_0_0004": int((absolute.norm(dim=1) >= 0.0004).sum()),
        "loss_baseline": baseline["loss"],
        "loss_candidate": candidate["loss"],
        "psnr_baseline": baseline["psnr"],
        "psnr_candidate": candidate["psnr"],
        "tolerances": {"forward": 0.0, "gradient_max_atol": 5e-5,
                       "xyz_relative_l2": 0.002, "oracle_atol": 2e-7},
    }
    if result["forward_max_abs_diff"] != 0.0:
        result["status"] = "failed"
    for key in ("signed_screen_grad_max_abs_diff", "xyz_grad_max_abs_diff", "opacity_grad_max_abs_diff"):
        if result[key] > 5e-5:
            result["status"] = "failed"
    if (result["xyz_grad_relative_l2_diff"] > 0.002 or
            result["absolute_eight_pixel_oracle_max_error"] > 2e-7 or
            not finite or not triangle_ok or result["absolute_gradient_nonzero_count"] == 0):
        result["status"] = "failed"
    (args.output / "subprocesses.json").write_text(json.dumps(records, indent=2) + "\n")
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
