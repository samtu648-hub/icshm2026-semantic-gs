#!/usr/bin/env python3
"""Load an R1 candidate's native optimizer checkpoint into a fresh model instance."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--candidate", choices=("standard-3dgs", "mip-splatting", "scaffold-gs", "3dgs-mcmc"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-camera-count", type=int, default=0, help="Required for Scaffold-GS appearance embeddings.")
    parser.add_argument("--n-offsets", type=int, default=10, help="Scaffold-GS n_offsets used by the checkpoint.")
    return parser.parse_args()


def optimizer_params() -> object:
    import argparse as argparse_module

    arguments = importlib.import_module("arguments")
    parser = argparse_module.ArgumentParser(add_help=False)
    optimization = arguments.OptimizationParams(parser)
    return optimization.extract(parser.parse_args([]))


def main() -> int:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    sys.path.insert(0, str(args.candidate_root.resolve()))
    import torch

    gaussian_model = importlib.import_module("scene.gaussian_model").GaussianModel
    model_args, iteration = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    if args.candidate == "standard-3dgs":
        if args.train_camera_count <= 0:
            raise ValueError("--train-camera-count must be positive for Standard 3DGS exposure state")
        model = gaussian_model(3, "default")
        model._exposure = torch.nn.Parameter(
            torch.eye(3, 4, device="cuda")[None].repeat(args.train_camera_count, 1, 1).requires_grad_(True)
        )
        model.pretrained_exposures = None
    elif args.candidate in {"mip-splatting", "3dgs-mcmc"}:
        model = gaussian_model(3)
    else:
        if args.train_camera_count <= 0:
            raise ValueError("--train-camera-count must be positive for Scaffold-GS")
        model = gaussian_model(n_offsets=args.n_offsets)
        model.set_appearance(args.train_camera_count)
    model.restore(model_args, optimizer_params())
    if args.candidate == "scaffold-gs":
        parameter_count = int(model.get_anchor.shape[0])
    else:
        parameter_count = int(model.get_xyz.shape[0])
    print(
        json.dumps(
            {
                "candidate": args.candidate,
                "checkpoint_iteration": iteration,
                "model_parameter_count": parameter_count,
                "cuda_available": torch.cuda.is_available(),
                "passes": parameter_count > 0 and torch.cuda.is_available(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
