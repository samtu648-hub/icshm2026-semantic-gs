#!/usr/bin/env python3
"""Run a tiny CUDA/import check for the RGB151 long-axis split hook."""

from __future__ import annotations

import argparse
import inspect
import math
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quality_protocol import write_new_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.evidence.exists():
        raise FileExistsError(f"Evidence already exists: {args.evidence}")

    started = time.monotonic()
    result: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": "RGB-151-STANDARD-ABSGRAD-LAS-DEV30",
        "status": "starting",
        "seed": args.seed,
        "command": sys.argv,
    }
    try:
        sys.path.insert(0, str(args.source))
        import torch  # type: ignore[import-not-found]
        import scene  # type: ignore[import-not-found]
        from utils.general_utils import build_rotation  # type: ignore[import-not-found]

        from src.standard_long_axis_split import (
            install,
            long_axis_children,
            projected_covariance,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the RGB151 GPU preflight")
        original = scene.GaussianModel.densify_and_split
        signature = str(inspect.signature(original))
        state = install(
            scene.GaussianModel,
            enabled=True,
            split_distance=0.45,
            opacity_multiplier=0.6,
            preserve_native_rng_draw=True,
        )
        if scene.GaussianModel.densify_and_split is original:
            raise RuntimeError("Long-axis hook was not installed")

        device = torch.device("cuda")
        xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], device=device)
        scales = torch.tensor([[4.0, 1.0, 0.5], [1.0, 4.0, 2.0]], device=device)
        half = math.sqrt(0.5)
        quaternions = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [half, 0.0, 0.0, half]], device=device
        )
        rotations = build_rotation(quaternions)
        opacity = torch.tensor([[0.8], [0.5]], device=device)

        torch.cuda.manual_seed_all(args.seed)
        repeated_scales = scales.repeat(2, 1)
        torch.normal(mean=torch.zeros_like(repeated_scales), std=repeated_scales)
        expected_rng = torch.cuda.get_rng_state()
        torch.cuda.manual_seed_all(args.seed)
        torch.normal(mean=torch.zeros_like(repeated_scales), std=repeated_scales)
        child_xyz, child_scales, child_opacity, axes = long_axis_children(
            xyz, scales, rotations, opacity
        )
        actual_rng = torch.cuda.get_rng_state()
        covariance = projected_covariance(
            child_scales,
            rotations.repeat(2, 1, 1),
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device),
        )
        torch.cuda.synchronize()

        if not torch.equal(actual_rng, expected_rng):
            raise RuntimeError(
                "Deterministic LAS math changed the post-draw CUDA RNG state"
            )
        if not all(
            torch.isfinite(value).all()
            for value in (child_xyz, child_scales, child_opacity, covariance)
        ):
            raise FloatingPointError("Nonfinite CUDA preflight output")
        if not torch.all(child_scales > 0):
            raise FloatingPointError("Non-positive CUDA child scale")

        result.update(
            status="passed_gpu_preflight",
            torch=torch.__version__,
            cuda=torch.version.cuda,
            device=torch.cuda.get_device_name(),
            native_signature=signature,
            hook={
                key: value for key, value in state.items() if key != "original_split"
            },
            axes=axes.cpu().tolist(),
            child_xyz=child_xyz.cpu().tolist(),
            child_scales=child_scales.cpu().tolist(),
            child_opacity=child_opacity.cpu().tolist(),
            cuda_rng_preserved_after_native_draw=True,
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
        )
    except BaseException as exc:
        result.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        write_new_json(args.evidence, result)


if __name__ == "__main__":
    main()
