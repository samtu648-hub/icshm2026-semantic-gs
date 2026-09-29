#!/usr/bin/env python3
"""Run the pinned ImprovedGS source with the registered all-400 camera universe."""
from __future__ import annotations

import argparse
import inspect
import json
import random
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256, write_new_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--access", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("upstream_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.evidence.exists():
        raise FileExistsError(args.evidence)

    started = time.monotonic()
    record: dict[str, object] = {
        "schema_version": 1,
        "seed": args.seed,
        "status": "initializing",
        "sample_trace": [],
    }
    try:
        sys.path.insert(0, str(args.source))
        import numpy as np
        import torch
        import scene
        from scene import dataset_readers
        from scene import training_runtime
        from utils import general_utils

        access = json.loads(args.access.read_text())
        train_names = set(access["train"])
        evaluation_names = set(access["evaluation"])
        if len(train_names) != 400 or evaluation_names or access.get("partition") != "released_all400_v1":
            raise ValueError("Expected the registered released_all400_v1 access set")

        original_reader = scene.sceneLoadTypeCallbacks["Colmap"]

        def explicit_reader(*reader_args, **reader_kwargs):
            bound = inspect.signature(original_reader).bind(*reader_args, **reader_kwargs)
            bound.apply_defaults()
            bound.arguments["eval"] = False
            scene_path = Path(bound.arguments["path"])
            info = original_reader(*bound.args, **bound.kwargs)
            cameras = list(info.train_cameras) + list(info.test_cameras)

            def canonical_name(camera) -> str:
                name = Path(camera.image_name).name
                return name if name.endswith(".png") else name + ".png"

            actual = [canonical_name(camera) for camera in cameras]
            if len(actual) != len(set(actual)) or set(actual) != train_names:
                raise ValueError("ImprovedGS reader changed the registered all-400 image universe")
            train = [camera for camera in cameras if canonical_name(camera) in train_names]
            canonical_ply = scene_path / "sparse/0/points3D.ply"
            common_cloud = dataset_readers.fetchPly(str(canonical_ply))
            if len(common_cloud.points) != 72_292 or not np.isfinite(common_cloud.points).all():
                raise ValueError("Unexpected common initialization point cloud")
            record["actual_train"] = sorted(canonical_name(camera) for camera in train)
            record["actual_evaluation"] = []
            record["initialization"] = {
                "canonical_ply_sha256": sha256(canonical_ply),
                "common_point_count": len(common_cloud.points),
                "policy": "RGB130/RGB120 original 303-view DLT initialization; all400 photometric training",
            }
            return info._replace(
                train_cameras=train,
                test_cameras=[],
                point_cloud=common_cloud,
                ply_path=str(canonical_ply),
                nerf_normalization=dataset_readers.getNerfppNorm(train),
            )

        scene.sceneLoadTypeCallbacks["Colmap"] = explicit_reader

        original_safe_state = general_utils.safe_state

        def seeded_state(*state_args, **state_kwargs):
            original_safe_state(*state_args, **state_kwargs)
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

        general_utils.safe_state = seeded_state

        camera_rng = torch.Generator(device="cpu")
        camera_rng.manual_seed(args.seed + 17_031)

        def audited_sample(scene_object, viewpoint_stack):
            if not viewpoint_stack:
                viewpoint_stack.extend(scene_object.getTrainCameras().copy())
            sample_index = int(torch.randint(0, len(viewpoint_stack), (1,), generator=camera_rng).item())
            camera = viewpoint_stack.pop(sample_index)
            trace = record["sample_trace"]
            if len(trace) < 1_000:
                name = Path(camera.image_name).name
                trace.append(name if name.endswith(".png") else name + ".png")
            return camera

        training_runtime.sample_training_viewpoint = audited_sample

        torch.cuda.reset_peak_memory_stats()
        record["torch"] = torch.__version__
        record["cuda"] = torch.version.cuda
        record["device"] = torch.cuda.get_device_name()
        upstream_args = args.upstream_args[1:] if args.upstream_args[:1] == ["--"] else args.upstream_args
        sys.argv = [str(args.source / "train.py"), *upstream_args]
        runpy.run_path(str(args.source / "train.py"), run_name="__main__")
        torch.cuda.synchronize()
        record["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        record["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
        record["status"] = "completed_native_execution"
    except BaseException as exc:
        record.update(status="failed_native_execution", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            import torch

            if torch.cuda.is_initialized():
                record["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
                record["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
        except Exception:
            pass
        record["seconds"] = time.monotonic() - started
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        write_new_json(args.evidence, record)


if __name__ == "__main__":
    main()
